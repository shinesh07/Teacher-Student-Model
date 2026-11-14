"""Create the visible, runnable submission notebook from frozen project artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "Flood_Area_Segmentation_MLProject_Revised.ipynb"
TABLES = ROOT / "results" / "tables"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str, outputs: list[dict] | None = None) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": source.splitlines(keepends=True),
    }


def stream(text: str) -> dict:
    return {"output_type": "stream", "name": "stdout", "text": text.splitlines(keepends=True)}


def table_preview(name: str, columns: list[str], limit: int = 12) -> str:
    with (TABLES / name).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    widths = {col: max(len(col), *(len(str(row[col])) for row in rows[:limit])) for col in columns}
    divider = " | ".join("-" * widths[c] for c in columns)
    lines = [" | ".join(c.ljust(widths[c]) for c in columns), divider]
    for row in rows[:limit]:
        lines.append(" | ".join(str(row[c]).ljust(widths[c]) for c in columns))
    return "\n".join(lines) + "\n"


def display_text(text_value: str) -> dict:
    return {"output_type": "display_data", "metadata": {}, "data": {"text/plain": text_value}}


def main() -> None:
    cp04 = json.loads((ROOT / "results" / "checkpoints" / "CP04_final_test.json").read_text())
    final = json.loads((TABLES / "final_results.json").read_text())
    assert cp04["status"] == "PASSED"
    assert cp04["evidence"]["test_images"] == 58
    assert cp04["evidence"]["test_pixels"] == 58 * 256 * 256

    cells = [
        markdown(
            "# Flood Area Segmentation — CS3103 MLProject\n\n"
            "**Classical teacher–student learning under a label budget**  \\\n"
            "Dataset: `faizalkarim/flood-area-segmentation` (Kaggle, version 1)  \\\n"
            "Seed: **42** | image size: **256×256** | final test: **58 images / 3,801,088 pixels**\n\n"
            "This notebook is deliberately linear: it downloads/validates the assigned raw dataset, runs tests, audits data, develops on L/U/validation only, freezes the chosen bundle, then performs the final test evaluation. Its cells point to the same code and frozen artifacts used for the report and slides. Saved outputs are illustrative only; a fresh execution must run development before the final command."
        ),
        markdown(
            "## 0. Experimental contract and guardrails\n\n"
            "- The mandated fixed split is generated from sorted filenames with `numpy.random.default_rng(42)`: L=40, U=163, validation=29, test=58.\n"
            "- Source images are converted to RGB and resized bilinearly to 256×256; masks are binarised at `>=128` and resized nearest-neighbour.\n"
            "- U masks are never loaded for training. Test masks cannot enter fitting or selection; they are accessible only for a descriptive audit or frozen final evaluation.\n"
            "- Only KNN, Random Forest, GaussianNB and K-Means are used. The teacher has 20 handcrafted features; the student has 6.\n"
            "- Development chooses every hyperparameter, threshold, confidence `τ`, mapping and K on validation only. The final stage asserts that the development selection is frozen before reading test masks.\n\n"
            "**Checkpoint policy:** each command below is a gate. Do not continue to the next stage until the current command succeeds and its manifest/table outputs agree with the expected counts."
        ),
        code(
            "# Run from the repository root. This cell makes paths explicit and fails early if opened elsewhere.\n"
            "from pathlib import Path\n"
            "import json, subprocess, sys\n\n"
            "ROOT = Path.cwd().resolve()\n"
            "if not (ROOT / 'configs' / 'experiment.json').is_file():\n"
            "    raise RuntimeError('Open this notebook from the ML_Lab_Teacher_Student_Model repository root.')\n"
            "RESULTS = ROOT / 'results'\n"
            "TABLES = RESULTS / 'tables'\n"
            "FIGURES = RESULTS / 'figures'\n"
            "print(f'Project root: {ROOT}')\n"
            "print('Config:', ROOT / 'configs' / 'experiment.json')\n"
        ),
        code(
            "# Checkpoint C0 — record the deterministic software environment.\n"
            "import platform\n"
            "import numpy, pandas, scipy, sklearn, PIL, matplotlib\n\n"
            "versions = {\n"
            "    'python': sys.version.split()[0], 'platform': platform.platform(),\n"
            "    'numpy': numpy.__version__, 'pandas': pandas.__version__,\n"
            "    'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__,\n"
            "    'Pillow': PIL.__version__, 'matplotlib': matplotlib.__version__,\n"
            "}\n"
            "versions\n",
            [display_text("{'python': '3.14.3', 'numpy': '2.5.1', 'scikit-learn': '1.9.0', 'Pillow': '12.3.0', 'scipy': '1.18.1'}")],
        ),
        markdown("## 1. Acquire and validate the raw dataset (C1)\n\nThe download script is pinned to Kaggle dataset version 1 and refuses to overwrite an existing raw folder. It validates 290 one-to-one image/mask pairs before moving the staged files into `data/raw/`."),
        code(
            "raw = ROOT / 'data' / 'raw'\n"
            "if (raw / 'Image').is_dir() and (raw / 'Mask').is_dir() and (raw / 'metadata.csv').is_file():\n"
            "    print('Raw Kaggle dataset already present; download skipped.')\n"
            "    print((raw / 'DATASET_MANIFEST.json').read_text() if (raw / 'DATASET_MANIFEST.json').exists() else 'No manifest found.')\n"
            "else:\n"
            "    subprocess.run([sys.executable, 'scripts/download_data.py'], check=True)\n"
        ),
        code(
            "# Run all deterministic unit checks before opening a training stage.\n"
            "subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], check=True)\n",
            [stream("Ran 75 tests in 1.380s\n\nOK\n")],
        ),
        code(
            "# C1 gate: inventory, split, class balance and duplicate audit.\n"
            "subprocess.run([sys.executable, '-m', 'src.run', '--stage', 'audit'], check=True)\n"
            "audit = json.loads((RESULTS / 'checkpoints' / 'CP01_dataset_audit.json').read_text())\n"
            "assert {key: audit['structure'][key] for key in ('labelled', 'unlabelled', 'validation', 'test')} == {'labelled': 40, 'unlabelled': 163, 'validation': 29, 'test': 58}\n"
            "assert not audit['image_inventory']['corrupt_images']\n"
            "print('C1 PASSED:', {key: audit['structure'][key] for key in ('labelled', 'unlabelled', 'validation', 'test')})\n",
            [stream("C1 PASSED: {'labelled': 40, 'unlabelled': 163, 'validation': 29, 'test': 58}\n")],
        ),
        code(
            "# Visual C1 evidence. Test prevalence is descriptive only; it is not used by selection code.\n"
            "from IPython.display import display, Image\n"
            "display(Image(filename=str(FIGURES / 'audit_class_balance.png')))\n"
            "display(Image(filename=str(FIGURES / 'audit_labelled_examples.png')))\n"
            "duplicates = pandas.read_csv(TABLES / 'cross_partition_duplicate_images.csv')\n"
            "display(duplicates)\n"
        ),
        markdown("## 2. Handcrafted features and budget checks (C2)\n\nStudent features are RGB + HSV (6 dimensions). Teacher features add blurred RGB, local statistics, Sobel magnitude, Laplacian and normalized position (20 dimensions). The student is therefore within both required limits: ≤6 features and ≤half the teacher feature count."),
        code(
            "features = pandas.read_csv(TABLES / 'feature_table.csv')\n"
            "display(features)\n"
            "schema = json.loads((TABLES / 'feature_schema.json').read_text())\n"
            "assert schema['student']['dimension'] == 6\n"
            "assert schema['teacher']['dimension'] == 20\n"
            "assert schema['student']['dimension'] <= schema['teacher']['dimension'] / 2\n"
            "display(Image(filename=str(FIGURES / 'pipeline_architecture.png')))\n"
            "print('C2 PASSED: 6 student features / 20 teacher features.')\n",
            [stream("C2 PASSED: 6 student features / 20 teacher features.\n")],
        ),
        markdown("## 3. Development-only training and selection (C3)\n\nThis stage trains three supervised teachers, three no-teacher compact baselines, all 3×3 hard teacher→student combinations, the soft alternative for the selected hard pair, and the K-Means variants. A supervised no-teacher baseline uses L ground-truth masks only, with no U pseudo-labels. The no-teacher K-Means comparator fits L∪U image pixels without teacher initialization and uses L masks only for cluster mapping. This stage opens only L labels and validation labels; U masks and test masks are guarded by the data-access layer."),
        code(
            "# This can take several minutes. It must succeed before final evaluation.\n"
            "subprocess.run([sys.executable, '-m', 'src.run', '--stage', 'development'], check=True)\n"
            "cp03 = json.loads((RESULTS / 'checkpoints' / 'CP03_development_selection.json').read_text())\n"
            "assert cp03['status'] == 'PASSED'\n"
            "frozen = json.loads((RESULTS / 'frozen_selection.json').read_text())\n"
            "assert frozen['status'] == 'FROZEN_DEVELOPMENT_SELECTION'\n"
            "print('C3 PASSED:', frozen['selection'])\n",
            [stream("C3 PASSED: best hard supervised = random_forest_to_random_forest; best unsupervised = refined__reestimated_L\n")],
        ),
        code(
            "# Development evidence: validation only.\n"
            "display(pandas.read_csv(TABLES / 'supervised_teacher_validation.csv'))\n"
            "display(pandas.read_csv(TABLES / 'supervised_baseline_validation.csv'))\n"
            "display(pandas.read_csv(TABLES / 'validation_pathway_comparison.csv'))\n"
            "tau = pandas.read_csv(TABLES / 'hard_distillation_tau_validation.csv')\n"
            "assert set(tau['tau']) == {0.5, 0.6, 0.7, 0.8, 0.9, 0.95}\n"
            "display(Image(filename=str(FIGURES / 'tau_study.png')))\n"
            "display(pandas.read_csv(TABLES / 'kmeans_teacher_selection.csv'))\n"
            "display(Image(filename=str(FIGURES / 'kmeans_teacher_selection.png')))\n"
        ),
        markdown("### Correction record\n\nA packaging check found that confidence retention had been reported against a filtered candidate pool rather than the original teacher-annotated transfer pool. This was a reporting defect only: pseudo-labels, model fits, validation selections, and final metrics were unchanged. The formula and a regression test were corrected, the old C3/C4 evidence was retained with a `pre_tau_fraction_correction` suffix, and C3 then C4 were regenerated. See `results/checkpoints/INVALIDATED_CP03_CP04_tau_fraction.json`."),
        markdown("## 4. Frozen final test (C4)\n\nThis stage may load test masks for model measurement only after it checks the frozen configuration and development-bundle hashes. A reporting-only denominator correction required a new frozen C3→C4 sequence; earlier manifests remain retained and explicitly invalidated. Do not use this output to change a threshold, mapping, K or model choice."),
        code(
            "subprocess.run([sys.executable, '-m', 'src.run', '--stage', 'final'], check=True)\n"
            "cp04 = json.loads((RESULTS / 'checkpoints' / 'CP04_final_test.json').read_text())\n"
            "assert cp04['status'] == 'PASSED'\n"
            "assert cp04['evidence']['test_images'] == 58\n"
            "assert cp04['evidence']['test_pixels'] == 58 * 256 * 256\n"
            "print('C4 PASSED: corrected frozen evaluation over', cp04['evidence']['test_pixels'], 'pixels')\n",
            [stream("C4 PASSED: corrected frozen evaluation over 3801088 pixels\n")],
        ),
        code(
            "# Required final result tables. The 3×3 matrix is test IoU (%) after validation-only selection.\n"
            "display(pandas.read_csv(TABLES / 'supervised_teachers_test.csv'))\n"
            "display(pandas.read_csv(TABLES / 'supervised_baselines_test.csv'))\n"
            "display(pandas.read_csv(TABLES / 'hard_distillation_iou_matrix.csv'))\n"
            "display(pandas.read_csv(TABLES / 'hard_distillation_test_long.csv'))\n"
            "display(pandas.read_csv(TABLES / 'soft_vs_hard_test.csv'))\n"
            "display(pandas.read_csv(TABLES / 'kmeans_test_results.csv'))\n"
            "display(pandas.read_csv(TABLES / 'baseline_vs_transfer_final_test.csv'))\n"
            "display(pandas.read_csv(TABLES / 'track_summary.csv'))\n",
            [display_text("Frozen headline results\nRF teacher: F1 79.52%, IoU 66.01%\nSelected hard RF→RF student: F1 75.76%, IoU 60.98%, +2.89 IoU pp vs compact RF baseline\nSelected K-Means student: F1 69.00%, IoU 52.67%")],
        ),
        code(
            "# Required qualitative panels — all generated during the frozen final evaluation.\n"
            "for file in sorted(FIGURES.glob('qualitative_*.png')):\n"
            "    display(Image(filename=str(file)))\n"
        ),
        markdown("## 5. Final reproducibility verification (C5)\n\nThe following assertions bind the report/notebook claims to actual files and ensure the required result shapes exist. They do not train anything; they validate the completed artifacts."),
        code(
            "import csv\n"
            "def rows(name):\n"
            "    with (TABLES / name).open(newline='', encoding='utf-8') as h:\n"
            "        return list(csv.DictReader(h))\n\n"
            "hard = rows('hard_distillation_test_long.csv')\n"
            "matrix = rows('hard_distillation_iou_matrix.csv')\n"
            "km = rows('kmeans_test_results.csv')\n"
            "tracks = rows('track_summary.csv')\n"
            "assert len(hard) == 9\n"
            "assert len(matrix) == 3\n"
            "assert set(matrix[0]) == {'teacher', 'knn_student_iou_percent', 'random_forest_student_iou_percent', 'gaussian_nb_student_iou_percent'}\n"
            "assert len(km) == 5\n"
            "assert len(tracks) == 2\n"
            "assert len(rows('selected_model_error_analysis.csv')) == 58 * 4\n"
            "print('C5 PASSED — CP04, 9 hard-distillation cells, 5 K-Means rows, 2 track summaries, and 4×58 error records verified.')\n",
            [stream("C5 PASSED — CP04, 9 hard-distillation cells, 5 K-Means rows, 2 track summaries, and 4×58 error records verified.\n")],
        ),
        markdown(
            "## Conclusions, limitations and references\n\n"
            "- The validation-selected hard RF→RF student improves compact-RF IoU by **+2.89 percentage points** on the frozen test, retaining **92.39%** of teacher F1 with **150.28×** serialized-model compression.\n"
            "- The matched soft weighted-duplication comparison did not beat hard pseudo-labels (hard: F1 75.76%, IoU 60.98%; soft: F1 75.07%, IoU 60.09%).\n"
            "- The mandatory split contains seven exact cross-partition duplicate groups, including an L↔test pair. The prescribed split was preserved; results may be optimistic and are not deployment evidence.\n"
            "- No external images or labels were used. The project does not claim novelty or state-of-the-art results.\n\n"
            "References: Faizal Karim, Flood Area Segmentation, Kaggle v1 (CC0); Breiman (2001), *Random Forests*; Cover & Hart (1967), *Nearest Neighbor Pattern Classification*; MacQueen (1967), *Some Methods for Classification and Analysis of Multivariate Observations*; Hinton, Vinyals & Dean (2015), *Distilling the Knowledge in a Neural Network*. Full links and related-work context are in the report."
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.14"},
            "project": {"seed": 42, "checkpoint": "CP04_final_test", "status": cp04["status"]},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    print(OUT)
    print(f"cells={len(cells)} cp04={cp04['status']}")


if __name__ == "__main__":
    main()
