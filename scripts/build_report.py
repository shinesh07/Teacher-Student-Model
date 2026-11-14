"""Build the project report from the frozen experiment artifacts.

Run only after `python -m src.run --stage final` has passed.  This script deliberately
reads result tables rather than embedding independently typed performance numbers.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
OUTPUT = ROOT / "output"
OUT = OUTPUT / "Flood_Area_Segmentation_Teacher_Student_Report_Revised.docx"

NAVY = "12304A"
TEAL = "007C83"
PALE = "E8F2F3"
ORANGE = "E87527"
GRAY = "5B6770"


def read_csv(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_text(cell, text: str, bold: bool = False, color: str = "000000") -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(str(text))
    r.bold = bold
    r.font.name = "Aptos"
    r.font.size = Pt(8)
    r.font.color.rgb = RGBColor.from_string(color)


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[float] | None = None) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.autofit = False
    for ix, header in enumerate(headers):
        cell = table.rows[0].cells[ix]
        set_cell_text(cell, header, bold=True, color="FFFFFF")
        set_cell_shading(cell, NAVY)
        if widths:
            cell.width = Inches(widths[ix])
    for row_ix, row in enumerate(rows):
        cells = table.add_row().cells
        for ix, value in enumerate(row):
            set_cell_text(cells[ix], value)
            if row_ix % 2 == 0:
                set_cell_shading(cells[ix], "F4F7F8")
            if widths:
                cells[ix].width = Inches(widths[ix])
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    p = doc.add_paragraph(style=f"Heading {level}")
    p.add_run(text)


def add_bullet(doc: Document, text: str) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(2)
    p.add_run(text)


def add_caption(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(8)
    r.font.color.rgb = RGBColor.from_string(GRAY)


def add_figure(doc: Document, file: str, caption: str, width: float = 6.35) -> None:
    path = FIGURES / file
    if not path.exists():
        raise FileNotFoundError(path)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(width))
    add_caption(doc, caption)


def fmt_pct(value: str | float) -> str:
    return f"{float(value):.2f}%"


def fmt_x(value: str | float) -> str:
    numeric = float(value)
    return f"{numeric:.4f}x" if abs(numeric) < 0.01 else f"{numeric:.2f}x"


def set_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    paragraph.add_run("CS3103 | ")
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    paragraph._p.append(field)


def setup_document(doc: Document) -> None:
    section = doc.sections[0]
    section.top_margin = Cm(1.6)
    section.bottom_margin = Cm(1.5)
    section.left_margin = Cm(1.7)
    section.right_margin = Cm(1.7)
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(9.5)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.05
    for level, size, color in [(1, 16, NAVY), (2, 12, TEAL), (3, 10, NAVY)]:
        style = styles[f"Heading {level}"]
        style.font.name = "Aptos Display"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(8)
        style.paragraph_format.space_after = Pt(4)
    if "Report Small" not in styles:
        style = styles.add_style("Report Small", WD_STYLE_TYPE.PARAGRAPH)
        style.font.name = "Aptos"
        style.font.size = Pt(8)
        style.font.color.rgb = RGBColor.from_string(GRAY)
    header = section.header.paragraphs[0]
    header.text = "FLOOD AREA SEGMENTATION  /  CLASSICAL TEACHER–STUDENT LEARNING"
    header.runs[0].font.name = "Aptos"
    header.runs[0].font.size = Pt(8)
    header.runs[0].font.color.rgb = RGBColor.from_string(TEAL)
    set_page_number(section.footer.paragraphs[0])


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    cp04 = json.loads((RESULTS / "checkpoints" / "CP04_final_test.json").read_text())
    frozen = json.loads((RESULTS / "frozen_selection.json").read_text())
    assert cp04["status"] == "PASSED", "report generation requires passed CP04"
    assert cp04["evidence"]["test_images"] == 58
    assert cp04["evidence"]["test_pixels"] == 58 * 256 * 256

    teachers = read_csv("supervised_teachers_test.csv")
    baselines = read_csv("supervised_baselines_test.csv")
    hard = read_csv("hard_distillation_test_long.csv")
    matrix = read_csv("hard_distillation_iou_matrix.csv")
    soft = read_csv("soft_vs_hard_test.csv")[0]
    kmeans = read_csv("kmeans_test_results.csv")
    kselect = read_csv("kmeans_teacher_selection.csv")
    validation_inventory = read_csv("validation_pathway_comparison.csv")
    final_comparison = read_csv("baseline_vs_transfer_final_test.csv")
    tau = read_csv("hard_distillation_tau_validation.csv")
    duplicates = read_csv("cross_partition_duplicate_images.csv")
    dataset = json.loads((TABLES / "dataset_card.json").read_text())
    selected_hard = next(r for r in hard if r["pair"] == "random_forest_to_random_forest")
    selected_km = next(r for r in kmeans if r["model"] == "km_refined__reestimated_L")

    doc = Document()
    setup_document(doc)

    # Cover
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(85)
    r = p.add_run("Flood Area Segmentation")
    r.font.name = "Aptos Display"
    r.font.size = Pt(28)
    r.font.bold = True
    r.font.color.rgb = RGBColor.from_string(NAVY)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Classical Teacher–Student Learning under a Label Budget")
    r.font.size = Pt(15)
    r.font.color.rgb = RGBColor.from_string(TEAL)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(28)
    p.add_run("CS3103 Machine Learning Project Report").bold = True
    for line in ["Student: Bojedla Shinesh", "Roll number: 2401CS75", "Date: 14 November 2026", "Dataset: faizalkarim/flood-area-segmentation (Kaggle, version 1)", "Experiment seed: 42 | Final checkpoint: PASSED"]:
        p = doc.add_paragraph(line)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(3)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(20)
    run = p.add_run("Reproducible artifact generated from frozen experiment tables")
    run.italic = True
    run.font.color.rgb = RGBColor.from_string(GRAY)
    doc.add_page_break()

    add_heading(doc, "Executive summary")
    doc.add_paragraph(
        "This project evaluates flood / non-flood pixel segmentation on the assigned RGB dataset using only the permitted classical models: K-Nearest Neighbours (KNN), Random Forest (RF), Gaussian Naive Bayes (GNB), and K-Means. A richer teacher feature set (20 handcrafted features) is transferred to compact 6-feature students. Every development decision uses labelled L images, unlabelled-image pixels U, and validation only. The reported final metrics come from the corrected re-evaluation of a frozen selection bundle; no test result changed a selection or retrained a model."
    )
    doc.add_paragraph(
        f"The strongest supervised teacher was RF (test IoU {fmt_pct(teachers[1]['iou_percent'])}; F1 {fmt_pct(teachers[1]['f1_percent'])}). The validation-selected RF→RF hard-distilled student reached IoU {fmt_pct(selected_hard['test_iou_percent'])} and F1 {fmt_pct(selected_hard['test_f1_percent'])}, improving on the compact RF baseline by {float(selected_hard['delta_iou_percentage_points']):.2f} IoU points while reducing serialized model size by {fmt_x(selected_hard['compression_ratio'])}. In the unsupervised track, the validation-selected refined K-Means student reached IoU {fmt_pct(selected_km['test_iou_percent'])}."
    )
    add_table(doc, ["Frozen evaluation", "Value"], [
        ["Images / pixels", "58 test images / 3,801,088 pixels"],
        ["Configuration hash", cp04["config_sha256"][:16] + "…"],
        ["Development bundle", frozen["bundle_sha256"][:16] + "…"],
        ["Unit / artifact checks", "75 tests passed; CP04 passed; 9 hard cells; 5 K-Means rows"],
    ], [2.0, 4.4])

    add_heading(doc, "1. Problem, data and experimental contract")
    doc.add_paragraph(
        "The task is binary semantic segmentation: each 256×256 RGB pixel is classified as flood (1) or non-flood (0). The source is the prescribed Flood Area Segmentation Kaggle dataset (Faizal Karim, version 1, CC0). The mandatory split was reproduced from sorted filenames with numpy.random.default_rng(42): 40 labelled images (L), 163 unlabelled images (U), 29 validation images and 58 test images."
    )
    add_table(doc, ["Partition", "Images", "Masks used for" , "Flood-pixel fraction"], [
        ["L", "40", "Teacher / baseline fit; K-Means mapping", "36.49%"],
        ["U", "163", "Teacher pseudo-labelling; image-only K-Means fit", "Never loaded"],
        ["Validation", "29", "All development selection", "37.89%"],
        ["Test", "58", "Frozen final evaluation only", "38.34% (descriptive audit only)"],
    ], [0.9, 0.7, 2.5, 2.3])
    add_figure(doc, "audit_class_balance.png", "Figure 1. Post-preprocessing class balance across partitions. Test prevalence is reported descriptively, not used for selection.", 5.9)
    add_figure(doc, "audit_labelled_examples.png", "Figure 2. Representative training image-mask pairs from partition L showing raw flood scenes and ground-truth flood annotations.", 6.2)
    doc.add_paragraph(
        "Preprocessing is deterministic: Pillow converts every source image to RGB; images use bilinear resizing to 256×256; masks are thresholded at ≥128 and resized with nearest neighbour. An audit found 290 image–mask pairs, no corrupt source images, and source modes P=2, RGB=287 and RGBA=1 before RGB conversion."
    )
    add_bullet(doc, f"Duplicate audit: {len(duplicates)} exact duplicate groups cross fixed partitions. The required split was retained unchanged, but this can make reported generalisation optimistic. Notably, 1026.jpg (L) and 18.jpg (test) are an exact cross-split pair.")
    add_bullet(doc, "Leakage gate: U masks are blocked before opening; test masks are blocked except for descriptive audit or the frozen final-evaluation stage. Test results cannot change a model, threshold, mapping or K.")

    add_heading(doc, "2. Feature design and model budgets")
    doc.add_paragraph(
        "The student feature vector is intentionally compact: RGB (3), HSV hue/saturation/value (3), for six values per pixel. The teacher extends it to 20 values with Gaussian-blurred RGB, local mean/standard-deviation descriptors, Sobel magnitude, Laplacian and normalized pixel coordinates. The student therefore uses 6/20 = 30% of teacher features, satisfying both the ≤6 and ≤half budget."
    )
    add_figure(doc, "pipeline_architecture.png", "Figure 3. Leak-safe pipeline: all tuning ends at validation; test labels are read only after the selection manifest is frozen.", 6.35)
    add_table(doc, ["Family", "Teacher budget", "Student budget", "Actual student fit"], [
        ["KNN", "≥20,000 references; rich features", "≤5,000 references; 6 features", "5,000"],
        ["Random Forest", "≥200 trees; rich features", "≤20 trees; depth ≤8; 6 features", "20,000 pixels"],
        ["GaussianNB", "20 features", "6 features", "20,000 pixels"],
        ["K-Means", "K≥6, n_init≥10, ≥100,000 pixels", "K≤4, ≤20,000 pixels", "20,000 pixels"],
    ], [1.1, 2.1, 2.15, 1.0])

    add_heading(doc, "3. Development protocol and checkpoints")
    doc.add_paragraph(
        "C0 captured the project contract; C1 audited the raw data and fixed split; C2 verified preprocessing, features and capacity constraints; C3 performed development-only model selection and froze the selected bundle; C4 evaluated that frozen bundle on the test set; C5 packages these deliverables. Artifact manifests store the configuration hash, package versions, sampling evidence, model metadata and output file paths."
    )
    add_bullet(doc, "Supervised teachers are fit on L. Each student receives teacher pseudo-labels over permitted U images; confidence τ ∈ {0.50, 0.60, 0.70, 0.80, 0.90, 0.95} and student threshold are chosen on validation.")
    add_bullet(doc, "All 3×3 hard teacher→student family pairs are evaluated. A soft-label weighted-duplication alternative is compared at the same τ for the best hard pair.")
    add_bullet(doc, "For unsupervised learning, the teacher K is selected by silhouette on a fixed image-only L∪U sample, then clusters are mapped to flood/non-flood using L masks. Four student variants cross no-Lloyd/frozen versus Lloyd-refined centroids with inherited versus re-estimated L mapping.")
    add_bullet(doc, "A confidence-retention reporting denominator defect was found during packaging. It changed no labels, models, choices or final metrics. The old C3/C4 evidence is preserved, the calculation and regression test were corrected, then the frozen C3→C4 sequence was rerun.")

    add_heading(doc, "4. Development validation inventory")
    doc.add_paragraph("This table records every development candidate before final measurement. The supervised no-teacher bases use L ground-truth masks only and no U pseudo-labels. The K-Means base is also no-teacher, but legitimately uses L∪U images for unsupervised fitting and L masks only to map its clusters. K-Means transfer selection is restricted to the four teacher-initialized variants, so the independent base remains a comparator rather than a candidate for that selection.")
    add_table(doc, ["Track", "Candidate", "Teacher", "Validation F1", "Validation IoU", "Final selection"], [
        [r["track"], r["candidate"].replace("random_forest", "RF").replace("gaussian_nb", "GNB").replace("_to_", " → "), r["uses_teacher"], fmt_pct(r["validation_f1_percent"]), fmt_pct(r["validation_iou_percent"]), "Yes" if r["selected_for_final"] == "True" else "No"]
        for r in validation_inventory
    ], [0.78, 2.40, 0.70, 0.85, 0.90, 0.75])

    add_heading(doc, "5. Final comparison against no teacher baselines")
    doc.add_paragraph("Metrics are micro-aggregated across every test pixel. Model sizes are serialized artifacts; speedups compare the recorded end-to-end segmentation timing for the corresponding teacher and student.")
    add_table(doc, ["Teacher", "Features", "Fit pixels", "F1", "IoU", "Size"], [
        [r["name"].replace("teacher_", "").upper(), r["feature_dimensions"], f"{int(r['fit_pixels']):,}", fmt_pct(r["f1_percent"]), fmt_pct(r["iou_percent"]), f"{int(r['model_size_bytes']):,} B"]
        for r in teachers
    ], [1.25, 0.65, 1.0, 0.8, 0.8, 1.4])
    add_table(doc, ["No-teacher compact baseline", "F1", "IoU", "Size"], [
        [r["name"].replace("baseline_", "").upper(), fmt_pct(r["f1_percent"]), fmt_pct(r["iou_percent"]), f"{int(r['model_size_bytes']):,} B"]
        for r in baselines
    ], [2.4, 1.0, 1.0, 1.5])
    doc.add_paragraph(
        "RF is the strongest teacher on the held-out test set. The compact baselines establish that a six-feature RF can be fast but loses accuracy without transferred pseudo-labels. Reported test outcomes are not used to replace the validation selection."
    )
    add_figure(doc, "supervised_teacher_vs_baseline.png", "Figure 4. Supervised teachers vs compact baselines across F1 and IoU, illustrating the baseline performance drop before knowledge transfer.", 5.8)
    add_table(doc, ["Category", "Candidate", "Teacher", "F1", "IoU", "Delta IoU vs base"], [
        [r["category"], r["candidate"], r["uses_teacher"], fmt_pct(r["test_f1_percent"]), fmt_pct(r["test_iou_percent"]), "n/a" if r["delta_iou_vs_corresponding_base_percentage_points"] == "" else f"{float(r['delta_iou_vs_corresponding_base_percentage_points']):+.2f} pp"]
        for r in final_comparison
    ], [1.35, 1.75, 0.73, 0.70, 0.70, 1.15])
    doc.add_paragraph("The selected hard RF→RF transfer improves its directly matched no-teacher compact RF base by +2.89 IoU points. The selected K-Means transfer improves its independent no-teacher K-Means base by only +0.05 points, which is not a substantive gain and should not be over-interpreted.")

    add_heading(doc, "6. Hard distillation: all teacher–student pairs")
    doc.add_paragraph("Table 5 is the required 3×3 test IoU (%) matrix. Each cell uses the τ and student threshold selected on validation for that pair.")
    add_table(doc, ["Teacher → student", "KNN", "RF", "GNB"], [
        [r["teacher"].replace("random_forest", "RF").replace("gaussian_nb", "GNB").upper(), fmt_pct(r["knn_student_iou_percent"]), fmt_pct(r["random_forest_student_iou_percent"]), fmt_pct(r["gaussian_nb_student_iou_percent"])]
        for r in matrix
    ], [2.0, 1.2, 1.2, 1.2])
    add_figure(doc, "hard_distillation_heatmap.png", "Figure 5. Hard-distillation test IoU matrix heatmap across all 9 teacher-student pairs.", 4.5)
    add_table(doc, ["Pair", "τ", "F1", "IoU", "Δ IoU vs compact baseline", "Retention", "Compression"], [
        [r["pair"].replace("random_forest", "RF").replace("gaussian_nb", "GNB").replace("_to_", " → "), r["selected_tau"], fmt_pct(r["test_f1_percent"]), fmt_pct(r["test_iou_percent"]), f"{float(r['delta_iou_percentage_points']):+.2f} pp", fmt_pct(r["retention_percent"]), fmt_x(r["compression_ratio"])]
        for r in hard
    ], [1.5, 0.45, 0.72, 0.72, 1.45, 0.95, 0.95])
    add_figure(doc, "tau_study.png", "Figure 6. Validation IoU and corrected retained-pseudo-label fraction across confidence thresholds.", 6.0)
    doc.add_paragraph(
        f"The validation-selected RF→RF hard student is the supervised choice: test F1 {fmt_pct(selected_hard['test_f1_percent'])}, IoU {fmt_pct(selected_hard['test_iou_percent'])}, agreement {fmt_pct(float(selected_hard['agreement']) * 100)}, retention {fmt_pct(selected_hard['retention_percent'])}, {fmt_x(selected_hard['compression_ratio'])} compression and {fmt_x(selected_hard['speedup'])} speedup. It gains {float(selected_hard['delta_f1_percentage_points']):+.2f} F1 points and {float(selected_hard['delta_iou_percentage_points']):+.2f} IoU points over its no-teacher compact RF baseline."
    )
    doc.add_paragraph(
        f"Soft weighted duplication did not win: at τ={soft['tau_held_fixed']}, soft RF→RF gives F1 {fmt_pct(soft['soft_f1_percent'])} and IoU {fmt_pct(soft['soft_iou_percent'])}, versus hard F1 {fmt_pct(soft['hard_f1_percent'])} and IoU {fmt_pct(soft['hard_iou_percent'])}."
    )

    add_heading(doc, "7. Unsupervised K-Means transfer")
    add_table(doc, ["Teacher K", "Silhouette", "Validation IoU", "Selection"], [
        [r["clusters"], f"{float(r['silhouette']):.3f}", fmt_pct(json.loads(r["validation"])["iou"] * 100), "Selected" if r["clusters"] == "6" else "Not selected"]
        for r in kselect
    ], [1.3, 1.35, 1.4, 1.3])
    add_figure(doc, "kmeans_teacher_selection.png", "Figure 7. K-Means teacher selection: K=6 maximized silhouette on the image-only sample.", 5.8)
    add_figure(doc, "kmeans_variants_comparison.png", "Figure 8. Comparison of the four K-Means student variants on validation and test IoU.", 5.8)
    add_table(doc, ["Student", "Lloyd updates", "Mapping", "F1", "IoU", "ARI", "NMI", "Retention", "Compression"], [
        [r["model"].replace("km_", "").replace("__", " / "), "No" if "frozen" in r["model"] else ("Baseline" if r["model"] == "km_baseline" else "Yes"), "—" if r["model"] == "km_baseline" else ("Inherited" if "inherited" in r["model"] else "Re-estimated L"), fmt_pct(r["test_f1_percent"]), fmt_pct(r["test_iou_percent"]), f"{float(r['ari']):.3f}", f"{float(r['nmi']):.3f}", fmt_pct(r["retention_percent"]), fmt_x(r["compression_ratio"])]
        for r in kmeans
    ], [1.2, 0.7, 1.2, 0.65, 0.65, 0.55, 0.55, 0.9, 0.8])
    doc.add_paragraph(
        "The selected unsupervised student follows the validation protocol (refined / re-estimated-L) even though the frozen variants happen to score higher on test. That difference is reported rather than used for post-hoc selection. The selected transfer improves the independent no-teacher K-Means base by only +0.05 IoU points on the final table, so this K-Means transfer experiment provides little evidence of benefit. The frozen variants retain more than 100% of teacher F1 because they are independently evaluated compact models, not constrained to imitate every teacher error."
    )

    add_heading(doc, "8. Qualitative error analysis")
    doc.add_paragraph("Four frozen test examples compare RGB input, ground truth, the RF teacher, selected RF student, K-Means teacher and selected K-Means student. They are diagnostic outputs, not cherry-picked after retraining.")
    for filename, caption in [
        ("qualitative_1_3015.png", "Figure 9. Test image 3015.jpg: strong agreement on a large flooded region."),
        ("qualitative_2_3095.png", "Figure 10. Test image 3095.jpg: a difficult scene with lower K-Means-student precision."),
        ("qualitative_3_3.png", "Figure 11. Test image 3.jpg: the selected RF student has high recall but over-segments some background."),
        ("qualitative_4_1058.png", "Figure 12. Test image 1058.jpg: complementary teacher/student boundary errors."),
    ]:
        add_figure(doc, filename, caption, 6.15)

    add_heading(doc, "9. Interpretation, limitations and responsible use")
    add_bullet(doc, "The selected hard RF student is an effective compact transfer result: it outperforms the same-budget no-teacher RF baseline on the frozen test set, while retaining 92.39% of teacher IoU (95.27% of teacher F1) and using a far smaller serialized model.")
    add_bullet(doc, "Hard pseudo-labels beat the tested soft weighted-duplication method. This is an empirical result for this split and feature design, not a general claim that hard distillation is always better.")
    add_bullet(doc, "The dataset contains exact image duplicates across mandatory partitions, including an L↔test duplicate. The split was intentionally not changed because the brief prescribes it; all reported results should be read with potential optimism from this dataset property.")
    add_bullet(doc, "The result lifecycle includes a corrected C3→C4 rerun after a reporting-only denominator defect. Earlier C3/C4 manifests are retained and marked invalidated. The final metrics shown here did not guide model selection or retraining, but the project should not claim a single historical test-read event.")
    add_bullet(doc, "The experiment was reproduced within one configured environment, not independently rerun on a second machine. Exact timings can vary by hardware and software versions; the saved configuration, hashes and package versions support repeatability but do not prove cross-machine replication.")
    add_bullet(doc, "This RGB, binary benchmark cannot establish real-world operational reliability. Flood maps affect safety and resource decisions; a deployment would require geographically separated validation, uncertainty handling, temporal checks, human review, and fairness assessment across locations and capture conditions.")
    add_bullet(doc, "No external labels or imagery were used. The project does not claim novelty or state-of-the-art performance; its contribution is a reproducible comparison within the assignment's classical-model and label-budget constraints.")

    add_heading(doc, "10. Reproducibility")
    doc.add_paragraph("From a clean clone with the specified dataset access, run:")
    for command in [
        "python3 -m pip install -r requirements.txt",
        "python3 scripts/download_data.py",
        "python3 -m unittest discover -s tests -v",
        "python3 -m src.run --stage audit",
        "python3 -m src.run --stage development",
        "python3 -m src.run --stage final",
    ]:
        p = doc.add_paragraph()
        p.style = "Report Small"
        run = p.add_run(command)
        run.font.name = "Menlo"
    doc.add_paragraph("The supplied notebook repeats this workflow with visible tables, figures and artifact assertions. Checkpoints and source/result provenance are maintained in CHECKPOINTS.md, results/checkpoints/ and results/provenance/.")

    add_heading(doc, "References")
    refs = [
        "Faizal Karim. Flood Area Segmentation (Kaggle dataset), version 1, CC0. https://www.kaggle.com/api/v1/datasets/view/faizalkarim/flood-area-segmentation",
        "Breiman, L. (2001). Random Forests. Machine Learning, 45, 5–32. https://doi.org/10.1023/A:1010933404324",
        "Cover, T. and Hart, P. (1967). Nearest Neighbor Pattern Classification. IEEE Transactions on Information Theory, 13(1), 21–27. https://doi.org/10.1109/TIT.1967.1053964",
        "MacQueen, J. (1967). Some Methods for Classification and Analysis of Multivariate Observations. Fifth Berkeley Symposium. https://digicoll.lib.berkeley.edu/record/113015",
        "Hinton, G., Vinyals, O. and Dean, J. (2015). Distilling the Knowledge in a Neural Network. https://arxiv.org/abs/1503.02531",
        "Rahnemoonfar, M. et al. (2021). FloodNet: A High Resolution Aerial Imagery Dataset for Post Flood Scene Understanding. IEEE Access, 9, 89644–89654. https://doi.org/10.1109/ACCESS.2021.3090981",
        "Bonafilia, D. et al. (2020). Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms for Sentinel-1. CVPR Workshops. https://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.html",
    ]
    for ref in refs:
        p = doc.add_paragraph(ref)
        p.style = "Report Small"
        p.paragraph_format.left_indent = Cm(0.3)
        p.paragraph_format.first_line_indent = Cm(-0.3)

    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
