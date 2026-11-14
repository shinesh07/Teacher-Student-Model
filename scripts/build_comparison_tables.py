"""Derive leak-safe baseline/pathway comparisons from frozen development/final artifacts.

This packaging step reads no images or masks and never fits a model.  It makes the
meaning of the no-teacher baseline explicit alongside all validation candidates and a
compact final-test comparison table for the report, deck and notebook.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.artifacts import write_rows_csv


TABLES = ROOT / "results" / "tables"


def percentage(value: float) -> float:
    return float(value) * 100.0


def validation_row(
    *, track: str, candidate: str, route: str, uses_teacher: str, training_signal: str,
    feature_dimensions: int, validation: dict, selected_for_final: bool, selection_note: str,
) -> dict:
    return {
        "track": track,
        "candidate": candidate,
        "route": route,
        "uses_teacher": uses_teacher,
        "training_signal": training_signal,
        "feature_dimensions": feature_dimensions,
        "validation_f1": float(validation["f1"]),
        "validation_iou": float(validation["iou"]),
        "validation_f1_percent": percentage(validation["f1"]),
        "validation_iou_percent": percentage(validation["iou"]),
        "selected_for_final": selected_for_final,
        "selection_note": selection_note,
    }


def build_validation_rows(development: dict) -> list[dict]:
    selection = development["selection"]
    rows: list[dict] = []
    for family, record in development["supervised_teachers"].items():
        rows.append(validation_row(
            track="supervised", candidate=f"Teacher {family}", route="rich teacher",
            uses_teacher="upper reference", training_signal="L ground-truth masks only",
            feature_dimensions=20, validation=record["selected"]["validation"],
            selected_for_final=False, selection_note="Teacher reference; all three are retained for final reporting.",
        ))
    for family, record in development["supervised_no_teacher_baselines"].items():
        rows.append(validation_row(
            track="supervised", candidate=f"No-teacher base {family}", route="compact baseline",
            uses_teacher="no", training_signal="L ground-truth masks only; no U pseudo-labels",
            feature_dimensions=6, validation=record["selected"]["validation"],
            selected_for_final=False, selection_note="Direct compact baseline used for fair student-family comparison.",
        ))
    selected_key = selection["best_supervised_student"]["key"]
    selected_kind = selection["best_supervised_student"]["kind"]
    for pair, record in development["hard_distillation"].items():
        rows.append(validation_row(
            track="supervised", candidate=f"Hard {pair}", route="hard distillation",
            uses_teacher="yes", training_signal="teacher pseudo-labels on U images; compact student fit",
            feature_dimensions=6, validation=record["selected"]["validation"],
            selected_for_final=selected_kind == "hard" and pair == selected_key,
            selection_note="Best hard pair selected by validation IoU then F1." if pair == selection["best_hard_supervised_key"] else "Compared on validation only.",
        ))
    soft = development["soft_distillation"]
    rows.append(validation_row(
        track="supervised", candidate=f"Soft {soft['hard_pair']}", route="soft weighted duplication",
        uses_teacher="yes", training_signal="same teacher/U source budget as matched hard pair",
        feature_dimensions=6, validation=soft["validation"],
        selected_for_final=selected_kind == "soft" and soft["hard_pair"] == selected_key,
        selection_note="Matched soft alternative; its source-pixel budget and tau are held fixed.",
    ))
    unsup = development["unsupervised"]
    for trial in unsup["teacher_trials"]:
        rows.append(validation_row(
            track="unsupervised", candidate=f"K-Means teacher K={trial['clusters']}", route="K-Means teacher trial",
            uses_teacher="upper reference", training_signal="L∪U image pixels; L masks only for cluster-to-class mapping",
            feature_dimensions=20, validation=trial["validation"],
            selected_for_final=int(trial["clusters"]) == int(unsup["teacher_selected"]["clusters"]),
            selection_note="Teacher K selected by silhouette, not validation score.",
        ))
    rows.append(validation_row(
        track="unsupervised", candidate="K-Means base (2 clusters)", route="independent compact baseline",
        uses_teacher="no", training_signal="L∪U image pixels; L masks only for compact cluster mapping",
        feature_dimensions=6, validation=unsup["baseline"]["validation"],
        selected_for_final=False, selection_note="Independent no-teacher comparator; not part of distilled-variant selection.",
    ))
    for key, variant in unsup["variants"].items():
        rows.append(validation_row(
            track="unsupervised", candidate=f"K-Means {key}", route="teacher-initialized compact variant",
            uses_teacher="yes", training_signal="K=6 teacher cluster assignments plus L mapping",
            feature_dimensions=6, validation=variant["validation"],
            selected_for_final=key == unsup["selected_variant_key"],
            selection_note="Selection is among the four teacher-initialized variants only.",
        ))
    return rows


def final_row(*, category: str, candidate: str, uses_teacher: str, training_signal: str, row: dict, base: dict | None, selected: bool, note: str) -> dict:
    f1 = float(row["f1"] if "f1" in row else row["test_f1"])
    iou = float(row["iou"] if "iou" in row else row["test_iou"])
    f1_percent = float(row["f1_percent"] if "f1_percent" in row else row["test_f1_percent"])
    iou_percent = float(row["iou_percent"] if "iou_percent" in row else row["test_iou_percent"])
    base_iou = None if base is None else float(base["iou"] if "iou" in base else base["test_iou"])
    delta = "" if base_iou is None else (iou - base_iou) * 100.0
    return {
        "category": category,
        "candidate": candidate,
        "uses_teacher": uses_teacher,
        "training_signal": training_signal,
        "test_f1_percent": f1_percent,
        "test_iou_percent": iou_percent,
        "delta_iou_vs_corresponding_base_percentage_points": delta,
        "selected_for_final": selected,
        "note": note,
    }


def build_final_rows(final: dict) -> list[dict]:
    selection = final["frozen_selection"]
    rows_by_name = {row["name"]: row for row in final["models"]}
    hard = {row["pair"]: row for row in final["hard_distillation"]}
    kmeans = {row["model"]: row for row in final["kmeans"]}
    selected_supervised_key = selection["best_supervised_student"]["key"]
    selected_supervised_kind = selection["best_supervised_student"]["kind"]
    selected_supervised_name = f"{'soft' if selected_supervised_kind == 'soft' else 'hard'}_{selected_supervised_key}"
    selected_unsupervised_name = f"km_{selection['best_unsupervised_variant']}"
    rf_base = rows_by_name["baseline_random_forest"]
    km_base = kmeans["km_baseline"]
    selected_hard_row = hard[selected_supervised_key] if selected_supervised_kind == "hard" else None
    soft = final["soft_vs_hard"]
    output: list[dict] = []
    output.append(final_row(
        category="Supervised upper reference", candidate="Teacher Random Forest", uses_teacher="upper reference",
        training_signal="L ground-truth masks; 20 rich features", row=rows_by_name["teacher_random_forest"], base=rf_base,
        selected=False, note="Accuracy reference; it is not a compact student.",
    ))
    for family in ("knn", "random_forest", "gaussian_nb"):
        base = rows_by_name[f"baseline_{family}"]
        output.append(final_row(
            category="No-teacher supervised base", candidate=f"Compact {family}", uses_teacher="no",
            training_signal="L ground-truth masks only; no U pseudo-labels", row=base, base=base,
            selected=False, note="Direct compact baseline for this student family.",
        ))
    selected_row = rows_by_name[selected_supervised_name]
    output.append(final_row(
        category="Selected supervised transfer", candidate=selected_supervised_name.replace("hard_", "Hard ").replace("soft_", "Soft ").replace("_to_", " → "),
        uses_teacher="yes", training_signal="teacher pseudo-labels on U images; 6-feature compact student", row=selected_row, base=rf_base,
        selected=True, note="Selected on validation before test masks were opened.",
    ))
    if selected_hard_row is not None:
        soft_name = f"soft_{selected_supervised_key}"
        output.append(final_row(
            category="Matched soft transfer", candidate=soft_name.replace("soft_", "Soft ").replace("_to_", " → "), uses_teacher="yes",
            training_signal="same teacher/U source budget as hard comparison", row=rows_by_name[soft_name], base=rf_base,
            selected=False, note=f"Matched soft alternative at tau={soft['tau_held_fixed']}.",
        ))
    output.append(final_row(
        category="Unsupervised upper reference", candidate="K-Means teacher K=6", uses_teacher="upper reference",
        training_signal="L∪U image pixels; L masks only for cluster mapping", row=rows_by_name["km_teacher"], base=km_base,
        selected=False, note="K selected by image-only silhouette.",
    ))
    output.append(final_row(
        category="No-teacher K-Means base", candidate="K-Means base (2 clusters)", uses_teacher="no",
        training_signal="L∪U image pixels; L masks only for compact cluster mapping", row=km_base, base=km_base,
        selected=False, note="Independent compact K-Means comparator.",
    ))
    output.append(final_row(
        category="Selected K-Means transfer", candidate=selected_unsupervised_name.replace("km_", "K-Means ").replace("__", " / "), uses_teacher="yes",
        training_signal="K=6 teacher cluster assignments plus L mapping", row=rows_by_name[selected_unsupervised_name], base=km_base,
        selected=True, note="Selected among teacher-initialized variants on validation only.",
    ))
    return output


def main() -> None:
    development = json.loads((TABLES / "development_selection.json").read_text())
    final = json.loads((TABLES / "final_results.json").read_text())
    validation_rows = build_validation_rows(development)
    final_rows = build_final_rows(final)
    assert len(validation_rows) == 24, len(validation_rows)
    assert len(final_rows) == 9, len(final_rows)
    write_rows_csv(TABLES / "validation_pathway_comparison.csv", validation_rows)
    write_rows_csv(TABLES / "baseline_vs_transfer_final_test.csv", final_rows)
    print("Wrote validation_pathway_comparison.csv (24 development candidates)")
    print("Wrote baseline_vs_transfer_final_test.csv (9 final comparison rows)")


if __name__ == "__main__":
    main()
