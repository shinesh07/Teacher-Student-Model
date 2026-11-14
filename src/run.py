"""Leak-aware command-line stages for the CS3103 flood segmentation project.

``audit`` validates raw data, ``development`` fits/selects with L/U/validation only,
and ``final`` is the only stage allowed to load test masks after selection is frozen.
"""

from __future__ import annotations

import argparse
import json
import pickle
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .artifacts import (
    plot_kmeans_selection,
    plot_tau_study,
    write_json,
    write_pipeline_diagram,
    write_rows_csv,
)
from .audit import audit_dataset
from .compliance import write_checkpoint_manifest
from .config import PROJECT_ROOT, config_digest, load_config
from .data import FloodDataset, dataset_from_config
from .evaluation import AggregateEvaluation, evaluate_model_on_images, write_qualitative_panel
from .experiment import (
    best_hard_distillation_key,
    build_development_pools,
    train_hard_distilled_students,
    train_no_teacher_students,
    train_soft_distilled_best_pair,
    train_supervised_teachers,
    train_unsupervised_track,
)
from .metrics import (
    agreement,
    cluster_similarity,
    compression_ratio,
    efficiency_score,
    evaluate_binary_segmentation,
    percentage_point_gain,
    retention_percent,
    speedup,
)
from .modeling import FittedPixelModel
from .provenance import EventLogger, sha256_file, write_json_atomic


RESULTS = PROJECT_ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
CHECKPOINTS = RESULTS / "checkpoints"
PROVENANCE = RESULTS / "provenance"
MODELS = PROJECT_ROOT / "models"
BUNDLE_PATH = MODELS / "development_bundle.pkl"
FROZEN_PATH = RESULTS / "frozen_selection.json"


class StageError(RuntimeError):
    """Raised when a stage cannot safely start or complete."""


def _prepare_directories() -> None:
    for directory in (TABLES, FIGURES, CHECKPOINTS, PROVENANCE, MODELS):
        directory.mkdir(parents=True, exist_ok=True)


def _save_pickle_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary = Path(handle.name)
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise StageError(f"Required artifact does not exist: {path}") from error
    if not isinstance(value, dict):
        raise StageError(f"Expected a JSON object in {path}")
    return value


def _model_descriptor(model: Any) -> dict[str, Any]:
    if isinstance(model, FittedPixelModel):
        return {
            "model_class": type(model.raw_model).__name__, "family": model.family,
            "role": model.role, "feature_set": model.feature_set,
            "feature_dimensions": model.feature_count, "threshold": model.threshold,
            "fit_pixels": model.fit_count, "task": model.task,
        }
    return {
        "model_class": type(model).__name__, "family": "kmeans",
        "role": "teacher" if int(getattr(model, "n_clusters", 0)) >= 6 else "student",
        "feature_set": None, "feature_dimensions": int(getattr(model, "n_features_in_", 0)),
        "threshold": None, "fit_pixels": None, "task": "clustering",
    }


def _validation_score(record: dict[str, Any]) -> tuple[float, float]:
    selected = record["selected"]
    return float(selected["validation"]["iou"]), float(selected["validation"]["f1"])


def _development_selection(
    hard_records: dict[str, dict[str, Any]], soft_key: str, soft_record: dict[str, Any], unsup_records: dict[str, Any]
) -> dict[str, Any]:
    """Freeze all choices from validation records only, before test access."""
    hard_key = best_hard_distillation_key(hard_records)
    hard_iou, hard_f1 = _validation_score(hard_records[hard_key])
    soft_iou, soft_f1 = float(soft_record["validation"]["iou"]), float(soft_record["validation"]["f1"])
    best_supervised = (
        {"kind": "soft", "key": soft_key, "validation_iou": soft_iou, "validation_f1": soft_f1}
        if (soft_iou, soft_f1) > (hard_iou, hard_f1)
        else {"kind": "hard", "key": hard_key, "validation_iou": hard_iou, "validation_f1": hard_f1}
    )
    return {
        "selection_scope": "L ground truth, L-union-U images and validation only; no test mask opened",
        "best_hard_supervised_key": hard_key,
        "soft_comparison_pair": soft_key,
        "best_supervised_student": best_supervised,
        "best_unsupervised_variant": unsup_records["selected_variant_key"],
        "kmeans_teacher_selection": "maximum silhouette on fixed L-union-U image-only sample",
    }


def _feature_rows(config: dict[str, Any], timing: dict[str, float | int]) -> list[dict[str, Any]]:
    justifications = {
        "red": "Direct RGB colour.", "green": "Direct RGB colour.", "blue": "Direct RGB colour.",
        "hue": "Colour family independent of brightness.", "saturation": "Colour purity cue.",
        "value": "Brightness cue.",
        "blurred_red_sigma1": "Fine local colour context.", "blurred_green_sigma1": "Fine local colour context.",
        "blurred_blue_sigma1": "Fine local colour context.",
        "blurred_red_sigma3": "Broader local colour context.", "blurred_green_sigma3": "Broader local colour context.",
        "blurred_blue_sigma3": "Broader local colour context.",
        "gray_mean_7": "Fine local smoothness.", "gray_std_7": "Fine texture variation.",
        "gray_mean_15": "Coarse local smoothness.", "gray_std_15": "Coarse texture variation.",
        "sobel_magnitude": "Edge strength.", "laplacian": "Second-order texture/edge response.",
        "row_position": "Normalized vertical location.", "column_position": "Normalized horizontal location.",
    }
    rows: list[dict[str, Any]] = []
    for feature_set in ("student", "teacher"):
        names = config["features"][feature_set]
        for dimension, feature in enumerate(names, start=1):
            rows.append({
                "feature_set": feature_set, "dimension": dimension, "feature": feature,
                "justification": justifications[feature], "set_dimensions": len(names),
                "mean_extraction_seconds_per_image": timing[f"{feature_set}_mean_seconds_per_image"],
            })
    return rows


def run_audit(config: dict[str, Any], dataset: FloodDataset) -> dict[str, Any]:
    _prepare_directories()
    report = audit_dataset(dataset, config, FIGURES)
    write_json(CHECKPOINTS / "CP01_dataset_audit.json", report)
    write_json(TABLES / "dataset_card.json", report)
    write_rows_csv(TABLES / "cross_partition_duplicate_images.csv", report["image_inventory"]["cross_partition_duplicate_groups"])
    return report


def run_development(config: dict[str, Any], dataset: FloodDataset) -> dict[str, Any]:
    """Train every required development model without opening a test mask."""
    _prepare_directories()
    structure = dataset.validate_structure()
    events = EventLogger(PROVENANCE / "events.jsonl", f"development-{config_digest(config)[:12]}")
    pools = build_development_pools(dataset, config, events)
    teachers, teacher_records = train_supervised_teachers(pools, config, events)
    baselines, baseline_records = train_no_teacher_students(pools, config, events)
    hard_models, hard_records, hard_sets = train_hard_distilled_students(pools, teachers, config, events)
    soft_key, soft_model, soft_record = train_soft_distilled_best_pair(
        pools, config, teachers, hard_models, hard_records, hard_sets, events
    )
    unsup_artifacts, unsup_records = train_unsupervised_track(pools, dataset, config, events)
    selection = _development_selection(hard_records, soft_key, soft_record, unsup_records)
    records = {
        "structure": structure,
        "feature_timing": pools.feature_timing,
        "sampling": {
            "labelled": pools.labelled_samples,
            "labelled_cluster_mapping": pools.labelled_mapping_samples,
            "transfer": pools.transfer_samples,
            "validation": pools.validation_samples,
        },
        "supervised_teachers": teacher_records,
        "supervised_no_teacher_baselines": baseline_records,
        "hard_distillation": hard_records,
        "soft_distillation": soft_record,
        "unsupervised": unsup_records,
        "selection": selection,
    }
    bundle = {
        "config_sha256": config_digest(config),
        "teachers": teachers,
        "baselines": baselines,
        "hard_models": hard_models,
        "soft_key": soft_key,
        "soft_model": soft_model,
        "unsupervised": unsup_artifacts,
        "records": records,
    }
    _save_pickle_atomic(BUNDLE_PATH, bundle)
    bundle_sha256 = sha256_file(BUNDLE_PATH)
    frozen = {
        "status": "FROZEN_DEVELOPMENT_SELECTION", "config_sha256": config_digest(config),
        "bundle_path": str(BUNDLE_PATH.relative_to(PROJECT_ROOT)), "bundle_sha256": bundle_sha256,
        "selection": selection,
    }
    write_json(FROZEN_PATH, frozen)
    write_json(TABLES / "development_selection.json", records)
    write_json(TABLES / "feature_schema.json", {
        "teacher": config["features"]["teacher"], "student": config["features"]["student"],
        "feature_timing": pools.feature_timing,
    })
    write_rows_csv(TABLES / "feature_table.csv", _feature_rows(config, pools.feature_timing))
    write_pipeline_diagram(FIGURES / "pipeline_architecture.png")
    write_rows_csv(TABLES / "supervised_teacher_validation.csv", [
        {"model": family, **record["selected"]["validation"], "threshold": record["selected"]["threshold"],
         "fit_pixels": record["selected"]["fit_pixels"], "parameters": record["selected"]["parameters"]}
        for family, record in teacher_records.items()
    ])
    write_rows_csv(TABLES / "supervised_baseline_validation.csv", [
        {"model": family, **record["selected"]["validation"], "threshold": record["selected"]["threshold"],
         "fit_pixels": record["selected"]["fit_pixels"], "parameters": record["selected"]["parameters"]}
        for family, record in baseline_records.items()
    ])
    tau_rows: list[dict[str, Any]] = []
    for key, record in hard_records.items():
        for row in record["tau_study"]:
            tau_rows.append({"pair": key, **row})
    write_rows_csv(TABLES / "hard_distillation_tau_validation.csv", tau_rows)
    # The brief asks for a curve for at least one pair. Use the pre-test best hard cell.
    plot_tau_study(hard_records[selection["best_hard_supervised_key"]]["tau_study"], FIGURES / "tau_study.png")
    write_rows_csv(TABLES / "kmeans_teacher_selection.csv", unsup_records["teacher_trials"])
    plot_kmeans_selection(unsup_records["teacher_trials"], FIGURES / "kmeans_teacher_selection.png")
    manifest = write_checkpoint_manifest(
        "CP03_development_selection", config,
        {
            "structure": structure, "bundle_path": str(BUNDLE_PATH), "bundle_sha256": bundle_sha256,
            "frozen_manifest": str(FROZEN_PATH), "selected": selection,
            "models": {
                "teachers": {key: _model_descriptor(value) for key, value in teachers.items()},
                "baselines": {key: _model_descriptor(value) for key, value in baselines.items()},
                "hard_students": {key: _model_descriptor(value) for key, value in hard_models.items()},
                "soft_student": _model_descriptor(soft_model),
            },
        }, output_dir=CHECKPOINTS,
    )
    events.record(
        "development", "freeze_selection", mask_ids_read=[], test_masks_opened=False,
        checkpoint=str(manifest), frozen_manifest=str(FROZEN_PATH), bundle_sha256=bundle_sha256,
    )
    return {"records": records, "bundle_sha256": bundle_sha256, "frozen": frozen, "checkpoint": str(manifest)}


def _load_frozen_bundle(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify freeze/config/model integrity before final test labels can be read."""
    frozen = _load_json(FROZEN_PATH)
    expected_hash = config_digest(config)
    if frozen.get("status") != "FROZEN_DEVELOPMENT_SELECTION":
        raise StageError("Development selection has not been frozen")
    if frozen.get("config_sha256") != expected_hash:
        raise StageError("Frozen selection was built from a different experiment config")
    if not BUNDLE_PATH.is_file():
        raise StageError("Frozen selection references a missing fitted-model bundle")
    if frozen.get("bundle_sha256") != sha256_file(BUNDLE_PATH):
        raise StageError("Fitted-model bundle hash differs from its frozen manifest")
    with BUNDLE_PATH.open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, dict) or bundle.get("config_sha256") != expected_hash:
        raise StageError("Fitted-model bundle is incompatible with the frozen experiment config")
    return frozen, bundle


def _load_final_test_data(dataset: FloodDataset) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    """Load every final-test image and mask after freeze verification only."""
    names = list(dataset.splits().test)
    images = [dataset.load_image(name) for name in names]
    masks = [dataset.load_mask(name, "test", "final_evaluation") for name in names]
    if len(names) != 58 or any(image.shape != (256, 256, 3) for image in images):
        raise StageError("Final test split does not contain the expected 58 processed RGB images")
    if any(mask.shape != (256, 256) for mask in masks):
        raise StageError("Final test masks do not have the required processed shape")
    return names, images, masks


def _prediction_vector(evaluation: AggregateEvaluation) -> np.ndarray:
    return np.concatenate([segment.prediction.reshape(-1) for segment in evaluation.segments]).astype(np.uint8)


def _assignment_vector(evaluation: AggregateEvaluation) -> np.ndarray:
    assignments = [segment.raw_assignments for segment in evaluation.segments]
    if any(value is None for value in assignments):
        raise StageError("Raw cluster assignments are required for this K-Means comparison")
    return np.concatenate([np.asarray(value).reshape(-1) for value in assignments if value is not None]).astype(np.int64)


def _test_row(
    name: str, track: str, group: str, model: Any, evaluation: AggregateEvaluation
) -> dict[str, Any]:
    metrics = evaluation.metrics
    descriptor = _model_descriptor(model)
    return {
        "name": name, "track": track, "group": group, **descriptor,
        **metrics,
        "precision_percent": 100.0 * float(metrics["precision"]),
        "recall_percent": 100.0 * float(metrics["recall"]),
        "f1_percent": 100.0 * float(metrics["f1"]),
        "iou_percent": 100.0 * float(metrics["iou"]),
        "model_size_bytes": evaluation.raw_model_size_bytes,
        "mean_segmentation_seconds_per_image": evaluation.mean_seconds_per_image,
        "mean_feature_seconds_per_image": evaluation.mean_feature_seconds_per_image,
    }


def _evaluate_candidate(
    *,
    name: str,
    track: str,
    group: str,
    model: Any,
    feature_set: str,
    images: list[np.ndarray],
    masks: list[np.ndarray],
    mapping: Any = None,
    chunk_size: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray | None, AggregateEvaluation]:
    evaluation = evaluate_model_on_images(
        images, masks, model, feature_set, mapping=mapping, chunk_size=chunk_size
    )
    expected = len(images) * 256 * 256
    if int(evaluation.metrics["n_pixels"]) != expected:
        raise StageError(f"{name} did not evaluate every test pixel: {evaluation.metrics['n_pixels']} != {expected}")
    assignments = _assignment_vector(evaluation) if track == "unsupervised" else None
    return _test_row(name, track, group, model, evaluation), _prediction_vector(evaluation), assignments, evaluation


def _comparison_fields(teacher: dict[str, Any], student: dict[str, Any], agreement_value: float) -> dict[str, float]:
    """Recompute every derived teacher-student quantity from final raw results."""
    return {
        "agreement": agreement_value,
        "retention_percent": retention_percent(float(student["iou"]), float(teacher["iou"])),
        "compression_ratio": compression_ratio(float(teacher["model_size_bytes"]), float(student["model_size_bytes"])),
        "speedup": speedup(
            float(teacher["mean_segmentation_seconds_per_image"]),
            float(student["mean_segmentation_seconds_per_image"]),
        ),
    }


def _majority_baseline(masks: list[np.ndarray], labelled_flood_fraction: float) -> dict[str, Any]:
    """Simple generic baseline selected from L prevalence only, scored only at final stage."""
    value = int(labelled_flood_fraction >= 0.5)
    truth = np.concatenate([mask.reshape(-1) for mask in masks])
    prediction = np.full(truth.shape, value, dtype=np.uint8)
    metrics = evaluate_binary_segmentation(truth, prediction)
    return {
        "name": "majority_pixel_baseline", "track": "generic", "group": "simple_baseline",
        "model_class": "ConstantPixelPredictor", "family": "constant", "role": "baseline",
        "feature_set": "none", "feature_dimensions": 0, "threshold": None, "fit_pixels": None,
        "task": "classification", **metrics,
        "precision_percent": 100.0 * float(metrics["precision"]),
        "recall_percent": 100.0 * float(metrics["recall"]),
        "f1_percent": 100.0 * float(metrics["f1"]), "iou_percent": 100.0 * float(metrics["iou"]),
        "model_size_bytes": 0, "mean_segmentation_seconds_per_image": 0.0,
        "mean_feature_seconds_per_image": 0.0,
        "labelled_flood_fraction_used": labelled_flood_fraction, "constant_prediction": value,
    }


def run_final(config: dict[str, Any], dataset: FloodDataset) -> dict[str, Any]:
    """Perform one frozen full-test evaluation of every report-required candidate."""
    _prepare_directories()
    # This check must remain above `_load_final_test_data`: it is the concrete barrier
    # that prevents test labels from being opened during an unfrozen experiment.
    frozen, bundle = _load_frozen_bundle(config)
    records = bundle["records"]
    audit_manifest = _load_json(CHECKPOINTS / "CP01_dataset_audit.json")
    if audit_manifest.get("config_sha256") != config_digest(config):
        raise StageError("Dataset audit is stale for the current config; rerun --stage audit before final")
    events = EventLogger(PROVENANCE / "events.jsonl", f"final-{config_digest(config)[:12]}")
    names, images, masks = _load_final_test_data(dataset)
    events.record(
        "final", "load_test_masks_once_after_freeze", image_ids=names, mask_ids_read=names,
        mask_purpose="final_evaluation", frozen_manifest=str(FROZEN_PATH),
    )
    chunk_size = int(config["execution"]["prediction_chunk_pixels"])
    rows_by_name: dict[str, dict[str, Any]] = {}
    predictions: dict[str, np.ndarray] = {}
    assignments: dict[str, np.ndarray] = {}
    evaluations: dict[str, AggregateEvaluation] = {}

    def evaluate_and_keep(
        name: str, track: str, group: str, model: Any, feature_set: str, mapping: Any = None
    ) -> None:
        row, prediction, raw_assignments, evaluated = _evaluate_candidate(
            name=name, track=track, group=group, model=model, feature_set=feature_set,
            images=images, masks=masks, mapping=mapping, chunk_size=chunk_size,
        )
        rows_by_name[name] = row
        predictions[name] = prediction
        if raw_assignments is not None:
            assignments[name] = raw_assignments
        evaluations[name] = evaluated
        events.record(
            "final", "evaluate_frozen_candidate", candidate=name, track=track,
            pixels=int(row["n_pixels"]), mean_seconds_per_image=row["mean_segmentation_seconds_per_image"],
        )

    # Part B: upper references and compact GT-only lower references.
    for family, model in bundle["teachers"].items():
        evaluate_and_keep(f"teacher_{family}", "supervised", "teacher", model, "teacher")
    for family, model in bundle["baselines"].items():
        evaluate_and_keep(f"baseline_{family}", "supervised", "no_teacher_baseline", model, "student")
    for key, model in bundle["hard_models"].items():
        evaluate_and_keep(f"hard_{key}", "supervised", "hard_distilled", model, "student")
    evaluate_and_keep(f"soft_{bundle['soft_key']}", "supervised", "soft_distilled", bundle["soft_model"], "student")

    # Part C: the rich K-Means teacher, independent compact baseline, and all four
    # teacher-initialized variants selected earlier without test labels.
    unsup = bundle["unsupervised"]
    evaluate_and_keep("km_teacher", "unsupervised", "teacher", unsup["teacher"], "teacher", unsup["teacher_mapping"])
    evaluate_and_keep("km_baseline", "unsupervised", "no_teacher_baseline", unsup["baseline"], "student", unsup["baseline_mapping"])
    for key, variant in unsup["variants"].items():
        evaluate_and_keep(f"km_{key}", "unsupervised", "distilled_variant", variant["model"], "student", variant["mapping"])

    majority = _majority_baseline(
        masks, float(audit_manifest["class_statistics"]["labelled"]["flood_pixel_fraction"])
    )
    rows_by_name[majority["name"]] = majority

    hard_rows: list[dict[str, Any]] = []
    hard_matrix: list[dict[str, Any]] = []
    for teacher_family in ("knn", "random_forest", "gaussian_nb"):
        matrix_row: dict[str, Any] = {"teacher": teacher_family}
        for student_family in ("knn", "random_forest", "gaussian_nb"):
            key = f"{teacher_family}_to_{student_family}"
            teacher_row, student_row = rows_by_name[f"teacher_{teacher_family}"], rows_by_name[f"hard_{key}"]
            agreement_value = agreement(predictions[f"teacher_{teacher_family}"], predictions[f"hard_{key}"])
            comparison = _comparison_fields(teacher_row, student_row, agreement_value)
            row = {
                "pair": key, "teacher": teacher_family, "student": student_family,
                "selected_tau": records["hard_distillation"][key]["selected_pseudo_training"]["tau"],
                "test_precision": student_row["precision"], "test_recall": student_row["recall"],
                "test_f1": student_row["f1"], "test_iou": student_row["iou"],
                "test_f1_percent": student_row["f1_percent"], "test_iou_percent": student_row["iou_percent"],
                "delta_iou_percentage_points": percentage_point_gain(student_row["iou"], rows_by_name[f"baseline_{student_family}"]["iou"]),
                "delta_f1_percentage_points": percentage_point_gain(student_row["f1"], rows_by_name[f"baseline_{student_family}"]["f1"]),
                **comparison,
            }
            hard_rows.append(row)
            matrix_row[f"{student_family}_student_iou_percent"] = student_row["iou_percent"]
        hard_matrix.append(matrix_row)

    soft_key = bundle["soft_key"]
    soft_teacher = records["soft_distillation"]["teacher_family"]
    soft_student = records["soft_distillation"]["student_family"]
    soft_name = f"soft_{soft_key}"
    hard_name = f"hard_{soft_key}"
    soft_comparison = {
        "pair": soft_key, "teacher": soft_teacher, "student": soft_student,
        "hard_f1": rows_by_name[hard_name]["f1"], "hard_iou": rows_by_name[hard_name]["iou"],
        "soft_f1": rows_by_name[soft_name]["f1"], "soft_iou": rows_by_name[soft_name]["iou"],
        "hard_f1_percent": rows_by_name[hard_name]["f1_percent"], "hard_iou_percent": rows_by_name[hard_name]["iou_percent"],
        "soft_f1_percent": rows_by_name[soft_name]["f1_percent"], "soft_iou_percent": rows_by_name[soft_name]["iou_percent"],
        "f1_change_percentage_points": percentage_point_gain(rows_by_name[soft_name]["f1"], rows_by_name[hard_name]["f1"]),
        "iou_change_percentage_points": percentage_point_gain(rows_by_name[soft_name]["iou"], rows_by_name[hard_name]["iou"]),
        "method": records["soft_distillation"]["method"],
        "tau_held_fixed": records["soft_distillation"]["tau_held_fixed"],
    }

    kmeans_rows: list[dict[str, Any]] = []
    teacher_clusters = assignments["km_teacher"]
    for name in ["km_baseline", *[f"km_{key}" for key in unsup["variants"]]]:
        student_row = rows_by_name[name]
        similarity = cluster_similarity(teacher_clusters, assignments[name])
        comparison = _comparison_fields(rows_by_name["km_teacher"], student_row, agreement(predictions["km_teacher"], predictions[name]))
        kmeans_rows.append({
            "model": name, "test_precision": student_row["precision"], "test_recall": student_row["recall"],
            "test_f1": student_row["f1"], "test_iou": student_row["iou"],
            "test_f1_percent": student_row["f1_percent"], "test_iou_percent": student_row["iou_percent"],
            "ari": similarity["ari"], "nmi": similarity["nmi"], **comparison,
        })

    selection = frozen["selection"]
    if selection["best_supervised_student"]["kind"] == "soft":
        selected_supervised_name = f"soft_{selection['best_supervised_student']['key']}"
    else:
        selected_supervised_name = f"hard_{selection['best_supervised_student']['key']}"
    selected_supervised_key = selection["best_supervised_student"]["key"]
    selected_supervised_teacher = records["hard_distillation"][selected_supervised_key]["teacher_family"]
    selected_unsupervised_name = f"km_{selection['best_unsupervised_variant']}"
    supervised_summary = {
        "track": "supervised", "teacher": f"teacher_{selected_supervised_teacher}",
        "student": selected_supervised_name, **_comparison_fields(
            rows_by_name[f"teacher_{selected_supervised_teacher}"], rows_by_name[selected_supervised_name],
            agreement(predictions[f"teacher_{selected_supervised_teacher}"], predictions[selected_supervised_name]),
        ),
        "f1_student": rows_by_name[selected_supervised_name]["f1"],
        "iou_student": rows_by_name[selected_supervised_name]["iou"],
    }
    supervised_summary["z"] = efficiency_score(supervised_summary["f1_student"], supervised_summary["iou_student"])
    unsupervised_summary = {
        "track": "unsupervised", "teacher": "km_teacher", "student": selected_unsupervised_name,
        **_comparison_fields(rows_by_name["km_teacher"], rows_by_name[selected_unsupervised_name], agreement(predictions["km_teacher"], predictions[selected_unsupervised_name])),
        "f1_student": rows_by_name[selected_unsupervised_name]["f1"],
        "iou_student": rows_by_name[selected_unsupervised_name]["iou"],
    }
    unsupervised_summary["z"] = efficiency_score(unsupervised_summary["f1_student"], unsupervised_summary["iou_student"])
    track_summary = [supervised_summary, unsupervised_summary]

    # Required qualitative evidence: four fixed test images, each with RGB, GT,
    # frozen best supervised teacher/student and K-Means teacher/student masks.
    qualitative_paths: list[str] = []
    for image_index, filename in enumerate(names[:4]):
        reshaped = {
            "Best supervised student": predictions[selected_supervised_name].reshape(58, 256, 256)[image_index],
            "Best supervised teacher": predictions[f"teacher_{selected_supervised_teacher}"].reshape(58, 256, 256)[image_index],
            "K-Means student": predictions[selected_unsupervised_name].reshape(58, 256, 256)[image_index],
            "K-Means teacher": predictions["km_teacher"].reshape(58, 256, 256)[image_index],
        }
        destination = FIGURES / f"qualitative_{image_index + 1}_{Path(filename).stem}.png"
        write_qualitative_panel(images[image_index], masks[image_index], reshaped, destination, title=f"Test image {filename}")
        qualitative_paths.append(str(destination))

    error_rows: list[dict[str, Any]] = []
    selected_for_errors = {
        "supervised_teacher": f"teacher_{selected_supervised_teacher}",
        "supervised_student": selected_supervised_name,
        "kmeans_teacher": "km_teacher",
        "kmeans_student": selected_unsupervised_name,
    }
    for image_index, filename in enumerate(names):
        truth = masks[image_index]
        for label, candidate in selected_for_errors.items():
            metric = evaluate_binary_segmentation(truth, predictions[candidate].reshape(58, 256, 256)[image_index])
            error_rows.append({"filename": filename, "model_role": label, "candidate": candidate, **metric})

    all_rows = list(rows_by_name.values())
    write_rows_csv(TABLES / "final_metrics_all.csv", all_rows)
    write_rows_csv(TABLES / "supervised_teachers_test.csv", [row for row in all_rows if row["group"] == "teacher" and row["track"] == "supervised"])
    write_rows_csv(TABLES / "supervised_baselines_test.csv", [row for row in all_rows if row["group"] == "no_teacher_baseline" and row["track"] == "supervised"])
    write_rows_csv(TABLES / "hard_distillation_test_long.csv", hard_rows)
    write_rows_csv(TABLES / "hard_distillation_iou_matrix.csv", hard_matrix)
    write_rows_csv(TABLES / "soft_vs_hard_test.csv", [soft_comparison])
    write_rows_csv(TABLES / "kmeans_test_results.csv", kmeans_rows)
    write_rows_csv(TABLES / "track_summary.csv", track_summary)
    write_rows_csv(TABLES / "selected_model_error_analysis.csv", error_rows)
    result = {
        "frozen_selection": selection, "test_image_ids": names, "test_pixel_count": 58 * 256 * 256,
        "models": all_rows, "hard_distillation": hard_rows, "soft_vs_hard": soft_comparison,
        "kmeans": kmeans_rows, "track_summary": track_summary, "qualitative_figures": qualitative_paths,
        "error_analysis_csv": str(TABLES / "selected_model_error_analysis.csv"),
    }
    write_json(TABLES / "final_results.json", result)
    checkpoint = write_checkpoint_manifest(
        "CP04_final_test", config,
        {
            "frozen_manifest": str(FROZEN_PATH), "bundle_sha256": frozen["bundle_sha256"],
            "test_images": len(names), "test_pixels": 58 * 256 * 256,
            "result_json": str(TABLES / "final_results.json"), "track_summary": track_summary,
            "qualitative_figures": qualitative_paths,
        }, output_dir=CHECKPOINTS,
    )
    events.record(
        "final", "complete_final_evaluation", mask_ids_read=names, checkpoint=str(checkpoint),
        test_pixels=58 * 256 * 256, selected_track_summary=track_summary,
    )
    return {"result": result, "checkpoint": str(checkpoint)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("audit", "development", "final", "all"), required=True,
        help="Experiment stage to execute. 'final' refuses an unfrozen development selection.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional experiment JSON path.")
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = dataset_from_config(config)
    summary: dict[str, Any] = {}
    if args.stage in {"audit", "all"}:
        summary["audit"] = run_audit(config, dataset)
    if args.stage in {"development", "all"}:
        summary["development"] = run_development(config, dataset)
    if args.stage in {"final", "all"}:
        summary["final"] = run_final(config, dataset)
    print(json.dumps({"completed_stages": list(summary), "config_sha256": config_digest(config)}, indent=2))


if __name__ == "__main__":
    main()
