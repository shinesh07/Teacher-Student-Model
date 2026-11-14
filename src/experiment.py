"""Development-stage training for the reproducible flood-segmentation project.

This module deliberately does not expose test-set evaluation. It may read validation
masks to choose configurations, and labelled masks to fit permitted models, but it
never requests test-mask access. Final testing is isolated in ``src.run``.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable, Literal

import numpy as np
from sklearn.metrics import silhouette_score

from .data import FloodDataset
from .distillation import (
    DistillationError,
    PseudoTrainingSet,
    select_pseudo_training_set,
    weighted_soft_duplication,
)
from .features import (
    STUDENT_FEATURE_NAMES,
    TEACHER_FEATURE_NAMES,
    extract_features,
    extract_with_timing,
    validate_feature_budget,
)
from .metrics import evaluate_binary_segmentation
from .modeling import FittedPixelModel, fit_pixel_model
from .compliance import assert_pair_capacity
from .provenance import EventLogger
from .sampling import PixelSelection, select_pixel_indices
from .unsupervised import (
    ClusterMapping,
    FrozenCentroidPredictor,
    derive_teacher_grouping,
    fit_student_kmeans,
    fit_teacher_kmeans,
    make_distilled_student,
    majority_cluster_mapping,
)


SupervisedFamily = Literal["knn", "random_forest", "gaussian_nb"]


class DevelopmentContractError(RuntimeError):
    """Raised when a development-only run would violate an experiment invariant."""


@dataclass
class DevelopmentPools:
    """Sampled arrays retained for development fitting and validation only."""

    labelled_teacher: np.ndarray
    labelled_student: np.ndarray
    labelled_targets: np.ndarray
    labelled_samples: list[dict[str, object]]
    labelled_mapping_teacher: np.ndarray
    labelled_mapping_student: np.ndarray
    labelled_mapping_targets: np.ndarray
    labelled_mapping_samples: list[dict[str, object]]
    transfer_teacher: np.ndarray
    transfer_student: np.ndarray
    transfer_samples: list[dict[str, object]]
    validation_teacher: np.ndarray
    validation_student: np.ndarray
    validation_targets: np.ndarray
    validation_samples: list[dict[str, object]]
    feature_timing: dict[str, float | int]


def _validate_feature_schema(config: dict[str, Any]) -> None:
    """Ensure configuration and extractor schemas cannot silently drift."""
    if tuple(config["features"]["student"]) != STUDENT_FEATURE_NAMES:
        raise DevelopmentContractError("Student feature schema differs from implementation")
    if tuple(config["features"]["teacher"]) != TEACHER_FEATURE_NAMES:
        raise DevelopmentContractError("Teacher feature schema differs from implementation")
    probe = np.zeros((2, 2, 3), dtype=np.float32)
    validate_feature_budget(extract_features(probe, "student"), extract_features(probe, "teacher"))


def _select_rows(
    *,
    image_id: str,
    teacher_features: np.ndarray,
    student_features: np.ndarray,
    count: int,
    seed: int,
    namespace: str,
    labels: np.ndarray | None = None,
    balanced: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, object]]:
    """Select identical pixels from rich and compact feature matrices."""
    if teacher_features.shape[0] != student_features.shape[0]:
        raise DevelopmentContractError("Teacher and student matrices have different pixel counts")
    flattened_labels = None if labels is None else labels.reshape(-1)
    if flattened_labels is not None and flattened_labels.shape[0] != teacher_features.shape[0]:
        raise DevelopmentContractError("Labels are not aligned to the feature matrices")
    selection: PixelSelection = select_pixel_indices(
        image_id=image_id,
        population_size=teacher_features.shape[0],
        count=count,
        base_seed=seed,
        strategy="balanced" if balanced else "uniform",
        labels=flattened_labels if balanced else None,
        namespace=namespace,
    )
    index = selection.pixel_indices
    selected_labels = None if flattened_labels is None else flattened_labels[index].astype(np.uint8, copy=False)
    return teacher_features[index], student_features[index], selected_labels, selection.as_manifest()


def build_development_pools(
    dataset: FloodDataset, config: dict[str, Any], events: EventLogger | None = None
) -> DevelopmentPools:
    """Build sampled development arrays while enforcing split-specific mask access.

    L samples use balanced ground-truth sampling for fitting. L union U transfer samples
    are uniform and intentionally do not open a mask for U. Validation is sampled
    uniformly so its prevalence remains suitable for hyperparameter selection.
    """
    _validate_feature_schema(config)
    seed = int(config["seed"])
    sampling = config["sampling"]
    splits = dataset.splits()

    labelled_teacher: list[np.ndarray] = []
    labelled_student: list[np.ndarray] = []
    labelled_targets: list[np.ndarray] = []
    labelled_samples: list[dict[str, object]] = []
    labelled_mapping_teacher: list[np.ndarray] = []
    labelled_mapping_student: list[np.ndarray] = []
    labelled_mapping_targets: list[np.ndarray] = []
    labelled_mapping_samples: list[dict[str, object]] = []
    teacher_feature_seconds: list[float] = []
    student_feature_seconds: list[float] = []
    for filename in splits.labelled:
        image = dataset.load_image(filename)
        target = dataset.load_mask(filename, "labelled", "teacher_fit")
        rich_extracted, compact_extracted = extract_with_timing(image, "teacher"), extract_with_timing(image, "student")
        rich, compact = rich_extracted.values, compact_extracted.values
        teacher_feature_seconds.append(rich_extracted.elapsed_seconds)
        student_feature_seconds.append(compact_extracted.elapsed_seconds)
        selected_rich, selected_compact, selected_target, manifest = _select_rows(
            image_id=filename,
            teacher_features=rich,
            student_features=compact,
            count=int(sampling["labelled_pixels_per_image"]),
            seed=seed,
            namespace="labelled-ground-truth",
            labels=target,
            balanced=True,
        )
        assert selected_target is not None
        labelled_teacher.append(selected_rich)
        labelled_student.append(selected_compact)
        labelled_targets.append(selected_target)
        labelled_samples.append(manifest)
        map_rich, map_compact, map_target, map_manifest = _select_rows(
            image_id=filename,
            teacher_features=rich,
            student_features=compact,
            count=int(sampling["labelled_pixels_per_image"]),
            seed=seed,
            namespace="labelled-cluster-mapping",
            labels=target,
            balanced=False,
        )
        assert map_target is not None
        labelled_mapping_teacher.append(map_rich)
        labelled_mapping_student.append(map_compact)
        labelled_mapping_targets.append(map_target)
        labelled_mapping_samples.append(map_manifest)
    if events:
        events.record(
            "development", "sample_labelled_ground_truth", image_ids=list(splits.labelled),
            mask_ids_read=list(splits.labelled), mask_purpose="teacher_fit",
            pixels_per_image=int(sampling["labelled_pixels_per_image"]), sampling="balanced",
        )

    transfer_teacher: list[np.ndarray] = []
    transfer_student: list[np.ndarray] = []
    transfer_samples: list[dict[str, object]] = []
    transfer_names = (*splits.labelled, *splits.unlabelled)
    for filename in transfer_names:
        image = dataset.load_image(filename)
        rich_extracted, compact_extracted = extract_with_timing(image, "teacher"), extract_with_timing(image, "student")
        rich, compact = rich_extracted.values, compact_extracted.values
        teacher_feature_seconds.append(rich_extracted.elapsed_seconds)
        student_feature_seconds.append(compact_extracted.elapsed_seconds)
        selected_rich, selected_compact, selected_target, manifest = _select_rows(
            image_id=filename,
            teacher_features=rich,
            student_features=compact,
            count=int(sampling["transfer_pixels_per_image"]),
            seed=seed,
            namespace="transfer-images-only",
        )
        if selected_target is not None:
            raise DevelopmentContractError("Transfer-set code must never receive a target array")
        transfer_teacher.append(selected_rich)
        transfer_student.append(selected_compact)
        transfer_samples.append(manifest)
    if events:
        events.record(
            "development", "sample_transfer_images_only", image_ids=list(transfer_names),
            mask_ids_read=[], mask_purpose=None,
            pixels_per_image=int(sampling["transfer_pixels_per_image"]), sampling="uniform",
        )

    validation_teacher: list[np.ndarray] = []
    validation_student: list[np.ndarray] = []
    validation_targets: list[np.ndarray] = []
    validation_samples: list[dict[str, object]] = []
    for filename in splits.validation:
        image = dataset.load_image(filename)
        target = dataset.load_mask(filename, "validation", "validation")
        rich_extracted, compact_extracted = extract_with_timing(image, "teacher"), extract_with_timing(image, "student")
        rich, compact = rich_extracted.values, compact_extracted.values
        teacher_feature_seconds.append(rich_extracted.elapsed_seconds)
        student_feature_seconds.append(compact_extracted.elapsed_seconds)
        selected_rich, selected_compact, selected_target, manifest = _select_rows(
            image_id=filename,
            teacher_features=rich,
            student_features=compact,
            count=int(sampling["validation_pixels_per_image"]),
            seed=seed,
            namespace="validation-selection",
            labels=target,
        )
        assert selected_target is not None
        validation_teacher.append(selected_rich)
        validation_student.append(selected_compact)
        validation_targets.append(selected_target)
        validation_samples.append(manifest)
    if events:
        events.record(
            "development", "sample_validation_for_selection", image_ids=list(splits.validation),
            mask_ids_read=list(splits.validation), mask_purpose="validation",
            pixels_per_image=int(sampling["validation_pixels_per_image"]), sampling="uniform",
        )

    pools = DevelopmentPools(
        labelled_teacher=np.concatenate(labelled_teacher),
        labelled_student=np.concatenate(labelled_student),
        labelled_targets=np.concatenate(labelled_targets),
        labelled_samples=labelled_samples,
        labelled_mapping_teacher=np.concatenate(labelled_mapping_teacher),
        labelled_mapping_student=np.concatenate(labelled_mapping_student),
        labelled_mapping_targets=np.concatenate(labelled_mapping_targets),
        labelled_mapping_samples=labelled_mapping_samples,
        transfer_teacher=np.concatenate(transfer_teacher),
        transfer_student=np.concatenate(transfer_student),
        transfer_samples=transfer_samples,
        validation_teacher=np.concatenate(validation_teacher),
        validation_student=np.concatenate(validation_student),
        validation_targets=np.concatenate(validation_targets),
        validation_samples=validation_samples,
        feature_timing={
            "images_extracted": len(teacher_feature_seconds),
            "teacher_mean_seconds_per_image": float(np.mean(teacher_feature_seconds)),
            "teacher_median_seconds_per_image": float(np.median(teacher_feature_seconds)),
            "student_mean_seconds_per_image": float(np.mean(student_feature_seconds)),
            "student_median_seconds_per_image": float(np.median(student_feature_seconds)),
        },
    )
    expected_labelled = 40 * int(sampling["labelled_pixels_per_image"])
    expected_transfer = 203 * int(sampling["transfer_pixels_per_image"])
    expected_validation = 29 * int(sampling["validation_pixels_per_image"])
    if pools.labelled_teacher.shape != (expected_labelled, 20):
        raise DevelopmentContractError("Unexpected labelled teacher sample shape")
    if pools.transfer_teacher.shape != (expected_transfer, 20):
        raise DevelopmentContractError("Unexpected transfer sample size")
    if pools.validation_teacher.shape != (expected_validation, 20):
        raise DevelopmentContractError("Unexpected validation sample size")
    return pools


def select_global_rows(
    features: np.ndarray,
    labels: np.ndarray | None,
    *,
    count: int,
    seed: int,
    namespace: str,
    balanced: bool,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, object]]:
    """Deterministically choose a model-specific budget from a sampled pool."""
    selection = select_pixel_indices(
        image_id=f"pooled::{namespace}", population_size=features.shape[0], count=count,
        base_seed=seed, strategy="balanced" if balanced else "uniform",
        labels=labels if balanced else None, namespace=namespace,
    )
    picked_labels = None if labels is None else labels[selection.pixel_indices]
    return features[selection.pixel_indices], picked_labels, selection.as_manifest()


def _require_two_classes(targets: np.ndarray, context: str) -> None:
    classes = np.unique(targets)
    if classes.size != 2 or not np.array_equal(np.sort(classes), np.array([0, 1])):
        raise DevelopmentContractError(f"{context} needs both classes, observed {classes.tolist()}")


def _threshold_selection(
    scores: np.ndarray, targets: np.ndarray, thresholds: Iterable[float]
) -> tuple[float, dict[str, float | int], list[dict[str, float | int]]]:
    """Select a threshold by validation IoU with deterministic tie breaks."""
    candidates: list[dict[str, float | int]] = []
    for threshold in thresholds:
        metric = evaluate_binary_segmentation(targets, (scores >= float(threshold)).astype(np.uint8))
        candidates.append({"threshold": float(threshold), **metric})
    if not candidates:
        raise DevelopmentContractError("At least one validation decision threshold is required")
    winner = max(
        candidates,
        key=lambda item: (
            float(item["iou"]), float(item["f1"]),
            -abs(float(item["threshold"]) - 0.5), -float(item["threshold"]),
        ),
    )
    return float(winner["threshold"]), winner, candidates


def _candidate_parameters(
    config: dict[str, Any], family: SupervisedFamily, role: Literal["teacher", "student"]
) -> list[dict[str, Any]]:
    """Expand only the predeclared validation search space from configuration."""
    specification = config["supervised"][role]
    if family == "knn":
        return [{"n_neighbors": int(value)} for value in specification["knn_neighbors"]]
    if family == "random_forest":
        return [
            {"n_estimators": int(specification["rf_trees"]), "max_depth": value}
            for value in specification["rf_max_depth"]
        ]
    if family == "gaussian_nb":
        return [{"var_smoothing": float(value)} for value in specification["gnb_var_smoothing"]]
    raise DevelopmentContractError(f"Unknown supervised family {family!r}")


def _fit_tune_supervised(
    *,
    config: dict[str, Any],
    family: SupervisedFamily,
    role: Literal["teacher", "student"],
    feature_set: Literal["teacher", "student"],
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    label: str,
) -> tuple[FittedPixelModel, dict[str, Any]]:
    """Fit fixed candidates and choose strictly from validation evidence."""
    _require_two_classes(train_targets, f"{label} training")
    trials: list[dict[str, Any]] = []
    best_model: FittedPixelModel | None = None
    best_record: dict[str, Any] | None = None
    for parameters in _candidate_parameters(config, family, role):
        started = perf_counter()
        model = fit_pixel_model(
            family, role, feature_set, train_features, train_targets,
            random_state=int(config["seed"]),
            feature_names=TEACHER_FEATURE_NAMES if feature_set == "teacher" else STUDENT_FEATURE_NAMES,
            **parameters,
        )
        training_seconds = perf_counter() - started
        scores = model.predict_scores(
            validation_features, chunk_size=int(config["execution"]["prediction_chunk_pixels"])
        )
        threshold, best_metric, threshold_trials = _threshold_selection(
            scores, validation_targets, config["supervised"]["decision_thresholds"]
        )
        model.threshold = threshold
        record: dict[str, Any] = {
            "parameters": parameters, "threshold": threshold, "validation": best_metric,
            "threshold_trials": threshold_trials, "training_seconds": training_seconds,
            "capacity": model.capacity_summary(), "fit_pixels": model.fit_count,
        }
        trials.append(record)
        if best_record is None or (
            float(record["validation"]["iou"]), float(record["validation"]["f1"])
        ) > (
            float(best_record["validation"]["iou"]), float(best_record["validation"]["f1"])
        ):
            best_model, best_record = model, record
    if best_model is None or best_record is None:  # pragma: no cover - config guarantees candidates
        raise DevelopmentContractError(f"No candidate was fitted for {label}")
    return best_model, {"label": label, "selected": best_record, "trials": trials}


def fit_budget_for_family(config: dict[str, Any], family: SupervisedFamily, role: str) -> int:
    """Return the declared per-family fitting budget used for fair comparisons."""
    sampling = config["sampling"]
    if role == "teacher" and family == "knn":
        return int(sampling["teacher_knn_reference_pixels"])
    if role == "student" and family == "knn":
        return int(sampling["student_knn_reference_pixels"])
    if role == "teacher":
        return 40 * int(sampling["labelled_pixels_per_image"])
    return int(sampling["student_fit_pixels"])


def train_supervised_teachers(
    pools: DevelopmentPools, config: dict[str, Any], events: EventLogger | None = None
) -> tuple[dict[SupervisedFamily, FittedPixelModel], dict[str, Any]]:
    """Fit KNN-T, RF-T and GNB-T on L truth; select only on validation."""
    models: dict[SupervisedFamily, FittedPixelModel] = {}
    records: dict[str, Any] = {}
    for family in ("knn", "random_forest", "gaussian_nb"):
        count = fit_budget_for_family(config, family, "teacher")
        features, targets, manifest = select_global_rows(
            pools.labelled_teacher, pools.labelled_targets, count=count,
            seed=int(config["seed"]), namespace=f"teacher-{family}", balanced=True,
        )
        assert targets is not None
        model, record = _fit_tune_supervised(
            config=config, family=family, role="teacher", feature_set="teacher",
            train_features=features, train_targets=targets,
            validation_features=pools.validation_teacher, validation_targets=pools.validation_targets,
            label=f"{family}-teacher",
        )
        record["training_sample"] = manifest
        models[family], records[family] = model, record
        if events:
            events.record(
                "development", "fit_supervised_teacher", model_family=family,
                model_fit_ids="labelled_only", mask_ids_read="labelled_only",
                scaler_fit_ids="labelled_only" if family == "knn" else [], fit_pixels=count,
                selected_validation=record["selected"],
            )
    return models, records


def train_no_teacher_students(
    pools: DevelopmentPools, config: dict[str, Any], events: EventLogger | None = None
) -> tuple[dict[SupervisedFamily, FittedPixelModel], dict[str, Any]]:
    """Fit the three GT-only compact-feature baselines under student budgets."""
    models: dict[SupervisedFamily, FittedPixelModel] = {}
    records: dict[str, Any] = {}
    for family in ("knn", "random_forest", "gaussian_nb"):
        count = fit_budget_for_family(config, family, "student")
        features, targets, manifest = select_global_rows(
            pools.labelled_student, pools.labelled_targets, count=count,
            seed=int(config["seed"]), namespace=f"baseline-student-{family}", balanced=True,
        )
        assert targets is not None
        model, record = _fit_tune_supervised(
            config=config, family=family, role="student", feature_set="student",
            train_features=features, train_targets=targets,
            validation_features=pools.validation_student, validation_targets=pools.validation_targets,
            label=f"{family}-no-teacher-student",
        )
        record["training_sample"] = manifest
        models[family], records[family] = model, record
        if events:
            events.record(
                "development", "fit_no_teacher_student", model_family=family,
                model_fit_ids="labelled_only", mask_ids_read="labelled_only",
                scaler_fit_ids="labelled_only" if family == "knn" else [], fit_pixels=count,
                selected_validation=record["selected"],
            )
    return models, records


def _hard_key(teacher_family: SupervisedFamily, student_family: SupervisedFamily) -> str:
    return f"{teacher_family}_to_{student_family}"


def train_hard_distilled_students(
    pools: DevelopmentPools,
    teachers: dict[SupervisedFamily, FittedPixelModel],
    config: dict[str, Any],
    events: EventLogger | None = None,
) -> tuple[
    dict[str, FittedPixelModel],
    dict[str, dict[str, Any]],
    dict[str, PseudoTrainingSet],
]:
    """Create all nine hard-label supervised teacher-to-student cells.

    Each tau candidate has the fixed student fit budget. Candidates unable to supply it
    are recorded as invalid rather than silently shrinking a student and making its
    no-teacher comparison unfair.
    """
    models: dict[str, FittedPixelModel] = {}
    records: dict[str, dict[str, Any]] = {}
    selected_sets: dict[str, PseudoTrainingSet] = {}
    seed = int(config["seed"])
    chunk_size = int(config["execution"]["prediction_chunk_pixels"])
    postprocess = config["postprocessing"]
    for teacher_family, teacher in teachers.items():
        transfer_scores = teacher.predict_scores(pools.transfer_teacher, chunk_size=chunk_size)
        if events:
            events.record(
                "development", "teacher_pseudo_labels_transfer", teacher_family=teacher_family,
                image_ids="labelled_union_unlabelled", mask_ids_read=[], transfer_rows=int(transfer_scores.size),
            )
        for student_family in ("knn", "random_forest", "gaussian_nb"):
            key = _hard_key(teacher_family, student_family)
            capacity = fit_budget_for_family(config, student_family, "student")
            tau_rows: list[dict[str, Any]] = []
            best_model: FittedPixelModel | None = None
            best_record: dict[str, Any] | None = None
            best_set: PseudoTrainingSet | None = None
            for tau in config["supervised"]["confidence_thresholds"]:
                try:
                    pseudo_set = select_pseudo_training_set(
                        pools.transfer_student, transfer_scores, tau=float(tau), capacity=capacity,
                        base_seed=seed, strategy="balanced", namespace=key,
                    )
                except DistillationError as error:
                    tau_rows.append({"tau": float(tau), "valid": False, "reason": str(error)})
                    continue
                model, trial = _fit_tune_supervised(
                    config=config, family=student_family, role="student", feature_set="student",
                    train_features=pseudo_set.features, train_targets=pseudo_set.labels,
                    validation_features=pools.validation_student, validation_targets=pools.validation_targets,
                    label=f"hard-distilled-{key}-tau-{float(tau):.2f}",
                )
                trial["pseudo_training"] = pseudo_set.as_manifest()
                tau_rows.append({
                    "tau": float(tau), "valid": True,
                    "retained_fraction": pseudo_set.retained_fraction,
                    "retained_rows": pseudo_set.retained_count,
                    "validation_iou": trial["selected"]["validation"]["iou"],
                    "validation_f1": trial["selected"]["validation"]["f1"],
                    "student_fit_rows": pseudo_set.selected_count,
                })
                if best_record is None or (
                    float(trial["selected"]["validation"]["iou"]),
                    float(trial["selected"]["validation"]["f1"]),
                ) > (
                    float(best_record["selected"]["validation"]["iou"]),
                    float(best_record["selected"]["validation"]["f1"]),
                ):
                    best_model, best_record, best_set = model, trial, pseudo_set
            if best_model is None or best_record is None or best_set is None:
                raise DevelopmentContractError(f"No viable hard-distillation tau for {key}")
            capacity_evidence = assert_pair_capacity(
                teacher.raw_model, best_model.raw_model,
                teacher_feature_count=teacher.feature_count,
                student_feature_count=best_model.feature_count,
                teacher_fit_pixels=teacher.fit_count,
                student_fit_pixels=best_model.fit_count,
                teacher_postprocessing=postprocess, student_postprocessing=postprocess,
                require_rich_teacher=True,
            )
            record = {
                "teacher_family": teacher_family,
                "student_family": student_family,
                "selected": best_record["selected"],
                "all_parameter_trials_at_selected_tau": best_record["trials"],
                "selected_pseudo_training": best_set.as_manifest(),
                "tau_study": tau_rows,
                "capacity": capacity_evidence,
            }
            models[key], records[key], selected_sets[key] = best_model, record, best_set
            if events:
                events.record(
                    "development", "fit_hard_distilled_student", teacher_family=teacher_family,
                    student_family=student_family, model_fit_ids="teacher_outputs_on_labelled_union_unlabelled",
                    mask_ids_read=[], fit_pixels=best_model.fit_count,
                    selected_tau=best_set.tau, selected_validation=best_record["selected"],
                )
    return models, records, selected_sets


def best_hard_distillation_key(records: dict[str, dict[str, Any]]) -> str:
    """Select the one B.3 cell used for the required controlled soft-label study."""
    if not records:
        raise DevelopmentContractError("No hard-distillation records are available")
    return max(
        records,
        key=lambda key: (
            float(records[key]["selected"]["validation"]["iou"]),
            float(records[key]["selected"]["validation"]["f1"]),
            key,
        ),
    )


def train_soft_distilled_best_pair(
    pools: DevelopmentPools,
    config: dict[str, Any],
    teachers: dict[SupervisedFamily, FittedPixelModel],
    hard_models: dict[str, FittedPixelModel],
    hard_records: dict[str, dict[str, Any]],
    hard_sets: dict[str, PseudoTrainingSet],
    events: EventLogger | None = None,
) -> tuple[str, FittedPixelModel, dict[str, Any]]:
    """Train a soft counterpart to the best hard cell without changing tau or budget."""
    key = best_hard_distillation_key(hard_records)
    hard = hard_models[key]
    pseudo = hard_sets[key]
    teacher = teachers[hard_records[key]["teacher_family"]]
    parameters = dict(hard_records[key]["selected"]["parameters"])
    if hard.family == "knn":
        # sklearn's KNN classifier has no sample weights, so use the allowed regression
        # route and retain the same source-pixel capacity and confidence filter.
        soft = fit_pixel_model(
            "knn", "student", "student", pseudo.features, pseudo.scores,
            task="regression", random_state=int(config["seed"]),
            feature_names=STUDENT_FEATURE_NAMES, **parameters,
        )
        soft_method = "KNeighborsRegressor on teacher probability"
        source_rows = pseudo.selected_count
    else:
        duplicated = weighted_soft_duplication(pseudo.features, pseudo.scores)
        soft = fit_pixel_model(
            hard.family, "student", "student", duplicated.features, duplicated.targets,
            sample_weight=duplicated.sample_weight, random_state=int(config["seed"]),
            feature_names=STUDENT_FEATURE_NAMES, **parameters,
        )
        soft_method = "weighted duplication"
        source_rows = duplicated.source_rows
    scores = soft.predict_scores(
        pools.validation_student, chunk_size=int(config["execution"]["prediction_chunk_pixels"])
    )
    threshold, validation, threshold_trials = _threshold_selection(
        scores, pools.validation_targets, config["supervised"]["decision_thresholds"]
    )
    soft.threshold = threshold
    pair_capacity = assert_pair_capacity(
        teacher.raw_model, soft.raw_model,
        teacher_feature_count=teacher.feature_count, student_feature_count=soft.feature_count,
        teacher_fit_pixels=teacher.fit_count, student_fit_pixels=soft.fit_count,
        teacher_postprocessing=config["postprocessing"], student_postprocessing=config["postprocessing"],
        require_rich_teacher=True,
    )
    record = {
        "hard_pair": key,
        "teacher_family": hard_records[key]["teacher_family"],
        "student_family": hard.family,
        "tau_held_fixed": pseudo.tau,
        "source_pixel_budget_held_fixed": source_rows,
        "parameters_held_fixed": parameters,
        "method": soft_method,
        "threshold": threshold,
        "validation": validation,
        "threshold_trials": threshold_trials,
        "capacity": pair_capacity,
    }
    if events:
        events.record(
            "development", "fit_soft_distilled_student", hard_pair=key,
            model_fit_ids="teacher_probability_outputs_on_labelled_union_unlabelled",
            mask_ids_read=[], source_pixel_budget=source_rows, selected_validation=validation,
        )
    return key, soft, record


def _select_unsupervised_fit_rows(
    pools: DevelopmentPools, count: int, seed: int, namespace: str
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Choose matching rich/compact image-only rows for K-Means fitting."""
    selection = select_pixel_indices(
        image_id="pooled::unsupervised_L_plus_U", population_size=pools.transfer_teacher.shape[0],
        count=count, base_seed=seed, strategy="uniform", namespace=namespace,
    )
    index = selection.pixel_indices
    return pools.transfer_teacher[index], pools.transfer_student[index], selection.as_manifest()


def _cluster_assignments(model: Any, features: np.ndarray, chunk_size: int) -> np.ndarray:
    """Predict raw cluster IDs in chunks for sklearn or frozen-centroid K-Means."""
    values: list[np.ndarray] = []
    for start in range(0, features.shape[0], chunk_size):
        block = features[start : start + chunk_size]
        if isinstance(model, FrozenCentroidPredictor):
            values.append(model.predict(block, chunk_size=chunk_size))
        else:
            values.append(np.asarray(model.predict(block), dtype=np.int64))
    return np.concatenate(values).astype(np.int64, copy=False)


def _mapping_from_labelled_pool(
    model: Any,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    chunk_size: int,
) -> ClusterMapping:
    """Map a fitted clusterer with L-only, uniformly sampled labels."""
    assignments = _cluster_assignments(model, features, chunk_size)
    return majority_cluster_mapping(
        assignments, targets, n_clusters=int(getattr(model, "n_clusters"))
    )


def mapping_from_all_labelled_images(
    model: Any,
    dataset: FloodDataset,
    feature_set: Literal["teacher", "student"],
    *,
    chunk_size: int,
    events: EventLogger | None = None,
) -> ClusterMapping:
    """Map clusters using every pixel of L and no other ground truth.

    The assignment describes the cluster-to-class mapping as majority voting over
    labelled pixels in L. This streaming implementation honors that literally while
    avoiding a 2.6-million-row in-memory feature cache. It is deliberately separate
    from training samples, so an unsupervised fit cannot accidentally receive masks.
    """
    clusters = int(getattr(model, "n_clusters"))
    support = np.zeros(clusters, dtype=np.int64)
    positives = np.zeros(clusters, dtype=np.int64)
    all_positive = 0
    all_pixels = 0
    names = dataset.splits().labelled
    for filename in names:
        image = dataset.load_image(filename)
        features = extract_features(image, feature_set)
        assignments = _cluster_assignments(model, features, chunk_size)
        target = dataset.load_mask(filename, "labelled", "cluster_mapping").reshape(-1)
        if assignments.shape != target.shape:
            raise DevelopmentContractError("Cluster assignments and L mask pixels are misaligned")
        support += np.bincount(assignments, minlength=clusters).astype(np.int64)
        positives += np.bincount(assignments, weights=target, minlength=clusters).astype(np.int64)
        all_positive += int(target.sum())
        all_pixels += int(target.size)
    fallback = int(all_positive > (all_pixels - all_positive))
    mapping = np.full(clusters, fallback, dtype=np.uint8)
    observed = support > 0
    mapping[observed] = (positives[observed] > (support[observed] - positives[observed])).astype(np.uint8)
    if events:
        events.record(
            "development", "map_clusters_using_all_labelled_pixels", feature_set=feature_set,
            image_ids=list(names), mask_ids_read=list(names), mask_purpose="cluster_mapping",
            pixels=int(all_pixels), model_clusters=clusters,
        )
    return ClusterMapping(mapping, support, fallback)


def _validation_kmeans_metrics(
    model: Any,
    mapping: ClusterMapping,
    features: np.ndarray,
    targets: np.ndarray,
    chunk_size: int,
) -> dict[str, float | int]:
    assignments = _cluster_assignments(model, features, chunk_size)
    return evaluate_binary_segmentation(targets, mapping.apply(assignments))


def train_unsupervised_track(
    pools: DevelopmentPools,
    dataset: FloodDataset,
    config: dict[str, Any],
    events: EventLogger | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train the complete K-Means teacher/student track without U masks.

    KT is selected by maximum silhouette score on a fixed training-pixel sample; the
    associated validation IoU is logged as confirmation, not used to override the
    unsupervised choice. All cluster-to-class mappings consume every L pixel and no
    other labels.
    """
    seed = int(config["seed"])
    chunk_size = int(config["execution"]["prediction_chunk_pixels"])
    teacher_fit_count = int(config["sampling"]["kmeans_teacher_fit_pixels"])
    student_fit_count = int(config["sampling"]["kmeans_student_fit_pixels"])
    rich_fit, compact_fit, teacher_sample_manifest = _select_unsupervised_fit_rows(
        pools, teacher_fit_count, seed, "kmeans-teacher-fit"
    )
    teacher_trials: list[dict[str, Any]] = []
    chosen_teacher: Any | None = None
    chosen_mapping: ClusterMapping | None = None
    chosen_trial: dict[str, Any] | None = None
    for clusters in config["unsupervised"]["teacher_clusters"]:
        model = fit_teacher_kmeans(
            rich_fit, n_clusters=int(clusters), n_init=int(config["unsupervised"]["teacher_n_init"]),
            random_state=seed, minibatch=True, batch_size=int(config["unsupervised"]["batch_size"]),
        )
        silhouette_selection = select_pixel_indices(
            image_id=f"teacher-kmeans-{clusters}", population_size=rich_fit.shape[0],
            count=min(2_000, rich_fit.shape[0]), base_seed=seed, strategy="uniform",
            namespace="kmeans-silhouette",
        )
        sample_indices = silhouette_selection.pixel_indices
        sample_labels = _cluster_assignments(model, rich_fit[sample_indices], chunk_size)
        if np.unique(sample_labels).size < 2:
            raise DevelopmentContractError(
                f"K-Means with K_T={clusters} produced fewer than two clusters on its silhouette sample"
            )
        silhouette = float(silhouette_score(rich_fit[sample_indices], sample_labels))
        mapping = mapping_from_all_labelled_images(
            model, dataset, "teacher", chunk_size=chunk_size, events=events
        )
        validation = _validation_kmeans_metrics(
            model, mapping, pools.validation_teacher, pools.validation_targets, chunk_size
        )
        capacity = model.capacity_summary() if hasattr(model, "capacity_summary") else None
        # Use the centralized checker even for a raw sklearn K-Means object.
        from .compliance import assert_model_capacity
        capacity = assert_model_capacity(model, "teacher", feature_count=20, recorded_fit_pixels=teacher_fit_count)
        trial = {
            "clusters": int(clusters), "silhouette": silhouette, "inertia": float(model.inertia_),
            "validation": validation, "mapping": {
                "cluster_to_class": mapping.cluster_to_class.tolist(),
                "support": mapping.support.tolist(), "fallback_class": mapping.fallback_class,
            }, "capacity": capacity, "silhouette_sample": silhouette_selection.as_manifest(False),
        }
        teacher_trials.append(trial)
        if chosen_trial is None or (silhouette, -int(clusters)) > (
            float(chosen_trial["silhouette"]), -int(chosen_trial["clusters"])
        ):
            chosen_teacher, chosen_mapping, chosen_trial = model, mapping, trial
    if chosen_teacher is None or chosen_mapping is None or chosen_trial is None:  # pragma: no cover
        raise DevelopmentContractError("No K-Means teacher candidate was fitted")
    if events:
        events.record(
            "development", "fit_kmeans_teacher", model_fit_ids="labelled_union_unlabelled_images_only",
            mask_ids_read="labelled_only_for_mapping", fit_pixels=teacher_fit_count,
            selected_clusters=chosen_teacher.n_clusters, selection_metric="silhouette",
        )

    # No-teacher compact K-Means uses the exact same compact sample capacity as all
    # distilled students but starts independently with k-means++.
    _, baseline_compact, baseline_manifest = _select_unsupervised_fit_rows(
        pools, student_fit_count, seed, "kmeans-baseline-student-fit"
    )
    baseline = fit_student_kmeans(
        baseline_compact, n_clusters=int(config["unsupervised"]["student_clusters"]),
        init="k-means++", n_init=int(config["unsupervised"]["student_n_init"]), random_state=seed,
    )
    baseline_mapping = mapping_from_all_labelled_images(
        baseline, dataset, "student", chunk_size=chunk_size, events=events
    )
    baseline_validation = _validation_kmeans_metrics(
        baseline, baseline_mapping, pools.validation_student, pools.validation_targets, chunk_size
    )
    from .compliance import assert_model_capacity
    baseline_capacity = assert_model_capacity(
        baseline, "student", feature_count=6, recorded_fit_pixels=student_fit_count
    )

    # Teacher assignments are formed only on the already image-only L union U fit
    # sample. The compact initial centroids hence never receive U ground truth.
    teacher_assignment_all = _cluster_assignments(chosen_teacher, rich_fit, chunk_size)
    student_selector = select_pixel_indices(
        image_id="pooled::kmeans-distilled-student", population_size=rich_fit.shape[0],
        count=student_fit_count, base_seed=seed, strategy="uniform", namespace="kmeans-student-fit",
    )
    student_indices = student_selector.pixel_indices
    compact_student_fit = compact_fit[student_indices]
    teacher_assignment_fit = teacher_assignment_all[student_indices]
    grouping = derive_teacher_grouping(
        chosen_teacher.cluster_centers_, chosen_mapping,
        student_clusters=int(config["unsupervised"]["student_clusters"]), random_state=seed,
    )
    variants: dict[str, dict[str, Any]] = {}
    for refine in (False, True):
        student, initial_centers = make_distilled_student(
            compact_student_fit, teacher_assignment_fit, grouping, refine=refine, random_state=seed
        )
        inherited = grouping.inherited_mapping()
        reestimated = mapping_from_all_labelled_images(
            student, dataset, "student", chunk_size=chunk_size, events=events
        )
        for mapping_name, mapping in (("inherited", inherited), ("reestimated_L", reestimated)):
            key = f"{'refined' if refine else 'frozen'}__{mapping_name}"
            validation = _validation_kmeans_metrics(
                student, mapping, pools.validation_student, pools.validation_targets, chunk_size
            )
            capacity = assert_pair_capacity(
                chosen_teacher, student, teacher_feature_count=20, student_feature_count=6,
                teacher_fit_pixels=teacher_fit_count, student_fit_pixels=student_fit_count,
                teacher_postprocessing=config["postprocessing"], student_postprocessing=config["postprocessing"],
                require_rich_teacher=True,
            )
            variants[key] = {
                "model": student, "mapping": mapping, "refine": refine,
                "mapping_name": mapping_name, "initial_centers": initial_centers,
                "validation": validation, "capacity": capacity,
            }
    selected_variant_key = max(
        variants,
        key=lambda key: (float(variants[key]["validation"]["iou"]), float(variants[key]["validation"]["f1"]), key),
    )
    artifacts = {
        "teacher": chosen_teacher, "teacher_mapping": chosen_mapping,
        "baseline": baseline, "baseline_mapping": baseline_mapping,
        "variants": variants, "selected_variant_key": selected_variant_key,
    }
    records = {
        "teacher_fit_sample": teacher_sample_manifest, "teacher_trials": teacher_trials,
        "teacher_selected": chosen_trial, "baseline_fit_sample": baseline_manifest,
        "baseline": {"validation": baseline_validation, "capacity": baseline_capacity},
        "student_fit_sample": student_selector.as_manifest(),
        "grouping": {
            "teacher_cluster_to_student": grouping.teacher_cluster_to_student.tolist(),
            "student_cluster_to_class": grouping.student_cluster_to_class.tolist(),
        },
        "variants": {
            key: {
                "refine": value["refine"], "mapping_name": value["mapping_name"],
                "validation": value["validation"], "capacity": value["capacity"],
                "initial_centers": value["initial_centers"].tolist(),
                "mapping": value["mapping"].cluster_to_class.tolist(),
            }
            for key, value in variants.items()
        },
        "selected_variant_key": selected_variant_key,
    }
    if events:
        events.record(
            "development", "fit_kmeans_students", model_fit_ids="labelled_union_unlabelled_images_only",
            mask_ids_read="labelled_only_for_mapping", fit_pixels=student_fit_count,
            selected_variant=selected_variant_key,
        )
    return artifacts, records
