"""Array-only hard and soft teacher-student distillation primitives.

Nothing in this module opens a dataset file or accepts a mask object.  A teacher's
probabilities and direct image-derived student features are its only inputs during
pseudo-label construction.  The one validation helper accepts an explicitly named
``validation_targets`` array because threshold selection is intentionally a separate,
validation-only operation performed by the experiment runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

import numpy as np

from .metrics import evaluate_binary_segmentation
from .sampling import PixelSelection, SamplingStrategy, select_pixel_indices


class DistillationError(ValueError):
    """Raised when teacher outputs or a transfer set violate the distillation contract."""


def _as_feature_matrix(values: Any, *, name: str = "features") -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise DistillationError(f"{name} must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise DistillationError(f"{name} contains NaN or infinity")
    return matrix


def _as_probability_vector(values: Any, *, name: str = "teacher_scores") -> np.ndarray:
    scores = np.asarray(values, dtype=np.float32).reshape(-1)
    if scores.size < 1:
        raise DistillationError(f"{name} must contain at least one probability")
    if not np.isfinite(scores).all() or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise DistillationError(f"{name} must contain finite probabilities in [0, 1]")
    return scores


def _as_binary_validation_targets(values: Any) -> np.ndarray:
    targets = np.asarray(values).reshape(-1)
    if targets.size < 1 or not np.all(np.isin(targets, (0, 1, False, True))):
        raise DistillationError("validation_targets must be a non-empty binary 0/1 vector")
    return targets.astype(np.uint8, copy=False)


def _validate_tau(tau: float) -> float:
    value = float(tau)
    if not np.isfinite(value) or value < 0.5 or value > 1.0:
        raise DistillationError("tau must be a finite confidence threshold in [0.5, 1.0]")
    return value


@dataclass(frozen=True)
class PseudoLabels:
    """Hard labels and confidence derived solely from a teacher score vector."""

    scores: np.ndarray
    hard_labels: np.ndarray
    confidence: np.ndarray

    def __post_init__(self) -> None:
        scores = _as_probability_vector(self.scores)
        hard = np.asarray(self.hard_labels)
        confidence = np.asarray(self.confidence, dtype=np.float32).reshape(-1)
        if hard.ndim != 1 or hard.size != scores.size or not np.all(np.isin(hard, (0, 1, False, True))):
            raise DistillationError("hard_labels must be a binary vector aligned with scores")
        if confidence.size != scores.size or not np.isfinite(confidence).all():
            raise DistillationError("confidence must be a finite vector aligned with scores")
        expected_hard = (scores >= 0.5).astype(np.uint8)
        expected_confidence = np.maximum(scores, 1.0 - scores)
        if not np.array_equal(np.asarray(hard, dtype=np.uint8), expected_hard):
            raise DistillationError("hard_labels must use the exact rule score >= 0.5")
        if not np.allclose(confidence, expected_confidence, rtol=0.0, atol=1e-7):
            raise DistillationError("confidence must equal max(score, 1 - score)")
        object.__setattr__(self, "scores", scores.copy())
        object.__setattr__(self, "hard_labels", expected_hard)
        object.__setattr__(self, "confidence", expected_confidence.astype(np.float32, copy=False))

    @property
    def count(self) -> int:
        return int(self.scores.size)

    def retained_indices(self, tau: float) -> np.ndarray:
        """Return transfer rows satisfying the inclusive project confidence rule."""
        threshold = _validate_tau(tau)
        return np.flatnonzero(self.confidence >= threshold).astype(np.int64, copy=False)

    def retained_fraction(self, tau: float) -> float:
        """Return the fraction of teacher-annotated pixels retained at ``tau``."""
        return float(self.retained_indices(tau).size / self.count)


def hard_pseudo_labels(teacher_scores: Any) -> PseudoLabels:
    """Derive pseudo-labels with the exact rules in the assignment brief.

    ``score == 0.5`` is a flood hard label and has confidence 0.5.  Confidence
    filtering subsequently uses ``>= tau`` rather than an accidental strict inequality.
    """
    scores = _as_probability_vector(teacher_scores)
    return PseudoLabels(
        scores=scores,
        hard_labels=(scores >= 0.5).astype(np.uint8),
        confidence=np.maximum(scores, 1.0 - scores).astype(np.float32),
    )


@dataclass(frozen=True)
class PseudoTrainingSet:
    """A fixed-capacity, reproducibly sampled hard-distillation transfer set.

    ``source_indices`` refer to rows in the caller's original feature/probability pool.
    They are not image labels, and can be written to a manifest to reconstruct exactly
    which teacher outputs reached the student.
    """

    features: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    source_indices: np.ndarray
    candidate_source_indices: np.ndarray
    transfer_pool_count: int
    tau: float
    selection: PixelSelection

    def __post_init__(self) -> None:
        features = _as_feature_matrix(self.features)
        scores = _as_probability_vector(self.scores)
        labels = np.asarray(self.labels).reshape(-1)
        source = np.asarray(self.source_indices)
        candidates = np.asarray(self.candidate_source_indices)
        transfer_pool_count = self.transfer_pool_count
        if scores.size != features.shape[0] or labels.size != features.shape[0]:
            raise DistillationError("features, scores and labels must have aligned rows")
        if not np.all(np.isin(labels, (0, 1, False, True))):
            raise DistillationError("pseudo-labels must be binary")
        if source.ndim != 1 or source.size != features.shape[0] or not np.issubdtype(source.dtype, np.integer):
            raise DistillationError("source_indices must be one integer index per selected row")
        if candidates.ndim != 1 or not np.issubdtype(candidates.dtype, np.integer):
            raise DistillationError("candidate_source_indices must be an integer vector")
        if not isinstance(transfer_pool_count, (int, np.integer)) or transfer_pool_count < 1:
            raise DistillationError("transfer_pool_count must be a positive integer")
        if np.any(candidates < 0) or np.any(candidates >= int(transfer_pool_count)):
            raise DistillationError("candidate source indices lie outside the original transfer pool")
        if source.size != self.selection.requested_count:
            raise DistillationError("selection size must agree with selected transfer rows")
        if len(np.unique(source)) != source.size or len(np.unique(candidates)) != candidates.size:
            raise DistillationError("source indices must not contain duplicates")
        if not np.all(np.isin(source, candidates)):
            raise DistillationError("every selected source index must have passed confidence filtering")
        derived = hard_pseudo_labels(scores).hard_labels
        if not np.array_equal(labels.astype(np.uint8), derived):
            raise DistillationError("selected labels must be derived from selected teacher scores")
        if np.unique(labels).size < 2:
            raise DistillationError("pseudo-training set must contain both pseudo-classes")
        _validate_tau(self.tau)
        object.__setattr__(self, "features", features.copy())
        object.__setattr__(self, "scores", scores.copy())
        object.__setattr__(self, "labels", labels.astype(np.uint8, copy=True))
        object.__setattr__(self, "source_indices", source.astype(np.int64, copy=True))
        object.__setattr__(self, "candidate_source_indices", candidates.astype(np.int64, copy=True))
        object.__setattr__(self, "transfer_pool_count", int(transfer_pool_count))

    @property
    def retained_count(self) -> int:
        return int(self.candidate_source_indices.size)

    @property
    def selected_count(self) -> int:
        return int(self.source_indices.size)

    @property
    def retained_fraction(self) -> float:
        """Fraction kept by confidence filtering from the original transfer pool."""
        return float(self.retained_count / self.transfer_pool_count)

    def as_manifest(self, *, include_indices: bool = False) -> dict[str, object]:
        """Return JSON-ready pseudo-label and sampling evidence."""
        result: dict[str, object] = {
            "tau": self.tau,
            "transfer_pool_rows": self.transfer_pool_count,
            "confidence_retained_rows": self.retained_count,
            "confidence_retained_fraction": self.retained_fraction,
            "student_fit_rows": self.selected_count,
            "pseudo_class_counts": {
                "non_flood": int(np.count_nonzero(self.labels == 0)),
                "flood": int(np.count_nonzero(self.labels == 1)),
            },
            "selection": self.selection.as_manifest(include_indices=False),
        }
        if include_indices:
            result["source_indices"] = self.source_indices.tolist()
            result["candidate_source_indices"] = self.candidate_source_indices.tolist()
        return result


def select_pseudo_training_set(
    student_features: Any,
    teacher_scores: Any,
    *,
    tau: float,
    capacity: int,
    base_seed: int = 42,
    strategy: SamplingStrategy = "balanced",
    namespace: str = "distillation-transfer",
) -> PseudoTrainingSet:
    """Confidence-filter then sample an exact, fixed-capacity transfer set.

    This function only uses teacher probabilities and compact image features.  It never
    receives ground truth.  A tau candidate is rejected if it cannot supply the fixed
    declared capacity or both pseudo-classes; silently shrinking the student would make
    the no-teacher baseline incomparable.
    """
    features = _as_feature_matrix(student_features, name="student_features")
    pseudo = hard_pseudo_labels(teacher_scores)
    if features.shape[0] != pseudo.count:
        raise DistillationError("student_features and teacher_scores must have equal row counts")
    if not isinstance(capacity, (int, np.integer)) or capacity < 2:
        raise DistillationError("capacity must be an integer of at least two pixels")
    threshold = _validate_tau(tau)
    candidates = pseudo.retained_indices(threshold)
    if candidates.size < int(capacity):
        raise DistillationError(
            f"tau={threshold:g} retained {candidates.size} transfer pixels, below fixed capacity {capacity}"
        )
    candidate_labels = pseudo.hard_labels[candidates]
    if np.unique(candidate_labels).size < 2:
        raise DistillationError(
            f"tau={threshold:g} retained only one pseudo-class; a binary student cannot be fitted"
        )
    selection = select_pixel_indices(
        image_id="transfer_pool",
        population_size=int(candidates.size),
        count=int(capacity),
        base_seed=int(base_seed),
        strategy=strategy,
        labels=candidate_labels if strategy == "balanced" else None,
        namespace=f"{namespace}/tau={threshold:.6f}",
    )
    selected_source_indices = candidates[selection.pixel_indices]
    return PseudoTrainingSet(
        features=features[selected_source_indices],
        scores=pseudo.scores[selected_source_indices],
        labels=pseudo.hard_labels[selected_source_indices],
        source_indices=selected_source_indices,
        candidate_source_indices=candidates,
        transfer_pool_count=int(features.shape[0]),
        tau=threshold,
        selection=selection,
    )


@dataclass(frozen=True)
class SoftDistillationBatch:
    """Weighted duplicated examples implementing soft binary distillation.

    For each original pixel `x` with teacher probability `q`, the output includes
    `(x, 0)` weighted `1-q` and `(x, 1)` weighted `q`.  RF and GNB can consume this
    directly via ``sample_weight``; KNN should use the regression route instead.
    """

    features: np.ndarray
    targets: np.ndarray
    sample_weight: np.ndarray
    source_rows: int

    def __post_init__(self) -> None:
        features = _as_feature_matrix(self.features)
        targets = np.asarray(self.targets).reshape(-1)
        weights = np.asarray(self.sample_weight, dtype=np.float32).reshape(-1)
        if features.shape[0] != 2 * self.source_rows:
            raise DistillationError("soft batch must contain exactly two rows per source pixel")
        if targets.size != features.shape[0] or not np.all(np.isin(targets, (0, 1, False, True))):
            raise DistillationError("soft targets must be binary and aligned with features")
        if weights.size != features.shape[0] or not np.isfinite(weights).all() or np.any(weights < 0):
            raise DistillationError("soft sample weights must be finite non-negative values")
        if not np.allclose(weights[: self.source_rows] + weights[self.source_rows :], 1.0, atol=1e-6):
            raise DistillationError("paired soft weights must sum to one per source pixel")
        if not np.array_equal(targets[: self.source_rows], np.zeros(self.source_rows, dtype=targets.dtype)):
            raise DistillationError("first half of a soft batch must be non-flood targets")
        if not np.array_equal(targets[self.source_rows :], np.ones(self.source_rows, dtype=targets.dtype)):
            raise DistillationError("second half of a soft batch must be flood targets")
        object.__setattr__(self, "features", features.copy())
        object.__setattr__(self, "targets", targets.astype(np.uint8, copy=True))
        object.__setattr__(self, "sample_weight", weights.astype(np.float32, copy=True))


def weighted_soft_duplication(features: Any, teacher_scores: Any) -> SoftDistillationBatch:
    """Create the exact weighted-duplication training batch from soft scores."""
    values = _as_feature_matrix(features)
    scores = _as_probability_vector(teacher_scores)
    if values.shape[0] != scores.size:
        raise DistillationError("features and teacher_scores must have equal row counts")
    count = scores.size
    return SoftDistillationBatch(
        features=np.concatenate((values, values), axis=0),
        targets=np.concatenate((np.zeros(count, dtype=np.uint8), np.ones(count, dtype=np.uint8))),
        sample_weight=np.concatenate((1.0 - scores, scores)).astype(np.float32),
        source_rows=int(count),
    )


@dataclass(frozen=True)
class ThresholdStudy:
    """Validation-only score threshold study with a deterministic selection rule."""

    selected_threshold: float
    selected_metrics: dict[str, float | int]
    rows: tuple[dict[str, float | int], ...]

    def as_manifest(self) -> dict[str, object]:
        return {
            "selection_metric": "validation_iou_micro",
            "tie_break": "higher F1, then threshold closest to 0.5, then lower threshold",
            "selected_threshold": self.selected_threshold,
            "selected_metrics": self.selected_metrics,
            "threshold_rows": list(self.rows),
        }


def tune_decision_threshold(
    scores: Any,
    validation_targets: Any,
    thresholds: Iterable[float],
) -> ThresholdStudy:
    """Choose a binary decision threshold using validation labels only.

    IoU is the primary selection metric because flood area is the assignment's positive
    segmentation target. Ties are deterministically resolved by F1, proximity to 0.5,
    then the lower threshold. The caller must ensure `validation_targets` came from the
    fixed validation partition; this module does not access any files.
    """
    probabilities = _as_probability_vector(scores, name="scores")
    targets = _as_binary_validation_targets(validation_targets)
    if probabilities.size != targets.size:
        raise DistillationError("scores and validation_targets must have equal row counts")
    candidates = sorted({float(value) for value in thresholds})
    if not candidates:
        raise DistillationError("At least one candidate decision threshold is required")
    if any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in candidates):
        raise DistillationError("All decision thresholds must be finite values in [0, 1]")
    rows: list[dict[str, float | int]] = []
    for threshold in candidates:
        metrics = evaluate_binary_segmentation(targets, (probabilities >= threshold).astype(np.uint8))
        rows.append({"threshold": threshold, **metrics})
    # `max` is stable, but make the full deterministic ordering visible rather than
    # relying on candidate list order when scores are tied exactly.
    selected = max(
        rows,
        key=lambda row: (
            float(row["iou"]),
            float(row["f1"]),
            -abs(float(row["threshold"]) - 0.5),
            -float(row["threshold"]),
        ),
    )
    return ThresholdStudy(
        selected_threshold=float(selected["threshold"]),
        selected_metrics={key: value for key, value in selected.items() if key != "threshold"},
        rows=tuple(rows),
    )
