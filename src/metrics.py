"""Metric primitives for leak-free flood segmentation experiments.

The project specification requires micro-averaged pixel metrics.  These helpers accept
one image or an arbitrary collection of images, flatten all pixels together, and compute
the confusion counts once.  This deliberately avoids the common mistake of averaging
per-image IoUs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


class MetricInputError(ValueError):
    """Raised when labels, scores, or scalar metrics are outside their contract."""


@dataclass(frozen=True)
class Confusion:
    """Micro-aggregated binary confusion counts, with flood as positive (one)."""

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def total(self) -> int:
        return self.true_positive + self.false_positive + self.false_negative + self.true_negative

    def as_dict(self) -> dict[str, int]:
        return {
            "tp": self.true_positive,
            "fp": self.false_positive,
            "fn": self.false_negative,
            "tn": self.true_negative,
            "n_pixels": self.total,
        }


def _as_binary_vector(values: Any, *, name: str) -> np.ndarray:
    """Return a flattened uint8 binary vector and reject silent label coercion."""
    array = np.asarray(values)
    if array.size == 0:
        raise MetricInputError(f"{name} must contain at least one pixel")
    if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
        raise MetricInputError(f"{name} must be a numeric or boolean binary array")
    unique = np.unique(array)
    if not np.all(np.isin(unique, (0, 1, False, True))):
        raise MetricInputError(f"{name} must contain only binary labels 0/1; found {unique[:8]!r}")
    return array.astype(np.uint8, copy=False).reshape(-1)


def micro_confusion(y_true: Any, y_pred: Any) -> Confusion:
    """Compute TP/FP/FN/TN over every supplied pixel, not per-image averages."""
    truth = _as_binary_vector(y_true, name="y_true")
    prediction = _as_binary_vector(y_pred, name="y_pred")
    if truth.shape != prediction.shape:
        raise MetricInputError(
            f"y_true and y_pred contain different numbers of pixels: {truth.size} and {prediction.size}"
        )
    return Confusion(
        true_positive=int(np.count_nonzero((truth == 1) & (prediction == 1))),
        false_positive=int(np.count_nonzero((truth == 0) & (prediction == 1))),
        false_negative=int(np.count_nonzero((truth == 1) & (prediction == 0))),
        true_negative=int(np.count_nonzero((truth == 0) & (prediction == 0))),
    )


def _safe_ratio(numerator: int | float, denominator: int | float, *, undefined: float) -> float:
    return float(numerator / denominator) if denominator else float(undefined)


def segmentation_metrics(confusion: Confusion, *, undefined: float = 0.0) -> dict[str, float | int]:
    """Derive project-required precision, recall, F1, and flood IoU from one confusion.

    ``undefined`` documents the chosen zero-denominator policy.  The default (zero) is
    suitable for this flood-positive evaluation and makes reports reproducible.
    """
    precision = _safe_ratio(
        confusion.true_positive,
        confusion.true_positive + confusion.false_positive,
        undefined=undefined,
    )
    recall = _safe_ratio(
        confusion.true_positive,
        confusion.true_positive + confusion.false_negative,
        undefined=undefined,
    )
    f1 = _safe_ratio(2 * precision * recall, precision + recall, undefined=undefined)
    iou = _safe_ratio(
        confusion.true_positive,
        confusion.true_positive + confusion.false_positive + confusion.false_negative,
        undefined=undefined,
    )
    return {
        **confusion.as_dict(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def evaluate_binary_segmentation(y_true: Any, y_pred: Any, *, undefined: float = 0.0) -> dict[str, float | int]:
    """Convenience composition of :func:`micro_confusion` and :func:`segmentation_metrics`."""
    return segmentation_metrics(micro_confusion(y_true, y_pred), undefined=undefined)


def agreement(y_a: Any, y_b: Any) -> float:
    """Return the pixel-wise agreement between two binary prediction maps."""
    first = _as_binary_vector(y_a, name="y_a")
    second = _as_binary_vector(y_b, name="y_b")
    if first.shape != second.shape:
        raise MetricInputError("Agreement requires prediction arrays with the same number of pixels")
    return float(np.mean(first == second))


def _finite_nonnegative(value: float, *, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise MetricInputError(f"{name} must be a finite non-negative value; got {value!r}")
    return value


def retention_percent(student_iou: float, teacher_iou: float) -> float:
    """Return 100 * IoU_student / IoU_teacher, or NaN when teacher IoU is zero."""
    student = _finite_nonnegative(student_iou, name="student_iou")
    teacher = _finite_nonnegative(teacher_iou, name="teacher_iou")
    return float("nan") if teacher == 0 else 100.0 * student / teacher


def compression_ratio(teacher_size_bytes: int | float, student_size_bytes: int | float) -> float:
    """Return size_teacher / size_student and reject zero-sized artifacts."""
    teacher = _finite_nonnegative(teacher_size_bytes, name="teacher_size_bytes")
    student = _finite_nonnegative(student_size_bytes, name="student_size_bytes")
    if student == 0:
        raise MetricInputError("student_size_bytes must be greater than zero")
    return teacher / student


def speedup(teacher_seconds: float, student_seconds: float) -> float:
    """Return t_teacher / t_student and reject invalid timing measurements."""
    teacher = _finite_nonnegative(teacher_seconds, name="teacher_seconds")
    student = _finite_nonnegative(student_seconds, name="student_seconds")
    if student == 0:
        raise MetricInputError("student_seconds must be greater than zero")
    return teacher / student


def percentage_point_gain(distilled_score: float, baseline_score: float) -> float:
    """Return a fraction-scale score difference in percentage points."""
    distilled = _finite_nonnegative(distilled_score, name="distilled_score")
    baseline = _finite_nonnegative(baseline_score, name="baseline_score")
    if distilled > 1 or baseline > 1:
        raise MetricInputError("Scores must be fractions in [0, 1], not percentages")
    return 100.0 * (distilled - baseline)


def efficiency_score(f1: float, iou: float, *, inputs_are_percent: bool = False) -> float:
    """Compute the project's Z = 0.6 F1_% + 0.4 IoU_% efficiency score.

    By default ``f1`` and ``iou`` are fractions in [0, 1], which is the form emitted by
    :func:`evaluate_binary_segmentation`. Set ``inputs_are_percent=True`` when inputs are
    already expressed on the 0--100 scale.
    """
    f1_value = _finite_nonnegative(f1, name="f1")
    iou_value = _finite_nonnegative(iou, name="iou")
    upper = 100.0 if inputs_are_percent else 1.0
    if f1_value > upper or iou_value > upper:
        unit = "percent" if inputs_are_percent else "fraction"
        raise MetricInputError(f"F1 and IoU must be {unit}-scale values <= {upper}")
    if not inputs_are_percent:
        f1_value *= 100.0
        iou_value *= 100.0
    return 0.6 * f1_value + 0.4 * iou_value


def cluster_similarity(teacher_clusters: Iterable[int], student_clusters: Iterable[int]) -> dict[str, float]:
    """Calculate ARI and NMI from raw cluster assignments in the exact same pixel order."""
    teacher = np.asarray(list(teacher_clusters)).reshape(-1)
    student = np.asarray(list(student_clusters)).reshape(-1)
    if teacher.size == 0 or student.size == 0:
        raise MetricInputError("Cluster similarity requires at least one assignment")
    if teacher.shape != student.shape:
        raise MetricInputError("Teacher and student cluster assignments must have equal length")
    return {
        "ari": float(adjusted_rand_score(teacher, student)),
        "nmi": float(normalized_mutual_info_score(teacher, student)),
    }


def metrics_record(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """Produce JSON-friendly primary metric output for a model evaluation artifact."""
    return dict(evaluate_binary_segmentation(y_true, y_pred))
