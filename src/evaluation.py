"""Full-image inference, timing, micro-evaluation, and qualitative rendering.

This module deliberately works with arrays supplied by the experiment runner.  It does
not know about filenames, partitions, or masks on disk, so it cannot bypass the
leak-aware data-access layer.  The runner decides whether an array comes from
validation or the one permitted final-test pass; this module only makes prediction and
measurement reproducible once those arrays are in memory.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from .features import FeatureSetName, extract_with_timing
from .metrics import evaluate_binary_segmentation
from .modeling import FittedPixelModel
from .unsupervised import ClusterMapping, FrozenCentroidPredictor


class EvaluationError(ValueError):
    """Raised when an in-memory prediction/evaluation request is inconsistent."""


ClusterModel = KMeans | MiniBatchKMeans | FrozenCentroidPredictor
PixelModel = FittedPixelModel | ClusterModel


@dataclass(frozen=True)
class SegmentedImage:
    """Prediction and timing evidence for one complete image.

    ``segmentation_seconds`` begins immediately before feature extraction, so it
    matches the project definition of segmentation time.  ``raw_assignments`` is kept
    for the K-Means track's ARI/NMI calculation and is intentionally absent for a
    supervised classifier.
    """

    prediction: np.ndarray
    feature_set: FeatureSetName
    feature_dimensions: int
    feature_seconds: float
    segmentation_seconds: float
    scores: np.ndarray | None = None
    raw_assignments: np.ndarray | None = None

    def __post_init__(self) -> None:
        prediction = _binary_mask(self.prediction, name="prediction")
        if self.scores is not None:
            scores = np.asarray(self.scores, dtype=np.float32)
            if scores.shape != prediction.shape or not np.isfinite(scores).all():
                raise EvaluationError("scores must be finite and have the prediction's image shape")
            object.__setattr__(self, "scores", scores.copy())
        if self.raw_assignments is not None:
            assignments = np.asarray(self.raw_assignments)
            if assignments.shape != prediction.shape:
                raise EvaluationError(
                    "raw_assignments must have the prediction's image shape"
                )
            if not np.issubdtype(assignments.dtype, np.integer) or np.any(assignments < 0):
                raise EvaluationError("raw_assignments must be non-negative integer cluster IDs")
            object.__setattr__(self, "raw_assignments", assignments.astype(np.int64, copy=True))
        if self.feature_dimensions < 1:
            raise EvaluationError("feature_dimensions must be positive")
        if self.feature_seconds < 0 or self.segmentation_seconds < 0:
            raise EvaluationError("Timing values cannot be negative")
        object.__setattr__(self, "prediction", prediction)

    @property
    def pixels(self) -> int:
        """Return the number of evaluated pixels in this image."""
        return int(self.prediction.size)


@dataclass(frozen=True)
class AggregateEvaluation:
    """Micro-aggregated metrics and timing over a fixed collection of full images."""

    metrics: dict[str, float | int]
    image_count: int
    pixels_evaluated: int
    mean_seconds_per_image: float
    mean_feature_seconds_per_image: float
    raw_model_size_bytes: int
    segments: tuple[SegmentedImage, ...]

    def __post_init__(self) -> None:
        if self.image_count < 1 or self.pixels_evaluated < 1:
            raise EvaluationError("Aggregate evaluation must contain at least one image and one pixel")
        if len(self.segments) != self.image_count:
            raise EvaluationError("image_count and segments length disagree")
        if self.pixels_evaluated != sum(segment.pixels for segment in self.segments):
            raise EvaluationError("pixels_evaluated does not match the supplied complete-image segments")
        if int(self.metrics.get("n_pixels", -1)) != self.pixels_evaluated:
            raise EvaluationError("Micro-metric pixel count does not match segment pixel totals")
        if self.mean_seconds_per_image < 0 or self.mean_feature_seconds_per_image < 0:
            raise EvaluationError("Aggregate timing values cannot be negative")
        if self.raw_model_size_bytes < 1:
            raise EvaluationError("A serializable raw model must occupy at least one byte")

    def as_record(self) -> dict[str, float | int]:
        """Return JSON-friendly metric/efficiency fields without duplicating masks."""
        return {
            **self.metrics,
            "images": self.image_count,
            "mean_seconds_per_image": self.mean_seconds_per_image,
            "mean_feature_seconds_per_image": self.mean_feature_seconds_per_image,
            "raw_model_size_bytes": self.raw_model_size_bytes,
        }


def _binary_mask(values: Any, *, name: str) -> np.ndarray:
    """Validate a 2-D binary mask and make a compact immutable-safe copy."""
    array = np.asarray(values)
    if array.ndim != 2 or array.size == 0:
        raise EvaluationError(f"{name} must be a non-empty two-dimensional mask")
    if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
        raise EvaluationError(f"{name} must be a numeric or boolean binary mask")
    if not np.all(np.isin(np.unique(array), (0, 1, False, True))):
        raise EvaluationError(f"{name} must contain only binary labels 0/1")
    return array.astype(np.uint8, copy=True)


def _positive_scores(model: FittedPixelModel, features: np.ndarray, chunk_size: int) -> np.ndarray:
    """Delegate supervised score inference while preserving the wrapper's scaling rules."""
    return model.predict_scores(features, chunk_size=chunk_size)


def _chunked_cluster_assignments(
    model: ClusterModel, features: np.ndarray, *, chunk_size: int
) -> np.ndarray:
    """Predict K-Means cluster IDs in bounded chunks for a whole 256 x 256 image."""
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise EvaluationError("chunk_size must be a positive integer")
    if isinstance(model, FrozenCentroidPredictor):
        return model.predict(features, chunk_size=int(chunk_size)).astype(np.int64, copy=False)
    assignments: list[np.ndarray] = []
    for start in range(0, len(features), int(chunk_size)):
        assignments.append(np.asarray(model.predict(features[start : start + int(chunk_size)])))
    result = np.concatenate(assignments).astype(np.int64, copy=False)
    if result.shape != (len(features),) or np.any(result < 0):
        raise EvaluationError("K-Means prediction did not return one non-negative assignment per pixel")
    return result


def raw_model_size_bytes(model: PixelModel) -> int:
    """Return the assignment-specified pickle size of the raw deployed estimator.

    A :class:`FittedPixelModel` intentionally exposes its sklearn estimator separately
    from its scaler; the brief defines model size as ``len(pickle.dumps(model))`` for
    the raw model, so the scaler is not silently substituted into that measurement.
    """
    raw_model = model.raw_model if isinstance(model, FittedPixelModel) else model
    return len(pickle.dumps(raw_model))


def segment_image(
    image: np.ndarray,
    model: PixelModel,
    feature_set: FeatureSetName,
    *,
    mapping: ClusterMapping | None = None,
    chunk_size: int = 16_384,
) -> SegmentedImage:
    """Segment every pixel of one RGB image and measure total inference time.

    Parameters
    ----------
    image:
        Preprocessed RGB float image. Its shape is preserved in the returned mask.
    model:
        Either a supervised :class:`FittedPixelModel`, or a K-Means-style predictor.
    feature_set:
        The direct feature family used by this model (``"teacher"`` or ``"student"``).
    mapping:
        Required for K-Means: a precomputed L-only cluster-to-class mapping. It is
        never inferred here, keeping validation/test labels out of this function.
    chunk_size:
        Maximum rows per prediction call; feature extraction still happens once for the
        entire image, as required for the timing measurement.
    """
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise EvaluationError("chunk_size must be a positive integer")
    start = perf_counter()
    extracted = extract_with_timing(image, feature_set)
    features = extracted.values
    height, width = extracted.height, extracted.width

    if isinstance(model, FittedPixelModel):
        if mapping is not None:
            raise EvaluationError("Supervised pixel models must not receive a cluster mapping")
        if model.feature_set != feature_set:
            raise EvaluationError(
                f"Model expects {model.feature_set!r} features but got {feature_set!r}"
            )
        scores = _positive_scores(model, features, int(chunk_size))
        prediction = (scores >= model.threshold).astype(np.uint8).reshape(height, width)
        return SegmentedImage(
            prediction=prediction,
            scores=scores.reshape(height, width),
            raw_assignments=None,
            feature_set=feature_set,
            feature_dimensions=extracted.dimensions,
            feature_seconds=extracted.elapsed_seconds,
            segmentation_seconds=perf_counter() - start,
        )

    if not isinstance(model, (KMeans, MiniBatchKMeans, FrozenCentroidPredictor)):
        raise EvaluationError(
            "model must be a FittedPixelModel, KMeans, MiniBatchKMeans, or FrozenCentroidPredictor"
        )
    if mapping is None:
        raise EvaluationError("A precomputed labelled-data ClusterMapping is required for K-Means")
    expected_features = int(getattr(model, "n_features_in_"))
    if expected_features != extracted.dimensions:
        raise EvaluationError(
            f"K-Means expects {expected_features} features but {feature_set!r} produces "
            f"{extracted.dimensions}"
        )
    assignments = _chunked_cluster_assignments(model, features, chunk_size=int(chunk_size))
    prediction = mapping.apply(assignments).reshape(height, width)
    return SegmentedImage(
        prediction=prediction,
        scores=None,
        raw_assignments=assignments.reshape(height, width),
        feature_set=feature_set,
        feature_dimensions=extracted.dimensions,
        feature_seconds=extracted.elapsed_seconds,
        segmentation_seconds=perf_counter() - start,
    )


def evaluate_model_on_images(
    images: Sequence[np.ndarray] | Iterable[np.ndarray],
    masks: Sequence[np.ndarray] | Iterable[np.ndarray],
    model: PixelModel,
    feature_set: FeatureSetName,
    *,
    mapping: ClusterMapping | None = None,
    chunk_size: int = 16_384,
) -> AggregateEvaluation:
    """Segment all supplied images and calculate one micro-averaged metric record.

    The function does not subsample or average per-image scores.  Its caller therefore
    obtains the brief's required full-pixel micro average whenever it passes every
    256 x 256 image from a split.
    """
    image_list, mask_list = list(images), list(masks)
    if not image_list:
        raise EvaluationError("At least one image is required for evaluation")
    if len(image_list) != len(mask_list):
        raise EvaluationError("images and masks must contain the same number of elements")
    segments: list[SegmentedImage] = []
    flattened_masks: list[np.ndarray] = []
    flattened_predictions: list[np.ndarray] = []
    for image, mask in zip(image_list, mask_list):
        truth = _binary_mask(mask, name="mask")
        segment = segment_image(
            image,
            model,
            feature_set,
            mapping=mapping,
            chunk_size=chunk_size,
        )
        if truth.shape != segment.prediction.shape:
            raise EvaluationError(
                f"Mask shape {truth.shape} does not match predicted shape {segment.prediction.shape}"
            )
        segments.append(segment)
        flattened_masks.append(truth.reshape(-1))
        flattened_predictions.append(segment.prediction.reshape(-1))
    truth_all = np.concatenate(flattened_masks)
    prediction_all = np.concatenate(flattened_predictions)
    metrics = evaluate_binary_segmentation(truth_all, prediction_all)
    return AggregateEvaluation(
        metrics=metrics,
        image_count=len(segments),
        pixels_evaluated=int(truth_all.size),
        mean_seconds_per_image=float(np.mean([segment.segmentation_seconds for segment in segments])),
        mean_feature_seconds_per_image=float(np.mean([segment.feature_seconds for segment in segments])),
        raw_model_size_bytes=raw_model_size_bytes(model),
        segments=tuple(segments),
    )


def write_qualitative_panel(
    image: np.ndarray,
    mask: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    destination: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Render one deterministic image/ground-truth/prediction comparison panel.

    Prediction labels are sorted, so the same dictionary contents always appear in the
    same left-to-right order.  This is intentionally a display function only: callers
    must supply masks after the evaluation protocol authorizes their use.
    """
    rgb = np.asarray(image, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or not np.isfinite(rgb).all():
        raise EvaluationError("image must be a finite H x W x 3 RGB array")
    if np.any(rgb < 0.0) or np.any(rgb > 1.0):
        raise EvaluationError("image must be normalized to [0, 1]")
    truth = _binary_mask(mask, name="mask")
    if truth.shape != rgb.shape[:2]:
        raise EvaluationError("mask shape must match the image's height and width")
    if not predictions:
        raise EvaluationError("At least one named prediction is required for a qualitative panel")
    ordered_predictions = [(str(name), _binary_mask(values, name=f"prediction {name!r}"))
                           for name, values in sorted(predictions.items(), key=lambda item: str(item[0]))]
    for name, prediction in ordered_predictions:
        if prediction.shape != truth.shape:
            raise EvaluationError(f"prediction {name!r} has shape {prediction.shape}, expected {truth.shape}")

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    columns = 2 + len(ordered_predictions)
    figure, axes = plt.subplots(1, columns, figsize=(3.1 * columns, 3.65), constrained_layout=True)
    axes = np.atleast_1d(axes)
    axes[0].imshow(rgb)
    axes[0].set_title("RGB image")
    axes[1].imshow(truth, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title("Ground truth")
    for axis, (name, prediction) in zip(axes[2:], ordered_predictions):
        axis.imshow(prediction, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(name)
    for axis in axes:
        axis.axis("off")
    if title:
        figure.suptitle(title)
    figure.savefig(destination_path, dpi=160, metadata={"Software": "CS3103 evaluation pipeline"})
    plt.close(figure)
    return destination_path
