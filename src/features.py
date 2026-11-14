"""Deterministic, hand-crafted pixel features for flood segmentation.

Feature extractors accept an already-preprocessed RGB image in ``[0, 1]`` and return
one row per pixel in raster order.  They deliberately do not accept masks, filenames or
model objects, so neither labels nor a learned representation can leak into a feature
matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from scipy import ndimage


FeatureSetName = Literal["student", "teacher"]

STUDENT_FEATURE_NAMES: tuple[str, ...] = (
    "red",
    "green",
    "blue",
    "hue",
    "saturation",
    "value",
)

TEACHER_FEATURE_NAMES: tuple[str, ...] = (
    *STUDENT_FEATURE_NAMES,
    "blurred_red_sigma1",
    "blurred_green_sigma1",
    "blurred_blue_sigma1",
    "blurred_red_sigma3",
    "blurred_green_sigma3",
    "blurred_blue_sigma3",
    "gray_mean_7",
    "gray_std_7",
    "gray_mean_15",
    "gray_std_15",
    "sobel_magnitude",
    "laplacian",
    "row_position",
    "column_position",
)


class FeatureContractError(ValueError):
    """Raised when an input or generated matrix violates the feature contract."""


@dataclass(frozen=True)
class FeatureExtraction:
    """One feature matrix and extraction metadata suitable for experiment logs."""

    values: np.ndarray
    names: tuple[str, ...]
    height: int
    width: int
    elapsed_seconds: float

    @property
    def pixels(self) -> int:
        return self.height * self.width

    @property
    def dimensions(self) -> int:
        return self.values.shape[1]


def feature_names(feature_set: FeatureSetName) -> tuple[str, ...]:
    """Return the fixed, reportable schema for a feature family."""
    if feature_set == "student":
        return STUDENT_FEATURE_NAMES
    if feature_set == "teacher":
        return TEACHER_FEATURE_NAMES
    raise FeatureContractError(f"Unknown feature set {feature_set!r}; use 'student' or 'teacher'.")


def _validate_rgb(image: np.ndarray) -> np.ndarray:
    """Validate the public RGB-float input contract and create a stable float32 view."""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise FeatureContractError(
            f"Expected a height x width x 3 RGB array, received shape {array.shape}."
        )
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise FeatureContractError("Images must have positive height and width.")
    if not np.issubdtype(array.dtype, np.number):
        raise FeatureContractError("RGB image values must be numeric.")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise FeatureContractError("RGB image contains NaN or infinity.")
    lower, upper = float(array.min()), float(array.max())
    if lower < 0.0 or upper > 1.0:
        raise FeatureContractError(
            "Expected an RGB float image normalised to [0, 1]; "
            f"received value range [{lower:.6g}, {upper:.6g}]."
        )
    return array


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert normalised RGB to HSV without a dependency on OpenCV or scikit-image.

    The implementation follows the usual sector formula and is vectorised.  Hue for
    achromatic pixels is defined as zero, matching common image-library behaviour.
    """
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maximum = np.maximum(np.maximum(red, green), blue)
    minimum = np.minimum(np.minimum(red, green), blue)
    chroma = maximum - minimum

    hue = np.zeros_like(maximum, dtype=np.float32)
    nonzero = chroma > 0.0
    red_max = nonzero & (maximum == red)
    green_max = nonzero & (maximum == green)
    blue_max = nonzero & (maximum == blue)
    hue[red_max] = np.mod((green[red_max] - blue[red_max]) / chroma[red_max], 6.0)
    hue[green_max] = ((blue[green_max] - red[green_max]) / chroma[green_max]) + 2.0
    hue[blue_max] = ((red[blue_max] - green[blue_max]) / chroma[blue_max]) + 4.0
    hue /= 6.0
    hue = np.mod(hue, 1.0)

    saturation = np.divide(
        chroma,
        maximum,
        out=np.zeros_like(maximum, dtype=np.float32),
        where=maximum > 0.0,
    )
    return np.stack((hue, saturation, maximum), axis=-1).astype(np.float32, copy=False)


def _local_mean_std(gray: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return reflection-padded local mean and standard deviation for an odd window."""
    if window < 1 or window % 2 == 0:
        raise FeatureContractError("Local-statistic windows must be positive odd integers.")
    mean = ndimage.uniform_filter(gray, size=window, mode="reflect")
    mean_square = ndimage.uniform_filter(gray * gray, size=window, mode="reflect")
    variance = np.maximum(mean_square - mean * mean, 0.0)
    return mean.astype(np.float32, copy=False), np.sqrt(variance).astype(np.float32, copy=False)


def _teacher_feature_maps(rgb: np.ndarray, hsv: np.ndarray) -> tuple[np.ndarray, ...]:
    """Build the exact rich feature map ordering declared in the experiment config."""
    gray = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(
        np.float32, copy=False
    )
    blurred_sigma1 = ndimage.gaussian_filter(rgb, sigma=(1.0, 1.0, 0.0), mode="reflect")
    blurred_sigma3 = ndimage.gaussian_filter(rgb, sigma=(3.0, 3.0, 0.0), mode="reflect")
    mean7, std7 = _local_mean_std(gray, 7)
    mean15, std15 = _local_mean_std(gray, 15)
    dy = ndimage.sobel(gray, axis=0, mode="reflect")
    dx = ndimage.sobel(gray, axis=1, mode="reflect")
    sobel_magnitude = np.hypot(dx, dy).astype(np.float32, copy=False)
    laplacian = ndimage.laplace(gray, mode="reflect").astype(np.float32, copy=False)
    height, width = gray.shape
    row_position = np.broadcast_to(np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None], gray.shape)
    column_position = np.broadcast_to(np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :], gray.shape)
    return (
        rgb[..., 0],
        rgb[..., 1],
        rgb[..., 2],
        hsv[..., 0],
        hsv[..., 1],
        hsv[..., 2],
        blurred_sigma1[..., 0],
        blurred_sigma1[..., 1],
        blurred_sigma1[..., 2],
        blurred_sigma3[..., 0],
        blurred_sigma3[..., 1],
        blurred_sigma3[..., 2],
        mean7,
        std7,
        mean15,
        std15,
        sobel_magnitude,
        laplacian,
        row_position,
        column_position,
    )


def extract_features(image: np.ndarray, feature_set: FeatureSetName) -> np.ndarray:
    """Extract direct, hand-crafted per-pixel features in raster order.

    Parameters
    ----------
    image:
        A H x W x 3 RGB float array in [0, 1], normally supplied by
        :meth:`src.data.FloodDataset.load_image`.
    feature_set:
        ``"student"`` returns six RGB/HSV channels. ``"teacher"`` returns the
        student channels plus spatial colour/context/edge features (20 total).
    """
    rgb = _validate_rgb(image)
    hsv = rgb_to_hsv(rgb)
    if feature_set == "student":
        maps = (
            rgb[..., 0],
            rgb[..., 1],
            rgb[..., 2],
            hsv[..., 0],
            hsv[..., 1],
            hsv[..., 2],
        )
    elif feature_set == "teacher":
        maps = _teacher_feature_maps(rgb, hsv)
    else:
        raise FeatureContractError(f"Unknown feature set {feature_set!r}; use 'student' or 'teacher'.")

    expected_names = feature_names(feature_set)
    features = np.stack(maps, axis=-1).reshape(-1, len(expected_names)).astype(np.float32, copy=False)
    if features.shape[1] != len(expected_names) or not np.isfinite(features).all():
        raise FeatureContractError("Feature extraction produced an invalid matrix.")
    return features


def extract_with_timing(image: np.ndarray, feature_set: FeatureSetName) -> FeatureExtraction:
    """Extract features and record wall-clock time, including only feature computation."""
    start = perf_counter()
    values = extract_features(image, feature_set)
    elapsed = perf_counter() - start
    height, width = image.shape[:2]
    return FeatureExtraction(
        values=values,
        names=feature_names(feature_set),
        height=height,
        width=width,
        elapsed_seconds=elapsed,
    )


def validate_feature_budget(student: np.ndarray, teacher: np.ndarray) -> None:
    """Assert the report's feature-capacity constraints using actual matrices."""
    if student.ndim != 2 or teacher.ndim != 2:
        raise FeatureContractError("Feature matrices must be two-dimensional.")
    student_dimensions, teacher_dimensions = student.shape[1], teacher.shape[1]
    if student_dimensions > 6:
        raise FeatureContractError(f"Student has {student_dimensions} features; the limit is 6.")
    if teacher_dimensions < 12:
        raise FeatureContractError("Teacher must use at least 12 rich features.")
    if student_dimensions > teacher_dimensions / 2:
        raise FeatureContractError(
            f"Student has {student_dimensions} features, exceeding half of teacher's {teacher_dimensions}."
        )
