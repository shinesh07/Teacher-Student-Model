"""Deterministic, provenance-preserving sampling for pixel-level experiments.

The project permits pixel sampling but requires the number of pixels per image to be
reported.  This module has no file or mask access: callers must explicitly pass feature
rows (and, when permitted, labels), so it cannot bypass the dataset's anti-leakage
gatekeeper.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np


SamplingStrategy = Literal["uniform", "balanced"]


class SamplingError(ValueError):
    """Raised when a requested sample cannot satisfy its declared contract."""


def stable_seed(base_seed: int, namespace: str) -> int:
    """Derive a reproducible 32-bit seed without Python's randomized ``hash``.

    The namespace should encode both the operation and image identifier, for example
    ``"labelled/1001.jpg/teacher"``.  This makes sampling insensitive to process
    order while keeping every experiment tied to the global project seed.
    """
    if not isinstance(base_seed, (int, np.integer)):
        raise SamplingError("base_seed must be an integer")
    if not isinstance(namespace, str) or not namespace:
        raise SamplingError("namespace must be a non-empty string")
    payload = f"{int(base_seed)}\0{namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def _validate_count(population_size: int, count: int) -> None:
    if not isinstance(population_size, (int, np.integer)) or population_size < 1:
        raise SamplingError("population_size must be a positive integer")
    if not isinstance(count, (int, np.integer)) or count < 1:
        raise SamplingError("count must be a positive integer")
    if count > population_size:
        raise SamplingError(
            f"Requested {count} rows from a population of {population_size}; replacement is forbidden."
        )


def _validate_labels(labels: np.ndarray | None, population_size: int) -> np.ndarray:
    if labels is None:
        raise SamplingError("Balanced sampling requires labels.")
    result = np.asarray(labels)
    if result.ndim != 1 or len(result) != population_size:
        raise SamplingError(
            "labels must be one-dimensional with exactly one item per candidate row."
        )
    if not np.isfinite(result).all():
        raise SamplingError("labels contain NaN or infinity")
    return result


def _balanced_quotas(labels: np.ndarray, count: int, generator: np.random.Generator) -> dict[object, int]:
    """Allocate near-equal class quotas while respecting rare-class availability."""
    classes, inverse, available = np.unique(labels, return_inverse=True, return_counts=True)
    if count < len(classes):
        # It is mathematically impossible to preserve every class.  A deterministic
        # subset prevents an accidental label-order preference and remains explicit.
        selected = generator.choice(len(classes), size=count, replace=False)
        return {classes[index].item() if hasattr(classes[index], "item") else classes[index]: 1 for index in selected}

    quotas = np.full(len(classes), count // len(classes), dtype=np.int64)
    residual = int(count % len(classes))
    # Randomize only which equally eligible class gets an indivisible remainder.
    for index in generator.permutation(len(classes))[:residual]:
        quotas[index] += 1
    quotas = np.minimum(quotas, available)

    remaining = int(count - quotas.sum())
    # Redistribute a rare class's unused quota round-robin over classes that still have
    # rows, keeping the final balance as close as the data permits.
    while remaining:
        eligible = np.flatnonzero(quotas < available)
        if not len(eligible):  # Defensive; count <= population has already been checked.
            raise SamplingError("Unable to allocate the requested balanced sample.")
        for index in generator.permutation(eligible):
            if remaining == 0:
                break
            if quotas[index] < available[index]:
                quotas[index] += 1
                remaining -= 1
    return {
        classes[index].item() if hasattr(classes[index], "item") else classes[index]: int(quotas[index])
        for index in range(len(classes))
        if quotas[index]
    }


@dataclass(frozen=True)
class PixelSelection:
    """The full, serializable provenance for rows selected from one image."""

    image_id: str
    pixel_indices: np.ndarray
    population_size: int
    requested_count: int
    seed: int
    strategy: SamplingStrategy

    def __post_init__(self) -> None:
        indices = np.asarray(self.pixel_indices)
        if not isinstance(self.image_id, str) or not self.image_id:
            raise SamplingError("image_id must be a non-empty string")
        _validate_count(self.population_size, self.requested_count)
        if indices.ndim != 1 or len(indices) != self.requested_count:
            raise SamplingError("pixel_indices must contain exactly requested_count one-dimensional indices")
        if not np.issubdtype(indices.dtype, np.integer):
            raise SamplingError("pixel_indices must be integers")
        if np.any(indices < 0) or np.any(indices >= self.population_size):
            raise SamplingError("pixel_indices are outside the source image")
        if len(np.unique(indices)) != len(indices):
            raise SamplingError("pixel_indices contain duplicate rows")

    @property
    def indices_sha256(self) -> str:
        """Compact integrity value for large persisted index lists."""
        indices = np.asarray(self.pixel_indices, dtype="<i8")
        return hashlib.sha256(indices.tobytes()).hexdigest()

    def as_manifest(self, include_indices: bool = True) -> dict[str, object]:
        """Return JSON-compatible provenance suitable for a checkpoint manifest."""
        manifest: dict[str, object] = {
            "image_id": self.image_id,
            "population_size": self.population_size,
            "requested_count": self.requested_count,
            "selected_count": int(len(self.pixel_indices)),
            "seed": self.seed,
            "strategy": self.strategy,
            "pixel_indices_sha256": self.indices_sha256,
        }
        if include_indices:
            manifest["pixel_indices"] = np.asarray(self.pixel_indices, dtype=np.int64).tolist()
        return manifest


@dataclass(frozen=True)
class SampledRows:
    """Feature rows (and optional labels) paired with their exact source indices."""

    features: np.ndarray
    selection: PixelSelection
    labels: np.ndarray | None = None

    def __post_init__(self) -> None:
        features = np.asarray(self.features)
        if features.ndim != 2 or len(features) != self.selection.requested_count:
            raise SamplingError("features must be 2-D with exactly one row per selected pixel")
        if not np.isfinite(features).all():
            raise SamplingError("sampled features contain NaN or infinity")
        if self.labels is not None:
            labels = np.asarray(self.labels)
            if labels.ndim != 1 or len(labels) != len(features):
                raise SamplingError("sampled labels must be 1-D with one value per selected row")


def select_pixel_indices(
    *,
    image_id: str,
    population_size: int,
    count: int,
    base_seed: int,
    strategy: SamplingStrategy = "uniform",
    labels: np.ndarray | None = None,
    namespace: str = "pixel-sampling",
) -> PixelSelection:
    """Choose distinct pixel positions using a reproducible image-specific stream.

    ``balanced`` approximately equalizes the classes while taking every available row
    of a rare class when the desired half/quota is impossible. It is appropriate only
    when the caller is permitted to use labels; pseudo-label callers may also use it.
    """
    _validate_count(population_size, count)
    if strategy not in {"uniform", "balanced"}:
        raise SamplingError("strategy must be 'uniform' or 'balanced'")
    seed = stable_seed(base_seed, f"{namespace}\0{image_id}\0{strategy}\0{count}")
    generator = np.random.default_rng(seed)
    if strategy == "uniform":
        indices = generator.choice(population_size, size=count, replace=False)
    else:
        validated_labels = _validate_labels(labels, population_size)
        indices_by_class = {
            value.item() if hasattr(value, "item") else value: np.flatnonzero(validated_labels == value)
            for value in np.unique(validated_labels)
        }
        quotas = _balanced_quotas(validated_labels, count, generator)
        pieces = [
            generator.choice(indices_by_class[value], size=quota, replace=False)
            for value, quota in quotas.items()
        ]
        indices = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
    # Sorting makes row order independent of the per-class concatenation and clearer in
    # a saved manifest, while preserving the selected set and its reproducibility.
    indices = np.sort(np.asarray(indices, dtype=np.int64))
    return PixelSelection(
        image_id=image_id,
        pixel_indices=indices,
        population_size=int(population_size),
        requested_count=int(count),
        seed=seed,
        strategy=strategy,
    )


def sample_rows(
    *,
    image_id: str,
    features: np.ndarray,
    count: int,
    base_seed: int,
    labels: np.ndarray | None = None,
    strategy: SamplingStrategy = "uniform",
    namespace: str = "pixel-sampling",
) -> SampledRows:
    """Sample feature rows and optional aligned labels without any data-file access."""
    values = np.asarray(features)
    if values.ndim != 2:
        raise SamplingError("features must be a two-dimensional matrix")
    selection = select_pixel_indices(
        image_id=image_id,
        population_size=len(values),
        count=count,
        base_seed=base_seed,
        strategy=strategy,
        labels=labels,
        namespace=namespace,
    )
    selected_labels = None if labels is None else np.asarray(labels)[selection.pixel_indices]
    return SampledRows(
        features=values[selection.pixel_indices],
        labels=selected_labels,
        selection=selection,
    )


def stack_samples(samples: Iterable[SampledRows], require_labels: bool | None = None) -> tuple[np.ndarray, np.ndarray | None, list[dict[str, object]]]:
    """Combine same-schema samples and return arrays plus per-image provenance.

    ``require_labels=True`` is useful for supervised fitting; ``False`` protects an
    unlabeled operation from accidentally accepting labels; ``None`` preserves the
    samples' common label state.
    """
    collected = list(samples)
    if not collected:
        raise SamplingError("At least one sample is required")
    dimensions = {np.asarray(sample.features).shape[1] for sample in collected}
    if len(dimensions) != 1:
        raise SamplingError("All samples must use the same feature dimension")
    has_labels = [sample.labels is not None for sample in collected]
    if len(set(has_labels)) != 1:
        raise SamplingError("Cannot stack a mixture of labelled and unlabelled samples")
    labels_present = has_labels[0]
    if require_labels is not None and labels_present != require_labels:
        expected = "labelled" if require_labels else "unlabelled"
        raise SamplingError(f"This operation requires {expected} samples")
    features = np.concatenate([np.asarray(sample.features) for sample in collected], axis=0)
    labels = (
        np.concatenate([np.asarray(sample.labels) for sample in collected], axis=0)
        if labels_present
        else None
    )
    manifest = [sample.selection.as_manifest() for sample in collected]
    return features, labels, manifest
