"""Dataset-card, structural-audit and leak-safe EDA functions."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .config import config_digest
from .data import FloodDataset
from .provenance import runtime_context, sha256_file, write_json_atomic


def _source_image_inventory(dataset: FloodDataset) -> dict[str, Any]:
    """Inspect raw image headers and bytes without looking at unlabelled labels."""
    modes: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    hashes: dict[str, list[str]] = {}
    corrupt: list[str] = []
    for filename in dataset.filenames():
        path = dataset.image_path(filename)
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                modes[image.mode] += 1
                dimensions[f"{image.width}x{image.height}"] += 1
        except Exception as error:  # pragma: no cover - regression protection for real data
            corrupt.append(f"{filename}: {error}")
            continue
        digest = sha256_file(path)
        hashes.setdefault(digest, []).append(filename)
    splits = dataset.splits()
    duplicate_groups = [names for names in hashes.values() if len(names) > 1]
    duplicate_details = [
        {
            "filenames": sorted(names),
            "partitions": sorted({splits.partition_for(name) for name in names}),
        }
        for names in duplicate_groups
    ]
    cross_partition_duplicates = [
        item for item in duplicate_details if len(item["partitions"]) > 1
    ]
    return {
        "modes": dict(sorted(modes.items())),
        "source_dimensions": dict(sorted(dimensions.items())),
        "corrupt_images": corrupt,
        "exact_duplicate_groups": duplicate_details,
        "cross_partition_duplicate_groups": cross_partition_duplicates,
        "fixed_split_duplicate_warning": (
            "Exact duplicate source photographs span fixed partitions. The required split "
            "was retained unchanged; this is documented as a validity limitation, not "
            "corrected through data-driven re-splitting."
            if cross_partition_duplicates
            else None
        ),
        "raw_image_inventory_sha256": sha256_file_list(dataset.image_dir),
    }


def sha256_file_list(directory: Path) -> str:
    """Produce one deterministic manifest hash over names and file hashes."""
    rows = [
        {"name": path.name, "sha256": sha256_file(path)}
        for path in sorted(directory.iterdir())
        if path.is_file()
    ]
    from .provenance import stable_digest

    return stable_digest(rows)


def _metadata_audit(dataset: FloodDataset) -> dict[str, Any]:
    path = dataset.root / "metadata.csv"
    if not path.is_file():
        return {"exists": False}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames or []
    image_values = [row.get("Image", "") for row in rows]
    mask_values = [row.get("Mask", "") for row in rows]
    expected = set(dataset.filenames())
    paired = all(
        value and f"{Path(image).stem}.png" == value for image, value in zip(image_values, mask_values)
    )
    return {
        "exists": True,
        "columns": fields,
        "rows": len(rows),
        "missing_cells": sum(value == "" for row in rows for value in row.values()),
        "image_filenames_match_dataset": set(image_values) == expected,
        "mask_stems_match_images": paired,
        "metadata_sha256": sha256_file(path),
    }


def _mask_class_statistics(dataset: FloodDataset) -> dict[str, dict[str, float | int]]:
    """Stream permitted masks and report post-preprocessing flood-pixel fractions."""
    statistics: dict[str, dict[str, float | int]] = {}
    split_purposes = {
        "labelled": "audit",
        "validation": "audit",
        "test": "audit",
    }
    splits = dataset.splits()
    for partition, purpose in split_purposes.items():
        positive = 0
        total = 0
        for filename in splits.names(partition):
            mask = dataset.load_mask(filename, partition, purpose)  # type: ignore[arg-type]
            positive += int(mask.sum())
            total += int(mask.size)
        statistics[partition] = {
            "images": len(splits.names(partition)),
            "positive_pixels": positive,
            "total_pixels": total,
            "flood_pixel_fraction": positive / total if total else float("nan"),
        }
    return statistics


def _labelled_rgb_statistics(dataset: FloodDataset) -> dict[str, list[float] | int]:
    """Compute train-only descriptive RGB statistics for documented feature design."""
    count = 0
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_square_sum = np.zeros(3, dtype=np.float64)
    for filename in dataset.splits().labelled:
        image = dataset.load_image(filename)
        pixels = image.reshape(-1, 3).astype(np.float64)
        count += len(pixels)
        channel_sum += pixels.sum(axis=0)
        channel_square_sum += np.square(pixels).sum(axis=0)
    mean = channel_sum / count
    variance = np.maximum(channel_square_sum / count - np.square(mean), 0.0)
    return {
        "pixels": count,
        "mean_rgb": mean.tolist(),
        "std_rgb": np.sqrt(variance).tolist(),
    }


def write_audit_figures(
    dataset: FloodDataset, class_stats: dict[str, dict[str, float | int]], output_dir: Path
) -> list[str]:
    """Save labeled-data EDA figures; test ground truth is not visualized here."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: list[str] = []

    labels = ["L", "Validation", "Test"]
    fractions = [
        float(class_stats[partition]["flood_pixel_fraction"])
        for partition in ("labelled", "validation", "test")
    ]
    figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    bars = axis.bar(labels, fractions, color=["#3b82f6", "#f59e0b", "#64748b"])
    axis.set_ylabel("Flood-pixel fraction")
    axis.set_ylim(0, max(0.05, max(fractions) * 1.2))
    axis.set_title("Post-preprocessing flood prevalence by fixed split")
    for bar, fraction in zip(bars, fractions):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{fraction:.3%}", ha="center", va="bottom")
    class_path = output_dir / "audit_class_balance.png"
    figure.savefig(class_path, dpi=160)
    plt.close(figure)
    figure_paths.append(str(class_path))

    examples = dataset.splits().labelled[:4]
    figure, axes = plt.subplots(2, len(examples), figsize=(3.2 * len(examples), 5), constrained_layout=True)
    for index, filename in enumerate(examples):
        image = dataset.load_image(filename)
        mask = dataset.load_mask(filename, "labelled", "audit")
        axes[0, index].imshow(image)
        axes[0, index].set_title(filename)
        axes[1, index].imshow(mask, cmap="Blues", vmin=0, vmax=1)
        for axis in axes[:, index]:
            axis.axis("off")
    axes[0, 0].set_ylabel("RGB image")
    axes[1, 0].set_ylabel("Binarised mask")
    examples_path = output_dir / "audit_labelled_examples.png"
    figure.savefig(examples_path, dpi=160)
    plt.close(figure)
    figure_paths.append(str(examples_path))
    return figure_paths


def audit_dataset(dataset: FloodDataset, config: dict[str, Any], figure_dir: Path | None = None) -> dict[str, Any]:
    """Perform and return the complete non-fitting data audit.

    No model, scaler or feature-selection method appears here. The only test-label
    information accessed is the project-required aggregate flood fraction.
    """
    structure = dataset.validate_structure()
    image_inventory = _source_image_inventory(dataset)
    if image_inventory["corrupt_images"]:
        raise RuntimeError(f"Corrupt input images found: {image_inventory['corrupt_images'][:3]}")
    class_stats = _mask_class_statistics(dataset)
    figures = write_audit_figures(dataset, class_stats, figure_dir) if figure_dir else []
    return {
        "checkpoint": "CP01_dataset_audit",
        "config_sha256": config_digest(config),
        "dataset_source": {
            "slug": config["dataset"]["slug"],
            "root": str(dataset.root),
            "mask_threshold": dataset.mask_positive_threshold,
            "processed_size": list(dataset.image_size),
            "rgb_loader": "Pillow convert('RGB'); no BGR interpretation is used",
        },
        "structure": structure,
        "metadata": _metadata_audit(dataset),
        "image_inventory": image_inventory,
        "class_statistics": class_stats,
        "labelled_rgb_statistics": _labelled_rgb_statistics(dataset),
        "figures": figures,
        "runtime": runtime_context(("numpy", "Pillow", "matplotlib", "scikit-learn", "scipy")),
    }


def write_audit_checkpoint(report: dict[str, Any], path: Path) -> None:
    """Persist a successful audit only after all its assertions completed."""
    write_json_atomic(path, report)
