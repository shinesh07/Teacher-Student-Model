"""Small, deterministic writers for report-ready experiment artifacts.

The experiment runner owns model selection and evaluation.  This module deliberately
does not know about data splits, masks, estimators, or model objects; it only turns
already-verified JSON-like evidence into durable tables and labeled figures.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from .provenance import write_json_atomic


class ArtifactError(ValueError):
    """Raised when a report artifact cannot be generated from trustworthy inputs."""


def write_json(path: str | Path, payload: Any) -> Path:
    """Atomically write JSON evidence and return its final path."""
    destination = Path(path)
    try:
        json.dumps(payload, sort_keys=True, default=str)
    except (TypeError, ValueError) as error:
        raise ArtifactError("JSON payload is not serializable") from error
    write_json_atomic(destination, payload)
    return destination


def _csv_cell(value: Any) -> str | int | float:
    """Make nested evidence readable and lossless in a CSV cell."""
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def write_rows_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Atomically write mapping rows as a deterministic CSV table.

    Columns are the sorted union of all keys.  This prevents a different row order or
    an optional late metric from silently dropping a column, while still making a
    table with no rows explicit rather than producing an ambiguous empty file.
    """
    destination = Path(path)
    collected = list(rows)
    if not all(isinstance(row, Mapping) for row in collected):
        raise ArtifactError("CSV rows must all be mappings")
    fieldnames = sorted({str(key) for row in collected for key in row})
    if not fieldnames:
        raise ArtifactError("CSV output needs at least one row with at least one key")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=destination.parent,
        prefix=f".{destination.name}.", suffix=".tmp", delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        writer = csv.DictWriter(temporary, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in collected:
            normalized = {str(key): _csv_cell(value) for key, value in row.items()}
            writer.writerow(normalized)
    try:
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def _validate_plot_destination(destination: str | Path) -> Path:
    path = Path(destination)
    if path.suffix.lower() not in {".png", ".pdf", ".svg"}:
        raise ArtifactError("Figure destination must use .png, .pdf, or .svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _finite_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ArtifactError(f"{label} must be numeric") from error
    if not np.isfinite(number):
        raise ArtifactError(f"{label} must be finite")
    return number


def _save_figure(figure: plt.Figure, destination: Path) -> Path:
    """Save deterministic image metadata, then always release Matplotlib memory."""
    try:
        figure.savefig(
            destination,
            dpi=160,
            metadata={"Software": "CS3103 reproducible ML pipeline", "CreationDate": None},
        )
    finally:
        plt.close(figure)
    return destination


def write_pipeline_diagram(
    destination: str | Path,
    *,
    title: str = "Flood segmentation teacher-student pipeline",
) -> Path:
    """Render a compact architecture diagram with explicit training/inference flow.

    The figure intentionally distinguishes labels and teacher outputs from the direct
    compact-feature student inference route, satisfying the guidelines' diagram rule
    without implying that test masks enter a fit operation.
    """
    path = _validate_plot_destination(destination)
    figure, axis = plt.subplots(figsize=(13.0, 4.9), constrained_layout=True)
    axis.set_xlim(0.0, 13.0)
    axis.set_ylim(0.0, 5.0)
    axis.axis("off")
    axis.set_title(title, fontsize=15, fontweight="bold", pad=12)

    blocks = [
        (0.3, 3.3, 1.7, 0.82, "L images\n+GT masks", "#dbeafe"),
        (0.3, 1.2, 1.7, 0.82, "U images\n(no masks)", "#e2e8f0"),
        (2.45, 3.3, 1.9, 0.82, "Rich features\n20 dimensions", "#fef3c7"),
        (4.9, 3.3, 1.8, 0.82, "Teacher\nKNN / RF / GNB", "#fed7aa"),
        (7.2, 3.3, 1.9, 0.82, "Confidence filter\nτ selected on validation", "#fde68a"),
        (9.55, 3.3, 1.8, 0.82, "Student\n≤6 features", "#bbf7d0"),
        (11.65, 3.3, 1.1, 0.82, "Flood\nmask", "#bfdbfe"),
        (2.45, 1.2, 1.9, 0.82, "Compact features\n6 dimensions", "#dcfce7"),
        (7.2, 1.2, 1.9, 0.82, "Validation only\nthreshold / τ", "#e9d5ff"),
        (9.55, 1.2, 3.2, 0.82, "Test images: compact features → frozen student → prediction", "#f1f5f9"),
    ]
    for x, y, width, height, text, color in blocks:
        patch = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.05,rounding_size=0.10",
            linewidth=1.1, edgecolor="#334155", facecolor=color,
        )
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=8.8)

    arrows = [
        ((2.0, 3.71), (2.45, 3.71)),
        ((4.35, 3.71), (4.9, 3.71)),
        ((6.7, 3.71), (7.2, 3.71)),
        ((9.1, 3.71), (9.55, 3.71)),
        ((11.35, 3.71), (11.65, 3.71)),
        ((2.0, 1.61), (2.45, 1.61)),
        ((4.35, 1.61), (9.55, 1.61)),
        ((2.0, 1.61), (4.9, 3.3)),
        ((4.35, 1.61), (9.55, 3.3)),
        ((7.2, 2.02), (7.2, 3.3)),
    ]
    for start, end in arrows:
        axis.annotate(
            "", xy=end, xytext=start,
            arrowprops={"arrowstyle": "->", "color": "#475569", "lw": 1.2},
        )
    axis.text(
        0.3, 0.36,
        "Leakage control: models fit on L or L∪U images only; validation selects configurations; "
        "test masks are restricted to descriptive audit or frozen final metrics.",
        fontsize=8.6, color="#334155",
    )
    return _save_figure(figure, path)


def plot_tau_study(
    rows: Iterable[Mapping[str, Any]],
    destination: str | Path,
    *,
    title: str = "Hard-distillation confidence study",
) -> Path:
    """Plot validation IoU and confidence-retained fraction over tau values.

    Invalid tau candidates are intentionally omitted from curves; their reasons belong
    in the accompanying CSV/JSON evidence instead of becoming fabricated numeric points.
    """
    points: list[tuple[float, float, float]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ArtifactError("Tau-study rows must be mappings")
        if row.get("valid", True) is False:
            continue
        try:
            tau = _finite_number(row["tau"], "tau")
            iou = _finite_number(row["validation_iou"], "validation_iou")
            retained = _finite_number(row["retained_fraction"], "retained_fraction")
        except KeyError as error:
            raise ArtifactError(f"Tau-study row is missing {error.args[0]!r}") from error
        if not 0.0 <= tau <= 1.0 or not 0.0 <= iou <= 1.0 or not 0.0 <= retained <= 1.0:
            raise ArtifactError("tau, validation_iou, and retained_fraction must lie in [0, 1]")
        points.append((tau, iou, retained))
    if not points:
        raise ArtifactError("Tau study contains no valid plottable candidates")
    points.sort(key=lambda point: point[0])
    tau, iou, retained = (np.asarray(values, dtype=float) for values in zip(*points))
    path = _validate_plot_destination(destination)
    figure, left = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    right = left.twinx()
    left.plot(tau, iou * 100.0, marker="o", color="#2563eb", linewidth=2, label="Validation IoU")
    right.plot(tau, retained * 100.0, marker="s", color="#d97706", linewidth=2, label="Pixels retained")
    left.set_xlabel("Confidence threshold τ")
    left.set_ylabel("Validation flood IoU (%)", color="#2563eb")
    right.set_ylabel("Transfer pixels retained (%)", color="#d97706")
    left.set_ylim(0.0, 100.0)
    right.set_ylim(0.0, 100.0)
    left.set_xticks(tau)
    left.grid(axis="y", alpha=0.25)
    left.set_title(title)
    lines = left.get_lines() + right.get_lines()
    left.legend(lines, [line.get_label() for line in lines], loc="best")
    return _save_figure(figure, path)


def plot_kmeans_selection(
    rows: Iterable[Mapping[str, Any]],
    destination: str | Path,
    *,
    title: str = "K-Means teacher cluster-count selection",
) -> Path:
    """Plot unsupervised silhouette and inertia evidence against K_T."""
    points: list[tuple[int, float, float]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ArtifactError("K-Means selection rows must be mappings")
        try:
            clusters = int(row["clusters"])
            silhouette = _finite_number(row["silhouette"], "silhouette")
            inertia = _finite_number(row["inertia"], "inertia")
        except KeyError as error:
            raise ArtifactError(f"K-Means selection row is missing {error.args[0]!r}") from error
        if clusters < 2 or not -1.0 <= silhouette <= 1.0 or inertia < 0.0:
            raise ArtifactError("Invalid K-Means clusters, silhouette, or inertia values")
        points.append((clusters, silhouette, inertia))
    if not points:
        raise ArtifactError("K-Means selection contains no points")
    points.sort(key=lambda point: point[0])
    clusters, silhouette, inertia = (np.asarray(values) for values in zip(*points))
    path = _validate_plot_destination(destination)
    figure, left = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    right = left.twinx()
    left.plot(clusters, silhouette, marker="o", color="#7c3aed", linewidth=2, label="Silhouette")
    right.plot(clusters, inertia, marker="s", color="#0f766e", linewidth=2, label="Inertia")
    left.set_xlabel("Teacher clusters K_T")
    left.set_ylabel("Silhouette score", color="#7c3aed")
    right.set_ylabel("Inertia", color="#0f766e")
    left.set_xticks(clusters)
    left.set_ylim(-1.0, 1.0)
    left.grid(axis="y", alpha=0.25)
    left.set_title(title)
    lines = left.get_lines() + right.get_lines()
    left.legend(lines, [line.get_label() for line in lines], loc="best")
    return _save_figure(figure, path)
