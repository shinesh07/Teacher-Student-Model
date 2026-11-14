"""Dataset access with fixed splits and explicit anti-leakage permissions.

The only code allowed to construct paths to ground-truth masks lives here. Training
code receives arrays, never raw mask locations, which makes accidental access to U or
test labels fail loudly rather than silently contaminating an experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from PIL import Image


Partition = Literal["labelled", "unlabelled", "validation", "test"]
MaskPurpose = Literal[
    "teacher_fit",
    "baseline_fit",
    "cluster_mapping",
    "validation",
    "audit",
    "final_evaluation",
]


class DatasetContractError(RuntimeError):
    """Raised when raw data violate the published dataset contract."""


class MaskAccessError(PermissionError):
    """Raised before an impermissible mask path can be opened."""


@dataclass(frozen=True)
class Splits:
    """Deterministic fixed split required by the project brief."""

    labelled: tuple[str, ...]
    unlabelled: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def names(self, partition: Partition) -> tuple[str, ...]:
        return getattr(self, partition)

    def partition_for(self, filename: str) -> Partition:
        for partition in ("labelled", "unlabelled", "validation", "test"):
            if filename in self.names(partition):
                return partition
        raise KeyError(f"{filename!r} is not present in the fixed split")

    def as_dict(self) -> dict[str, list[str]]:
        return {
            partition: list(self.names(partition))
            for partition in ("labelled", "unlabelled", "validation", "test")
        }


def generate_fixed_splits(filenames: Iterable[str], seed: int = 42) -> Splits:
    """Implement the brief's split code exactly in its essential operations."""
    files = sorted(
        name for name in filenames if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(files) != 290:
        raise DatasetContractError(f"Expected 290 images, found {len(files)}")
    indices = np.random.default_rng(seed).permutation(len(files))
    return Splits(
        test=tuple(files[i] for i in indices[:58]),
        validation=tuple(files[i] for i in indices[58:87]),
        labelled=tuple(files[i] for i in indices[87:127]),
        unlabelled=tuple(files[i] for i in indices[127:]),
    )


@dataclass
class FloodDataset:
    """Validated raw dataset plus a gatekeeper for label access."""

    root: Path
    image_dir_name: str = "Image"
    mask_dir_name: str = "Mask"
    image_size: tuple[int, int] = (256, 256)
    mask_positive_threshold: int = 128
    seed: int = 42

    @property
    def image_dir(self) -> Path:
        return self.root / self.image_dir_name

    @property
    def mask_dir(self) -> Path:
        return self.root / self.mask_dir_name

    def filenames(self) -> tuple[str, ...]:
        if not self.image_dir.is_dir():
            raise DatasetContractError(f"Image directory is missing: {self.image_dir}")
        return tuple(
            sorted(
                path.name
                for path in self.image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
        )

    def splits(self) -> Splits:
        return generate_fixed_splits(self.filenames(), self.seed)

    def image_path(self, filename: str) -> Path:
        path = self.image_dir / filename
        if not path.is_file():
            raise DatasetContractError(f"Missing image: {path}")
        return path

    def _assert_mask_permission(self, partition: Partition, purpose: MaskPurpose) -> None:
        if partition == "unlabelled":
            raise MaskAccessError("Masks of U are never available to this pipeline.")
        # Part A explicitly asks us to report the test flood-pixel fraction. That is a
        # descriptive audit, not a tuning signal, so it is the sole pre-final-test
        # exception. All fitting, thresholding, feature selection and mapping paths
        # remain blocked by their distinct purposes.
        if partition == "test" and purpose not in {"audit", "final_evaluation"}:
            raise MaskAccessError(
                "Test masks are available only for the descriptive audit or frozen final evaluation."
            )
        if partition == "validation" and purpose not in {"validation", "audit"}:
            raise MaskAccessError("Validation masks may only support audit or validation selection.")
        if partition == "labelled" and purpose not in {
            "teacher_fit",
            "baseline_fit",
            "cluster_mapping",
            "audit",
        }:
            raise MaskAccessError("Labelled masks cannot be used for this requested purpose.")

    def mask_path(self, filename: str, partition: Partition, purpose: MaskPurpose) -> Path:
        """Return a mask location only after enforcing the project data-use rules."""
        self._assert_mask_permission(partition, purpose)
        expected = self.splits().partition_for(filename)
        if expected != partition:
            raise MaskAccessError(
                f"Filename {filename!r} belongs to {expected!r}, not {partition!r}."
            )
        path = self.mask_dir / f"{Path(filename).stem}.png"
        if not path.is_file():
            raise DatasetContractError(f"Missing corresponding mask: {path}")
        return path

    def load_image(self, filename: str) -> np.ndarray:
        """Load a source photograph as a 256 x 256 RGB float image in [0, 1]."""
        with Image.open(self.image_path(filename)) as source:
            # Configuration uses NumPy's (height, width) convention while Pillow's
            # resize API expects (width, height). The project size is square, but this
            # explicit conversion keeps the invariant testable for any future size.
            pillow_size = (self.image_size[1], self.image_size[0])
            image = source.convert("RGB").resize(pillow_size, Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        if array.shape != (*self.image_size, 3):
            raise DatasetContractError(f"Unexpected processed image shape {array.shape}")
        return array

    def load_mask(
        self, filename: str, partition: Partition, purpose: MaskPurpose
    ) -> np.ndarray:
        """Binarise before nearest-neighbour resizing, as specified in the brief."""
        with Image.open(self.mask_path(filename, partition, purpose)) as source:
            raw = np.asarray(source.convert("L"), dtype=np.uint8)
        binary = (raw >= self.mask_positive_threshold).astype(np.uint8)
        binary_image = Image.fromarray(binary, mode="L")
        pillow_size = (self.image_size[1], self.image_size[0])
        resized = binary_image.resize(pillow_size, Image.Resampling.NEAREST)
        result = (np.asarray(resized, dtype=np.uint8) > 0).astype(np.uint8)
        if result.shape != self.image_size or not set(np.unique(result)).issubset({0, 1}):
            raise DatasetContractError("Mask preprocessing did not produce a binary 256 x 256 mask")
        return result

    def validate_structure(self) -> dict[str, int]:
        """Check file counts and exact stem pairing without reading unlabelled mask pixels."""
        names = self.filenames()
        if not self.mask_dir.is_dir():
            raise DatasetContractError(f"Mask directory is missing: {self.mask_dir}")
        mask_names = tuple(sorted(path.name for path in self.mask_dir.glob("*.png")))
        expected_masks = tuple(f"{Path(name).stem}.png" for name in names)
        missing = sorted(set(expected_masks) - set(mask_names))
        unexpected = sorted(set(mask_names) - set(expected_masks))
        if missing or unexpected or len(mask_names) != 290:
            raise DatasetContractError(
                f"Mask pairing failed: missing={missing[:5]}, unexpected={unexpected[:5]}, "
                f"mask_count={len(mask_names)}"
            )
        split_sizes = {name: len(self.splits().names(name)) for name in (
            "labelled", "unlabelled", "validation", "test"
        )}
        if split_sizes != {"labelled": 40, "unlabelled": 163, "validation": 29, "test": 58}:
            raise DatasetContractError(f"Wrong split sizes: {split_sizes}")
        return {"images": len(names), "masks": len(mask_names), **split_sizes}


def dataset_from_config(config: dict) -> FloodDataset:
    """Construct the dataset object from the checked-in config without accessing data."""
    from .config import resolve_project_path

    spec = config["dataset"]
    return FloodDataset(
        root=resolve_project_path(spec["root"]),
        image_dir_name=spec["image_dir"],
        mask_dir_name=spec["mask_dir"],
        image_size=tuple(spec["image_size"]),
        mask_positive_threshold=spec["mask_positive_threshold"],
        seed=config["seed"],
    )
