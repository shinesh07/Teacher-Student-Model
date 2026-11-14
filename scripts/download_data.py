#!/usr/bin/env python3
"""Download and validate only the public dataset named in the project brief.

The Kaggle endpoint is pinned to version 1. Extraction is staged and refuses to replace
an existing raw dataset, preventing accidental loss of a manually verified download.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


SLUG = "faizalkarim/flood-area-segmentation"
VERSION = 1
URL = f"https://www.kaggle.com/api/v1/datasets/download/{SLUG}?datasetVersionNumber={VERSION}"
EXPECTED_IMAGES = 290


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"Unsafe archive member: {member.filename!r}")
        bundle.extractall(destination)


def validate_layout(root: Path) -> dict[str, object]:
    image_dir, mask_dir = root / "Image", root / "Mask"
    metadata = root / "metadata.csv"
    images = sorted(path for path in image_dir.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    masks = sorted(mask_dir.glob("*.png"))
    image_stems, mask_stems = {path.stem for path in images}, {path.stem for path in masks}
    if not image_dir.is_dir() or not mask_dir.is_dir() or not metadata.is_file():
        raise RuntimeError("Archive must contain Image/, Mask/, and metadata.csv at its root")
    if len(images) != EXPECTED_IMAGES or len(masks) != EXPECTED_IMAGES:
        raise RuntimeError(f"Expected {EXPECTED_IMAGES} images and masks; got {len(images)} and {len(masks)}")
    if image_stems != mask_stems:
        raise RuntimeError("Image and mask stems do not match exactly")
    return {
        "images": len(images),
        "masks": len(masks),
        "metadata": metadata.name,
        "image_extensions": sorted({path.suffix.lower() for path in images}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    destination = args.destination.resolve()
    target_items = [destination / name for name in ("Image", "Mask", "metadata.csv")]
    if any(path.exists() for path in target_items):
        raise SystemExit(f"Refusing to overwrite an existing raw dataset in {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="flood-download-") as temporary:
        staging = Path(temporary)
        archive = staging / "dataset.zip"
        request = urllib.request.Request(URL, headers={"User-Agent": "CS3103-reproducible-project/1.0"})
        print(f"Downloading {SLUG} version {VERSION} from Kaggle...")
        with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        extracted = staging / "extracted"
        extracted.mkdir()
        safe_extract(archive, extracted)
        validation = validate_layout(extracted)
        for name in ("Image", "Mask", "metadata.csv"):
            shutil.move(str(extracted / name), str(destination / name))
        manifest = {
            "dataset_slug": SLUG,
            "dataset_version": VERSION,
            "download_url": URL,
            "archive_sha256": sha256(archive),
            **validation,
        }
        with (destination / "DATASET_MANIFEST.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
