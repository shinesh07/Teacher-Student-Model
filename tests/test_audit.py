from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.audit import _metadata_audit, _mask_class_statistics, sha256_file_list
from src.data import FloodDataset


class AuditHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        (root / "Image").mkdir()
        (root / "Mask").mkdir()
        rows = ["Image,Mask"]
        for index in range(290):
            name = f"scene_{index:03d}"
            Image.fromarray(np.full((2, 3, 3), index % 255, dtype=np.uint8), "RGB").save(root / "Image" / f"{name}.jpg")
            Image.fromarray(np.array([[127, 128, 255], [0, 127, 128]], dtype=np.uint8), "L").save(root / "Mask" / f"{name}.png")
            rows.append(f"{name}.jpg,{name}.png")
        (root / "metadata.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        self.dataset = FloodDataset(root=root, image_size=(3, 2))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_mask_audit_uses_only_allowed_partitions_and_binarises_threshold(self) -> None:
        stats = _mask_class_statistics(self.dataset)
        self.assertEqual(set(stats), {"labelled", "validation", "test"})
        self.assertEqual(stats["labelled"]["total_pixels"], 40 * 6)
        self.assertAlmostEqual(float(stats["validation"]["flood_pixel_fraction"]), 3 / 6)

    def test_metadata_audit_validates_all_mappings(self) -> None:
        report = _metadata_audit(self.dataset)
        self.assertTrue(report["exists"])
        self.assertTrue(report["image_filenames_match_dataset"])
        self.assertTrue(report["mask_stems_match_images"])
        self.assertEqual(report["missing_cells"], 0)

    def test_file_list_hash_is_stable(self) -> None:
        first = sha256_file_list(self.dataset.image_dir)
        second = sha256_file_list(self.dataset.image_dir)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
