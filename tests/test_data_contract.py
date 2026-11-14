from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data import FloodDataset, MaskAccessError, generate_fixed_splits


class FixedSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.files = [f"scene_{index:03d}.jpg" for index in range(290)]

    def test_sizes_are_the_published_sizes(self) -> None:
        splits = generate_fixed_splits(self.files)
        self.assertEqual((len(splits.labelled), len(splits.unlabelled), len(splits.validation), len(splits.test)), (40, 163, 29, 58))
        all_names = set(splits.labelled + splits.unlabelled + splits.validation + splits.test)
        self.assertEqual(len(all_names), 290)

    def test_membership_matches_the_published_rng_procedure(self) -> None:
        files = sorted(self.files)
        indices = np.random.default_rng(42).permutation(len(files))
        splits = generate_fixed_splits(self.files)
        self.assertEqual(list(splits.test), [files[i] for i in indices[:58]])
        self.assertEqual(list(splits.validation), [files[i] for i in indices[58:87]])
        self.assertEqual(list(splits.labelled), [files[i] for i in indices[87:127]])
        self.assertEqual(list(splits.unlabelled), [files[i] for i in indices[127:]])


class MaskPermissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        (root / "Image").mkdir()
        (root / "Mask").mkdir()
        for index in range(290):
            (root / "Image" / f"scene_{index:03d}.jpg").touch()
            (root / "Mask" / f"scene_{index:03d}.png").touch()
        self.dataset = FloodDataset(root=root)
        self.splits = self.dataset.splits()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_unlabelled_mask_is_blocked_before_opening(self) -> None:
        with self.assertRaises(MaskAccessError):
            self.dataset.mask_path(self.splits.unlabelled[0], "unlabelled", "teacher_fit")

    def test_test_mask_is_blocked_outside_final_evaluation(self) -> None:
        with self.assertRaises(MaskAccessError):
            self.dataset.mask_path(self.splits.test[0], "test", "validation")

    def test_test_mask_audit_is_allowed_but_is_a_distinct_nonfitting_purpose(self) -> None:
        self.assertTrue(self.dataset.mask_path(self.splits.test[0], "test", "audit").is_file())

    def test_wrong_partition_cannot_bypass_a_gate(self) -> None:
        with self.assertRaises(MaskAccessError):
            self.dataset.mask_path(self.splits.test[0], "labelled", "teacher_fit")

    def test_final_test_and_labelled_fit_are_the_only_relevant_allowed_paths(self) -> None:
        self.assertTrue(self.dataset.mask_path(self.splits.test[0], "test", "final_evaluation").is_file())
        self.assertTrue(self.dataset.mask_path(self.splits.labelled[0], "labelled", "teacher_fit").is_file())


if __name__ == "__main__":
    unittest.main()
