from __future__ import annotations

import unittest

import numpy as np

from src.sampling import (
    SamplingError,
    sample_rows,
    select_pixel_indices,
    stable_seed,
    stack_samples,
)


class SamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = np.arange(400, dtype=np.float32).reshape(100, 4)
        self.labels = np.array([0] * 90 + [1] * 10, dtype=np.uint8)

    def test_stable_seed_uses_namespace_without_process_hashing(self) -> None:
        self.assertEqual(stable_seed(42, "a/1.jpg"), stable_seed(42, "a/1.jpg"))
        self.assertNotEqual(stable_seed(42, "a/1.jpg"), stable_seed(42, "a/2.jpg"))

    def test_uniform_selection_is_exact_unique_sorted_and_reproducible(self) -> None:
        first = select_pixel_indices(
            image_id="one.jpg", population_size=100, count=25, base_seed=42
        )
        second = select_pixel_indices(
            image_id="one.jpg", population_size=100, count=25, base_seed=42
        )
        self.assertEqual(len(first.pixel_indices), 25)
        self.assertEqual(len(np.unique(first.pixel_indices)), 25)
        self.assertTrue(np.all(first.pixel_indices[:-1] < first.pixel_indices[1:]))
        np.testing.assert_array_equal(first.pixel_indices, second.pixel_indices)
        self.assertEqual(first.indices_sha256, second.indices_sha256)

    def test_balanced_selection_preserves_minority_when_possible(self) -> None:
        selected = select_pixel_indices(
            image_id="one.jpg",
            population_size=100,
            count=20,
            base_seed=42,
            strategy="balanced",
            labels=self.labels,
        )
        picked = self.labels[selected.pixel_indices]
        self.assertEqual(int((picked == 0).sum()), 10)
        self.assertEqual(int((picked == 1).sum()), 10)

    def test_balanced_selection_redistributes_when_minority_is_exhausted(self) -> None:
        selected = select_pixel_indices(
            image_id="one.jpg",
            population_size=100,
            count=50,
            base_seed=42,
            strategy="balanced",
            labels=self.labels,
        )
        picked = self.labels[selected.pixel_indices]
        self.assertEqual(len(picked), 50)
        self.assertEqual(int((picked == 1).sum()), 10)
        self.assertEqual(int((picked == 0).sum()), 40)

    def test_sample_rows_aligns_features_labels_and_manifest(self) -> None:
        sampled = sample_rows(
            image_id="one.jpg",
            features=self.features,
            labels=self.labels,
            count=20,
            base_seed=42,
            strategy="balanced",
        )
        np.testing.assert_array_equal(
            sampled.features[:, 0] / 4,
            sampled.selection.pixel_indices.astype(np.float32),
        )
        np.testing.assert_array_equal(sampled.labels, self.labels[sampled.selection.pixel_indices])
        manifest = sampled.selection.as_manifest()
        self.assertEqual(manifest["selected_count"], 20)
        self.assertEqual(manifest["pixel_indices"], sampled.selection.pixel_indices.tolist())

    def test_stack_samples_returns_single_schema_and_provenance(self) -> None:
        first = sample_rows(
            image_id="one.jpg", features=self.features, labels=self.labels, count=10, base_seed=42
        )
        second = sample_rows(
            image_id="two.jpg", features=self.features, labels=self.labels, count=15, base_seed=42
        )
        features, labels, manifest = stack_samples([first, second], require_labels=True)
        self.assertEqual(features.shape, (25, 4))
        self.assertEqual(labels.shape, (25,))
        self.assertEqual([entry["image_id"] for entry in manifest], ["one.jpg", "two.jpg"])

    def test_invalid_requests_fail_instead_of_sampling_with_replacement(self) -> None:
        with self.assertRaises(SamplingError):
            select_pixel_indices(image_id="x", population_size=5, count=6, base_seed=42)
        with self.assertRaises(SamplingError):
            select_pixel_indices(
                image_id="x", population_size=5, count=2, base_seed=42, strategy="balanced"
            )
        with self.assertRaises(SamplingError):
            sample_rows(image_id="x", features=np.zeros(5), count=2, base_seed=42)
        unlabelled = sample_rows(image_id="x", features=self.features, count=10, base_seed=42)
        with self.assertRaises(SamplingError):
            stack_samples([unlabelled], require_labels=True)


if __name__ == "__main__":
    unittest.main()
