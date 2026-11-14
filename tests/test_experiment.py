from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np

from src.experiment import (
    DevelopmentContractError,
    _threshold_selection,
    _validate_feature_schema,
    fit_budget_for_family,
    mapping_from_all_labelled_images,
    select_global_rows,
)


@dataclass(frozen=True)
class _FakeSplits:
    labelled: tuple[str, ...]


class _FakeDataset:
    """Minimal in-memory source used to prove the L-only K-Means mapping path."""

    def __init__(self) -> None:
        self.mask_calls: list[tuple[str, str, str]] = []
        self._names = ("labelled_a.jpg", "labelled_b.jpg")
        self._image = np.array(
            [
                [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
                [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            ],
            dtype=np.float32,
        )
        self._masks = {
            name: np.array([[0, 1], [0, 1]], dtype=np.uint8) for name in self._names
        }

    def splits(self) -> _FakeSplits:
        return _FakeSplits(self._names)

    def load_image(self, filename: str) -> np.ndarray:
        if filename not in self._names:
            raise AssertionError(f"unexpected image access: {filename}")
        return self._image.copy()

    def load_mask(self, filename: str, partition: str, purpose: str) -> np.ndarray:
        self.mask_calls.append((filename, partition, purpose))
        if filename not in self._names:
            raise AssertionError(f"unexpected mask access: {filename}")
        if (partition, purpose) != ("labelled", "cluster_mapping"):
            raise AssertionError(f"unexpected mask access policy: {(partition, purpose)!r}")
        return self._masks[filename].copy()


class _AlternatingClusterModel:
    """Cheap model double that makes raw-cluster majority votes observable."""

    n_clusters = 2

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (np.arange(len(features), dtype=np.int64) % 2).astype(np.int64)


class ExperimentHelperTests(unittest.TestCase):
    def test_threshold_selection_uses_iou_then_documented_lower_tie_break(self) -> None:
        # Both thresholds yield exactly the same perfect mask. They are equally close
        # to 0.5, so the documented final tie break must choose the lower threshold.
        threshold, selected, rows = _threshold_selection(
            np.array([0.2, 0.8], dtype=np.float32),
            np.array([0, 1], dtype=np.uint8),
            [0.6, 0.4],
        )
        self.assertEqual(threshold, 0.4)
        self.assertEqual(selected["iou"], 1.0)
        self.assertEqual([row["threshold"] for row in rows], [0.6, 0.4])

    def test_threshold_selection_rejects_an_empty_grid(self) -> None:
        with self.assertRaises(DevelopmentContractError):
            _threshold_selection(np.array([0.5]), np.array([1], dtype=np.uint8), [])

    def test_global_sample_is_deterministic_aligned_and_exact(self) -> None:
        features = np.arange(120, dtype=np.float32).reshape(30, 4)
        labels = np.array([0] * 20 + [1] * 10, dtype=np.uint8)
        first = select_global_rows(
            features, labels, count=10, seed=42, namespace="baseline", balanced=True
        )
        second = select_global_rows(
            features, labels, count=10, seed=42, namespace="baseline", balanced=True
        )
        selected_features, selected_labels, manifest = first
        np.testing.assert_array_equal(selected_features, second[0])
        np.testing.assert_array_equal(selected_labels, second[1])
        np.testing.assert_array_equal(selected_features[:, 0] / 4, manifest["pixel_indices"])
        self.assertEqual(selected_labels.shape, (10,))
        self.assertEqual(int((selected_labels == 0).sum()), 5)
        self.assertEqual(int((selected_labels == 1).sum()), 5)
        self.assertEqual(manifest["selected_count"], 10)

    def test_feature_schema_mismatch_stops_before_dataset_access(self) -> None:
        with self.assertRaises(DevelopmentContractError):
            _validate_feature_schema({"features": {"student": ["wrong"], "teacher": []}})

    def test_mapping_streams_all_and_only_labelled_masks(self) -> None:
        dataset = _FakeDataset()
        mapping = mapping_from_all_labelled_images(
            _AlternatingClusterModel(), dataset, "student", chunk_size=2
        )
        np.testing.assert_array_equal(mapping.cluster_to_class, [0, 1])
        np.testing.assert_array_equal(mapping.support, [4, 4])
        self.assertEqual(
            dataset.mask_calls,
            [
                ("labelled_a.jpg", "labelled", "cluster_mapping"),
                ("labelled_b.jpg", "labelled", "cluster_mapping"),
            ],
        )

    def test_fixed_fit_budgets_match_capacity_contract(self) -> None:
        config = {
            "sampling": {
                "teacher_knn_reference_pixels": 30_000,
                "student_knn_reference_pixels": 5_000,
                "labelled_pixels_per_image": 1_250,
                "student_fit_pixels": 20_000,
            }
        }
        self.assertEqual(fit_budget_for_family(config, "knn", "teacher"), 30_000)
        self.assertEqual(fit_budget_for_family(config, "knn", "student"), 5_000)
        self.assertEqual(fit_budget_for_family(config, "random_forest", "teacher"), 50_000)
        self.assertEqual(fit_budget_for_family(config, "gaussian_nb", "student"), 20_000)


if __name__ == "__main__":
    unittest.main()
