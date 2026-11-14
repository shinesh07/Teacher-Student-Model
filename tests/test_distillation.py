from __future__ import annotations

import unittest

import numpy as np

from src.distillation import (
    DistillationError,
    hard_pseudo_labels,
    select_pseudo_training_set,
    tune_decision_threshold,
    weighted_soft_duplication,
)


class PseudoLabelTests(unittest.TestCase):
    def test_exact_hard_and_confidence_boundaries_are_inclusive(self) -> None:
        pseudo = hard_pseudo_labels([0.0, 0.49, 0.5, 0.51, 1.0])
        np.testing.assert_array_equal(pseudo.hard_labels, [0, 0, 1, 1, 1])
        np.testing.assert_allclose(pseudo.confidence, [1.0, 0.51, 0.5, 0.51, 1.0])
        np.testing.assert_array_equal(pseudo.retained_indices(0.5), [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(pseudo.retained_indices(0.51), [0, 1, 3, 4])

    def test_invalid_probabilities_and_tau_fail_loudly(self) -> None:
        with self.assertRaises(DistillationError):
            hard_pseudo_labels([-0.01, 0.5])
        with self.assertRaises(DistillationError):
            hard_pseudo_labels([0.5]).retained_indices(0.49)


class TransferSamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = np.arange(400, dtype=np.float32).reshape(100, 4)
        self.scores = np.concatenate(
            (np.linspace(0.01, 0.2, 50), np.linspace(0.8, 0.99, 50))
        ).astype(np.float32)

    def test_fixed_capacity_balanced_selection_is_deterministic_and_manifestable(self) -> None:
        first = select_pseudo_training_set(
            self.features, self.scores, tau=0.8, capacity=20, base_seed=42
        )
        second = select_pseudo_training_set(
            self.features, self.scores, tau=0.8, capacity=20, base_seed=42
        )
        self.assertEqual(first.selected_count, 20)
        self.assertEqual(first.retained_count, 100)
        np.testing.assert_array_equal(first.source_indices, second.source_indices)
        self.assertEqual(set(first.labels.tolist()), {0, 1})
        self.assertEqual(first.as_manifest()["student_fit_rows"], 20)
        self.assertAlmostEqual(float(first.as_manifest()["confidence_retained_fraction"]), 1.0)

    def test_retained_fraction_uses_the_original_transfer_pool_not_filtered_candidates(self) -> None:
        # At tau 0.95, only the extreme-confidence tails survive. The denominator must
        # remain all 100 teacher-annotated transfer pixels.
        result = select_pseudo_training_set(self.features, self.scores, tau=0.95, capacity=10)
        self.assertEqual(result.as_manifest()["transfer_pool_rows"], 100)
        self.assertAlmostEqual(
            float(result.as_manifest()["confidence_retained_fraction"]), result.retained_count / 100
        )
        self.assertLess(float(result.as_manifest()["confidence_retained_fraction"]), 1.0)

    def test_insufficient_or_one_class_transfer_sets_are_rejected(self) -> None:
        with self.assertRaises(DistillationError):
            select_pseudo_training_set(self.features, self.scores, tau=0.99, capacity=3)
        with self.assertRaises(DistillationError):
            select_pseudo_training_set(self.features, np.full(100, 0.99), tau=0.9, capacity=20)

    def test_feature_score_alignment_is_enforced(self) -> None:
        with self.assertRaises(DistillationError):
            select_pseudo_training_set(self.features[:10], self.scores, tau=0.5, capacity=2)


class SoftAndThresholdTests(unittest.TestCase):
    def test_weighted_duplication_has_exact_two_targets_and_per_pixel_unit_weight(self) -> None:
        features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        batch = weighted_soft_duplication(features, [0.2, 0.8])
        self.assertEqual(batch.features.shape, (4, 2))
        np.testing.assert_array_equal(batch.targets, [0, 0, 1, 1])
        np.testing.assert_allclose(batch.sample_weight, [0.8, 0.2, 0.2, 0.8])
        np.testing.assert_allclose(batch.sample_weight[:2] + batch.sample_weight[2:], [1.0, 1.0])

    def test_validation_threshold_study_uses_iou_then_explicit_ties(self) -> None:
        study = tune_decision_threshold(
            [0.1, 0.4, 0.6, 0.9], [0, 0, 1, 1], [0.3, 0.5, 0.7]
        )
        self.assertEqual(study.selected_threshold, 0.5)
        self.assertEqual(float(study.selected_metrics["iou"]), 1.0)
        self.assertEqual(len(study.rows), 3)
        self.assertEqual(study.as_manifest()["selection_metric"], "validation_iou_micro")

    def test_validation_targets_are_checked(self) -> None:
        with self.assertRaises(DistillationError):
            tune_decision_threshold([0.5], [2], [0.5])
        with self.assertRaises(DistillationError):
            tune_decision_threshold([0.5], [1], [])


if __name__ == "__main__":
    unittest.main()
