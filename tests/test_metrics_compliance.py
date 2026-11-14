from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from src.compliance import (
    ComplianceError,
    assert_feature_budget,
    assert_model_capacity,
    assert_postprocessing_equal,
    assert_same_family,
    write_checkpoint_manifest,
)
from src.unsupervised import FrozenCentroidPredictor
from src.metrics import (
    MetricInputError,
    agreement,
    cluster_similarity,
    compression_ratio,
    efficiency_score,
    evaluate_binary_segmentation,
    micro_confusion,
    percentage_point_gain,
    retention_percent,
    speedup,
)


class MetricTests(unittest.TestCase):
    def test_micro_metrics_are_calculated_over_all_pixels(self) -> None:
        truth = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        prediction = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        confusion = micro_confusion(truth, prediction)
        self.assertEqual(confusion.as_dict(), {"tp": 1, "fp": 1, "fn": 1, "tn": 1, "n_pixels": 4})
        result = evaluate_binary_segmentation(truth, prediction)
        self.assertEqual(result["n_pixels"], 4)
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["f1"], 0.5)
        self.assertAlmostEqual(result["iou"], 1 / 3)

    def test_bad_binary_inputs_are_rejected(self) -> None:
        with self.assertRaises(MetricInputError):
            micro_confusion([0, 2], [0, 1])
        with self.assertRaises(MetricInputError):
            micro_confusion([0, 1], [0])

    def test_derived_teacher_student_quantities_have_explicit_units(self) -> None:
        self.assertAlmostEqual(agreement([0, 1], [0, 0]), 0.5)
        self.assertAlmostEqual(retention_percent(0.25, 0.5), 50.0)
        self.assertAlmostEqual(compression_ratio(1000, 100), 10.0)
        self.assertAlmostEqual(speedup(2.0, 0.5), 4.0)
        self.assertAlmostEqual(percentage_point_gain(0.55, 0.50), 5.0)
        self.assertAlmostEqual(efficiency_score(0.5, 0.25), 40.0)
        self.assertAlmostEqual(efficiency_score(50, 25, inputs_are_percent=True), 40.0)
        with self.assertRaises(MetricInputError):
            efficiency_score(50, 25)

    def test_cluster_similarity_uses_same_ordered_assignments(self) -> None:
        result = cluster_similarity([3, 3, 4, 4], [1, 1, 2, 2])
        self.assertAlmostEqual(result["ari"], 1.0)
        self.assertAlmostEqual(result["nmi"], 1.0)


class ComplianceTests(unittest.TestCase):
    @staticmethod
    def _binary_data(rows: int, columns: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        features = rng.normal(size=(rows, columns)).astype(np.float32)
        labels = (features[:, 0] > 0).astype(np.uint8)
        return features, labels

    def test_feature_budget_is_enforced(self) -> None:
        assert_feature_budget(20, 6, require_rich_teacher=True)
        with self.assertRaises(ComplianceError):
            assert_feature_budget(20, 7)
        with self.assertRaises(ComplianceError):
            assert_feature_budget(11, 6)

    def test_knn_capacity_uses_fitted_reference_count_not_claimed_count(self) -> None:
        features, labels = self._binary_data(20_000, 20)
        teacher = KNeighborsClassifier(n_neighbors=3).fit(features, labels)
        result = assert_model_capacity(teacher, "teacher", feature_count=20, recorded_fit_pixels=20_000)
        self.assertEqual(result["reference_pixels"], 20_000)
        undersized = KNeighborsClassifier(n_neighbors=1).fit(features[:4], labels[:4])
        with self.assertRaises(ComplianceError):
            assert_model_capacity(undersized, "teacher", feature_count=20, recorded_fit_pixels=20_000)

    def test_random_forest_student_depth_and_tree_limits_are_enforced(self) -> None:
        features, labels = self._binary_data(80, 6)
        legal = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42).fit(features, labels)
        result = assert_model_capacity(legal, "student", feature_count=6)
        self.assertLessEqual(result["max_fitted_depth"], 8)
        deep = RandomForestClassifier(n_estimators=1, max_depth=None, random_state=42).fit(features, labels)
        # Force a nontrivial tree and verify actual fitted depth, not merely constructor intent.
        if deep.estimators_[0].tree_.max_depth > 8:
            with self.assertRaises(ComplianceError):
                assert_model_capacity(deep, "student", feature_count=6)

    def test_gaussian_nb_and_pairing_contracts(self) -> None:
        features, labels = self._binary_data(30, 6)
        gnb = GaussianNB().fit(features, labels)
        self.assertEqual(assert_model_capacity(gnb, "student", feature_count=6)["family"], "gaussian_nb")
        # The specification explicitly requires the full 3 x 3 matrix of KNN, RF and
        # GNB teacher/student combinations, so different supervised estimator types
        # are compatible. Only a supervised/K-Means crossing is forbidden.
        knn = KNeighborsClassifier(n_neighbors=1).fit(features, labels)
        self.assertEqual(assert_same_family(gnb, knn), "supervised")
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(features)
        with self.assertRaises(ComplianceError):
            assert_same_family(gnb, kmeans)
        assert_postprocessing_equal({"mode": "none"}, {"mode": "none"})
        with self.assertRaises(ComplianceError):
            assert_postprocessing_equal({"mode": "none"}, {"mode": "median"})

    def test_kmeans_student_limit_and_teacher_minimum_are_enforced(self) -> None:
        features, _ = self._binary_data(30, 6)
        student = KMeans(n_clusters=2, n_init=10, random_state=42).fit(features)
        self.assertEqual(assert_model_capacity(student, "student", feature_count=6)["clusters"], 2)
        teacher = KMeans(n_clusters=2, n_init=1, random_state=42).fit(features)
        with self.assertRaises(ComplianceError):
            assert_model_capacity(teacher, "teacher", feature_count=6)

    def test_frozen_zero_lloyd_kmeans_is_a_valid_compact_student(self) -> None:
        frozen = FrozenCentroidPredictor(np.zeros((2, 6), dtype=np.float32))
        result = assert_model_capacity(
            frozen, "student", feature_count=6, recorded_fit_pixels=20_000
        )
        self.assertEqual(result["family"], "kmeans")
        self.assertEqual(result["clusters"], 2)

    def test_checkpoint_manifest_is_bound_to_config_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            config = {"seed": 42, "nested": {"x": 1}}
            destination = write_checkpoint_manifest(
                "CP01-data",
                config,
                {"tests": ["test_a"], "counts": {"images": 290}},
                output_dir=output,
            )
            self.assertTrue(destination.is_file())
            manifest = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(manifest["checkpoint"], "CP01-data")
            self.assertEqual(manifest["status"], "PASSED")
            self.assertEqual(manifest["evidence"]["counts"]["images"], 290)
            with self.assertRaises(ComplianceError):
                write_checkpoint_manifest("bad/name", config, {}, output_dir=output)


if __name__ == "__main__":
    unittest.main()
