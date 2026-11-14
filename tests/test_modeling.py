from __future__ import annotations

import unittest

import numpy as np

from src.modeling import ModelingError, fit_pixel_model, make_estimator


class ModelingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        generator = np.random.default_rng(42)
        cls.features = generator.normal(size=(240, 6)).astype(np.float32)
        cls.targets = (cls.features[:, 0] + cls.features[:, 1] > 0).astype(np.uint8)

    def test_knn_uses_scaling_kd_tree_and_chunked_binary_probabilities(self) -> None:
        model = fit_pixel_model(
            "knn", "student", "student", self.features, self.targets, n_neighbors=11
        )
        self.assertIsNotNone(model.scaler)
        self.assertEqual(model.raw_model.algorithm, "kd_tree")
        self.assertEqual(model.raw_model.n_jobs, -1)
        one_chunk = model.predict_proba(self.features[:37], chunk_size=37)
        many_chunks = model.predict_proba(self.features[:37], chunk_size=5)
        np.testing.assert_allclose(one_chunk, many_chunks)
        self.assertEqual(one_chunk.shape, (37, 2))
        np.testing.assert_allclose(one_chunk.sum(axis=1), 1.0)
        self.assertTrue(set(model.predict(self.features[:37])).issubset({0, 1}))

    def test_knn_student_capacity_and_teacher_minimum_are_checked_on_actual_rows(self) -> None:
        oversized = np.random.default_rng(1).normal(size=(5_001, 6)).astype(np.float32)
        labels = (oversized[:, 0] > 0).astype(np.uint8)
        with self.assertRaises(ModelingError):
            fit_pixel_model("knn", "student", "student", oversized, labels, n_neighbors=3)
        with self.assertRaises(ModelingError):
            fit_pixel_model("knn", "teacher", "teacher", self.features, self.targets, n_neighbors=3)

    def test_random_forest_static_budgets_fail_before_fit(self) -> None:
        self.assertEqual(make_estimator("random_forest", "teacher").n_estimators, 200)
        self.assertEqual(make_estimator("random_forest", "student").n_estimators, 20)
        with self.assertRaises(ModelingError):
            make_estimator("random_forest", "teacher", n_estimators=199)
        with self.assertRaises(ModelingError):
            make_estimator("random_forest", "student", n_estimators=21, max_depth=8)
        with self.assertRaises(ModelingError):
            make_estimator("random_forest", "student", n_estimators=20, max_depth=9)

    def test_random_forest_and_gnb_keep_unscaled_inputs_and_fit(self) -> None:
        forest = fit_pixel_model(
            "random_forest",
            "student",
            "student",
            self.features,
            self.targets,
            n_estimators=5,
            max_depth=4,
        )
        gnb = fit_pixel_model("gaussian_nb", "student", "student", self.features, self.targets)
        self.assertIsNone(forest.scaler)
        self.assertIsNone(gnb.scaler)
        self.assertEqual(forest.feature_count, 6)
        self.assertGreater(gnb.model_size_bytes, 0)

    def test_weighted_soft_fitting_and_regression_paths(self) -> None:
        scores = self.targets.astype(np.float32) * 0.8 + 0.1
        duplicated_features = np.repeat(self.features[:80], 2, axis=0)
        duplicated_targets = np.tile(np.array([1, 0], dtype=np.uint8), 80)
        duplicated_weights = np.column_stack((scores[:80], 1.0 - scores[:80])).reshape(-1)
        weighted = fit_pixel_model(
            "gaussian_nb",
            "student",
            "student",
            duplicated_features,
            duplicated_targets,
            sample_weight=duplicated_weights,
        )
        regressor = fit_pixel_model(
            "knn",
            "student",
            "student",
            self.features,
            scores,
            task="regression",
            n_neighbors=7,
        )
        self.assertEqual(weighted.task, "classification")
        self.assertEqual(regressor.task, "regression")
        self.assertTrue(np.all((regressor.predict_scores(self.features[:20]) >= 0) & (regressor.predict_scores(self.features[:20]) <= 1)))

    def test_bad_inputs_and_unsupported_weighted_knn_fail_loudly(self) -> None:
        with self.assertRaises(ModelingError):
            fit_pixel_model("knn", "student", "student", self.features, self.targets, sample_weight=np.ones(240))
        with self.assertRaises(ModelingError):
            fit_pixel_model("gaussian_nb", "student", "student", self.features, np.full(240, 2))
        with self.assertRaises(ModelingError):
            fit_pixel_model("gaussian_nb", "student", "student", self.features, self.targets, threshold=1.1)


if __name__ == "__main__":
    unittest.main()
