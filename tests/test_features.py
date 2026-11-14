from __future__ import annotations

import unittest

import numpy as np

from src.features import (
    STUDENT_FEATURE_NAMES,
    TEACHER_FEATURE_NAMES,
    FeatureContractError,
    extract_features,
    extract_with_timing,
    feature_names,
    rgb_to_hsv,
    validate_feature_budget,
)


class FeatureExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = np.random.default_rng(42)
        self.image = generator.random((11, 13, 3), dtype=np.float32)

    def test_declared_schemas_match_required_dimensions(self) -> None:
        self.assertEqual(feature_names("student"), STUDENT_FEATURE_NAMES)
        self.assertEqual(feature_names("teacher"), TEACHER_FEATURE_NAMES)
        self.assertEqual(len(STUDENT_FEATURE_NAMES), 6)
        self.assertEqual(len(TEACHER_FEATURE_NAMES), 20)

    def test_student_shape_raster_order_and_rgb_identity(self) -> None:
        features = extract_features(self.image, "student")
        self.assertEqual(features.shape, (11 * 13, 6))
        np.testing.assert_allclose(features[:, :3], self.image.reshape(-1, 3), rtol=0, atol=0)

    def test_teacher_shape_is_finite_and_deterministic(self) -> None:
        first = extract_features(self.image, "teacher")
        second = extract_features(self.image.copy(), "teacher")
        self.assertEqual(first.shape, (11 * 13, 20))
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_array_equal(first, second)

    def test_hsv_known_colours(self) -> None:
        rgb = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.5]]], dtype=np.float32)
        hsv = rgb_to_hsv(rgb)
        np.testing.assert_allclose(hsv[0, :, 0], [0.0, 1.0 / 3.0, 2.0 / 3.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(hsv[0, :, 1], [1.0, 1.0, 1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(hsv[0, :, 2], [1.0, 1.0, 1.0, 0.5], atol=1e-6)

    def test_position_channels_are_normalised(self) -> None:
        features = extract_features(self.image, "teacher").reshape(11, 13, 20)
        np.testing.assert_allclose(features[:, 0, 18], np.linspace(0.0, 1.0, 11))
        np.testing.assert_allclose(features[0, :, 19], np.linspace(0.0, 1.0, 13))

    def test_timing_metadata_matches_feature_matrix(self) -> None:
        extraction = extract_with_timing(self.image, "teacher")
        self.assertEqual(extraction.values.shape, (143, 20))
        self.assertEqual(extraction.names, TEACHER_FEATURE_NAMES)
        self.assertGreaterEqual(extraction.elapsed_seconds, 0.0)

    def test_budget_is_satisfied_by_actual_extractors(self) -> None:
        student = extract_features(self.image, "student")
        teacher = extract_features(self.image, "teacher")
        validate_feature_budget(student, teacher)

    def test_bad_shape_range_and_nonfinite_values_fail_loudly(self) -> None:
        with self.assertRaises(FeatureContractError):
            extract_features(np.zeros((5, 5), dtype=np.float32), "student")
        with self.assertRaises(FeatureContractError):
            extract_features(np.full((5, 5, 3), 1.1, dtype=np.float32), "student")
        invalid = np.zeros((5, 5, 3), dtype=np.float32)
        invalid[0, 0, 0] = np.nan
        with self.assertRaises(FeatureContractError):
            extract_features(invalid, "student")


if __name__ == "__main__":
    unittest.main()
