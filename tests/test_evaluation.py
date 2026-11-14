from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from src.evaluation import (
    EvaluationError,
    evaluate_model_on_images,
    raw_model_size_bytes,
    segment_image,
    write_qualitative_panel,
)
from src.features import extract_features
from src.modeling import fit_pixel_model
from src.unsupervised import ClusterMapping, FrozenCentroidPredictor


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = np.random.default_rng(42)
        self.image = generator.random((7, 9, 3), dtype=np.float32)
        self.image_two = generator.random((7, 9, 3), dtype=np.float32)
        self.mask = (self.image[..., 0] > 0.5).astype(np.uint8)
        self.mask_two = (self.image_two[..., 0] > 0.5).astype(np.uint8)
        training_images = generator.random((8, 6, 3), dtype=np.float32)
        training_features = extract_features(training_images, "student")
        targets = (training_features[:, 0] > 0.5).astype(np.uint8)
        self.supervised = fit_pixel_model(
            "gaussian_nb",
            "student",
            "student",
            training_features,
            targets,
            feature_names=("red", "green", "blue", "hue", "saturation", "value"),
        )

    def test_supervised_segmentation_measures_features_and_prediction(self) -> None:
        result = segment_image(self.image, self.supervised, "student", chunk_size=11)
        self.assertEqual(result.prediction.shape, self.mask.shape)
        self.assertEqual(result.scores.shape, self.mask.shape)
        self.assertIsNone(result.raw_assignments)
        self.assertEqual(result.feature_dimensions, 6)
        self.assertGreaterEqual(result.feature_seconds, 0.0)
        self.assertGreaterEqual(result.segmentation_seconds, result.feature_seconds)
        self.assertTrue(set(np.unique(result.prediction)).issubset({0, 1}))
        self.assertGreater(raw_model_size_bytes(self.supervised), 0)

    def test_micro_aggregation_uses_every_pixel_not_per_image_average(self) -> None:
        result = evaluate_model_on_images(
            [self.image, self.image_two],
            [self.mask, self.mask_two],
            self.supervised,
            "student",
            chunk_size=13,
        )
        self.assertEqual(result.image_count, 2)
        self.assertEqual(result.pixels_evaluated, 2 * 7 * 9)
        self.assertEqual(result.metrics["n_pixels"], 2 * 7 * 9)
        self.assertGreater(result.raw_model_size_bytes, 0)
        self.assertEqual(len(result.segments), 2)

    def test_kmeans_and_frozen_centroid_use_supplied_mapping(self) -> None:
        features = extract_features(self.image, "student")
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(features)
        mapping = ClusterMapping(np.array([0, 1], dtype=np.uint8), np.array([1, 1]), 0)
        clustered = segment_image(self.image, kmeans, "student", mapping=mapping, chunk_size=17)
        frozen = FrozenCentroidPredictor(kmeans.cluster_centers_)
        frozen_result = segment_image(self.image, frozen, "student", mapping=mapping, chunk_size=17)
        self.assertEqual(clustered.raw_assignments.shape, self.mask.shape)
        self.assertIsNone(clustered.scores)
        self.assertEqual(frozen_result.prediction.shape, self.mask.shape)
        self.assertTrue(set(np.unique(clustered.prediction)).issubset({0, 1}))
        self.assertGreater(raw_model_size_bytes(kmeans), 0)

    def test_missing_or_wrong_mapping_and_shape_fail_loudly(self) -> None:
        features = extract_features(self.image, "student")
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(features)
        with self.assertRaises(EvaluationError):
            segment_image(self.image, kmeans, "student")
        with self.assertRaises(EvaluationError):
            evaluate_model_on_images([self.image], [np.zeros((2, 2), dtype=np.uint8)], self.supervised, "student")
        with self.assertRaises(EvaluationError):
            segment_image(self.image, self.supervised, "teacher")

    def test_qualitative_panel_is_written_with_sorted_prediction_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = write_qualitative_panel(
                self.image,
                self.mask,
                {"z-model": self.mask_two, "a-model": self.mask},
                Path(temporary) / "panel.png",
                title="Controlled qualitative comparison",
            )
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
