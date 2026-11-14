from __future__ import annotations

import unittest

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from src.unsupervised import (
    ClusterMapping,
    FrozenCentroidPredictor,
    UnsupervisedContractError,
    assignments_to_mask,
    derive_teacher_grouping,
    fit_student_kmeans,
    fit_teacher_kmeans,
    initialize_student_centroids,
    majority_cluster_mapping,
    make_distilled_student,
)


class MappingTests(unittest.TestCase):
    def test_majority_mapping_uses_labelled_votes_and_deterministic_ties(self) -> None:
        mapping = majority_cluster_mapping(
            np.array([0, 0, 1, 1, 2, 2]),
            np.array([0, 1, 1, 1, 0, 1]),
            n_clusters=4,
        )
        # Cluster 0 ties -> non-flood, cluster 1 -> flood, cluster 2 ties -> non-flood,
        # absent cluster 3 falls back to global flood majority.
        np.testing.assert_array_equal(mapping.cluster_to_class, [0, 1, 0, 1])
        np.testing.assert_array_equal(mapping.support, [2, 2, 2, 0])
        np.testing.assert_array_equal(assignments_to_mask([3, 2, 1, 0], mapping), [1, 0, 1, 0])

    def test_mapping_rejects_non_binary_or_misaligned_labels(self) -> None:
        with self.assertRaises(UnsupervisedContractError):
            majority_cluster_mapping([0, 1], [0, 2])
        with self.assertRaises(UnsupervisedContractError):
            majority_cluster_mapping([0, 1], [0])


class GroupingAndCentroidTests(unittest.TestCase):
    def setUp(self) -> None:
        self.centers = np.array(
            [[0.0, 0.0], [0.2, 0.1], [2.0, 2.0], [2.2, 2.1], [8.0, 8.0], [8.2, 8.1]],
            dtype=np.float32,
        )
        self.mapping = ClusterMapping(
            np.array([0, 0, 1, 1, 1, 1], dtype=np.uint8),
            np.ones(6, dtype=np.int64),
            0,
        )

    def test_two_group_case_is_exactly_teacher_class_mapping(self) -> None:
        grouping = derive_teacher_grouping(self.centers, self.mapping, student_clusters=2)
        np.testing.assert_array_equal(grouping.teacher_cluster_to_student, self.mapping.cluster_to_class)
        np.testing.assert_array_equal(grouping.student_cluster_to_class, [0, 1])
        np.testing.assert_array_equal(grouping.inherited_mapping().cluster_to_class, [0, 1])

    def test_general_grouping_stays_within_semantic_class(self) -> None:
        grouping = derive_teacher_grouping(self.centers, self.mapping, student_clusters=4)
        self.assertEqual(grouping.student_clusters, 4)
        for teacher_cluster, student_cluster in enumerate(grouping.teacher_cluster_to_student):
            self.assertEqual(
                int(self.mapping.cluster_to_class[teacher_cluster]),
                int(grouping.student_cluster_to_class[student_cluster]),
            )

    def test_initial_centroids_are_means_of_teacher_groups(self) -> None:
        grouping = derive_teacher_grouping(self.centers, self.mapping, student_clusters=2)
        features = np.array([[1.0], [3.0], [10.0], [20.0], [30.0]], dtype=np.float32)
        teacher_assignments = np.array([0, 1, 2, 4, 5])
        initial = initialize_student_centroids(features, teacher_assignments, grouping)
        np.testing.assert_allclose(initial, [[2.0], [20.0]])

    def test_missing_teacher_group_fit_pixels_fails(self) -> None:
        grouping = derive_teacher_grouping(self.centers, self.mapping, student_clusters=2)
        with self.assertRaises(UnsupervisedContractError):
            initialize_student_centroids(np.array([[1.0]], dtype=np.float32), np.array([0]), grouping)


class StudentModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = np.array(
            [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [4.9, 5.0], [5.1, 5.0], [5.0, 5.1]],
            dtype=np.float32,
        )
        self.teacher_assignments = np.array([0, 0, 0, 1, 1, 1])
        self.grouping = derive_teacher_grouping(
            np.array([[0.0, 0.0], [5.0, 5.0]], dtype=np.float32),
            ClusterMapping(np.array([0, 1], dtype=np.uint8), np.array([3, 3]), 0),
            student_clusters=2,
        )

    def test_frozen_predictor_uses_nearest_centroid_in_chunks(self) -> None:
        predictor = FrozenCentroidPredictor(np.array([[0.0, 0.0], [5.0, 5.0]], dtype=np.float32))
        np.testing.assert_array_equal(predictor.predict(self.features, chunk_size=2), [0, 0, 0, 1, 1, 1])
        self.assertEqual(predictor.n_features_in_, 2)
        self.assertEqual(predictor.n_clusters, 2)

    def test_no_refinement_and_refinement_variants_are_available(self) -> None:
        frozen, initial = make_distilled_student(
            self.features, self.teacher_assignments, self.grouping, refine=False
        )
        refined, refined_initial = make_distilled_student(
            self.features, self.teacher_assignments, self.grouping, refine=True
        )
        self.assertIsInstance(frozen, FrozenCentroidPredictor)
        self.assertIsInstance(refined, KMeans)
        np.testing.assert_allclose(initial, refined_initial)
        np.testing.assert_array_equal(frozen.predict(self.features), [0, 0, 0, 1, 1, 1])
        self.assertEqual(refined.n_init, 1)

    def test_kmeans_factories_support_teacher_and_baseline_student(self) -> None:
        generator = np.random.default_rng(42)
        six_clouds = np.concatenate(
            [generator.normal(loc=index * 4.0, scale=0.1, size=(12, 2)) for index in range(6)]
        ).astype(np.float32)
        teacher = fit_teacher_kmeans(six_clouds, n_clusters=6, n_init=10, minibatch=True)
        baseline = fit_student_kmeans(self.features, n_clusters=2, init="k-means++")
        self.assertIsInstance(teacher, MiniBatchKMeans)
        self.assertEqual(teacher.n_init, 10)
        self.assertIsInstance(baseline, KMeans)
        self.assertEqual(baseline.n_clusters, 2)

    def test_invalid_student_capacity_or_centroid_shape_fails(self) -> None:
        with self.assertRaises(UnsupervisedContractError):
            fit_student_kmeans(self.features, n_clusters=5)
        with self.assertRaises(UnsupervisedContractError):
            fit_student_kmeans(self.features, n_clusters=2, init=np.zeros((3, 2), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
