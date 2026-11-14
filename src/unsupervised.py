"""Leak-safe building blocks for the K-Means teacher-student track.

This module deliberately works only with arrays already produced by the orchestrator.
It never opens images or masks: the caller is therefore responsible for passing masks
only from the labelled partition when constructing a cluster-to-class mapping.

The no-refinement student is represented by :class:`FrozenCentroidPredictor`. It is the
exact nearest-centroid partition implied by the teacher-derived compact centroids,
before any Lloyd iterations can move it. The refinement variant is a regular sklearn
``KMeans`` initialized from those same centroids with ``n_init=1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


KMeansEstimator = KMeans | MiniBatchKMeans


class UnsupervisedContractError(ValueError):
    """Raised when cluster, feature, or label arrays violate the track contract."""


def _as_feature_matrix(values: Any, *, name: str) -> np.ndarray:
    """Validate and return a finite two-dimensional float matrix."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise UnsupervisedContractError(
            f"{name} must be a non-empty 2D feature matrix; got shape {array.shape}"
        )
    if not np.isfinite(array).all():
        raise UnsupervisedContractError(f"{name} contains NaN or infinity")
    return array


def _as_cluster_vector(values: Any, *, name: str, n_clusters: int | None = None) -> np.ndarray:
    """Validate cluster IDs and optionally require them to lie in a known range."""
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise UnsupervisedContractError(f"{name} must be a non-empty one-dimensional vector")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.issubdtype(array.dtype, np.number) or not np.all(np.equal(array, np.floor(array))):
            raise UnsupervisedContractError(f"{name} must contain integer cluster IDs")
    array = array.astype(np.int64, copy=False)
    if np.any(array < 0):
        raise UnsupervisedContractError(f"{name} cannot contain negative cluster IDs")
    if n_clusters is not None and np.any(array >= n_clusters):
        raise UnsupervisedContractError(
            f"{name} contains an ID outside [0, {n_clusters - 1}]"
        )
    return array


def _as_binary_labels(values: Any, *, name: str) -> np.ndarray:
    """Validate binary ground-truth labels used only for permitted mapping steps."""
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise UnsupervisedContractError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isin(array, (0, 1, False, True))):
        raise UnsupervisedContractError(f"{name} must contain only binary labels 0/1")
    return array.astype(np.uint8, copy=False)


@dataclass(frozen=True)
class ClusterMapping:
    """An immutable mapping from numbered clusters to the flood/non-flood class.

    ``support`` records labelled-pixel support per cluster. It lets the report describe
    clusters that did not receive a labelled pixel and therefore fell back to the
    globally most common labelled class.
    """

    cluster_to_class: np.ndarray
    support: np.ndarray
    fallback_class: int

    def __post_init__(self) -> None:
        mapping = _as_binary_labels(self.cluster_to_class, name="cluster_to_class")
        support = np.asarray(self.support)
        if support.ndim != 1 or support.shape != mapping.shape or np.any(support < 0):
            raise UnsupervisedContractError("support must be a non-negative vector matching cluster_to_class")
        if self.fallback_class not in (0, 1):
            raise UnsupervisedContractError("fallback_class must be 0 or 1")
        object.__setattr__(self, "cluster_to_class", mapping.copy())
        object.__setattr__(self, "support", support.astype(np.int64, copy=True))

    @property
    def n_clusters(self) -> int:
        return int(self.cluster_to_class.size)

    def apply(self, assignments: Any) -> np.ndarray:
        """Convert raw cluster assignments to a binary mask vector."""
        clusters = _as_cluster_vector(assignments, name="assignments", n_clusters=self.n_clusters)
        return self.cluster_to_class[clusters].astype(np.uint8, copy=False)


def majority_cluster_mapping(
    labelled_assignments: Any,
    labelled_targets: Any,
    *,
    n_clusters: int | None = None,
) -> ClusterMapping:
    """Map clusters to flood labels by labelled-pixel majority vote only.

    Ties deliberately map to non-flood (0), which is deterministic and avoids inventing
    a positive label. A cluster with no labelled support receives the global labelled
    majority, again with a 0 tie break. The caller must ensure both inputs were obtained
    only from the labelled set ``L``.
    """
    assignments = _as_cluster_vector(labelled_assignments, name="labelled_assignments")
    targets = _as_binary_labels(labelled_targets, name="labelled_targets")
    if assignments.shape != targets.shape:
        raise UnsupervisedContractError("labelled_assignments and labelled_targets must have equal length")
    inferred_clusters = int(assignments.max()) + 1
    count = inferred_clusters if n_clusters is None else int(n_clusters)
    if count < inferred_clusters or count <= 0:
        raise UnsupervisedContractError(
            f"n_clusters={count} cannot represent maximum assignment ID {inferred_clusters - 1}"
        )
    support = np.bincount(assignments, minlength=count).astype(np.int64)
    positive = np.bincount(assignments, weights=targets, minlength=count).astype(np.int64)
    negative = support - positive
    global_fallback = int(targets.sum() > (targets.size - targets.sum()))
    mapping = np.full(count, global_fallback, dtype=np.uint8)
    represented = support > 0
    mapping[represented] = (positive[represented] > negative[represented]).astype(np.uint8)
    return ClusterMapping(mapping, support, global_fallback)


@dataclass(frozen=True)
class ClusterGrouping:
    """Teacher-cluster merging used to create a smaller student partition."""

    teacher_cluster_to_student: np.ndarray
    student_cluster_to_class: np.ndarray

    def __post_init__(self) -> None:
        teacher_to_student = _as_cluster_vector(
            self.teacher_cluster_to_student, name="teacher_cluster_to_student"
        )
        student_to_class = _as_binary_labels(
            self.student_cluster_to_class, name="student_cluster_to_class"
        )
        if teacher_to_student.max() >= student_to_class.size:
            raise UnsupervisedContractError(
                "teacher_cluster_to_student contains a group missing from student_cluster_to_class"
            )
        if set(teacher_to_student.tolist()) != set(range(student_to_class.size)):
            raise UnsupervisedContractError("Every numbered student group must receive a teacher cluster")
        object.__setattr__(self, "teacher_cluster_to_student", teacher_to_student.copy())
        object.__setattr__(self, "student_cluster_to_class", student_to_class.copy())

    @property
    def teacher_clusters(self) -> int:
        return int(self.teacher_cluster_to_student.size)

    @property
    def student_clusters(self) -> int:
        return int(self.student_cluster_to_class.size)

    def groups_for(self, teacher_assignments: Any) -> np.ndarray:
        """Return the compact student-group assignment for teacher-assigned pixels."""
        assignments = _as_cluster_vector(
            teacher_assignments, name="teacher_assignments", n_clusters=self.teacher_clusters
        )
        return self.teacher_cluster_to_student[assignments]

    def inherited_mapping(self) -> ClusterMapping:
        """Return the teacher-derived mapping inherited without reading student labels."""
        support = np.bincount(
            self.teacher_cluster_to_student, minlength=self.student_clusters
        ).astype(np.int64)
        fallback = int(self.student_cluster_to_class.sum() > (self.student_clusters / 2))
        return ClusterMapping(self.student_cluster_to_class, support, fallback)


def _allocate_groups_per_class(cluster_classes: np.ndarray, student_clusters: int) -> dict[int, int]:
    """Allocate a requested compact cluster count across the two semantic classes."""
    available = {label: int(np.count_nonzero(cluster_classes == label)) for label in (0, 1)}
    if not all(available.values()):
        raise UnsupervisedContractError(
            "Teacher cluster mapping must contain both flood and non-flood clusters to build a binary student"
        )
    if student_clusters < 2:
        raise UnsupervisedContractError("student_clusters must be at least 2 for binary segmentation")
    if student_clusters > sum(available.values()):
        raise UnsupervisedContractError(
            "student_clusters cannot exceed the number of available teacher clusters"
        )
    allocation = {0: 1, 1: 1}
    while sum(allocation.values()) < student_clusters:
        candidates = [label for label in (0, 1) if allocation[label] < available[label]]
        # Give the next group to the semantically broader currently least-compressed
        # class; tuple ordering makes ties deterministic (class 0 first).
        chosen = max(candidates, key=lambda label: (available[label] / allocation[label], -label))
        allocation[chosen] += 1
    return allocation


def derive_teacher_grouping(
    teacher_centers: Any,
    teacher_mapping: ClusterMapping,
    *,
    student_clusters: int = 2,
    random_state: int = 42,
) -> ClusterGrouping:
    """Merge rich teacher clusters into at most four compact student groups.

    For the required two-cluster case, the grouping is precisely ``g = pi_T``: all
    flood teacher clusters merge into group 1 and all others into group 0. For three or
    four student groups, teacher centers are partitioned *within their inherited class*
    using deterministic K-Means. Thus a compact group can never mix teacher flood and
    non-flood semantics before optional refinement.
    """
    centers = _as_feature_matrix(teacher_centers, name="teacher_centers")
    if centers.shape[0] != teacher_mapping.n_clusters:
        raise UnsupervisedContractError(
            "teacher_centers row count must equal the cluster mapping length"
        )
    if student_clusters > 4:
        raise UnsupervisedContractError("Student K-Means capacity is at most 4 clusters")
    allocation = _allocate_groups_per_class(teacher_mapping.cluster_to_class, int(student_clusters))
    group_for_teacher = np.empty(teacher_mapping.n_clusters, dtype=np.int64)
    group_classes: list[int] = []
    next_group = 0
    for semantic_class in (0, 1):
        cluster_indices = np.flatnonzero(teacher_mapping.cluster_to_class == semantic_class)
        local_groups = allocation[semantic_class]
        if local_groups == 1:
            group_for_teacher[cluster_indices] = next_group
            group_classes.append(semantic_class)
            next_group += 1
            continue
        # This is a grouping operation on at most K_T centroids, not a second pixel
        # learner. The result is stable under the fixed seed and n_init=10.
        local = KMeans(n_clusters=local_groups, n_init=10, random_state=random_state)
        local_labels = local.fit_predict(centers[cluster_indices])
        group_for_teacher[cluster_indices] = local_labels + next_group
        group_classes.extend([semantic_class] * local_groups)
        next_group += local_groups
    return ClusterGrouping(group_for_teacher, np.asarray(group_classes, dtype=np.uint8))


def initialize_student_centroids(
    compact_features: Any,
    teacher_assignments: Any,
    grouping: ClusterGrouping,
) -> np.ndarray:
    """Compute compact centroids from teacher-grouped pixels exactly as specified."""
    features = _as_feature_matrix(compact_features, name="compact_features")
    assignments = _as_cluster_vector(
        teacher_assignments, name="teacher_assignments", n_clusters=grouping.teacher_clusters
    )
    if features.shape[0] != assignments.size:
        raise UnsupervisedContractError(
            "compact_features and teacher_assignments must represent the same pixels"
        )
    groups = grouping.groups_for(assignments)
    centroids = np.empty((grouping.student_clusters, features.shape[1]), dtype=np.float32)
    for group in range(grouping.student_clusters):
        selected = features[groups == group]
        if selected.size == 0:
            raise UnsupervisedContractError(
                f"Teacher grouping leaves student group {group} without any fit pixels"
            )
        centroids[group] = selected.mean(axis=0, dtype=np.float64)
    if not np.isfinite(centroids).all():
        raise UnsupervisedContractError("Student centroid initialization contains non-finite values")
    return centroids


@dataclass(frozen=True)
class FrozenCentroidPredictor:
    """A no-Lloyd-iteration compact K-Means partition with fixed centroids."""

    cluster_centers_: np.ndarray
    # This explicit marker lets the capacity checker recognize the assignment's
    # required zero-Lloyd K-Means variant without pretending it is a different
    # classifier family. It is intentionally valid only as a compact student.
    is_frozen_kmeans: bool = True

    def __post_init__(self) -> None:
        centers = _as_feature_matrix(self.cluster_centers_, name="cluster_centers_")
        object.__setattr__(self, "cluster_centers_", centers.copy())

    @property
    def n_clusters(self) -> int:
        return int(self.cluster_centers_.shape[0])

    @property
    def n_features_in_(self) -> int:
        return int(self.cluster_centers_.shape[1])

    def predict(self, values: Any, *, chunk_size: int = 65_536) -> np.ndarray:
        """Assign each row to its nearest frozen centroid using bounded memory."""
        features = _as_feature_matrix(values, name="values")
        if features.shape[1] != self.n_features_in_:
            raise UnsupervisedContractError(
                f"Expected {self.n_features_in_} compact features; received {features.shape[1]}"
            )
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise UnsupervisedContractError("chunk_size must be a positive integer")
        result = np.empty(features.shape[0], dtype=np.int64)
        center_norms = np.einsum("ij,ij->i", self.cluster_centers_, self.cluster_centers_)
        for start in range(0, features.shape[0], chunk_size):
            stop = min(start + chunk_size, features.shape[0])
            block = features[start:stop]
            # ||x-c||² = ||x||² + ||c||² - 2x.c is markedly smaller than broadcasting
            # a (chunk, cluster, feature) tensor for full 256 x 256 images.
            distances = (
                np.einsum("ij,ij->i", block, block)[:, None]
                + center_norms[None, :]
                - 2.0 * block @ self.cluster_centers_.T
            )
            result[start:stop] = np.argmin(distances, axis=1)
        return result


def fit_teacher_kmeans(
    rich_features: Any,
    *,
    n_clusters: int,
    n_init: int = 10,
    random_state: int = 42,
    minibatch: bool = True,
    batch_size: int = 4096,
) -> KMeansEstimator:
    """Fit a compliant rich-feature teacher on preselected L union U image pixels."""
    features = _as_feature_matrix(rich_features, name="rich_features")
    if n_clusters < 6:
        raise UnsupervisedContractError("Teacher K-Means needs at least 6 clusters")
    if n_clusters > features.shape[0]:
        raise UnsupervisedContractError("n_clusters cannot exceed the number of fit pixels")
    if n_init < 10:
        raise UnsupervisedContractError("Teacher K-Means needs n_init >= 10")
    if minibatch:
        model: KMeansEstimator = MiniBatchKMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            random_state=random_state,
            batch_size=min(batch_size, features.shape[0]),
            reassignment_ratio=0.0,
        )
    else:
        model = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
    model.fit(features)
    return model


def fit_student_kmeans(
    compact_features: Any,
    *,
    n_clusters: int,
    init: Literal["k-means++"] | np.ndarray = "k-means++",
    random_state: int = 42,
    n_init: int | None = None,
    max_iter: int = 300,
) -> KMeans:
    """Fit a compact K-Means student, preserving supplied-centroid order when given."""
    features = _as_feature_matrix(compact_features, name="compact_features")
    if n_clusters < 1 or n_clusters > 4:
        raise UnsupervisedContractError("Student K-Means must use between 1 and 4 clusters")
    if n_clusters > features.shape[0]:
        raise UnsupervisedContractError("n_clusters cannot exceed the number of fit pixels")
    if isinstance(init, str):
        if init != "k-means++":
            raise UnsupervisedContractError("Student initialization must be 'k-means++' or an array")
        resolved_n_init = 10 if n_init is None else int(n_init)
    else:
        centers = _as_feature_matrix(init, name="init")
        if centers.shape != (n_clusters, features.shape[1]):
            raise UnsupervisedContractError(
                "Supplied compact centroids must have shape "
                f"({n_clusters}, {features.shape[1]}), got {centers.shape}"
            )
        init = centers
        resolved_n_init = 1 if n_init is None else int(n_init)
        if resolved_n_init != 1:
            raise UnsupervisedContractError("Teacher-derived student initialization requires n_init=1")
    if resolved_n_init <= 0 or max_iter <= 0:
        raise UnsupervisedContractError("n_init and max_iter must both be positive")
    model = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=resolved_n_init,
        random_state=random_state,
        max_iter=max_iter,
    )
    model.fit(features)
    return model


def make_distilled_student(
    compact_features: Any,
    teacher_assignments: Any,
    grouping: ClusterGrouping,
    *,
    refine: bool,
    random_state: int = 42,
) -> tuple[FrozenCentroidPredictor | KMeans, np.ndarray]:
    """Construct either no-refinement or Lloyd-refined compact student and its init.

    The returned centroid matrix is retained for reproducibility and makes it possible
    to demonstrate exactly how a refined model drifted from the teacher partition.
    """
    features = _as_feature_matrix(compact_features, name="compact_features")
    initial_centers = initialize_student_centroids(features, teacher_assignments, grouping)
    if not refine:
        return FrozenCentroidPredictor(initial_centers), initial_centers
    return (
        fit_student_kmeans(
            features,
            n_clusters=grouping.student_clusters,
            init=initial_centers,
            n_init=1,
            random_state=random_state,
        ),
        initial_centers,
    )


def assignments_to_mask(assignments: Any, mapping: ClusterMapping) -> np.ndarray:
    """Named convenience wrapper clarifying conversion of raw clusters to flood masks."""
    return mapping.apply(assignments)
