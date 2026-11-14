"""Budget-aware sklearn wrappers for classical pixel-wise segmentation models.

The wrappers deliberately keep the raw sklearn estimator visible so that the assignment's
capacity checks and pickle-size measurement operate on the exact fitted object.  They
also centralize feature scaling and chunked inference; no caller can accidentally fit a
KNN scaler on validation or test pixels through this module.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from .compliance import ComplianceError, ModelFamily, ModelRole, assert_model_capacity, model_family


FeatureSet = Literal["teacher", "student"]
LearningTask = Literal["classification", "regression"]


class ModelingError(ValueError):
    """Raised when model construction, fitting, or prediction breaks the contract."""


def _require_role(role: str) -> ModelRole:
    if role not in {"teacher", "student"}:
        raise ModelingError(f"role must be 'teacher' or 'student', received {role!r}")
    return role  # type: ignore[return-value]


def _require_feature_set(feature_set: str) -> FeatureSet:
    if feature_set not in {"teacher", "student"}:
        raise ModelingError(
            f"feature_set must be 'teacher' or 'student', received {feature_set!r}"
        )
    return feature_set  # type: ignore[return-value]


def _require_task(task: str) -> LearningTask:
    if task not in {"classification", "regression"}:
        raise ModelingError(
            f"task must be 'classification' or 'regression', received {task!r}"
        )
    return task  # type: ignore[return-value]


def _validate_threshold(threshold: float) -> float:
    value = float(threshold)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ModelingError(f"threshold must be a finite probability in [0, 1], got {threshold!r}")
    return value


def _validate_feature_matrix(features: Any, *, expected_columns: int | None = None) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ModelingError(
            f"features must be a non-empty two-dimensional matrix, received {values.shape}"
        )
    if expected_columns is not None and values.shape[1] != expected_columns:
        raise ModelingError(
            f"features have {values.shape[1]} columns; model was fitted with {expected_columns}"
        )
    if not np.isfinite(values).all():
        raise ModelingError("features contain NaN or infinity")
    return values


def _validate_targets(targets: Any, rows: int, task: LearningTask) -> np.ndarray:
    values = np.asarray(targets).reshape(-1)
    if values.size != rows:
        raise ModelingError(f"targets have {values.size} rows but features have {rows}")
    if not np.issubdtype(values.dtype, np.number) and values.dtype != np.bool_:
        raise ModelingError("targets must be numeric or boolean")
    values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ModelingError("targets contain NaN or infinity")
    if task == "classification":
        if not np.all(np.isin(values, (0.0, 1.0))):
            raise ModelingError("binary classification targets must contain only 0 and 1")
        return values.astype(np.uint8, copy=False)
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ModelingError("soft-label regression targets must lie in [0, 1]")
    return values.astype(np.float32, copy=False)


def _validate_sample_weight(sample_weight: Any | None, rows: int) -> np.ndarray | None:
    if sample_weight is None:
        return None
    weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if weights.size != rows:
        raise ModelingError(f"sample_weight has {weights.size} rows but features have {rows}")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ModelingError("sample_weight must contain finite non-negative values")
    if not np.any(weights > 0.0):
        raise ModelingError("sample_weight must have at least one positive value")
    return weights


def _validate_capacity_at_construction(
    family: ModelFamily, role: ModelRole, *, n_estimators: int | None, max_depth: int | None
) -> None:
    """Reject static capacity violations before expensive fitting begins."""
    if family != "random_forest":
        return
    if not isinstance(n_estimators, (int, np.integer)) or n_estimators < 1:
        raise ModelingError("Random Forest requires a positive n_estimators")
    if max_depth is not None and (
        not isinstance(max_depth, (int, np.integer)) or max_depth < 1
    ):
        raise ModelingError("max_depth must be a positive integer or None")
    if role == "teacher" and n_estimators < 200:
        raise ModelingError("Random Forest teacher requires at least 200 trees")
    if role == "student":
        if n_estimators > 20:
            raise ModelingError("Random Forest student permits at most 20 trees")
        if max_depth is not None and max_depth > 8:
            raise ModelingError("Random Forest student permits max_depth <= 8")


def make_estimator(
    family: ModelFamily,
    role: ModelRole,
    *,
    task: LearningTask = "classification",
    n_neighbors: int = 15,
    n_estimators: int | None = None,
    max_depth: int | None = None,
    var_smoothing: float = 1e-9,
    random_state: int = 42,
    class_weight: str | dict[int, float] | None = "balanced_subsample",
) -> Any:
    """Build one permitted sklearn estimator with assignment-compliant defaults.

    The dynamic KNN reference count is checked after fitting, because it depends on the
    actual matrix rather than a declared configuration.  Static Random-Forest limits are
    checked here before computation.
    """
    checked_role = _require_role(role)
    checked_task = _require_task(task)
    if family not in {"knn", "random_forest", "gaussian_nb"}:
        raise ModelingError(f"Unsupported supervised family {family!r}")
    if not isinstance(n_neighbors, (int, np.integer)) or n_neighbors < 1:
        raise ModelingError("n_neighbors must be a positive integer")
    if not np.isfinite(var_smoothing) or var_smoothing <= 0:
        raise ModelingError("var_smoothing must be a positive finite value")
    if family == "random_forest" and n_estimators is None:
        n_estimators = 200 if checked_role == "teacher" else 20
    _validate_capacity_at_construction(
        family, checked_role, n_estimators=n_estimators, max_depth=max_depth
    )

    if family == "knn":
        common = {
            "n_neighbors": int(n_neighbors),
            "weights": "distance",
            "algorithm": "kd_tree",
            "leaf_size": 40,
            "p": 2,
            "n_jobs": -1,
        }
        return KNeighborsClassifier(**common) if checked_task == "classification" else KNeighborsRegressor(**common)

    if family == "random_forest":
        # The default was resolved before the capacity check above.
        assert n_estimators is not None
        common = {
            "n_estimators": int(n_estimators),
            "max_depth": max_depth,
            "random_state": int(random_state),
            "n_jobs": -1,
        }
        if checked_task == "classification":
            return RandomForestClassifier(**common, class_weight=class_weight)
        return RandomForestRegressor(**common)

    if checked_task == "regression":
        raise ModelingError("GaussianNB does not support soft-label regression")
    return GaussianNB(var_smoothing=float(var_smoothing))


@dataclass
class FittedPixelModel:
    """Fitted pixel model plus feature-transform and deployment metadata.

    ``estimator`` is intentionally public and exactly the sklearn object on which the
    capacity checker and `pickle.dumps` must operate.  The scaler exists only for KNN,
    as required by the project brief; RF and GaussianNB receive original handcrafted
    feature values.
    """

    estimator: Any
    scaler: StandardScaler | None
    family: ModelFamily
    role: ModelRole
    feature_set: FeatureSet
    threshold: float
    fit_count: int
    task: LearningTask = "classification"
    feature_names: tuple[str, ...] = ()

    @property
    def raw_model(self) -> Any:
        """Return the raw sklearn object used for grading and serialization."""
        return self.estimator

    @property
    def feature_count(self) -> int:
        """Return the actual fitted input width, never a configuration claim."""
        value = getattr(self.estimator, "n_features_in_", None)
        if value is None:
            raise ModelingError("Estimator is not fitted")
        return int(value)

    @property
    def model_size_bytes(self) -> int:
        """Assignment-specified raw-estimator pickle size in bytes."""
        return len(pickle.dumps(self.estimator))

    def _transform(self, features: Any) -> np.ndarray:
        matrix = _validate_feature_matrix(features, expected_columns=self.feature_count)
        return self.scaler.transform(matrix) if self.scaler is not None else matrix

    def predict_scores(self, features: Any, *, chunk_size: int = 16_384) -> np.ndarray:
        """Return flood probabilities/scores in deterministic raster-preserving chunks."""
        if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
            raise ModelingError("chunk_size must be a positive integer")
        matrix = self._transform(features)
        scores: list[np.ndarray] = []
        for start in range(0, matrix.shape[0], int(chunk_size)):
            chunk = matrix[start : start + int(chunk_size)]
            if self.task == "regression":
                prediction = np.asarray(self.estimator.predict(chunk), dtype=np.float32).reshape(-1)
            else:
                probabilities = np.asarray(self.estimator.predict_proba(chunk), dtype=np.float32)
                classes = np.asarray(self.estimator.classes_)
                flood_column = np.flatnonzero(classes == 1)
                prediction = (
                    probabilities[:, int(flood_column[0])]
                    if flood_column.size
                    else np.zeros(chunk.shape[0], dtype=np.float32)
                )
            scores.append(np.clip(prediction, 0.0, 1.0).astype(np.float32, copy=False))
        return np.concatenate(scores)

    def predict_proba(self, features: Any, *, chunk_size: int = 16_384) -> np.ndarray:
        """Return a consistently ordered [P(non-flood), P(flood)] probability matrix."""
        flood_scores = self.predict_scores(features, chunk_size=chunk_size)
        return np.column_stack((1.0 - flood_scores, flood_scores)).astype(np.float32, copy=False)

    def predict(self, features: Any, *, chunk_size: int = 16_384) -> np.ndarray:
        """Threshold chunked scores into the project's binary flood labels."""
        return (self.predict_scores(features, chunk_size=chunk_size) >= self.threshold).astype(np.uint8)

    def capacity_summary(self) -> dict[str, Any]:
        """Run hard limits against the actual fitted estimator and fit count."""
        return assert_model_capacity(
            self.estimator,
            self.role,
            feature_count=self.feature_count,
            recorded_fit_pixels=self.fit_count,
        )


def fit_pixel_model(
    family: ModelFamily,
    role: ModelRole,
    feature_set: FeatureSet,
    features: Any,
    targets: Any,
    *,
    task: LearningTask = "classification",
    threshold: float = 0.5,
    n_neighbors: int = 15,
    n_estimators: int | None = None,
    max_depth: int | None = None,
    var_smoothing: float = 1e-9,
    random_state: int = 42,
    class_weight: str | dict[int, float] | None = "balanced_subsample",
    sample_weight: Any | None = None,
    feature_names: tuple[str, ...] = (),
) -> FittedPixelModel:
    """Fit one permitted supervised pixel model without hidden preprocessing.

    KNN alone receives a `StandardScaler`, which is fit only on the caller-provided
    training matrix.  The caller's data-access layer is responsible for ensuring that
    matrix comes from the allowed split; the wrapper makes its fit count explicit for
    subsequent capacity verification.
    """
    checked_role = _require_role(role)
    checked_features = _require_feature_set(feature_set)
    checked_task = _require_task(task)
    checked_threshold = _validate_threshold(threshold)
    matrix = _validate_feature_matrix(features)
    target_values = _validate_targets(targets, matrix.shape[0], checked_task)
    weights = _validate_sample_weight(sample_weight, matrix.shape[0])
    estimator = make_estimator(
        family,
        checked_role,
        task=checked_task,
        n_neighbors=n_neighbors,
        n_estimators=n_estimators,
        max_depth=max_depth,
        var_smoothing=var_smoothing,
        random_state=random_state,
        class_weight=class_weight,
    )
    resolved_family = model_family(estimator)
    if resolved_family == "knn":
        if weights is not None:
            raise ModelingError(
                "scikit-learn KNN does not accept sample_weight; use soft-label regression instead"
            )
        if n_neighbors > matrix.shape[0]:
            raise ModelingError(
                f"n_neighbors={n_neighbors} exceeds the {matrix.shape[0]} fit rows"
            )
        scaler = StandardScaler()
        fit_matrix = scaler.fit_transform(matrix)
        estimator.fit(fit_matrix, target_values)
    else:
        scaler = None
        if weights is None:
            estimator.fit(matrix, target_values)
        else:
            estimator.fit(matrix, target_values, sample_weight=weights)

    fitted = FittedPixelModel(
        estimator=estimator,
        scaler=scaler,
        family=resolved_family,
        role=checked_role,
        feature_set=checked_features,
        threshold=checked_threshold,
        fit_count=int(matrix.shape[0]),
        task=checked_task,
        feature_names=tuple(feature_names),
    )
    # A hard capacity violation is never allowed to escape as a plausible model.
    try:
        fitted.capacity_summary()
    except ComplianceError as error:
        raise ModelingError(str(error)) from error
    return fitted
