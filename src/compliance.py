"""Executable capacity, pairing, and checkpoint checks for the assignment contract."""

from __future__ import annotations

import json
import os
import platform
import re
import sys
import tempfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from .config import PROJECT_ROOT, config_digest


ModelRole = Literal["teacher", "student"]
ModelFamily = Literal["knn", "random_forest", "gaussian_nb", "kmeans"]


class ComplianceError(ValueError):
    """Raised when an artifact violates a hard project constraint."""


def model_family(model: Any) -> ModelFamily:
    """Identify one of the explicitly permitted estimator families.

    This is intentionally instance-based rather than name-based, so an unrelated class
    cannot masquerade as a permitted model in the graded pipeline.
    """
    if isinstance(model, (KNeighborsClassifier, KNeighborsRegressor)):
        return "knn"
    if isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
        return "random_forest"
    if isinstance(model, GaussianNB):
        return "gaussian_nb"
    if isinstance(model, (KMeans, MiniBatchKMeans)) or bool(
        getattr(model, "is_frozen_kmeans", False)
    ):
        return "kmeans"
    raise ComplianceError(
        "Only sklearn KNN, Random Forest, GaussianNB, KMeans, or MiniBatchKMeans "
        f"are permitted; received {type(model).__module__}.{type(model).__name__}."
    )


def _require_role(role: str) -> ModelRole:
    if role not in {"teacher", "student"}:
        raise ComplianceError(f"role must be 'teacher' or 'student', received {role!r}")
    return role  # type: ignore[return-value]


def _fitted_feature_count(model: Any) -> int | None:
    count = getattr(model, "n_features_in_", None)
    return int(count) if count is not None else None


def _fitted_row_count(model: Any) -> int | None:
    """Get a model's actual fit-row count where sklearn exposes one."""
    count = getattr(model, "n_samples_fit_", None)
    if count is not None:
        return int(count)
    labels = getattr(model, "labels_", None)
    if labels is not None:
        return int(np.asarray(labels).size)
    return None


def _observed_fit_pixels(model: Any, recorded_fit_pixels: int | None) -> int | None:
    actual = _fitted_row_count(model)
    if recorded_fit_pixels is not None:
        if not isinstance(recorded_fit_pixels, (int, np.integer)) or recorded_fit_pixels <= 0:
            raise ComplianceError("recorded_fit_pixels must be a positive integer when supplied")
        if actual is not None and actual != int(recorded_fit_pixels):
            raise ComplianceError(
                f"Recorded fit pixels ({recorded_fit_pixels}) disagree with fitted model rows ({actual})"
            )
        return int(recorded_fit_pixels)
    return actual


def assert_feature_budget(
    teacher_feature_count: int,
    student_feature_count: int,
    *,
    require_rich_teacher: bool = False,
) -> None:
    """Enforce the hard compact-student feature budget from the brief.

    The brief recommends (rather than mandates) dT >= 12, hence that part is opt-in.
    """
    if teacher_feature_count <= 0 or student_feature_count <= 0:
        raise ComplianceError("Teacher and student feature counts must both be positive")
    if require_rich_teacher and teacher_feature_count < 12:
        raise ComplianceError("Teacher feature set must contain at least 12 features for this check")
    if student_feature_count > 6:
        raise ComplianceError(f"Student uses {student_feature_count} features; the maximum is 6")
    if student_feature_count * 2 > teacher_feature_count:
        raise ComplianceError(
            "Student feature count must be at most half the teacher feature count: "
            f"dS={student_feature_count}, dT={teacher_feature_count}"
        )


def assert_same_family(teacher: Any, student: Any) -> Literal["supervised", "unsupervised"]:
    """Enforce the brief's *track* compatibility, not an estimator-name restriction.

    The brief explicitly requires all nine KNN/RF/GNB teacher-to-student cells in Part
    B.3.  Thus KNN, Random Forest and GaussianNB are one supervised family for this
    check; only a K-Means model may teach a K-Means model.  Earlier versions of this
    helper incorrectly demanded identical sklearn classes, which would have made the
    mandated 3 x 3 matrix impossible.
    """
    teacher_family, student_family = model_family(teacher), model_family(student)
    teacher_is_unsupervised = teacher_family == "kmeans"
    student_is_unsupervised = student_family == "kmeans"
    if teacher_is_unsupervised != student_is_unsupervised:
        raise ComplianceError(
            "Teacher/student track mismatch: supervised KNN/RF/GNB models cannot be "
            "paired with K-Means."
        )
    return "unsupervised" if teacher_is_unsupervised else "supervised"


def assert_postprocessing_equal(
    teacher_postprocessing: Mapping[str, Any] | None,
    student_postprocessing: Mapping[str, Any] | None,
) -> None:
    """Require a comparison to use byte-for-byte identical postprocessing settings."""
    teacher_payload = json.dumps(teacher_postprocessing or {}, sort_keys=True, separators=(",", ":"))
    student_payload = json.dumps(student_postprocessing or {}, sort_keys=True, separators=(",", ":"))
    if teacher_payload != student_payload:
        raise ComplianceError(
            "Teacher and student must use identical postprocessing for a fair comparison"
        )


def _assert_fitted_dimensions(model: Any, feature_count: int | None) -> None:
    if feature_count is None:
        return
    if not isinstance(feature_count, (int, np.integer)) or feature_count <= 0:
        raise ComplianceError("feature_count must be a positive integer when supplied")
    fitted = _fitted_feature_count(model)
    if fitted is None:
        raise ComplianceError(f"{type(model).__name__} is not fitted (n_features_in_ is absent)")
    if fitted != int(feature_count):
        raise ComplianceError(
            f"Declared feature count ({feature_count}) disagrees with fitted model count ({fitted})"
        )


def assert_model_capacity(
    model: Any,
    role: ModelRole,
    *,
    feature_count: int | None = None,
    recorded_fit_pixels: int | None = None,
) -> dict[str, Any]:
    """Validate one fitted model against its family-specific capacity budget.

    ``recorded_fit_pixels`` is required for models that do not retain an exact training
    row count.  When sklearn exposes an actual count, a mismatch is an error instead of
    trusting a configuration value.
    """
    checked_role = _require_role(role)
    family = model_family(model)
    _assert_fitted_dimensions(model, feature_count)
    observed_fit_pixels = _observed_fit_pixels(model, recorded_fit_pixels)
    summary: dict[str, Any] = {
        "family": family,
        "role": checked_role,
        "model_class": type(model).__name__,
        "feature_count": _fitted_feature_count(model),
        "fit_pixels": observed_fit_pixels,
    }

    if family == "knn":
        if observed_fit_pixels is None:
            raise ComplianceError("KNN capacity requires its fitted reference-pixel count")
        lower, upper = (20_000, None) if checked_role == "teacher" else (None, 5_000)
        if lower is not None and observed_fit_pixels < lower:
            raise ComplianceError(f"KNN teacher needs >= {lower} reference pixels; found {observed_fit_pixels}")
        if upper is not None and observed_fit_pixels > upper:
            raise ComplianceError(f"KNN student needs <= {upper} reference pixels; found {observed_fit_pixels}")
        summary["reference_pixels"] = observed_fit_pixels

    elif family == "random_forest":
        trees = int(getattr(model, "n_estimators", 0))
        fitted_estimators = getattr(model, "estimators_", None)
        if fitted_estimators is None:
            raise ComplianceError("Random Forest must be fitted before capacity can be checked")
        if len(fitted_estimators) != trees:
            raise ComplianceError(
                f"Random Forest declares {trees} trees but fitted {len(fitted_estimators)}"
            )
        depths = [int(tree.tree_.max_depth) for tree in fitted_estimators]
        if checked_role == "teacher" and trees < 200:
            raise ComplianceError(f"Random Forest teacher needs >= 200 trees; found {trees}")
        if checked_role == "student":
            if trees > 20:
                raise ComplianceError(f"Random Forest student needs <= 20 trees; found {trees}")
            if depths and max(depths) > 8:
                raise ComplianceError(
                    f"Random Forest student tree depth must be <= 8; found {max(depths)}"
                )
        summary.update({"trees": trees, "max_fitted_depth": max(depths, default=0)})

    elif family == "gaussian_nb":
        if not hasattr(model, "class_count_"):
            raise ComplianceError("GaussianNB must be fitted before capacity can be checked")
        summary["classes"] = int(np.asarray(model.class_count_).size)

    elif family == "kmeans":
        clusters = int(getattr(model, "n_clusters", 0))
        frozen = bool(getattr(model, "is_frozen_kmeans", False))
        n_init = getattr(model, "n_init", 0 if frozen else None)
        if not hasattr(model, "cluster_centers_"):
            raise ComplianceError("K-Means must be fitted before capacity can be checked")
        if checked_role == "teacher":
            if frozen:
                raise ComplianceError("A zero-Lloyd frozen-centroid model is allowed only as a student")
            if clusters < 6:
                raise ComplianceError(f"K-Means teacher needs >= 6 clusters; found {clusters}")
            if not isinstance(n_init, (int, np.integer)) or int(n_init) < 10:
                raise ComplianceError(f"K-Means teacher needs n_init >= 10; found {n_init!r}")
            if observed_fit_pixels is None or observed_fit_pixels < 100_000:
                raise ComplianceError(
                    "K-Means teacher needs >= 100000 fit pixels; "
                    f"found {observed_fit_pixels!r}"
                )
        else:
            if clusters > 4:
                raise ComplianceError(f"K-Means student needs <= 4 clusters; found {clusters}")
            if observed_fit_pixels is None:
                raise ComplianceError("K-Means student capacity requires a recorded fit-pixel count")
            if observed_fit_pixels > 20_000:
                raise ComplianceError(
                    f"K-Means student needs <= 20000 fit pixels; found {observed_fit_pixels}"
                )
        summary.update({"clusters": clusters, "n_init": n_init})

    return summary


def assert_pair_capacity(
    teacher: Any,
    student: Any,
    *,
    teacher_feature_count: int,
    student_feature_count: int,
    teacher_fit_pixels: int | None = None,
    student_fit_pixels: int | None = None,
    teacher_postprocessing: Mapping[str, Any] | None = None,
    student_postprocessing: Mapping[str, Any] | None = None,
    require_rich_teacher: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run all objective checks needed before reporting a teacher/student comparison."""
    assert_feature_budget(
        teacher_feature_count,
        student_feature_count,
        require_rich_teacher=require_rich_teacher,
    )
    assert_same_family(teacher, student)
    assert_postprocessing_equal(teacher_postprocessing, student_postprocessing)
    return {
        "teacher": assert_model_capacity(
            teacher,
            "teacher",
            feature_count=teacher_feature_count,
            recorded_fit_pixels=teacher_fit_pixels,
        ),
        "student": assert_model_capacity(
            student,
            "student",
            feature_count=student_feature_count,
            recorded_fit_pixels=student_fit_pixels,
        ),
    }


def _package_versions() -> dict[str, str]:
    packages = ("numpy", "scipy", "scikit-learn", "Pillow")
    result: dict[str, str] = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def write_checkpoint_manifest(
    checkpoint: str,
    config: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    status: str = "PASSED",
) -> Path:
    """Atomically save an evidence-bound checkpoint manifest.

    ``evidence`` should contain concrete test output and artifact paths. It is kept
    caller-defined so data, feature, model, and report stages can use the same manifest
    format without losing relevant details.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", checkpoint):
        raise ComplianceError("checkpoint must contain only alphanumerics, dot, underscore, or hyphen")
    if status != "PASSED":
        raise ComplianceError("Only passed gates may create a checkpoint manifest")
    try:
        json.dumps(evidence, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ComplianceError("Checkpoint evidence must be JSON serializable") from exc

    target_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "results" / "checkpoints"
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint": checkpoint,
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_digest(dict(config)),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "evidence": dict(evidence),
    }
    destination = target_dir / f"{checkpoint}.json"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target_dir, prefix=f".{checkpoint}-", suffix=".tmp", delete=False
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_path = Path(handle.name)
    try:
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination
