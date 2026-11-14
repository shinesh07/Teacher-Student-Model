"""Configuration loading and stable experiment identity helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.json"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the version-controlled experiment contract."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def config_digest(config: dict[str, Any]) -> str:
    """Return a stable hash used to bind artifacts to one configuration."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a configuration-relative path inside this repository."""
    candidate = Path(value)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
