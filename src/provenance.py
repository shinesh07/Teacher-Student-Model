"""Small, append-only evidence utilities used by every pipeline checkpoint.

The project guidelines ask for decisions to be justified, recorded and verified.  These
helpers make that rule executable: each stage writes its inputs, configuration hash,
software context and checks into a JSON artifact rather than relying on notebook state.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    """Return a sortable, timezone-explicit timestamp."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """Hash one file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_digest(value: Any) -> str:
    """Hash JSON-compatible evidence with deterministic serialization."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def runtime_context(packages: Iterable[str] = ()) -> dict[str, Any]:
    """Capture enough environment detail for a mentor to reproduce an experiment."""
    resolved_packages: dict[str, str] = {}
    for name in packages:
        try:
            resolved_packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            resolved_packages[name] = "not-installed"
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": resolved_packages,
    }


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write a complete JSON document atomically, never a half-written checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as temporary:
        temporary.write(serialized)
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


class EventLogger:
    """Append structured data-use evidence to a JSONL file.

    Events intentionally contain image identifiers and mask-purpose metadata but no
    masks, image pixels or labels. That makes leakage review possible without copying
    the data into experiment logs.
    """

    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, phase: str, operation: str, **details: Any) -> dict[str, Any]:
        event = {
            "timestamp": utc_now(),
            "run_id": self.run_id,
            "phase": phase,
            "operation": operation,
            **details,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")
        return event
