from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Segment names must be non-empty alphanumerics, hyphens, or underscores only.
# This prevents path traversal (no dots, slashes, or other separators).
_SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class ArtifactStoreError(Exception):
    """Raised when an artifact operation fails due to invalid input or I/O."""


class ArtifactStore:
    """Filesystem artifact management for SynthQuanta.

    All large/immutable artifacts (datasets, model weights, experiment
    checkpoints, benchmark dumps) live under a configurable root directory.
    PostgreSQL stores only path references and metadata.

    Security invariant: no API-supplied string can escape the artifact root.
    Every path component is validated before joining.
    """

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._ensure_root_dirs()

    # ------------------------------------------------------------------
    # Public path helpers
    # ------------------------------------------------------------------

    def dataset_path(self, dataset_id: str) -> Path:
        return self._safe_subpath("datasets", dataset_id)

    def model_path(self, model_id: str) -> Path:
        return self._safe_subpath("models", model_id)

    def experiment_path(self, experiment_id: str) -> Path:
        return self._safe_subpath("experiments", experiment_id)

    def benchmark_path(self, benchmark_id: str) -> Path:
        return self._safe_subpath("benchmarks", benchmark_id)

    def evaluation_path(self, evaluation_id: str) -> Path:
        return self._safe_subpath("evaluations", evaluation_id)

    # ------------------------------------------------------------------
    # Directory management
    # ------------------------------------------------------------------

    def ensure_dir(self, path: Path) -> Path:
        """Create directory and all parents. Returns the path."""
        path.mkdir(parents=True, exist_ok=True)
        return path

    def exists(self, path: Path) -> bool:
        return path.exists()

    # ------------------------------------------------------------------
    # JSON metadata helpers
    # ------------------------------------------------------------------

    def write_metadata(self, path: Path, data: dict[str, Any]) -> None:
        """Write a JSON metadata file to path/metadata.json."""
        self.ensure_dir(path)
        metadata_file = path / "metadata.json"
        metadata_file.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.debug("Metadata written to %s", metadata_file)

    def read_metadata(self, path: Path) -> dict[str, Any]:
        """Read path/metadata.json and return as dict. Raises if missing."""
        metadata_file = path / "metadata.json"
        if not metadata_file.exists():
            raise ArtifactStoreError(f"Metadata not found: {metadata_file}")
        return json.loads(metadata_file.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Relative path (for storing in database)
    # ------------------------------------------------------------------

    def relative_path(self, absolute: Path) -> str:
        """Return a path relative to the artifact root, for storing in PostgreSQL."""
        return str(absolute.relative_to(self._root))

    @property
    def root(self) -> Path:
        return self._root

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _safe_subpath(self, category: str, *parts: str) -> Path:
        """Build a safe path under root/category/parts.

        Raises ArtifactStoreError if any part would escape the artifact root
        or contains disallowed characters.
        """
        self._validate_segment(category)
        for part in parts:
            self._validate_segment(part)

        candidate = self._root / category
        for part in parts:
            candidate = candidate / part

        # Final guard: resolved path must remain under the artifact root
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self._root)
        except ValueError:
            raise ArtifactStoreError(
                f"Path traversal attempt detected: {candidate!r} escapes artifact root"
            )

        return resolved

    @staticmethod
    def _validate_segment(segment: str) -> None:
        if not segment or not _SAFE_SEGMENT_RE.match(segment):
            raise ArtifactStoreError(
                f"Invalid path segment {segment!r}. "
                "Only alphanumerics, hyphens, and underscores are allowed."
            )

    def _ensure_root_dirs(self) -> None:
        for subdir in ("datasets", "models", "experiments", "benchmarks", "evaluations"):
            (self._root / subdir).mkdir(parents=True, exist_ok=True)
        logger.debug("Artifact root initialized at %s", self._root)
