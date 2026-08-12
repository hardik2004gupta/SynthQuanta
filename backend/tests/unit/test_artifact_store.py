from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.artifact_store import ArtifactStore, ArtifactStoreError


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(root=tmp_path)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def test_store_creates_subdirectories(tmp_path: Path) -> None:
    ArtifactStore(root=tmp_path)
    for subdir in ("datasets", "models", "experiments", "benchmarks"):
        assert (tmp_path / subdir).is_dir()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def test_dataset_path_is_under_root(store: ArtifactStore, tmp_path: Path) -> None:
    path = store.dataset_path("DS-0001")
    assert str(path).startswith(str(tmp_path))
    assert "datasets" in str(path)
    assert "DS-0001" in str(path)


def test_model_path_is_under_root(store: ArtifactStore, tmp_path: Path) -> None:
    path = store.model_path("MODEL-0001")
    assert "models" in str(path)


def test_experiment_path_is_under_root(store: ArtifactStore, tmp_path: Path) -> None:
    path = store.experiment_path("EXP-0001")
    assert "experiments" in str(path)


def test_benchmark_path_is_under_root(store: ArtifactStore, tmp_path: Path) -> None:
    path = store.benchmark_path("BENCH-0001")
    assert "benchmarks" in str(path)


# ---------------------------------------------------------------------------
# Path traversal protection (security requirement from CLAUDE.md §18)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_id", [
    "../evil",
    "../../etc/passwd",
    "/absolute/path",
    "valid/../evil",
    "has spaces",
    "has.dots",
    "",
])
def test_rejects_path_traversal(store: ArtifactStore, bad_id: str) -> None:
    with pytest.raises(ArtifactStoreError):
        store.dataset_path(bad_id)


def test_rejects_slashes_in_segment(store: ArtifactStore) -> None:
    with pytest.raises(ArtifactStoreError):
        store.dataset_path("DS-0001/evil")


# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------

def test_ensure_dir_creates_nested(store: ArtifactStore, tmp_path: Path) -> None:
    target = tmp_path / "new" / "nested" / "dir"
    result = store.ensure_dir(target)
    assert result.is_dir()
    assert result == target


# ---------------------------------------------------------------------------
# Metadata read/write
# ---------------------------------------------------------------------------

def test_write_and_read_metadata(store: ArtifactStore, tmp_path: Path) -> None:
    artifact_dir = tmp_path / "datasets" / "DS-test"
    data = {"seed": 42, "sample_count": 1000, "status": "COMPLETED"}
    store.write_metadata(artifact_dir, data)

    result = store.read_metadata(artifact_dir)
    assert result["seed"] == 42
    assert result["sample_count"] == 1000


def test_read_metadata_raises_if_missing(store: ArtifactStore, tmp_path: Path) -> None:
    with pytest.raises(ArtifactStoreError):
        store.read_metadata(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# Relative path helper
# ---------------------------------------------------------------------------

def test_relative_path(store: ArtifactStore, tmp_path: Path) -> None:
    absolute = store.dataset_path("DS-0001")
    rel = store.relative_path(absolute)
    assert not Path(rel).is_absolute()
    assert "DS-0001" in rel
