from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Force test environment before importing app modules
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("DATABASE_URL", "postgresql://synthquanta:synthquanta@localhost:5432/synthquanta_test")
os.environ.setdefault("LOG_LEVEL", "WARNING")


@pytest.fixture(scope="session")
def tmp_artifact_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temporary artifact root for test isolation."""
    return tmp_path_factory.mktemp("artifacts")


@pytest.fixture(scope="session")
def client() -> TestClient:
    from app.main import app
    return TestClient(app, raise_server_exceptions=True)
