"""Integration tests for the Benchmark API.

Tests the FastAPI → BenchmarkService → DB chain using in-memory SQLite.
Error cases tested without real artifacts.
Full benchmark lifecycle is in tests/e2e/final/test_final_e2e.py.
"""
from __future__ import annotations

import os
import time
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.session import get_db
from app.main import app

# ---------------------------------------------------------------------------
# SQLite in-memory test DB
# ---------------------------------------------------------------------------

_SQLITE_URL = "sqlite:///./test_benchmark_api.db"
_test_engine = create_engine(_SQLITE_URL, connect_args={"check_same_thread": False})
_TestSession = sessionmaker(autocommit=False, autoflush=False, bind=_test_engine)


@pytest.fixture(scope="module", autouse=True)
def create_tables():
    Base.metadata.create_all(bind=_test_engine)
    yield
    Base.metadata.drop_all(bind=_test_engine)
    _test_engine.dispose()
    for _ in range(3):
        try:
            if os.path.exists("test_benchmark_api.db"):
                os.remove("test_benchmark_api.db")
            break
        except PermissionError:
            time.sleep(0.2)


def _override_get_db():
    db = _TestSession()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="module")
def test_client():
    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.pop(get_db, None)


# ---------------------------------------------------------------------------
# Run endpoint — error cases
# ---------------------------------------------------------------------------

class TestBenchmarkRun:
    def test_run_unknown_model_returns_error(self, test_client):
        resp = test_client.post(
            "/api/v1/benchmarks/run",
            json={"model_id": str(uuid.uuid4()), "iterations": 5, "warmup": 2},
        )
        assert resp.status_code in (422, 404)

    def test_run_unknown_quantization_returns_error(self, test_client):
        resp = test_client.post(
            "/api/v1/benchmarks/run",
            json={"quantization_id": str(uuid.uuid4()), "iterations": 5, "warmup": 2},
        )
        assert resp.status_code in (422, 404)

    def test_run_empty_body_returns_error(self, test_client):
        resp = test_client.post("/api/v1/benchmarks/run", json={})
        assert resp.status_code in (422, 400)

    def test_run_invalid_iterations_returns_422(self, test_client):
        resp = test_client.post(
            "/api/v1/benchmarks/run",
            json={"model_id": str(uuid.uuid4()), "iterations": 0, "warmup": 0},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Get / list endpoints
# ---------------------------------------------------------------------------

class TestBenchmarkGet:
    def test_get_unknown_returns_404(self, test_client):
        resp = test_client.get(f"/api/v1/benchmarks/{uuid.uuid4()}")
        assert resp.status_code == 404

    def test_list_returns_200_empty(self, test_client):
        resp = test_client.get("/api/v1/benchmarks")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
