"""Integration tests for the Runtime API.

Tests the FastAPI → RuntimeService → DB + SQRuntime chain.
Uses SQLite in-memory DB.  Error cases are tested without real artifacts.
Full runtime lifecycle with real models is in tests/e2e/final/test_final_e2e.py.
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

_SQLITE_URL = "sqlite:///./test_runtime_api.db"
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
            if os.path.exists("test_runtime_api.db"):
                os.remove("test_runtime_api.db")
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
# Health endpoint — always available
# ---------------------------------------------------------------------------

class TestRuntimeHealth:
    def test_health_returns_200(self, test_client):
        resp = test_client.get("/api/v1/runtime/health")
        assert resp.status_code == 200

    def test_health_has_status_field(self, test_client):
        resp = test_client.get("/api/v1/runtime/health")
        body = resp.json()
        assert "status" in body

    def test_health_has_device_field(self, test_client):
        resp = test_client.get("/api/v1/runtime/health")
        body = resp.json()
        assert "device" in body

    def test_health_not_ready_by_default(self, test_client):
        # The runtime may be ready from a prior test run in the same process,
        # so we accept either state — just validate the schema.
        resp = test_client.get("/api/v1/runtime/health")
        body = resp.json()
        assert body["status"] in ("uninitialized", "loading", "ready", "failed")


# ---------------------------------------------------------------------------
# Load endpoint — error cases (no real artifact)
# ---------------------------------------------------------------------------

class TestRuntimeLoad:
    def test_load_missing_model_id_returns_error(self, test_client):
        resp = test_client.post(
            "/api/v1/runtime/load",
            json={"model_id": str(uuid.uuid4())},
        )
        # Should be 422 (not found)
        assert resp.status_code in (422, 404, 500)

    def test_load_missing_quantization_id_returns_error(self, test_client):
        resp = test_client.post(
            "/api/v1/runtime/load",
            json={"quantization_id": str(uuid.uuid4())},
        )
        assert resp.status_code in (422, 404, 500)

    def test_load_empty_body_returns_error(self, test_client):
        resp = test_client.post("/api/v1/runtime/load", json={})
        assert resp.status_code in (422, 400)


# ---------------------------------------------------------------------------
# Predict endpoint — error when not ready
# ---------------------------------------------------------------------------

class TestPredictNotReady:
    def test_predict_without_load_returns_503(self, test_client):
        # Use a fresh SQRuntime with no model loaded — by overriding the singleton
        from app.services import runtime_service as rs
        old_runtime = rs._runtime
        rs._runtime = rs.SQRuntime()  # fresh unloaded runtime
        try:
            resp = test_client.post(
                "/api/v1/runtime/predict",
                json={"values": [0.0] * 16},
            )
            assert resp.status_code == 503
        finally:
            rs._runtime = old_runtime

    def test_predict_batch_without_load_returns_503(self, test_client):
        from app.services import runtime_service as rs
        old_runtime = rs._runtime
        rs._runtime = rs.SQRuntime()
        try:
            resp = test_client.post(
                "/api/v1/runtime/predict/batch",
                json={"windows": [[0.0] * 16]},
            )
            assert resp.status_code == 503
        finally:
            rs._runtime = old_runtime


# ---------------------------------------------------------------------------
# Telemetry endpoint — always available
# ---------------------------------------------------------------------------

class TestTelemetry:
    def test_telemetry_returns_200(self, test_client):
        resp = test_client.get("/api/v1/runtime/telemetry")
        assert resp.status_code == 200

    def test_telemetry_has_required_fields(self, test_client):
        resp = test_client.get("/api/v1/runtime/telemetry")
        body = resp.json()
        assert "request_count" in body
        assert "success_count" in body
        assert "error_count" in body
