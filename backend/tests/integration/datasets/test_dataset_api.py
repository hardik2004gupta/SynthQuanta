"""Integration tests for the Dataset API.

These tests go through the full FastAPI → DatasetService → engine chain.
The database dependency is replaced with an in-memory SQLite DB so the
tests run without a live PostgreSQL connection.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.db.base import Base
from app.db.session import get_db
from app.main import app


# ---------------------------------------------------------------------------
# In-memory SQLite override — no PostgreSQL required for integration tests
# ---------------------------------------------------------------------------

_SQLITE_URL = "sqlite:///./test_datasets.db"

_test_engine = create_engine(
    _SQLITE_URL,
    connect_args={"check_same_thread": False},
)
_TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_test_engine)


@pytest.fixture(scope="module", autouse=True)
def create_tables():
    Base.metadata.create_all(bind=_test_engine)
    yield
    Base.metadata.drop_all(bind=_test_engine)
    _test_engine.dispose()
    import os, time
    # On Windows the file may still be held briefly after dispose(); retry once.
    for _ in range(3):
        try:
            if os.path.exists("test_datasets.db"):
                os.remove("test_datasets.db")
            break
        except PermissionError:
            time.sleep(0.2)


def _override_get_db():
    db = _TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="module")
def api_client():
    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Minimal generation config
# ---------------------------------------------------------------------------

MINIMAL_CONFIG = {
    "name": "integration-test-dataset",
    "seed": 42,
    "signal": {
        "type": "sinusoidal",
        "duration": 2.0,
        "sampling_rate": 100.0,
        "amplitude": 1.0,
        "frequency": 5.0,
    },
    "faults": {},
    "window_size": 64,
}


# ---------------------------------------------------------------------------
# POST /api/v1/datasets
# ---------------------------------------------------------------------------

class TestGenerateDataset:
    def test_generate_returns_201(self, api_client: TestClient) -> None:
        response = api_client.post("/api/v1/datasets", json=MINIMAL_CONFIG)
        assert response.status_code == 201

    def test_response_has_dataset_id(self, api_client: TestClient) -> None:
        response = api_client.post("/api/v1/datasets", json=MINIMAL_CONFIG)
        body = response.json()
        assert "dataset_id" in body
        assert body["dataset_id"]

    def test_response_has_human_id(self, api_client: TestClient) -> None:
        response = api_client.post("/api/v1/datasets", json=MINIMAL_CONFIG)
        body = response.json()
        assert "human_id" in body
        assert body["human_id"].startswith("DS-")

    def test_status_completed(self, api_client: TestClient) -> None:
        response = api_client.post("/api/v1/datasets", json=MINIMAL_CONFIG)
        assert response.json()["status"] == "COMPLETED"

    def test_sample_count_correct(self, api_client: TestClient) -> None:
        response = api_client.post("/api/v1/datasets", json=MINIMAL_CONFIG)
        body = response.json()
        # 2.0s * 100Hz = 200 samples
        assert body["sample_count"] == 200

    def test_signal_preview_present(self, api_client: TestClient) -> None:
        response = api_client.post("/api/v1/datasets", json=MINIMAL_CONFIG)
        preview = response.json()["signal_preview"]
        assert isinstance(preview, list)
        assert len(preview) > 0
        assert len(preview[0]) == 2  # [timestamp, value]

    def test_validation_included(self, api_client: TestClient) -> None:
        response = api_client.post("/api/v1/datasets", json=MINIMAL_CONFIG)
        val = response.json()["validation"]
        assert "valid" in val
        assert val["valid"] is True

    def test_split_counts_present(self, api_client: TestClient) -> None:
        response = api_client.post("/api/v1/datasets", json=MINIMAL_CONFIG)
        splits = response.json()["split_counts"]
        assert "train" in splits
        assert "validation" in splits
        assert "iid_test" in splits
        assert "shift_test" in splits

    def test_fault_annotations_empty_when_no_faults(self, api_client: TestClient) -> None:
        response = api_client.post("/api/v1/datasets", json=MINIMAL_CONFIG)
        assert response.json()["fault_annotations"] == []

    def test_fault_annotations_populated_when_faults_enabled(self, api_client: TestClient) -> None:
        cfg = {**MINIMAL_CONFIG, "faults": {"noise": {"enabled": True, "std": 0.1}}}
        response = api_client.post("/api/v1/datasets", json=cfg)
        assert len(response.json()["fault_annotations"]) == 1


# ---------------------------------------------------------------------------
# Validation of bad requests
# ---------------------------------------------------------------------------

class TestGenerateValidation:
    def test_invalid_signal_type(self, api_client: TestClient) -> None:
        cfg = {**MINIMAL_CONFIG, "signal": {**MINIMAL_CONFIG["signal"], "type": "INVALID"}}
        response = api_client.post("/api/v1/datasets", json=cfg)
        assert response.status_code == 422

    def test_zero_duration(self, api_client: TestClient) -> None:
        cfg = {**MINIMAL_CONFIG, "signal": {**MINIMAL_CONFIG["signal"], "duration": 0}}
        response = api_client.post("/api/v1/datasets", json=cfg)
        assert response.status_code == 422

    def test_zero_sampling_rate(self, api_client: TestClient) -> None:
        cfg = {**MINIMAL_CONFIG, "signal": {**MINIMAL_CONFIG["signal"], "sampling_rate": 0}}
        response = api_client.post("/api/v1/datasets", json=cfg)
        assert response.status_code == 422

    def test_invalid_window_size(self, api_client: TestClient) -> None:
        response = api_client.post("/api/v1/datasets", json={**MINIMAL_CONFIG, "window_size": 0})
        assert response.status_code == 422

    def test_noise_invalid_fracs(self, api_client: TestClient) -> None:
        cfg = {
            **MINIMAL_CONFIG,
            "faults": {"noise": {"enabled": True, "std": 0.1, "start_frac": 0.8, "end_frac": 0.2}},
        }
        response = api_client.post("/api/v1/datasets", json=cfg)
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/datasets
# ---------------------------------------------------------------------------

class TestListDatasets:
    def test_list_returns_200(self, api_client: TestClient) -> None:
        response = api_client.get("/api/v1/datasets")
        assert response.status_code == 200

    def test_list_response_structure(self, api_client: TestClient) -> None:
        # Generate at least one dataset first
        api_client.post("/api/v1/datasets", json=MINIMAL_CONFIG)
        response = api_client.get("/api/v1/datasets")
        body = response.json()
        assert "items" in body
        assert "total" in body
        assert "limit" in body
        assert "offset" in body

    def test_list_contains_generated_datasets(self, api_client: TestClient) -> None:
        response = api_client.get("/api/v1/datasets")
        body = response.json()
        assert body["total"] >= 1

    def test_list_pagination(self, api_client: TestClient) -> None:
        response = api_client.get("/api/v1/datasets?limit=1&offset=0")
        assert response.status_code == 200
        body = response.json()
        assert len(body["items"]) <= 1


# ---------------------------------------------------------------------------
# GET /api/v1/datasets/{dataset_id}
# ---------------------------------------------------------------------------

class TestGetDataset:
    def test_get_existing(self, api_client: TestClient) -> None:
        create_resp = api_client.post("/api/v1/datasets", json=MINIMAL_CONFIG)
        dataset_id = create_resp.json()["dataset_id"]
        get_resp = api_client.get(f"/api/v1/datasets/{dataset_id}")
        assert get_resp.status_code == 200

    def test_get_returns_same_id(self, api_client: TestClient) -> None:
        create_resp = api_client.post("/api/v1/datasets", json=MINIMAL_CONFIG)
        dataset_id = create_resp.json()["dataset_id"]
        body = api_client.get(f"/api/v1/datasets/{dataset_id}").json()
        assert body["dataset_id"] == dataset_id

    def test_get_missing_returns_404(self, api_client: TestClient) -> None:
        response = api_client.get("/api/v1/datasets/does-not-exist-xyz")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Determinism integration test (critical)
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_content(self, api_client: TestClient) -> None:
        """Two requests with identical config+seed must produce identical data."""
        cfg = {
            "name": "repro-test",
            "seed": 42,
            "signal": {
                "type": "sinusoidal",
                "duration": 1.0,
                "sampling_rate": 100.0,
                "amplitude": 1.0,
                "frequency": 10.0,
            },
            "faults": {"noise": {"enabled": True, "std": 0.05}},
            "window_size": 64,
        }
        resp_a = api_client.post("/api/v1/datasets", json=cfg).json()
        resp_b = api_client.post("/api/v1/datasets", json=cfg).json()

        # IDs differ (separate artifacts), but previews must be identical
        assert resp_a["dataset_id"] != resp_b["dataset_id"]
        assert resp_a["signal_preview"] == resp_b["signal_preview"]
        assert resp_a["sample_count"] == resp_b["sample_count"]
        assert resp_a["window_count"] == resp_b["window_count"]
        assert resp_a["fault_count"] == resp_b["fault_count"]

    def test_different_seeds_different_content(self, api_client: TestClient) -> None:
        cfg_42 = {
            "name": "repro-42",
            "seed": 42,
            "signal": {"type": "sinusoidal", "duration": 1.0, "sampling_rate": 100.0,
                       "amplitude": 1.0, "frequency": 10.0},
            "faults": {"noise": {"enabled": True, "std": 0.1}},
            "window_size": 64,
        }
        cfg_43 = {**cfg_42, "seed": 43, "name": "repro-43"}
        resp_42 = api_client.post("/api/v1/datasets", json=cfg_42).json()
        resp_43 = api_client.post("/api/v1/datasets", json=cfg_43).json()
        # Previews should differ since noise is seed-dependent
        assert resp_42["signal_preview"] != resp_43["signal_preview"]
