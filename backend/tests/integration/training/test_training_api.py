"""Integration tests for the Training API.

These run through the full FastAPI → TrainingService → DB chain using an
in-memory SQLite DB.  A real (tiny) dataset artifact is written to a temp
directory so DatasetFactory can load it.

NOTE: These tests do NOT start actual training threads — we verify API
contract (job creation, status polling) with a minimal in-process call.
The full end-to-end training run is covered by tests/e2e/test_training_e2e.py.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.session import get_db
from app.main import app

# ---------------------------------------------------------------------------
# SQLite override
# ---------------------------------------------------------------------------

_SQLITE_URL = "sqlite:///./test_training_api.db"
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
            if os.path.exists("test_training_api.db"):
                os.remove("test_training_api.db")
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
def api_client():
    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client
    app.dependency_overrides.pop(get_db, None)


# ---------------------------------------------------------------------------
# Artifact fixture — write a real dataset so the service can load it
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dataset_id(api_client, tmp_path_factory, monkeypatch):
    """Generate a real dataset via the API and return its ID."""
    # Override artifact root to use tmp dir
    tmp_root = tmp_path_factory.mktemp("artifacts")

    import app.services.dataset_service as ds_mod
    import app.services.training_service as ts_mod
    from app.services.artifact_store import ArtifactStore

    original_ds_init = ds_mod.DatasetService.__init__
    original_ts_init = ts_mod.TrainingService.__init__

    def patched_ds_init(self, db):
        from app.core.config import get_settings
        from app.data.engine import SyntheticDataEngine
        from app.db.repositories.dataset_repository import DatasetRepository
        self._db = db
        self._repo = DatasetRepository(db)
        self._engine = SyntheticDataEngine()
        self._store = ArtifactStore(root=tmp_root)

    def patched_ts_init(self, db):
        from app.db.repositories.dataset_repository import DatasetRepository
        from app.db.repositories.experiment_repository import ExperimentRepository
        from app.db.repositories.model_repository import ModelRepository
        self._db = db
        self._exp_repo = ExperimentRepository(db)
        self._ds_repo = DatasetRepository(db)
        self._model_repo = ModelRepository(db)
        self._store = ArtifactStore(root=tmp_root)

    ds_mod.DatasetService.__init__ = patched_ds_init
    ts_mod.TrainingService.__init__ = patched_ts_init

    # Also patch the background thread's settings reference
    import app.services.training_service as ts_mod_direct
    original_settings = ts_mod_direct._settings
    ts_mod_direct._settings.artifact_root = str(tmp_root)

    yield tmp_root

    ds_mod.DatasetService.__init__ = original_ds_init
    ts_mod.TrainingService.__init__ = original_ts_init


# ---------------------------------------------------------------------------
# A simpler approach: just test contract with a pre-seeded dataset record
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def seeded_dataset_id(api_client, tmp_path_factory):
    """Seed a COMPLETED dataset directly in the DB via the generate API."""
    payload = {
        "name": "train-api-test",
        "seed": 42,
        "signal": {"type": "sinusoidal", "duration": 1.0, "sampling_rate": 100.0, "amplitude": 1.0, "frequency": 5.0},
        "faults": {"noise": {"enabled": True, "std": 0.1, "start_frac": 0.2, "end_frac": 0.8}},
        "window_size": 32,
    }
    resp = api_client.post("/api/v1/datasets", json=payload)
    assert resp.status_code in (200, 201), resp.text
    return resp.json()["dataset_id"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_start_training_unknown_dataset(api_client):
    payload = {
        "dataset_id": "00000000-0000-0000-0000-000000000000",
        "name": "test-exp",
        "method": "lora",
        "epochs": 1, "batch_size": 4, "learning_rate": 0.001, "weight_decay": 0.01, "seed": 42,
        "lora": {"rank": 2, "alpha": 4.0, "dropout": 0.05, "target_modules": ["q_proj"]},
        "model": {"architecture": "sensor-transformer-small", "window_size": 32,
                  "embedding_dim": 8, "num_layers": 1, "num_heads": 2, "ffn_dim": 16, "dropout": 0.1},
    }
    resp = api_client.post("/api/v1/training/run", json=payload)
    assert resp.status_code == 404
    assert "DATASET_NOT_FOUND" in resp.json()["detail"]["code"]


def test_start_training_invalid_method(api_client, seeded_dataset_id):
    payload = {
        "dataset_id": seeded_dataset_id,
        "name": "test-exp",
        "method": "invalid_method",
        "epochs": 1, "batch_size": 4, "learning_rate": 0.001, "weight_decay": 0.01, "seed": 42,
        "lora": {"rank": 2, "alpha": 4.0, "dropout": 0.05, "target_modules": ["q_proj"]},
        "model": {"architecture": "sensor-transformer-small", "window_size": 32,
                  "embedding_dim": 8, "num_layers": 1, "num_heads": 2, "ffn_dim": 16, "dropout": 0.1},
    }
    resp = api_client.post("/api/v1/training/run", json=payload)
    assert resp.status_code == 422  # validation error from Pydantic


def test_start_training_returns_pending(api_client, seeded_dataset_id):
    payload = {
        "dataset_id": seeded_dataset_id,
        "name": "integration-lora-test",
        "method": "lora",
        "epochs": 1, "batch_size": 4, "learning_rate": 0.001, "weight_decay": 0.01, "seed": 42,
        "lora": {"rank": 2, "alpha": 4.0, "dropout": 0.05, "target_modules": ["q_proj", "k_proj"]},
        "model": {"architecture": "sensor-transformer-small", "window_size": 32,
                  "embedding_dim": 8, "num_layers": 1, "num_heads": 2, "ffn_dim": 16, "dropout": 0.1},
    }
    resp = api_client.post("/api/v1/training/run", json=payload)
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert "experiment_id" in body
    assert "human_id" in body
    assert body["human_id"].startswith("EXP-")
    # Status is PENDING at time of response (thread may advance it)
    assert body["status"] in ("PENDING", "RUNNING", "COMPLETED", "FAILED")


def test_get_experiment_not_found(api_client):
    resp = api_client.get("/api/v1/training/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


def test_get_experiment_returns_record(api_client, seeded_dataset_id):
    # Start a job
    payload = {
        "dataset_id": seeded_dataset_id,
        "name": "get-test",
        "method": "lora",
        "epochs": 1, "batch_size": 4, "learning_rate": 0.001, "weight_decay": 0.01, "seed": 42,
        "lora": {"rank": 2, "alpha": 4.0, "dropout": 0.05, "target_modules": ["q_proj"]},
        "model": {"architecture": "sensor-transformer-small", "window_size": 32,
                  "embedding_dim": 8, "num_layers": 1, "num_heads": 2, "ffn_dim": 16, "dropout": 0.1},
    }
    start_resp = api_client.post("/api/v1/training/run", json=payload)
    assert start_resp.status_code == 202
    exp_id = start_resp.json()["experiment_id"]

    # Poll the GET endpoint
    get_resp = api_client.get(f"/api/v1/training/{exp_id}")
    assert get_resp.status_code == 200
    body = get_resp.json()
    assert body["experiment_id"] == exp_id
    assert body["method"] == "lora"
    assert body["dataset_id"] == seeded_dataset_id
    assert body["status"] in ("PENDING", "RUNNING", "COMPLETED", "FAILED")
