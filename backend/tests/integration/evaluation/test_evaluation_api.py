"""Integration tests for the Evaluation API.

Tests the FastAPI → EvaluationService → DB chain using in-memory SQLite.
Does NOT run actual model inference — we test API contract only:
  - 404 for unknown experiment
  - 409 for non-COMPLETED experiment
  - 202 for a COMPLETED experiment (job is created and returns evaluation_id)
  - GET returns the evaluation record

Full evaluation with real model is covered by tests/e2e/test_evaluation_e2e.py.
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

_SQLITE_URL = "sqlite:///./test_evaluation_api.db"
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
            if os.path.exists("test_evaluation_api.db"):
                os.remove("test_evaluation_api.db")
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
# Seed a COMPLETED experiment directly via DB (bypassing real training)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def completed_experiment_id(api_client, tmp_path_factory):
    """Seed a COMPLETED experiment + minimal checkpoint via the dataset and DB."""
    import json
    import numpy as np

    from app.db.models.dataset import Dataset
    from app.db.models.experiment import Experiment
    from app.db.models.ml_model import MLModel

    tmp_root = tmp_path_factory.mktemp("eval_artifacts")

    # --- Patch artifact store for EvaluationService ---
    import app.services.evaluation_service as ev_mod
    from app.services.artifact_store import ArtifactStore

    original_ev_init = ev_mod.EvaluationService.__init__

    def patched_ev_init(self, db):
        from app.db.repositories.dataset_repository import DatasetRepository
        from app.db.repositories.evaluation_repository import EvaluationRepository
        from app.db.repositories.experiment_repository import ExperimentRepository
        from app.db.repositories.model_repository import ModelRepository
        self._db = db
        self._eval_repo = EvaluationRepository(db)
        self._exp_repo = ExperimentRepository(db)
        self._ds_repo = DatasetRepository(db)
        self._model_repo = ModelRepository(db)
        self._store = ArtifactStore(root=tmp_root)

    ev_mod.EvaluationService.__init__ = patched_ev_init

    # Also patch the settings used by the background thread
    import app.services.evaluation_service as ev_direct
    ev_direct._settings.artifact_root = str(tmp_root)

    # Write a minimal dataset artifact
    ds_id = str(uuid.uuid4())
    ds_artifact_path = f"datasets/DS-EVAL-TEST"
    ds_dir = tmp_root / ds_artifact_path
    ds_dir.mkdir(parents=True, exist_ok=True)

    # Minimal npz
    n = 200
    values = np.random.randn(n).astype(np.float32)
    timestamps = np.arange(n, dtype=np.float64) / 100.0
    labels = np.zeros(n, dtype=np.int64)
    np.savez(ds_dir / "data.npz", values=values, timestamps=timestamps, labels=labels)

    ds_meta = {
        "seed": 42,
        "signal": {"type": "sinusoidal", "duration": 2.0, "sampling_rate": 100.0, "amplitude": 1.0, "frequency": 5.0},
        "faults": {},
        "window_size": 32,
        "splits": {"train": 140, "validation": 20, "iid_test": 20, "shift_test": 20},
    }
    (ds_dir / "metadata.json").write_text(json.dumps(ds_meta), encoding="utf-8")

    # Create a minimal checkpoint (enough for load_model_from_checkpoint pattern)
    model_id = str(uuid.uuid4())
    model_artifact_path = "models/MODEL-EVAL-TEST"
    ckpt_dir = tmp_root / model_artifact_path
    (ckpt_dir / "base").mkdir(parents=True, exist_ok=True)

    import torch
    from app.ml.models.sensor_transformer import ModelConfig, SensorTransformer

    cfg = ModelConfig(window_size=32, embedding_dim=8, num_layers=1,
                      num_heads=2, ffn_dim=16, dropout=0.0)
    model = SensorTransformer(cfg)
    torch.save(model.state_dict(), ckpt_dir / "base" / "model.pt")

    ckpt_meta = {
        "config": {"model": {"architecture": "sensor-transformer-small",
                              "window_size": 32, "embedding_dim": 8,
                              "num_layers": 1, "num_heads": 2, "ffn_dim": 16,
                              "dropout": 0.0}},
        "norm_stats": {"mean": 0.0, "std": 1.0},
        "method": "full",
        "lora": None,
    }
    (ckpt_dir / "metadata.json").write_text(json.dumps(ckpt_meta), encoding="utf-8")

    # Seed DB records
    db = _TestSession()
    try:
        # Dataset record
        ds_record = Dataset(
            id=ds_id, human_id="DS-EVAL-TEST", name="eval-test-dataset",
            seed=42, status="COMPLETED",
            configuration=ds_meta, artifact_path=ds_artifact_path,
            sample_count=n, window_count=6,
        )
        db.add(ds_record)

        # Experiment record (COMPLETED)
        exp_id = str(uuid.uuid4())
        exp_record = Experiment(
            id=exp_id, human_id="EXP-EVAL-TEST", name="eval-test-exp",
            dataset_id=ds_id, method="full",
            configuration={"model": cfg.__dict__},
            status="COMPLETED",
            artifact_path=model_artifact_path,
            metrics={"best_val_loss": 0.5},
        )
        db.add(exp_record)

        # MLModel record
        ml_record = MLModel(
            id=model_id, human_id="MODEL-EVAL-TEST", name="eval-test-model",
            version="v1", base_model="sensor-transformer-small",
            experiment_id=exp_id, precision="fp32",
            artifact_path=model_artifact_path, status="COMPLETED",
        )
        db.add(ml_record)
        db.commit()
    finally:
        db.close()

    yield exp_id

    ev_mod.EvaluationService.__init__ = original_ev_init


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------

class TestEvaluationRunEndpoint:
    def test_unknown_experiment_returns_404(self, api_client):
        resp = api_client.post(
            "/api/v1/evaluation/run",
            json={"experiment_id": "00000000-0000-0000-0000-000000000000"},
        )
        assert resp.status_code == 404
        assert "EXPERIMENT_NOT_FOUND" in resp.json()["detail"]["code"]

    def test_pending_experiment_returns_409(self, api_client):
        # Create a PENDING experiment
        db = _TestSession()
        from app.db.models.experiment import Experiment
        exp_id = str(uuid.uuid4())
        ds_id = str(uuid.uuid4())
        try:
            from app.db.models.dataset import Dataset
            ds = Dataset(
                id=ds_id, human_id="DS-PEND", name="pend", seed=1,
                status="COMPLETED",
                configuration={}, artifact_path="datasets/DS-PEND", sample_count=100, window_count=0,
            )
            db.add(ds)
            exp = Experiment(
                id=exp_id, human_id="EXP-PEND", name="pend-exp",
                dataset_id=ds_id, method="lora", configuration={}, status="PENDING",
            )
            db.add(exp)
            db.commit()
        finally:
            db.close()

        resp = api_client.post("/api/v1/evaluation/run", json={"experiment_id": str(exp_id)})
        assert resp.status_code == 409
        assert "EXPERIMENT_NOT_READY" in resp.json()["detail"]["code"]

    def test_completed_experiment_starts_job(self, api_client, completed_experiment_id):
        resp = api_client.post(
            "/api/v1/evaluation/run",
            json={"experiment_id": completed_experiment_id, "include_shift": False},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "evaluation_id" in data
        assert "human_id" in data
        assert data["human_id"].startswith("EVAL-")
        assert data["status"] in ("PENDING", "RUNNING")

    def test_start_response_has_experiment_id(self, api_client, completed_experiment_id):
        resp = api_client.post(
            "/api/v1/evaluation/run",
            json={"experiment_id": completed_experiment_id, "include_shift": False},
        )
        assert resp.status_code == 202
        assert resp.json()["experiment_id"] == completed_experiment_id


class TestEvaluationGetEndpoint:
    def test_get_unknown_returns_404(self, api_client):
        resp = api_client.get("/api/v1/evaluation/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404

    def test_get_created_evaluation(self, api_client, completed_experiment_id):
        # Start a job first
        start_resp = api_client.post(
            "/api/v1/evaluation/run",
            json={"experiment_id": completed_experiment_id, "include_shift": False},
        )
        assert start_resp.status_code == 202
        eval_id = start_resp.json()["evaluation_id"]

        # Poll for it
        get_resp = api_client.get(f"/api/v1/evaluation/{eval_id}")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["evaluation_id"] == eval_id
        assert data["experiment_id"] == completed_experiment_id
        assert data["status"] in ("PENDING", "RUNNING", "COMPLETED", "FAILED")

    def test_list_evaluations(self, api_client, completed_experiment_id):
        resp = api_client.get("/api/v1/evaluation?limit=10")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
