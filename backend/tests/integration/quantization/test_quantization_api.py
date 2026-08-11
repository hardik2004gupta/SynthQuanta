"""Integration tests for the Quantization API.

Tests the FastAPI → QuantizationService → DB chain using in-memory SQLite.
Does NOT run actual model quantization — we test API contract only:
  - 404 for unknown model
  - 409 for non-fp32 model
  - 409 for non-COMPLETED model
  - 202 for a valid COMPLETED fp32 model
  - GET returns the quantization record

Full quantization with real model is covered by tests/e2e/test_quantization_e2e.py.
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

_SQLITE_URL = "sqlite:///./test_quantization_api.db"
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
            if os.path.exists("test_quantization_api.db"):
                os.remove("test_quantization_api.db")
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
# Seed a COMPLETED fp32 model directly via DB (bypassing real training)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def completed_fp32_model_id(api_client, tmp_path_factory):
    """Seed a COMPLETED fp32 MLModel + minimal checkpoint in the DB."""
    import json
    import torch
    from app.db.models.dataset import Dataset
    from app.db.models.experiment import Experiment
    from app.db.models.ml_model import MLModel
    from app.ml.models.sensor_transformer import ModelConfig, SensorTransformer

    tmp_root = tmp_path_factory.mktemp("quant_artifacts")

    # Patch QuantizationService artifact root
    import app.services.quantization_service as qs_mod
    qs_mod._settings.artifact_root = str(tmp_root)

    # Create minimal checkpoint
    model_id = str(uuid.uuid4())
    model_artifact_path = "models/MODEL-QUANT-TEST"
    ckpt_dir = tmp_root / model_artifact_path
    (ckpt_dir / "base").mkdir(parents=True, exist_ok=True)

    cfg = ModelConfig(window_size=16, embedding_dim=8, num_layers=1,
                      num_heads=2, ffn_dim=16, dropout=0.0)
    model = SensorTransformer(cfg)
    torch.save(model.state_dict(), ckpt_dir / "base" / "model.pt")

    ckpt_meta = {
        "config": {"model": {"architecture": "sensor-transformer-small",
                              "window_size": 16, "embedding_dim": 8,
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
        ds_id = str(uuid.uuid4())
        ds = Dataset(
            id=ds_id, human_id="DS-QUANT-TEST", name="quant-test-ds",
            seed=42, status="COMPLETED",
            configuration={}, artifact_path="datasets/DS-QUANT-TEST",
            sample_count=200, window_count=10,
        )
        db.add(ds)

        exp_id = str(uuid.uuid4())
        exp = Experiment(
            id=exp_id, human_id="EXP-QUANT-TEST", name="quant-test-exp",
            dataset_id=ds_id, method="full",
            configuration={}, status="COMPLETED",
            artifact_path=model_artifact_path,
            metrics={"best_val_loss": 0.5},
        )
        db.add(exp)

        ml = MLModel(
            id=model_id, human_id="MODEL-QUANT-TEST", name="quant-test-model",
            version="v1", base_model="sensor-transformer-small",
            experiment_id=exp_id, precision="fp32",
            artifact_path=model_artifact_path, status="COMPLETED",
        )
        db.add(ml)
        db.commit()
    finally:
        db.close()

    return model_id


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------

class TestQuantizationRunEndpoint:
    def test_unknown_model_returns_404(self, api_client):
        resp = api_client.post(
            "/api/v1/quantization/run",
            json={"source_model_id": "00000000-0000-0000-0000-000000000000"},
        )
        assert resp.status_code == 404
        assert "MODEL_NOT_FOUND" in resp.json()["detail"]["code"]

    def test_int8_model_returns_409(self, api_client):
        from app.db.models.ml_model import MLModel
        db = _TestSession()
        int8_model_id = str(uuid.uuid4())
        try:
            int8 = MLModel(
                id=int8_model_id, human_id="MODEL-INT8-TEST", name="int8-model",
                version="v1", base_model="sensor-transformer-small",
                precision="int8", artifact_path="models/MODEL-INT8-TEST",
                status="COMPLETED",
            )
            db.add(int8)
            db.commit()
        finally:
            db.close()

        resp = api_client.post(
            "/api/v1/quantization/run",
            json={"source_model_id": int8_model_id},
        )
        assert resp.status_code == 409
        assert "MODEL_NOT_FP32" in resp.json()["detail"]["code"]

    def test_pending_model_returns_409(self, api_client):
        from app.db.models.ml_model import MLModel
        db = _TestSession()
        pending_id = str(uuid.uuid4())
        try:
            m = MLModel(
                id=pending_id, human_id="MODEL-PEND-TEST", name="pend-model",
                version="v1", base_model="sensor-transformer-small",
                precision="fp32", artifact_path="models/MODEL-PEND-TEST",
                status="PENDING",
            )
            db.add(m)
            db.commit()
        finally:
            db.close()

        resp = api_client.post(
            "/api/v1/quantization/run",
            json={"source_model_id": pending_id},
        )
        assert resp.status_code == 409
        assert "MODEL_NOT_READY" in resp.json()["detail"]["code"]

    def test_valid_model_starts_job(self, api_client, completed_fp32_model_id):
        resp = api_client.post(
            "/api/v1/quantization/run",
            json={"source_model_id": completed_fp32_model_id},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "quantization_id" in data
        assert "human_id" in data
        assert data["human_id"].startswith("QUANT-")
        assert data["status"] in ("PENDING", "RUNNING")
        assert data["source_model_id"] == completed_fp32_model_id

    def test_start_response_has_human_id(self, api_client, completed_fp32_model_id):
        resp = api_client.post(
            "/api/v1/quantization/run",
            json={"source_model_id": completed_fp32_model_id},
        )
        assert resp.status_code == 202
        assert resp.json()["human_id"].startswith("QUANT-")


class TestQuantizationGetEndpoint:
    def test_get_unknown_returns_404(self, api_client):
        resp = api_client.get("/api/v1/quantization/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404

    def test_get_created_quantization(self, api_client, completed_fp32_model_id):
        start_resp = api_client.post(
            "/api/v1/quantization/run",
            json={"source_model_id": completed_fp32_model_id},
        )
        assert start_resp.status_code == 202
        quant_id = start_resp.json()["quantization_id"]

        get_resp = api_client.get(f"/api/v1/quantization/{quant_id}")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["quantization_id"] == quant_id
        assert data["source_model_id"] == completed_fp32_model_id
        assert data["status"] in ("PENDING", "RUNNING", "COMPLETED", "FAILED")

    def test_list_quantizations(self, api_client):
        resp = api_client.get("/api/v1/quantization?limit=10")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
