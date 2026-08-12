"""Unit tests for SQRuntime — load, predict, health, telemetry.

Tests the runtime directly without FastAPI (testability rule §12).
Creates tiny artifacts in tmp_path.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from app.ml.adapters.lora import LoRAConfig
from app.ml.datasets.dataset import DatasetFactory
from app.ml.models.sensor_transformer import ModelConfig
from app.ml.quantization.engine import QuantizationEngine
from app.ml.training.config import TrainingConfig
from app.ml.training.trainer import Trainer
from app.runtime.postprocessing import FAULT_LABELS
from app.runtime.runtime import RuntimeState, SQRuntime, SQRuntimeError


# ---------------------------------------------------------------------------
# Shared tiny pipeline fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_artifacts(tmp_path_factory):
    """Build minimal FP32 checkpoint + INT8 artifact for runtime tests."""
    tmp = tmp_path_factory.mktemp("sqruntime")
    window_size = 16
    n_samples = 400

    # Dataset
    rng = np.random.default_rng(99)
    ts = np.linspace(0.0, 4.0, n_samples)
    vs = np.sin(2 * np.pi * 3 * ts).astype(np.float32)
    labels = np.zeros(n_samples, dtype=np.int64)
    labels[100:150] = 1  # NOISE

    ds_dir = tmp / "DS-RT"
    ds_dir.mkdir()
    np.savez(str(ds_dir / "data.npz"), timestamps=ts, values=vs, labels=labels)
    (ds_dir / "metadata.json").write_text(json.dumps({
        "human_id": "DS-RT",
        "sample_count": n_samples,
        "window_size": window_size,
        "seed": 99,
        "configuration": {
            "seed": 99,
            "signal": {"type": "sinusoidal", "duration": 4.0, "sampling_rate": 100.0,
                       "amplitude": 1.0, "frequency": 3.0},
            "faults": {"noise": {"enabled": True}},
        },
    }))

    # Train tiny model
    ckpt_dir = tmp / "MODEL-RT"
    cfg = TrainingConfig(
        method="lora",
        epochs=2,
        batch_size=16,
        learning_rate=5e-4,
        seed=99,
        lora=LoRAConfig(rank=2, target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]),
        model=ModelConfig(window_size=window_size, embedding_dim=8, num_layers=1,
                         num_heads=2, ffn_dim=16, dropout=0.0),
    )
    splits = DatasetFactory.from_artifact(ds_dir, window_size=window_size)
    Trainer(
        train_dataset=splits.train,
        val_dataset=splits.validation,
        config=cfg,
        checkpoint_dir=ckpt_dir,
        norm_stats=splits.norm_stats,
    ).train()

    # Quantize
    from app.ml.training.trainer import load_model_from_checkpoint
    fp32_model, ckpt_meta = load_model_from_checkpoint(ckpt_dir, device="cpu")
    engine = QuantizationEngine()
    q_model = engine.quantize(fp32_model)
    int8_dir = tmp / "INT8-RT"
    engine.save_quantized(q_model, int8_dir, ckpt_meta)

    return {
        "fp32_dir": ckpt_dir,
        "int8_dir": int8_dir,
        "window_size": window_size,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRuntimeInitialState:
    def test_initially_uninitialized(self):
        rt = SQRuntime()
        assert rt.health().status == RuntimeState.UNINITIALIZED

    def test_not_ready_initially(self):
        rt = SQRuntime()
        assert not rt.is_ready()


class TestFP32Load:
    def test_load_fp32_sets_ready(self, tiny_artifacts):
        rt = SQRuntime()
        rt.load(
            artifact_dir=tiny_artifacts["fp32_dir"],
            precision="fp32",
            model_id="test-fp32",
            artifact_path="models/MODEL-RT",
        )
        assert rt.is_ready()
        assert rt.health().status == RuntimeState.READY
        assert rt.health().precision == "fp32"

    def test_load_fp32_health_details(self, tiny_artifacts):
        rt = SQRuntime()
        rt.load(
            artifact_dir=tiny_artifacts["fp32_dir"],
            precision="fp32",
            model_id="test-fp32",
            artifact_path="models/MODEL-RT",
        )
        h = rt.health()
        assert h.model_id == "test-fp32"
        assert h.loaded_at is not None
        assert h.device == "cpu"

    def test_unsupported_precision_raises(self):
        rt = SQRuntime()
        with pytest.raises(SQRuntimeError):
            rt.load(Path("/nonexistent"), precision="fp16", model_id="x", artifact_path="x")

    def test_missing_artifact_fails(self):
        rt = SQRuntime()
        with pytest.raises(SQRuntimeError):
            rt.load(Path("/nonexistent/path"), precision="fp32", model_id="x", artifact_path="x")
        assert rt.health().status == RuntimeState.FAILED


class TestINT8Load:
    def test_load_int8_sets_ready(self, tiny_artifacts):
        rt = SQRuntime()
        rt.load(
            artifact_dir=tiny_artifacts["int8_dir"],
            precision="int8",
            model_id="test-int8",
            artifact_path="quantizations/INT8-RT",
        )
        assert rt.is_ready()
        assert rt.health().precision == "int8"


class TestPredict:
    def test_predict_returns_result(self, tiny_artifacts):
        rt = SQRuntime()
        rt.load(
            artifact_dir=tiny_artifacts["fp32_dir"],
            precision="fp32",
            model_id="test-fp32",
            artifact_path="models/MODEL-RT",
        )
        ws = tiny_artifacts["window_size"]
        result = rt.predict([0.0] * ws)
        assert result.predicted_class in FAULT_LABELS
        assert 0.0 <= result.confidence <= 1.0
        assert result.latency_ms > 0.0
        assert len(result.probabilities) == 7

    def test_predict_wrong_length_raises(self, tiny_artifacts):
        rt = SQRuntime()
        rt.load(
            artifact_dir=tiny_artifacts["fp32_dir"],
            precision="fp32",
            model_id="test-fp32",
            artifact_path="models/MODEL-RT",
        )
        with pytest.raises(SQRuntimeError, match="Preprocessing"):
            rt.predict([0.0] * 5)

    def test_predict_nan_input_raises(self, tiny_artifacts):
        rt = SQRuntime()
        rt.load(
            artifact_dir=tiny_artifacts["fp32_dir"],
            precision="fp32",
            model_id="test-fp32",
            artifact_path="models/MODEL-RT",
        )
        ws = tiny_artifacts["window_size"]
        values = [0.0] * ws
        values[0] = float("nan")
        with pytest.raises(SQRuntimeError, match="Preprocessing"):
            rt.predict(values)

    def test_predict_not_ready_raises(self):
        rt = SQRuntime()
        with pytest.raises(SQRuntimeError, match="not ready"):
            rt.predict([0.0] * 16)

    def test_predict_updates_telemetry(self, tiny_artifacts):
        rt = SQRuntime()
        rt.load(
            artifact_dir=tiny_artifacts["fp32_dir"],
            precision="fp32",
            model_id="test-fp32",
            artifact_path="models/MODEL-RT",
        )
        ws = tiny_artifacts["window_size"]
        rt.predict([0.0] * ws)
        rt.predict([1.0] * ws)
        t = rt.get_telemetry()
        assert t.request_count == 2
        assert t.success_count == 2


class TestBatchPredict:
    def test_predict_batch_shape(self, tiny_artifacts):
        rt = SQRuntime()
        rt.load(
            artifact_dir=tiny_artifacts["fp32_dir"],
            precision="fp32",
            model_id="test-fp32",
            artifact_path="models/MODEL-RT",
        )
        ws = tiny_artifacts["window_size"]
        windows = [[float(i % 3)] * ws for i in range(4)]
        results = rt.predict_batch(windows)
        assert len(results) == 4
        for r in results:
            assert r.predicted_class in FAULT_LABELS

    def test_predict_batch_empty_raises(self, tiny_artifacts):
        rt = SQRuntime()
        rt.load(
            artifact_dir=tiny_artifacts["fp32_dir"],
            precision="fp32",
            model_id="test-fp32",
            artifact_path="models/MODEL-RT",
        )
        with pytest.raises(SQRuntimeError, match="at least one"):
            rt.predict_batch([])

    def test_single_consistency(self, tiny_artifacts):
        """Single and batch prediction on same input should agree on class."""
        rt = SQRuntime()
        rt.load(
            artifact_dir=tiny_artifacts["fp32_dir"],
            precision="fp32",
            model_id="test-fp32",
            artifact_path="models/MODEL-RT",
        )
        ws = tiny_artifacts["window_size"]
        values = [0.5] * ws
        single = rt.predict(values)
        batch = rt.predict_batch([values])
        assert single.predicted_class == batch[0].predicted_class


class TestUnload:
    def test_unload_returns_to_uninitialized(self, tiny_artifacts):
        rt = SQRuntime()
        rt.load(
            artifact_dir=tiny_artifacts["fp32_dir"],
            precision="fp32",
            model_id="test-fp32",
            artifact_path="models/MODEL-RT",
        )
        rt.unload()
        assert rt.health().status == RuntimeState.UNINITIALIZED
        assert not rt.is_ready()
