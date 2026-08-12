"""End-to-end quantization test — Phase 5.

Runs the complete quantization lifecycle without any mocking:
  1. Train a tiny FP32 model (2 epochs, LoRA)
  2. Quantize FP32 → INT8 using QuantizationEngine
  3. Validate the quantized model forward pass (smoke-test)
  4. Save the INT8 artifact
  5. Reload the INT8 artifact
  6. Run FP32 vs INT8 comparison (F1, size, latency)
  7. Verify all required metrics are present and in valid ranges

This tests QuantizationEngine and run_comparison() directly — no FastAPI, no DB.
Runs in seconds on CPU.
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
from app.ml.quantization.comparison import run_comparison
from app.ml.quantization.engine import QuantizationEngine
from app.ml.training.config import TrainingConfig
from app.ml.training.trainer import Trainer, load_model_from_checkpoint


# ---------------------------------------------------------------------------
# Shared tiny artifact
# ---------------------------------------------------------------------------

def _create_tiny_artifact(artifact_dir: Path, n_samples: int = 600, window_size: int = 16) -> Path:
    rng = np.random.default_rng(77)
    timestamps = np.linspace(0.0, 6.0, n_samples)
    values = np.sin(2 * np.pi * 5 * timestamps) + rng.normal(0, 0.1, n_samples)
    values = values.astype(np.float32)
    labels = np.zeros(n_samples, dtype=np.int64)
    seg = n_samples // 7
    for fault_label in range(1, 7):
        start = fault_label * seg
        end = start + seg // 2
        labels[start:end] = fault_label

    ds_dir = artifact_dir / "DS-QUANT-E2E"
    ds_dir.mkdir(parents=True)
    np.savez(str(ds_dir / "data.npz"), timestamps=timestamps, values=values, labels=labels)
    meta = {
        "human_id": "DS-QUANT-E2E",
        "sample_count": n_samples,
        "window_size": window_size,
        "seed": 77,
        "configuration": {
            "seed": 77,
            "signal": {"type": "sinusoidal", "duration": 6.0, "sampling_rate": 100.0,
                        "amplitude": 1.0, "frequency": 5.0},
            "faults": {"noise": {"enabled": True, "std": 0.1}},
        },
    }
    (ds_dir / "metadata.json").write_text(json.dumps(meta))
    return ds_dir


@pytest.fixture(scope="module")
def quant_pipeline(tmp_path_factory):
    """Train tiny model, quantize, compare — shared across all tests in this module."""
    tmp = tmp_path_factory.mktemp("quant_e2e")
    window_size = 16
    ds_dir = _create_tiny_artifact(tmp, n_samples=600, window_size=window_size)
    ckpt_dir = tmp / "CKPT-QUANT-E2E"

    # 1. Train tiny LoRA model
    cfg = TrainingConfig(
        method="lora",
        epochs=2,
        batch_size=16,
        learning_rate=5e-4,
        seed=42,
        lora=LoRAConfig(rank=2, target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]),
        model=ModelConfig(window_size=window_size, embedding_dim=8, num_layers=1,
                          num_heads=2, ffn_dim=16, dropout=0.0),
    )
    splits = DatasetFactory.from_artifact(ds_dir, window_size=window_size)
    trainer = Trainer(
        train_dataset=splits.train,
        val_dataset=splits.validation,
        config=cfg,
        checkpoint_dir=ckpt_dir,
        norm_stats=splits.norm_stats,
    )
    trainer.train()

    # 2. Load FP32 model and quantize
    fp32_model, ckpt_meta = load_model_from_checkpoint(ckpt_dir, device="cpu")
    fp32_model.eval()

    engine = QuantizationEngine()
    quantized_model = engine.quantize(fp32_model)

    # 3. Save INT8 artifact
    int8_dir = tmp / "INT8-QUANT-E2E"
    int8_file, int8_meta = engine.save_quantized(
        quantized_model,
        artifact_dir=int8_dir,
        source_metadata=ckpt_meta,
    )

    # 4. Validate inference
    QuantizationEngine.validate_inference(quantized_model, window_size=window_size)

    # 5. Reload INT8
    loaded_q, loaded_meta = QuantizationEngine.load_quantized(int8_dir)

    # 6. Run comparison
    comparison = run_comparison(
        fp32_checkpoint_dir=ckpt_dir,
        int8_model=quantized_model,
        int8_model_file=int8_file,
        dataset_dir=ds_dir,
        evaluation_batch_size=32,
        benchmark_iterations=20,
        benchmark_warmup=5,
    )

    return {
        "quantized_model": quantized_model,
        "loaded_model": loaded_q,
        "int8_file": int8_file,
        "int8_dir": int8_dir,
        "int8_meta": int8_meta,
        "loaded_meta": loaded_meta,
        "comparison": comparison,
        "window_size": window_size,
        "ckpt_dir": ckpt_dir,
    }


# ---------------------------------------------------------------------------
# Quantization tests
# ---------------------------------------------------------------------------

class TestQuantizedModel:
    def test_quantized_model_is_module(self, quant_pipeline):
        assert isinstance(quant_pipeline["quantized_model"], torch.nn.Module)

    def test_quantized_forward_pass_shape(self, quant_pipeline):
        q = quant_pipeline["quantized_model"]
        ws = quant_pipeline["window_size"]
        x = torch.randn(4, ws, 1)
        with torch.no_grad():
            out = q(x)
        assert out.logits.shape == (4, 7)

    def test_quantized_logits_no_nan(self, quant_pipeline):
        q = quant_pipeline["quantized_model"]
        ws = quant_pipeline["window_size"]
        x = torch.randn(4, ws, 1)
        with torch.no_grad():
            out = q(x)
        assert not torch.isnan(out.logits).any()
        assert not torch.isinf(out.logits).any()


class TestArtifactPersistence:
    def test_int8_file_exists(self, quant_pipeline):
        assert quant_pipeline["int8_file"].exists()

    def test_int8_file_nonzero_size(self, quant_pipeline):
        assert quant_pipeline["int8_file"].stat().st_size > 0

    def test_metadata_json_exists(self, quant_pipeline):
        assert (quant_pipeline["int8_dir"] / "metadata.json").exists()

    def test_metadata_has_model_config(self, quant_pipeline):
        meta = quant_pipeline["int8_meta"]
        assert "model_config" in meta
        assert meta["model_config"].get("window_size") == quant_pipeline["window_size"]

    def test_metadata_has_backend(self, quant_pipeline):
        from app.ml.quantization.backend_detect import _KNOWN_BACKENDS
        meta = quant_pipeline["int8_meta"]
        assert meta.get("backend") in _KNOWN_BACKENDS


class TestReloadedModel:
    def test_loaded_is_module(self, quant_pipeline):
        assert isinstance(quant_pipeline["loaded_model"], torch.nn.Module)

    def test_loaded_forward_pass(self, quant_pipeline):
        q = quant_pipeline["loaded_model"]
        ws = quant_pipeline["window_size"]
        x = torch.randn(2, ws, 1)
        with torch.no_grad():
            out = q(x)
        assert out.logits.shape == (2, 7)

    def test_loaded_meta_has_norm_stats(self, quant_pipeline):
        meta = quant_pipeline["loaded_meta"]
        assert "norm_stats" in meta


class TestComparison:
    def test_fp32_f1_in_range(self, quant_pipeline):
        cmp = quant_pipeline["comparison"]
        assert 0.0 <= cmp.fp32_macro_f1 <= 1.0

    def test_int8_f1_in_range(self, quant_pipeline):
        cmp = quant_pipeline["comparison"]
        assert 0.0 <= cmp.int8_macro_f1 <= 1.0

    def test_f1_delta_consistent(self, quant_pipeline):
        cmp = quant_pipeline["comparison"]
        expected = cmp.fp32_macro_f1 - cmp.int8_macro_f1
        assert abs(cmp.f1_delta - expected) < 1e-6

    def test_fp32_size_positive(self, quant_pipeline):
        cmp = quant_pipeline["comparison"]
        assert cmp.fp32_size_bytes > 0

    def test_int8_size_positive(self, quant_pipeline):
        cmp = quant_pipeline["comparison"]
        assert cmp.int8_size_bytes > 0

    def test_size_reduction_ratio_positive(self, quant_pipeline):
        cmp = quant_pipeline["comparison"]
        assert cmp.size_reduction_ratio > 0.0

    def test_fp32_latency_positive(self, quant_pipeline):
        cmp = quant_pipeline["comparison"]
        assert cmp.fp32_latency_ms > 0.0

    def test_int8_latency_positive(self, quant_pipeline):
        cmp = quant_pipeline["comparison"]
        assert cmp.int8_latency_ms > 0.0

    def test_latency_speedup_positive(self, quant_pipeline):
        cmp = quant_pipeline["comparison"]
        assert cmp.latency_speedup > 0.0

    def test_n_test_windows_positive(self, quant_pipeline):
        cmp = quant_pipeline["comparison"]
        assert cmp.n_test_windows > 0

    def test_to_dict_serializable(self, quant_pipeline):
        import json as _json
        cmp = quant_pipeline["comparison"]
        d = cmp.to_dict()
        _json.dumps(d)
        for key in ("fp32_macro_f1", "int8_macro_f1", "f1_delta",
                    "fp32_size_bytes", "int8_size_bytes", "size_reduction_ratio",
                    "fp32_latency_ms", "int8_latency_ms", "latency_speedup",
                    "n_test_windows"):
            assert key in d
