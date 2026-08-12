"""Unit tests for QuantizationEngine — quantize, save, load, validate."""
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from app.ml.models.sensor_transformer import ModelConfig, SensorTransformer
from app.ml.quantization.engine import QuantizationEngine, QuantizationError
from app.ml.quantization.config import QuantizationConfig


def _tiny_model(window_size: int = 16) -> SensorTransformer:
    cfg = ModelConfig(window_size=window_size, embedding_dim=8, num_layers=1,
                      num_heads=2, ffn_dim=16, dropout=0.0)
    return SensorTransformer(cfg)


class TestQuantize:
    def test_returns_module(self, tmp_path):
        model = _tiny_model()
        engine = QuantizationEngine()
        q = engine.quantize(model)
        assert isinstance(q, nn.Module)

    def test_quantized_model_not_same_object(self, tmp_path):
        model = _tiny_model()
        engine = QuantizationEngine()
        q = engine.quantize(model)
        assert q is not model

    def test_original_model_unmodified(self, tmp_path):
        model = _tiny_model()
        original_type = type(list(model.children())[0])
        engine = QuantizationEngine()
        engine.quantize(model)
        # Original model's first child should still be the same type
        assert type(list(model.children())[0]) is original_type

    def test_quantized_forward_pass_works(self, tmp_path):
        model = _tiny_model(window_size=16)
        engine = QuantizationEngine()
        q = engine.quantize(model)
        q.eval()
        x = torch.randn(2, 16, 1)
        with torch.no_grad():
            output = q(x)
        assert output.logits.shape == (2, 7)

    def test_backend_property(self):
        from app.ml.quantization.backend_detect import _KNOWN_BACKENDS
        engine = QuantizationEngine()
        assert engine.backend in _KNOWN_BACKENDS


class TestSaveLoad:
    def test_save_creates_files(self, tmp_path):
        model = _tiny_model()
        engine = QuantizationEngine()
        q = engine.quantize(model)
        artifact_dir = tmp_path / "q_model"
        source_meta = {"config": {"model": {"window_size": 16, "embedding_dim": 8,
                                              "num_layers": 1, "num_heads": 2,
                                              "ffn_dim": 16, "dropout": 0.0}},
                       "norm_stats": {"mean": 0.0, "std": 1.0}}
        model_file, _ = engine.save_quantized(q, artifact_dir, source_meta)
        assert model_file.exists()
        assert (artifact_dir / "metadata.json").exists()

    def test_saved_metadata_has_required_fields(self, tmp_path):
        model = _tiny_model()
        engine = QuantizationEngine()
        q = engine.quantize(model)
        artifact_dir = tmp_path / "q_model"
        source_meta = {"config": {"model": {"window_size": 16, "embedding_dim": 8,
                                              "num_layers": 1, "num_heads": 2,
                                              "ffn_dim": 16, "dropout": 0.0}},
                       "norm_stats": {"mean": 0.0, "std": 1.0}}
        _, meta = engine.save_quantized(q, artifact_dir, source_meta)
        for key in ("quantization_method", "backend", "model_config", "norm_stats"):
            assert key in meta

    def test_load_quantized_returns_module(self, tmp_path):
        model = _tiny_model()
        engine = QuantizationEngine()
        q = engine.quantize(model)
        artifact_dir = tmp_path / "q_model"
        source_meta = {"config": {"model": {"window_size": 16, "embedding_dim": 8,
                                              "num_layers": 1, "num_heads": 2,
                                              "ffn_dim": 16, "dropout": 0.0}},
                       "norm_stats": {"mean": 0.0, "std": 1.0}}
        engine.save_quantized(q, artifact_dir, source_meta)
        loaded, meta = QuantizationEngine.load_quantized(artifact_dir)
        assert isinstance(loaded, nn.Module)
        assert isinstance(meta, dict)

    def test_loaded_model_forward_pass(self, tmp_path):
        window_size = 16
        model = _tiny_model(window_size)
        engine = QuantizationEngine()
        q = engine.quantize(model)
        artifact_dir = tmp_path / "q_model"
        source_meta = {"config": {"model": {"window_size": window_size,
                                              "embedding_dim": 8, "num_layers": 1,
                                              "num_heads": 2, "ffn_dim": 16,
                                              "dropout": 0.0}},
                       "norm_stats": {"mean": 0.0, "std": 1.0}}
        engine.save_quantized(q, artifact_dir, source_meta)
        loaded, _ = QuantizationEngine.load_quantized(artifact_dir)
        loaded.eval()
        x = torch.randn(1, window_size, 1)
        with torch.no_grad():
            output = loaded(x)
        assert output.logits.shape == (1, 7)

    def test_load_missing_model_file_raises(self, tmp_path):
        artifact_dir = tmp_path / "empty"
        artifact_dir.mkdir()
        with pytest.raises(QuantizationError, match="not found"):
            QuantizationEngine.load_quantized(artifact_dir)

    def test_load_missing_metadata_raises(self, tmp_path):
        artifact_dir = tmp_path / "only_weights"
        artifact_dir.mkdir()
        (artifact_dir / "model_quantized.pt").write_bytes(b"x")
        with pytest.raises(QuantizationError, match="metadata"):
            QuantizationEngine.load_quantized(artifact_dir)


class TestValidateInference:
    def test_valid_model_does_not_raise(self):
        model = _tiny_model(window_size=16)
        engine = QuantizationEngine()
        q = engine.quantize(model)
        QuantizationEngine.validate_inference(q, window_size=16)  # must not raise

    def test_file_size_returns_int(self, tmp_path):
        model = _tiny_model()
        engine = QuantizationEngine()
        q = engine.quantize(model)
        artifact_dir = tmp_path / "q_model"
        source_meta = {"config": {"model": {"window_size": 16, "embedding_dim": 8,
                                              "num_layers": 1, "num_heads": 2,
                                              "ffn_dim": 16, "dropout": 0.0}},
                       "norm_stats": {"mean": 0.0, "std": 1.0}}
        model_file, _ = engine.save_quantized(q, artifact_dir, source_meta)
        size = QuantizationEngine.file_size_bytes(model_file)
        assert isinstance(size, int)
        assert size > 0

    def test_file_size_missing_returns_zero(self, tmp_path):
        missing = tmp_path / "nonexistent.pt"
        assert QuantizationEngine.file_size_bytes(missing) == 0
