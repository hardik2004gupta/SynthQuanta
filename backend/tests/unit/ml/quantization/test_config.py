"""Unit tests for QuantizationConfig."""
import pytest

from app.ml.quantization.config import QuantizationConfig


class TestQuantizationConfig:
    def test_default_method_is_dynamic_int8(self):
        cfg = QuantizationConfig()
        assert cfg.method == "dynamic_int8"

    def test_validate_passes_for_dynamic_int8(self):
        cfg = QuantizationConfig(method="dynamic_int8")
        cfg.validate()  # must not raise

    def test_validate_rejects_fp16(self):
        cfg = QuantizationConfig(method="fp16")
        with pytest.raises(ValueError, match="MVP supports 'dynamic_int8' only"):
            cfg.validate()

    def test_validate_rejects_static_int8(self):
        cfg = QuantizationConfig(method="static_int8")
        with pytest.raises(ValueError):
            cfg.validate()

    def test_validate_rejects_zero_iterations(self):
        cfg = QuantizationConfig(benchmark_iterations=0)
        with pytest.raises(ValueError, match="benchmark_iterations"):
            cfg.validate()

    def test_validate_allows_zero_warmup(self):
        cfg = QuantizationConfig(benchmark_warmup=0)
        cfg.validate()  # 0 warmup is allowed

    def test_to_dict_has_all_fields(self):
        cfg = QuantizationConfig()
        d = cfg.to_dict()
        for key in ("method", "n_calibration_windows", "benchmark_iterations",
                    "benchmark_warmup", "benchmark_batch_size"):
            assert key in d

    def test_to_dict_method_matches(self):
        cfg = QuantizationConfig(method="dynamic_int8", benchmark_iterations=30)
        d = cfg.to_dict()
        assert d["method"] == "dynamic_int8"
        assert d["benchmark_iterations"] == 30
