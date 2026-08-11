"""Unit tests for runtime Preprocessor."""
import numpy as np
import pytest
import torch

from app.runtime.preprocessing import Preprocessor, PreprocessingError


class TestPreprocessor:
    def test_basic_output_shape(self):
        p = Preprocessor(window_size=16, mean=0.0, std=1.0)
        t = p.preprocess([0.0] * 16)
        assert t.shape == (1, 16, 1)

    def test_output_is_float_tensor(self):
        p = Preprocessor(window_size=8, mean=0.0, std=1.0)
        t = p.preprocess([1.0] * 8)
        assert t.dtype == torch.float32

    def test_normalization_applied(self):
        p = Preprocessor(window_size=4, mean=2.0, std=2.0)
        t = p.preprocess([2.0, 4.0, 0.0, 2.0])
        expected = np.array([(2 - 2) / 2, (4 - 2) / 2, (0 - 2) / 2, (2 - 2) / 2], dtype=np.float32)
        np.testing.assert_allclose(t.squeeze().numpy(), expected, rtol=1e-5)

    def test_wrong_length_raises(self):
        p = Preprocessor(window_size=16, mean=0.0, std=1.0)
        with pytest.raises(PreprocessingError, match="Expected 16"):
            p.preprocess([0.0] * 10)

    def test_nan_raises(self):
        p = Preprocessor(window_size=4, mean=0.0, std=1.0)
        with pytest.raises(PreprocessingError, match="NaN"):
            p.preprocess([float("nan"), 1.0, 2.0, 3.0])

    def test_inf_raises(self):
        p = Preprocessor(window_size=4, mean=0.0, std=1.0)
        with pytest.raises(PreprocessingError, match="Inf"):
            p.preprocess([float("inf"), 1.0, 2.0, 3.0])

    def test_accepts_numpy_array(self):
        p = Preprocessor(window_size=8, mean=0.0, std=1.0)
        arr = np.zeros(8, dtype=np.float32)
        t = p.preprocess(arr)
        assert t.shape == (1, 8, 1)

    def test_std_too_small_uses_clamp(self):
        p = Preprocessor(window_size=4, mean=0.0, std=1e-15)
        t = p.preprocess([0.0] * 4)
        assert not torch.isnan(t).any()
        assert not torch.isinf(t).any()

    def test_preprocess_batch_shape(self):
        p = Preprocessor(window_size=8, mean=0.0, std=1.0)
        batch = [[0.0] * 8, [1.0] * 8, [2.0] * 8]
        t = p.preprocess_batch(batch)
        assert t.shape == (3, 8, 1)

    def test_preprocess_batch_empty_raises(self):
        p = Preprocessor(window_size=8, mean=0.0, std=1.0)
        with pytest.raises(PreprocessingError, match="at least one"):
            p.preprocess_batch([])
