"""Runtime preprocessing pipeline — converts raw sensor input to model tensors.

Must match the normalization used during training exactly.
Norm stats (mean, std) are loaded from the model artifact metadata.
"""
from __future__ import annotations

import numpy as np
import torch


class PreprocessingError(ValueError):
    """Raised when input validation or normalization fails."""


class Preprocessor:
    """Validates and normalizes a sensor window into a model-ready tensor.

    Normalization: (x - mean) / max(std, 1e-8)
    Output shape: (1, window_size, 1) — ready for SensorTransformer.forward()
    """

    def __init__(self, window_size: int, mean: float, std: float) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self._window_size = window_size
        self._mean = float(mean)
        self._std = max(float(std), 1e-8)

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        return self._std

    def preprocess(self, values: list[float] | np.ndarray) -> torch.Tensor:
        """Validate and normalize a single sensor window.

        Args:
            values: Exactly window_size float values.

        Returns:
            Tensor of shape (1, window_size, 1).

        Raises:
            PreprocessingError: if length, type, or numerical validity fails.
        """
        if hasattr(values, "__len__"):
            n = len(values)
        else:
            raise PreprocessingError("values must be a sequence (list or ndarray)")

        if n != self._window_size:
            raise PreprocessingError(
                f"Expected {self._window_size} values, got {n}"
            )

        arr = np.asarray(values, dtype=np.float32)

        if np.any(np.isnan(arr)):
            raise PreprocessingError("Input contains NaN values")
        if np.any(np.isinf(arr)):
            raise PreprocessingError("Input contains Inf values")

        arr = (arr - self._mean) / self._std

        # shape: (1, window_size, 1)
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(-1)

    def preprocess_batch(self, batch: list[list[float]] | list[np.ndarray]) -> torch.Tensor:
        """Validate and normalize a batch of sensor windows.

        Args:
            batch: List of N windows, each with window_size values.

        Returns:
            Tensor of shape (N, window_size, 1).

        Raises:
            PreprocessingError: on any invalid window.
        """
        if not batch:
            raise PreprocessingError("Batch must contain at least one window")
        tensors = [self.preprocess(w).squeeze(0) for w in batch]  # each (window_size, 1)
        return torch.stack(tensors, dim=0)  # (N, window_size, 1)
