"""QuantizationConfig — configuration for the INT8 quantization pass."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QuantizationConfig:
    """Configuration for a quantization job.

    method must be "dynamic_int8" — the only supported path for MVP (FP32 → INT8).
    FP16 is a post-MVP extension (CLAUDE.md §25 resolution).
    """

    method: str = "dynamic_int8"
    # For dynamic quantization, calibration is not required.
    # Stored as metadata for audit purposes only.
    n_calibration_windows: int = 100
    # Latency benchmark parameters (used in comparison step)
    benchmark_iterations: int = 50
    benchmark_warmup: int = 10
    benchmark_batch_size: int = 1

    def validate(self) -> None:
        if self.method != "dynamic_int8":
            raise ValueError(
                f"Unsupported quantization method: {self.method!r}. "
                "MVP supports 'dynamic_int8' only (CLAUDE.md §25)."
            )
        if self.benchmark_iterations < 1:
            raise ValueError("benchmark_iterations must be >= 1")
        if self.benchmark_warmup < 0:
            raise ValueError("benchmark_warmup must be >= 0")

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "n_calibration_windows": self.n_calibration_windows,
            "benchmark_iterations": self.benchmark_iterations,
            "benchmark_warmup": self.benchmark_warmup,
            "benchmark_batch_size": self.benchmark_batch_size,
        }
