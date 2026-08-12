"""Memory measurement utilities for the BenchmarkEngine.

Uses tracemalloc (stdlib, always available) to measure Python heap
allocations during benchmarking.  For CPU/local MVP this gives a useful
proxy for model memory consumption.
"""
from __future__ import annotations

import tracemalloc
from contextlib import contextmanager
from typing import Generator


@contextmanager
def measure_peak_memory_mb() -> Generator[dict, None, None]:
    """Context manager that measures peak Python heap allocation.

    Usage:
        with measure_peak_memory_mb() as mem_result:
            # ... run inference ...
        peak_mb = mem_result["peak_mb"]

    Yields a mutable dict so the caller can read the result after exit.
    """
    result: dict = {"peak_mb": None, "method": "tracemalloc"}
    tracemalloc.start()
    try:
        yield result
    finally:
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result["peak_mb"] = round(peak / 1024 / 1024, 4)


def get_model_size_mb(artifact_path_bytes: int) -> float:
    """Convert artifact size in bytes to megabytes."""
    return round(artifact_path_bytes / 1024 / 1024, 4)
