"""Latency measurement utilities for the BenchmarkEngine."""
from __future__ import annotations

import numpy as np


def compute_latency_stats(latencies_ms: list[float]) -> dict:
    """Compute percentile statistics from a list of latency samples.

    Args:
        latencies_ms: Measured inference latencies in milliseconds.

    Returns:
        dict with p50, p90, p95, p99, min, max, mean, count — all in ms.
    """
    if not latencies_ms:
        raise ValueError("latencies_ms must not be empty")

    arr = np.array(latencies_ms, dtype=np.float64)
    return {
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "p99": round(float(np.percentile(arr, 99)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "mean": round(float(np.mean(arr)), 4),
        "count": len(latencies_ms),
    }
