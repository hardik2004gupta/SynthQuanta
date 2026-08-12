"""In-memory runtime telemetry — tracks request counts and latency.

Intentionally lightweight: no external infrastructure required.
All state is process-local and resets when the process restarts.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TelemetrySummary:
    """Point-in-time snapshot of runtime telemetry."""

    request_count: int
    success_count: int
    error_count: int
    last_latency_ms: Optional[float]
    mean_latency_ms: Optional[float]
    p50_latency_ms: Optional[float]
    p95_latency_ms: Optional[float]
    p99_latency_ms: Optional[float]

    def to_dict(self) -> dict:
        return {
            "request_count": self.request_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "last_latency_ms": self.last_latency_ms,
            "mean_latency_ms": self.mean_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
        }


class RuntimeTelemetry:
    """Thread-safe in-memory telemetry collector."""

    _MAX_SAMPLES = 10_000   # retain at most 10k latency samples

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._request_count = 0
        self._success_count = 0
        self._error_count = 0
        self._latencies: list[float] = []
        self._last_latency: float | None = None

    def record(self, latency_ms: float, success: bool) -> None:
        """Record one completed inference call."""
        with self._lock:
            self._request_count += 1
            if success:
                self._success_count += 1
                self._last_latency = latency_ms
                if len(self._latencies) < self._MAX_SAMPLES:
                    self._latencies.append(latency_ms)
            else:
                self._error_count += 1

    def to_summary(self) -> TelemetrySummary:
        """Return a point-in-time snapshot (thread-safe copy)."""
        with self._lock:
            rc = self._request_count
            sc = self._success_count
            ec = self._error_count
            last = self._last_latency
            lats = list(self._latencies)

        if lats:
            arr = np.array(lats)
            mean_ms = float(np.mean(arr))
            p50 = float(np.percentile(arr, 50))
            p95 = float(np.percentile(arr, 95))
            p99 = float(np.percentile(arr, 99))
        else:
            mean_ms = p50 = p95 = p99 = None

        return TelemetrySummary(
            request_count=rc,
            success_count=sc,
            error_count=ec,
            last_latency_ms=last,
            mean_latency_ms=mean_ms,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
        )

    def reset(self) -> None:
        """Reset all counters — called when a new model is loaded."""
        with self._lock:
            self._request_count = 0
            self._success_count = 0
            self._error_count = 0
            self._latencies = []
            self._last_latency = None
