"""BenchmarkEngine — benchmarks SQRuntime, not raw model internals.

Architecture (§29–§37 of Phase 6 spec):
    BenchmarkEngine
          ↓
    SQRuntime.predict() / SQRuntime.predict_batch()

Rules:
  - All latency numbers come from actual timed calls.
  - Warmup iterations are EXCLUDED from statistics.
  - Memory measured via tracemalloc.
  - Failure on any batch size → BatchResult.status="FAILED" + diagnostic.
  - Never fabricate performance numbers.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.benchmarking.latency import compute_latency_stats
from app.benchmarking.memory import measure_peak_memory_mb

logger = logging.getLogger(__name__)

# Default batch sizes to test (§35 Phase 6 spec)
DEFAULT_BATCH_SIZES: list[int] = [1, 4, 8, 16]


@dataclass
class BatchResult:
    """Benchmark result for one (runtime, batch_size) combination."""

    batch_size: int
    iterations: int
    warmup_count: int
    latency_stats: dict                 # {p50, p90, p95, p99, min, max, mean, count}
    throughput_rps: float               # requests/second (one request = one batch)
    throughput_sps: float               # samples/second = rps * batch_size
    memory_peak_mb: Optional[float]
    duration_seconds: float
    status: str = "COMPLETED"
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "iterations": self.iterations,
            "warmup_count": self.warmup_count,
            "latency_stats": self.latency_stats,
            "throughput_rps": round(self.throughput_rps, 4),
            "throughput_sps": round(self.throughput_sps, 4),
            "memory_peak_mb": self.memory_peak_mb,
            "duration_seconds": round(self.duration_seconds, 4),
            "status": self.status,
            "error": self.error,
        }


@dataclass
class BenchmarkRun:
    """Full benchmark run result across multiple batch sizes."""

    runtime_variant: str           # "fp32" | "int8"
    device: str
    backend: Optional[str]
    batch_results: list[BatchResult]
    total_duration_seconds: float

    @property
    def primary_result(self) -> Optional[BatchResult]:
        """Return the batch_size=1 result (or first completed result)."""
        for r in self.batch_results:
            if r.batch_size == 1 and r.status == "COMPLETED":
                return r
        for r in self.batch_results:
            if r.status == "COMPLETED":
                return r
        return None

    def to_dict(self) -> dict:
        return {
            "runtime_variant": self.runtime_variant,
            "device": self.device,
            "backend": self.backend,
            "batch_results": [r.to_dict() for r in self.batch_results],
            "total_duration_seconds": round(self.total_duration_seconds, 4),
        }


class BenchmarkEngine:
    """Benchmarks a SQRuntime over multiple batch sizes.

    Must be instantiated without the runtime; the runtime is passed at
    benchmark time so the engine can be reused across different runtimes.
    """

    def _make_windows(
        self, window_size: int, count: int, seed: int
    ) -> list[list[float]]:
        """Generate deterministic synthetic windows for benchmarking."""
        rng = np.random.default_rng(seed)
        return [
            list(rng.normal(0.0, 1.0, window_size).astype(np.float64))
            for _ in range(count)
        ]

    def _warmup(self, runtime, windows: list[list[float]], warmup_count: int, batch_size: int) -> None:
        """Run warmup iterations — NOT counted in results."""
        for i in range(warmup_count):
            if batch_size == 1:
                runtime.predict(windows[i % len(windows)])
            else:
                batch = [windows[(i * batch_size + j) % len(windows)] for j in range(batch_size)]
                runtime.predict_batch(batch)

    def run_single_batch_size(
        self,
        runtime,
        batch_size: int,
        iterations: int,
        warmup_count: int,
        window_size: int,
        seed: int,
    ) -> BatchResult:
        """Benchmark one (runtime, batch_size) combination.

        Args:
            runtime:      SQRuntime instance (must be in READY state).
            batch_size:   Number of windows per inference call.
            iterations:   Number of measured inference calls (after warmup).
            warmup_count: Warmup calls excluded from statistics.
            window_size:  Length of each sensor window.
            seed:         RNG seed for deterministic window generation.

        Returns:
            BatchResult with actual latency stats, throughput, and memory.
        """
        # Generate enough windows for warmup + measurement
        total_needed = (warmup_count + iterations) * batch_size
        windows = self._make_windows(window_size, max(total_needed, batch_size * 2), seed)

        run_start = time.perf_counter()

        try:
            self._warmup(runtime, windows, warmup_count, batch_size)
        except Exception as exc:
            return BatchResult(
                batch_size=batch_size,
                iterations=0,
                warmup_count=warmup_count,
                latency_stats={},
                throughput_rps=0.0,
                throughput_sps=0.0,
                memory_peak_mb=None,
                duration_seconds=time.perf_counter() - run_start,
                status="FAILED",
                error=f"Warmup failed: {exc}",
            )

        latencies: list[float] = []
        mem_result: dict = {"peak_mb": None}

        try:
            with measure_peak_memory_mb() as mem_result:
                t_measure_start = time.perf_counter()
                for i in range(iterations):
                    if batch_size == 1:
                        result = runtime.predict(windows[i % len(windows)])
                        latencies.append(result.latency_ms)
                    else:
                        batch = [windows[(i * batch_size + j) % len(windows)] for j in range(batch_size)]
                        t0 = time.perf_counter()
                        runtime.predict_batch(batch)
                        latencies.append((time.perf_counter() - t0) * 1000.0)
                elapsed = time.perf_counter() - t_measure_start
        except Exception as exc:
            return BatchResult(
                batch_size=batch_size,
                iterations=len(latencies),
                warmup_count=warmup_count,
                latency_stats=compute_latency_stats(latencies) if latencies else {},
                throughput_rps=0.0,
                throughput_sps=0.0,
                memory_peak_mb=mem_result.get("peak_mb"),
                duration_seconds=time.perf_counter() - run_start,
                status="FAILED",
                error=f"Measurement failed at iteration {len(latencies)}: {exc}",
            )

        stats = compute_latency_stats(latencies)
        throughput_rps = iterations / max(elapsed, 1e-9)
        throughput_sps = throughput_rps * batch_size
        total_duration = time.perf_counter() - run_start

        logger.info(
            "Benchmark batch_size=%d: P95=%.2f ms, throughput=%.1f req/s",
            batch_size, stats["p95"], throughput_rps,
        )

        return BatchResult(
            batch_size=batch_size,
            iterations=iterations,
            warmup_count=warmup_count,
            latency_stats=stats,
            throughput_rps=round(throughput_rps, 4),
            throughput_sps=round(throughput_sps, 4),
            memory_peak_mb=mem_result.get("peak_mb"),
            duration_seconds=round(total_duration, 4),
            status="COMPLETED",
        )

    def run(
        self,
        runtime,
        batch_sizes: list[int] | None = None,
        iterations: int = 50,
        warmup_count: int = 10,
        seed: int = 42,
    ) -> BenchmarkRun:
        """Run benchmark across all requested batch sizes.

        Args:
            runtime:      SQRuntime instance in READY state.
            batch_sizes:  List of batch sizes to test. Defaults to [1, 4, 8, 16].
            iterations:   Measured calls per batch size.
            warmup_count: Warmup calls per batch size.
            seed:         RNG seed for input generation.

        Returns:
            BenchmarkRun with results for all batch sizes.
        """
        if not runtime.is_ready():
            raise ValueError("SQRuntime must be in READY state before benchmarking")

        health = runtime.health()
        runtime_variant = health.precision or "unknown"
        device = health.device
        backend = health.backend

        # Infer window_size from the preprocessor
        try:
            _model, preprocessor = runtime._require_ready()
            window_size = preprocessor.window_size
        except Exception as exc:
            raise ValueError(f"Cannot determine window_size from runtime: {exc}") from exc

        sizes = batch_sizes or DEFAULT_BATCH_SIZES
        run_start = time.perf_counter()
        batch_results: list[BatchResult] = []

        for bs in sizes:
            logger.info(
                "Benchmarking batch_size=%d (%d warmup + %d measured)...",
                bs, warmup_count, iterations,
            )
            result = self.run_single_batch_size(
                runtime=runtime,
                batch_size=bs,
                iterations=iterations,
                warmup_count=warmup_count,
                window_size=window_size,
                seed=seed + bs,  # vary seed per batch size
            )
            batch_results.append(result)

        total_duration = time.perf_counter() - run_start
        return BenchmarkRun(
            runtime_variant=runtime_variant,
            device=device,
            backend=backend,
            batch_results=batch_results,
            total_duration_seconds=round(total_duration, 4),
        )
