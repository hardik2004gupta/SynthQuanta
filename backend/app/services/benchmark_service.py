"""BenchmarkService — orchestrates benchmark jobs.

Architecture:
    POST /benchmarks/run  → BenchmarkService.start_benchmark_job()
                            → validate artifact
                            → create Benchmark record (PENDING)
                            → spawn background thread
                            → return immediately with benchmark_id

    background thread     → load artifact into a fresh SQRuntime
                          → BenchmarkEngine.run()
                          → persist results
                          → mark COMPLETED | FAILED

    GET /benchmarks/{id}  → BenchmarkService.get_benchmark()
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from sqlalchemy.orm import Session

from app.benchmarking.engine import BenchmarkEngine
from app.core.config import get_settings
from app.db.repositories.benchmark_repository import BenchmarkRepository
from app.db.repositories.model_repository import ModelRepository
from app.db.repositories.quantization_repository import QuantizationRepository
from app.db.session import SessionLocal
from app.ml.training.reproducibility import collect_hardware_info
from app.runtime.runtime import SQRuntime
from app.schemas.benchmark import (
    BatchResult,
    BenchmarkResponse,
    BenchmarkRunRequest,
    BenchmarkStartResponse,
)
from app.services.artifact_store import ArtifactStore

logger = logging.getLogger(__name__)
_settings = get_settings()


class BenchmarkServiceError(Exception):
    """Raised by BenchmarkService for validated failures."""


def _next_bench_human_id(repo: BenchmarkRepository) -> str:
    n = repo.count() + 1
    return f"BENCH-{n:04d}"


class BenchmarkService:
    """Business logic for benchmark job lifecycle."""

    def __init__(self, db: Session) -> None:
        self._db = db
        self._bench_repo = BenchmarkRepository(db)
        self._model_repo = ModelRepository(db)
        self._quant_repo = QuantizationRepository(db)
        self._store = ArtifactStore(root=_settings.artifact_root_path)

    # ------------------------------------------------------------------
    # Start a benchmark job (non-blocking)
    # ------------------------------------------------------------------

    def start_benchmark_job(self, request: BenchmarkRunRequest) -> BenchmarkStartResponse:
        """Validate, create the Benchmark record, and start the background thread."""
        # Resolve model + artifact
        model_id, artifact_path, precision = self._resolve_artifact(request)
        artifact_dir = _settings.artifact_root_path / artifact_path

        human_id = _next_bench_human_id(self._bench_repo)

        record = self._bench_repo.create(
            human_id=human_id,
            model_id=model_id,
            runtime_variant=precision,
            device=collect_hardware_info()["device"],
            batch_size=request.batch_sizes[0] if request.batch_sizes else 1,
            iterations=request.iterations,
            warmup_count=request.warmup,
            status="PENDING",
        )
        self._bench_repo.commit()
        benchmark_id = str(record.id)

        logger.info(
            "Benchmark job created: %s (%s) | model=%s | precision=%s",
            human_id, benchmark_id, model_id, precision,
        )

        thread = threading.Thread(
            target=_run_benchmark_thread,
            args=(
                benchmark_id,
                model_id,
                artifact_path,
                precision,
                request.batch_sizes,
                request.iterations,
                request.warmup,
                request.seed,
            ),
            name=f"bench-{human_id}",
            daemon=True,
        )
        thread.start()

        return BenchmarkStartResponse(
            benchmark_id=benchmark_id,
            human_id=human_id,
            model_id=model_id,
            runtime_variant=precision,
            status="PENDING",
            message=(
                f"Benchmark job {human_id} started. "
                f"Poll GET /api/v1/benchmarks/{benchmark_id} for status."
            ),
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_benchmark(self, benchmark_id: str) -> BenchmarkResponse:
        record = self._bench_repo.get_by_id(benchmark_id)
        if record is None:
            raise BenchmarkServiceError(f"Benchmark {benchmark_id!r} not found.")
        return _to_response(record)

    def list_benchmarks(self, limit: int = 100, offset: int = 0) -> list[BenchmarkResponse]:
        rows, _ = self._bench_repo.list_all(limit=limit, offset=offset)
        return [_to_response(r) for r in rows]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_artifact(self, request: BenchmarkRunRequest) -> tuple[str, str, str]:
        """Return (model_id, artifact_path, precision) or raise."""
        if request.quantization_id:
            quant = self._quant_repo.get_by_id(request.quantization_id)
            if quant is None:
                raise BenchmarkServiceError(f"Quantization {request.quantization_id!r} not found.")
            if quant.status != "COMPLETED":
                raise BenchmarkServiceError(
                    f"Quantization {quant.human_id} is not COMPLETED (current: {quant.status})."
                )
            if not quant.artifact_path:
                raise BenchmarkServiceError(
                    f"Quantization {quant.human_id} has no artifact_path."
                )
            model_id = str(quant.quantized_model_id or quant.source_model_id)
            return model_id, quant.artifact_path, "int8"

        elif request.model_id:
            model = self._model_repo.get_by_id(request.model_id)
            if model is None:
                raise BenchmarkServiceError(f"Model {request.model_id!r} not found.")
            if model.status != "COMPLETED":
                raise BenchmarkServiceError(
                    f"Model {model.human_id} is not COMPLETED (current: {model.status})."
                )
            return str(model.id), model.artifact_path, model.precision

        else:
            raise BenchmarkServiceError(
                "Either model_id or quantization_id must be provided."
            )


# ---------------------------------------------------------------------------
# Background thread
# ---------------------------------------------------------------------------

def _run_benchmark_thread(
    benchmark_id: str,
    model_id: str,
    artifact_path: str,
    precision: str,
    batch_sizes: list[int],
    iterations: int,
    warmup: int,
    seed: int,
) -> None:
    db = SessionLocal()
    bench_repo = BenchmarkRepository(db)

    try:
        record = bench_repo.get_by_id(benchmark_id)
        if record is None:
            logger.error("Benchmark record %s not found in thread", benchmark_id)
            return

        bench_repo.update(record, status="RUNNING")
        bench_repo.commit()

        _do_benchmark(
            record, bench_repo, artifact_path, precision,
            model_id, batch_sizes, iterations, warmup, seed,
        )
    except Exception as exc:
        logger.exception("Unexpected error in benchmark thread %s", benchmark_id)
        try:
            record = bench_repo.get_by_id(benchmark_id)
            if record:
                bench_repo.update(record, status="FAILED", error=str(exc))
                bench_repo.commit()
        except Exception:
            pass
    finally:
        db.close()


def _do_benchmark(
    record,
    bench_repo: BenchmarkRepository,
    artifact_path: str,
    precision: str,
    model_id: str,
    batch_sizes: list[int],
    iterations: int,
    warmup: int,
    seed: int,
) -> None:
    artifact_dir = _settings.artifact_root_path / artifact_path
    hardware_info = collect_hardware_info()

    runtime = SQRuntime()
    try:
        runtime.load(
            artifact_dir=artifact_dir,
            precision=precision,
            model_id=model_id,
            artifact_path=artifact_path,
        )
    except Exception as exc:
        bench_repo.update(
            record,
            status="FAILED",
            error=f"Runtime load failed: {exc}",
            hardware_info=hardware_info,
        )
        bench_repo.commit()
        return

    engine = BenchmarkEngine()
    try:
        bench_run = engine.run(
            runtime=runtime,
            batch_sizes=batch_sizes,
            iterations=iterations,
            warmup_count=warmup,
            seed=seed,
        )
    except Exception as exc:
        bench_repo.update(
            record,
            status="FAILED",
            error=f"BenchmarkEngine.run() failed: {exc}",
            hardware_info=hardware_info,
        )
        bench_repo.commit()
        return

    # Extract primary stats (batch_size=1 or first completed)
    primary = bench_run.primary_result
    latency_metrics = primary.latency_stats if primary else None
    throughput = primary.throughput_rps if primary else None
    memory = {"peak_mb": primary.memory_peak_mb, "method": "tracemalloc"} if primary else None

    bench_results_json = [r.to_dict() for r in bench_run.batch_results]

    # Persist benchmark artifact
    store = ArtifactStore(root=_settings.artifact_root_path)
    bench_dir = store.benchmark_path(record.human_id)
    store.write_metadata(bench_dir, {
        "human_id": record.human_id,
        "benchmark_id": str(record.id),
        "model_id": model_id,
        "precision": precision,
        "batch_results": bench_results_json,
        "hardware_info": hardware_info,
    })
    artifact_rel = store.relative_path(bench_dir)

    bench_repo.update(
        record,
        status="COMPLETED",
        latency_metrics=latency_metrics,
        throughput=throughput,
        memory=memory,
        batch_results=bench_results_json,
        hardware_info=hardware_info,
        artifact_path=artifact_rel,
        duration_seconds=bench_run.total_duration_seconds,
    )
    bench_repo.commit()
    logger.info(
        "Benchmark %s COMPLETED: P95=%.2f ms, throughput=%.1f req/s",
        record.human_id,
        latency_metrics.get("p95", 0) if latency_metrics else 0,
        throughput or 0,
    )


# ---------------------------------------------------------------------------
# Response helper
# ---------------------------------------------------------------------------

def _to_response(record) -> BenchmarkResponse:
    batch_results_list = None
    if record.batch_results:
        batch_results_list = [
            BatchResult(
                batch_size=r["batch_size"],
                iterations=r["iterations"],
                warmup_count=r["warmup_count"],
                latency_stats=r.get("latency_stats", {}),
                throughput_rps=r.get("throughput_rps", 0.0),
                throughput_sps=r.get("throughput_sps", 0.0),
                memory_peak_mb=r.get("memory_peak_mb"),
                duration_seconds=r.get("duration_seconds", 0.0),
                status=r.get("status", "COMPLETED"),
                error=r.get("error"),
            )
            for r in record.batch_results
        ]

    return BenchmarkResponse(
        benchmark_id=str(record.id),
        human_id=record.human_id,
        model_id=str(record.model_id),
        runtime_variant=record.runtime_variant,
        device=record.device,
        status=record.status,
        iterations=record.iterations,
        warmup_count=record.warmup_count,
        batch_results=batch_results_list,
        latency_metrics=record.latency_metrics,
        throughput=record.throughput,
        memory=record.memory,
        hardware_info=record.hardware_info,
        artifact_path=record.artifact_path,
        duration_seconds=record.duration_seconds,
        error=record.error,
        created_at=record.created_at.isoformat(),
    )
