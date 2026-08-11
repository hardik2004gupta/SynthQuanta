"""QuantizationService — orchestrates INT8 quantization jobs.

Architecture:
  POST /quantization/run  → QuantizationService.start_quantization_job()
                            → validate source model (must be fp32, COMPLETED)
                            → create Quantization record (PENDING)
                            → spawn background thread
                            → return immediately with quantization_id

  background thread       → load FP32 checkpoint
                          → QuantizationEngine.quantize()
                          → validate_inference() smoke-test
                          → save INT8 artifact
                          → register INT8 MLModel record
                          → run_comparison() (FP32 vs INT8 F1 + size + latency)
                          → mark COMPLETED | FAILED

  GET /quantization/{id}  → QuantizationService.get_quantization()

Rules (§11 CLAUDE.md; Phase 5 spec):
  - FP32 artifact is immutable after creation.
  - Quantization failure → FAILED + explicit diagnostic. Never silently fall back.
  - All comparison numbers from actual execution.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models.ml_model import MLModel
from app.db.models.quantization import Quantization
from app.db.repositories.dataset_repository import DatasetRepository
from app.db.repositories.experiment_repository import ExperimentRepository
from app.db.repositories.model_repository import ModelRepository
from app.db.repositories.quantization_repository import QuantizationRepository
from app.db.session import SessionLocal
from app.ml.quantization.comparison import run_comparison
from app.ml.quantization.config import QuantizationConfig
from app.ml.quantization.engine import QuantizationEngine, QuantizationError
from app.ml.training.reproducibility import collect_hardware_info
from app.schemas.quantization import (
    ComparisonMetrics,
    QuantizationResponse,
    QuantizationRunRequest,
    QuantizationStartResponse,
)
from app.services.artifact_store import ArtifactStore

logger = logging.getLogger(__name__)
_settings = get_settings()


class QuantizationServiceError(Exception):
    """Raised by QuantizationService for validated failures."""


def _next_quant_human_id(repo: QuantizationRepository) -> str:
    n = repo.count() + 1
    return f"QUANT-{n:04d}"


def _next_model_human_id(repo: ModelRepository) -> str:
    n = repo.count() + 1
    return f"MODEL-{n:04d}"


class QuantizationService:
    """Business logic for quantization job lifecycle."""

    def __init__(self, db: Session) -> None:
        self._db = db
        self._quant_repo = QuantizationRepository(db)
        self._model_repo = ModelRepository(db)
        self._exp_repo = ExperimentRepository(db)
        self._ds_repo = DatasetRepository(db)
        self._store = ArtifactStore(root=Path(_settings.artifact_root))

    # ------------------------------------------------------------------
    # Start job (called from API route — returns immediately)
    # ------------------------------------------------------------------

    def start_quantization_job(
        self, request: QuantizationRunRequest
    ) -> QuantizationStartResponse:
        # 1. Validate source model exists and is FP32 + COMPLETED
        source_model = self._model_repo.get_by_id(request.source_model_id)
        if source_model is None:
            raise QuantizationServiceError(
                f"MODEL_NOT_FOUND: No model with id={request.source_model_id}"
            )
        if source_model.precision != "fp32":
            raise QuantizationServiceError(
                f"MODEL_NOT_FP32: Model {source_model.human_id} has precision="
                f"'{source_model.precision}'. Only fp32 models can be quantized."
            )
        if source_model.status != "COMPLETED":
            raise QuantizationServiceError(
                f"MODEL_NOT_READY: Model {source_model.human_id} has status="
                f"'{source_model.status}'. Only COMPLETED models can be quantized."
            )

        # 2. Resolve dataset_id (explicit → experiment dataset → None)
        dataset_id = request.dataset_id
        if dataset_id is None and source_model.experiment_id:
            exp = self._exp_repo.get_by_id(source_model.experiment_id)
            if exp is not None:
                dataset_id = exp.dataset_id

        # 3. Create Quantization record
        human_id = _next_quant_human_id(self._quant_repo)
        quant_record = self._quant_repo.create(
            id=str(uuid.uuid4()),
            human_id=human_id,
            source_model_id=source_model.id,
            dataset_id=dataset_id,
            status="PENDING",
            method="dynamic_int8",
        )
        self._quant_repo.commit()

        quantization_id = quant_record.id

        # 4. Spawn background thread
        config = QuantizationConfig(
            method="dynamic_int8",
            benchmark_iterations=request.benchmark_iterations,
            benchmark_warmup=request.benchmark_warmup,
        )
        t = threading.Thread(
            target=_run_quantization_thread,
            args=(quantization_id, source_model.id, dataset_id, config),
            daemon=True,
        )
        t.start()

        return QuantizationStartResponse(
            quantization_id=quantization_id,
            human_id=human_id,
            source_model_id=source_model.id,
            status="PENDING",
            message=f"Quantization job {human_id} queued",
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_quantization(self, quantization_id: str) -> QuantizationResponse:
        record = self._quant_repo.get_by_id(quantization_id)
        if record is None:
            raise QuantizationServiceError(
                f"QUANTIZATION_NOT_FOUND: No quantization with id={quantization_id}"
            )
        return _to_response(record)

    def list_quantizations(
        self, limit: int = 50, offset: int = 0
    ) -> list[QuantizationResponse]:
        rows, _ = self._quant_repo.list_all(limit=limit, offset=offset)
        return [_to_response(r) for r in rows]


# ---------------------------------------------------------------------------
# Background thread (runs in a new DB session — isolated from the request)
# ---------------------------------------------------------------------------

def _run_quantization_thread(
    quantization_id: str,
    source_model_id: str,
    dataset_id: str | None,
    config: QuantizationConfig,
) -> None:
    db = SessionLocal()
    try:
        quant_repo = QuantizationRepository(db)
        model_repo = ModelRepository(db)
        store = ArtifactStore(root=Path(_settings.artifact_root))

        record = quant_repo.get_by_id(quantization_id)
        if record is None:
            logger.error("Quantization record %s vanished before thread started", quantization_id)
            return

        quant_repo.update(record, status="RUNNING")
        quant_repo.commit()

        t0 = time.perf_counter()
        try:
            _do_quantize(
                record=record,
                source_model_id=source_model_id,
                dataset_id=dataset_id,
                config=config,
                quant_repo=quant_repo,
                model_repo=model_repo,
                store=store,
            )
            duration = time.perf_counter() - t0
            quant_repo.update(record, status="COMPLETED", duration_seconds=duration)
            quant_repo.commit()
            logger.info("Quantization %s completed in %.1fs", record.human_id, duration)

        except Exception as exc:
            duration = time.perf_counter() - t0
            error_msg = str(exc)
            logger.exception("Quantization %s FAILED: %s", record.human_id, error_msg)
            quant_repo.update(
                record,
                status="FAILED",
                error=error_msg,
                duration_seconds=duration,
            )
            quant_repo.commit()

    finally:
        db.close()


def _do_quantize(
    record: Quantization,
    source_model_id: str,
    dataset_id: str | None,
    config: QuantizationConfig,
    quant_repo: QuantizationRepository,
    model_repo: ModelRepository,
    store: ArtifactStore,
) -> None:
    from app.ml.training.trainer import load_model_from_checkpoint

    # --- Resolve FP32 model paths ---
    source_model = model_repo.get_by_id(source_model_id)
    if source_model is None:
        raise QuantizationError(f"Source model {source_model_id} not found")

    fp32_checkpoint_dir = store.root / source_model.artifact_path
    if not fp32_checkpoint_dir.exists():
        raise QuantizationError(
            f"FP32 checkpoint directory not found: {fp32_checkpoint_dir}"
        )

    # --- Load FP32 model ---
    fp32_model, ckpt_meta = load_model_from_checkpoint(fp32_checkpoint_dir, device="cpu")
    fp32_model.eval()

    window_size: int = ckpt_meta["config"]["model"]["window_size"]

    # --- Quantize ---
    engine = QuantizationEngine(config)
    quantized_model = engine.quantize(fp32_model)

    # --- Smoke-test (mandatory — never skip) ---
    QuantizationEngine.validate_inference(quantized_model, window_size)

    # --- Persist INT8 artifact ---
    int8_human_id = f"MODEL-{model_repo.count() + 1:04d}"
    int8_artifact_rel = f"models/{int8_human_id}-INT8"
    int8_artifact_dir = store.root / int8_artifact_rel

    int8_file, int8_meta = engine.save_quantized(
        quantized_model,
        artifact_dir=int8_artifact_dir,
        source_metadata=ckpt_meta,
    )

    # Update quantization record with backend + artifact path
    quant_repo.update(
        record,
        backend=engine.backend,
        artifact_path=int8_artifact_rel,
    )
    quant_repo.commit()

    # --- Register INT8 MLModel record ---
    int8_model_id = str(uuid.uuid4())
    int8_model_record = model_repo.create(
        id=int8_model_id,
        human_id=int8_human_id,
        name=f"{source_model.name}-int8",
        version=source_model.version,
        base_model=source_model.base_model,
        experiment_id=source_model.experiment_id,
        precision="int8",
        quantization="dynamic_int8",
        artifact_path=int8_artifact_rel,
        status="COMPLETED",
        total_parameters=source_model.total_parameters,
        trainable_parameters=source_model.trainable_parameters,
    )
    model_repo.commit()

    # Link INT8 model to quantization record
    quant_repo.update(record, quantized_model_id=int8_model_id)
    quant_repo.commit()

    # --- Quality + size + latency comparison (only if dataset available) ---
    if dataset_id is not None:
        from app.db.repositories.dataset_repository import DatasetRepository
        db_session = quant_repo._db
        ds_repo = DatasetRepository(db_session)
        ds_record = ds_repo.get_by_id(dataset_id)

        if ds_record is not None and ds_record.artifact_path:
            dataset_dir = store.root / ds_record.artifact_path
            if dataset_dir.exists():
                comparison = run_comparison(
                    fp32_checkpoint_dir=fp32_checkpoint_dir,
                    int8_model=quantized_model,
                    int8_model_file=int8_file,
                    dataset_dir=dataset_dir,
                    evaluation_batch_size=64,
                    benchmark_iterations=config.benchmark_iterations,
                    benchmark_warmup=config.benchmark_warmup,
                )
                quant_repo.update(
                    record,
                    fp32_macro_f1=comparison.fp32_macro_f1,
                    int8_macro_f1=comparison.int8_macro_f1,
                    f1_delta=comparison.f1_delta,
                    fp32_size_bytes=comparison.fp32_size_bytes,
                    int8_size_bytes=comparison.int8_size_bytes,
                    size_reduction_ratio=comparison.size_reduction_ratio,
                    fp32_latency_ms=comparison.fp32_latency_ms,
                    int8_latency_ms=comparison.int8_latency_ms,
                    latency_speedup=comparison.latency_speedup,
                    n_test_windows=comparison.n_test_windows,
                )
                quant_repo.commit()

                # Attach metrics to INT8 model record too
                model_repo.update(
                    int8_model_record,
                    metrics={
                        "int8_macro_f1": comparison.int8_macro_f1,
                        "f1_delta": comparison.f1_delta,
                        "size_reduction_ratio": comparison.size_reduction_ratio,
                        "latency_speedup": comparison.latency_speedup,
                    },
                )
                model_repo.commit()
            else:
                logger.warning(
                    "Dataset directory not found at %s; skipping comparison", dataset_dir
                )
        else:
            logger.warning("Dataset record %s missing artifact_path; skipping comparison", dataset_id)
    else:
        logger.info("No dataset_id provided; skipping quality/size/latency comparison")


# ---------------------------------------------------------------------------
# DB record → Pydantic response
# ---------------------------------------------------------------------------

def _to_response(record: Quantization) -> QuantizationResponse:
    comparison: ComparisonMetrics | None = None
    if record.status == "COMPLETED" and record.fp32_macro_f1 is not None:
        comparison = ComparisonMetrics(
            fp32_macro_f1=record.fp32_macro_f1,
            int8_macro_f1=record.int8_macro_f1 or 0.0,
            f1_delta=record.f1_delta or 0.0,
            fp32_size_bytes=record.fp32_size_bytes or 0,
            int8_size_bytes=record.int8_size_bytes or 0,
            size_reduction_ratio=record.size_reduction_ratio or 0.0,
            fp32_latency_ms=record.fp32_latency_ms or 0.0,
            int8_latency_ms=record.int8_latency_ms or 0.0,
            latency_speedup=record.latency_speedup or 0.0,
            n_test_windows=record.n_test_windows or 0,
        )

    return QuantizationResponse(
        quantization_id=record.id,
        human_id=record.human_id,
        source_model_id=record.source_model_id,
        quantized_model_id=record.quantized_model_id,
        dataset_id=record.dataset_id,
        status=record.status,
        method=record.method,
        backend=record.backend,
        comparison=comparison,
        artifact_path=record.artifact_path,
        duration_seconds=record.duration_seconds,
        error=record.error,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )
