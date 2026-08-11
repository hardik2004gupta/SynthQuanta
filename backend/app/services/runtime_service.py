"""RuntimeService — manages the SQRuntime singleton lifecycle.

Architecture (§26–§28 Phase 6 spec):
    API Route
        ↓
    RuntimeService        ← this module
        ↓
    SQRuntime (singleton)
        ↓
    Model artifact

The runtime is a process-level singleton.  It loads once and stays loaded
until a new POST /runtime/load replaces it.  All inference requests share
the same loaded model — thread-safe because PyTorch eval-mode forward
passes are inherently reentrant.
"""
from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.repositories.model_repository import ModelRepository
from app.db.repositories.quantization_repository import QuantizationRepository
from app.runtime.runtime import SQRuntime, SQRuntimeError
from app.schemas.runtime import (
    BatchPredictionResponse,
    BatchPredictRequest,
    PredictionResponse,
    PredictRequest,
    RuntimeHealthResponse,
    RuntimeLoadRequest,
    TelemetryResponse,
)

logger = logging.getLogger(__name__)
_settings = get_settings()

# ---------------------------------------------------------------------------
# Process-level singleton runtime
# ---------------------------------------------------------------------------

_runtime: SQRuntime = SQRuntime()


def get_runtime() -> SQRuntime:
    """Return the process-level SQRuntime instance."""
    return _runtime


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class RuntimeServiceError(Exception):
    """Raised by RuntimeService for validated failures."""


class RuntimeService:
    """Business logic for SQRuntime lifecycle and inference."""

    def __init__(self, db: Session) -> None:
        self._db = db
        self._model_repo = ModelRepository(db)
        self._quant_repo = QuantizationRepository(db)
        self._artifact_root = _settings.artifact_root_path

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_model(self, request: RuntimeLoadRequest) -> RuntimeHealthResponse:
        """Load a model artifact into SQRuntime.

        Resolves the artifact from the DB using model_id or quantization_id.
        Never accepts raw filesystem paths from the caller.

        Returns the current health state after loading.
        """
        runtime = get_runtime()

        if request.quantization_id:
            return self._load_int8(runtime, request.quantization_id)
        elif request.model_id:
            return self._load_fp32(runtime, request.model_id)
        else:
            raise RuntimeServiceError(
                "Either model_id or quantization_id must be provided."
            )

    def _load_fp32(self, runtime: SQRuntime, model_id: str) -> RuntimeHealthResponse:
        model_record = self._model_repo.get_by_id(model_id)
        if model_record is None:
            raise RuntimeServiceError(f"Model {model_id!r} not found.")
        if model_record.status != "COMPLETED":
            raise RuntimeServiceError(
                f"Model {model_record.human_id} is not COMPLETED "
                f"(current: {model_record.status})."
            )
        if model_record.precision != "fp32":
            raise RuntimeServiceError(
                f"Model {model_record.human_id} has precision "
                f"{model_record.precision!r}, not 'fp32'. "
                "Use quantization_id to load an INT8 model."
            )

        artifact_dir = self._artifact_root / model_record.artifact_path
        logger.info(
            "Loading FP32 model %s from %s", model_record.human_id, artifact_dir
        )

        runtime.load(
            artifact_dir=artifact_dir,
            precision="fp32",
            model_id=str(model_record.id),
            artifact_path=model_record.artifact_path,
        )
        return _health_to_response(runtime.health())

    def _load_int8(self, runtime: SQRuntime, quantization_id: str) -> RuntimeHealthResponse:
        quant_record = self._quant_repo.get_by_id(quantization_id)
        if quant_record is None:
            raise RuntimeServiceError(f"Quantization {quantization_id!r} not found.")
        if quant_record.status != "COMPLETED":
            raise RuntimeServiceError(
                f"Quantization {quant_record.human_id} is not COMPLETED "
                f"(current: {quant_record.status})."
            )
        if quant_record.artifact_path is None:
            raise RuntimeServiceError(
                f"Quantization {quant_record.human_id} has no artifact_path."
            )

        # The quantization artifact_path points to the INT8 artifact directory
        artifact_dir = self._artifact_root / quant_record.artifact_path
        model_id = quant_record.quantized_model_id or quant_record.source_model_id

        logger.info(
            "Loading INT8 model from quantization %s at %s",
            quant_record.human_id, artifact_dir,
        )

        runtime.load(
            artifact_dir=artifact_dir,
            precision="int8",
            model_id=str(model_id) if model_id else "unknown",
            artifact_path=quant_record.artifact_path,
        )
        return _health_to_response(runtime.health())

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def get_health(self) -> RuntimeHealthResponse:
        runtime = get_runtime()
        return _health_to_response(runtime.health())

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, request: PredictRequest) -> PredictionResponse:
        runtime = get_runtime()
        if not runtime.is_ready():
            raise RuntimeServiceError(
                "SQRuntime is not ready. Load a model first via POST /api/v1/runtime/load."
            )
        result = runtime.predict(request.values)
        return PredictionResponse(
            predicted_class=result.predicted_class,
            predicted_class_index=result.predicted_class_index,
            confidence=result.confidence,
            probabilities=result.probabilities,
            latency_ms=result.latency_ms,
        )

    def predict_batch(self, request: BatchPredictRequest) -> BatchPredictionResponse:
        runtime = get_runtime()
        if not runtime.is_ready():
            raise RuntimeServiceError(
                "SQRuntime is not ready. Load a model first via POST /api/v1/runtime/load."
            )
        results = runtime.predict_batch(request.windows)
        return BatchPredictionResponse(
            predictions=[
                PredictionResponse(
                    predicted_class=r.predicted_class,
                    predicted_class_index=r.predicted_class_index,
                    confidence=r.confidence,
                    probabilities=r.probabilities,
                    latency_ms=r.latency_ms,
                )
                for r in results
            ]
        )

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def get_telemetry(self) -> TelemetryResponse:
        runtime = get_runtime()
        t = runtime.get_telemetry()
        return TelemetryResponse(
            request_count=t.request_count,
            success_count=t.success_count,
            error_count=t.error_count,
            last_latency_ms=t.last_latency_ms,
            mean_latency_ms=t.mean_latency_ms,
            p50_latency_ms=t.p50_latency_ms,
            p95_latency_ms=t.p95_latency_ms,
            p99_latency_ms=t.p99_latency_ms,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _health_to_response(health) -> RuntimeHealthResponse:
    return RuntimeHealthResponse(
        status=health.status,
        model_id=health.model_id,
        artifact_path=health.artifact_path,
        precision=health.precision,
        runtime_variant=health.runtime_variant,
        device=health.device,
        backend=health.backend,
        loaded_at=health.loaded_at.isoformat() if health.loaded_at else None,
        request_count=health.request_count,
        error=health.error,
    )
