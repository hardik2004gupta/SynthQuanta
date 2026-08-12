"""SQRuntime API routes — thin delegation to RuntimeService.

Routes:
    POST /runtime/load          → load a model artifact into SQRuntime
    GET  /runtime/health        → current runtime state
    POST /runtime/predict       → single-window inference
    POST /runtime/predict/batch → batch inference
    GET  /runtime/telemetry     → telemetry summary
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.common import ErrorDetail, ErrorResponse
from app.schemas.runtime import (
    BatchPredictionResponse,
    BatchPredictRequest,
    PredictionResponse,
    PredictRequest,
    RuntimeHealthResponse,
    RuntimeLoadRequest,
    TelemetryResponse,
)
from app.services.runtime_service import RuntimeService, RuntimeServiceError
from app.runtime.runtime import SQRuntimeError
from app.runtime.preprocessing import PreprocessingError

router = APIRouter(prefix="/runtime", tags=["runtime"])


def _svc(db: Session = Depends(get_db)) -> RuntimeService:
    return RuntimeService(db)


@router.post("/load", response_model=RuntimeHealthResponse, status_code=status.HTTP_200_OK)
def load_runtime(request: RuntimeLoadRequest, svc: RuntimeService = Depends(_svc)):
    """Load a model artifact into SQRuntime."""
    try:
        return svc.load_model(request)
    except RuntimeServiceError as exc:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error=ErrorDetail(code="RUNTIME_LOAD_ERROR", message=str(exc))
            ).model_dump(),
        )
    except SQRuntimeError as exc:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error=ErrorDetail(code="RUNTIME_LOAD_FAILED", message=str(exc))
            ).model_dump(),
        )


@router.get("/health", response_model=RuntimeHealthResponse)
def runtime_health(svc: RuntimeService = Depends(_svc)):
    """Return the current runtime state — always safe to call."""
    return svc.get_health()


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictRequest, svc: RuntimeService = Depends(_svc)):
    """Run single-window inference."""
    try:
        return svc.predict(request)
    except RuntimeServiceError as exc:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                error=ErrorDetail(code="RUNTIME_NOT_READY", message=str(exc))
            ).model_dump(),
        )
    except SQRuntimeError as exc:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error=ErrorDetail(code="INFERENCE_ERROR", message=str(exc))
            ).model_dump(),
        )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictRequest, svc: RuntimeService = Depends(_svc)):
    """Run batch inference on multiple windows."""
    try:
        return svc.predict_batch(request)
    except RuntimeServiceError as exc:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                error=ErrorDetail(code="RUNTIME_NOT_READY", message=str(exc))
            ).model_dump(),
        )
    except SQRuntimeError as exc:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error=ErrorDetail(code="BATCH_INFERENCE_ERROR", message=str(exc))
            ).model_dump(),
        )


@router.get("/telemetry", response_model=TelemetryResponse)
def get_telemetry(svc: RuntimeService = Depends(_svc)):
    """Return current runtime telemetry."""
    return svc.get_telemetry()
