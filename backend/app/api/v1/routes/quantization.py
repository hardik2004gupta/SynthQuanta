"""Quantization API routes (Phase 5).

POST /api/v1/quantization/run          → start a quantization job
GET  /api/v1/quantization/{id}         → poll job status + results
GET  /api/v1/quantization              → list all quantization records
GET  /api/v1/models/{model_id}/quantization → list quantizations for a model

Routes are thin — all logic is delegated to QuantizationService.
"""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.quantization import (
    QuantizationResponse,
    QuantizationRunRequest,
    QuantizationStartResponse,
)
from app.services.quantization_service import QuantizationService, QuantizationServiceError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/quantization", tags=["quantization"])


def _service(db: Annotated[Session, Depends(get_db)]) -> QuantizationService:
    return QuantizationService(db)


@router.post("/run", response_model=QuantizationStartResponse, status_code=202)
def start_quantization(
    request: QuantizationRunRequest,
    svc: Annotated[QuantizationService, Depends(_service)],
) -> QuantizationStartResponse:
    """Start an INT8 quantization job for a FP32 model.

    Returns immediately with a quantization_id for polling.
    The job runs in the background (PENDING → RUNNING → COMPLETED | FAILED).
    """
    try:
        return svc.start_quantization_job(request)
    except QuantizationServiceError as exc:
        msg = str(exc)
        if "MODEL_NOT_FOUND" in msg:
            raise HTTPException(
                status_code=404,
                detail={"code": "MODEL_NOT_FOUND", "message": msg},
            )
        if "MODEL_NOT_FP32" in msg:
            raise HTTPException(
                status_code=409,
                detail={"code": "MODEL_NOT_FP32", "message": msg},
            )
        if "MODEL_NOT_READY" in msg:
            raise HTTPException(
                status_code=409,
                detail={"code": "MODEL_NOT_READY", "message": msg},
            )
        raise HTTPException(status_code=500, detail={"code": "QUANTIZATION_ERROR", "message": msg})


@router.get("/{quantization_id}", response_model=QuantizationResponse)
def get_quantization(
    quantization_id: str,
    svc: Annotated[QuantizationService, Depends(_service)],
) -> QuantizationResponse:
    """Get quantization job status and comparison results."""
    try:
        return svc.get_quantization(quantization_id)
    except QuantizationServiceError as exc:
        raise HTTPException(
            status_code=404,
            detail={"code": "QUANTIZATION_NOT_FOUND", "message": str(exc)},
        )


@router.get("", response_model=list[QuantizationResponse])
def list_quantizations(
    svc: Annotated[QuantizationService, Depends(_service)],
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> list[QuantizationResponse]:
    """List all quantization records, newest first."""
    return svc.list_quantizations(limit=limit, offset=offset)
