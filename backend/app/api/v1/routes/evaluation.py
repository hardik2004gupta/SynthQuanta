"""Evaluation API routes — thin delegation to EvaluationService.

Routes:
  POST /evaluation/run              start an evaluation job (non-blocking)
  GET  /evaluation/{evaluation_id}  poll status and retrieve results
  GET  /evaluation                  list recent evaluations
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.common import ErrorResponse
from app.schemas.evaluation import (
    EvaluationResponse,
    EvaluationRunRequest,
    EvaluationStartResponse,
)
from app.services.evaluation_service import EvaluationService, EvaluationServiceError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post(
    "/run",
    response_model=EvaluationStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        404: {"model": ErrorResponse, "description": "Experiment not found"},
        409: {"model": ErrorResponse, "description": "Experiment not ready"},
        422: {"description": "Validation error"},
    },
)
def start_evaluation(
    request: EvaluationRunRequest,
    db: Session = Depends(get_db),
) -> EvaluationStartResponse:
    """Start an evaluation job.

    Returns immediately with `evaluation_id` and `status: PENDING`.
    Poll `GET /evaluation/{evaluation_id}` to track progress.
    """
    svc = EvaluationService(db)
    try:
        return svc.start_evaluation_job(request)
    except EvaluationServiceError as exc:
        msg = str(exc)
        if "not found" in msg.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "EXPERIMENT_NOT_FOUND", "message": msg},
            )
        if "not in completed" in msg.lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"code": "EXPERIMENT_NOT_READY", "message": msg},
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "EVALUATION_ERROR", "message": msg},
        )


@router.get(
    "/{evaluation_id}",
    response_model=EvaluationResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Evaluation not found"},
    },
)
def get_evaluation(
    evaluation_id: str,
    db: Session = Depends(get_db),
) -> EvaluationResponse:
    """Retrieve evaluation status and (when complete) results."""
    svc = EvaluationService(db)
    try:
        return svc.get_evaluation(evaluation_id)
    except EvaluationServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "EVALUATION_NOT_FOUND", "message": str(exc)},
        )


@router.get(
    "",
    response_model=list[EvaluationResponse],
    responses={},
)
def list_evaluations(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[EvaluationResponse]:
    """List recent evaluations."""
    svc = EvaluationService(db)
    rows, _ = svc.list_evaluations(limit=limit, offset=offset)
    return rows
