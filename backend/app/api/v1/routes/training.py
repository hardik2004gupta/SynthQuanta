"""Training API routes — thin delegation to TrainingService.

Routes:
  POST /training/run              start a training job (non-blocking)
  GET  /training/{experiment_id}  poll status and retrieve results
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.common import ErrorDetail, ErrorResponse
from app.schemas.training import ExperimentResponse, TrainingRunRequest, TrainingStartResponse
from app.services.training_service import TrainingService, TrainingServiceError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training", tags=["training"])


@router.post(
    "/run",
    response_model=TrainingStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        404: {"model": ErrorResponse, "description": "Dataset not found"},
        409: {"model": ErrorResponse, "description": "Dataset not ready"},
        422: {"description": "Validation error"},
    },
)
def start_training(
    request: TrainingRunRequest,
    db: Session = Depends(get_db),
) -> TrainingStartResponse:
    """Start a training job.

    Returns immediately with `experiment_id` and `status: PENDING`.
    Poll `GET /training/{experiment_id}` to track progress.
    """
    svc = TrainingService(db)
    try:
        return svc.start_training_job(request)
    except TrainingServiceError as exc:
        msg = str(exc)
        if "not found" in msg.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "DATASET_NOT_FOUND", "message": msg},
            )
        if "not in completed" in msg.lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"code": "DATASET_NOT_READY", "message": msg},
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "TRAINING_ERROR", "message": msg},
        )


@router.get(
    "/{experiment_id}",
    response_model=ExperimentResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Experiment not found"},
    },
)
def get_experiment(
    experiment_id: str,
    db: Session = Depends(get_db),
) -> ExperimentResponse:
    """Retrieve experiment status and (when complete) training results."""
    svc = TrainingService(db)
    try:
        return svc.get_experiment(experiment_id)
    except TrainingServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "EXPERIMENT_NOT_FOUND", "message": str(exc)},
        )
