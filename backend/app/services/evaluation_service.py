"""EvaluationService — orchestrates evaluation jobs.

Architecture:
  POST /evaluation/run  → EvaluationService.start_evaluation_job()
                          → create Evaluation record (PENDING)
                          → spawn background thread
                          → return immediately with evaluation_id

  background thread     → EvaluationService._run_in_background()
                          → load model checkpoint (via EvaluationEngine)
                          → run IID evaluation
                          → (optionally) run 5 distribution-shift scenarios
                          → persist results + artifact
                          → mark COMPLETED | FAILED

  GET /evaluation/{id}  → EvaluationService.get_evaluation()

Rules enforced (§46, §27 of phase spec):
  - Never fabricate metrics — all numbers from actual model outputs
  - Never retrain for shift scenarios — same checkpoint, different data
  - Failures marked FAILED with explicit diagnostic
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models.evaluation import Evaluation
from app.db.repositories.dataset_repository import DatasetRepository
from app.db.repositories.evaluation_repository import EvaluationRepository
from app.db.repositories.experiment_repository import ExperimentRepository
from app.db.repositories.model_repository import ModelRepository
from app.db.session import SessionLocal
from app.ml.evaluation.distribution_shift import DistributionShiftEngine
from app.ml.evaluation.engine import EvaluationEngine
from app.ml.training.trainer import load_model_from_checkpoint
from app.schemas.evaluation import EvaluationResponse, EvaluationRunRequest, EvaluationStartResponse
from app.services.artifact_store import ArtifactStore

logger = logging.getLogger(__name__)
_settings = get_settings()


class EvaluationServiceError(Exception):
    """Raised by EvaluationService for validated failures."""


class EvaluationService:
    """Business logic for evaluation job lifecycle."""

    def __init__(self, db: Session) -> None:
        self._db = db
        self._eval_repo = EvaluationRepository(db)
        self._exp_repo = ExperimentRepository(db)
        self._ds_repo = DatasetRepository(db)
        self._model_repo = ModelRepository(db)
        self._store = ArtifactStore(root=_settings.artifact_root_path)

    # ------------------------------------------------------------------
    # Start a job (non-blocking)
    # ------------------------------------------------------------------

    def start_evaluation_job(self, request: EvaluationRunRequest) -> EvaluationStartResponse:
        # 1. Verify experiment exists and is COMPLETED
        experiment = self._exp_repo.get_by_id(request.experiment_id)
        if experiment is None:
            raise EvaluationServiceError(
                f"Experiment {request.experiment_id!r} not found."
            )
        if experiment.status != "COMPLETED":
            raise EvaluationServiceError(
                f"Experiment {experiment.human_id} is not in COMPLETED state "
                f"(current: {experiment.status}). Train the model first."
            )
        if not experiment.artifact_path:
            raise EvaluationServiceError(
                f"Experiment {experiment.human_id} has no checkpoint artifact path."
            )

        # 2. Find the associated model record
        models = self._model_repo.get_by_experiment(str(experiment.id))
        model_record = models[0] if models else None
        model_id = str(model_record.id) if model_record else None

        # 3. Human-readable ID
        eval_count = self._eval_repo.count()
        human_id = f"EVAL-{eval_count + 1:04d}"

        # 4. Create record
        record = self._eval_repo.create(
            human_id=human_id,
            experiment_id=str(experiment.id),
            model_id=model_id,
            dataset_id=str(experiment.dataset_id),
            evaluation_type="iid+shift" if request.include_shift else "iid",
            status="PENDING",
        )
        self._eval_repo.commit()
        evaluation_id = str(record.id)

        logger.info(
            "Evaluation job created: %s (%s) | experiment=%s | shift=%s",
            human_id, evaluation_id, experiment.human_id, request.include_shift,
        )

        # 5. Spawn background thread
        thread = threading.Thread(
            target=_run_evaluation_thread,
            args=(
                evaluation_id,
                str(experiment.artifact_path),
                str(experiment.dataset_id),
                request.batch_size,
                request.include_shift,
            ),
            name=f"eval-{human_id}",
            daemon=True,
        )
        thread.start()

        return EvaluationStartResponse(
            evaluation_id=evaluation_id,
            human_id=human_id,
            experiment_id=str(experiment.id),
            status="PENDING",
            message=(
                f"Evaluation job {human_id} started. "
                f"Poll GET /api/v1/evaluation/{evaluation_id} for status."
            ),
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_evaluation(self, evaluation_id: str) -> EvaluationResponse:
        record = self._eval_repo.get_by_id(evaluation_id)
        if record is None:
            raise EvaluationServiceError(f"Evaluation {evaluation_id!r} not found.")
        return _evaluation_to_response(record)

    def list_evaluations(
        self, limit: int = 50, offset: int = 0
    ) -> tuple[list[EvaluationResponse], int]:
        rows, total = self._eval_repo.list_all(limit=limit, offset=offset)
        return [_evaluation_to_response(r) for r in rows], total


# ---------------------------------------------------------------------------
# Background thread (owns its own DB session)
# ---------------------------------------------------------------------------

def _run_evaluation_thread(
    evaluation_id: str,
    checkpoint_artifact_path: str,
    dataset_id: str,
    batch_size: int,
    include_shift: bool,
) -> None:
    db: Session = SessionLocal()
    eval_repo = EvaluationRepository(db)
    ds_repo = DatasetRepository(db)
    store = ArtifactStore(root=_settings.artifact_root_path)
    artifact_root = _settings.artifact_root_path

    try:
        record = eval_repo.get_by_id(evaluation_id)
        if record is None:
            logger.error("Thread cannot find evaluation %s — aborting", evaluation_id)
            return

        eval_repo.update(record, status="RUNNING")
        eval_repo.commit()

        # Resolve paths
        checkpoint_dir = artifact_root / checkpoint_artifact_path
        dataset_record = ds_repo.get_by_id(dataset_id)
        if dataset_record is None or not dataset_record.artifact_path:
            raise EvaluationServiceError(
                f"Dataset {dataset_id!r} not found or has no artifact."
            )
        dataset_dir = artifact_root / dataset_record.artifact_path

        logger.info(
            "[%s] Starting IID evaluation | checkpoint=%s | dataset=%s",
            record.human_id, checkpoint_dir.name, dataset_dir.name,
        )

        # --- IID evaluation ---
        engine = EvaluationEngine(
            checkpoint_dir=checkpoint_dir,
            dataset_dir=dataset_dir,
            batch_size=batch_size,
        )
        iid_result = engine.evaluate()

        if iid_result.status == "FAILED":
            raise RuntimeError(f"IID evaluation failed: {iid_result.error}")

        iid_metrics = iid_result.metrics.to_dict()
        iid_localization = iid_result.localization.to_dict()

        # --- Distribution shift ---
        shift_results_list: list[dict] = []
        if include_shift:
            logger.info("[%s] Starting distribution-shift evaluation", record.human_id)
            try:
                model, ckpt_meta = load_model_from_checkpoint(
                    checkpoint_dir, device="cpu"
                )
                model.eval()

                window_size = ckpt_meta["config"]["model"]["window_size"]
                norm_stats_raw = ckpt_meta.get("norm_stats", {})
                norm_stats = (
                    float(norm_stats_raw.get("mean", 0.0)),
                    float(norm_stats_raw.get("std", 1.0)),
                )
                base_config = dataset_record.configuration or {}

                shift_engine = DistributionShiftEngine(
                    model=model,
                    base_config=base_config,
                    norm_stats=norm_stats,
                    iid_f1=iid_result.metrics.macro_f1,
                    window_size=window_size,
                    batch_size=batch_size,
                )
                shift_scenarios = shift_engine.run_all()
                shift_results_list = [s.to_dict() for s in shift_scenarios]
                logger.info(
                    "[%s] Shift evaluation complete: %d scenarios",
                    record.human_id, len(shift_scenarios),
                )
            except Exception as exc:
                logger.warning(
                    "[%s] Distribution-shift evaluation failed (IID still valid): %s",
                    record.human_id, exc,
                )
                shift_results_list = [{"error": str(exc)}]

        # --- Persist artifact ---
        eval_artifact_dir = store.evaluation_path(record.human_id)
        store.ensure_dir(eval_artifact_dir)
        full_results = {
            "iid": {
                "metrics": iid_metrics,
                "localization": iid_localization,
                "n_windows": iid_result.n_windows,
                "duration_seconds": iid_result.duration_seconds,
            },
            "distribution_shift": shift_results_list,
        }
        (eval_artifact_dir / "results.json").write_text(
            json.dumps(full_results, indent=2, default=str), encoding="utf-8"
        )
        artifact_path = f"evaluations/{record.human_id}"

        # Aggregate summary metrics (for quick dashboard display)
        summary_metrics = {
            "macro_f1": iid_result.metrics.macro_f1,
            "weighted_f1": iid_result.metrics.weighted_f1,
            "false_alarm_rate": iid_result.metrics.false_alarm_rate,
            "mean_iou": iid_result.localization.mean_iou,
            "n_windows": iid_result.n_windows,
            "n_shift_scenarios": len(shift_results_list),
        }

        eval_repo.update(
            record,
            status="COMPLETED",
            metrics=summary_metrics,
            results=full_results,
            duration_seconds=iid_result.duration_seconds,
            hardware_info=iid_result.hardware_info,
            artifact_path=artifact_path,
        )
        eval_repo.commit()

        logger.info(
            "[%s] COMPLETED | macro_f1=%.4f | mean_iou=%.4f | %.1f s",
            record.human_id,
            iid_result.metrics.macro_f1,
            iid_result.localization.mean_iou,
            iid_result.duration_seconds,
        )

    except Exception as exc:
        logger.exception("[%s] Evaluation FAILED: %s", evaluation_id, exc)
        try:
            err_record = eval_repo.get_by_id(evaluation_id)
            if err_record is not None:
                eval_repo.update(
                    err_record,
                    status="FAILED",
                    metrics={"error": str(exc)},
                )
                eval_repo.commit()
        except Exception:
            logger.exception("Failed to mark evaluation FAILED in DB")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _evaluation_to_response(record: Evaluation) -> EvaluationResponse:
    return EvaluationResponse(
        evaluation_id=str(record.id),
        human_id=record.human_id,
        experiment_id=str(record.experiment_id),
        model_id=str(record.model_id) if record.model_id else None,
        dataset_id=str(record.dataset_id),
        status=record.status,
        evaluation_type=record.evaluation_type,
        metrics=record.metrics,
        results=record.results,
        duration_seconds=record.duration_seconds,
        hardware_info=record.hardware_info,
        artifact_path=record.artifact_path,
        created_at=record.created_at,
    )
