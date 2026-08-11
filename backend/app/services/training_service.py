"""TrainingService — orchestrates training jobs.

Architecture:
  POST /training/run  → TrainingService.start_training_job()
                        → creates Experiment record (PENDING)
                        → spawns background thread
                        → returns immediately with experiment_id

  background thread   → TrainingService._run_in_background()
                        → loads dataset artifact
                        → builds Trainer
                        → runs training loop (with per-epoch DB callbacks)
                        → persists checkpoint + MLModel record
                        → marks Experiment COMPLETED | FAILED

  GET /training/{id}  → TrainingService.get_experiment()
                        → queries DB → returns current state

Rules enforced here (§NON-NEGOTIABLE §12, §16):
  - Never fabricate training metrics
  - Never silently substitute LoRA when QLoRA requested
  - Never put training logic in API routes
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models.experiment import Experiment
from app.db.models.ml_model import MLModel
from app.db.repositories.dataset_repository import DatasetRepository
from app.db.repositories.experiment_repository import ExperimentRepository
from app.db.repositories.model_repository import ModelRepository
from app.db.session import SessionLocal
from app.ml.adapters.lora import LoRAConfig
from app.ml.datasets.dataset import DatasetFactory
from app.ml.models.sensor_transformer import ModelConfig
from app.ml.training.config import TrainingConfig
from app.ml.training.trainer import EpochMetrics, Trainer, TrainerError, TrainingResult
from app.schemas.training import (
    EpochMetricsSchema,
    ExperimentResponse,
    TrainingRunRequest,
    TrainingStartResponse,
)
from app.services.artifact_store import ArtifactStore

logger = logging.getLogger(__name__)
_settings = get_settings()


class TrainingServiceError(Exception):
    """Raised by TrainingService for validated failures."""


class TrainingService:
    """Business logic for training experiment lifecycle."""

    def __init__(self, db: Session) -> None:
        self._db = db
        self._exp_repo = ExperimentRepository(db)
        self._ds_repo = DatasetRepository(db)
        self._model_repo = ModelRepository(db)
        self._store = ArtifactStore(root=_settings.artifact_root_path)

    # ------------------------------------------------------------------
    # Start a training job (non-blocking)
    # ------------------------------------------------------------------

    def start_training_job(self, request: TrainingRunRequest) -> TrainingStartResponse:
        """Validate, create the Experiment record, and start the background thread."""
        # 1. Verify dataset exists
        dataset = self._ds_repo.get_by_id(request.dataset_id)
        if dataset is None:
            raise TrainingServiceError(f"Dataset {request.dataset_id!r} not found.")
        if dataset.status != "COMPLETED":
            raise TrainingServiceError(
                f"Dataset {dataset.human_id} is not in COMPLETED state "
                f"(current: {dataset.status}).  Generate the dataset first."
            )

        # 2. Assign a human-readable experiment ID
        exp_count = self._exp_repo.count()
        human_id = f"EXP-{exp_count + 1:04d}"

        # 3. Build training config from request
        training_cfg = _build_training_config(request)

        # 4. Create Experiment record in PENDING state
        config_dict = training_cfg.to_dict()
        record = self._exp_repo.create(
            human_id=human_id,
            name=request.name,
            dataset_id=request.dataset_id,
            method=request.method,
            configuration=config_dict,
            status="PENDING",
        )
        self._exp_repo.commit()
        experiment_id = str(record.id)

        logger.info(
            "Training job created: %s (%s) | dataset=%s | method=%s",
            human_id,
            experiment_id,
            dataset.human_id,
            request.method,
        )

        # 5. Spawn background thread
        thread = threading.Thread(
            target=_run_training_thread,
            args=(experiment_id, training_cfg, dataset.artifact_path),
            name=f"train-{human_id}",
            daemon=True,
        )
        thread.start()

        return TrainingStartResponse(
            experiment_id=experiment_id,
            human_id=human_id,
            status="PENDING",
            message=(
                f"Training job {human_id} started. "
                "Poll GET /api/v1/training/{experiment_id} for status."
            ),
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_experiment(self, experiment_id: str) -> ExperimentResponse:
        record = self._exp_repo.get_by_id(experiment_id)
        if record is None:
            raise TrainingServiceError(f"Experiment {experiment_id!r} not found.")
        return _experiment_to_response(record, self._model_repo)

    def list_experiments(self, limit: int = 50, offset: int = 0) -> tuple[list[ExperimentResponse], int]:
        rows, total = self._exp_repo.list_all(limit=limit, offset=offset)
        return [_experiment_to_response(r, self._model_repo) for r in rows], total


# ---------------------------------------------------------------------------
# Background thread (owns its own DB session)
# ---------------------------------------------------------------------------

def _run_training_thread(
    experiment_id: str,
    training_cfg: TrainingConfig,
    dataset_artifact_path: str,
) -> None:
    """Runs in a daemon thread.  Creates its own SQLAlchemy session."""
    db: Session = SessionLocal()
    exp_repo = ExperimentRepository(db)
    model_repo = ModelRepository(db)
    store = ArtifactStore(root=_settings.artifact_root_path)

    try:
        record = exp_repo.get_by_id(experiment_id)
        if record is None:
            logger.error("Thread cannot find experiment %s — aborting", experiment_id)
            return

        # Mark RUNNING
        exp_repo.update(record, status="RUNNING")
        exp_repo.commit()

        # Resolve the dataset artifact directory
        artifact_root = _settings.artifact_root_path
        dataset_dir = artifact_root / dataset_artifact_path

        logger.info(
            "[%s] Loading dataset from %s", record.human_id, dataset_dir
        )

        # Load dataset splits
        window_size = training_cfg.model.window_size
        splits = DatasetFactory.from_artifact(dataset_dir, window_size=window_size)

        # Determine checkpoint directory
        model_count = model_repo.count()
        model_human_id = f"MODEL-{model_count + 1:04d}"
        checkpoint_dir = artifact_root / "models" / model_human_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Progress callback — writes training_history to DB after each epoch
        def on_epoch(ep: EpochMetrics) -> None:
            try:
                inner_db: Session = SessionLocal()
                inner_repo = ExperimentRepository(inner_db)
                try:
                    inner_rec = inner_repo.get_by_id(experiment_id)
                    if inner_rec is not None:
                        current_history = list(inner_rec.training_history or [])
                        current_history.append({
                            "epoch": ep.epoch,
                            "train_loss": ep.train_loss,
                            "val_loss": ep.val_loss,
                            "val_accuracy": ep.val_accuracy,
                            "duration_seconds": ep.duration_seconds,
                            "learning_rate": ep.learning_rate,
                        })
                        inner_repo.update(inner_rec, training_history=current_history)
                        inner_repo.commit()
                finally:
                    inner_db.close()
            except Exception:
                logger.warning("on_epoch callback failed", exc_info=True)

        # Train
        trainer = Trainer(
            train_dataset=splits.train,
            val_dataset=splits.validation,
            config=training_cfg,
            checkpoint_dir=checkpoint_dir,
            norm_stats=splits.norm_stats,
            progress_cb=on_epoch,
        )

        result: TrainingResult = trainer.train()

        # Persist MLModel record
        lora_cfg_dict = training_cfg.lora.to_dict() if training_cfg.method in ("lora", "qlora") else None
        model_artifact_path = f"models/{model_human_id}"

        model_record = model_repo.create(
            human_id=model_human_id,
            name=f"{record.name}-model",
            version="v1",
            base_model=training_cfg.model.architecture,
            experiment_id=experiment_id,
            precision="fp32",
            artifact_path=model_artifact_path,
            status="COMPLETED",
            total_parameters=result.total_parameters,
            trainable_parameters=result.trainable_parameters,
            lora_config=lora_cfg_dict,
        )
        model_repo.commit()

        # Summary metrics attached to the experiment record
        summary_metrics = {
            "best_val_loss": result.best_val_loss,
            "best_epoch": result.best_epoch,
            "total_epochs": result.total_epochs,
            "final_train_loss": result.training_history[-1].train_loss if result.training_history else None,
            "final_val_loss": result.training_history[-1].val_loss if result.training_history else None,
            "final_val_accuracy": result.training_history[-1].val_accuracy if result.training_history else None,
            "total_parameters": result.total_parameters,
            "trainable_parameters": result.trainable_parameters,
        }

        # Flatten training history for the experiment record
        history_dicts = [
            {
                "epoch": ep.epoch,
                "train_loss": ep.train_loss,
                "val_loss": ep.val_loss,
                "val_accuracy": ep.val_accuracy,
                "duration_seconds": ep.duration_seconds,
                "learning_rate": ep.learning_rate,
            }
            for ep in result.training_history
        ]

        exp_repo.update(
            record,
            status="COMPLETED",
            metrics=summary_metrics,
            training_history=history_dicts,
            artifact_path=model_artifact_path,
            duration_seconds=result.duration_seconds,
            hardware_info=result.hardware_info,
        )
        exp_repo.commit()

        logger.info(
            "[%s] COMPLETED in %.1f s | best_val_loss=%.4f | model=%s",
            record.human_id,
            result.duration_seconds,
            result.best_val_loss,
            model_human_id,
        )

    except (TrainerError, Exception) as exc:
        logger.exception("[%s] Training FAILED: %s", experiment_id, exc)
        try:
            err_record = exp_repo.get_by_id(experiment_id)
            if err_record is not None:
                current_metrics = dict(err_record.metrics or {})
                current_metrics["error"] = str(exc)
                exp_repo.update(
                    err_record,
                    status="FAILED",
                    metrics=current_metrics,
                )
                exp_repo.commit()
        except Exception:
            logger.exception("Failed to mark experiment FAILED in DB")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_training_config(request: TrainingRunRequest) -> TrainingConfig:
    """Convert API request schema to internal TrainingConfig."""
    lora_cfg = LoRAConfig(
        rank=request.lora.rank,
        alpha=request.lora.alpha,
        dropout=request.lora.dropout,
        target_modules=list(request.lora.target_modules),
    )
    model_cfg = ModelConfig(
        architecture=request.model.architecture,
        window_size=request.model.window_size,
        embedding_dim=request.model.embedding_dim,
        num_layers=request.model.num_layers,
        num_heads=request.model.num_heads,
        ffn_dim=request.model.ffn_dim,
        dropout=request.model.dropout,
    )
    cfg = TrainingConfig(
        method=request.method,
        epochs=request.epochs,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        weight_decay=request.weight_decay,
        seed=request.seed,
        lora=lora_cfg,
        model=model_cfg,
    )
    cfg.validate()
    return cfg


def _experiment_to_response(record: Experiment, model_repo: ModelRepository) -> ExperimentResponse:
    """Map an ORM Experiment to the API response schema."""
    # Find the most recent model for this experiment
    models = model_repo.get_by_experiment(str(record.id))
    first_model: MLModel | None = models[0] if models else None

    history = None
    if record.training_history:
        history = [
            EpochMetricsSchema(**ep) for ep in record.training_history
        ]

    error_msg = None
    if record.status == "FAILED" and record.metrics:
        error_msg = record.metrics.get("error")

    return ExperimentResponse(
        experiment_id=str(record.id),
        human_id=record.human_id,
        name=record.name,
        dataset_id=str(record.dataset_id),
        method=record.method,
        status=record.status,
        configuration=record.configuration or {},
        metrics=record.metrics,
        training_history=history,
        artifact_path=record.artifact_path,
        duration_seconds=record.duration_seconds,
        hardware_info=record.hardware_info,
        error_message=error_msg,
        created_at=record.created_at,
        model_id=str(first_model.id) if first_model else None,
        model_human_id=first_model.human_id if first_model else None,
    )
