"""DatasetService — orchestrates the full dataset generation workflow.

Pipeline:
    validate request → assign human_id → create DB record (PENDING)
    → generate data → validate → persist artifact
    → update DB record (COMPLETED) → return response

On failure at any stage: DB record is marked FAILED with diagnostic.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.data.engine import DataEngineError, SyntheticDataEngine
from app.data.models import FaultType, RawDataset, WindowedDataset
from app.db.models.dataset import Dataset
from app.db.repositories.dataset_repository import DatasetRepository
from app.schemas.dataset import (
    DatasetGenerateRequest,
    DatasetResponse,
    DatasetSummary,
    FaultAnnotationSchema,
    ValidationSummary,
)
from app.services.artifact_store import ArtifactStore, ArtifactStoreError

logger = logging.getLogger(__name__)
_settings = get_settings()

# Maximum number of points returned in signal_preview (for waveform chart)
_PREVIEW_POINTS = 500


class DatasetServiceError(Exception):
    """Raised when the service layer detects a non-recoverable condition."""


class DatasetService:
    """Owns all business logic for dataset generation and retrieval."""

    def __init__(self, db: Session) -> None:
        self._db = db
        self._repo = DatasetRepository(db)
        self._engine = SyntheticDataEngine()
        self._store = ArtifactStore(root=_settings.artifact_root_path)

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(self, request: DatasetGenerateRequest) -> DatasetResponse:
        config = request.to_engine_config()

        # Assign human-readable ID before anything else so it can be used
        # as the artifact directory name.
        total = self._repo.count()
        human_id = f"DS-{total + 1:04d}"
        safe_id = human_id.replace("-", "_")  # e.g. DS_0001 — safe for filesystem

        # Create DB record immediately so the ID exists during generation
        record = self._repo.create(
            human_id=human_id,
            name=request.name,
            seed=request.seed,
            configuration=config,
            status="PENDING",
        )
        self._repo.commit()
        dataset_id = record.id

        try:
            # Generate data
            raw, windowed, validation = self._engine.generate(config)

            # Persist artifact
            artifact_dir = self._store.dataset_path(safe_id)
            self._store.ensure_dir(artifact_dir)
            self._persist_npz(artifact_dir, raw)
            artifact_rel = self._store.relative_path(artifact_dir)

            # Build validation summary for DB
            val_summary = self._build_validation_summary(raw, validation)

            # Update DB record → COMPLETED
            self._repo.update(
                record,
                status="COMPLETED",
                artifact_path=artifact_rel,
                sample_count=len(raw.timestamps),
                window_count=windowed.total_windows,
                fault_count=len(raw.fault_annotations),
                validation_summary=val_summary,
                configuration={
                    **config,
                    "human_id": human_id,
                },
            )
            self._repo.commit()

        except (DataEngineError, ArtifactStoreError, Exception) as exc:
            logger.exception("Dataset generation failed for %s", human_id)
            try:
                self._repo.update(record, status="FAILED")
                self._repo.commit()
            except Exception:
                self._repo.rollback()
            raise DatasetServiceError(f"Dataset generation failed: {exc}") from exc

        return self._to_response(record, raw, windowed, validation)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def list_datasets(
        self, limit: int = 50, offset: int = 0
    ) -> tuple[list[DatasetSummary], int]:
        rows, total = self._repo.list_all(limit=limit, offset=offset)
        summaries = [self._to_summary(r) for r in rows]
        return summaries, total

    def get_dataset(self, dataset_id: str) -> DatasetResponse:
        record = self._repo.get_by_id(dataset_id)
        if record is None:
            raise DatasetServiceError(f"Dataset not found: {dataset_id}")

        if record.status != "COMPLETED":
            # Return a minimal response for non-completed datasets
            return self._partial_response(record)

        # Try to load full data from artifact
        try:
            artifact_path = _settings.artifact_root_path / record.artifact_path
            raw, windowed = self._load_artifact(artifact_path, record)
            return self._to_response(record, raw, windowed, validation=None)
        except Exception as exc:
            logger.warning("Could not load artifact for %s: %s", dataset_id, exc)
            return self._partial_response(record)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _persist_npz(self, artifact_dir: Path, raw: RawDataset) -> None:
        """Save the numerical dataset as a compressed NPZ file."""
        npz_path = artifact_dir / "data.npz"
        np.savez_compressed(
            str(npz_path),
            timestamps=raw.timestamps,
            values=raw.values,
            original_values=raw.original_values,
            labels=raw.labels,
        )

        meta = {
            "seed": raw.seed,
            "signal_type": raw.signal_type.value,
            "configuration": raw.configuration,
            "fault_annotations": [a.to_dict() for a in raw.fault_annotations],
            "sample_count": len(raw.timestamps),
        }
        metadata_path = artifact_dir / "metadata.json"
        metadata_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    def _load_artifact(
        self, artifact_dir: Path, record: Dataset
    ) -> tuple[RawDataset, WindowedDataset]:
        """Reconstruct RawDataset from the NPZ + metadata on disk."""
        from app.data.models import FaultAnnotation, SignalType
        from app.data.windowing import WindowingEngine

        npz = np.load(str(artifact_dir / "data.npz"))
        meta = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

        annotations = [
            FaultAnnotation(
                fault_id=a["fault_id"],
                fault_type=FaultType(a["fault_type"]),
                start_index=int(a["start_index"]),
                end_index=int(a["end_index"]),
                severity=float(a["severity"]),
                parameters=a.get("parameters", {}),
            )
            for a in meta.get("fault_annotations", [])
        ]

        raw = RawDataset(
            timestamps=npz["timestamps"],
            values=npz["values"],
            original_values=npz["original_values"],
            labels=npz["labels"],
            fault_annotations=annotations,
            signal_type=SignalType(meta["signal_type"]),
            seed=int(meta["seed"]),
            configuration=meta["configuration"],
        )

        window_size = int(record.configuration.get("window_size", 128))
        windowed = WindowingEngine(window_size=window_size).build(raw)
        return raw, windowed

    def _build_validation_summary(
        self, raw: RawDataset, validation: Any
    ) -> dict[str, Any]:
        valid_vals = raw.values[~np.isnan(raw.values) & ~np.isinf(raw.values)]
        return {
            "valid": validation.valid,
            "issue_count": len(validation.issues),
            "nan_count": int(np.sum(np.isnan(raw.values))),
            "gap_count": validation.temporal.gap_count,
            "annotation_count": len(raw.fault_annotations),
            "fault_types_present": validation.ground_truth.fault_types_present,
            "statistics": {
                "mean": float(np.mean(valid_vals)) if len(valid_vals) else 0.0,
                "std": float(np.std(valid_vals)) if len(valid_vals) else 0.0,
                "min": float(np.min(valid_vals)) if len(valid_vals) else 0.0,
                "max": float(np.max(valid_vals)) if len(valid_vals) else 0.0,
            },
        }

    def _downsample(
        self, timestamps: np.ndarray, values: np.ndarray
    ) -> list[list[float]]:
        """Return up to _PREVIEW_POINTS [t, v] pairs for the waveform chart."""
        n = len(timestamps)
        if n <= _PREVIEW_POINTS:
            idx = np.arange(n)
        else:
            idx = np.round(np.linspace(0, n - 1, _PREVIEW_POINTS)).astype(int)

        preview = []
        for i in idx:
            v = values[i]
            # NaN → null in JSON (stored as Python None — Pydantic serialises it correctly)
            preview.append([float(timestamps[i]), None if np.isnan(v) else float(v)])
        return preview

    def _to_summary(self, record: Dataset) -> DatasetSummary:
        cfg = record.configuration
        signal_cfg = cfg.get("signal", {})
        return DatasetSummary(
            dataset_id=record.id,
            human_id=record.human_id,
            name=record.name,
            status=record.status,
            seed=record.seed,
            sample_count=record.sample_count,
            window_count=record.window_count,
            fault_count=record.fault_count,
            signal_type=signal_cfg.get("type", "unknown"),
            duration=signal_cfg.get("duration", 0.0),
            sampling_rate=signal_cfg.get("sampling_rate", 0.0),
            created_at=record.created_at,
        )

    def _to_response(
        self,
        record: Dataset,
        raw: RawDataset,
        windowed: WindowedDataset,
        validation: Any,
    ) -> DatasetResponse:
        cfg = record.configuration
        signal_cfg = cfg.get("signal", {})

        annotations = [
            FaultAnnotationSchema(
                fault_id=a.fault_id,
                fault_type=a.fault_type.value,
                start_index=a.start_index,
                end_index=a.end_index,
                severity=a.severity,
                parameters=a.parameters,
            )
            for a in raw.fault_annotations
        ]

        if validation is not None:
            val_summary = self._build_validation_summary(raw, validation)
        else:
            val_summary = record.validation_summary or {}

        return DatasetResponse(
            dataset_id=record.id,
            human_id=record.human_id,
            name=record.name,
            status=record.status,
            seed=record.seed,
            sample_count=record.sample_count,
            window_count=record.window_count,
            fault_count=record.fault_count,
            signal_type=signal_cfg.get("type", raw.signal_type.value),
            duration=signal_cfg.get("duration", 0.0),
            sampling_rate=signal_cfg.get("sampling_rate", 0.0),
            fault_annotations=annotations,
            validation=ValidationSummary(**val_summary) if isinstance(val_summary, dict) and val_summary else ValidationSummary(
                valid=True,
                issue_count=0,
                nan_count=0,
                gap_count=0,
                annotation_count=0,
                fault_types_present=[],
                statistics={},
            ),
            signal_preview=self._downsample(raw.timestamps, raw.values),
            split_counts={
                "train": len(windowed.splits.train),
                "validation": len(windowed.splits.validation),
                "iid_test": len(windowed.splits.iid_test),
                "shift_test": len(windowed.splits.shift_test),
            },
            configuration=cfg,
            artifact_path=record.artifact_path or "",
            created_at=record.created_at,
        )

    def _partial_response(self, record: Dataset) -> DatasetResponse:
        """Minimal response for datasets that aren't COMPLETED."""
        cfg = record.configuration
        signal_cfg = cfg.get("signal", {})
        val_summary = record.validation_summary or {}

        return DatasetResponse(
            dataset_id=record.id,
            human_id=record.human_id,
            name=record.name,
            status=record.status,
            seed=record.seed,
            sample_count=record.sample_count,
            window_count=record.window_count,
            fault_count=record.fault_count,
            signal_type=signal_cfg.get("type", "unknown"),
            duration=signal_cfg.get("duration", 0.0),
            sampling_rate=signal_cfg.get("sampling_rate", 0.0),
            fault_annotations=[],
            validation=ValidationSummary(
                valid=val_summary.get("valid", False),
                issue_count=val_summary.get("issue_count", 0),
                nan_count=val_summary.get("nan_count", 0),
                gap_count=val_summary.get("gap_count", 0),
                annotation_count=val_summary.get("annotation_count", 0),
                fault_types_present=val_summary.get("fault_types_present", []),
                statistics=val_summary.get("statistics", {}),
            ),
            signal_preview=[],
            split_counts={},
            configuration=cfg,
            artifact_path=record.artifact_path or "",
            created_at=record.created_at,
        )
