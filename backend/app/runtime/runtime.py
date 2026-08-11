"""SQRuntime — Synthetic Quality Runtime for SynthQuanta.

Architecture (§12 CLAUDE.md; Phase 6 spec §7–§28):
    API Route
        ↓
    RuntimeService
        ↓
    SQRuntime  ← this module
        ↓
    ModelLoader | Preprocessor | InferenceEngine | Postprocessor | Telemetry

State machine:
    UNINITIALIZED → LOADING → READY → FAILED

Rules:
  - Never report READY before a successful validation forward pass.
  - Never update model parameters during inference.
  - Never silently swallow inference failures — propagate them.
  - All telemetry derived from actual execution.
  - Thread-safe: concurrent requests reuse the loaded model safely.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from app.ml.models.sensor_transformer import ModelConfig, SensorTransformer
from app.ml.quantization.engine import QuantizationEngine
from app.ml.training.trainer import load_model_from_checkpoint
from app.runtime.health import RuntimeHealth
from app.runtime.postprocessing import Postprocessor, PredictionResult
from app.runtime.preprocessing import Preprocessor, PreprocessingError
from app.runtime.telemetry import RuntimeTelemetry

logger = logging.getLogger(__name__)


class RuntimeState:
    UNINITIALIZED = "uninitialized"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


class SQRuntimeError(Exception):
    """Raised when a runtime operation fails."""


class SQRuntime:
    """SynthQuanta model serving runtime.

    One SQRuntime instance holds one loaded model.  It is designed to be
    long-lived (kept in RuntimeService, not recreated per request).

    Thread safety: the _lock guards state transitions.  Concurrent
    torch.no_grad() forward passes are safe because the model is in eval
    mode and parameters are never mutated during serving.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = RuntimeState.UNINITIALIZED
        self._model: nn.Module | None = None
        self._preprocessor: Preprocessor | None = None
        self._postprocessor = Postprocessor()
        self._telemetry = RuntimeTelemetry()
        self._model_id: str | None = None
        self._artifact_path: str | None = None
        self._precision: str | None = None      # "fp32" | "int8"
        self._device: str = "cpu"
        self._backend: str | None = None
        self._loaded_at: datetime | None = None
        self._last_error: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(
        self,
        artifact_dir: Path,
        precision: str,
        model_id: str,
        artifact_path: str,
    ) -> None:
        """Load a model artifact into the runtime.

        Args:
            artifact_dir:   Filesystem path to the artifact directory.
            precision:      "fp32" or "int8".
            model_id:       DB UUID of the model record.
            artifact_path:  Relative path (stored in DB) for reference.

        Raises:
            SQRuntimeError: if loading or validation fails.
        """
        with self._lock:
            self._state = RuntimeState.LOADING
            self._model = None
            self._preprocessor = None
            self._model_id = model_id
            self._artifact_path = artifact_path
            self._precision = precision
            self._last_error = None
            self._telemetry.reset()

        try:
            model, metadata, preprocessor = self._load_artifact(artifact_dir, precision)
            self._validate(model, preprocessor)
            with self._lock:
                self._model = model
                self._preprocessor = preprocessor
                self._backend = metadata.get("backend")
                self._loaded_at = datetime.now(timezone.utc)
                self._state = RuntimeState.READY
            logger.info(
                "SQRuntime READY [precision=%s, model_id=%s]", precision, model_id
            )
        except Exception as exc:
            err_msg = str(exc)
            with self._lock:
                self._state = RuntimeState.FAILED
                self._last_error = err_msg
            logger.error("SQRuntime load FAILED: %s", err_msg)
            raise SQRuntimeError(f"Runtime load failed: {err_msg}") from exc

    def unload(self) -> None:
        """Unload the current model, returning to UNINITIALIZED."""
        with self._lock:
            self._model = None
            self._preprocessor = None
            self._state = RuntimeState.UNINITIALIZED
            self._model_id = None
            self._artifact_path = None
            self._precision = None
            self._backend = None
            self._loaded_at = None
            self._last_error = None
            self._telemetry.reset()
        logger.info("SQRuntime unloaded")

    def is_ready(self) -> bool:
        return self._state == RuntimeState.READY

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, values: list[float]) -> PredictionResult:
        """Run single-window inference.

        Args:
            values: Exactly window_size float values (sensor readings).

        Returns:
            PredictionResult with class, confidence, probabilities, latency.

        Raises:
            SQRuntimeError: if runtime not ready, input invalid, or inference fails.
        """
        model, preprocessor = self._require_ready()
        try:
            tensor = preprocessor.preprocess(values)
        except PreprocessingError as exc:
            raise SQRuntimeError(f"Preprocessing failed: {exc}") from exc

        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                output = model(tensor)
        except Exception as exc:
            self._telemetry.record(0.0, success=False)
            raise SQRuntimeError(f"Inference failed: {exc}") from exc
        latency_ms = (time.perf_counter() - t0) * 1000.0

        self._telemetry.record(latency_ms, success=True)
        return self._postprocessor.postprocess(output, latency_ms)

    def predict_batch(self, windows: list[list[float]]) -> list[PredictionResult]:
        """Run batch inference on multiple windows.

        Args:
            windows: List of windows, each with exactly window_size floats.

        Returns:
            List of PredictionResult, one per input window.

        Raises:
            SQRuntimeError: if runtime not ready, input invalid, or inference fails.
        """
        if not windows:
            raise SQRuntimeError("Batch must contain at least one window")

        model, preprocessor = self._require_ready()
        try:
            tensor = preprocessor.preprocess_batch(windows)
        except PreprocessingError as exc:
            raise SQRuntimeError(f"Batch preprocessing failed: {exc}") from exc

        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                output = model(tensor)
        except Exception as exc:
            self._telemetry.record(0.0, success=False)
            raise SQRuntimeError(f"Batch inference failed: {exc}") from exc
        latency_ms = (time.perf_counter() - t0) * 1000.0

        self._telemetry.record(latency_ms, success=True)
        return self._postprocessor.postprocess_batch(output, latency_ms)

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def health(self) -> RuntimeHealth:
        """Return current runtime health — always safe to call."""
        with self._lock:
            state = self._state
            model_id = self._model_id
            artifact_path = self._artifact_path
            precision = self._precision
            device = self._device
            backend = self._backend
            loaded_at = self._loaded_at
            error = self._last_error
            rc = self._telemetry.to_summary().request_count

        return RuntimeHealth(
            status=state,
            model_id=model_id,
            artifact_path=artifact_path,
            precision=precision,
            runtime_variant=precision,
            device=device,
            backend=backend,
            loaded_at=loaded_at,
            request_count=rc,
            error=error,
        )

    def get_telemetry(self):
        return self._telemetry.to_summary()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_ready(self) -> tuple[nn.Module, Preprocessor]:
        """Return model + preprocessor, or raise if not ready."""
        with self._lock:
            state = self._state
            model = self._model
            preprocessor = self._preprocessor
        if state != RuntimeState.READY or model is None or preprocessor is None:
            raise SQRuntimeError(
                f"Runtime is not ready (current state: {state}). "
                "Load a model first via POST /api/v1/runtime/load."
            )
        return model, preprocessor

    def _load_artifact(
        self, artifact_dir: Path, precision: str
    ) -> tuple[nn.Module, dict, Preprocessor]:
        """Dispatch to FP32 or INT8 loader.  Returns (model, metadata, preprocessor)."""
        artifact_dir = Path(artifact_dir)
        if precision == "fp32":
            return self._load_fp32(artifact_dir)
        elif precision == "int8":
            return self._load_int8(artifact_dir)
        else:
            raise SQRuntimeError(f"Unsupported precision: {precision!r}")

    def _load_fp32(self, checkpoint_dir: Path) -> tuple[nn.Module, dict, Preprocessor]:
        """Load FP32 model from checkpoint directory."""
        model, metadata = load_model_from_checkpoint(checkpoint_dir, device=self._device)
        model.eval()

        window_size = metadata["config"]["model"]["window_size"]
        norm_stats = metadata.get("norm_stats", {})
        mean = float(norm_stats.get("mean", 0.0))
        std = float(norm_stats.get("std", 1.0))

        return model, metadata, Preprocessor(window_size, mean, std)

    def _load_int8(self, artifact_dir: Path) -> tuple[nn.Module, dict, Preprocessor]:
        """Load INT8 quantized model from artifact directory."""
        model, metadata = QuantizationEngine.load_quantized(artifact_dir)
        model.eval()

        model_cfg = metadata.get("model_config", {})
        window_size = model_cfg.get("window_size", 128)
        norm_stats = metadata.get("norm_stats", {})
        mean = float(norm_stats.get("mean", 0.0))
        std = float(norm_stats.get("std", 1.0))

        return model, metadata, Preprocessor(window_size, mean, std)

    def _validate(self, model: nn.Module, preprocessor: Preprocessor) -> None:
        """Run a smoke-test forward pass to confirm the model is executable."""
        import numpy as np
        dummy = np.zeros(preprocessor.window_size, dtype=np.float32)
        tensor = preprocessor.preprocess(dummy)
        with torch.no_grad():
            output = model(tensor)
        logits = output.logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise SQRuntimeError("Validation forward pass produced NaN/Inf logits")
