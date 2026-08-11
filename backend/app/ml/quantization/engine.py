"""QuantizationEngine — applies PyTorch dynamic INT8 quantization.

Architecture (§11, §12 of CLAUDE.md; §5 of Phase 5 spec):
    QuantizationEngine
        ↓
    torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        ↓
    save state dict + metadata to artifact directory
        ↓
    (caller) run comparison via evaluate_dataset()

Rules:
  - Never silently fall back to FP32.  Failure → QuantizationError.
  - FP32 artifact is immutable; INT8 is a separate artifact.
  - Quantized model must produce valid inference results (validated by caller).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn

from app.ml.models.sensor_transformer import ModelConfig, SensorTransformer
from app.ml.quantization.backend_detect import detect_quantized_backend
from app.ml.quantization.config import QuantizationConfig

logger = logging.getLogger(__name__)

_QUANTIZED_FILE = "model_quantized.pt"
_META_FILE = "metadata.json"


class QuantizationError(Exception):
    """Raised when quantization fails.  Caller must mark the job FAILED."""


class QuantizationEngine:
    """Applies dynamic INT8 quantization to a SensorTransformer.

    This engine is independently testable without FastAPI.
    """

    def __init__(self, config: QuantizationConfig | None = None) -> None:
        self._config = config or QuantizationConfig()
        self._config.validate()
        self._backend = detect_quantized_backend()

    @property
    def backend(self) -> str:
        return self._backend

    # ------------------------------------------------------------------
    # Quantize
    # ------------------------------------------------------------------

    def quantize(self, model: SensorTransformer) -> nn.Module:
        """Apply dynamic INT8 quantization.  Returns a new quantized module.

        The original model is NOT modified (quantize_dynamic returns a new object).

        Raises QuantizationError on failure — never swallowed.
        """
        try:
            torch.backends.quantized.engine = self._backend
            model.eval()
            # torch.ao.quantization is the canonical path (torch.quantization is an alias)
            quantized = torch.ao.quantization.quantize_dynamic(
                model,
                {nn.Linear},
                dtype=torch.qint8,
            )
            quantized.eval()
            logger.info(
                "Dynamic INT8 quantization complete (backend=%s)", self._backend
            )
            return quantized
        except Exception as exc:
            raise QuantizationError(
                f"torch.quantization.quantize_dynamic failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------

    def save_quantized(
        self,
        quantized_model: nn.Module,
        artifact_dir: Path,
        source_metadata: dict,
    ) -> tuple[Path, dict]:
        """Save quantized model state dict and metadata.

        Args:
            quantized_model:  The quantized nn.Module (from self.quantize()).
            artifact_dir:     Directory to write files into (created if absent).
            source_metadata:  Checkpoint metadata from the FP32 model (for lineage).

        Returns:
            (model_file_path, persisted_metadata_dict)
        """
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        model_file = artifact_dir / _QUANTIZED_FILE
        meta_file = artifact_dir / _META_FILE

        torch.save(quantized_model.state_dict(), str(model_file))

        meta = {
            "quantization_method": self._config.method,
            "backend": self._backend,
            "model_config": source_metadata.get("config", {}).get("model", {}),
            "norm_stats": source_metadata.get("norm_stats", {}),
            "source_checkpoint_config": source_metadata.get("config", {}),
        }
        meta_file.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

        logger.info("Quantized artifact saved: %s", artifact_dir)
        return model_file, meta

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    @staticmethod
    def load_quantized(artifact_dir: Path) -> tuple[nn.Module, dict]:
        """Reload a previously saved quantized model.

        Reconstructs the model architecture, re-applies quantize_dynamic to
        obtain the correct module structure, then loads the saved state dict.

        Returns (quantized_model, metadata_dict).
        Raises QuantizationError if files are missing or malformed.
        """
        artifact_dir = Path(artifact_dir)
        model_file = artifact_dir / _QUANTIZED_FILE
        meta_file = artifact_dir / _META_FILE

        if not model_file.exists():
            raise QuantizationError(f"Quantized model file not found: {model_file}")
        if not meta_file.exists():
            raise QuantizationError(f"Quantization metadata not found: {meta_file}")

        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        model_cfg_dict = meta.get("model_config", {})
        if not model_cfg_dict:
            raise QuantizationError("metadata.json missing 'model_config' field")

        model_cfg = ModelConfig.from_dict(model_cfg_dict)

        backend = meta.get("backend", detect_quantized_backend())
        torch.backends.quantized.engine = backend

        base = SensorTransformer(model_cfg)
        base.eval()
        quantized = torch.ao.quantization.quantize_dynamic(
            base, {nn.Linear}, dtype=torch.qint8
        )

        state = torch.load(str(model_file), map_location="cpu", weights_only=False)
        quantized.load_state_dict(state)
        quantized.eval()

        logger.info("Quantized model loaded from %s", artifact_dir)
        return quantized, meta

    # ------------------------------------------------------------------
    # Validate (smoke-test forward pass)
    # ------------------------------------------------------------------

    @staticmethod
    def validate_inference(
        quantized_model: nn.Module,
        window_size: int,
    ) -> None:
        """Run a single forward pass to verify the quantized model is functional.

        Raises QuantizationError if the forward pass fails or produces NaN/Inf.
        """
        x = torch.randn(1, window_size, 1)
        try:
            with torch.no_grad():
                output = quantized_model(x)
        except Exception as exc:
            raise QuantizationError(
                f"Quantized model failed smoke-test forward pass: {exc}"
            ) from exc

        logits = output.logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise QuantizationError(
                "Quantized model produced NaN/Inf logits in smoke-test"
            )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def file_size_bytes(path: Path) -> int:
        """Return file size in bytes, or 0 if file does not exist."""
        try:
            return os.path.getsize(str(path))
        except OSError:
            return 0
