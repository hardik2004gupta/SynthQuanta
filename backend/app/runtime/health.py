"""RuntimeHealth dataclass — exposes observable runtime state."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class RuntimeHealth:
    """Point-in-time snapshot of SQRuntime state."""

    status: str                        # uninitialized | loading | ready | failed
    model_id: Optional[str]           # DB UUID of the loaded model
    artifact_path: Optional[str]      # relative artifact path
    precision: Optional[str]          # fp32 | int8
    runtime_variant: Optional[str]    # same as precision — exposed explicitly
    device: str                       # cpu | cuda
    backend: Optional[str]            # quantization backend if int8
    loaded_at: Optional[datetime]     # when the model became READY
    request_count: int                # total inference requests since load
    error: Optional[str]              # last failure message

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "model_id": self.model_id,
            "artifact_path": self.artifact_path,
            "precision": self.precision,
            "runtime_variant": self.runtime_variant,
            "device": self.device,
            "backend": self.backend,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "request_count": self.request_count,
            "error": self.error,
        }
