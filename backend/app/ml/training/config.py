"""Training configuration for SynthQuanta Phase 3.

Keeps all hyperparameters in one typed dataclass so every experiment
records exactly what was run.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.ml.adapters.lora import LoRAConfig
from app.ml.models.sensor_transformer import ModelConfig

ALLOWED_METHODS = ("full", "lora", "qlora")


@dataclass
class TrainingConfig:
    """Complete training configuration for one experiment.

    method:
        "full"  — all parameters trainable (standard fine-tuning)
        "lora"  — LoRA adapters on target linear layers; base weights frozen
        "qlora" — QLoRA (requires CUDA + bitsandbytes); fails with diagnostic
                  if the environment cannot support it
    """
    method: str = "lora"
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    validation_frequency: int = 1   # validate every N epochs
    seed: int = 42

    # LoRA / QLoRA hyperparameters (used when method ∈ {"lora", "qlora"})
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # Model architecture (can override defaults)
    model: ModelConfig = field(default_factory=ModelConfig)

    def validate(self) -> None:
        if self.method not in ALLOWED_METHODS:
            raise ValueError(
                f"Training method {self.method!r} is not supported.  "
                f"Choose from: {ALLOWED_METHODS}"
            )
        if self.epochs < 1:
            raise ValueError(f"epochs must be ≥ 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be ≥ 1, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.method in ("lora", "qlora"):
            self.lora.validate()
        self.model.validate()

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "validation_frequency": self.validation_frequency,
            "seed": self.seed,
            "lora": self.lora.to_dict() if self.method in ("lora", "qlora") else None,
            "model": self.model.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        """Reconstruct from a previously serialized dict."""
        lora_d = d.get("lora") or {}
        model_d = d.get("model") or {}
        return cls(
            method=d.get("method", "lora"),
            epochs=d.get("epochs", 5),
            batch_size=d.get("batch_size", 16),
            learning_rate=d.get("learning_rate", 1e-3),
            weight_decay=d.get("weight_decay", 0.01),
            validation_frequency=d.get("validation_frequency", 1),
            seed=d.get("seed", 42),
            lora=LoRAConfig(**lora_d) if lora_d else LoRAConfig(),
            model=ModelConfig(**{k: v for k, v in model_d.items() if k in ModelConfig.__dataclass_fields__}) if model_d else ModelConfig(),
        )
