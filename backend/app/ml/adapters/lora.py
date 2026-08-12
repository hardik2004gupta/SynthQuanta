"""LoRA (Low-Rank Adaptation) for SensorTransformer.

Manual implementation — PEFT is not used because PEFT requires a
HuggingFace PreTrainedModel interface.  Our custom SensorTransformer uses
explicit named nn.Linear layers, making direct replacement straightforward.

How it works:
    For each targeted nn.Linear W (shape out × in):
        frozen W is stored as a buffer (no gradient, present in state_dict)
        trainable A ∈ R^{rank × in}   (LoRA down-projection)
        trainable B ∈ R^{out × rank}  (LoRA up-projection, zero-init)
        output = W x + (alpha/rank) * B A x

Default target modules for SensorTransformer:
    q_proj, k_proj, v_proj, out_proj   (attention projections in each block)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    rank: int = 4
    alpha: float = 8.0
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "out_proj"]
    )

    def validate(self) -> None:
        if self.rank < 1:
            raise ValueError(f"LoRA rank must be ≥ 1, got {self.rank}")
        if self.alpha <= 0:
            raise ValueError(f"LoRA alpha must be > 0, got {self.alpha}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"LoRA dropout must be in [0, 1), got {self.dropout}")
        if not self.target_modules:
            raise ValueError("LoRA target_modules list cannot be empty")

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
        }


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with LoRA adapters.

    The base weight is stored as a non-trainable buffer so it is included in
    state_dict() (checkpoint reload works) but excluded from optimizer updates.
    Only lora_A and lora_B are trainable parameters.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling = alpha / rank

        # Frozen base weights stored as buffers
        self.register_buffer("weight", torch.empty(out_features, in_features))
        if has_bias:
            self.register_buffer("bias", torch.zeros(out_features))
        else:
            self.register_buffer("bias", None)  # type: ignore[arg-type]

        # Trainable LoRA adapters
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout)

        # Initialize A with Kaiming, B stays zero → delta W = 0 at init
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, rank: int, alpha: float, dropout: float
    ) -> "LoRALinear":
        """Create a LoRALinear by copying weights from an existing nn.Linear."""
        lora = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            has_bias=linear.bias is not None,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        with torch.no_grad():
            lora.weight.copy_(linear.weight.data)
            if linear.bias is not None and lora.bias is not None:
                lora.bias.copy_(linear.bias.data)
        return lora

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora_delta = (self.lora_dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return base + self.scaling * lora_delta

    def extra_repr(self) -> str:
        r = self.lora_A.shape[0]
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={r}, scaling={self.scaling:.4f}"
        )


# ---------------------------------------------------------------------------
# Applying LoRA to a model
# ---------------------------------------------------------------------------

def apply_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """Replace target nn.Linear layers with LoRALinear wrappers in-place.

    After this call:
    - All parameters that are NOT part of a LoRA adapter are frozen
      (requires_grad = False).
    - Only lora_A and lora_B tensors are trainable.

    Args:
        model:  The model to adapt (modified in-place).
        config: LoRA configuration.

    Returns:
        The same model (modified in-place) for chaining.
    """
    config.validate()

    # Step 1: freeze all existing parameters first
    for param in model.parameters():
        param.requires_grad_(False)

    # Step 2: replace target Linear modules
    _replace_recursive(model, config)

    return model


def _replace_recursive(
    module: nn.Module,
    config: LoRAConfig,
) -> None:
    """Recursively replace named children that match target_modules."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in config.target_modules:
            setattr(
                module,
                name,
                LoRALinear.from_linear(child, config.rank, config.alpha, config.dropout),
            )
        else:
            _replace_recursive(child, config)


def count_lora_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total_parameters, trainable_lora_parameters).

    total_parameters counts all leaf tensors (params + buffers that
    hold frozen base weights).
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_total = sum(p.numel() for p in model.parameters())
    buf_total = sum(b.numel() for b in model.buffers() if b is not None)
    return param_total + buf_total, trainable
