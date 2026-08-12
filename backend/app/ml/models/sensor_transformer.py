"""Compact Sensor Quality Transformer — SynthQuanta Phase 3 model.

Architecture (§9, §5–§10 of phase spec):
    Input (batch, window_size, 1)
        ↓  linear projection → embedding_dim
    Temporal Embedding (linear + learned positional)
        ↓
    LayerNorm
        ↓
    N × TransformerBlock (explicit Q/K/V projections — LoRA targets)
        ↓
    Global average pooling
        ↓
    Classification head → 7 classes
        ↓
    ModelOutput (logits, probs, fault_class, confidence)

Designed for local CPU training on small sensor windows.  Do NOT increase
the defaults without evidence that the existing defaults are insufficient.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_CLASSES = 7  # NORMAL + 6 fault types


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """All architectural hyperparameters for SensorTransformer.

    Designed to be small enough to train on CPU in reasonable time.
    """
    architecture: str = "sensor-transformer-small"
    window_size: int = 128        # must match dataset window_size
    input_features: int = 1       # sensor values only
    embedding_dim: int = 32
    num_layers: int = 2
    num_heads: int = 4            # head_dim = embedding_dim / num_heads = 8
    ffn_dim: int = 64
    dropout: float = 0.1
    num_classes: int = NUM_CLASSES

    def validate(self) -> None:
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.window_size < 1:
            raise ValueError(f"window_size must be ≥ 1, got {self.window_size}")
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be ≥ 2, got {self.num_classes}")

    def to_dict(self) -> dict:
        return {
            "architecture": self.architecture,
            "window_size": self.window_size,
            "input_features": self.input_features,
            "embedding_dim": self.embedding_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout,
            "num_classes": self.num_classes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Model output
# ---------------------------------------------------------------------------

@dataclass
class ModelOutput:
    logits: torch.Tensor      # (batch, num_classes)
    probs: torch.Tensor       # (batch, num_classes) — softmax
    fault_class: torch.Tensor  # (batch,) — argmax
    confidence: torch.Tensor   # (batch,) — max probability


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _TransformerBlock(nn.Module):
    """Single Transformer encoder block with explicit Q / K / V projections.

    Using separate linear layers (rather than nn.MultiheadAttention's merged
    in_proj_weight) makes it straightforward to target individual projections
    with LoRA adapters by name: "q_proj", "k_proj", "v_proj", "out_proj".
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d = config.embedding_dim
        self.num_heads = config.num_heads
        self.head_dim = d // config.num_heads
        self.scale = self.head_dim ** -0.5

        # Attention projections (LoRA targets)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d)

        # Feed-forward network
        self.ff1 = nn.Linear(d, config.ffn_dim)
        self.ff2 = nn.Linear(config.ffn_dim, d)

        # Norms and dropout
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.attn_drop = nn.Dropout(config.dropout)
        self.ff_drop = nn.Dropout(config.dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(proj.weight)
        nn.init.kaiming_uniform_(self.ff1.weight, nonlinearity="relu")
        nn.init.zeros_(self.ff1.bias)
        nn.init.xavier_uniform_(self.ff2.weight)
        nn.init.zeros_(self.ff2.bias)

    def _self_attention(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        H, dh = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, dh).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, dh).transpose(1, 2)

        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, d)
        return self.out_proj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual attention
        x = x + self._self_attention(self.norm1(x))
        # Pre-norm residual feed-forward
        x = x + self.ff_drop(self.ff2(F.relu(self.ff1(self.norm2(x)))))
        return x


# ---------------------------------------------------------------------------
# SensorTransformer
# ---------------------------------------------------------------------------

class SensorTransformer(nn.Module):
    """Compact time-series Transformer for 7-class sensor-fault classification."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        d = config.embedding_dim

        # Temporal embedding: project 1 sensor feature → d
        self.embed = nn.Linear(config.input_features, d)
        # Learned positional encodings
        self.pos_embed = nn.Embedding(config.window_size, d)
        # Pre-transformer norm
        self.pre_norm = nn.LayerNorm(d)
        self.embed_drop = nn.Dropout(config.dropout)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList(
            [_TransformerBlock(config) for _ in range(config.num_layers)]
        )

        # Classification head (mean-pooled representation → num_classes)
        self.head = nn.Linear(d, config.num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.embed.weight, nonlinearity="linear")
        nn.init.zeros_(self.embed.bias)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Args:
            x: (batch, window_size, input_features)
        Returns:
            ModelOutput with logits, probs, fault_class, confidence
        """
        if x.dim() != 3:
            raise ValueError(
                f"SensorTransformer expects 3D input (batch, seq_len, features), "
                f"got shape {tuple(x.shape)}"
            )
        B, T, F = x.shape
        if T != self.config.window_size:
            raise ValueError(
                f"Input seq_len {T} does not match config.window_size {self.config.window_size}"
            )
        if F != self.config.input_features:
            raise ValueError(
                f"Input features {F} does not match config.input_features {self.config.input_features}"
            )

        positions = torch.arange(T, device=x.device)
        h = self.embed(x) + self.pos_embed(positions)   # (B, T, d)
        h = self.pre_norm(h)
        h = self.embed_drop(h)

        for block in self.blocks:
            h = block(h)

        # Global average pooling over the time dimension
        h = h.mean(dim=1)   # (B, d)

        logits = self.head(h)                      # (B, num_classes)
        probs = torch.softmax(logits, dim=-1)      # (B, num_classes)
        fault_class = probs.argmax(dim=-1)         # (B,)
        confidence = probs.max(dim=-1).values      # (B,)

        return ModelOutput(
            logits=logits,
            probs=probs,
            fault_class=fault_class,
            confidence=confidence,
        )

    def count_parameters(self) -> tuple[int, int]:
        """Return (total_parameters, trainable_parameters)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Also count buffers that were originally trainable (LoRA frozen weights)
        for buf in self.buffers():
            total += buf.numel()
        return total, trainable
