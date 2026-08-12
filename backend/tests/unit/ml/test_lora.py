"""Unit tests for LoRA adapter implementation."""
import pytest
import torch
import torch.nn as nn

from app.ml.adapters.lora import LoRAConfig, LoRALinear, apply_lora, count_lora_parameters
from app.ml.models.sensor_transformer import ModelConfig, SensorTransformer


@pytest.fixture
def tiny_model() -> SensorTransformer:
    cfg = ModelConfig(window_size=16, embedding_dim=8, num_layers=1, num_heads=2, ffn_dim=16)
    return SensorTransformer(cfg)


def test_lora_config_defaults():
    cfg = LoRAConfig()
    cfg.validate()
    assert cfg.rank == 4


def test_lora_config_invalid_rank():
    with pytest.raises(ValueError):
        LoRAConfig(rank=0).validate()


def test_lora_linear_forward_matches_base():
    """At init, lora_B == 0 so LoRALinear output == base linear output."""
    lin = nn.Linear(8, 8, bias=True)
    lora = LoRALinear.from_linear(lin, rank=2, alpha=4.0, dropout=0.0)
    x = torch.randn(3, 8)
    with torch.no_grad():
        expected = lin(x)
        actual = lora(x)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_lora_linear_buffers_not_in_parameters():
    lin = nn.Linear(8, 8)
    lora = LoRALinear.from_linear(lin, rank=2, alpha=4.0, dropout=0.0)
    param_names = {n for n, _ in lora.named_parameters()}
    assert "weight" not in param_names
    assert "lora_A" in param_names
    assert "lora_B" in param_names


def test_apply_lora_freezes_base_parameters(tiny_model):
    cfg = LoRAConfig(rank=2, target_modules=["q_proj", "k_proj"])
    apply_lora(tiny_model, cfg)
    for name, param in tiny_model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            assert param.requires_grad, f"{name} should be trainable"
        else:
            assert not param.requires_grad, f"{name} should be frozen"


def test_apply_lora_reduces_trainable_params(tiny_model):
    total_before, _ = tiny_model.count_parameters()
    cfg = LoRAConfig(rank=2, target_modules=["q_proj", "k_proj", "v_proj", "out_proj"])
    apply_lora(tiny_model, cfg)
    total_after, trainable_after = count_lora_parameters(tiny_model)
    assert trainable_after < total_before
    assert trainable_after > 0


def test_apply_lora_forward_still_works(tiny_model):
    cfg = LoRAConfig(rank=2)
    apply_lora(tiny_model, cfg)
    x = torch.randn(2, 16, 1)
    out = tiny_model(x)
    assert out.logits.shape == (2, 7)


def test_count_lora_parameters(tiny_model):
    cfg = LoRAConfig(rank=2, target_modules=["q_proj"])
    apply_lora(tiny_model, cfg)
    total, trainable = count_lora_parameters(tiny_model)
    assert total > trainable
    assert trainable > 0


def test_lora_zero_delta_at_init(tiny_model):
    """LoRA starts with zero contribution (lora_B init = 0).

    Use eval mode to disable dropout so both forward passes are deterministic.
    """
    tiny_model.eval()
    x = torch.randn(1, 16, 1)
    with torch.no_grad():
        out_base = tiny_model(x)
    cfg = LoRAConfig(rank=2)
    apply_lora(tiny_model, cfg)
    tiny_model.eval()
    with torch.no_grad():
        out_lora = tiny_model(x)
    # logits should be identical before any gradient step
    assert torch.allclose(out_base.logits, out_lora.logits, atol=1e-5)
