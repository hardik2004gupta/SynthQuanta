"""Unit tests for SensorTransformer model architecture."""
import pytest
import torch

from app.ml.models.sensor_transformer import ModelConfig, ModelOutput, SensorTransformer


@pytest.fixture
def tiny_config() -> ModelConfig:
    return ModelConfig(
        window_size=32,
        input_features=1,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        ffn_dim=16,
        dropout=0.0,
        num_classes=7,
    )


def test_model_config_defaults():
    cfg = ModelConfig()
    cfg.validate()
    assert cfg.num_classes == 7
    assert cfg.window_size == 128


def test_model_config_invalid_heads():
    cfg = ModelConfig(embedding_dim=10, num_heads=3)
    with pytest.raises(ValueError, match="divisible"):
        cfg.validate()


def test_model_config_roundtrip():
    cfg = ModelConfig(embedding_dim=16, num_layers=2, num_heads=4)
    cfg2 = ModelConfig.from_dict(cfg.to_dict())
    assert cfg2.embedding_dim == 16
    assert cfg2.num_heads == 4


def test_forward_output_shape(tiny_config):
    model = SensorTransformer(tiny_config)
    x = torch.randn(4, tiny_config.window_size, tiny_config.input_features)
    out = model(x)
    assert isinstance(out, ModelOutput)
    assert out.logits.shape == (4, 7)
    assert out.probs.shape == (4, 7)
    assert out.fault_class.shape == (4,)
    assert out.confidence.shape == (4,)


def test_forward_probs_sum_to_one(tiny_config):
    model = SensorTransformer(tiny_config)
    x = torch.randn(2, tiny_config.window_size, 1)
    out = model(x)
    assert torch.allclose(out.probs.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_forward_fault_class_is_argmax(tiny_config):
    model = SensorTransformer(tiny_config)
    x = torch.randn(3, tiny_config.window_size, 1)
    out = model(x)
    expected = out.probs.argmax(dim=-1)
    assert torch.equal(out.fault_class, expected)


def test_forward_rejects_wrong_window_size(tiny_config):
    model = SensorTransformer(tiny_config)
    x = torch.randn(1, 99, 1)  # wrong window size
    with pytest.raises(ValueError, match="window_size"):
        model(x)


def test_count_parameters(tiny_config):
    model = SensorTransformer(tiny_config)
    total, trainable = model.count_parameters()
    assert total > 0
    assert trainable == total  # no LoRA applied yet


def test_batch_size_one(tiny_config):
    model = SensorTransformer(tiny_config)
    x = torch.randn(1, tiny_config.window_size, 1)
    out = model(x)
    assert out.logits.shape == (1, 7)


def test_deterministic_with_seed(tiny_config):
    torch.manual_seed(0)
    model1 = SensorTransformer(tiny_config)
    torch.manual_seed(0)
    model2 = SensorTransformer(tiny_config)
    x = torch.randn(2, tiny_config.window_size, 1)
    torch.manual_seed(42)
    out1 = model1(x)
    torch.manual_seed(42)
    out2 = model2(x)
    assert torch.allclose(out1.logits, out2.logits)
