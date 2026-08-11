"""Unit tests for runtime Postprocessor and PredictionResult."""
import torch
import pytest

from app.ml.models.sensor_transformer import ModelConfig, ModelOutput, SensorTransformer
from app.runtime.postprocessing import (
    FAULT_LABELS,
    NUM_CLASSES,
    Postprocessor,
    PredictionResult,
)


def _make_output(probs=None):
    """Construct a fake ModelOutput for testing."""
    if probs is None:
        probs = torch.zeros(1, NUM_CLASSES)
        probs[0, 0] = 1.0  # NORMAL with 100% confidence
    fault_class = probs.argmax(dim=-1)
    confidence = probs.max(dim=-1).values
    return ModelOutput(
        logits=torch.zeros_like(probs),
        probs=probs,
        fault_class=fault_class,
        confidence=confidence,
    )


class TestFaultLabels:
    def test_num_classes(self):
        assert NUM_CLASSES == 7

    def test_labels_correct(self):
        assert FAULT_LABELS[0] == "NORMAL"
        assert FAULT_LABELS[1] == "NOISE"
        assert FAULT_LABELS[6] == "SAMPLING_JITTER"


class TestPostprocessor:
    def test_normal_prediction(self):
        pp = Postprocessor()
        output = _make_output()
        result = pp.postprocess(output, latency_ms=1.5)
        assert isinstance(result, PredictionResult)
        assert result.predicted_class == "NORMAL"
        assert result.predicted_class_index == 0
        assert abs(result.confidence - 1.0) < 1e-5

    def test_latency_stored(self):
        pp = Postprocessor()
        output = _make_output()
        result = pp.postprocess(output, latency_ms=3.14)
        assert abs(result.latency_ms - 3.14) < 0.01

    def test_probabilities_sum_to_one(self):
        pp = Postprocessor()
        probs = torch.softmax(torch.randn(1, NUM_CLASSES), dim=-1)
        output = _make_output(probs)
        result = pp.postprocess(output, latency_ms=1.0)
        total = sum(result.probabilities.values())
        assert abs(total - 1.0) < 1e-4

    def test_probabilities_has_all_labels(self):
        pp = Postprocessor()
        output = _make_output()
        result = pp.postprocess(output, latency_ms=1.0)
        assert set(result.probabilities.keys()) == set(FAULT_LABELS)

    def test_noise_prediction(self):
        pp = Postprocessor()
        probs = torch.zeros(1, NUM_CLASSES)
        probs[0, 1] = 1.0  # NOISE
        output = _make_output(probs)
        result = pp.postprocess(output, latency_ms=1.0)
        assert result.predicted_class == "NOISE"
        assert result.predicted_class_index == 1

    def test_batch_postprocessing(self):
        pp = Postprocessor()
        probs = torch.zeros(3, NUM_CLASSES)
        probs[0, 0] = 1.0  # NORMAL
        probs[1, 2] = 1.0  # DRIFT
        probs[2, 4] = 1.0  # CLIPPING
        output = ModelOutput(
            logits=torch.zeros(3, NUM_CLASSES),
            probs=probs,
            fault_class=probs.argmax(dim=-1),
            confidence=probs.max(dim=-1).values,
        )
        results = pp.postprocess_batch(output, latency_ms=6.0)
        assert len(results) == 3
        assert results[0].predicted_class == "NORMAL"
        assert results[1].predicted_class == "DRIFT"
        assert results[2].predicted_class == "CLIPPING"
