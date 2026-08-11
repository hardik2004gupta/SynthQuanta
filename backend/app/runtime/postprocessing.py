"""Runtime postprocessing — converts raw model output to structured prediction.

Class labels must match the training label ordering exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

# Canonical class labels — index matches the integer class label used in training.
# NORMAL=0, NOISE=1, DRIFT=2, DROPOUT=3, CLIPPING=4, TIMESTAMP_GAP=5, SAMPLING_JITTER=6
FAULT_LABELS: list[str] = [
    "NORMAL",
    "NOISE",
    "DRIFT",
    "DROPOUT",
    "CLIPPING",
    "TIMESTAMP_GAP",
    "SAMPLING_JITTER",
]

NUM_CLASSES = len(FAULT_LABELS)


@dataclass
class PredictionResult:
    """Structured output from a single inference call."""

    predicted_class: str
    predicted_class_index: int
    confidence: float                    # max probability [0, 1]
    probabilities: dict[str, float]      # {class_label: probability}
    latency_ms: float                    # wall-clock time of this call


class Postprocessor:
    """Converts ModelOutput logits to a structured PredictionResult."""

    def postprocess(self, output, latency_ms: float) -> PredictionResult:
        """Convert a single-item ModelOutput to PredictionResult.

        Args:
            output:      ModelOutput from SensorTransformer (batch_size == 1).
            latency_ms:  Measured wall-clock time for this forward pass.
        """
        # Take the first (and only) item in the batch
        probs = output.probs[0].detach().cpu()       # (num_classes,)
        fault_class = int(output.fault_class[0].item())
        confidence = float(output.confidence[0].item())

        prob_dict = {
            FAULT_LABELS[i]: round(float(probs[i].item()), 6)
            for i in range(min(len(FAULT_LABELS), probs.shape[0]))
        }

        predicted_label = FAULT_LABELS[fault_class] if fault_class < len(FAULT_LABELS) else "UNKNOWN"

        return PredictionResult(
            predicted_class=predicted_label,
            predicted_class_index=fault_class,
            confidence=round(confidence, 6),
            probabilities=prob_dict,
            latency_ms=round(latency_ms, 4),
        )

    def postprocess_batch(self, output, latency_ms: float) -> list[PredictionResult]:
        """Convert a batched ModelOutput to a list of PredictionResults.

        Latency is split equally across items (batch overhead amortized).
        """
        batch_size = output.probs.shape[0]
        per_item_latency = latency_ms / max(batch_size, 1)

        results = []
        for i in range(batch_size):
            probs_i = output.probs[i].detach().cpu()
            fault_class_i = int(output.fault_class[i].item())
            confidence_i = float(output.confidence[i].item())

            prob_dict = {
                FAULT_LABELS[j]: round(float(probs_i[j].item()), 6)
                for j in range(min(len(FAULT_LABELS), probs_i.shape[0]))
            }
            predicted_label = FAULT_LABELS[fault_class_i] if fault_class_i < len(FAULT_LABELS) else "UNKNOWN"

            results.append(PredictionResult(
                predicted_class=predicted_label,
                predicted_class_index=fault_class_i,
                confidence=round(confidence_i, 6),
                probabilities=prob_dict,
                latency_ms=round(per_item_latency, 4),
            ))
        return results
