"""Distribution Shift Engine — SynthQuanta Phase 4.

Tests the same trained model against deliberately shifted data distributions.
NEVER retrains the model.  NEVER modifies model weights.

Five required shift scenarios (§22–§26 of phase spec):
    1. noise_shift       — higher noise σ
    2. amplitude_shift   — higher signal amplitude
    3. frequency_shift   — higher carrier frequency
    4. severity_shift    — more extreme fault magnitudes
    5. compound_shift    — multiple simultaneous faults

Each scenario:
    - builds a modified data-generation config
    - calls SyntheticDataEngine to generate shifted data (deterministic with seed)
    - loads shifted windows into SensorWindowDataset using training norm_stats
    - runs model inference
    - computes metrics vs. IID baseline
    - returns ShiftScenarioResult
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from app.data.engine import SyntheticDataEngine
from app.ml.datasets.dataset import SensorWindowDataset, _majority_label, _temporal_splits
from app.ml.evaluation.engine import evaluate_dataset
from app.ml.evaluation.metrics import ClassificationMetrics
from app.ml.models.sensor_transformer import SensorTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

SHIFT_SCENARIOS = [
    "noise_shift",
    "amplitude_shift",
    "frequency_shift",
    "severity_shift",
    "compound_shift",
]


@dataclass
class ShiftScenarioResult:
    scenario: str
    iid_macro_f1: float
    shifted_macro_f1: float
    absolute_degradation: float
    relative_degradation: float
    robustness_ratio: float
    n_shifted_windows: int
    configuration: dict
    seed: int
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "iid_macro_f1": self.iid_macro_f1,
            "shifted_macro_f1": self.shifted_macro_f1,
            "absolute_degradation": self.absolute_degradation,
            "relative_degradation": self.relative_degradation,
            "robustness_ratio": self.robustness_ratio,
            "n_shifted_windows": self.n_shifted_windows,
            "configuration": self.configuration,
            "seed": self.seed,
            "error": self.error,
        }


class DistributionShiftEngine:
    """Evaluate a trained model under five distribution-shift scenarios.

    Args:
        model:         Trained SensorTransformer (must be in eval() mode).
        base_config:   Original dataset generation config (from artifact metadata).
        norm_stats:    (mean, std) from the training split — used to normalise shifted data.
        iid_f1:        Macro F1 already computed on the IID test split.
        window_size:   Window size used during training/evaluation.
        seed_offset:   Added to base seed for each shift scenario (for determinism).
        batch_size:    Inference batch size.
    """

    def __init__(
        self,
        model: SensorTransformer,
        base_config: dict[str, Any],
        norm_stats: tuple[float, float],
        iid_f1: float,
        window_size: int,
        seed_offset: int = 10000,
        batch_size: int = 64,
    ) -> None:
        self._model = model
        self._base_config = base_config
        self._norm_stats = norm_stats
        self._iid_f1 = iid_f1
        self._window_size = window_size
        self._seed_offset = seed_offset
        self._batch_size = batch_size
        self._engine = SyntheticDataEngine()
        self._device = next(model.parameters()).device

    def run_all(self) -> list[ShiftScenarioResult]:
        results = []
        for scenario in SHIFT_SCENARIOS:
            logger.info("Running shift scenario: %s", scenario)
            result = self._run_scenario(scenario)
            results.append(result)
        return results

    def _run_scenario(self, scenario: str) -> ShiftScenarioResult:
        base_seed = int(self._base_config.get("seed", 42))
        shift_seed = base_seed + self._seed_offset + SHIFT_SCENARIOS.index(scenario)

        try:
            shifted_cfg = self._build_shifted_config(scenario, shift_seed)
            shifted_ds = self._generate_shifted_dataset(shifted_cfg)

            if len(shifted_ds) == 0:
                raise ValueError("Shifted dataset produced 0 windows")

            with torch.no_grad():
                metrics, _, _, _ = evaluate_dataset(
                    self._model, shifted_ds,
                    batch_size=self._batch_size,
                    device=self._device,
                )

            shifted_f1 = metrics.macro_f1
            iid_f1 = self._iid_f1
            absolute_deg = iid_f1 - shifted_f1
            relative_deg = absolute_deg / iid_f1 if iid_f1 > 0.0 else 0.0
            robustness = shifted_f1 / iid_f1 if iid_f1 > 0.0 else 0.0

            logger.info(
                "  %s: iid_f1=%.4f  shifted_f1=%.4f  Δ=%.4f  robustness=%.4f",
                scenario, iid_f1, shifted_f1, absolute_deg, robustness,
            )
            return ShiftScenarioResult(
                scenario=scenario,
                iid_macro_f1=iid_f1,
                shifted_macro_f1=shifted_f1,
                absolute_degradation=absolute_deg,
                relative_degradation=relative_deg,
                robustness_ratio=robustness,
                n_shifted_windows=len(shifted_ds),
                configuration=shifted_cfg,
                seed=shift_seed,
            )

        except Exception as exc:
            logger.exception("Shift scenario %s failed: %s", scenario, exc)
            return ShiftScenarioResult(
                scenario=scenario,
                iid_macro_f1=self._iid_f1,
                shifted_macro_f1=0.0,
                absolute_degradation=self._iid_f1,
                relative_degradation=1.0,
                robustness_ratio=0.0,
                n_shifted_windows=0,
                configuration={},
                seed=shift_seed,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Config builders for each scenario
    # ------------------------------------------------------------------

    def _build_shifted_config(self, scenario: str, seed: int) -> dict[str, Any]:
        cfg = copy.deepcopy(self._base_config)
        cfg["seed"] = seed
        sig = cfg.setdefault("signal", {})
        faults = cfg.setdefault("faults", {})

        if scenario == "noise_shift":
            # Increase noise std 3× (train σ≈0.10, shift σ≈0.30)
            noise = faults.setdefault("noise", {})
            base_std = float(noise.get("std", 0.10))
            noise["enabled"] = True
            noise["std"] = max(base_std * 3.0, 0.30)
            noise.setdefault("start_frac", 0.2)
            noise.setdefault("end_frac", 0.8)

        elif scenario == "amplitude_shift":
            # Double the signal amplitude
            base_amp = float(sig.get("amplitude", 1.0))
            sig["amplitude"] = base_amp * 2.0

        elif scenario == "frequency_shift":
            # Increase carrier frequency by 50%
            base_freq = float(sig.get("frequency", 10.0))
            sig["frequency"] = base_freq * 1.5

        elif scenario == "severity_shift":
            # Make existing faults more extreme
            if faults.get("noise", {}).get("enabled"):
                faults["noise"]["std"] = float(faults["noise"].get("std", 0.1)) * 4.0
            if faults.get("drift", {}).get("enabled"):
                faults["drift"]["magnitude"] = float(faults["drift"].get("magnitude", 0.5)) * 3.0
            if faults.get("dropout", {}).get("enabled"):
                faults["dropout"]["severity"] = 1.0  # full dropout
            if faults.get("clipping", {}).get("enabled"):
                # Tighten clipping bounds (more extreme clipping)
                lower = float(faults["clipping"].get("lower", -1.0))
                upper = float(faults["clipping"].get("upper", 1.0))
                faults["clipping"]["lower"] = lower * 0.5
                faults["clipping"]["upper"] = upper * 0.5
            # If no faults in base, add a severe noise fault
            if not any(v.get("enabled") for v in faults.values() if isinstance(v, dict)):
                faults["noise"] = {"enabled": True, "std": 0.5, "start_frac": 0.3, "end_frac": 0.7}

        elif scenario == "compound_shift":
            # Multiple simultaneous faults regardless of original config
            faults["noise"] = {"enabled": True, "std": 0.15, "start_frac": 0.2, "end_frac": 0.8}
            faults["drift"] = {
                "enabled": True, "magnitude": 0.4, "direction": "positive",
                "start_frac": 0.3, "end_frac": 0.7,
            }
            faults["clipping"] = {"enabled": True, "lower": -1.2, "upper": 1.2}

        return cfg

    def _generate_shifted_dataset(self, cfg: dict[str, Any]) -> SensorWindowDataset:
        """Generate shifted data and return as SensorWindowDataset using training norm_stats."""
        raw, _, _ = self._engine.generate(cfg)

        n = len(raw.values)
        win_size = self._window_size
        stride = win_size
        n_windows = (n - win_size) // stride + 1 if n >= win_size else 0
        if n_windows == 0:
            raise ValueError(
                f"Shifted data has only {n} samples — not enough for window_size={win_size}"
            )

        values = raw.values.astype(np.float32)
        labels = raw.labels.astype(np.int64)

        win_values = np.stack([
            values[i * stride : i * stride + win_size] for i in range(n_windows)
        ])
        win_labels = np.array([
            _majority_label(labels[i * stride : i * stride + win_size])
            for i in range(n_windows)
        ], dtype=np.int64)

        # Use training norm_stats (no leakage from shifted data)
        return SensorWindowDataset(win_values, win_labels, norm_stats=self._norm_stats)
