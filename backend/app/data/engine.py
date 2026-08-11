"""SyntheticDataEngine — orchestrates the complete data generation pipeline.

Pipeline:
    Configuration → Signal Generator → Fault Engine → Ground Truth
    → Validator → Windowing → RawDataset + WindowedDataset
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.data.faults import (
    ClippingFault,
    DriftFault,
    DropoutFault,
    Fault,
    NoiseFault,
    SamplingJitterFault,
    TimestampGapFault,
)
from app.data.models import (
    FAULT_LABEL,
    FaultType,
    RawDataset,
    SignalType,
    WindowedDataset,
)
from app.data.signals import (
    CompositeSignal,
    PeriodicSignal,
    SinusoidalSignal,
    TrendSignal,
)
from app.data.signals.composite import SineComponent
from app.data.signals.periodic import WaveformType
from app.data.validation import DatasetValidator, ValidationResult
from app.data.windowing import WindowingEngine

logger = logging.getLogger(__name__)

# Generator version — bump whenever the generation algorithm changes in a
# way that would produce different output for the same seed/config.
GENERATOR_VERSION = "1.0.0"


class DataEngineError(Exception):
    """Raised when dataset generation fails in an unrecoverable way."""


class SyntheticDataEngine:
    """Entry point for synthetic sensor-data generation.

    Usage:
        engine = SyntheticDataEngine()
        raw, windowed, validation = engine.generate(config)

    All randomness flows from numpy.random.default_rng(seed).
    Same config + same seed → same output, guaranteed.
    """

    def __init__(self) -> None:
        self._validator = DatasetValidator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self, config: dict[str, Any]
    ) -> tuple[RawDataset, WindowedDataset, ValidationResult]:
        """Generate a complete dataset from configuration.

        Args:
            config: Validated generation configuration dict. Must contain at
                    minimum: seed, signal (type, duration, sampling_rate),
                    window_size.

        Returns:
            (raw_dataset, windowed_dataset, validation_result)

        Raises:
            DataEngineError: if generation fails (not a config validation error).
        """
        seed = int(config["seed"])
        rng = np.random.default_rng(seed)

        signal_cfg = config["signal"]
        fault_cfg = config.get("faults", {})
        window_size = int(config.get("window_size", 128))

        # 1 — generate clean signal
        logger.debug("Generating %s signal (duration=%.1fs, rate=%.0fHz)",
                     signal_cfg["type"], signal_cfg["duration"], signal_cfg["sampling_rate"])
        generator = self._build_signal_generator(signal_cfg)
        signal = generator.generate(
            duration=float(signal_cfg["duration"]),
            sampling_rate=float(signal_cfg["sampling_rate"]),
            rng=rng,
        )

        if len(signal.timestamps) < 2:
            raise DataEngineError("Generated signal has fewer than 2 samples")

        original_values = signal.values.copy()
        timestamps = signal.timestamps.copy()
        values = signal.values.copy()

        # 2 — apply faults in deterministic order
        faults = self._build_faults(fault_cfg)
        from app.data.models import FaultAnnotation
        annotations: list[FaultAnnotation] = []
        fault_counter = 0

        for fault in faults:
            fault_id = f"fault_{fault_counter:04d}"
            result = fault.apply(timestamps, values, rng, fault_id)
            timestamps = result.timestamps
            values = result.values
            annotations.append(result.annotation)
            fault_counter += 1

        # 3 — build per-sample label array
        n = len(timestamps)
        labels = np.zeros(n, dtype=np.int32)
        for ann in annotations:
            label = FAULT_LABEL[ann.fault_type]
            labels[ann.start_index: ann.end_index] = label

        raw = RawDataset(
            timestamps=timestamps,
            values=values,
            original_values=original_values,
            fault_annotations=annotations,
            labels=labels,
            signal_type=signal.signal_type,
            seed=seed,
            configuration={
                **config,
                "generator_version": GENERATOR_VERSION,
            },
        )

        # 4 — validate
        validation = self._validator.validate(raw)
        logger.debug(
            "Validation %s — %d issue(s)",
            "PASSED" if validation.valid else "FAILED",
            len(validation.issues),
        )

        # 5 — window
        windowing = WindowingEngine(window_size=window_size)
        windowed = windowing.build(raw)

        logger.info(
            "Dataset generated: %d samples, %d windows, %d faults, seed=%d",
            n, windowed.total_windows, len(annotations), seed,
        )
        return raw, windowed, validation

    # ------------------------------------------------------------------
    # Builder helpers
    # ------------------------------------------------------------------

    def _build_signal_generator(self, cfg: dict[str, Any]):
        sig_type = SignalType(cfg["type"])
        amplitude = float(cfg.get("amplitude", 1.0))
        frequency = float(cfg.get("frequency", 10.0))
        phase = float(cfg.get("phase", 0.0))
        baseline = float(cfg.get("baseline", 0.0))

        if sig_type == SignalType.SINUSOIDAL:
            return SinusoidalSignal(
                amplitude=amplitude,
                frequency=frequency,
                phase=phase,
                baseline=baseline,
            )
        if sig_type == SignalType.COMPOSITE:
            components = []
            raw_comps = cfg.get("components")
            if raw_comps:
                for c in raw_comps:
                    components.append(SineComponent(
                        amplitude=float(c.get("amplitude", 1.0)),
                        frequency=float(c.get("frequency", 10.0)),
                        phase=float(c.get("phase", 0.0)),
                    ))
            else:
                components = [
                    SineComponent(amplitude=amplitude, frequency=frequency, phase=phase),
                    SineComponent(amplitude=amplitude * 0.3, frequency=frequency * 2.5, phase=0.5),
                    SineComponent(amplitude=amplitude * 0.1, frequency=frequency * 5.0, phase=1.0),
                ]
            return CompositeSignal(components=components, baseline=baseline)
        if sig_type == SignalType.TREND:
            return TrendSignal(
                slope=float(cfg.get("slope", 0.05)),
                baseline=baseline,
                overlay_amplitude=float(cfg.get("overlay_amplitude", amplitude * 0.2)),
                overlay_frequency=float(cfg.get("overlay_frequency", frequency)),
                background_noise_std=float(cfg.get("background_noise_std", 0.0)),
            )
        if sig_type == SignalType.PERIODIC:
            return PeriodicSignal(
                amplitude=amplitude,
                frequency=frequency,
                waveform=WaveformType(cfg.get("waveform", "sawtooth")),
                baseline=baseline,
            )
        raise DataEngineError(f"Unknown signal type: {cfg['type']}")

    def _build_faults(self, fault_cfg: dict[str, Any]) -> list[Fault]:
        """Build fault list in deterministic order from configuration."""
        faults: list[Fault] = []

        # Canonical application order (must be stable for reproducibility)
        if fault_cfg.get("noise", {}).get("enabled"):
            cfg = fault_cfg["noise"]
            faults.append(NoiseFault(
                std=float(cfg.get("std", 0.1)),
                start_frac=float(cfg.get("start_frac", 0.2)),
                end_frac=float(cfg.get("end_frac", 0.8)),
            ))

        if fault_cfg.get("drift", {}).get("enabled"):
            cfg = fault_cfg["drift"]
            faults.append(DriftFault(
                magnitude=float(cfg.get("magnitude", 0.5)),
                direction=str(cfg.get("direction", "positive")),
                start_frac=float(cfg.get("start_frac", 0.1)),
                end_frac=float(cfg.get("end_frac", 0.6)),
            ))

        if fault_cfg.get("dropout", {}).get("enabled"):
            cfg = fault_cfg["dropout"]
            faults.append(DropoutFault(
                start_frac=float(cfg.get("start_frac", 0.4)),
                end_frac=float(cfg.get("end_frac", 0.5)),
                severity=float(cfg.get("severity", 1.0)),
            ))

        if fault_cfg.get("clipping", {}).get("enabled"):
            cfg = fault_cfg["clipping"]
            faults.append(ClippingFault(
                lower=float(cfg.get("lower", -1.0)),
                upper=float(cfg.get("upper", 1.0)),
            ))

        if fault_cfg.get("timestamp_gap", {}).get("enabled"):
            cfg = fault_cfg["timestamp_gap"]
            faults.append(TimestampGapFault(
                position_frac=float(cfg.get("position_frac", 0.5)),
                gap_seconds=float(cfg.get("gap_seconds", 1.0)),
            ))

        if fault_cfg.get("sampling_jitter", {}).get("enabled"):
            cfg = fault_cfg["sampling_jitter"]
            faults.append(SamplingJitterFault(
                jitter_std_seconds=float(cfg.get("jitter_std_seconds", 0.001)),
                start_frac=float(cfg.get("start_frac", 0.0)),
                end_frac=float(cfg.get("end_frac", 1.0)),
            ))

        return faults
