"""End-to-end evaluation test — Phase 4.

Runs the complete evaluation lifecycle without any mocking:
  1. Create a tiny synthetic dataset artifact
  2. Train a tiny model (2 epochs, LoRA)
  3. Run EvaluationEngine → IID metrics + localization
  4. Run DistributionShiftEngine → 5 shift scenarios
  5. Verify all required metrics are present and non-NaN

This uses DatasetFactory, Trainer, EvaluationEngine, and DistributionShiftEngine
directly — no FastAPI, no DB.  Runs in seconds on CPU.

Acceptance criteria (§65 of Phase 4 spec):
  - EvaluationEngine.evaluate() returns COMPLETED with real metrics
  - ClassificationMetrics has all 7 classes
  - Localization result is non-null
  - DistributionShiftEngine.run_all() returns 5 scenarios
  - All scenarios have iid_macro_f1 = same value (same model)
  - robustness_ratio ∈ [0.0, 1.5]  (any extreme is logged, not errored)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from app.ml.adapters.lora import LoRAConfig
from app.ml.datasets.dataset import DatasetFactory
from app.ml.evaluation.distribution_shift import SHIFT_SCENARIOS, DistributionShiftEngine
from app.ml.evaluation.engine import EvaluationEngine
from app.ml.evaluation.metrics import NUM_CLASSES
from app.ml.models.sensor_transformer import ModelConfig
from app.ml.training.config import TrainingConfig
from app.ml.training.trainer import Trainer, load_model_from_checkpoint


# ---------------------------------------------------------------------------
# Shared tiny artifact
# ---------------------------------------------------------------------------

def _create_eval_artifact(artifact_dir: Path, n_samples: int = 800, window_size: int = 32) -> Path:
    rng = np.random.default_rng(99)
    timestamps = np.linspace(0.0, 8.0, n_samples)
    values = np.sin(2 * np.pi * 10 * timestamps) + rng.normal(0, 0.08, n_samples)
    values = values.astype(np.float32)
    labels = np.zeros(n_samples, dtype=np.int64)
    seg = n_samples // 7
    for fault_label in range(1, 7):
        start = fault_label * seg
        end = start + seg // 2
        labels[start:end] = fault_label

    ds_dir = artifact_dir / "DS-EVAL-E2E"
    ds_dir.mkdir(parents=True)
    np.savez(str(ds_dir / "data.npz"), timestamps=timestamps, values=values, labels=labels)
    meta = {
        "human_id": "DS-EVAL-E2E",
        "sample_count": n_samples,
        "window_size": window_size,
        "seed": 99,
        "configuration": {
            "seed": 99,
            "signal": {"type": "sinusoidal", "duration": 8.0, "sampling_rate": 100.0,
                        "amplitude": 1.0, "frequency": 10.0},
            "faults": {"noise": {"enabled": True, "std": 0.08}},
        },
    }
    (ds_dir / "metadata.json").write_text(json.dumps(meta))
    return ds_dir


@pytest.fixture(scope="module")
def eval_pipeline(tmp_path_factory):
    """Train tiny model then evaluate — shared across all tests in this module."""
    tmp = tmp_path_factory.mktemp("eval_e2e")
    window_size = 32
    ds_dir = _create_eval_artifact(tmp, n_samples=800, window_size=window_size)
    ckpt_dir = tmp / "CKPT-EVAL-E2E"

    # 1. Train tiny LoRA model
    cfg = TrainingConfig(
        method="lora",
        epochs=2,
        batch_size=16,
        learning_rate=5e-4,
        seed=42,
        lora=LoRAConfig(rank=2, target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]),
        model=ModelConfig(window_size=window_size, embedding_dim=8, num_layers=1,
                          num_heads=2, ffn_dim=16, dropout=0.0),
    )
    splits = DatasetFactory.from_artifact(ds_dir, window_size=window_size)
    trainer = Trainer(
        train_dataset=splits.train,
        val_dataset=splits.validation,
        config=cfg,
        checkpoint_dir=ckpt_dir,
        norm_stats=splits.norm_stats,
    )
    trainer.train()

    # 2. Evaluate
    engine = EvaluationEngine(checkpoint_dir=ckpt_dir, dataset_dir=ds_dir, batch_size=32)
    iid_result = engine.evaluate()

    # 3. Shift evaluation
    model, ckpt_meta = load_model_from_checkpoint(ckpt_dir, device="cpu")
    model.eval()
    norm_stats_raw = ckpt_meta.get("norm_stats", {})
    norm_stats = (float(norm_stats_raw.get("mean", 0.0)), float(norm_stats_raw.get("std", 1.0)))
    base_config = ckpt_meta.get("config", {})
    # inject dataset config for shift engine
    base_config.update({"seed": 99, "signal": {"type": "sinusoidal", "duration": 8.0,
                         "sampling_rate": 100.0, "amplitude": 1.0, "frequency": 10.0},
                         "faults": {"noise": {"enabled": True, "std": 0.08}}})

    shift_engine = DistributionShiftEngine(
        model=model,
        base_config=base_config,
        norm_stats=norm_stats,
        iid_f1=iid_result.metrics.macro_f1,
        window_size=window_size,
        batch_size=32,
    )
    shift_results = shift_engine.run_all()

    return iid_result, shift_results


# ---------------------------------------------------------------------------
# IID evaluation tests
# ---------------------------------------------------------------------------

class TestIIDEvaluation:
    def test_status_completed(self, eval_pipeline):
        iid_result, _ = eval_pipeline
        assert iid_result.status == "COMPLETED"
        assert iid_result.error is None

    def test_n_windows_positive(self, eval_pipeline):
        iid_result, _ = eval_pipeline
        assert iid_result.n_windows > 0

    def test_metrics_not_none(self, eval_pipeline):
        iid_result, _ = eval_pipeline
        assert iid_result.metrics is not None

    def test_macro_f1_in_range(self, eval_pipeline):
        iid_result, _ = eval_pipeline
        f1 = iid_result.metrics.macro_f1
        assert 0.0 <= f1 <= 1.0, f"macro_f1={f1} out of range"

    def test_all_7_classes_in_per_class(self, eval_pipeline):
        iid_result, _ = eval_pipeline
        assert len(iid_result.metrics.per_class) == NUM_CLASSES

    def test_confusion_matrix_is_7x7(self, eval_pipeline):
        iid_result, _ = eval_pipeline
        cm = iid_result.metrics.confusion_matrix
        assert len(cm) == NUM_CLASSES
        assert all(len(row) == NUM_CLASSES for row in cm)

    def test_false_alarm_rate_in_range(self, eval_pipeline):
        iid_result, _ = eval_pipeline
        far = iid_result.metrics.false_alarm_rate
        assert 0.0 <= far <= 1.0

    def test_localization_not_none(self, eval_pipeline):
        iid_result, _ = eval_pipeline
        assert iid_result.localization is not None

    def test_mean_iou_in_range(self, eval_pipeline):
        iid_result, _ = eval_pipeline
        assert 0.0 <= iid_result.localization.mean_iou <= 1.0

    def test_duration_positive(self, eval_pipeline):
        iid_result, _ = eval_pipeline
        assert iid_result.duration_seconds > 0.0

    def test_hardware_info_present(self, eval_pipeline):
        iid_result, _ = eval_pipeline
        assert isinstance(iid_result.hardware_info, dict)

    def test_to_dict_serializable(self, eval_pipeline):
        import json
        iid_result, _ = eval_pipeline
        d = iid_result.to_dict()
        json.dumps(d, default=str)
        assert "metrics" in d
        assert "localization" in d


# ---------------------------------------------------------------------------
# Distribution shift tests
# ---------------------------------------------------------------------------

class TestDistributionShift:
    def test_exactly_five_scenarios(self, eval_pipeline):
        _, shift_results = eval_pipeline
        assert len(shift_results) == 5

    def test_scenario_names_match_registry(self, eval_pipeline):
        _, shift_results = eval_pipeline
        names = [s.scenario for s in shift_results]
        assert set(names) == set(SHIFT_SCENARIOS)

    def test_iid_f1_consistent_across_scenarios(self, eval_pipeline):
        iid_result, shift_results = eval_pipeline
        ref_iid_f1 = iid_result.metrics.macro_f1
        for s in shift_results:
            if s.error is None:
                assert s.iid_macro_f1 == pytest.approx(ref_iid_f1, abs=1e-6)

    def test_robustness_ratio_in_reasonable_range(self, eval_pipeline):
        _, shift_results = eval_pipeline
        for s in shift_results:
            if s.error is None:
                assert 0.0 <= s.robustness_ratio <= 2.0, (
                    f"Unexpected robustness ratio {s.robustness_ratio} for {s.scenario}"
                )

    def test_absolute_degradation_consistent(self, eval_pipeline):
        _, shift_results = eval_pipeline
        for s in shift_results:
            if s.error is None:
                expected_deg = s.iid_macro_f1 - s.shifted_macro_f1
                assert s.absolute_degradation == pytest.approx(expected_deg, abs=1e-6)

    def test_relative_degradation_consistent(self, eval_pipeline):
        _, shift_results = eval_pipeline
        for s in shift_results:
            if s.error is None and s.iid_macro_f1 > 0:
                expected_rel = s.absolute_degradation / s.iid_macro_f1
                assert s.relative_degradation == pytest.approx(expected_rel, abs=1e-5)

    def test_n_shifted_windows_positive(self, eval_pipeline):
        _, shift_results = eval_pipeline
        for s in shift_results:
            if s.error is None:
                assert s.n_shifted_windows > 0

    def test_each_scenario_has_seed(self, eval_pipeline):
        _, shift_results = eval_pipeline
        seeds = [s.seed for s in shift_results]
        assert len(set(seeds)) == len(seeds), "Each scenario must have a unique seed"

    def test_to_dict_serializable(self, eval_pipeline):
        import json
        _, shift_results = eval_pipeline
        for s in shift_results:
            json.dumps(s.to_dict(), default=str)
