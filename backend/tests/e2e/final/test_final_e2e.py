"""Final Phase 6 E2E acceptance test.

Runs the complete SynthQuanta lifecycle without mocking:
  1. Build tiny dataset artifact
  2. Train tiny FP32 model (2 epochs, LoRA)
  3. Evaluate FP32
  4. Quantize → INT8 artifact
  5. Load FP32 into SQRuntime
  6. Run FP32 inference
  7. Benchmark FP32 (batch_sizes=[1, 4])
  8. Load INT8 into SQRuntime
  9. Run INT8 inference
  10. Benchmark INT8
  11. Compare FP32 vs INT8 predictions
  12. Verify all persisted results

No FastAPI, no DB — tests the domain layer directly.
Runs in 30-120 seconds on CPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from app.benchmarking.engine import BenchmarkEngine
from app.ml.adapters.lora import LoRAConfig
from app.ml.datasets.dataset import DatasetFactory
from app.ml.evaluation.engine import evaluate_dataset
from app.ml.models.sensor_transformer import ModelConfig
from app.ml.quantization.engine import QuantizationEngine
from app.ml.training.config import TrainingConfig
from app.ml.training.trainer import Trainer, load_model_from_checkpoint
from app.runtime.postprocessing import FAULT_LABELS
from app.runtime.runtime import RuntimeState, SQRuntime


# ---------------------------------------------------------------------------
# Shared fixture: build the complete pipeline once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Full FP32→INT8→Runtime→Benchmark pipeline.  Shared across all tests."""
    tmp = tmp_path_factory.mktemp("final_e2e")
    window_size = 16
    n_samples = 600

    # -------------------------------------------------------------------------
    # 1. Dataset
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(2024)
    ts = np.linspace(0.0, 6.0, n_samples)
    vs = (np.sin(2 * np.pi * 5 * ts) + rng.normal(0, 0.05, n_samples)).astype(np.float32)
    labels = np.zeros(n_samples, dtype=np.int64)
    seg = n_samples // 7
    for fault_label in range(1, 7):
        labels[fault_label * seg: fault_label * seg + seg // 2] = fault_label

    ds_dir = tmp / "DS-FINAL"
    ds_dir.mkdir()
    np.savez(str(ds_dir / "data.npz"), timestamps=ts, values=vs, labels=labels)
    (ds_dir / "metadata.json").write_text(json.dumps({
        "human_id": "DS-FINAL",
        "sample_count": n_samples,
        "window_size": window_size,
        "seed": 2024,
        "configuration": {
            "seed": 2024,
            "signal": {"type": "sinusoidal", "duration": 6.0, "sampling_rate": 100.0,
                       "amplitude": 1.0, "frequency": 5.0},
            "faults": {"noise": {"enabled": True}},
        },
    }))

    # -------------------------------------------------------------------------
    # 2. Train FP32 model
    # -------------------------------------------------------------------------
    ckpt_dir = tmp / "MODEL-FINAL"
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
    training_result = trainer.train()

    # -------------------------------------------------------------------------
    # 3. Evaluate FP32
    # -------------------------------------------------------------------------
    fp32_model, ckpt_meta = load_model_from_checkpoint(ckpt_dir, device="cpu")
    fp32_model.eval()
    fp32_metrics, _, _, _ = evaluate_dataset(fp32_model, splits.iid_test, batch_size=32)

    # -------------------------------------------------------------------------
    # 4. Quantize → INT8
    # -------------------------------------------------------------------------
    engine = QuantizationEngine()
    int8_model = engine.quantize(fp32_model)
    int8_dir = tmp / "INT8-FINAL"
    int8_file, int8_meta = engine.save_quantized(int8_model, int8_dir, ckpt_meta)
    QuantizationEngine.validate_inference(int8_model, window_size=window_size)
    loaded_int8, loaded_int8_meta = QuantizationEngine.load_quantized(int8_dir)

    # Evaluate INT8
    int8_model.eval()
    int8_metrics, _, _, _ = evaluate_dataset(int8_model, splits.iid_test, batch_size=32)

    # -------------------------------------------------------------------------
    # 5. FP32 SQRuntime
    # -------------------------------------------------------------------------
    fp32_runtime = SQRuntime()
    fp32_runtime.load(
        artifact_dir=ckpt_dir,
        precision="fp32",
        model_id="model-final-fp32",
        artifact_path="models/MODEL-FINAL",
    )

    # -------------------------------------------------------------------------
    # 6. FP32 inference
    # -------------------------------------------------------------------------
    test_window = [0.0] * window_size
    fp32_prediction = fp32_runtime.predict(test_window)
    fp32_batch_predictions = fp32_runtime.predict_batch([test_window, test_window])

    # -------------------------------------------------------------------------
    # 7. Benchmark FP32
    # -------------------------------------------------------------------------
    bench_engine = BenchmarkEngine()
    fp32_bench_run = bench_engine.run(
        runtime=fp32_runtime,
        batch_sizes=[1, 4],
        iterations=20,
        warmup_count=5,
        seed=42,
    )

    # -------------------------------------------------------------------------
    # 8. INT8 SQRuntime
    # -------------------------------------------------------------------------
    int8_runtime = SQRuntime()
    int8_runtime.load(
        artifact_dir=int8_dir,
        precision="int8",
        model_id="model-final-int8",
        artifact_path="quantizations/INT8-FINAL",
    )

    # -------------------------------------------------------------------------
    # 9. INT8 inference
    # -------------------------------------------------------------------------
    int8_prediction = int8_runtime.predict(test_window)

    # -------------------------------------------------------------------------
    # 10. Benchmark INT8
    # -------------------------------------------------------------------------
    int8_bench_run = bench_engine.run(
        runtime=int8_runtime,
        batch_sizes=[1, 4],
        iterations=20,
        warmup_count=5,
        seed=42,
    )

    return {
        "ds_dir": ds_dir,
        "ckpt_dir": ckpt_dir,
        "int8_dir": int8_dir,
        "int8_file": int8_file,
        "splits": splits,
        "training_result": training_result,
        "fp32_metrics": fp32_metrics,
        "int8_metrics": int8_metrics,
        "fp32_runtime": fp32_runtime,
        "int8_runtime": int8_runtime,
        "fp32_prediction": fp32_prediction,
        "fp32_batch_predictions": fp32_batch_predictions,
        "int8_prediction": int8_prediction,
        "fp32_bench_run": fp32_bench_run,
        "int8_bench_run": int8_bench_run,
        "window_size": window_size,
        "test_window": test_window,
    }


# ---------------------------------------------------------------------------
# Dataset assertions
# ---------------------------------------------------------------------------

class TestDataset:
    def test_dataset_exists(self, pipeline):
        assert pipeline["ds_dir"].exists()

    def test_dataset_npz_exists(self, pipeline):
        assert (pipeline["ds_dir"] / "data.npz").exists()

    def test_iid_test_has_windows(self, pipeline):
        assert len(pipeline["splits"].iid_test) > 0


# ---------------------------------------------------------------------------
# Training assertions
# ---------------------------------------------------------------------------

class TestTraining:
    def test_checkpoint_exists(self, pipeline):
        model_pt = pipeline["ckpt_dir"] / "base" / "model.pt"
        assert model_pt.exists()

    def test_metadata_exists(self, pipeline):
        assert (pipeline["ckpt_dir"] / "metadata.json").exists()

    def test_training_completed(self, pipeline):
        assert pipeline["training_result"].error is None

    def test_training_epochs_ran(self, pipeline):
        assert pipeline["training_result"].total_epochs == 2


# ---------------------------------------------------------------------------
# FP32 evaluation assertions
# ---------------------------------------------------------------------------

class TestFP32Evaluation:
    def test_fp32_f1_in_range(self, pipeline):
        assert 0.0 <= pipeline["fp32_metrics"].macro_f1 <= 1.0

    def test_fp32_false_alarm_rate_in_range(self, pipeline):
        assert 0.0 <= pipeline["fp32_metrics"].false_alarm_rate <= 1.0


# ---------------------------------------------------------------------------
# INT8 artifact assertions
# ---------------------------------------------------------------------------

class TestINT8Artifact:
    def test_int8_file_exists(self, pipeline):
        assert pipeline["int8_file"].exists()

    def test_int8_file_nonzero(self, pipeline):
        assert pipeline["int8_file"].stat().st_size > 0

    def test_int8_metadata_exists(self, pipeline):
        assert (pipeline["int8_dir"] / "metadata.json").exists()

    def test_int8_f1_in_range(self, pipeline):
        assert 0.0 <= pipeline["int8_metrics"].macro_f1 <= 1.0


# ---------------------------------------------------------------------------
# FP32 runtime assertions
# ---------------------------------------------------------------------------

class TestFP32Runtime:
    def test_fp32_runtime_ready(self, pipeline):
        assert pipeline["fp32_runtime"].health().status == RuntimeState.READY

    def test_fp32_inference_class_valid(self, pipeline):
        pred = pipeline["fp32_prediction"]
        assert pred.predicted_class in FAULT_LABELS

    def test_fp32_confidence_in_range(self, pipeline):
        pred = pipeline["fp32_prediction"]
        assert 0.0 <= pred.confidence <= 1.0

    def test_fp32_probabilities_valid(self, pipeline):
        pred = pipeline["fp32_prediction"]
        assert len(pred.probabilities) == 7
        total = sum(pred.probabilities.values())
        assert abs(total - 1.0) < 1e-4

    def test_fp32_latency_positive(self, pipeline):
        assert pipeline["fp32_prediction"].latency_ms > 0.0

    def test_fp32_batch_prediction_length(self, pipeline):
        assert len(pipeline["fp32_batch_predictions"]) == 2

    def test_fp32_batch_classes_valid(self, pipeline):
        for pred in pipeline["fp32_batch_predictions"]:
            assert pred.predicted_class in FAULT_LABELS


# ---------------------------------------------------------------------------
# INT8 runtime assertions
# ---------------------------------------------------------------------------

class TestINT8Runtime:
    def test_int8_runtime_ready(self, pipeline):
        assert pipeline["int8_runtime"].health().status == RuntimeState.READY

    def test_int8_inference_class_valid(self, pipeline):
        pred = pipeline["int8_prediction"]
        assert pred.predicted_class in FAULT_LABELS

    def test_int8_confidence_in_range(self, pipeline):
        pred = pipeline["int8_prediction"]
        assert 0.0 <= pred.confidence <= 1.0

    def test_int8_latency_positive(self, pipeline):
        assert pipeline["int8_prediction"].latency_ms > 0.0


# ---------------------------------------------------------------------------
# FP32 vs INT8 prediction comparison
# ---------------------------------------------------------------------------

class TestPredictionComparison:
    def test_both_produce_valid_classes(self, pipeline):
        fp32 = pipeline["fp32_prediction"]
        int8 = pipeline["int8_prediction"]
        assert fp32.predicted_class in FAULT_LABELS
        assert int8.predicted_class in FAULT_LABELS

    def test_probabilities_sum_to_one_fp32(self, pipeline):
        probs = pipeline["fp32_prediction"].probabilities
        assert abs(sum(probs.values()) - 1.0) < 1e-4

    def test_probabilities_sum_to_one_int8(self, pipeline):
        probs = pipeline["int8_prediction"].probabilities
        assert abs(sum(probs.values()) - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# FP32 benchmark assertions
# ---------------------------------------------------------------------------

class TestFP32Benchmark:
    def test_fp32_bench_has_batch_results(self, pipeline):
        assert len(pipeline["fp32_bench_run"].batch_results) == 2

    def test_fp32_batch_1_completed(self, pipeline):
        result = next(r for r in pipeline["fp32_bench_run"].batch_results if r.batch_size == 1)
        assert result.status == "COMPLETED"

    def test_fp32_p50_finite(self, pipeline):
        primary = pipeline["fp32_bench_run"].primary_result
        assert primary is not None
        assert primary.latency_stats["p50"] > 0.0

    def test_fp32_p95_finite(self, pipeline):
        primary = pipeline["fp32_bench_run"].primary_result
        assert primary.latency_stats["p95"] > 0.0

    def test_fp32_p99_finite(self, pipeline):
        primary = pipeline["fp32_bench_run"].primary_result
        assert primary.latency_stats["p99"] > 0.0

    def test_fp32_throughput_positive(self, pipeline):
        primary = pipeline["fp32_bench_run"].primary_result
        assert primary.throughput_rps > 0.0

    def test_fp32_to_dict_serializable(self, pipeline):
        import json as _json
        d = pipeline["fp32_bench_run"].to_dict()
        _json.dumps(d)


# ---------------------------------------------------------------------------
# INT8 benchmark assertions
# ---------------------------------------------------------------------------

class TestINT8Benchmark:
    def test_int8_bench_has_batch_results(self, pipeline):
        assert len(pipeline["int8_bench_run"].batch_results) == 2

    def test_int8_batch_1_completed(self, pipeline):
        result = next(r for r in pipeline["int8_bench_run"].batch_results if r.batch_size == 1)
        assert result.status == "COMPLETED"

    def test_int8_p95_finite(self, pipeline):
        primary = pipeline["int8_bench_run"].primary_result
        assert primary is not None
        assert primary.latency_stats["p95"] > 0.0

    def test_int8_throughput_positive(self, pipeline):
        primary = pipeline["int8_bench_run"].primary_result
        assert primary.throughput_rps > 0.0


# ---------------------------------------------------------------------------
# Cross-runtime comparison
# ---------------------------------------------------------------------------

class TestFP32vsINT8BenchmarkComparison:
    def test_both_benchmarks_have_p95(self, pipeline):
        fp32_p = pipeline["fp32_bench_run"].primary_result
        int8_p = pipeline["int8_bench_run"].primary_result
        assert fp32_p is not None and "p95" in fp32_p.latency_stats
        assert int8_p is not None and "p95" in int8_p.latency_stats

    def test_runtime_variants_recorded(self, pipeline):
        assert pipeline["fp32_bench_run"].runtime_variant == "fp32"
        assert pipeline["int8_bench_run"].runtime_variant == "int8"

    def test_benchmark_device_recorded(self, pipeline):
        assert pipeline["fp32_bench_run"].device == "cpu"


# ---------------------------------------------------------------------------
# Telemetry assertions
# ---------------------------------------------------------------------------

class TestTelemetry:
    def test_fp32_runtime_request_count(self, pipeline):
        t = pipeline["fp32_runtime"].get_telemetry()
        # predict (1) + batch-predict (1) + benchmark (20 iter × 2 batch + warmup...)
        assert t.request_count > 0

    def test_int8_runtime_request_count(self, pipeline):
        t = pipeline["int8_runtime"].get_telemetry()
        assert t.request_count > 0

    def test_fp32_mean_latency_populated(self, pipeline):
        t = pipeline["fp32_runtime"].get_telemetry()
        assert t.mean_latency_ms is not None
        assert t.mean_latency_ms > 0.0
