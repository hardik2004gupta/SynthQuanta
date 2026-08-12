"""Unit tests for BenchmarkEngine — runs against a real SQRuntime."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.benchmarking.engine import BenchmarkEngine, DEFAULT_BATCH_SIZES
from app.ml.adapters.lora import LoRAConfig
from app.ml.datasets.dataset import DatasetFactory
from app.ml.models.sensor_transformer import ModelConfig
from app.ml.training.config import TrainingConfig
from app.ml.training.trainer import Trainer
from app.runtime.runtime import SQRuntime


# ---------------------------------------------------------------------------
# Shared runtime fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ready_runtime(tmp_path_factory):
    """Provide a READY SQRuntime for benchmark engine tests."""
    tmp = tmp_path_factory.mktemp("bench_engine")
    window_size = 16
    n_samples = 400

    rng = np.random.default_rng(55)
    ts = np.linspace(0.0, 4.0, n_samples)
    vs = np.sin(2 * np.pi * 5 * ts).astype(np.float32)
    labels = np.zeros(n_samples, dtype=np.int64)
    labels[80:120] = 2  # DRIFT

    ds_dir = tmp / "DS-BENCH"
    ds_dir.mkdir()
    np.savez(str(ds_dir / "data.npz"), timestamps=ts, values=vs, labels=labels)
    (ds_dir / "metadata.json").write_text(json.dumps({
        "human_id": "DS-BENCH",
        "sample_count": n_samples,
        "window_size": window_size,
        "seed": 55,
        "configuration": {
            "seed": 55,
            "signal": {"type": "sinusoidal", "duration": 4.0, "sampling_rate": 100.0,
                       "amplitude": 1.0, "frequency": 5.0},
            "faults": {},
        },
    }))

    ckpt_dir = tmp / "MODEL-BENCH"
    cfg = TrainingConfig(
        method="lora",
        epochs=1,
        batch_size=16,
        learning_rate=1e-3,
        seed=55,
        lora=LoRAConfig(rank=2, target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]),
        model=ModelConfig(window_size=window_size, embedding_dim=8, num_layers=1,
                         num_heads=2, ffn_dim=16, dropout=0.0),
    )
    splits = DatasetFactory.from_artifact(ds_dir, window_size=window_size)
    Trainer(
        train_dataset=splits.train,
        val_dataset=splits.validation,
        config=cfg,
        checkpoint_dir=ckpt_dir,
        norm_stats=splits.norm_stats,
    ).train()

    runtime = SQRuntime()
    runtime.load(
        artifact_dir=ckpt_dir,
        precision="fp32",
        model_id="model-bench-test",
        artifact_path="models/MODEL-BENCH",
    )
    return runtime, window_size


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBenchmarkEngineSingle:
    def test_single_batch_size_1_completed(self, ready_runtime):
        runtime, window_size = ready_runtime
        engine = BenchmarkEngine()
        result = engine.run_single_batch_size(
            runtime=runtime,
            batch_size=1,
            iterations=10,
            warmup_count=3,
            window_size=window_size,
            seed=42,
        )
        assert result.status == "COMPLETED"

    def test_latency_stats_populated(self, ready_runtime):
        runtime, window_size = ready_runtime
        engine = BenchmarkEngine()
        result = engine.run_single_batch_size(
            runtime=runtime,
            batch_size=1,
            iterations=10,
            warmup_count=2,
            window_size=window_size,
            seed=42,
        )
        assert result.latency_stats.get("p50") is not None
        assert result.latency_stats.get("p95") is not None
        assert result.latency_stats.get("p99") is not None

    def test_throughput_positive(self, ready_runtime):
        runtime, window_size = ready_runtime
        engine = BenchmarkEngine()
        result = engine.run_single_batch_size(
            runtime=runtime,
            batch_size=1,
            iterations=5,
            warmup_count=1,
            window_size=window_size,
            seed=42,
        )
        assert result.throughput_rps > 0.0

    def test_iterations_count_matches(self, ready_runtime):
        runtime, window_size = ready_runtime
        engine = BenchmarkEngine()
        result = engine.run_single_batch_size(
            runtime=runtime,
            batch_size=1,
            iterations=8,
            warmup_count=2,
            window_size=window_size,
            seed=42,
        )
        assert result.iterations == 8
        assert result.latency_stats.get("count") == 8

    def test_batch_size_4(self, ready_runtime):
        runtime, window_size = ready_runtime
        engine = BenchmarkEngine()
        result = engine.run_single_batch_size(
            runtime=runtime,
            batch_size=4,
            iterations=5,
            warmup_count=1,
            window_size=window_size,
            seed=42,
        )
        assert result.status == "COMPLETED"
        assert result.throughput_sps == pytest.approx(result.throughput_rps * 4, rel=1e-3)


class TestBenchmarkEngineMulti:
    def test_run_returns_benchmark_run(self, ready_runtime):
        runtime, _ = ready_runtime
        engine = BenchmarkEngine()
        bench_run = engine.run(
            runtime=runtime,
            batch_sizes=[1, 4],
            iterations=5,
            warmup_count=2,
            seed=42,
        )
        assert len(bench_run.batch_results) == 2

    def test_primary_result_is_batch_1(self, ready_runtime):
        runtime, _ = ready_runtime
        engine = BenchmarkEngine()
        bench_run = engine.run(
            runtime=runtime,
            batch_sizes=[1, 4],
            iterations=5,
            warmup_count=2,
            seed=42,
        )
        primary = bench_run.primary_result
        assert primary is not None
        assert primary.batch_size == 1

    def test_to_dict_serializable(self, ready_runtime):
        import json as _json
        runtime, _ = ready_runtime
        engine = BenchmarkEngine()
        bench_run = engine.run(
            runtime=runtime,
            batch_sizes=[1],
            iterations=3,
            warmup_count=1,
            seed=42,
        )
        d = bench_run.to_dict()
        _json.dumps(d)

    def test_not_ready_raises(self):
        rt = SQRuntime()
        engine = BenchmarkEngine()
        with pytest.raises(ValueError, match="READY"):
            engine.run(rt)
