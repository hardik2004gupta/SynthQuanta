"""Unit tests for latency statistics computation."""
import pytest

from app.benchmarking.latency import compute_latency_stats


class TestComputeLatencyStats:
    def test_basic_stats(self):
        latencies = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = compute_latency_stats(latencies)
        assert "p50" in stats
        assert "p95" in stats
        assert "p99" in stats
        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert "count" in stats

    def test_count_correct(self):
        stats = compute_latency_stats([1.0, 2.0, 3.0])
        assert stats["count"] == 3

    def test_min_max(self):
        stats = compute_latency_stats([5.0, 10.0, 1.0])
        assert abs(stats["min"] - 1.0) < 1e-6
        assert abs(stats["max"] - 10.0) < 1e-6

    def test_percentile_ordering(self):
        latencies = list(range(1, 101))
        stats = compute_latency_stats([float(x) for x in latencies])
        assert stats["p50"] <= stats["p95"]
        assert stats["p95"] <= stats["p99"]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_latency_stats([])

    def test_single_value(self):
        stats = compute_latency_stats([7.5])
        assert abs(stats["p50"] - 7.5) < 1e-4
        assert abs(stats["mean"] - 7.5) < 1e-4
