"""Unit tests for RuntimeTelemetry."""
import pytest

from app.runtime.telemetry import RuntimeTelemetry


class TestRuntimeTelemetry:
    def test_initial_state(self):
        t = RuntimeTelemetry()
        summary = t.to_summary()
        assert summary.request_count == 0
        assert summary.success_count == 0
        assert summary.error_count == 0
        assert summary.last_latency_ms is None

    def test_records_success(self):
        t = RuntimeTelemetry()
        t.record(5.0, success=True)
        s = t.to_summary()
        assert s.request_count == 1
        assert s.success_count == 1
        assert s.error_count == 0
        assert abs(s.last_latency_ms - 5.0) < 1e-9

    def test_records_failure(self):
        t = RuntimeTelemetry()
        t.record(0.0, success=False)
        s = t.to_summary()
        assert s.request_count == 1
        assert s.success_count == 0
        assert s.error_count == 1

    def test_mean_latency(self):
        t = RuntimeTelemetry()
        for lat in [10.0, 20.0, 30.0]:
            t.record(lat, success=True)
        s = t.to_summary()
        assert abs(s.mean_latency_ms - 20.0) < 1e-6

    def test_percentile_values_computed(self):
        t = RuntimeTelemetry()
        for lat in range(1, 101):
            t.record(float(lat), success=True)
        s = t.to_summary()
        assert s.p50_latency_ms is not None
        assert s.p95_latency_ms is not None
        assert s.p99_latency_ms is not None
        assert s.p50_latency_ms <= s.p95_latency_ms
        assert s.p95_latency_ms <= s.p99_latency_ms

    def test_reset_clears_all(self):
        t = RuntimeTelemetry()
        t.record(5.0, success=True)
        t.reset()
        s = t.to_summary()
        assert s.request_count == 0
        assert s.success_count == 0
        assert s.last_latency_ms is None

    def test_to_dict(self):
        t = RuntimeTelemetry()
        t.record(3.0, success=True)
        d = t.to_summary().to_dict()
        assert "request_count" in d
        assert "mean_latency_ms" in d
