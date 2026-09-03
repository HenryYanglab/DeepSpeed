# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import math
from concurrent.futures import ThreadPoolExecutor

import pytest

from deepspeed.runtime.fastoffload.telemetry.metrics import MetricsRegistry, RunningStats


def test_running_stats_empty_snapshot():
    snapshot = RunningStats().snapshot()

    assert snapshot.count == 0
    assert snapshot.total == 0.0
    assert snapshot.minimum is None
    assert snapshot.maximum is None
    assert snapshot.mean is None


def test_running_stats_observations_and_reset():
    stats = RunningStats()
    for value in (2, 4.0, 9):
        stats.observe(value)

    snapshot = stats.snapshot()
    assert snapshot.count == 3
    assert snapshot.total == 15.0
    assert snapshot.minimum == 2.0
    assert snapshot.maximum == 9.0
    assert snapshot.mean == 5.0

    stats.reset()
    assert stats.snapshot().count == 0


def test_registry_collects_all_metric_types():
    registry = MetricsRegistry()
    registry.increment("gradient_count")
    registry.increment("gradient_bytes", 16)
    registry.increment("gradient_bytes", 32)
    registry.set_gauge("micro_step", 3)
    registry.observe("backward_ms", 4.0)
    registry.observe("backward_ms", 8.0)

    snapshot = registry.snapshot()
    assert snapshot.counters == {"gradient_count": 1, "gradient_bytes": 48}
    assert snapshot.gauges == {"micro_step": 3}
    assert snapshot.histograms["backward_ms"].count == 2
    assert snapshot.histograms["backward_ms"].mean == 6.0


def test_window_reset_retains_gauges():
    registry = MetricsRegistry()
    registry.increment("steps", 2)
    registry.set_gauge("queue_depth", 4)
    registry.observe("step_ms", 10)

    completed_window = registry.snapshot(reset=True)
    new_window = registry.snapshot()

    assert completed_window.counters["steps"] == 2
    assert completed_window.histograms["step_ms"].count == 1
    assert new_window.counters == {}
    assert new_window.histograms == {}
    assert new_window.gauges == {"queue_depth": 4}


def test_reset_can_clear_or_retain_gauges():
    registry = MetricsRegistry()
    registry.set_gauge("queue_depth", 4)

    registry.reset(reset_gauges=False)
    assert registry.snapshot().gauges == {"queue_depth": 4}

    registry.reset()
    assert registry.snapshot().gauges == {}


@pytest.mark.parametrize("name", ["", "   ", None, 1])
def test_invalid_metric_names_are_rejected(name):
    registry = MetricsRegistry()

    with pytest.raises(ValueError):
        registry.increment(name)


@pytest.mark.parametrize("value", [True, "1", None])
def test_non_numeric_metric_values_are_rejected(value):
    registry = MetricsRegistry()

    with pytest.raises(TypeError):
        registry.set_gauge("metric", value)


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_non_finite_metric_values_are_rejected(value):
    registry = MetricsRegistry()

    with pytest.raises(ValueError):
        registry.observe("metric", value)


def test_negative_counter_increment_is_rejected():
    registry = MetricsRegistry()

    with pytest.raises(ValueError):
        registry.increment("bytes", -1)


def test_registry_updates_are_thread_safe():
    registry = MetricsRegistry()
    worker_count = 4
    increments_per_worker = 1000

    def update_metrics():
        for _ in range(increments_per_worker):
            registry.increment("events")
            registry.observe("event_cost", 1)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(update_metrics) for _ in range(worker_count)]
        for future in futures:
            future.result()

    snapshot = registry.snapshot()
    expected_count = worker_count * increments_per_worker
    assert snapshot.counters["events"] == expected_count
    assert snapshot.histograms["event_cost"].count == expected_count
