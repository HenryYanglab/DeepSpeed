# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import csv
import json
from unittest.mock import patch

import pytest

from deepspeed.runtime.fastoffload.config import TelemetryConfig
from deepspeed.runtime.fastoffload.context import GradientBucketContext, GradientContext, StepContext
from deepspeed.runtime.fastoffload.events import FastOffloadEvent
from deepspeed.runtime.fastoffload.telemetry.metrics import MetricsRegistry
from deepspeed.runtime.fastoffload.telemetry.observer import FastOffloadObserver
from deepspeed.runtime.fastoffload.telemetry.reporter import TelemetryReporter
from deepspeed.runtime.fastoffload.telemetry.timers import HostTimer


def make_step_context(timestamp_ns, **overrides):
    values = {
        "global_step": 0,
        "micro_step": 0,
        "rank": 0,
        "world_size": 1,
        "zero_stage": 2,
        "gradient_accumulation_boundary": True,
        "timestamp_ns": timestamp_ns,
    }
    values.update(overrides)
    return StepContext(**values)


def make_gradient_context(**overrides):
    values = {
        "parameter_id": 7,
        "parameter_name": "decoder.layer.weight",
        "group_id": 0,
        "partition_id": 0,
        "parameter_numel": 16,
        "shard_numel": 8,
        "shard_offset": 0,
        "element_size": 2,
        "shard_bytes": 16,
        "dtype": "torch.float16",
        "device": "cuda:0",
        "global_step": 0,
        "micro_step": 0,
        "rank": 0,
        "world_size": 1,
        "zero_stage": 2,
        "gradient_accumulation_boundary": True,
        "is_local_partition": True,
        "is_cpu_offload": True,
        "timestamp_ns": 100,
    }
    values.update(overrides)
    return GradientContext(**values)


def make_bucket_context(**overrides):
    values = {
        "bucket_id": 1,
        "parameter_count": 2,
        "total_numel": 32,
        "total_bytes": 64,
        "communication_dtype": "torch.float16",
        "overlap_comm": True,
        "global_step": 0,
        "micro_step": 0,
        "rank": 0,
        "world_size": 1,
        "zero_stage": 2,
        "gradient_accumulation_boundary": True,
        "timestamp_ns": 100,
    }
    values.update(overrides)
    return GradientBucketContext(**values)


def test_host_timer_records_elapsed_time_without_suppressing_errors():
    registry = MetricsRegistry()
    with patch("deepspeed.runtime.fastoffload.telemetry.timers.time.perf_counter_ns", side_effect=[100, 2_000_100]):
        with HostTimer(registry, "operation_ms"):
            pass

    assert registry.snapshot().histograms["operation_ms"].mean == 2.0

    with pytest.raises(RuntimeError):
        with HostTimer(registry, "failed_ms"):
            raise RuntimeError("failure")
    assert registry.snapshot().histograms["failed_ms"].count == 1


def test_disabled_host_timer_does_not_record_metric():
    registry = MetricsRegistry()

    with HostTimer(registry, "operation_ms", enabled=False):
        pass

    assert registry.snapshot().histograms == {}


def test_observer_collects_lifecycle_gradient_and_bucket_metrics():
    observer = FastOffloadObserver(TelemetryConfig(log_interval=10))
    observer.on_backward_begin(make_step_context(1_000_000))
    observer.on_gradient_ready(make_gradient_context())
    observer.on_gradient_reduced(make_gradient_context())
    observer.on_gradient_bucket(make_bucket_context())
    observer.on_backward_end(make_step_context(6_000_000))
    observer.on_step_begin(make_step_context(7_000_000))
    observer.on_step_end(make_step_context(10_000_000))

    snapshot = observer.snapshot()
    assert snapshot.counters["backward_count"] == 1
    assert snapshot.counters["gradient_ready_count"] == 1
    assert snapshot.counters["gradient_reduced_count"] == 1
    assert snapshot.counters["gradient_local_shard_numel"] == 8
    assert snapshot.counters["gradient_local_shard_bytes"] == 16
    assert snapshot.counters["potential_offload_bytes"] == 16
    assert snapshot.counters["gradient_bucket_count"] == 1
    assert snapshot.counters["gradient_bucket_bytes"] == 64
    assert snapshot.counters["step_count"] == 1
    assert snapshot.histograms["backward_host_ms"].mean == 5.0
    assert snapshot.histograms["optimizer_step_host_ms"].mean == 3.0
    assert snapshot.gauges["gradient_accumulation_boundary"] == 1


def test_nonlocal_or_nonoffloaded_gradient_is_not_potential_offload():
    observer = FastOffloadObserver(TelemetryConfig(host_timing=False))
    observer.on_gradient_reduced(make_gradient_context(is_local_partition=False))
    observer.on_gradient_reduced(make_gradient_context(is_cpu_offload=False))

    snapshot = observer.snapshot()
    assert snapshot.counters["gradient_reduced_count"] == 2
    assert snapshot.counters["gradient_local_shard_bytes"] == 16
    assert "potential_offload_bytes" not in snapshot.counters


def test_disabled_parameter_and_bucket_events_collect_no_data():
    config = TelemetryConfig(parameter_events=False, bucket_events=False, host_timing=False)
    observer = FastOffloadObserver(config)

    observer.on_gradient_ready(make_gradient_context())
    observer.on_gradient_reduced(make_gradient_context())
    observer.on_gradient_bucket(make_bucket_context())

    snapshot = observer.snapshot()
    assert snapshot.counters == {}
    assert snapshot.histograms == {}


def test_debug_event_buffer_has_fixed_capacity():
    observer = FastOffloadObserver(TelemetryConfig(debug_event_buffer_size=2, host_timing=False))
    observer.on_backward_begin(make_step_context(1))
    observer.on_gradient_ready(make_gradient_context())
    observer.on_backward_end(make_step_context(2))

    events = observer.debug_events
    assert len(events) == 2
    assert events[0][0] == FastOffloadEvent.gradient_ready
    assert events[1][0] == FastOffloadEvent.backward_end


def test_observer_rejects_invalid_lifecycle_order():
    observer = FastOffloadObserver(TelemetryConfig(host_timing=False))

    with pytest.raises(RuntimeError, match="without backward begin"):
        observer.on_backward_end(make_step_context(1))
    with pytest.raises(RuntimeError, match="without step begin"):
        observer.on_step_end(make_step_context(1))


def test_observer_rejects_reversed_phase_timestamps():
    observer = FastOffloadObserver(TelemetryConfig(host_timing=False))
    observer.on_backward_begin(make_step_context(10))

    with pytest.raises(ValueError, match="precedes"):
        observer.on_backward_end(make_step_context(9))


def test_reporter_emits_at_completed_step_interval():
    messages = []
    config = TelemetryConfig(log_interval=2, host_timing=False)
    reporter = TelemetryReporter(config, logging_fn=messages.append)
    observer = FastOffloadObserver(config, reporter=reporter)

    for step in range(2):
        observer.on_step_begin(make_step_context(step * 10, global_step=step))
        observer.on_step_end(make_step_context(step * 10 + 5, global_step=step))

    assert len(messages) == 1
    assert "completed_steps=2" in messages[0]
    assert "gradient_shards=0" in messages[0]


def test_default_reporter_prints_rank_zero_report(capsys):
    reporter = TelemetryReporter(TelemetryConfig())

    assert reporter.report(MetricsRegistry().snapshot(), completed_steps=1, rank=0) is True

    output = capsys.readouterr().out
    assert "[FastOffload Observer]" in output
    assert "completed_steps=1" in output


def test_reporter_rank_mode_controls_output():
    snapshot = MetricsRegistry().snapshot()
    rank_zero_messages = []
    rank_zero_reporter = TelemetryReporter(TelemetryConfig(), logging_fn=rank_zero_messages.append)
    assert rank_zero_reporter.report(snapshot, completed_steps=1, rank=1) is False
    assert rank_zero_messages == []

    per_rank_messages = []
    per_rank_config = TelemetryConfig(rank_mode="per_rank")
    per_rank_reporter = TelemetryReporter(per_rank_config, logging_fn=per_rank_messages.append)
    assert per_rank_reporter.report(snapshot, completed_steps=1, rank=1) is True
    assert "rank=1" in per_rank_messages[0]


def test_reporter_formats_durations_and_bytes():
    registry = MetricsRegistry()
    registry.increment("gradient_reduced_count", 3)
    registry.increment("potential_offload_bytes", 1024**3)
    registry.increment("actual_offload_bytes", 512 * 1024**2)
    registry.observe("gpu_staging_copy_ms", 0.25)
    registry.observe("copy_submit_host_ms", 0.125)
    registry.observe("d2h_copy_ms", 2.5)
    registry.observe("d2h_bandwidth_gbps", 10.25)
    registry.observe("inline_worker_ms", 1.5)
    registry.set_gauge("buffer_pool_high_watermark", 1)
    registry.set_gauge("queue_high_watermark", 3)
    registry.observe("queue_wait_ms", 0.5)
    registry.observe("step_flush_ms", 0.25)
    registry.observe("transfer_hidden_ratio", 0.75)
    registry.observe("gpu_source_hold_ms", 3.5)
    registry.observe("backward_host_ms", 12.5)
    registry.observe("optimizer_step_host_ms", 4.25)
    registry.observe("observer_callback_ms", 0.125)

    lines = TelemetryReporter.format_report(registry.snapshot(), completed_steps=2, rank=0)
    report = "\n".join(lines)
    assert "backward_mean_ms=12.50" in report
    assert "optimizer_step_mean_ms=4.25" in report
    assert "potential_offload=1.00 GiB" in report
    assert "actual_offload=512.00 MiB" in report
    assert "gpu_staging_copy_mean_ms=0.25" in report
    assert "copy_submit_host_mean_ms=0.1250" in report
    assert "d2h_copy_mean_ms=2.50" in report
    assert "d2h_bandwidth_mean_gbps=10.25" in report
    assert "inline_worker_mean_ms=1.50" in report
    assert "buffer_pool_high_watermark=1" in report
    assert "queue_high_watermark=3" in report
    assert "queue_wait_mean_ms=0.50" in report
    assert "step_flush_mean_ms=0.25" in report
    assert "transfer_hidden_ratio=0.7500" in report
    assert "gpu_source_hold_mean_ms=3.50" in report
    assert "observer_callback_mean_ms=0.1250" in report


def test_reporter_writes_jsonl_and_csv(tmp_path):
    jsonl_path = tmp_path / "metrics.jsonl"
    csv_path = tmp_path / "metrics.csv"
    config = TelemetryConfig(jsonl_path=str(jsonl_path), csv_path=str(csv_path))
    reporter = TelemetryReporter(config, logging_fn=lambda message: None)
    registry = MetricsRegistry()
    registry.increment("actual_offload_bytes", 32)
    registry.set_gauge("queue_high_watermark", 2)
    registry.observe("d2h_copy_ms", 1.5)

    reporter.report(registry.snapshot(), completed_steps=3, rank=0)

    record = json.loads(jsonl_path.read_text(encoding="utf-8"))
    assert record["completed_steps"] == 3
    assert record["counters"]["actual_offload_bytes"] == 32
    assert record["histograms"]["d2h_copy_ms"]["mean"] == 1.5
    with csv_path.open(encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert any(row["metric_name"] == "actual_offload_bytes" and row["value"] == "32" for row in rows)
    assert any(row["metric_name"] == "d2h_copy_ms" and row["mean"] == "1.5" for row in rows)


def test_close_reports_partial_window_once_and_rejects_events():
    messages = []
    config = TelemetryConfig(log_interval=10, host_timing=False)
    reporter = TelemetryReporter(config, logging_fn=messages.append)
    observer = FastOffloadObserver(config, reporter=reporter)
    observer.on_step_begin(make_step_context(1))
    observer.on_step_end(make_step_context(2))

    observer.close()
    observer.close()

    assert len(messages) == 1
    assert "[FastOffload Observer Final]" in messages[0]
    with pytest.raises(RuntimeError, match="closed"):
        observer.on_backward_begin(make_step_context(3))
