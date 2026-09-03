# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Metadata-only observer for FastOffload lifecycle events."""

import threading
import time
from collections import deque
from typing import Deque, Optional, Tuple, Union

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.fastoffload.config import TelemetryConfig
from deepspeed.runtime.fastoffload.context import GradientBucketContext, GradientContext, StepContext
from deepspeed.runtime.fastoffload.events import FastOffloadEvent

from .metrics import MetricsRegistry, MetricsSnapshot
from .reporter import TelemetryReporter

ObserverContext = Union[StepContext, GradientContext, GradientBucketContext]
DebugEvent = Tuple[FastOffloadEvent, ObserverContext]


class FastOffloadObserver:
    """Aggregate observer metadata without reading or retaining tensors."""

    def __init__(self,
                 config: TelemetryConfig,
                 metrics: Optional[MetricsRegistry] = None,
                 reporter: Optional[TelemetryReporter] = None) -> None:
        self._config = config
        self._metrics = metrics or MetricsRegistry()
        self._reporter = reporter or TelemetryReporter(config)
        self._backward_start_ns: Optional[int] = None
        self._step_start_ns: Optional[int] = None
        self._completed_steps = 0
        self._steps_since_report = 0
        self._last_rank = 0
        self._closed = False
        self._lock = threading.Lock()
        self._debug_events: Optional[Deque[DebugEvent]] = None
        if config.debug_event_buffer_size > 0:
            self._debug_events = deque(maxlen=config.debug_event_buffer_size)

    @property
    def metrics(self) -> MetricsRegistry:
        return self._metrics

    @property
    def debug_events(self) -> Tuple[DebugEvent, ...]:
        with self._lock:
            return tuple(self._debug_events or ())

    def on_backward_begin(self, context: StepContext) -> None:
        callback_start_ns = self._callback_begin(context, FastOffloadEvent.backward_begin)
        try:
            with self._lock:
                if self._backward_start_ns is not None:
                    raise RuntimeError("A backward pass is already active")
                self._backward_start_ns = context.timestamp_ns
            self._metrics.increment("backward_count")
            self._set_step_gauges(context)
        finally:
            self._callback_end(callback_start_ns)

    def on_gradient_ready(self, context: GradientContext) -> None:
        callback_start_ns = self._callback_begin(context, FastOffloadEvent.gradient_ready)
        try:
            if not self._config.parameter_events:
                return
            self._metrics.increment("gradient_ready_count")
            self._metrics.increment("gradient_parameter_numel", context.parameter_numel)
        finally:
            self._callback_end(callback_start_ns)

    def on_gradient_reduced(self, context: GradientContext) -> None:
        callback_start_ns = self._callback_begin(context, FastOffloadEvent.gradient_reduced)
        try:
            if not self._config.parameter_events:
                return
            self._metrics.increment("gradient_reduced_count")
            if context.is_local_partition:
                self._metrics.increment("gradient_local_shard_numel", context.shard_numel)
                self._metrics.increment("gradient_local_shard_bytes", context.shard_bytes)
                if context.is_cpu_offload:
                    self._metrics.increment("potential_offload_bytes", context.shard_bytes)
        finally:
            self._callback_end(callback_start_ns)

    def on_gradient_bucket(self, context: GradientBucketContext) -> None:
        self._ensure_open()
        callback_start_ns = time.perf_counter_ns() if self._config.host_timing else None
        try:
            if not self._config.bucket_events:
                return
            self._metrics.increment("gradient_bucket_count")
            self._metrics.increment("gradient_bucket_bytes", context.total_bytes)
            self._metrics.observe("gradient_bucket_parameter_count", context.parameter_count)
        finally:
            self._callback_end(callback_start_ns)

    def on_backward_end(self, context: StepContext) -> None:
        callback_start_ns = self._callback_begin(context, FastOffloadEvent.backward_end)
        try:
            with self._lock:
                start_ns = self._backward_start_ns
                self._backward_start_ns = None
            if start_ns is None:
                raise RuntimeError("Backward end received without backward begin")
            self._observe_phase_duration("backward_host_ms", start_ns, context.timestamp_ns)
            self._set_step_gauges(context)
        finally:
            self._callback_end(callback_start_ns)

    def on_step_begin(self, context: StepContext) -> None:
        callback_start_ns = self._callback_begin(context, FastOffloadEvent.step_begin)
        try:
            with self._lock:
                if self._step_start_ns is not None:
                    raise RuntimeError("An optimizer step is already active")
                self._step_start_ns = context.timestamp_ns
            self._set_step_gauges(context)
        finally:
            self._callback_end(callback_start_ns)

    def on_step_end(self, context: StepContext) -> None:
        callback_start_ns = self._callback_begin(context, FastOffloadEvent.step_end)
        should_report = False
        try:
            with self._lock:
                start_ns = self._step_start_ns
                self._step_start_ns = None
            if start_ns is None:
                raise RuntimeError("Step end received without step begin")
            self._observe_phase_duration("optimizer_step_host_ms", start_ns, context.timestamp_ns)
            self._metrics.increment("step_count")
            self._set_step_gauges(context)
            self._sample_memory_metrics()

            self._completed_steps += 1
            self._steps_since_report += 1
            should_report = self._reporter.should_report(self._completed_steps)
        finally:
            self._callback_end(callback_start_ns)

        if should_report:
            snapshot = self._metrics.snapshot(reset=True)
            self._reporter.report(snapshot, self._completed_steps, context.rank)
            self._steps_since_report = 0

    def snapshot(self, reset: bool = False) -> MetricsSnapshot:
        return self._metrics.snapshot(reset=reset)

    def close(self) -> None:
        """Emit a final partial window and reject subsequent events."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if self._steps_since_report > 0:
            snapshot = self._metrics.snapshot(reset=True)
            self._reporter.report(snapshot, self._completed_steps, self._last_rank, final=True)
            self._steps_since_report = 0

    def _callback_begin(self, context: ObserverContext, event: FastOffloadEvent) -> Optional[int]:
        self._ensure_open()
        self._last_rank = context.rank
        self._record_debug_event(event, context)
        return time.perf_counter_ns() if self._config.host_timing else None

    def _callback_end(self, start_ns: Optional[int]) -> None:
        if start_ns is None:
            return
        elapsed_ns = time.perf_counter_ns() - start_ns
        self._metrics.observe("observer_callback_ms", elapsed_ns / 1_000_000.0)

    def _ensure_open(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("FastOffloadObserver is closed")

    def _record_debug_event(self, event: FastOffloadEvent, context: ObserverContext) -> None:
        if self._debug_events is None:
            return
        with self._lock:
            self._debug_events.append((event, context))

    def _observe_phase_duration(self, metric_name: str, start_ns: int, end_ns: int) -> None:
        if end_ns < start_ns:
            raise ValueError(f"{metric_name} end timestamp precedes its start timestamp")
        if self._config.host_timing:
            self._metrics.observe(metric_name, (end_ns - start_ns) / 1_000_000.0)

    def _sample_memory_metrics(self) -> None:
        if not self._config.memory_metrics:
            return
        accelerator = get_accelerator()
        self._metrics.set_gauge("accelerator_allocated_bytes", accelerator.memory_allocated())
        self._metrics.set_gauge("accelerator_peak_allocated_bytes", accelerator.max_memory_allocated())
        self._metrics.set_gauge("accelerator_reserved_bytes", accelerator.memory_reserved())
        self._metrics.set_gauge("accelerator_peak_reserved_bytes", accelerator.max_memory_reserved())

    def _set_step_gauges(self, context: StepContext) -> None:
        self._metrics.set_gauge("global_step", context.global_step)
        self._metrics.set_gauge("micro_step", context.micro_step)
        self._metrics.set_gauge("gradient_accumulation_boundary", int(context.gradient_accumulation_boundary))
