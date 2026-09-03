# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Bounded event-driven scheduler for backward/D2H overlap."""

import time
from collections import deque
from typing import Deque

from deepspeed.runtime.fastoffload.actions import OffloadAction
from deepspeed.runtime.fastoffload.policies.all_offload import AllOffloadPolicy
from deepspeed.runtime.fastoffload.telemetry.metrics import MetricsRegistry
from deepspeed.runtime.fastoffload.transfer.asynchronous import AsynchronousTransferEngine
from deepspeed.runtime.fastoffload.transfer.context import GradientTransferView
from deepspeed.runtime.fastoffload.transfer.task import AsyncTransferTask


class OverlapScheduler:
    """Keep a bounded FIFO of D2H copies and advance them without global sync."""

    def __init__(self, policy: AllOffloadPolicy, transfer_engine: AsynchronousTransferEngine, metrics: MetricsRegistry,
                 max_inflight_tasks: int, max_inflight_bytes: int) -> None:
        self._policy = policy
        self._transfer_engine = transfer_engine
        self._metrics = metrics
        self._max_inflight_tasks = max_inflight_tasks
        self._max_inflight_bytes = max_inflight_bytes
        self._tasks: Deque[AsyncTransferTask] = deque()
        self._inflight_bytes = 0
        self._queue_high_watermark = 0
        self._copy_ms_since_step = 0.0
        self._closed = False

    @property
    def requires_stable_source(self) -> bool:
        return self._transfer_engine.requires_stable_source

    def submit(self, transfer_view: GradientTransferView) -> bool:
        self._ensure_open()
        decision = self._policy.decide(transfer_view)
        if decision.action == OffloadAction.keep_native:
            return False
        if decision.action != OffloadAction.offload_cpu:
            raise RuntimeError(f"Unsupported overlap offload action: {decision.action}")

        self.progress()
        while self._must_wait_for_capacity(transfer_view.nbytes):
            wait_start_ns = time.perf_counter_ns()
            self._complete_oldest(wait=True, overlapped=False)
            wait_ms = (time.perf_counter_ns() - wait_start_ns) / 1_000_000.0
            self._metrics.observe("queue_wait_ms", wait_ms)

        task = self._transfer_engine.submit(transfer_view)
        self._tasks.append(task)
        self._inflight_bytes += task.nbytes
        self._queue_high_watermark = max(self._queue_high_watermark, len(self._tasks))
        self._update_queue_metrics()
        return True

    def progress(self) -> None:
        self._ensure_open()
        while self._tasks and self._transfer_engine.is_complete(self._tasks[0]):
            self._complete_oldest(wait=False, overlapped=True)

    def prepare_step(self) -> None:
        self._ensure_open()
        flush_start_ns = time.perf_counter_ns()
        while self._tasks:
            self._complete_oldest(wait=True, overlapped=False)
        flush_ms = (time.perf_counter_ns() - flush_start_ns) / 1_000_000.0
        self._metrics.observe("step_flush_ms", flush_ms)
        if self._transfer_engine.overlaps_backward and self._copy_ms_since_step > 0:
            hidden_ratio = 1.0 - min(flush_ms / self._copy_ms_since_step, 1.0)
            self._metrics.observe("transfer_hidden_ratio", hidden_ratio)
        self._copy_ms_since_step = 0.0

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.prepare_step()
        finally:
            self._closed = True
            self._transfer_engine.close()

    def _must_wait_for_capacity(self, next_bytes: int) -> bool:
        if not self._tasks:
            return False
        task_limit_reached = len(self._tasks) >= self._max_inflight_tasks
        byte_limit_reached = self._inflight_bytes + next_bytes > self._max_inflight_bytes
        return task_limit_reached or byte_limit_reached

    def _complete_oldest(self, wait: bool, overlapped: bool) -> None:
        task = self._tasks.popleft()
        self._inflight_bytes -= task.nbytes
        try:
            copy_ms = self._transfer_engine.complete(task, wait=wait)
            if copy_ms is None:
                raise RuntimeError("Transfer task was not complete")
            self._copy_ms_since_step += copy_ms
            if overlapped and self._transfer_engine.overlaps_backward:
                self._metrics.observe("backward_overlap_ms", copy_ms)
        finally:
            self._update_queue_metrics()

    def _update_queue_metrics(self) -> None:
        self._metrics.set_gauge("inflight_tasks", len(self._tasks))
        self._metrics.set_gauge("inflight_bytes", self._inflight_bytes)
        self._metrics.set_gauge("queue_high_watermark", self._queue_high_watermark)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Overlap scheduler is closed")
