# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Event-driven asynchronous device-to-host transfer engine."""

import time
from dataclasses import replace
from typing import Any, Optional

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.fastoffload.telemetry.metrics import MetricsRegistry
from deepspeed.runtime.fastoffload.workers.inline import InlineWorker

from .buffer_pool import PinnedBufferPool
from .context import GradientTransferView
from .event_pool import TransferEventPool
from .gpu_buffer_pool import GpuBufferPool
from .task import AsyncTransferTask


class AsynchronousTransferEngine:
    """Submit D2H copies on a dedicated stream and complete them by event."""

    def __init__(self,
                 metrics: MetricsRegistry,
                 gpu_pool: Optional[GpuBufferPool] = None,
                 pool: Optional[PinnedBufferPool] = None,
                 worker: Optional[InlineWorker] = None,
                 accelerator: Optional[Any] = None,
                 event_pool_capacity: int = 1,
                 dedicated_copy_stream: bool = False) -> None:
        if (pool is None) != (worker is None):
            raise ValueError("CPU staging requires both a pinned pool and an inline worker")
        if dedicated_copy_stream != (gpu_pool is not None):
            raise ValueError("A dedicated copy stream requires a GPU staging pool")
        self._pool = pool
        self._gpu_pool = gpu_pool
        self._worker = worker
        self._metrics = metrics
        self._accelerator = accelerator or get_accelerator()
        self._dedicated_copy_stream = dedicated_copy_stream
        self._copy_stream = self._accelerator.Stream() if dedicated_copy_stream else None
        self._event_pool = TransferEventPool(self._accelerator, event_pool_capacity)
        self._next_task_id = 0

    @property
    def requires_stable_source(self) -> bool:
        return self._dedicated_copy_stream

    @property
    def overlaps_backward(self) -> bool:
        return self._dedicated_copy_stream

    def submit(self, transfer_view: GradientTransferView) -> AsyncTransferTask:
        submit_start_ns = time.perf_counter_ns()
        lease = None
        gpu_lease = None
        event_bundle = None
        try:
            if self._pool is not None:
                lease = self._pool.acquire(transfer_view.numel, transfer_view.destination.dtype)
            if self._gpu_pool is not None:
                gpu_lease = self._gpu_pool.acquire(transfer_view.numel, transfer_view.destination.dtype,
                                                   transfer_view.source.device)

            producer_stream = self._accelerator.current_stream()
            event_bundle = self._event_pool.acquire()
            staging_start_event = event_bundle.staging_start
            source_ready_event = event_bundle.source_ready
            copy_start_event = event_bundle.copy_start
            copy_complete_event = event_bundle.copy_complete
            target = transfer_view.destination if lease is None else lease.tensor
            if self._dedicated_copy_stream:
                staging_start_event.record(producer_stream)
                gpu_lease.tensor.copy_(transfer_view.source, non_blocking=True)
                source_ready_event.record(producer_stream)
                stable_view = replace(transfer_view, source=gpu_lease.tensor)
                with self._accelerator.stream(self._copy_stream):
                    self._copy_stream.wait_event(source_ready_event)
                    copy_start_event.record(self._copy_stream)
                    target.copy_(stable_view.source, non_blocking=True)
                    copy_complete_event.record(self._copy_stream)
            else:
                source_ready_event.record(producer_stream)
                stable_view = transfer_view
                copy_start_event.record(producer_stream)
                target.copy_(stable_view.source, non_blocking=True)
                copy_complete_event.record(producer_stream)
            if lease is not None:
                lease.mark_copy_in_flight()

            task = AsyncTransferTask(task_id=self._next_task_id,
                                     transfer_view=stable_view,
                                     lease=lease,
                                     gpu_lease=gpu_lease,
                                     event_bundle=event_bundle,
                                     staging_start_event=staging_start_event,
                                     source_ready_event=source_ready_event,
                                     copy_start_event=copy_start_event,
                                     copy_complete_event=copy_complete_event,
                                     submitted_ns=submit_start_ns)
            self._next_task_id += 1
            self._metrics.increment("copy_submit_count")
            strategy_metric = "dedicated_stream_submit_count" if self._dedicated_copy_stream else "producer_stream_submit_count"
            self._metrics.increment(strategy_metric)
            submit_ms = (time.perf_counter_ns() - submit_start_ns) / 1_000_000.0
            self._metrics.observe("copy_submit_host_ms", submit_ms)
            return task
        except Exception:
            if event_bundle is not None:
                self._event_pool.release(event_bundle)
            if gpu_lease is not None:
                gpu_lease.release()
            if lease is not None:
                lease.release()
            raise

    def is_complete(self, task: AsyncTransferTask) -> bool:
        return task.copy_complete_event.query()

    def complete(self, task: AsyncTransferTask, wait: bool = False) -> Optional[float]:
        if wait:
            task.copy_complete_event.synchronize()
        elif not self.is_complete(task):
            return None

        try:
            copy_ms = task.copy_start_event.elapsed_time(task.copy_complete_event)
            self._metrics.increment("copy_complete_count")
            self._metrics.increment("actual_offload_bytes", task.nbytes)
            if task.gpu_lease is not None:
                staging_ms = task.staging_start_event.elapsed_time(task.source_ready_event)
                self._metrics.observe("gpu_staging_copy_ms", staging_ms)
            self._metrics.observe("d2h_copy_ms", copy_ms)
            if copy_ms > 0:
                bandwidth_gbps = task.nbytes / (copy_ms * 1_000_000.0)
                self._metrics.observe("d2h_bandwidth_gbps", bandwidth_gbps)
            source_hold_ms = (time.perf_counter_ns() - task.submitted_ns) / 1_000_000.0
            self._metrics.observe("gpu_source_hold_ms", source_hold_ms)
            if task.lease is not None:
                task.lease.mark_filled()
                self._worker.consume(task.lease, task.transfer_view)
                self._metrics.increment("cpu_staging_copy_count")
            return copy_ms
        finally:
            self._event_pool.release(task.event_bundle)
            if task.gpu_lease is not None:
                task.gpu_lease.release()
            if task.lease is not None:
                task.lease.release()
            self._update_pool_metrics()

    def close(self) -> None:
        try:
            self._event_pool.close()
        finally:
            try:
                if self._gpu_pool is not None:
                    self._gpu_pool.close()
            finally:
                if self._pool is not None:
                    self._pool.close()

    def _update_pool_metrics(self) -> None:
        if self._pool is None:
            self._metrics.set_gauge("buffer_pool_in_use", 0)
            self._metrics.set_gauge("buffer_pool_high_watermark", 0)
            self._metrics.set_gauge("pinned_pool_allocated_bytes", 0)
            self._metrics.set_gauge("pinned_pool_peak_bytes", 0)
        else:
            self._metrics.set_gauge("buffer_pool_in_use", self._pool.in_use)
            self._metrics.set_gauge("buffer_pool_high_watermark", self._pool.high_watermark)
            self._metrics.set_gauge("oversized_buffer_allocations", self._pool.oversized_allocations)
            self._metrics.set_gauge("pinned_pool_allocated_bytes", self._pool.allocated_bytes)
            self._metrics.set_gauge("pinned_pool_peak_bytes", self._pool.peak_allocated_bytes)
        if self._gpu_pool is None:
            self._metrics.set_gauge("gpu_staging_pool_in_use", 0)
            self._metrics.set_gauge("gpu_staging_pool_high_watermark", 0)
            self._metrics.set_gauge("gpu_staging_pool_allocated_bytes", 0)
            self._metrics.set_gauge("gpu_staging_pool_peak_bytes", 0)
        else:
            self._metrics.set_gauge("gpu_staging_pool_in_use", self._gpu_pool.in_use)
            self._metrics.set_gauge("gpu_staging_pool_high_watermark", self._gpu_pool.high_watermark)
            self._metrics.set_gauge("gpu_staging_pool_allocated_bytes", self._gpu_pool.allocated_bytes)
            self._metrics.set_gauge("gpu_staging_pool_peak_bytes", self._gpu_pool.peak_allocated_bytes)
        self._metrics.set_gauge("event_pool_created", self._event_pool.created)
        self._metrics.set_gauge("event_pool_high_watermark", self._event_pool.high_watermark)
