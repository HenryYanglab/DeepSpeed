# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Synchronous device-to-host transfer implementation."""

import time
from typing import Optional

from deepspeed.runtime.fastoffload.telemetry.metrics import MetricsRegistry
from deepspeed.runtime.fastoffload.workers.inline import InlineWorker

from .buffer_pool import PinnedBufferPool
from .context import GradientTransferView


class SynchronousTransferEngine:
    """Synchronously copy a shard directly or through optional CPU staging."""

    def __init__(self,
                 metrics: MetricsRegistry,
                 pool: Optional[PinnedBufferPool] = None,
                 worker: Optional[InlineWorker] = None) -> None:
        if (pool is None) != (worker is None):
            raise ValueError("CPU staging requires both a pinned pool and an inline worker")
        self._pool = pool
        self._worker = worker
        self._metrics = metrics

    def transfer(self, transfer_view: GradientTransferView) -> None:
        lease = None
        try:
            target = transfer_view.destination
            if self._pool is not None:
                lease = self._pool.acquire(transfer_view.numel, transfer_view.destination.dtype)
                target = lease.tensor

            start_ns = time.perf_counter_ns()
            target.copy_(transfer_view.source, non_blocking=False)
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
            self._metrics.increment("actual_offload_bytes", transfer_view.nbytes)
            self._metrics.increment("d2h_copy_count")
            self._metrics.observe("d2h_copy_ms", elapsed_ms)
            if elapsed_ms > 0:
                bandwidth_gbps = transfer_view.nbytes / (elapsed_ms * 1_000_000.0)
                self._metrics.observe("d2h_bandwidth_gbps", bandwidth_gbps)

            if lease is not None:
                lease.mark_filled()
                self._worker.consume(lease, transfer_view)
                self._metrics.increment("cpu_staging_copy_count")
        finally:
            if lease is not None:
                lease.release()
            self._update_pool_metrics()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()

    def _update_pool_metrics(self) -> None:
        if self._pool is None:
            self._metrics.set_gauge("buffer_pool_in_use", 0)
            self._metrics.set_gauge("buffer_pool_high_watermark", 0)
            self._metrics.set_gauge("pinned_pool_allocated_bytes", 0)
            self._metrics.set_gauge("pinned_pool_peak_bytes", 0)
            return
        self._metrics.set_gauge("buffer_pool_in_use", self._pool.in_use)
        self._metrics.set_gauge("buffer_pool_high_watermark", self._pool.high_watermark)
        self._metrics.set_gauge("oversized_buffer_allocations", self._pool.oversized_allocations)
        self._metrics.set_gauge("pinned_pool_allocated_bytes", self._pool.allocated_bytes)
        self._metrics.set_gauge("pinned_pool_peak_bytes", self._pool.peak_allocated_bytes)
