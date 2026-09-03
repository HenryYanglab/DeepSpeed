# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from contextlib import nullcontext
from unittest.mock import Mock

import pytest
import torch

from deepspeed.runtime.fastoffload.actions import OffloadAction, OffloadDecision
from deepspeed.runtime.fastoffload.policies.all_offload import AllOffloadPolicy
from deepspeed.runtime.fastoffload.schedulers.overlap import OverlapScheduler
from deepspeed.runtime.fastoffload.schedulers.synchronous import SynchronousScheduler
from deepspeed.runtime.fastoffload.telemetry.metrics import MetricsRegistry
from deepspeed.runtime.fastoffload.transfer.asynchronous import AsynchronousTransferEngine
from deepspeed.runtime.fastoffload.transfer.buffer_pool import BufferState, PinnedBufferPool
from deepspeed.runtime.fastoffload.transfer.context import GradientTransferView
from deepspeed.runtime.fastoffload.transfer.gpu_buffer_pool import GpuBufferPool
from deepspeed.runtime.fastoffload.transfer.synchronous import SynchronousTransferEngine
from deepspeed.runtime.fastoffload.workers.inline import InlineWorker


def make_pool(buffer_count=1, buffer_size=32):
    return PinnedBufferPool(buffer_count,
                            buffer_size,
                            pin_memory_fn=lambda tensor: tensor,
                            unpin_memory_fn=lambda tensor: None)


def make_transfer_view(source=None, destination=None):
    source = torch.arange(8, dtype=torch.float32) if source is None else source
    destination = torch.zeros(8, dtype=torch.float32) if destination is None else destination
    return GradientTransferView(source=source,
                                destination=destination,
                                parameter_id=3,
                                group_id=0,
                                source_offset=0,
                                destination_offset=0,
                                numel=8)


def test_buffer_pool_reuses_slots_and_tracks_state():
    pool = make_pool()

    first = pool.acquire(8, torch.float32)
    tensor_pointer = first.tensor.data_ptr()
    assert first.state == BufferState.reserved
    first.mark_filled()
    first.mark_in_use()
    first.release()

    second = pool.acquire(8, torch.float32)
    assert second.tensor.data_ptr() == tensor_pointer
    assert pool.high_watermark == 1
    assert pool.allocated_bytes == 32
    assert pool.peak_allocated_bytes == 32
    second.release()
    pool.close()


def test_buffer_pool_is_bounded_and_rejects_double_release():
    pool = make_pool()
    lease = pool.acquire(8, torch.float32)
    with pytest.raises(RuntimeError, match="exhausted"):
        pool.acquire(1, torch.float32)
    lease.release()
    with pytest.raises(RuntimeError, match="already"):
        lease.release()
    pool.close()


def test_buffer_pool_grows_for_oversized_gradient():
    pool = make_pool(buffer_size=16)
    lease = pool.acquire(8, torch.float32)

    assert lease.tensor.numel() == 8
    assert pool.oversized_allocations == 1
    lease.release()
    pool.close()


def test_gpu_buffer_pool_reuses_allocation_and_tracks_peak_bytes():
    pool = GpuBufferPool(buffer_count=1, buffer_size=32)

    first = pool.acquire(8, torch.float32, torch.device("cpu"))
    pointer = first.tensor.data_ptr()
    first.release()
    second = pool.acquire(8, torch.float32, torch.device("cpu"))

    assert second.tensor.data_ptr() == pointer
    assert pool.allocated_bytes == 32
    assert pool.peak_allocated_bytes == 32
    assert pool.high_watermark == 1
    second.release()
    pool.close()


def test_sync_transfer_copies_values_and_records_actual_bytes():
    metrics = MetricsRegistry()
    pool = make_pool()
    worker = InlineWorker(metrics)
    engine = SynchronousTransferEngine(metrics, pool=pool, worker=worker)
    transfer_view = make_transfer_view()

    engine.transfer(transfer_view)

    assert torch.equal(transfer_view.destination, transfer_view.source)
    snapshot = metrics.snapshot()
    assert snapshot.counters["actual_offload_bytes"] == 32
    assert snapshot.counters["d2h_copy_count"] == 1
    assert snapshot.histograms["d2h_copy_ms"].count == 1
    assert snapshot.histograms["inline_worker_ms"].count == 1
    assert snapshot.gauges["buffer_pool_in_use"] == 0
    engine.close()


def test_sync_transfer_writes_directly_without_cpu_staging():
    metrics = MetricsRegistry()
    engine = SynchronousTransferEngine(metrics)
    transfer_view = make_transfer_view()

    engine.transfer(transfer_view)

    assert torch.equal(transfer_view.destination, transfer_view.source)
    snapshot = metrics.snapshot()
    assert snapshot.counters["actual_offload_bytes"] == 32
    assert "cpu_staging_copy_count" not in snapshot.counters
    assert snapshot.gauges["pinned_pool_peak_bytes"] == 0
    engine.close()


def test_sync_transfer_releases_buffer_when_worker_fails():
    metrics = MetricsRegistry()
    pool = make_pool()
    worker = Mock()
    worker.consume.side_effect = RuntimeError("worker failed")
    engine = SynchronousTransferEngine(metrics, pool=pool, worker=worker)

    with pytest.raises(RuntimeError, match="worker failed"):
        engine.transfer(make_transfer_view())

    assert pool.in_use == 0
    engine.close()


class FakeEvent:

    def __init__(self, enable_timing=False, initially_complete=False):
        self.complete = initially_complete

    def record(self, stream):
        pass

    def query(self):
        return self.complete

    def synchronize(self):
        self.complete = True

    def elapsed_time(self, other):
        return 2.0


class FakeStream:

    def wait_event(self, event):
        pass


class FakeAccelerator:

    def __init__(self, events_complete=False):
        self.events_complete = events_complete
        self.stream_instance = FakeStream()

    def Stream(self):
        return self.stream_instance

    def Event(self, enable_timing=False):
        return FakeEvent(enable_timing=enable_timing, initially_complete=self.events_complete)

    def current_stream(self):
        return self.stream_instance

    def stream(self, stream):
        return nullcontext()


def test_async_transfer_uses_event_before_inline_consumption():
    metrics = MetricsRegistry()
    pool = make_pool()
    gpu_pool = GpuBufferPool(buffer_count=1, buffer_size=32)
    worker = InlineWorker(metrics)
    engine = AsynchronousTransferEngine(metrics,
                                        gpu_pool=gpu_pool,
                                        pool=pool,
                                        worker=worker,
                                        accelerator=FakeAccelerator(),
                                        dedicated_copy_stream=True)
    transfer_view = make_transfer_view()

    task = engine.submit(transfer_view)

    assert task.lease.state == BufferState.copy_in_flight
    assert engine.complete(task, wait=False) is None
    assert torch.count_nonzero(transfer_view.destination) == 0
    assert engine.complete(task, wait=True) == 2.0
    assert torch.equal(transfer_view.destination, transfer_view.source)
    snapshot = metrics.snapshot()
    assert snapshot.counters["copy_submit_count"] == 1
    assert snapshot.counters["copy_complete_count"] == 1
    assert snapshot.counters["actual_offload_bytes"] == 32
    engine.close()


def test_async_transfer_writes_directly_without_inline_worker():
    metrics = MetricsRegistry()
    engine = AsynchronousTransferEngine(metrics, accelerator=FakeAccelerator())
    transfer_view = make_transfer_view()

    task = engine.submit(transfer_view)
    assert engine.complete(task, wait=True) == 2.0
    second_view = make_transfer_view()
    second_task = engine.submit(second_view)
    assert engine.complete(second_task, wait=True) == 2.0

    assert torch.equal(transfer_view.destination, transfer_view.source)
    assert torch.equal(second_view.destination, second_view.source)
    snapshot = metrics.snapshot()
    assert "cpu_staging_copy_count" not in snapshot.counters
    assert "gpu_staging_copy_ms" not in snapshot.histograms
    assert snapshot.counters["producer_stream_submit_count"] == 2
    assert snapshot.gauges["pinned_pool_peak_bytes"] == 0
    assert snapshot.gauges["gpu_staging_pool_peak_bytes"] == 0
    assert snapshot.gauges["event_pool_created"] == 1
    assert snapshot.gauges["event_pool_high_watermark"] == 1
    engine.close()


def test_overlap_scheduler_applies_bounded_queue_backpressure_and_flushes():
    metrics = MetricsRegistry()
    pool = make_pool(buffer_count=2)
    gpu_pool = GpuBufferPool(buffer_count=2, buffer_size=32)
    worker = InlineWorker(metrics)
    engine = AsynchronousTransferEngine(metrics,
                                        gpu_pool=gpu_pool,
                                        pool=pool,
                                        worker=worker,
                                        accelerator=FakeAccelerator(),
                                        dedicated_copy_stream=True)
    scheduler = OverlapScheduler(AllOffloadPolicy(), engine, metrics, max_inflight_tasks=1, max_inflight_bytes=64)

    assert scheduler.submit(make_transfer_view()) is True
    assert scheduler.submit(make_transfer_view()) is True
    scheduler.close()

    snapshot = metrics.snapshot()
    assert snapshot.counters["copy_submit_count"] == 2
    assert snapshot.counters["copy_complete_count"] == 2
    assert snapshot.histograms["queue_wait_ms"].count == 1
    assert snapshot.histograms["step_flush_ms"].count == 1
    assert snapshot.histograms["transfer_hidden_ratio"].count == 1
    assert snapshot.gauges["inflight_tasks"] == 0
    assert snapshot.gauges["inflight_bytes"] == 0


def test_all_offload_policy_and_scheduler_handle_transfer():
    policy = AllOffloadPolicy()
    transfer_engine = Mock()
    scheduler = SynchronousScheduler(policy, transfer_engine)
    transfer_view = make_transfer_view()

    assert policy.decide(transfer_view) == OffloadDecision(OffloadAction.offload_cpu)
    assert scheduler.submit(transfer_view) is True
    transfer_engine.transfer.assert_called_once_with(transfer_view)
    scheduler.close()
    transfer_engine.close.assert_called_once()
