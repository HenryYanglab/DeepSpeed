# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Versioned asynchronous coordination for hybrid CPU updates."""

import math
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Dict, Mapping, Optional

import torch

from .buffer import DoubleBufferedGradientAccumulator
from .context import HybridUpdateJob, HybridUpdateResult, clone_tensor_mapping

UpdateFunction = Callable[[HybridUpdateJob], Mapping[int, torch.Tensor]]
PrepareFunction = Callable[[Mapping[int, torch.Tensor], Mapping[int, torch.Tensor], int, str], tuple]
CommitFunction = Callable[[HybridUpdateResult], None]


class HybridUpdateCoordinator:
    """Accumulate B gradients while one frozen B/C job updates on CPU."""

    def __init__(self,
                 update_interval: int,
                 accumulation_device: str,
                 update_function: UpdateFunction,
                 second_gradient_reduction: str = "mean",
                 max_async_lag: int = 1,
                 prepare_function: Optional[PrepareFunction] = None,
                 clone_update_results: bool = True,
                 metrics=None) -> None:
        if update_interval < 2:
            raise ValueError("update_interval must be at least two")
        if second_gradient_reduction not in ("mean", "sum"):
            raise ValueError("second_gradient_reduction must be mean or sum")
        if max_async_lag < 1:
            raise ValueError("max_async_lag must be greater than zero")
        self._update_interval = update_interval
        self._reduction = second_gradient_reduction
        self._max_async_lag = max_async_lag
        self._update_function = update_function
        self._prepare_function = prepare_function
        self._clone_update_results = clone_update_results
        self._metrics = metrics
        self._accumulator = DoubleBufferedGradientAccumulator(accumulation_device)
        self._worker_affinity = ()
        self._executor = ThreadPoolExecutor(max_workers=1,
                                            thread_name_prefix="fastoffload-hybrid",
                                            initializer=self._initialize_worker)
        self._futures: Dict[int, Future] = {}
        self._ready: Dict[int, HybridUpdateResult] = {}
        self._transfer_sources: Dict[int, tuple] = {}
        self._submitted_version = 0
        self._committed_version = 0
        self._closed = False
        self._lock = threading.Lock()

    def set_worker_affinity(self, cores) -> None:
        if self._futures:
            raise RuntimeError("Cannot change Hybrid worker affinity after update submission")
        self._worker_affinity = tuple(cores)

    def _initialize_worker(self) -> None:
        if self._worker_affinity:
            os.sched_setaffinity(0, self._worker_affinity)

    @property
    def next_submitted_version(self) -> int:
        with self._lock:
            return self._submitted_version + 1

    @property
    def committed_version(self) -> int:
        with self._lock:
            return self._committed_version

    def restore_committed_version(self, version: int) -> None:
        if version < 0:
            raise ValueError("Committed version must not be negative")
        with self._lock:
            if self._futures:
                raise RuntimeError("Cannot restore a version with pending hybrid updates")
            self._submitted_version = version
            self._committed_version = version

    @property
    def next_ready_version(self) -> Optional[int]:
        with self._lock:
            if not self._ready:
                return None
            return min(result.version for result in self._ready.values())

    @property
    def pending_updates(self) -> int:
        with self._lock:
            return len(self._futures) + len(self._ready)

    @property
    def at_capacity(self) -> bool:
        return self.pending_updates >= self._max_async_lag

    @property
    def active_accumulated_steps(self) -> int:
        return self._accumulator.accumulated_steps(self._accumulator.active_id)

    @property
    def accumulator(self) -> DoubleBufferedGradientAccumulator:
        return self._accumulator

    def discard_active_gradients(self) -> None:
        self._accumulator.discard_active()

    def accumulate_second(self, gradients: Mapping[int, torch.Tensor]) -> None:
        self._ensure_open()
        self._accumulator.accumulate(gradients)

    def submit_boundary(self,
                        second_gradients: Mapping[int, torch.Tensor],
                        dense_gradients: Mapping[int, torch.Tensor],
                        lr: Optional[float] = None) -> int:
        self._ensure_open()
        if lr is not None:
            lr = float(lr)
            if not math.isfinite(lr) or lr < 0.0:
                raise ValueError("Hybrid job learning rate must be finite and non-negative")
        self.progress()
        if self.pending_updates >= self._max_async_lag:
            raise RuntimeError("Maximum asynchronous hybrid update lag reached")
        self._accumulator.accumulate(second_gradients)
        active_id = self._accumulator.active_id
        interval_steps = self._accumulator.accumulated_steps(active_id)
        if interval_steps != self._update_interval:
            raise RuntimeError(f"Hybrid boundary expected {self._update_interval} accumulated steps, "
                               f"found {interval_steps}")

        with self._lock:
            self._submitted_version += 1
            version = self._submitted_version
        buffer_id = self._accumulator.freeze_and_swap(version)
        if self._accumulator.accumulation_device.type == "gpu":
            self._accumulator.begin_copy_to_cpu(buffer_id)
        frozen = self._accumulator.frozen_gradients(buffer_id)
        transfer_event = None
        transfer_sources = ()
        if self._prepare_function is None:
            cpu_second = clone_tensor_mapping(frozen, torch.device("cpu"))
            if self._reduction == "mean":
                for gradient in cpu_second.values():
                    gradient.div_(interval_steps)
            cpu_dense = clone_tensor_mapping(dense_gradients, torch.device("cpu"))
        else:
            prepared = self._prepare_function(frozen, dense_gradients, interval_steps, self._reduction)
            if len(prepared) == 3:
                cpu_second, cpu_dense, transfer_event = prepared
            else:
                cpu_second, cpu_dense, transfer_event, transfer_sources = prepared
        self._accumulator.begin_update(buffer_id)
        job = HybridUpdateJob(version=version,
                              buffer_id=buffer_id,
                              interval_steps=interval_steps,
                              second_gradients=cpu_second,
                              dense_gradients=cpu_dense,
                              transfer_event=transfer_event,
                              lr=lr)
        with self._lock:
            if transfer_sources:
                self._transfer_sources[version] = transfer_sources
        future = self._executor.submit(self._run_update, job)
        with self._lock:
            self._futures[version] = future
        return version

    def progress(self) -> None:
        self._ensure_open()
        with self._lock:
            completed = [(version, future) for version, future in self._futures.items() if future.done()]
        for version, future in completed:
            result = future.result()
            with self._lock:
                self._futures.pop(version)
                self._ready[version] = result

    def commit_ready(self, commit_function: CommitFunction, max_versions: Optional[int] = None) -> int:
        self._ensure_open()
        self.progress()
        committed = 0
        with self._lock:
            ready_items = sorted(self._ready.items())
        for version, result in ready_items:
            if max_versions is not None and committed >= max_versions:
                break
            expected_version = self.committed_version + 1
            if result.version != expected_version:
                raise RuntimeError(
                    f"Hybrid commit version mismatch: expected {expected_version}, got {result.version}")
            commit_function(result)
            with self._lock:
                self._ready.pop(version)
                self._committed_version = result.version
            committed += 1
        return committed

    def wait_and_commit_oldest(self, commit_function: CommitFunction) -> None:
        self._ensure_open()
        with self._lock:
            futures = list(self._futures.values())
        if futures:
            futures[0].result()
        self.commit_ready(commit_function)

    def close(self, commit_function: Optional[CommitFunction] = None, discard_pending: bool = False) -> None:
        if self._closed:
            return
        if discard_pending:
            self._executor.shutdown(wait=True)
            with self._lock:
                self._futures.clear()
                self._ready.clear()
                self._transfer_sources.clear()
            self._closed = True
            return
        if commit_function is not None:
            while self.pending_updates:
                self.progress()
                if self.next_ready_version is not None:
                    self.commit_ready(commit_function)
                else:
                    time.sleep(0.01)
        elif self.pending_updates:
            raise RuntimeError("Cannot close hybrid coordinator with uncommitted updates")
        self._executor.shutdown(wait=True)
        self._closed = True

    def _run_update(self, job: HybridUpdateJob) -> HybridUpdateResult:
        if job.transfer_event is not None:
            transfer_start = time.perf_counter_ns()
            job.transfer_event.synchronize()
            if self._metrics is not None:
                self._metrics.observe("takeover_cpu_d2h_wait_host_ms", (time.perf_counter_ns() - transfer_start) / 1e6)
            with self._lock:
                self._transfer_sources.pop(job.version, None)
            job = HybridUpdateJob(version=job.version,
                                  buffer_id=job.buffer_id,
                                  interval_steps=job.interval_steps,
                                  second_gradients=job.second_gradients,
                                  dense_gradients=job.dense_gradients,
                                  lr=job.lr)
        self._accumulator.release_update_buffer(job.buffer_id)
        update_start = time.perf_counter_ns()
        updated_values = self._update_function(job)
        if self._metrics is not None:
            self._metrics.observe("takeover_cpu_adam_host_ms", (time.perf_counter_ns() - update_start) / 1e6)
        cpu_values = (clone_tensor_mapping(updated_values, torch.device("cpu"))
                      if self._clone_update_results else updated_values)
        return HybridUpdateResult(version=job.version, buffer_id=job.buffer_id, updated_values=cpu_values)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Hybrid update coordinator is closed")
