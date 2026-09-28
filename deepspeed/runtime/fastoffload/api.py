# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Process-local registration API for FastOffload observers."""

import threading
import weakref
from pathlib import Path
from typing import Any, Dict, Optional, Union

from deepspeed.runtime.fastoffload.adapters.zero2 import Zero2ObserverAdapter
from deepspeed.runtime.fastoffload.config import FastOffloadConfig, FastOffloadMode
from deepspeed.runtime.fastoffload.hybrid.shadow import HybridCompressedCollectiveShadow
from deepspeed.runtime.fastoffload.hybrid.takeover_runtime import Zero2TakeoverRuntime
from deepspeed.runtime.fastoffload.importance import (ImportanceRegistry, StreamingImportanceSelector,
                                                      create_importance_algorithm)
from deepspeed.runtime.fastoffload.controller import FastOffloadController, NullFastOffloadController
from deepspeed.runtime.fastoffload.policies.all_offload import AllOffloadPolicy
from deepspeed.runtime.fastoffload.schedulers.overlap import OverlapScheduler
from deepspeed.runtime.fastoffload.schedulers.synchronous import SynchronousScheduler
from deepspeed.runtime.fastoffload.telemetry.observer import FastOffloadObserver
from deepspeed.runtime.fastoffload.transfer.asynchronous import AsynchronousTransferEngine
from deepspeed.runtime.fastoffload.transfer.buffer_pool import PinnedBufferPool
from deepspeed.runtime.fastoffload.transfer.gpu_buffer_pool import GpuBufferPool
from deepspeed.runtime.fastoffload.transfer.synchronous import SynchronousTransferEngine
from deepspeed.runtime.fastoffload.workers.inline import InlineWorker

ConfigInput = Union[FastOffloadConfig, Dict[str, Any], str, Path]
Controller = Union[FastOffloadController, NullFastOffloadController]

_ACTIVE_HANDLE: Optional["FastOffloadHandle"] = None
_REGISTRY_LOCK = threading.Lock()


class FastOffloadHandle:
    """Own controllers created from one process-local FastOffload installation."""

    def __init__(self, config: FastOffloadConfig) -> None:
        self.config = config
        self._controllers: "weakref.WeakSet[FastOffloadController]" = weakref.WeakSet()
        self._closed = False
        self._lock = threading.Lock()

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def close(self) -> None:
        """Close attached observers and remove this installation if active."""
        global _ACTIVE_HANDLE
        with self._lock:
            if self._closed:
                return
            self._closed = True
            controllers = list(self._controllers)

        for controller in controllers:
            controller.close()

        with _REGISTRY_LOCK:
            if _ACTIVE_HANDLE is self:
                _ACTIVE_HANDLE = None

    def _attach(self, controller: FastOffloadController) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot attach a controller to a closed FastOffload handle")
            self._controllers.add(controller)

    def __enter__(self) -> "FastOffloadHandle":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False


def install(config: ConfigInput) -> FastOffloadHandle:
    """Install one process-local observer configuration before DeepSpeed initialization."""
    global _ACTIVE_HANDLE
    normalized_config = _normalize_config(config)
    handle = FastOffloadHandle(normalized_config)
    with _REGISTRY_LOCK:
        if _ACTIVE_HANDLE is not None and not _ACTIVE_HANDLE.closed:
            raise RuntimeError("FastOffload is already installed in this process")
        _ACTIVE_HANDLE = handle
    return handle


def create_zero2_controller(optimizer: Any) -> Controller:
    """Create the controller attached by a ZeRO-2 optimizer constructor."""
    with _REGISTRY_LOCK:
        handle = _ACTIVE_HANDLE
    if handle is None or handle.closed or not handle.config.enabled:
        return NullFastOffloadController()
    hybrid_config = handle.config.hybrid_update
    if hybrid_config.enabled and not (hybrid_config.compressed_collective_shadow or hybrid_config.zero2_takeover):
        raise NotImplementedError("Hybrid update requires compressed_collective_shadow=true or "
                                  "zero2_takeover=true for optimizer step takeover")
    if hybrid_config.enabled and hybrid_config.compressed_collective_shadow and optimizer.gradient_accumulation_steps != 1:
        raise ValueError("Hybrid compressed collective shadow currently requires gradient_accumulation_steps=1")
    adapter = Zero2ObserverAdapter(optimizer,
                                   max_parameter_name_length=handle.config.telemetry.max_parameter_name_length)
    observer = FastOffloadObserver(handle.config.telemetry)
    importance_selector = None
    if handle.config.importance.enabled:
        importance_algorithm = create_importance_algorithm(handle.config.importance.algorithm,
                                                           handle.config.importance.topk_ratio,
                                                           handle.config.importance.comparison_chunk_rows)
        importance_registry = ImportanceRegistry()
        rank = adapter.get_rank()
        importance_selector = StreamingImportanceSelector(adapter.iter_parameter_importance_views(),
                                                          importance_algorithm,
                                                          importance_registry,
                                                          observer.metrics,
                                                          warmup_steps=handle.config.importance.warmup_steps,
                                                          rank=rank,
                                                          output_path=handle.config.importance.output_path,
                                                          sparse_backward=handle.config.importance.sparse_backward)
    hybrid_shadow = None
    if hybrid_config.enabled and hybrid_config.compressed_collective_shadow:
        hybrid_shadow = HybridCompressedCollectiveShadow(adapter, importance_registry, observer.metrics,
                                                         hybrid_config.update_interval,
                                                         hybrid_config.compressed_bucket_bytes,
                                                         hybrid_config.parity_atol, hybrid_config.parity_rtol)
    takeover_runtime = None
    if hybrid_config.enabled and hybrid_config.zero2_takeover:
        optimizer_groups = optimizer.optimizer.param_groups
        hyperparameters = {(group["lr"], tuple(group["betas"]), group["eps"], group.get("weight_decay", 0.0))
                           for group in optimizer_groups}
        if len(hyperparameters) != 1:
            raise ValueError("ZeRO-2 takeover currently requires identical Adam hyperparameters across groups")
        lr, betas, eps, weight_decay = hyperparameters.pop()
        takeover_runtime = Zero2TakeoverRuntime(
            adapter, importance_registry, hybrid_config.update_interval, optimizer.gradient_accumulation_steps,
            hybrid_config.accumulation_device, hybrid_config.second_gradient_reduction, lr, betas, eps, weight_decay,
            hybrid_config.max_async_lag, hybrid_config.compressed_bucket_bytes, observer.metrics,
            hybrid_config.pt_reserved_cores_perc, hybrid_config.second_reduce_scatter_device)
    scheduler = None
    if handle.config.mode in (FastOffloadMode.sync_offload, FastOffloadMode.async_offload):
        if not handle.config.transfer.cpu_staging and not optimizer.cpu_offload_pin_memory:
            raise ValueError("Direct FastOffload D2H requires ZeRO offload_optimizer.pin_memory=true")
        pool = None
        worker = None
        if handle.config.transfer.cpu_staging:
            pool = PinnedBufferPool(buffer_count=handle.config.transfer.buffer_count,
                                    buffer_size=handle.config.transfer.buffer_size)
            worker = InlineWorker(observer.metrics)
        policy = AllOffloadPolicy()
        if handle.config.mode == FastOffloadMode.sync_offload:
            transfer_engine = SynchronousTransferEngine(observer.metrics, pool=pool, worker=worker)
            scheduler = SynchronousScheduler(policy, transfer_engine)
        else:
            dedicated_copy_stream = handle.config.transfer.async_strategy == "dedicated_stream"
            gpu_pool = None
            if dedicated_copy_stream:
                gpu_pool = GpuBufferPool(buffer_count=handle.config.transfer.buffer_count,
                                         buffer_size=handle.config.transfer.buffer_size)
            transfer_engine = AsynchronousTransferEngine(
                observer.metrics,
                gpu_pool=gpu_pool,
                pool=pool,
                worker=worker,
                event_pool_capacity=handle.config.scheduler.max_inflight_tasks,
                dedicated_copy_stream=dedicated_copy_stream)
            scheduler = OverlapScheduler(policy,
                                         transfer_engine,
                                         observer.metrics,
                                         max_inflight_tasks=handle.config.scheduler.max_inflight_tasks,
                                         max_inflight_bytes=handle.config.scheduler.max_inflight_bytes)
    controller = FastOffloadController(adapter,
                                       observer,
                                       handle.config.failure_policy,
                                       scheduler=scheduler,
                                       importance_selector=importance_selector,
                                       hybrid_shadow=hybrid_shadow,
                                       takeover_runtime=takeover_runtime)
    handle._attach(controller)
    return controller


def _normalize_config(config: ConfigInput) -> FastOffloadConfig:
    if isinstance(config, FastOffloadConfig):
        return config
    if isinstance(config, (str, Path)):
        return FastOffloadConfig.from_json(config)
    if isinstance(config, dict):
        return FastOffloadConfig.from_dict(config)
    raise TypeError("FastOffload config must be a FastOffloadConfig, dictionary, or JSON path")
