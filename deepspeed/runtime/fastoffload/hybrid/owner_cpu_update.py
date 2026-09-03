# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Asynchronous owner-local CPU updates for Hybrid B/C and dense parameters."""

import math
import os
import threading
import time
from typing import Dict, Iterable, Mapping

import psutil

import torch

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

from .context import HybridUpdateJob, HybridUpdateResult, clone_tensor_mapping
from .coordinator import HybridUpdateCoordinator
from .owner_collective import OwnerReducedGradientSet, OwnerReducedGradients, Zero2OwnerCollective
from .partition import ParameterPartitionLayout
from .selected_adam import SelectedColumnAdamW


def _run_native_cpu_adam_process(master, exp_avg, exp_avg_sq, transfer_gradients, gradient, return_parameters,
                                 initial_step, hyperparameters, command_queue, result_queue, ready, affinity):
    if affinity:
        os.sched_setaffinity(0, affinity)
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    parameter = torch.nn.Parameter(master, requires_grad=False)
    lr, betas, eps, weight_decay = hyperparameters
    optimizer = DeepSpeedCPUAdam([parameter], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, adamw_mode=True)
    optimizer.state[parameter] = {
        "step": initial_step,
        "exp_avg": exp_avg,
        "exp_avg_sq": exp_avg_sq,
    }
    ready.set()
    while True:
        command = command_queue.get()
        if command == "close":
            break
        _, slot = command
        gradient.copy_(transfer_gradients[slot])
        parameter.grad = gradient
        optimizer.step()
        parameter.grad = None
        return_parameters[0].copy_(parameter)
        result_queue.put(optimizer.state[parameter]["step"])


class Zero2OwnerCpuUpdater:
    """Accumulate owner B gradients and asynchronously update B/C values on CPU."""

    _SECOND_TAG = 0
    _DENSE_TAG = 1
    _FLAT_GRADIENT_ID = -1
    _SECOND_NUMEL_ID = -2

    def __init__(self,
                 adapter,
                 layouts: Iterable[ParameterPartitionLayout],
                 collectives: Mapping[int, Zero2OwnerCollective],
                 update_interval: int,
                 accumulation_device: str,
                 second_gradient_reduction: str,
                 lr: float,
                 betas: tuple[float, float],
                 eps: float,
                 weight_decay: float,
                 max_async_lag: int = 1,
                 metrics=None,
                 pt_reserved_cores_perc: float = 0.25) -> None:
        self._adapter = adapter
        self._layouts = {layout.parameter_id: layout for layout in layouts}
        self._collectives = dict(collectives)
        self._optimizer = SelectedColumnAdamW(lr, betas, eps, weight_decay)
        self._native_parameter = None
        self._native_exp_avg = None
        self._native_exp_avg_sq = None
        self._native_initial_step = 0
        self._native_keys = []
        self._native_offsets = {}
        self._native_dense_staged = False
        self._native_dense_scales = {}
        self._native_gradients = ()
        self._native_fp32_gradient = None
        self._native_return_parameters = ()
        self._native_registered_buffers = ()
        self._native_slot_versions = {}
        self._native_staging_version = None
        self._visibility_event = None
        self._return_visibility_event = None
        self._native_process = None
        self._native_command_queue = None
        self._native_result_queue = None
        self._worker_cores = ()
        self._native_lengths = []
        self._native_hyperparameters = (lr, betas, eps, weight_decay)
        self._published_version = 0
        self._publication_condition = threading.Condition()
        self._second_gradient_reduction = second_gradient_reduction
        self._max_async_lag = max_async_lag
        self._pt_reserved_cores_perc = pt_reserved_cores_perc
        self._transfer_stream = (get_accelerator().Stream()
                                 if accumulation_device == "gpu" and get_accelerator().is_available() else None)
        self._metadata: Dict[int, tuple] = {}
        self._coordinator = HybridUpdateCoordinator(update_interval=update_interval,
                                                    accumulation_device=accumulation_device,
                                                    update_function=self._update,
                                                    second_gradient_reduction=second_gradient_reduction,
                                                    max_async_lag=max_async_lag,
                                                    prepare_function=self._prepare_gradients,
                                                    clone_update_results=False,
                                                    metrics=metrics)

    def initialize_state(self, parameter_id: int, columns: torch.Tensor, dense: bool, step: int, exp_avg: torch.Tensor,
                         exp_avg_sq: torch.Tensor) -> None:
        layout = self._layouts[parameter_id]
        rank = self._adapter.get_partition_rank(layout.group_id)
        values = layout.extract_local_values(self._adapter.get_fp32_partition(layout.group_id), columns, rank).cpu()
        moments = layout.extract_local_values(exp_avg, columns, rank).cpu()
        variances = layout.extract_local_values(exp_avg_sq, columns, rank).cpu()
        tag = self._DENSE_TAG if dense else self._SECOND_TAG
        self._optimizer.initialize_state((parameter_id, tag), values, moments, variances, step)

    def finalize_state_initialization(self) -> None:
        keys = sorted(self._optimizer.state_keys(), key=lambda key: (key[1], key[0]))
        total_numel = sum(self._optimizer.master_values(key).numel() for key in keys)
        if total_numel >= 1_000_000:
            self._configure_cpu_affinity()
            self._initialize_native_optimizer(keys)

    def _configure_cpu_affinity(self) -> None:
        process = psutil.Process()
        affinity = process.cpu_affinity()
        physical_cores = psutil.cpu_count(logical=False) or len(affinity)
        affinity = [core for core in affinity if core < physical_cores]
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", get_accelerator().current_device()))
        expected_rank_cores = max(1, physical_cores // local_world_size)
        if local_world_size > 1 and len(affinity) > expected_rank_cores:
            start = local_rank * expected_rank_cores
            affinity = affinity[start:start + expected_rank_cores]
        split = max(1, math.ceil(len(affinity) * self._pt_reserved_cores_perc))
        training_cores = affinity[:split]
        worker_cores = affinity[split:]
        if not worker_cores:
            worker_cores = training_cores
        process.cpu_affinity(training_cores)
        self._worker_cores = tuple(worker_cores)
        self._coordinator.set_worker_affinity(worker_cores)

    @property
    def pending_updates(self) -> int:
        return self._coordinator.pending_updates

    @property
    def at_capacity(self) -> bool:
        return self._coordinator.at_capacity

    @property
    def committed_version(self) -> int:
        return self._coordinator.committed_version

    def prepare_native_boundary(self) -> None:
        if self._native_parameter is None:
            return
        while self._coordinator.at_capacity:
            self._wait_and_commit_next()
        version = self._coordinator.next_submitted_version
        if self._native_staging_version == version:
            return
        slot = self._native_slot(version)
        previous_version = self._native_slot_versions.get(slot)
        if previous_version is not None:
            if previous_version > self._coordinator.committed_version:
                raise RuntimeError(f"Owner CPU transfer slot {slot} is still owned by version {previous_version}")
            self._native_slot_versions.pop(slot)
        self._native_slot_versions[slot] = version
        self._native_staging_version = version

    def wait_for_visibility(self, synchronize: bool = False) -> None:
        visibility_event = self._visibility_event
        if visibility_event is None:
            return
        if synchronize:
            visibility_event.synchronize()
        else:
            get_accelerator().current_stream().wait_event(visibility_event)
        self._visibility_event = None

    def accumulate_second(self, reduced_by_group: Mapping[int, OwnerReducedGradientSet]) -> None:
        self._coordinator.accumulate_second(self._flatten(reduced_by_group))

    def submit_boundary(self,
                        second_by_group: Mapping[int, OwnerReducedGradientSet],
                        dense_by_group: Mapping[int, OwnerReducedGradientSet],
                        second_columns: Mapping[int, torch.Tensor],
                        dense_columns: Mapping[int, torch.Tensor],
                        dense_scale: float = 1.0) -> int:
        self._initialize_states(second_by_group, second_columns, self._SECOND_TAG)
        self._initialize_states(dense_by_group, dense_columns, self._DENSE_TAG)
        expected_version = self._coordinator.next_submitted_version
        if self._native_parameter is not None:
            self.prepare_native_boundary()
            if self._native_staging_version != expected_version:
                raise RuntimeError("Native owner gradient staging version changed before submission")
        self._native_dense_scales[expected_version] = dense_scale
        version = self._coordinator.submit_boundary(self._flatten(second_by_group), self._flatten(dense_by_group))
        self._metadata[version] = (self._metadata_only(second_by_group), self._metadata_only(dense_by_group),
                                   dict(second_columns), dict(dense_columns))
        self._native_dense_staged = False
        self._native_staging_version = None
        return version

    def native_dense_gradient_view(self, parameter_id: int) -> torch.Tensor:
        location = self._native_offsets.get((parameter_id, self._DENSE_TAG))
        if location is None:
            if self._native_gradients:
                return self._staging_gradient().narrow(0, 0, 0)
            dtype = self._adapter.get_parameter(parameter_id).dtype
            return torch.empty(0, dtype=dtype, device="cpu")
        offset, length = location
        return self._staging_gradient().narrow(0, offset, length)

    def stage_native_dense_gradient(self, parameter_id: int, gradient: torch.Tensor) -> torch.Tensor:
        if not self._native_gradients:
            return gradient.detach().cpu()
        self.prepare_native_boundary()
        offset, length = self._native_offsets[(parameter_id, self._DENSE_TAG)]
        if gradient.numel() != length:
            raise RuntimeError(f"Native dense gradient size mismatch for parameter {parameter_id}")
        target = self._staging_gradient().narrow(0, offset, length)
        producer_stream = get_accelerator().current_stream()
        transfer_stream = self._transfer_stream or producer_stream
        transfer_stream.wait_stream(producer_stream)
        with get_accelerator().stream(transfer_stream):
            target.copy_(gradient.reshape(-1), non_blocking=True)
        self._native_dense_staged = True
        return target

    def discard_native_staging(self) -> None:
        if self._native_staging_version is not None:
            slot = self._native_slot(self._native_staging_version)
            self._native_slot_versions.pop(slot, None)
        self._native_staging_version = None
        self._native_dense_staged = False

    def record_native_transfer_event(self):
        transfer_stream = self._transfer_stream or get_accelerator().current_stream()
        event = get_accelerator().Event()
        event.record(transfer_stream)
        return event

    def discard_active_gradients(self) -> None:
        self._coordinator.discard_active_gradients()

    def commit_if_all_ranks_ready(self) -> int:
        self._coordinator.progress()
        expected = self._coordinator.committed_version + 1
        local_ready = self._coordinator.next_ready_version == expected
        if not self._adapter.all_ranks_ready(local_ready):
            return 0
        return self._coordinator.commit_ready(self._commit, max_versions=1)

    def wait_and_commit_oldest(self) -> None:
        self._wait_and_commit_next()

    def _wait_and_commit_next(self) -> None:
        while self._coordinator.pending_updates:
            self._coordinator.progress()
            expected = self._coordinator.committed_version + 1
            local_ready = self._coordinator.next_ready_version == expected
            if self._adapter.all_ranks_ready(local_ready):
                committed = self._coordinator.commit_ready(self._commit, max_versions=1)
                if committed != 1:
                    raise RuntimeError(f"Expected owner CPU update version {expected} was not committed")
                return
            time.sleep(0.01)

    def state_dict(self):
        while self._coordinator.pending_updates:
            self._wait_and_commit_next()
        self.wait_for_visibility(synchronize=True)
        return {
            "optimizer": self._optimizer.state_dict(),
            "committed_version": self._coordinator.committed_version,
            "active_accumulator": self._coordinator.accumulator.active_state_dict(),
        }

    def load_state_dict(self, state_dict) -> None:
        if self._coordinator.pending_updates or self._coordinator.active_accumulated_steps:
            raise RuntimeError("Cannot restore owner CPU state while updates are active")
        optimizer_state = state_dict.get("optimizer")
        if not isinstance(optimizer_state, dict):
            raise ValueError("Owner CPU checkpoint is missing optimizer state")
        devices = {key: torch.device("cpu") for key in optimizer_state.get("states", {})}
        self._optimizer.load_state_dict(optimizer_state, devices)
        committed_version = int(state_dict.get("committed_version", 0))
        self._coordinator.restore_committed_version(committed_version)
        self._published_version = committed_version
        self._coordinator.accumulator.load_active_state_dict(state_dict.get("active_accumulator", {}))

    def close(self) -> None:
        while self._coordinator.pending_updates:
            self._wait_and_commit_next()
        self._coordinator.close()
        self.wait_for_visibility(synchronize=True)
        if self._native_process is not None:
            self._native_command_queue.put("close")
            self._native_process.join(timeout=60)
            if self._native_process.is_alive():
                self._native_process.terminate()
                raise RuntimeError("Owner CPU optimizer process failed to stop")
        for buffer in self._native_registered_buffers:
            status = torch.cuda.cudart().cudaHostUnregister(buffer.data_ptr())  #ignore-cuda
            if status != torch.cuda.cudart().cudaError.success:  #ignore-cuda
                raise RuntimeError(f"Failed to unregister owner CPU transfer buffer: {status}")
        self._native_registered_buffers = ()

    def _prepare_gradients(self, second_gradients: Mapping[int, torch.Tensor],
                           dense_gradients: Mapping[int, torch.Tensor], interval_steps: int, reduction: str):
        if self._native_parameter is None:
            cpu_second = clone_tensor_mapping(second_gradients, torch.device("cpu"))
            if reduction == "mean":
                for gradient in cpu_second.values():
                    gradient.div_(interval_steps)
            return cpu_second, clone_tensor_mapping(dense_gradients, torch.device("cpu")), None

        second_gradients = dict(second_gradients)
        dense_gradients = dict(dense_gradients)
        for (parameter_id, tag), (offset, length) in self._native_offsets.items():
            destination = second_gradients if tag == self._SECOND_TAG else dense_gradients
            if parameter_id not in destination:
                if length:
                    raise RuntimeError(f"Owner CPU gradient is missing for parameter {parameter_id}")
                destination[parameter_id] = self._staging_gradient().narrow(0, offset, 0)
        keys = ([(parameter_id, self._SECOND_TAG)
                 for parameter_id in sorted(second_gradients)] + [(parameter_id, self._DENSE_TAG)
                                                                  for parameter_id in sorted(dense_gradients)])
        if keys != self._native_keys:
            raise RuntimeError("Owner CPU gradient layout changed after native Adam initialization")
        dense_sources = [] if self._native_dense_staged else [
            dense_gradients[parameter_id] for parameter_id in sorted(dense_gradients)
        ]
        sources = [second_gradients[parameter_id] for parameter_id in sorted(second_gradients)] + dense_sources
        total_numel = self._native_parameter.numel()
        second_numel = sum(gradient.numel() for gradient in second_gradients.values())
        accelerator = get_accelerator()
        cpu_gradient = self._staging_gradient()
        if cpu_gradient.numel() != total_numel:
            raise RuntimeError("Shared owner CPU gradient buffer has an invalid size")
        transfer_event = accelerator.Event()
        producer_stream = accelerator.current_stream()
        transfer_stream = self._transfer_stream or producer_stream
        transfer_stream.wait_stream(producer_stream)
        offset = 0
        with accelerator.stream(transfer_stream):
            for gradient in sources:
                source = gradient.detach().view(-1)
                if source.dtype != cpu_gradient.dtype:
                    source = source.to(dtype=cpu_gradient.dtype)
                cpu_gradient.narrow(0, offset, source.numel()).copy_(source, non_blocking=True)
                offset += source.numel()
            transfer_event.record(transfer_stream)
        metadata = torch.tensor([second_numel], dtype=torch.int64)
        return ({
            self._FLAT_GRADIENT_ID: cpu_gradient,
            self._SECOND_NUMEL_ID: metadata
        }, {}, transfer_event, tuple(sources))

    def _update(self, job: HybridUpdateJob):
        if self._FLAT_GRADIENT_ID in job.second_gradients:
            flat_gradient = job.second_gradients[self._FLAT_GRADIENT_ID]
            second_numel = int(job.second_gradients[self._SECOND_NUMEL_ID].item())
            if self._second_gradient_reduction == "mean":
                flat_gradient.narrow(0, 0, second_numel).div_(job.interval_steps)
            dense_scale = self._native_dense_scales.pop(job.version, 1.0)
            if dense_scale != 1.0:
                flat_gradient.narrow(0, second_numel, flat_gradient.numel() - second_numel).div_(dense_scale)
            return self._native_step(flat_gradient, job.version)
        gradients = {
            (parameter_id, self._SECOND_TAG): gradient
            for parameter_id, gradient in job.second_gradients.items()
        }
        dense_scale = self._native_dense_scales.pop(job.version, 1.0)
        gradients.update({
            (parameter_id, self._DENSE_TAG): gradient.div(dense_scale)
            for parameter_id, gradient in job.dense_gradients.items()
        })
        if self._native_parameter is not None:
            raise RuntimeError("Native owner CPU gradients must use the flat transfer path")
        return {
            self._result_key(parameter_id, tag): self._step(parameter_id, tag, gradient)
            for (parameter_id, tag), gradient in gradients.items()
        }

    def _initialize_native_optimizer(self, keys: list[tuple[int, int]]) -> None:
        from multiprocessing import get_context
        master, exp_avg, exp_avg_sq, lengths, step = self._optimizer.flatten_states(keys)
        master.share_memory_()
        exp_avg.share_memory_()
        exp_avg_sq.share_memory_()
        first_parameter_id = keys[0][0]
        transfer_dtype = self._adapter.get_parameter(first_parameter_id).dtype
        gradients = tuple(
            torch.empty(master.numel(), dtype=transfer_dtype, device="cpu").share_memory_()
            for _ in range(self._max_async_lag))
        fp32_gradient = torch.empty_like(master).share_memory_()
        return_parameters = (torch.empty(master.numel(), dtype=transfer_dtype, device="cpu").share_memory_(), )
        cudart = torch.cuda.cudart()  #ignore-cuda
        registered_buffers = []
        for buffer in gradients + return_parameters:
            status = cudart.cudaHostRegister(buffer.data_ptr(), buffer.nbytes, 0)
            if status != cudart.cudaError.success:
                for registered in registered_buffers:
                    cudart.cudaHostUnregister(registered.data_ptr())
                raise RuntimeError(f"Failed to register owner CPU transfer buffer: {status}")
            registered_buffers.append(buffer)
        from deepspeed.utils.pin_memory_tracker import track_pinned_memory
        track_pinned_memory(sum(buffer.nbytes for buffer in registered_buffers))
        self._native_parameter = torch.nn.Parameter(master, requires_grad=False)
        self._native_exp_avg = exp_avg
        self._native_exp_avg_sq = exp_avg_sq
        self._native_initial_step = step
        self._native_keys = keys
        self._native_lengths = lengths
        offset = 0
        for key, length in zip(keys, lengths):
            self._native_offsets[key] = (offset, length)
            offset += length
        self._native_gradients = gradients
        self._native_fp32_gradient = fp32_gradient
        self._native_return_parameters = return_parameters
        self._native_registered_buffers = tuple(registered_buffers)

        context = get_context("spawn")
        command_queue = context.Queue(maxsize=1)
        result_queue = context.Queue(maxsize=1)
        ready = context.Event()
        process = context.Process(target=_run_native_cpu_adam_process,
                                  args=(master, exp_avg, exp_avg_sq, gradients, fp32_gradient, return_parameters, step,
                                        self._native_hyperparameters, command_queue, result_queue, ready,
                                        self._worker_cores))
        process.daemon = True
        process.start()
        if not ready.wait(timeout=600):
            process.terminate()
            raise RuntimeError("Owner CPU optimizer process failed to become ready")
        self._native_process = process
        self._native_command_queue = command_queue
        self._native_result_queue = result_queue

    def _native_step(self, flat_gradient: torch.Tensor, version: int):
        if self._native_parameter is None or self._native_process is None:
            raise RuntimeError("Native owner CPU optimizer state was not initialized")
        slot = self._native_slot(version)
        if flat_gradient.data_ptr() != self._native_gradients[slot].data_ptr():
            raise RuntimeError("Owner CPU optimizer received an unexpected gradient buffer")
        if not self._native_process.is_alive():
            raise RuntimeError("Owner CPU optimizer process exited unexpectedly")
        with self._publication_condition:
            while version > self._published_version + 1:
                self._publication_condition.wait(timeout=1.0)
        if self._return_visibility_event is not None:
            self._return_visibility_event.synchronize()
            self._return_visibility_event = None
        self._native_command_queue.put(("step", slot))
        step = self._native_result_queue.get()
        self._optimizer.set_state_steps(self._native_keys, step)
        result_values = self._native_return_parameters[0]
        updated = {}
        offset = 0
        for (parameter_id, tag), length in zip(self._native_keys, self._native_lengths):
            updated[self._result_key(parameter_id, tag)] = result_values.narrow(0, offset, length)
            offset += length
        self._native_parameter.grad = None
        return updated

    def _step(self, parameter_id: int, tag: int, gradient: torch.Tensor) -> torch.Tensor:
        key = (parameter_id, tag)
        try:
            values = self._optimizer.master_values(key)
        except KeyError as error:
            raise RuntimeError(f"Owner CPU optimizer state is unavailable for parameter {parameter_id}") from error
        return self._optimizer.step_values(key, values, gradient)

    def _commit(self, result: HybridUpdateResult) -> None:
        metadata = self._metadata.pop(result.version, None)
        if metadata is None:
            raise RuntimeError(f"Missing owner update metadata for version {result.version}")
        second_by_group, dense_by_group, second_columns, dense_columns = metadata
        self._commit_band(result, second_by_group, second_columns, self._SECOND_TAG)
        self._commit_band(result, dense_by_group, dense_columns, self._DENSE_TAG)
        if self._native_return_parameters:
            visibility_event = get_accelerator().Event()
            visibility_event.record(get_accelerator().current_stream())
            self._return_visibility_event = visibility_event
            self._visibility_event = visibility_event
        with self._publication_condition:
            self._published_version = result.version
            self._publication_condition.notify_all()

    def _commit_band(self, result: HybridUpdateResult, reduced_by_group: Mapping[int, OwnerReducedGradientSet],
                     columns: Mapping[int, torch.Tensor], tag: int) -> None:
        for group_id, reduced_set in sorted(reduced_by_group.items()):
            for reduced in getattr(reduced_set, "buckets", (reduced_set, )):
                updated = {}
                group_columns = {}
                for parameter_id in reduced.parameter_ids:
                    values = result.updated_values[self._result_key(parameter_id, tag)]
                    # Publish canonical Hybrid masters directly; the native flat master is no longer on the hot path.
                    device = get_accelerator().current_device_name() if dist.is_initialized() else values.device
                    updated[parameter_id] = values.to(device=device, non_blocking=True)
                    group_columns[parameter_id] = columns[parameter_id]
                owner_values = OwnerReducedGradients(group_id, reduced.rank, reduced.padded_numel, updated,
                                                     reduced.parameter_ids)
                gathered = self._collectives[group_id].all_gather(group_columns, owner_values)
                for compressed in gathered:
                    self._adapter.scatter_parameter_columns(compressed.parameter_id, compressed.columns,
                                                            compressed.values)

    def _initialize_states(self, reduced_by_group: Mapping[int, OwnerReducedGradientSet],
                           columns: Mapping[int, torch.Tensor], tag: int) -> None:
        for group_id, reduced_set in reduced_by_group.items():
            rank = self._adapter.get_partition_rank(group_id)
            for parameter_id, gradient in reduced_set.values.items():
                key = (parameter_id, tag)
                try:
                    self._optimizer.master_values(key)
                    continue
                except KeyError:
                    pass
                layout = self._layouts[parameter_id]
                values = layout.extract_local_values(self._adapter.get_fp32_partition(group_id), columns[parameter_id],
                                                     rank).cpu()
                zeros = torch.zeros_like(values, dtype=torch.float32)
                self._optimizer.initialize_state(key, values, zeros, zeros, 0)
                if values.numel() != gradient.numel():
                    raise RuntimeError(f"Owner gradient layout mismatch for parameter {parameter_id}")

    def _staging_gradient(self) -> torch.Tensor:
        if not self._native_gradients:
            raise RuntimeError("Native owner CPU gradient buffers are unavailable")
        if self._native_staging_version is None:
            self.prepare_native_boundary()
        return self._native_gradients[self._native_slot(self._native_staging_version)]

    def _native_slot(self, version: int) -> int:
        if version < 1:
            raise ValueError("Native owner CPU version must be positive")
        return (version - 1) % self._max_async_lag

    @staticmethod
    def _metadata_only(reduced_by_group: Mapping[int, OwnerReducedGradientSet]):
        return {
            group_id:
            OwnerReducedGradientSet(
                tuple(
                    OwnerReducedGradients(group_id=bucket.group_id,
                                          rank=bucket.rank,
                                          padded_numel=bucket.padded_numel,
                                          values={},
                                          parameter_ids=bucket.parameter_ids,
                                          packed_bytes=bucket.packed_bytes)
                    for bucket in getattr(reduced_set, "buckets", (reduced_set, ))))
            for group_id, reduced_set in reduced_by_group.items()
        }

    @staticmethod
    def _flatten(reduced_by_group: Mapping[int, OwnerReducedGradientSet]):
        return {
            parameter_id: gradient
            for reduced_set in reduced_by_group.values()
            for parameter_id, gradient in reduced_set.values.items()
        }

    @staticmethod
    def _result_key(parameter_id: int, tag: int) -> int:
        return parameter_id * 2 + tag
