# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""ZeRO Stage 2 metadata adapter for the FastOffload observer."""

import math
import time
from typing import Any, Iterable, Tuple

import torch

import deepspeed.comm as dist

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.fastoffload.context import GradientBucketContext, GradientContext, StepContext
from deepspeed.runtime.fastoffload.hybrid.collective import Zero2CompressedGradientReducer
from deepspeed.runtime.fastoffload.hybrid.owner_collective import Zero2OwnerCollective
from deepspeed.runtime.fastoffload.hybrid.partition import ParameterPartitionLayout
from deepspeed.runtime.fastoffload.importance.context import ParameterImportanceView
from deepspeed.runtime.fastoffload.transfer.context import GradientTransferView

from .base import ObserverAdapter


class Zero2ObserverAdapter(ObserverAdapter):
    """Read ZeRO-2 private state without reading gradient tensor values."""

    def __init__(self, optimizer: Any, max_parameter_name_length: int = 128) -> None:
        if not getattr(optimizer, "partition_gradients", False):
            raise ValueError("FastOffload observer requires ZeRO Stage 2")
        if not getattr(optimizer, "cpu_offload", False):
            raise ValueError("FastOffload observer requires CPU optimizer offload")
        self._optimizer = optimizer
        self._max_parameter_name_length = max_parameter_name_length
        self._global_step = 0
        self._micro_step = 0
        self._bucket_id = 0

    def create_backward_begin_context(self) -> StepContext:
        self._micro_step = max(int(self._optimizer.micro_step_id) + 1, 0)
        accumulation_steps = int(self._optimizer.gradient_accumulation_steps)
        is_boundary = (self._micro_step + 1) % accumulation_steps == 0
        return self._create_step_context(is_boundary)

    def create_step_context(self) -> StepContext:
        return self._create_step_context(self._optimizer.is_gradient_accumulation_boundary())

    def _create_step_context(self, is_boundary: bool) -> StepContext:
        rank, world_size = self._rank_and_world_size(self._optimizer.dp_process_group)
        return StepContext(global_step=self._global_step,
                           micro_step=self._micro_step,
                           rank=rank,
                           world_size=world_size,
                           zero_stage=2,
                           gradient_accumulation_boundary=is_boundary,
                           timestamp_ns=time.perf_counter_ns())

    def create_gradient_context(self, parameter: Any, group_id: int) -> GradientContext:
        self._validate_group_id(group_id)
        process_group = self._optimizer.real_dp_process_group[group_id]
        partition_id, world_size = self._rank_and_world_size(process_group)
        param_id = self._optimizer.get_param_id(parameter)
        is_local_partition = param_id in self._optimizer.grad_position

        shard_offset = 0
        shard_numel = 0
        metadata_tensor = parameter
        if is_local_partition:
            position_group_id, shard_offset, destination_offset, shard_numel = self._optimizer.grad_position[param_id]
            if position_group_id != group_id:
                raise RuntimeError("Gradient position group does not match the parameter group")
            fp32_partition = self._optimizer.single_partition_of_fp32_groups[group_id]
            if fp32_partition.grad is not None:
                metadata_tensor = fp32_partition.grad

        element_size = metadata_tensor.element_size()
        name = self._optimizer.param_names.get(parameter, f"parameter_{param_id}")
        name = name[:self._max_parameter_name_length]
        return GradientContext(parameter_id=param_id,
                               parameter_name=name,
                               group_id=group_id,
                               partition_id=partition_id,
                               parameter_numel=parameter.numel(),
                               shard_numel=shard_numel,
                               shard_offset=shard_offset,
                               element_size=element_size,
                               shard_bytes=shard_numel * element_size,
                               dtype=str(metadata_tensor.dtype),
                               device=str(metadata_tensor.device),
                               global_step=self._global_step,
                               micro_step=self._micro_step,
                               rank=partition_id,
                               world_size=world_size,
                               zero_stage=2,
                               gradient_accumulation_boundary=self._optimizer.is_gradient_accumulation_boundary(),
                               is_local_partition=is_local_partition,
                               is_cpu_offload=self._optimizer.cpu_offload,
                               timestamp_ns=time.perf_counter_ns())

    def create_bucket_context(self, communication_dtype: Any, bucket: Any) -> GradientBucketContext:
        rank, world_size = self._rank_and_world_size(self._optimizer.dp_process_group)
        item_size = getattr(communication_dtype, "itemsize", None)
        if item_size is None:
            item_size = bucket.grads[0].element_size() if bucket.grads else 0
        context = GradientBucketContext(
            bucket_id=self._bucket_id,
            parameter_count=len(bucket.params),
            total_numel=bucket.elements,
            total_bytes=bucket.elements * item_size,
            communication_dtype=str(communication_dtype),
            overlap_comm=self._optimizer.overlap_comm,
            global_step=self._global_step,
            micro_step=self._micro_step,
            rank=rank,
            world_size=world_size,
            zero_stage=2,
            gradient_accumulation_boundary=self._optimizer.is_gradient_accumulation_boundary(),
            timestamp_ns=time.perf_counter_ns())
        self._bucket_id += 1
        return context

    def get_parameter_id(self, parameter: Any) -> int:
        return self._optimizer.get_param_id(parameter)

    def get_parameter(self, parameter_id: int) -> Any:
        parameter = self._optimizer.param_dict.get(parameter_id)
        if parameter is None:
            raise ValueError(f"Unknown ZeRO parameter ID: {parameter_id}")
        return parameter

    def get_gradient_tensor(self, parameter: Any) -> Any:
        return self._optimizer.get_param_gradient_attribute(parameter)

    def get_data_parallel_group(self, group_id: int = 0) -> Any:
        self._validate_group_id(group_id)
        return self._optimizer.real_dp_process_group[group_id]

    def get_partition_rank(self, group_id: int) -> int:
        self._validate_group_id(group_id)
        rank, _ = self._rank_and_world_size(self._optimizer.real_dp_process_group[group_id])
        return rank

    def is_gradient_accumulation_boundary(self) -> bool:
        return self._optimizer.is_gradient_accumulation_boundary()

    def get_local_reduced_gradient(self, parameter):
        parameter_id = self.get_parameter_id(parameter)
        group_id, source_offset, _, numel = self._optimizer.grad_position[parameter_id]
        self._validate_group_id(group_id)
        gradient = self.get_gradient_tensor(parameter)
        if gradient is None:
            raise RuntimeError(f"Reduced gradient is unavailable for parameter {parameter_id}")
        return gradient.view(-1).narrow(0, source_offset, numel)

    def copy_local_reduced_gradient(self, parameter, destination: Any) -> None:
        parameter_id = self.get_parameter_id(parameter)
        group_id, source_offset, destination_offset, numel = self._optimizer.grad_position[parameter_id]
        self._validate_group_id(group_id)
        gradient = self.get_gradient_tensor(parameter)
        if gradient is None:
            raise RuntimeError(f"Reduced gradient is unavailable for parameter {parameter_id}")
        source = gradient.view(-1).narrow(0, source_offset, numel)
        target = destination.view(-1).narrow(0, destination_offset, numel)
        target.copy_(source, non_blocking=True)

    def get_cpu_gradient_partition(self, group_id: int) -> Any:
        self._validate_group_id(group_id)
        gradient = self._optimizer.single_partition_of_fp32_groups[group_id].grad
        if gradient is None:
            raise RuntimeError(f"Native owner gradient partition {group_id} is unavailable")
        return gradient.view(-1)

    def compute_native_gradient_numerics(self, loss_scale: float, clip_grad: float):
        self._optimizer.check_overflow(partition_gradients=self._optimizer.partition_gradients)
        overflow = bool(self._optimizer.overflow)
        scaled_norm = float(self._optimizer.scaled_global_norm())
        global_norm = scaled_norm / loss_scale
        clip_factor = 1.0
        if clip_grad > 0.0 and global_norm > clip_grad:
            clip_factor = global_norm / clip_grad
        return overflow, global_norm, loss_scale * clip_factor

    def get_fp32_partition(self, group_id: int) -> Any:
        self._validate_group_id(group_id)
        return self._optimizer.single_partition_of_fp32_groups[group_id].view(-1)

    def get_learning_rate(self) -> float:
        """Read the scheduler-controlled LR without exposing ZeRO state to the runtime."""
        rates = [float(group["lr"]) for group in self._optimizer.optimizer.param_groups]
        if not rates or any(lr != rates[0] for lr in rates):
            raise ValueError("ZeRO-2 takeover requires identical learning rates across optimizer groups")
        lr = rates[0]
        if not math.isfinite(lr) or lr < 0.0:
            raise ValueError("Takeover learning rate must be finite and non-negative")
        return lr

    def get_dense_adam_partition_state(self, group_id: int) -> tuple[int, Any, Any]:
        self._validate_group_id(group_id)
        partition = self._optimizer.single_partition_of_fp32_groups[group_id]
        state = self._optimizer.optimizer.state.get(partition)
        if not state or "exp_avg" not in state or "exp_avg_sq" not in state:
            raise RuntimeError(f"Dense Adam state is unavailable for ZeRO group {group_id}")
        step = state.get("step", 0)
        if torch.is_tensor(step):
            step = int(step.item())
        return int(step), state["exp_avg"].view(-1), state["exp_avg_sq"].view(-1)

    def get_fp32_gradient_partition(self, group_id: int) -> Any:
        self._validate_group_id(group_id)
        gradient = self._optimizer.single_partition_of_fp32_groups[group_id].grad
        if gradient is None:
            raise RuntimeError("ZeRO CPU gradient partition has not been initialized")
        return gradient.view(-1)

    @staticmethod
    def synchronize_gradient_transfers() -> None:
        get_accelerator().synchronize()

    def get_rank(self) -> int:
        rank, _ = self._rank_and_world_size(self._optimizer.dp_process_group)
        return rank

    def iter_parameter_importance_views(self) -> Iterable[ParameterImportanceView]:
        seen_parameter_ids = set()
        for group in self._optimizer.bit16_groups:
            for parameter in group:
                parameter_id = self._optimizer.get_param_id(parameter)
                if parameter_id in seen_parameter_ids:
                    continue
                seen_parameter_ids.add(parameter_id)
                parameter_name = self._optimizer.param_names.get(parameter, f"parameter_{parameter_id}")
                yield ParameterImportanceView(parameter_id=parameter_id,
                                              parameter_name=parameter_name,
                                              tensor=parameter)

    def iter_parameter_partition_layouts(self) -> Iterable[ParameterPartitionLayout]:
        for group_id, parameters in enumerate(self._optimizer.round_robin_bit16_groups):
            group_offset = 0
            partition_size = int(self._optimizer.partition_size[group_id])
            process_group = self._optimizer.real_dp_process_group[group_id]
            _, world_size = self._rank_and_world_size(process_group)
            for parameter in parameters:
                shape = tuple(parameter.shape) if parameter.dim() == 2 else (1, parameter.numel())
                yield ParameterPartitionLayout(parameter_id=self._optimizer.get_param_id(parameter),
                                               group_id=group_id,
                                               shape=shape,
                                               group_offset=group_offset,
                                               partition_size=partition_size,
                                               world_size=world_size)
                group_offset += parameter.numel()

    def all_ranks_ready(self, ready: bool, group_id: int = 0) -> bool:
        self._validate_group_id(group_id)
        device = get_accelerator().current_device_name()
        flag = torch.tensor(1 if ready else 0, dtype=torch.int32, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self._optimizer.real_dp_process_group[group_id])
        return bool(flag.item())

    def create_owner_collectives(self) -> dict[int, Zero2OwnerCollective]:
        layouts_by_group = {}
        for layout in self.iter_parameter_partition_layouts():
            layouts_by_group.setdefault(layout.group_id, []).append(layout)
        return {
            group_id: Zero2OwnerCollective(layouts, self._optimizer.real_dp_process_group[group_id])
            for group_id, layouts in layouts_by_group.items()
        }

    def create_compressed_gradient_reducer(self) -> Zero2CompressedGradientReducer:
        process_groups = {
            group_id: process_group
            for group_id, process_group in enumerate(self._optimizer.real_dp_process_group)
        }
        return Zero2CompressedGradientReducer(self.iter_parameter_partition_layouts(), process_groups)

    def read_fp32_partition_values(self, group_id: int, offsets: Any, device: Any) -> Any:
        self._validate_group_id(group_id)
        partition = self._optimizer.single_partition_of_fp32_groups[group_id].view(-1)
        device_offsets = offsets.to(device=partition.device, dtype=torch.long)
        return partition.detach().index_select(0, device_offsets).to(device=device)

    @torch.no_grad()
    def scatter_parameter_columns(self, parameter_id: int, columns: Any, values: Any) -> None:
        parameter = self._optimizer.param_dict.get(parameter_id)
        if parameter is None:
            raise ValueError(f"Unknown ZeRO parameter ID: {parameter_id}")
        device_columns = columns.to(device=parameter.device, dtype=torch.long)
        device_values = values.to(device=parameter.device, dtype=parameter.dtype)
        if parameter.dim() == 2:
            parameter.index_copy_(1, device_columns, device_values)
        else:
            parameter.view(1, -1).index_copy_(1, device_columns, device_values.view(1, -1))

    @torch.no_grad()
    def write_fp32_partition_values(self, group_id: int, offsets: Any, values: Any) -> None:
        self._validate_group_id(group_id)
        partition = self._optimizer.single_partition_of_fp32_groups[group_id].view(-1)
        device_offsets = offsets.to(device=partition.device, dtype=torch.long)
        device_values = values.to(device=partition.device, dtype=partition.dtype)
        partition.index_copy_(0, device_offsets, device_values)

    def publish_fp32_partitions(self, group_ids: Iterable[int]) -> None:
        group_ids = tuple(group_ids)
        for group_id in group_ids:
            self._validate_group_id(group_id)
        if group_ids:
            self._optimizer.update_lp_params()

    def create_gradient_transfer_view(self, parameter: Any, stable_source: bool = False) -> GradientTransferView:
        param_id = self._optimizer.get_param_id(parameter)
        if param_id not in self._optimizer.grad_position:
            raise ValueError("Cannot transfer a gradient outside the local ZeRO partition")
        group_id, source_offset, destination_offset, numel = self._optimizer.grad_position[param_id]
        self._validate_group_id(group_id)

        gradient = self._optimizer.get_param_gradient_attribute(parameter)
        if gradient is None:
            raise RuntimeError("Cannot transfer a parameter without a gradient")
        source = gradient.view(-1).narrow(0, source_offset, numel)
        target_dtype = self._optimizer.master_weights_and_grads_dtype
        if not stable_source and source.dtype != target_dtype:
            source = source.to(target_dtype)

        fp32_gradient = self._optimizer.single_partition_of_fp32_groups[group_id].grad
        if fp32_gradient is None:
            raise RuntimeError("ZeRO CPU gradient partition has not been initialized")
        destination = fp32_gradient.view(-1).narrow(0, destination_offset, numel)
        return GradientTransferView(source=source,
                                    destination=destination,
                                    parameter_id=param_id,
                                    group_id=group_id,
                                    source_offset=source_offset,
                                    destination_offset=destination_offset,
                                    numel=numel)

    def clear_gradient(self, parameter: Any) -> None:
        self._optimizer.clear_grad_attribute(parameter)

    def complete_step(self) -> None:
        self._global_step += 1
        self._micro_step = 0

    def _validate_group_id(self, group_id: int) -> None:
        if group_id < 0 or group_id >= len(self._optimizer.real_dp_process_group):
            raise ValueError(f"Invalid parameter group ID: {group_id}")

    @staticmethod
    def _rank_and_world_size(process_group: Any) -> Tuple[int, int]:
        if not dist.is_initialized():
            return 0, 1
        return dist.get_rank(group=process_group), dist.get_world_size(group=process_group)
