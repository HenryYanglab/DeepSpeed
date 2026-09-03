# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Immutable metadata passed through FastOffload observer callbacks."""

from dataclasses import dataclass


def _require_int(name: str, value: int, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < minimum:
        raise ValueError(f"{name} must be greater than or equal to {minimum}")


def _require_bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")


def _require_non_empty_string(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _validate_rank(rank: int, world_size: int) -> None:
    _require_int("world_size", world_size, minimum=1)
    _require_int("rank", rank)
    if rank >= world_size:
        raise ValueError("rank must be smaller than world_size")


def _validate_step_metadata(global_step: int, micro_step: int, zero_stage: int, timestamp_ns: int) -> None:
    _require_int("global_step", global_step)
    _require_int("micro_step", micro_step)
    _require_int("zero_stage", zero_stage)
    if zero_stage != 2:
        raise ValueError("Observer context currently supports only ZeRO Stage 2")
    _require_int("timestamp_ns", timestamp_ns)


@dataclass(frozen=True)
class StepContext:
    """Metadata describing a backward or optimizer-step boundary."""

    global_step: int
    micro_step: int
    rank: int
    world_size: int
    zero_stage: int
    gradient_accumulation_boundary: bool
    timestamp_ns: int

    def __post_init__(self) -> None:
        _validate_step_metadata(self.global_step, self.micro_step, self.zero_stage, self.timestamp_ns)
        _validate_rank(self.rank, self.world_size)
        _require_bool("gradient_accumulation_boundary", self.gradient_accumulation_boundary)


@dataclass(frozen=True)
class GradientContext:
    """Tensor-free metadata describing one parameter gradient shard."""

    parameter_id: int
    parameter_name: str
    group_id: int
    partition_id: int
    parameter_numel: int
    shard_numel: int
    shard_offset: int
    element_size: int
    shard_bytes: int
    dtype: str
    device: str
    global_step: int
    micro_step: int
    rank: int
    world_size: int
    zero_stage: int
    gradient_accumulation_boundary: bool
    is_local_partition: bool
    is_cpu_offload: bool
    timestamp_ns: int

    def __post_init__(self) -> None:
        _validate_step_metadata(self.global_step, self.micro_step, self.zero_stage, self.timestamp_ns)
        _validate_rank(self.rank, self.world_size)
        _require_int("parameter_id", self.parameter_id)
        _require_non_empty_string("parameter_name", self.parameter_name)
        _require_int("group_id", self.group_id)
        _require_int("partition_id", self.partition_id)
        if self.partition_id >= self.world_size:
            raise ValueError("partition_id must be smaller than world_size")

        _require_int("parameter_numel", self.parameter_numel, minimum=1)
        _require_int("shard_numel", self.shard_numel)
        _require_int("shard_offset", self.shard_offset)
        _require_int("element_size", self.element_size, minimum=1)
        _require_int("shard_bytes", self.shard_bytes)
        if self.shard_offset + self.shard_numel > self.parameter_numel:
            raise ValueError("Gradient shard exceeds the parameter bounds")
        expected_bytes = self.shard_numel * self.element_size
        if self.shard_bytes != expected_bytes:
            raise ValueError(f"shard_bytes must equal shard_numel * element_size ({expected_bytes})")

        _require_non_empty_string("dtype", self.dtype)
        _require_non_empty_string("device", self.device)
        _require_bool("gradient_accumulation_boundary", self.gradient_accumulation_boundary)
        _require_bool("is_local_partition", self.is_local_partition)
        _require_bool("is_cpu_offload", self.is_cpu_offload)


@dataclass(frozen=True)
class GradientBucketContext:
    """Tensor-free metadata describing one gradient communication bucket."""

    bucket_id: int
    parameter_count: int
    total_numel: int
    total_bytes: int
    communication_dtype: str
    overlap_comm: bool
    global_step: int
    micro_step: int
    rank: int
    world_size: int
    zero_stage: int
    gradient_accumulation_boundary: bool
    timestamp_ns: int

    def __post_init__(self) -> None:
        _validate_step_metadata(self.global_step, self.micro_step, self.zero_stage, self.timestamp_ns)
        _validate_rank(self.rank, self.world_size)
        _require_int("bucket_id", self.bucket_id)
        _require_int("parameter_count", self.parameter_count)
        _require_int("total_numel", self.total_numel)
        _require_int("total_bytes", self.total_bytes)
        _require_non_empty_string("communication_dtype", self.communication_dtype)
        _require_bool("overlap_comm", self.overlap_comm)
        _require_bool("gradient_accumulation_boundary", self.gradient_accumulation_boundary)
