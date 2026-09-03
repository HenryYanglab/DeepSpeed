# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Owner-padded reduce-scatter and all-gather for compressed ZeRO-2 values."""

from dataclasses import dataclass
from typing import Iterable, Mapping, Tuple

import torch

import deepspeed.comm as dist

from .layout import CompressedColumnGradient
from .partition import ParameterPartitionLayout


@dataclass(frozen=True)
class OwnerReducedGradientSet:
    buckets: Tuple["OwnerReducedGradients", ...]

    @property
    def bucket_count(self) -> int:
        return len(self.buckets)

    @property
    def peak_packed_bytes(self) -> int:
        return max((bucket.packed_bytes for bucket in self.buckets), default=0)

    @property
    def total_packed_bytes(self) -> int:
        return sum(bucket.packed_bytes for bucket in self.buckets)

    @property
    def values(self) -> Mapping[int, torch.Tensor]:
        merged = {}
        for bucket in self.buckets:
            overlap = set(merged).intersection(bucket.values)
            if overlap:
                raise RuntimeError(f"Owner gradient parameters occur in multiple buckets: {sorted(overlap)}")
            merged.update(bucket.values)
        return merged


@dataclass(frozen=True)
class OwnerReducedGradients:
    group_id: int
    rank: int
    padded_numel: int
    values: Mapping[int, torch.Tensor]
    parameter_ids: Tuple[int, ...] = ()
    packed_bytes: int = 0


class Zero2OwnerCollective:
    """Reduce gradients to owners and gather owner-updated values with deterministic padding."""

    def __init__(self, layouts: Iterable[ParameterPartitionLayout], process_group=None) -> None:
        layouts = tuple(layouts)
        if not layouts:
            raise ValueError("Owner collective requires at least one parameter layout")
        group_ids = {layout.group_id for layout in layouts}
        world_sizes = {layout.world_size for layout in layouts}
        if len(group_ids) != 1 or len(world_sizes) != 1:
            raise ValueError("Owner collective layouts must belong to one ZeRO group")
        self._layouts = {layout.parameter_id: layout for layout in layouts}
        self._group_id = layouts[0].group_id
        self._world_size = layouts[0].world_size
        self._process_group = process_group

    def reduce_scatter_buckets(self,
                               gradients: Iterable[CompressedColumnGradient],
                               bucket_bytes: int,
                               average: bool = True) -> OwnerReducedGradientSet:
        if bucket_bytes < 1:
            raise ValueError("Owner collective bucket_bytes must be positive")
        buckets = []
        current = []
        current_bytes = 0
        for gradient in sorted(gradients, key=lambda item: item.parameter_id):
            gradient_bytes = gradient.values.numel() * gradient.values.element_size()
            if current and current_bytes + gradient_bytes > bucket_bytes:
                buckets.append(self.reduce_scatter(current, average=average))
                current = []
                current_bytes = 0
            current.append(gradient)
            current_bytes += gradient_bytes
        if current:
            buckets.append(self.reduce_scatter(current, average=average))
        return OwnerReducedGradientSet(tuple(buckets))

    def reduce_scatter(self,
                       gradients: Iterable[CompressedColumnGradient],
                       average: bool = True) -> OwnerReducedGradients:
        gradients = tuple(sorted(gradients, key=lambda item: item.parameter_id))
        plan, owner_counts = self._build_plan(gradients)
        padded_numel = max(owner_counts) if owner_counts else 0
        if padded_numel == 0:
            rank = dist.get_rank(group=self._process_group) if dist.is_initialized() else 0
            return OwnerReducedGradients(self._group_id, rank, 0, {}, tuple(item.parameter_id for item in gradients),
                                         0)
        first = gradients[0].values
        packed = torch.zeros(self._world_size * padded_numel, dtype=first.dtype, device=first.device)
        gradients_by_id = {gradient.parameter_id: gradient for gradient in gradients}
        for parameter_id, owner, owner_start, count, compressed_start in plan:
            if not count:
                continue
            flat_values = gradients_by_id[parameter_id].values.reshape(-1)
            source = flat_values.narrow(0, compressed_start, count)
            destination_start = owner * padded_numel + owner_start
            packed.narrow(0, destination_start, count).copy_(source)
        rank = dist.get_rank(group=self._process_group) if dist.is_initialized() else 0
        output = torch.empty(padded_numel, dtype=first.dtype, device=first.device)
        if dist.is_initialized():
            dist.reduce_scatter_fn(output, packed, group=self._process_group)
            if average:
                output.div_(self._world_size)
        else:
            output.copy_(packed.narrow(0, 0, padded_numel))
        local_values = {}
        for parameter_id, owner, owner_start, count, _ in plan:
            if owner == rank:
                local_values[parameter_id] = output.narrow(0, owner_start, count)
        packed_bytes = packed.numel() * packed.element_size()
        return OwnerReducedGradients(self._group_id, rank, padded_numel, local_values,
                                     tuple(item.parameter_id for item in gradients), packed_bytes)

    def create_owner_gradient_set(self, columns: Mapping[int, torch.Tensor], values: Mapping[int, torch.Tensor],
                                  bucket_bytes: int) -> OwnerReducedGradientSet:
        gradients = []
        for parameter_id, parameter_columns in sorted(columns.items()):
            layout = self._layouts[parameter_id]
            rows, _ = layout.shape
            gradients.append(
                CompressedColumnGradient(parameter_id, parameter_columns,
                                         torch.empty((rows, parameter_columns.numel()), device="meta")))
        buckets = []
        current = []
        current_bytes = 0
        for gradient in gradients:
            value = values[gradient.parameter_id]
            rows, _ = self._layouts[gradient.parameter_id].shape
            gradient_bytes = rows * gradient.columns.numel() * value.element_size()
            if current and current_bytes + gradient_bytes > bucket_bytes:
                buckets.append(self._create_owner_metadata(current, values))
                current = []
                current_bytes = 0
            current.append(gradient)
            current_bytes += gradient_bytes
        if current:
            buckets.append(self._create_owner_metadata(current, values))
        return OwnerReducedGradientSet(tuple(buckets))

    def _create_owner_metadata(self, gradients, values):
        plan, owner_counts = self._build_plan(gradients)
        padded_numel = max(owner_counts) if owner_counts else 0
        rank = dist.get_rank(group=self._process_group) if dist.is_initialized() else 0
        local_values = {}
        for gradient in gradients:
            expected = sum(count for parameter_id, owner, _, count, _ in plan
                           if parameter_id == gradient.parameter_id and owner == rank)
            value = values[gradient.parameter_id]
            if value.numel() != expected:
                raise RuntimeError(f"Owner gradient size mismatch for parameter {gradient.parameter_id}")
            local_values[gradient.parameter_id] = value
        item_size = next(iter(values.values())).element_size() if values else 0
        packed_bytes = self._world_size * padded_numel * item_size
        return OwnerReducedGradients(self._group_id, rank, padded_numel, local_values,
                                     tuple(gradient.parameter_id for gradient in gradients), packed_bytes)

    def all_gather(self, columns: Mapping[int, torch.Tensor],
                   owner_values: OwnerReducedGradients) -> Tuple[CompressedColumnGradient, ...]:
        if owner_values.group_id != self._group_id or owner_values.rank >= self._world_size:
            raise ValueError("Owner values do not match this collective")
        metadata = []
        for parameter_id, parameter_columns in sorted(columns.items()):
            layout = self._layouts.get(parameter_id)
            if layout is None:
                raise ValueError(f"Missing owner layout for parameter {parameter_id}")
            rows, _ = layout.shape
            metadata.append(
                CompressedColumnGradient(parameter_id, parameter_columns,
                                         torch.empty((rows, parameter_columns.numel()), device="meta")))
        plan, owner_counts = self._build_plan(metadata)
        padded_numel = max(owner_counts) if owner_counts else 0
        if padded_numel != owner_values.padded_numel:
            raise ValueError("Owner all-gather layout differs from reduce-scatter layout")
        if not owner_values.values:
            if padded_numel:
                raise ValueError("Local owner values are missing")
            return ()
        first = next(iter(owner_values.values.values()))
        local = torch.zeros(padded_numel, dtype=first.dtype, device=first.device)
        for parameter_id, owner, owner_start, count, _ in plan:
            if owner != owner_values.rank or not count:
                continue
            values = owner_values.values.get(parameter_id)
            if values is None or values.numel() != count:
                raise ValueError(f"Owner values are incomplete for parameter {parameter_id}")
            local.narrow(0, owner_start, count).copy_(values.reshape(-1))
        gathered = torch.empty(self._world_size * padded_numel, dtype=first.dtype, device=first.device)
        if dist.is_initialized():
            dist.allgather_fn(gathered, local, group=self._process_group)
        else:
            gathered.copy_(local)
        results = []
        for metadata_gradient in metadata:
            layout = self._layouts[metadata_gradient.parameter_id]
            rows, _ = layout.shape
            flat = torch.empty(rows * metadata_gradient.columns.numel(), dtype=first.dtype, device=first.device)
            for parameter_id, owner, owner_start, count, compressed_start in plan:
                if parameter_id != metadata_gradient.parameter_id or not count:
                    continue
                source = gathered.narrow(0, owner * padded_numel + owner_start, count)
                flat.narrow(0, compressed_start, count).copy_(source)
            values = flat.view(rows, metadata_gradient.columns.numel())
            results.append(CompressedColumnGradient(metadata_gradient.parameter_id, metadata_gradient.columns, values))
        return tuple(results)

    def _build_plan(self, gradients: Iterable[CompressedColumnGradient]):
        owner_offsets = [0] * self._world_size
        plan = []
        for gradient in gradients:
            layout = self._layouts.get(gradient.parameter_id)
            if layout is None:
                raise ValueError(f"Missing owner layout for parameter {gradient.parameter_id}")
            compressed_start = 0
            for owner, count in enumerate(layout.owner_counts(gradient.columns)):
                plan.append((gradient.parameter_id, owner, owner_offsets[owner], count, compressed_start))
                owner_offsets[owner] += count
                compressed_start += count
        return plan, owner_offsets
