# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Packed collectives for compressed hybrid gradients."""

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

import torch

import deepspeed.comm as dist

from .layout import CompressedColumnGradient
from .partition import ParameterPartitionLayout


@dataclass(frozen=True)
class PackedGradientSegment:
    parameter_id: int
    columns: torch.Tensor
    shape: Tuple[int, ...]
    offset: int
    numel: int


class PackedCompressedGradients:
    """One contiguous communication buffer with reconstruction metadata."""

    def __init__(self, gradients: Iterable[CompressedColumnGradient]) -> None:
        items = tuple(gradients)
        if not items:
            raise ValueError("At least one compressed gradient is required")
        first_values = items[0].values
        if not first_values.is_floating_point():
            raise ValueError("Compressed gradients must use a floating-point dtype")
        flattened = []
        segments = []
        offset = 0
        seen_ids = set()
        for gradient in items:
            if gradient.parameter_id in seen_ids:
                raise ValueError("A packed buffer may contain each parameter only once")
            if gradient.values.device != first_values.device or gradient.values.dtype != first_values.dtype:
                raise ValueError("Packed gradients must have the same device and dtype")
            seen_ids.add(gradient.parameter_id)
            values = gradient.values.contiguous().view(-1)
            flattened.append(values)
            segments.append(
                PackedGradientSegment(parameter_id=gradient.parameter_id,
                                      columns=gradient.columns.detach().cpu().long(),
                                      shape=tuple(gradient.values.shape),
                                      offset=offset,
                                      numel=values.numel()))
            offset += values.numel()
        self.buffer = torch.cat(flattened)
        self.segments = tuple(segments)

    def unpack(self) -> Tuple[CompressedColumnGradient, ...]:
        gradients = []
        for segment in self.segments:
            values = self.buffer.narrow(0, segment.offset, segment.numel).view(segment.shape)
            gradients.append(
                CompressedColumnGradient(parameter_id=segment.parameter_id, columns=segment.columns, values=values))
        return tuple(gradients)


class Zero2CompressedGradientReducer:
    """Batch compressed gradients by ZeRO group and run one packed collective per group."""

    def __init__(self, layouts: Iterable[ParameterPartitionLayout], process_groups: Mapping[int, object]) -> None:
        self._layouts = {layout.parameter_id: layout for layout in layouts}
        self._process_groups = dict(process_groups)
        missing_groups = {layout.group_id for layout in self._layouts.values()} - self._process_groups.keys()
        if missing_groups:
            raise ValueError(f"Missing process groups for ZeRO groups: {sorted(missing_groups)}")

    def reduce(self, gradients: Iterable[CompressedColumnGradient]) -> Dict[int, CompressedColumnGradient]:
        by_group: Dict[int, list[CompressedColumnGradient]] = {}
        for gradient in gradients:
            layout = self._layouts.get(gradient.parameter_id)
            if layout is None:
                raise ValueError(f"Missing ZeRO partition layout for parameter {gradient.parameter_id}")
            by_group.setdefault(layout.group_id, []).append(gradient)

        reduced = {}
        for group_id, group_gradients in sorted(by_group.items()):
            process_group = self._process_groups[group_id]
            group_gradients.sort(key=lambda gradient: gradient.parameter_id)
            for gradient in CompressedGradientCollective.all_reduce(group_gradients, process_group):
                reduced[gradient.parameter_id] = gradient
        return reduced


class CompressedGradientCollective:
    """All-reduce only packed selected values, never a dense zero-filled tensor."""

    @staticmethod
    def all_reduce(gradients: Iterable[CompressedColumnGradient],
                   process_group=None,
                   average: bool = True) -> Tuple[CompressedColumnGradient, ...]:
        packed = PackedCompressedGradients(gradients)
        if dist.is_initialized():
            dist.all_reduce(packed.buffer, group=process_group)
            if average:
                world_size = dist.get_world_size(group=process_group)
                packed.buffer.div_(world_size)
        return packed.unpack()
