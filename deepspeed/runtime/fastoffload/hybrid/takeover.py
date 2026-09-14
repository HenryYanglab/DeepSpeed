# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Compressed gradient capture and owner reduction for ZeRO-2 takeover."""

from dataclasses import dataclass
from typing import Dict, Mapping

import torch

from .layout import CompressedColumnGradient, HybridColumnLayout
from .microbatch import CompressedMicrobatchAccumulator
from .owner_collective import OwnerReducedGradientSet


@dataclass(frozen=True)
class TakeoverGradientBatch:
    first: Mapping[int, OwnerReducedGradientSet]
    second: Mapping[int, OwnerReducedGradientSet]
    dense: Mapping[int, OwnerReducedGradientSet]
    first_columns: Mapping[int, torch.Tensor]
    second_columns: Mapping[int, torch.Tensor]
    dense_columns: Mapping[int, torch.Tensor]
    dense_boundary: bool
    native_boundary: bool = False


class Zero2TakeoverGradientPipeline:
    """Capture A/B/C gradients, apply GAS locally, and reduce-scatter them to owners."""

    def __init__(self,
                 adapter,
                 importance_registry,
                 update_interval: int,
                 gradient_accumulation_steps: int,
                 compressed_bucket_bytes: int = 134217728,
                 metrics=None) -> None:
        self._adapter = adapter
        self._registry = importance_registry
        if compressed_bucket_bytes < 1:
            raise ValueError("compressed_bucket_bytes must be positive")
        self._update_interval = update_interval
        self._compressed_bucket_bytes = compressed_bucket_bytes
        self._metrics = metrics
        self._bucket_peak_bytes = 0
        self._collectives = adapter.create_owner_collectives()
        self._layouts = {layout.parameter_id: layout for layout in adapter.iter_parameter_partition_layouts()}
        # DeepSpeed scales each microbatch loss by GAS before backward, so summation reproduces its native gradient.
        self._first_accumulator = CompressedMicrobatchAccumulator(gradient_accumulation_steps, reduction="sum")
        self._second_accumulator = CompressedMicrobatchAccumulator(gradient_accumulation_steps, reduction="sum")
        self._dense_accumulator = CompressedMicrobatchAccumulator(gradient_accumulation_steps, reduction="sum")
        self._first_columns: Dict[int, torch.Tensor] = {}
        self._second_columns: Dict[int, torch.Tensor] = {}
        self._dense_columns: Dict[int, torch.Tensor] = {}
        self._packed_positions: Dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._device_packed_positions: Dict[tuple[int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}
        self._micro_first: Dict[int, torch.Tensor] = {}
        self._micro_second: Dict[int, torch.Tensor] = {}
        self._micro_dense: Dict[int, torch.Tensor] = {}
        self._successful_steps = 0

    @property
    def active(self) -> bool:
        return self._registry.ready

    @property
    def dense_boundary(self) -> bool:
        return (self._successful_steps + 1) % self._update_interval == 0

    def capture(self, parameter, group_id: int) -> bool:
        if not self.active:
            return False
        parameter_id = self._adapter.get_parameter_id(parameter)
        gradient = self._adapter.get_gradient_tensor(parameter)
        layout = self._layouts.get(parameter_id)
        if gradient is None or layout is None:
            return False
        selection = self._registry.get(parameter_id)
        if selection is None:
            if not self._registry.is_dense(parameter_id) or not self.dense_boundary:
                return True
            columns = torch.arange(layout.shape[1], dtype=torch.long)
            values = gradient.detach().reshape(1, -1)
            self._dense_columns[parameter_id] = columns
            self._micro_dense[parameter_id] = values
            return True
        column_layout = HybridColumnLayout(parameter_id, gradient.shape[1], selection.first_indices,
                                           selection.second_indices)
        first = column_layout.pack_first(gradient)
        second = column_layout.pack_second(gradient)
        self._first_columns[parameter_id] = first.columns
        self._second_columns[parameter_id] = second.columns
        self._micro_first[parameter_id] = first.values
        self._micro_second[parameter_id] = second.values
        if self.dense_boundary:
            dense = column_layout.pack_dense_remainder(gradient)
            self._dense_columns[parameter_id] = dense.columns
            self._micro_dense[parameter_id] = dense.values
        return True

    def configure_packed_columns(self, parameter_id: int, columns: torch.Tensor) -> None:
        selection = self._registry.get(parameter_id)
        if selection is None:
            raise ValueError(f"Packed takeover columns have no matrix selection for parameter {parameter_id}")
        cpu_columns = columns.detach().to(device="cpu", dtype=torch.long)
        first_columns = selection.first_indices.to(device="cpu", dtype=torch.long)
        second_columns = selection.second_indices.to(device="cpu", dtype=torch.long)
        first_positions = torch.searchsorted(cpu_columns, first_columns)
        second_positions = torch.searchsorted(cpu_columns, second_columns)
        if (first_positions.numel() and first_positions.max().item() >= cpu_columns.numel()
                or second_positions.numel() and second_positions.max().item() >= cpu_columns.numel()
                or not torch.equal(cpu_columns.index_select(0, first_positions), first_columns)
                or not torch.equal(cpu_columns.index_select(0, second_positions), second_columns)):
            raise ValueError(f"Packed takeover columns differ from importance selection for parameter {parameter_id}")
        self._packed_positions[parameter_id] = (first_positions, second_positions)

    def capture_packed(self, parameter_id: int, columns: torch.Tensor, values: torch.Tensor) -> None:
        if not self.active:
            raise RuntimeError("Packed takeover gradient arrived before importance selection")
        selection = self._registry.get(parameter_id)
        if selection is None:
            raise ValueError(f"Packed takeover gradient has no matrix selection for parameter {parameter_id}")
        if values.dim() != 2 or values.shape[1] != columns.numel():
            raise ValueError("Packed takeover gradient shape does not match its columns")
        positions = self._packed_positions.get(parameter_id)
        if positions is None:
            raise RuntimeError(f"Packed takeover columns were not configured for parameter {parameter_id}")
        device_key = (parameter_id, values.device)
        device_positions = self._device_packed_positions.get(device_key)
        if device_positions is None:
            device_positions = tuple(position.to(device=values.device) for position in positions)
            self._device_packed_positions[device_key] = device_positions
        first_positions, second_positions = device_positions
        self._first_columns[parameter_id] = selection.first_indices
        self._second_columns[parameter_id] = selection.second_indices
        self._accumulate_micro_gradient(self._micro_first, parameter_id,
                                        values.index_select(1, first_positions).detach())
        self._accumulate_micro_gradient(self._micro_second, parameter_id,
                                        values.index_select(1, second_positions).detach())

    def finish_microbatch(self) -> TakeoverGradientBatch | None:
        first_boundary = self._first_accumulator.accumulate(self._micro_first, take_ownership=True)
        second_boundary = self._second_accumulator.accumulate(self._micro_second, take_ownership=True)
        dense_boundary = self._dense_accumulator.accumulate(self._micro_dense, take_ownership=True)
        self._micro_first = {}
        self._micro_second = {}
        self._micro_dense = {}
        if first_boundary != second_boundary or first_boundary != dense_boundary:
            raise RuntimeError("Hybrid GAS accumulators reached inconsistent boundaries")
        if not first_boundary:
            return None
        first = self._reduce_band(self._first_accumulator.take_boundary(), self._first_columns)
        second = self._reduce_band(self._second_accumulator.take_boundary(), self._second_columns)
        dense = self._reduce_band(self._dense_accumulator.take_boundary(), self._dense_columns)
        return TakeoverGradientBatch(first=first,
                                     second=second,
                                     dense=dense,
                                     first_columns=dict(self._first_columns),
                                     second_columns=dict(self._second_columns),
                                     dense_columns=dict(self._dense_columns),
                                     dense_boundary=self.dense_boundary)

    def complete_step(self, overflow: bool) -> None:
        if not overflow:
            self._successful_steps += 1

    def state_dict(self):
        return {
            "successful_steps": self._successful_steps,
            "importance_registry": self._registry.state_dict(),
        }

    def load_state_dict(self, state_dict) -> None:
        if self._micro_first or self._micro_second or self._micro_dense:
            raise RuntimeError("Cannot restore takeover state during a microbatch")
        if not self._registry.ready:
            self._registry.load_state_dict(state_dict["importance_registry"])
        self._successful_steps = int(state_dict.get("successful_steps", 0))
        if self._successful_steps < 0:
            raise ValueError("Successful takeover step count must not be negative")

    def discard_microbatches(self) -> None:
        self._first_accumulator.discard()
        self._second_accumulator.discard()
        self._dense_accumulator.discard()
        self._micro_first.clear()
        self._micro_second.clear()
        self._micro_dense.clear()

    @staticmethod
    def _accumulate_micro_gradient(destination: Dict[int, torch.Tensor], parameter_id: int,
                                   gradient: torch.Tensor) -> None:
        current = destination.get(parameter_id)
        if current is None:
            destination[parameter_id] = gradient
            return
        if current.shape != gradient.shape:
            raise ValueError(f"Packed gradient layout changed for parameter {parameter_id}")
        current.add_(gradient)

    def _reduce_band(self, values: Mapping[int, torch.Tensor], columns: Mapping[int, torch.Tensor]):
        by_group: Dict[int, list[CompressedColumnGradient]] = {}
        for parameter_id, gradient in values.items():
            layout = self._layouts[parameter_id]
            compressed = CompressedColumnGradient(parameter_id, columns[parameter_id], gradient)
            by_group.setdefault(layout.group_id, []).append(compressed)
        reduced_by_group = {
            group_id: self._collectives[group_id].reduce_scatter_buckets(gradients, self._compressed_bucket_bytes)
            for group_id, gradients in sorted(by_group.items())
        }
        if self._metrics is not None:
            bucket_count = sum(reduced.bucket_count for reduced in reduced_by_group.values())
            packed_bytes = sum(reduced.total_packed_bytes for reduced in reduced_by_group.values())
            peak_bytes = max((reduced.peak_packed_bytes for reduced in reduced_by_group.values()), default=0)
            self._bucket_peak_bytes = max(self._bucket_peak_bytes, peak_bytes)
            self._metrics.increment("takeover_owner_bucket_count", bucket_count)
            self._metrics.increment("takeover_owner_packed_bytes", packed_bytes)
            self._metrics.set_gauge("takeover_owner_bucket_peak_bytes", self._bucket_peak_bytes)
        return reduced_by_group
