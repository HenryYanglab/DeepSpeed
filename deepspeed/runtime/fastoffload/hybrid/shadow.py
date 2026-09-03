# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Gated shadow execution of compressed collectives beside native ZeRO-2."""

import math
from dataclasses import dataclass
from typing import Dict

import torch

from .layout import CompressedColumnGradient, HybridColumnLayout


@dataclass(frozen=True)
class _OwnedReducedGradient:
    columns: torch.Tensor
    local_offsets: torch.Tensor
    values: torch.Tensor


class HybridCompressedCollectiveShadow:
    """Validate bounded compressed communication while native ZeRO performs the update."""

    def __init__(self, adapter, importance_registry, metrics, update_interval: int, bucket_bytes: int,
                 parity_atol: float, parity_rtol: float) -> None:
        self._adapter = adapter
        self._registry = importance_registry
        self._metrics = metrics
        self._update_interval = update_interval
        self._bucket_bytes = bucket_bytes
        self._parity_atol = parity_atol
        self._parity_rtol = parity_rtol
        self._reducer = adapter.create_compressed_gradient_reducer()
        self._partition_layouts = {
            layout.parameter_id: layout
            for layout in adapter.iter_parameter_partition_layouts()
        }
        self._pending: Dict[int, CompressedColumnGradient] = {}
        self._pending_bytes = 0
        self._reduced: Dict[int, _OwnedReducedGradient] = {}
        self._native_values: Dict[int, torch.Tensor] = {}
        self._native_owner_values: Dict[int, torch.Tensor] = {}
        self._native_owner_offsets: Dict[int, torch.Tensor] = {}
        self._parity_complete = set()
        self._owner_complete = set()
        self._step = 0
        self._closed = False
        self._compressed_norm_squared = 0.0
        self._native_norm_squared = 0.0
        self._compressed_overflow = False
        self._native_overflow = False
        self._peaks: Dict[str, int] = {}

    def capture(self, parameter, group_id: int) -> None:
        if self._closed or not self._registry.ready:
            return
        parameter_id = self._adapter.get_parameter_id(parameter)
        selection = self._registry.get(parameter_id)
        if selection is None:
            return
        gradient = self._adapter.get_gradient_tensor(parameter)
        if gradient is None:
            return
        layout = HybridColumnLayout(parameter_id, gradient.shape[1], selection.first_indices, selection.second_indices)
        if self._is_dense_boundary():
            columns = torch.arange(layout.column_count, dtype=torch.long)
        else:
            columns = torch.cat((layout.first_columns, layout.second_columns)).sort().values
        device_columns = columns.to(device=gradient.device)
        values = gradient.detach().index_select(1, device_columns)
        compressed = CompressedColumnGradient(parameter_id, columns, values)
        self._pending[parameter_id] = compressed
        captured_bytes = values.numel() * values.element_size()
        self._pending_bytes += captured_bytes
        self._metrics.increment("hybrid_shadow_captured_parameter_count")
        self._metrics.increment("hybrid_shadow_captured_element_count", values.numel())
        self._metrics.set_gauge("hybrid_shadow_pending_bytes", self._pending_bytes)
        self._update_peak_gauge("hybrid_shadow_bucket_peak_bytes", self._pending_bytes)
        if self._pending_bytes >= self._bucket_bytes:
            self._flush_bucket()

    def validate_native_gradient(self, parameter, group_id: int) -> None:
        if self._closed:
            return
        parameter_id = self._adapter.get_parameter_id(parameter)
        columns = self._columns_for(parameter_id)
        if columns is None:
            return
        gradient = self._adapter.get_gradient_tensor(parameter)
        partition_layout = self._partition_layouts.get(parameter_id)
        if gradient is None or partition_layout is None:
            return
        rank = self._adapter.get_partition_rank(group_id)
        owner_ranks, _ = partition_layout.map_columns(columns)
        owner_mask = owner_ranks == rank
        columns = columns.to(device=gradient.device)
        selected_native = gradient.detach().index_select(1, columns).reshape(-1)
        owner_mask = owner_mask.to(device=selected_native.device)
        self._native_values[parameter_id] = selected_native[owner_mask].cpu()
        self._try_validate_parity(parameter_id)

    def validate_owner_partition(self, parameter, group_id: int) -> None:
        if self._closed:
            return
        parameter_id = self._adapter.get_parameter_id(parameter)
        columns = self._columns_for(parameter_id)
        partition_layout = self._partition_layouts.get(parameter_id)
        if columns is None or partition_layout is None:
            return
        rank = self._adapter.get_partition_rank(group_id)
        owner_ranks, local_offsets = partition_layout.map_columns(columns)
        offsets = local_offsets[owner_ranks == rank]
        if offsets.numel() == 0:
            self._owner_complete.add(parameter_id)
            self._release_if_complete(parameter_id)
            return
        self._adapter.synchronize_gradient_transfers()
        native_partition = self._adapter.get_fp32_gradient_partition(group_id)
        device_offsets = offsets.to(device=native_partition.device)
        self._native_owner_offsets[parameter_id] = offsets
        self._native_owner_values[parameter_id] = native_partition.detach().index_select(0, device_offsets).clone()
        self._try_validate_owner(parameter_id)

    def reduce(self) -> None:
        if self._closed:
            return
        self._flush_bucket()
        for parameter_id in tuple(self._reduced):
            self._try_validate_parity(parameter_id)
            self._try_validate_owner(parameter_id)
        self._finish_norm_and_overflow_checks()
        self._clear_backward_state()

    def complete_step(self) -> None:
        if not self._closed:
            self._step += 1

    def close(self) -> None:
        self._clear_backward_state()
        self._closed = True

    def _flush_bucket(self) -> None:
        if not self._pending:
            return
        reduced = self._reducer.reduce(self._pending.values())
        communicated_elements = sum(gradient.values.numel() for gradient in reduced.values())
        communicated_bytes = sum(gradient.values.numel() * gradient.values.element_size()
                                 for gradient in reduced.values())
        estimated_live_bytes = self._pending_bytes + communicated_bytes
        self._update_peak_gauge("hybrid_shadow_estimated_peak_live_bytes", estimated_live_bytes)
        owned_reduced = {}
        for parameter_id, gradient in reduced.items():
            partition_layout = self._partition_layouts[parameter_id]
            rank = self._adapter.get_partition_rank(partition_layout.group_id)
            owner_ranks, local_offsets = partition_layout.map_columns(gradient.columns)
            owner_mask = owner_ranks == rank
            device_mask = owner_mask.to(device=gradient.values.device)
            owned_values = gradient.values.reshape(-1)[device_mask].cpu()
            owned_reduced[parameter_id] = _OwnedReducedGradient(columns=gradient.columns,
                                                                local_offsets=local_offsets[owner_mask],
                                                                values=owned_values)
        self._reduced.update(owned_reduced)
        self._pending.clear()
        self._pending_bytes = 0
        self._metrics.set_gauge("hybrid_shadow_pending_bytes", 0)
        boundary_kind = "dense" if self._is_dense_boundary() else "selected"
        self._metrics.increment("hybrid_shadow_collective_count")
        self._metrics.increment(f"hybrid_shadow_{boundary_kind}_collective_count")
        self._metrics.increment("hybrid_shadow_communicated_element_count", communicated_elements)
        self._metrics.increment("hybrid_shadow_communicated_bytes", communicated_bytes)
        self._metrics.increment(f"hybrid_shadow_{boundary_kind}_communicated_bytes", communicated_bytes)
        self._metrics.increment("hybrid_shadow_bucket_count")
        self._update_peak_gauge("hybrid_shadow_reduced_cpu_resident_peak_bytes", self._resident_reduced_bytes())
        for parameter_id in reduced:
            self._try_validate_parity(parameter_id)
            self._try_validate_owner(parameter_id)

    def _try_validate_parity(self, parameter_id: int) -> None:
        if parameter_id in self._parity_complete:
            return
        compressed = self._reduced.get(parameter_id)
        native = self._native_values.get(parameter_id)
        if compressed is None or native is None:
            return
        values = compressed.values
        native = native.to(device=values.device, dtype=values.dtype)
        difference = (values - native).detach().float().abs()
        max_abs_difference = difference.max().item() if difference.numel() else 0.0
        mismatch = not torch.allclose(values, native, atol=self._parity_atol, rtol=self._parity_rtol, equal_nan=True)
        self._metrics.increment("hybrid_shadow_parity_check_count")
        self._metrics.observe("hybrid_shadow_parity_max_abs", max_abs_difference)
        if mismatch:
            self._metrics.increment("hybrid_shadow_parity_mismatch_count")
            raise RuntimeError(
                f"Compressed collective mismatch for parameter {parameter_id}: max_abs={max_abs_difference}")

        compressed_float = values.detach().float()
        native_float = native.detach().float()
        self._compressed_overflow |= not torch.isfinite(compressed_float).all().item()
        self._native_overflow |= not torch.isfinite(native_float).all().item()
        self._compressed_norm_squared += compressed_float.square().sum().item()
        self._native_norm_squared += native_float.square().sum().item()
        self._parity_complete.add(parameter_id)
        self._native_values.pop(parameter_id, None)
        self._release_if_complete(parameter_id)

    def _try_validate_owner(self, parameter_id: int) -> None:
        if parameter_id in self._owner_complete:
            return
        compressed = self._reduced.get(parameter_id)
        native_values = self._native_owner_values.get(parameter_id)
        offsets = self._native_owner_offsets.get(parameter_id)
        partition_layout = self._partition_layouts.get(parameter_id)
        if compressed is None or native_values is None or offsets is None or partition_layout is None:
            return
        expected_offsets = compressed.local_offsets
        expected_values = compressed.values
        if not torch.equal(offsets, expected_offsets):
            raise RuntimeError(f"Owner offset mismatch for parameter {parameter_id}")
        expected_values = expected_values.to(device=native_values.device, dtype=native_values.dtype)
        mismatch = not torch.allclose(
            expected_values, native_values, atol=self._parity_atol, rtol=self._parity_rtol, equal_nan=True)
        self._metrics.increment("hybrid_shadow_owner_write_check_count")
        self._metrics.increment("hybrid_shadow_owner_write_element_count", expected_values.numel())
        if mismatch:
            self._metrics.increment("hybrid_shadow_owner_write_mismatch_count")
            max_abs = (expected_values - native_values).detach().float().abs().max().item()
            raise RuntimeError(f"Owner partition mismatch for parameter {parameter_id}: max_abs={max_abs}")
        self._owner_complete.add(parameter_id)
        self._native_owner_values.pop(parameter_id, None)
        self._native_owner_offsets.pop(parameter_id, None)
        self._release_if_complete(parameter_id)

    def _finish_norm_and_overflow_checks(self) -> None:
        if not self._parity_complete:
            return
        compressed_norm = math.sqrt(self._compressed_norm_squared)
        native_norm = math.sqrt(self._native_norm_squared)
        norm_difference = abs(compressed_norm - native_norm)
        norm_tolerance = self._parity_atol + self._parity_rtol * abs(native_norm)
        self._metrics.increment("hybrid_shadow_norm_check_count")
        self._metrics.observe("hybrid_shadow_norm_abs_difference", norm_difference)
        self._metrics.increment("hybrid_shadow_overflow_check_count")
        self._metrics.set_gauge("hybrid_shadow_compressed_overflow", int(self._compressed_overflow))
        self._metrics.set_gauge("hybrid_shadow_native_overflow", int(self._native_overflow))
        if norm_difference > norm_tolerance:
            self._metrics.increment("hybrid_shadow_norm_mismatch_count")
            raise RuntimeError(
                f"Compressed norm mismatch: compressed={compressed_norm}, native={native_norm}, diff={norm_difference}"
            )
        if self._compressed_overflow != self._native_overflow:
            self._metrics.increment("hybrid_shadow_overflow_mismatch_count")
            raise RuntimeError("Compressed and native overflow decisions differ")

    def _release_if_complete(self, parameter_id: int) -> None:
        if parameter_id not in self._parity_complete or parameter_id not in self._owner_complete:
            return
        self._reduced.pop(parameter_id, None)

    def _columns_for(self, parameter_id: int):
        reduced = self._reduced.get(parameter_id)
        if reduced is not None:
            return reduced.columns
        pending = self._pending.get(parameter_id)
        return None if pending is None else pending.columns

    def _resident_reduced_bytes(self) -> int:
        return sum(item.values.numel() * item.values.element_size() for item in self._reduced.values())

    def _update_peak_gauge(self, name: str, value: int) -> None:
        if value > self._peaks.get(name, 0):
            self._peaks[name] = value
            self._metrics.set_gauge(name, value)

    def _clear_backward_state(self) -> None:
        self._pending.clear()
        self._reduced.clear()
        self._native_values.clear()
        self._native_owner_values.clear()
        self._native_owner_offsets.clear()
        self._parity_complete.clear()
        self._owner_complete.clear()
        self._pending_bytes = 0
        self._compressed_norm_squared = 0.0
        self._native_norm_squared = 0.0
        self._compressed_overflow = False
        self._native_overflow = False
        self._metrics.set_gauge("hybrid_shadow_pending_bytes", 0)

    def _is_dense_boundary(self) -> bool:
        return (self._step + 1) % self._update_interval == 0
