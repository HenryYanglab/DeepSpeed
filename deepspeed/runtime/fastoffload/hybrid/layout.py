# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Column layouts used to pack and scatter hybrid gradients and values."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CompressedColumnGradient:
    parameter_id: int
    columns: torch.Tensor
    values: torch.Tensor


class HybridColumnLayout:
    """Map dense parameter columns into disjoint A, B, and C groups."""

    def __init__(self, parameter_id: int, column_count: int, first_columns: torch.Tensor,
                 second_columns: torch.Tensor) -> None:
        if column_count < 1:
            raise ValueError("column_count must be greater than zero")
        first = first_columns.detach().cpu().long().unique(sorted=True)
        second = second_columns.detach().cpu().long().unique(sorted=True)
        if first.numel() and (first[0] < 0 or first[-1] >= column_count):
            raise ValueError("First-important column is out of range")
        if second.numel() and (second[0] < 0 or second[-1] >= column_count):
            raise ValueError("Second-important column is out of range")
        if torch.isin(first, second).any():
            raise ValueError("First- and second-important columns must not overlap")
        selected = torch.cat((first, second))
        dense_mask = torch.ones(column_count, dtype=torch.bool)
        dense_mask[selected] = False
        self.parameter_id = parameter_id
        self.column_count = column_count
        self.first_columns = first
        self.second_columns = second
        self.dense_columns = torch.arange(column_count, dtype=torch.long)[dense_mask]

    def pack_first(self, dense_gradient: torch.Tensor) -> CompressedColumnGradient:
        return self._pack(dense_gradient, self.first_columns)

    def pack_second(self, dense_gradient: torch.Tensor) -> CompressedColumnGradient:
        return self._pack(dense_gradient, self.second_columns)

    def pack_dense_remainder(self, dense_gradient: torch.Tensor) -> CompressedColumnGradient:
        return self._pack(dense_gradient, self.dense_columns)

    @torch.no_grad()
    def scatter_values(self, parameter: torch.Tensor, compressed: CompressedColumnGradient) -> None:
        self._validate_matrix(parameter)
        if compressed.parameter_id != self.parameter_id:
            raise ValueError("Compressed values belong to another parameter")
        columns = compressed.columns.to(device=parameter.device)
        values = compressed.values.to(device=parameter.device, dtype=parameter.dtype)
        parameter.index_copy_(1, columns, values)

    def _pack(self, dense_gradient: torch.Tensor, columns: torch.Tensor) -> CompressedColumnGradient:
        self._validate_matrix(dense_gradient)
        device_columns = columns.to(device=dense_gradient.device)
        values = dense_gradient.index_select(1, device_columns)
        return CompressedColumnGradient(parameter_id=self.parameter_id, columns=columns, values=values)

    def _validate_matrix(self, tensor: torch.Tensor) -> None:
        if tensor.dim() != 2 or tensor.shape[1] != self.column_count:
            raise ValueError("Tensor does not match hybrid column layout")
