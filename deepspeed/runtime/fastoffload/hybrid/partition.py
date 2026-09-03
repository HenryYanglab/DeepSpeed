# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Mapping between compressed matrix columns and ZeRO-2 flat partition owners."""

from dataclasses import dataclass
from typing import Tuple

import torch

from .layout import CompressedColumnGradient


@dataclass(frozen=True)
class OwnedColumnGradient:
    parameter_id: int
    group_id: int
    owner_ranks: torch.Tensor
    local_offsets: torch.Tensor
    values: torch.Tensor

    def for_rank(self, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = self.owner_ranks == rank
        return self.local_offsets[mask], self.values[mask]

    @torch.no_grad()
    def scatter_to_partition(self, partition: torch.Tensor, rank: int, accumulate: bool = False) -> int:
        if partition.dim() != 1:
            raise ValueError("ZeRO destination partition must be flat")
        offsets, values = self.for_rank(rank)
        if offsets.numel() == 0:
            return 0
        if offsets.max() >= partition.numel():
            raise ValueError("Owned gradient offset exceeds the ZeRO partition")
        offsets = offsets.to(device=partition.device)
        values = values.to(device=partition.device, dtype=partition.dtype)
        if accumulate:
            partition.index_add_(0, offsets, values)
        else:
            partition.index_copy_(0, offsets, values)
        return offsets.numel()


@dataclass(frozen=True)
class ParameterPartitionLayout:
    """Location of one row-major parameter in a flattened ZeRO parameter group."""

    parameter_id: int
    group_id: int
    shape: Tuple[int, int]
    group_offset: int
    partition_size: int
    world_size: int

    def __post_init__(self) -> None:
        rows, columns = self.shape
        if rows < 1 or columns < 1:
            raise ValueError("Hybrid partition layout requires a non-empty matrix")
        if self.group_offset < 0:
            raise ValueError("group_offset must not be negative")
        if self.partition_size < 1 or self.world_size < 1:
            raise ValueError("partition_size and world_size must be positive")
        last_offset = self.group_offset + rows * columns - 1
        if last_offset >= self.partition_size * self.world_size:
            raise ValueError("Parameter exceeds the padded ZeRO partition range")

    def owner_counts(self, columns: torch.Tensor) -> Tuple[int, ...]:
        columns = columns.detach().cpu().long()
        if columns.numel() and not torch.equal(columns, columns.sort().values):
            raise ValueError("Compressed columns must be sorted for compact owner mapping")
        _, column_count = self.shape
        parameter_numel = self.shape[0] * column_count

        def selected_before(parameter_offset: int) -> int:
            parameter_offset = min(max(parameter_offset, 0), parameter_numel)
            full_rows, partial_column = divmod(parameter_offset, column_count)
            partial_count = torch.searchsorted(columns, partial_column).item()
            return full_rows * columns.numel() + partial_count

        counts = []
        for owner in range(self.world_size):
            owner_start = owner * self.partition_size - self.group_offset
            owner_end = (owner + 1) * self.partition_size - self.group_offset
            counts.append(selected_before(owner_end) - selected_before(owner_start))
        if sum(counts) != self.shape[0] * columns.numel():
            raise RuntimeError("Compact owner mapping did not cover every selected value")
        return tuple(counts)

    def extract_fragment_values(self, fragment: torch.Tensor, columns: torch.Tensor, rank: int) -> torch.Tensor:
        columns = columns.detach().to(device=fragment.device, dtype=torch.long)
        _, column_count = self.shape
        parameter_numel = self.shape[0] * column_count
        parameter_start = max(0, rank * self.partition_size - self.group_offset)
        parameter_end = min(parameter_numel, (rank + 1) * self.partition_size - self.group_offset)
        if fragment.numel() != max(0, parameter_end - parameter_start):
            raise ValueError("Owner parameter fragment has an invalid size")
        if parameter_start >= parameter_end or columns.numel() == 0:
            return torch.empty(0, dtype=fragment.dtype, device=fragment.device)
        chunks = []
        first_full_row = (parameter_start + column_count - 1) // column_count
        last_full_row = parameter_end // column_count
        if parameter_start % column_count:
            valid_columns = columns[columns >= parameter_start % column_count]
            if parameter_start // column_count == (parameter_end - 1) // column_count:
                valid_columns = valid_columns[valid_columns < (parameter_end - 1) % column_count + 1]
            chunks.append(fragment.index_select(0, valid_columns - parameter_start % column_count))
        if last_full_row > first_full_row:
            fragment_start = first_full_row * column_count - parameter_start
            rows = fragment.narrow(0, fragment_start, (last_full_row - first_full_row) * column_count)
            chunks.append(rows.view(-1, column_count).index_select(1, columns).reshape(-1))
        if parameter_end % column_count and (parameter_end - 1) // column_count >= first_full_row:
            valid_columns = columns[columns < parameter_end % column_count]
            row_start = (parameter_end - 1) // column_count * column_count - parameter_start
            chunks.append(fragment.index_select(0, row_start + valid_columns))
        if not chunks:
            return torch.empty(0, dtype=fragment.dtype, device=fragment.device)
        return torch.cat(chunks) if len(chunks) > 1 else chunks[0]

    def extract_local_values(self, partition: torch.Tensor, columns: torch.Tensor, rank: int) -> torch.Tensor:
        if partition.dim() != 1 or partition.numel() != self.partition_size:
            raise ValueError("Owner partition tensor has an invalid shape")
        columns = columns.detach().to(device=partition.device, dtype=torch.long)
        _, column_count = self.shape
        parameter_numel = self.shape[0] * column_count
        parameter_start = max(0, rank * self.partition_size - self.group_offset)
        parameter_end = min(parameter_numel, (rank + 1) * self.partition_size - self.group_offset)
        if parameter_start >= parameter_end or columns.numel() == 0:
            return torch.empty(0, dtype=partition.dtype, device=partition.device)
        chunks = []
        first_full_row = (parameter_start + column_count - 1) // column_count
        last_full_row = parameter_end // column_count

        if parameter_start % column_count:
            row = parameter_start // column_count
            valid_columns = columns[columns >= parameter_start % column_count]
            if row == (parameter_end - 1) // column_count:
                valid_columns = valid_columns[valid_columns < (parameter_end - 1) % column_count + 1]
            dense_offsets = row * column_count + valid_columns + self.group_offset - rank * self.partition_size
            chunks.append(partition.index_select(0, dense_offsets))

        if last_full_row > first_full_row:
            dense_start = first_full_row * column_count + self.group_offset - rank * self.partition_size
            dense_rows = partition.narrow(0, dense_start, (last_full_row - first_full_row) * column_count)
            chunks.append(dense_rows.view(-1, column_count).index_select(1, columns).reshape(-1))

        if parameter_end % column_count and (parameter_end - 1) // column_count >= first_full_row:
            row = (parameter_end - 1) // column_count
            valid_columns = columns[columns < parameter_end % column_count]
            dense_offsets = row * column_count + valid_columns + self.group_offset - rank * self.partition_size
            chunks.append(partition.index_select(0, dense_offsets))

        if not chunks:
            return torch.empty(0, dtype=partition.dtype, device=partition.device)
        return torch.cat(chunks) if len(chunks) > 1 else chunks[0]

    def local_offsets(self, columns: torch.Tensor, rank: int) -> torch.Tensor:
        if rank < 0 or rank >= self.world_size:
            raise ValueError("Owner rank is outside the partition world")
        columns = columns.detach().cpu().long()
        _, column_count = self.shape
        parameter_numel = self.shape[0] * column_count
        parameter_start = max(0, rank * self.partition_size - self.group_offset)
        parameter_end = min(parameter_numel, (rank + 1) * self.partition_size - self.group_offset)
        if parameter_start >= parameter_end or columns.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        first_row = parameter_start // column_count
        last_row = (parameter_end - 1) // column_count
        row_offsets = torch.arange(first_row, last_row + 1, dtype=torch.long).unsqueeze(1) * column_count
        parameter_offsets = (row_offsets + columns.unsqueeze(0)).reshape(-1)
        in_partition = (parameter_offsets >= parameter_start) & (parameter_offsets < parameter_end)
        group_offsets = parameter_offsets[in_partition] + self.group_offset
        return group_offsets - rank * self.partition_size

    def map_columns(self, columns: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rows, column_count = self.shape
        columns = columns.detach().cpu().long()
        if columns.numel() and (columns.min() < 0 or columns.max() >= column_count):
            raise ValueError("Compressed gradient column is out of range")
        row_offsets = torch.arange(rows, dtype=torch.long).unsqueeze(1) * column_count
        parameter_offsets = row_offsets + columns.unsqueeze(0)
        group_offsets = parameter_offsets.reshape(-1) + self.group_offset
        owners = torch.div(group_offsets, self.partition_size, rounding_mode="floor")
        local_offsets = torch.remainder(group_offsets, self.partition_size)
        return owners, local_offsets

    def map_gradient(self, gradient: CompressedColumnGradient) -> OwnedColumnGradient:
        if gradient.parameter_id != self.parameter_id:
            raise ValueError("Compressed gradient belongs to another parameter")
        rows, _ = self.shape
        columns = gradient.columns.detach().cpu().long()
        expected_shape = (rows, columns.numel())
        if tuple(gradient.values.shape) != expected_shape:
            raise ValueError(f"Compressed gradient values must have shape {expected_shape}")
        owners, local_offsets = self.map_columns(columns)
        values = gradient.values.reshape(-1)
        return OwnedColumnGradient(parameter_id=self.parameter_id,
                                   group_id=self.group_id,
                                   owner_ranks=owners,
                                   local_offsets=local_offsets,
                                   values=values)
