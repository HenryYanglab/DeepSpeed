# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Versioned owner-partition commits for ZeRO-2 hybrid updates."""

from dataclasses import dataclass
from typing import Dict, Iterable

import torch

from .layout import CompressedColumnGradient
from .partition import ParameterPartitionLayout


@dataclass(frozen=True)
class OwnerPartitionUpdate:
    parameter_id: int
    columns: torch.Tensor
    values: torch.Tensor
    version: int


class Zero2OwnerPartitionCommitter:
    """Write updated values only to their owner FP32 partitions and publish with ZeRO all-gather."""

    def __init__(self, adapter, layouts: Iterable[ParameterPartitionLayout]) -> None:
        self._adapter = adapter
        self._layouts = {layout.parameter_id: layout for layout in layouts}
        self._staged: Dict[int, OwnerPartitionUpdate] = {}
        self._staged_version = None
        self._committed_version = 0

    @property
    def committed_version(self) -> int:
        return self._committed_version

    def stage(self, update: OwnerPartitionUpdate) -> None:
        if update.parameter_id not in self._layouts:
            raise ValueError(f"Missing owner layout for parameter {update.parameter_id}")
        if update.version <= self._committed_version:
            raise RuntimeError("Cannot stage a stale owner-partition update")
        if self._staged_version is None:
            self._staged_version = update.version
        elif self._staged_version != update.version:
            raise RuntimeError("One owner commit batch cannot mix versions")
        if update.parameter_id in self._staged:
            raise RuntimeError(f"Duplicate owner update for parameter {update.parameter_id}")
        self._staged[update.parameter_id] = update

    @torch.no_grad()
    def commit(self, expected_version: int) -> int:
        if not self._staged:
            return 0
        if self._staged_version != expected_version:
            raise RuntimeError(
                f"Owner commit version mismatch: expected {expected_version}, staged {self._staged_version}")
        if expected_version != self._committed_version + 1:
            raise RuntimeError("Owner commits must be applied in order")
        written = 0
        touched_groups = set()
        for parameter_id in sorted(self._staged):
            update = self._staged[parameter_id]
            layout = self._layouts[parameter_id]
            compressed = CompressedColumnGradient(parameter_id, update.columns, update.values)
            owned = layout.map_gradient(compressed)
            rank = self._adapter.get_partition_rank(layout.group_id)
            offsets, values = owned.for_rank(rank)
            if offsets.numel():
                self._adapter.write_fp32_partition_values(layout.group_id, offsets, values)
                written += offsets.numel()
                touched_groups.add(layout.group_id)
        self._adapter.publish_fp32_partitions(touched_groups)
        self._staged.clear()
        self._staged_version = None
        self._committed_version = expected_version
        return written

    def discard_staged(self) -> None:
        self._staged.clear()
        self._staged_version = None

    def restore_committed_version(self, version: int) -> None:
        if version < 0 or self._staged:
            raise ValueError("Cannot restore owner version with invalid state")
        self._committed_version = version
