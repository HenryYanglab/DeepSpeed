# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Owner-local GPU optimizer for first-important takeover values."""

from typing import Iterable, Mapping, Optional

import torch

from .owner_collective import OwnerReducedGradients, OwnerReducedGradientSet, Zero2OwnerCollective
from .partition import ParameterPartitionLayout
from .selected_adam import SelectedColumnAdamW


class Zero2OwnerGpuUpdater:
    """Update owner-local A values on GPU and publish selected values to every rank."""

    def __init__(self, adapter, layouts: Iterable[ParameterPartitionLayout],
                 collectives: Mapping[int, Zero2OwnerCollective], lr: float, betas: tuple[float, float], eps: float,
                 weight_decay: float) -> None:
        self._adapter = adapter
        self._layouts = {layout.parameter_id: layout for layout in layouts}
        self._collectives = dict(collectives)
        self._optimizer = SelectedColumnAdamW(lr, betas, eps, weight_decay)

    def initialize_state(self, parameter_id: int, columns: torch.Tensor, step: int, exp_avg: torch.Tensor,
                         exp_avg_sq: torch.Tensor, device: torch.device) -> None:
        layout = self._layouts[parameter_id]
        rank = self._adapter.get_partition_rank(layout.group_id)
        values = layout.extract_local_values(self._adapter.get_fp32_partition(layout.group_id), columns,
                                             rank).to(device)
        moments = layout.extract_local_values(exp_avg, columns, rank).to(device)
        variances = layout.extract_local_values(exp_avg_sq, columns, rank).to(device)
        self._optimizer.initialize_state((parameter_id, "first_owner"), values, moments, variances, step)

    @torch.no_grad()
    def step(self,
             reduced_by_group: Mapping[int, OwnerReducedGradientSet],
             columns: Mapping[int, torch.Tensor],
             lr: Optional[float] = None) -> int:
        updated_elements = 0
        for group_id, reduced_set in sorted(reduced_by_group.items()):
            for reduced in getattr(reduced_set, "buckets", (reduced_set, )):
                updated_values = {}
                for parameter_id, gradient in reduced.values.items():
                    key = (parameter_id, "first_owner")
                    try:
                        values = self._optimizer.master_values(key)
                    except KeyError as error:
                        raise RuntimeError(
                            f"Owner GPU optimizer state is unavailable for parameter {parameter_id}") from error
                    updated = self._optimizer.step_values(key, values, gradient, lr=lr)
                    # The Hybrid master is canonical; rebuilding sparse flat offsets every step costs more than Adam.
                    updated_values[parameter_id] = updated
                    updated_elements += updated.numel()
                owner_values = OwnerReducedGradients(group_id=group_id,
                                                     rank=reduced.rank,
                                                     padded_numel=reduced.padded_numel,
                                                     values=updated_values,
                                                     parameter_ids=reduced.parameter_ids)
                group_columns = {
                    parameter_id: columns[parameter_id]
                    for parameter_id, layout in self._layouts.items()
                    if layout.group_id == group_id and parameter_id in reduced.parameter_ids
                }
                gathered = self._collectives[group_id].all_gather(group_columns, owner_values)
                for compressed in gathered:
                    self._adapter.scatter_parameter_columns(compressed.parameter_id, compressed.columns,
                                                            compressed.values)
        return updated_elements

    def state_dict(self):
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict, device: torch.device) -> None:
        states = state_dict.get("states", {})
        devices = {key: device for key in states}
        self._optimizer.load_state_dict(state_dict, devices)
