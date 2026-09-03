# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Immutable jobs and results for hybrid asynchronous updates."""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import torch


@dataclass(frozen=True)
class HybridUpdateJob:
    """Own gradients frozen at one dense update boundary."""

    version: int
    buffer_id: int
    interval_steps: int
    second_gradients: Mapping[int, torch.Tensor]
    dense_gradients: Mapping[int, torch.Tensor]
    transfer_event: Optional[Any] = None


@dataclass(frozen=True)
class HybridUpdateResult:
    """CPU worker output waiting for a training-thread commit."""

    version: int
    buffer_id: int
    updated_values: Mapping[int, torch.Tensor]


def clone_tensor_mapping(values: Mapping[int, torch.Tensor], device: torch.device) -> Dict[int, torch.Tensor]:
    return {parameter_id: tensor.detach().to(device=device, copy=True) for parameter_id, tensor in values.items()}
