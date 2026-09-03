# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Tensor views and immutable results used by importance selection."""

from dataclasses import dataclass
from typing import Any, Tuple

import torch


@dataclass(frozen=True)
class ParameterImportanceView:
    """A named model parameter exposed through an optimizer adapter."""

    parameter_id: int
    parameter_name: str
    tensor: torch.Tensor

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.tensor.shape)


@dataclass(frozen=True)
class ColumnImportanceSelection:
    """Two non-overlapping column-importance bands for one parameter."""

    parameter_id: int
    parameter_name: str
    shape: Tuple[int, ...]
    first_indices: torch.Tensor
    second_indices: torch.Tensor
    first_min_score: float
    second_min_score: float
    algorithm: str

    @property
    def selected_columns(self) -> int:
        return self.first_indices.numel() + self.second_indices.numel()

    def state_dict(self) -> dict[str, Any]:
        return {
            "parameter_id": self.parameter_id,
            "parameter_name": self.parameter_name,
            "shape": self.shape,
            "first_indices": self.first_indices,
            "second_indices": self.second_indices,
            "first_min_score": self.first_min_score,
            "second_min_score": self.second_min_score,
            "algorithm": self.algorithm,
        }
