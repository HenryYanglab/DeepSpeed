# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Interfaces for pluggable parameter-importance algorithms."""

from abc import ABC, abstractmethod

import torch

from .context import ColumnImportanceSelection, ParameterImportanceView


class ImportanceAlgorithm(ABC):
    """Select two column bands without depending on DeepSpeed internals."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def select(self, view: ParameterImportanceView, reference: torch.Tensor,
               current: torch.Tensor) -> ColumnImportanceSelection:
        pass
