# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Abstract adapter boundary between DeepSpeed and FastOffload."""

from abc import ABC, abstractmethod
from typing import Any, Iterable

from deepspeed.runtime.fastoffload.context import GradientBucketContext, GradientContext, StepContext
from deepspeed.runtime.fastoffload.importance.context import ParameterImportanceView
from deepspeed.runtime.fastoffload.transfer.context import GradientTransferView


class ObserverAdapter(ABC):
    """Convert optimizer-private state into stable observer contexts."""

    @abstractmethod
    def create_backward_begin_context(self) -> StepContext:
        pass

    @abstractmethod
    def create_step_context(self) -> StepContext:
        pass

    @abstractmethod
    def create_gradient_context(self, parameter: Any, group_id: int) -> GradientContext:
        pass

    @abstractmethod
    def create_bucket_context(self, communication_dtype: Any, bucket: Any) -> GradientBucketContext:
        pass

    @abstractmethod
    def create_gradient_transfer_view(self, parameter: Any, stable_source: bool = False) -> GradientTransferView:
        pass

    def get_rank(self) -> int:
        raise NotImplementedError("Adapter does not expose its process rank")

    def iter_parameter_importance_views(self) -> Iterable[ParameterImportanceView]:
        raise NotImplementedError("Adapter does not expose parameters for importance selection")

    @abstractmethod
    def clear_gradient(self, parameter: Any) -> None:
        pass

    @abstractmethod
    def complete_step(self) -> None:
        pass
