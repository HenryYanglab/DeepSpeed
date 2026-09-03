# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""All-gradient policy used by the synchronous baseline."""

from deepspeed.runtime.fastoffload.actions import OffloadAction, OffloadDecision
from deepspeed.runtime.fastoffload.transfer.context import GradientTransferView


class AllOffloadPolicy:
    """Select every reduced local gradient shard for CPU offload."""

    def decide(self, transfer_view: GradientTransferView) -> OffloadDecision:
        return OffloadDecision(action=OffloadAction.offload_cpu)
