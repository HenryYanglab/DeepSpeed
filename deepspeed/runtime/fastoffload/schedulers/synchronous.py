# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Single-task scheduler for the synchronous offload baseline."""

from deepspeed.runtime.fastoffload.actions import OffloadAction
from deepspeed.runtime.fastoffload.policies.all_offload import AllOffloadPolicy
from deepspeed.runtime.fastoffload.transfer.context import GradientTransferView
from deepspeed.runtime.fastoffload.transfer.synchronous import SynchronousTransferEngine


class SynchronousScheduler:
    """Execute policy, transfer, and inline consumption before returning."""

    requires_stable_source = False

    def __init__(self, policy: AllOffloadPolicy, transfer_engine: SynchronousTransferEngine) -> None:
        self._policy = policy
        self._transfer_engine = transfer_engine

    def submit(self, transfer_view: GradientTransferView) -> bool:
        decision = self._policy.decide(transfer_view)
        if decision.action == OffloadAction.keep_native:
            return False
        if decision.action != OffloadAction.offload_cpu:
            raise RuntimeError(f"Unsupported synchronous offload action: {decision.action}")
        self._transfer_engine.transfer(transfer_view)
        return True

    def progress(self) -> None:
        pass

    def prepare_step(self) -> None:
        pass

    def close(self) -> None:
        self._transfer_engine.close()
