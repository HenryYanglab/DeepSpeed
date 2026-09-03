# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Task records for asynchronous gradient transfers."""

from dataclasses import dataclass
from typing import Any, Optional

from deepspeed.runtime.fastoffload.transfer.buffer_pool import PinnedBufferLease
from deepspeed.runtime.fastoffload.transfer.context import GradientTransferView
from deepspeed.runtime.fastoffload.transfer.gpu_buffer_pool import GpuBufferLease


@dataclass
class AsyncTransferTask:
    """Own resources required until one asynchronous D2H copy is consumed."""

    task_id: int
    transfer_view: GradientTransferView
    lease: Optional[PinnedBufferLease]
    gpu_lease: Optional[GpuBufferLease]
    event_bundle: Any
    staging_start_event: Any
    source_ready_event: Any
    copy_start_event: Any
    copy_complete_event: Any
    submitted_ns: int

    @property
    def nbytes(self) -> int:
        return self.transfer_view.nbytes
