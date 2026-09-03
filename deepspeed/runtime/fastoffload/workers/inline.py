# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Inline gradient consumer for the synchronous offload baseline."""

import time

from deepspeed.runtime.fastoffload.telemetry.metrics import MetricsRegistry
from deepspeed.runtime.fastoffload.transfer.buffer_pool import PinnedBufferLease
from deepspeed.runtime.fastoffload.transfer.context import GradientTransferView


class InlineWorker:
    """Copy a completed staging buffer into ZeRO's CPU gradient partition."""

    def __init__(self, metrics: MetricsRegistry) -> None:
        self._metrics = metrics

    def consume(self, lease: PinnedBufferLease, transfer_view: GradientTransferView) -> None:
        start_ns = time.perf_counter_ns()
        lease.mark_in_use()
        transfer_view.destination.copy_(lease.tensor, non_blocking=False)
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        self._metrics.observe("inline_worker_ms", elapsed_ms)
