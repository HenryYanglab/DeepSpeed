# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Structured decisions produced by FastOffload policies."""

from dataclasses import dataclass
from enum import Enum


class OffloadAction(str, Enum):
    """Actions supported by the synchronous offload baseline."""

    keep_native = "keep_native"
    offload_cpu = "offload_cpu"


@dataclass(frozen=True)
class OffloadDecision:
    """Describe how one reduced local gradient shard should be handled."""

    action: OffloadAction
    priority: int = 0
