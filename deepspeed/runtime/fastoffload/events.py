# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Lifecycle events emitted by FastOffload integration points."""

from enum import Enum


class FastOffloadEvent(str, Enum):
    """Stable lifecycle events understood by FastOffload controllers."""

    backward_begin = "backward_begin"
    gradient_ready = "gradient_ready"
    gradient_reduced = "gradient_reduced"
    backward_end = "backward_end"
    step_begin = "step_begin"
    step_end = "step_end"
    report = "report"
    shutdown = "shutdown"
