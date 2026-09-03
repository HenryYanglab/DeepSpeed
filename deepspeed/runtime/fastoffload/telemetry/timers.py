# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Host-side timing helpers that never synchronize an accelerator."""

import time
from typing import Optional, Type

from .metrics import MetricsRegistry


class HostTimer:
    """Record one host-side duration in a MetricsRegistry histogram."""

    def __init__(self, registry: MetricsRegistry, metric_name: str, enabled: bool = True) -> None:
        self._registry = registry
        self._metric_name = metric_name
        self._enabled = enabled
        self._start_ns: Optional[int] = None

    def __enter__(self) -> "HostTimer":
        if self._enabled:
            if self._start_ns is not None:
                raise RuntimeError("HostTimer cannot be entered more than once")
            self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], traceback) -> bool:
        if self._enabled:
            if self._start_ns is None:
                raise RuntimeError("HostTimer was not started")
            elapsed_ns = time.perf_counter_ns() - self._start_ns
            self._registry.observe(self._metric_name, elapsed_ns / 1_000_000.0)
            self._start_ns = None
        return False
