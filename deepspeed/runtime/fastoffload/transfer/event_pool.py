# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Reusable accelerator event bundles for asynchronous transfers."""

import threading
from dataclasses import dataclass
from typing import Any, List


@dataclass(frozen=True)
class TransferEventBundle:
    """Events required by one GPU staging and D2H operation."""

    staging_start: Any
    source_ready: Any
    copy_start: Any
    copy_complete: Any


class TransferEventPool:
    """Bound event creation to the configured maximum inflight task count."""

    def __init__(self, accelerator: Any, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be greater than zero")
        self._accelerator = accelerator
        self._capacity = capacity
        self._free: List[TransferEventBundle] = []
        self._created = 0
        self._in_use = 0
        self._high_watermark = 0
        self._closed = False
        self._lock = threading.Lock()

    @property
    def created(self) -> int:
        with self._lock:
            return self._created

    @property
    def high_watermark(self) -> int:
        with self._lock:
            return self._high_watermark

    def acquire(self) -> TransferEventBundle:
        with self._lock:
            if self._closed:
                raise RuntimeError("Transfer event pool is closed")
            if self._free:
                bundle = self._free.pop()
            elif self._created < self._capacity:
                bundle = self._create_bundle()
                self._created += 1
            else:
                raise RuntimeError("Transfer event pool is exhausted")
            self._in_use += 1
            self._high_watermark = max(self._high_watermark, self._in_use)
            return bundle

    def release(self, bundle: TransferEventBundle) -> None:
        with self._lock:
            if self._in_use < 1:
                raise RuntimeError("Transfer event bundle is not leased")
            self._free.append(bundle)
            self._in_use -= 1

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._in_use:
                raise RuntimeError("Cannot close transfer event pool with active bundles")
            self._free.clear()
            self._closed = True

    def _create_bundle(self) -> TransferEventBundle:
        return TransferEventBundle(staging_start=self._accelerator.Event(enable_timing=True),
                                   source_ready=self._accelerator.Event(enable_timing=True),
                                   copy_start=self._accelerator.Event(enable_timing=True),
                                   copy_complete=self._accelerator.Event(enable_timing=True))
