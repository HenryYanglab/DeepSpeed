# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Bounded reusable GPU staging buffers for asynchronous D2H transfers."""

import threading
from typing import List, Optional

import torch


class _GpuBufferSlot:

    def __init__(self) -> None:
        self.tensor: Optional[torch.Tensor] = None
        self.capacity_bytes = 0
        self.in_use = False


class GpuBufferLease:
    """Exclusive tensor view leased from a GPU staging pool."""

    def __init__(self, pool: "GpuBufferPool", slot_id: int, tensor: torch.Tensor) -> None:
        self._pool = pool
        self._slot_id = slot_id
        self.tensor = tensor
        self._released = False

    def release(self) -> None:
        if self._released:
            raise RuntimeError("GPU buffer lease has already been released")
        self._pool._release(self._slot_id)
        self._released = True


class GpuBufferPool:
    """Lazily allocate and reuse a bounded number of GPU staging tensors."""

    def __init__(self, buffer_count: int, buffer_size: int) -> None:
        if buffer_count < 1:
            raise ValueError("buffer_count must be greater than zero")
        if buffer_size < 1:
            raise ValueError("buffer_size must be greater than zero")
        self._buffer_size = buffer_size
        self._slots: List[_GpuBufferSlot] = [_GpuBufferSlot() for _ in range(buffer_count)]
        self._in_use = 0
        self._high_watermark = 0
        self._allocated_bytes = 0
        self._peak_allocated_bytes = 0
        self._closed = False
        self._lock = threading.Lock()

    @property
    def in_use(self) -> int:
        with self._lock:
            return self._in_use

    @property
    def high_watermark(self) -> int:
        with self._lock:
            return self._high_watermark

    @property
    def allocated_bytes(self) -> int:
        with self._lock:
            return self._allocated_bytes

    @property
    def peak_allocated_bytes(self) -> int:
        with self._lock:
            return self._peak_allocated_bytes

    def acquire(self, numel: int, dtype: torch.dtype, device: torch.device) -> GpuBufferLease:
        if numel < 1:
            raise ValueError("numel must be greater than zero")
        required_bytes = numel * torch.empty((), dtype=dtype).element_size()
        with self._lock:
            if self._closed:
                raise RuntimeError("GPU buffer pool is closed")
            slot_id = self._find_free_slot()
            if slot_id is None:
                raise RuntimeError("GPU buffer pool is exhausted")
            slot = self._slots[slot_id]
            needs_resize = (slot.tensor is None or slot.capacity_bytes < required_bytes or slot.tensor.dtype != dtype
                            or slot.tensor.device != device)
            if needs_resize:
                self._resize_slot(slot, required_bytes, dtype, device)
            slot.in_use = True
            self._in_use += 1
            self._high_watermark = max(self._high_watermark, self._in_use)
            tensor = slot.tensor.view(-1).narrow(0, 0, numel)
        return GpuBufferLease(self, slot_id, tensor)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._in_use:
                raise RuntimeError("Cannot close GPU buffer pool with active leases")
            for slot in self._slots:
                slot.tensor = None
                slot.capacity_bytes = 0
            self._allocated_bytes = 0
            self._closed = True

    def _find_free_slot(self) -> Optional[int]:
        for slot_id, slot in enumerate(self._slots):
            if not slot.in_use:
                return slot_id
        return None

    def _resize_slot(self, slot: _GpuBufferSlot, required_bytes: int, dtype: torch.dtype,
                     device: torch.device) -> None:
        self._allocated_bytes -= slot.capacity_bytes
        capacity_bytes = max(self._buffer_size, required_bytes)
        element_size = torch.empty((), dtype=dtype).element_size()
        capacity_numel = (capacity_bytes + element_size - 1) // element_size
        slot.tensor = torch.empty(capacity_numel, dtype=dtype, device=device)
        slot.capacity_bytes = capacity_numel * element_size
        self._allocated_bytes += slot.capacity_bytes
        self._peak_allocated_bytes = max(self._peak_allocated_bytes, self._allocated_bytes)

    def _release(self, slot_id: int) -> None:
        with self._lock:
            slot = self._slots[slot_id]
            if not slot.in_use:
                raise RuntimeError("GPU buffer is not leased")
            slot.in_use = False
            self._in_use -= 1
