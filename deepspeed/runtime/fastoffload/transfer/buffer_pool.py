# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Bounded reusable pinned-memory buffers for gradient transfers."""

import threading
from enum import Enum
from typing import Any, Callable, List, Optional

import torch

from deepspeed.accelerator import get_accelerator


class BufferState(str, Enum):
    """Lifecycle states for one pooled transfer buffer."""

    free = "free"
    reserved = "reserved"
    copy_in_flight = "copy_in_flight"
    filled = "filled"
    in_use = "in_use"


class _BufferSlot:

    def __init__(self) -> None:
        self.tensor: Optional[torch.Tensor] = None
        self.capacity_bytes = 0
        self.state = BufferState.free


class PinnedBufferLease:
    """Exclusive lease for a tensor view backed by one pool slot."""

    def __init__(self, pool: "PinnedBufferPool", slot_id: int, tensor: torch.Tensor) -> None:
        self._pool = pool
        self._slot_id = slot_id
        self.tensor = tensor
        self._released = False

    @property
    def state(self) -> BufferState:
        return self._pool._get_state(self._slot_id)

    def mark_copy_in_flight(self) -> None:
        self._pool._transition(self._slot_id, BufferState.reserved, BufferState.copy_in_flight)

    def mark_filled(self) -> None:
        self._pool._transition_from(self._slot_id, (BufferState.reserved, BufferState.copy_in_flight),
                                    BufferState.filled)

    def mark_in_use(self) -> None:
        self._pool._transition(self._slot_id, BufferState.filled, BufferState.in_use)

    def release(self) -> None:
        if self._released:
            raise RuntimeError("Pinned buffer lease has already been released")
        self._pool._release(self._slot_id)
        self._released = True


class PinnedBufferPool:
    """Lazily allocate at most ``buffer_count`` reusable pinned CPU buffers.

    A slot grows when an oversized shard is encountered and is then reused. This
    keeps allocation off the per-gradient hot path while bounding the number of
    page-locked allocations.
    """

    def __init__(self,
                 buffer_count: int,
                 buffer_size: int,
                 pin_memory_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 unpin_memory_fn: Optional[Callable[[torch.Tensor], Any]] = None) -> None:
        if buffer_count < 1:
            raise ValueError("buffer_count must be greater than zero")
        if buffer_size < 1:
            raise ValueError("buffer_size must be greater than zero")
        accelerator = get_accelerator()
        self._buffer_size = buffer_size
        self._pin_memory_fn = pin_memory_fn or accelerator.pin_memory
        self._unpin_memory_fn = unpin_memory_fn or accelerator.unpin_memory
        self._slots: List[_BufferSlot] = [_BufferSlot() for _ in range(buffer_count)]
        self._in_use = 0
        self._high_watermark = 0
        self._oversized_allocations = 0
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
    def oversized_allocations(self) -> int:
        with self._lock:
            return self._oversized_allocations

    @property
    def allocated_bytes(self) -> int:
        with self._lock:
            return self._allocated_bytes

    @property
    def peak_allocated_bytes(self) -> int:
        with self._lock:
            return self._peak_allocated_bytes

    def acquire(self, numel: int, dtype: torch.dtype) -> PinnedBufferLease:
        if numel < 1:
            raise ValueError("numel must be greater than zero")
        required_bytes = numel * torch.empty((), dtype=dtype).element_size()
        with self._lock:
            if self._closed:
                raise RuntimeError("Pinned buffer pool is closed")
            slot_id = self._find_free_slot()
            if slot_id is None:
                raise RuntimeError("Pinned buffer pool is exhausted")
            slot = self._slots[slot_id]
            if slot.tensor is None or slot.capacity_bytes < required_bytes or slot.tensor.dtype != dtype:
                self._resize_slot(slot, required_bytes, dtype)
            slot.state = BufferState.reserved
            self._in_use += 1
            self._high_watermark = max(self._high_watermark, self._in_use)
            tensor = slot.tensor.view(-1).narrow(0, 0, numel)
        return PinnedBufferLease(self, slot_id, tensor)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._in_use:
                raise RuntimeError("Cannot close pinned buffer pool with active leases")
            tensors = [slot.tensor for slot in self._slots if slot.tensor is not None]
            for slot in self._slots:
                slot.tensor = None
                slot.capacity_bytes = 0
            self._allocated_bytes = 0
            self._closed = True
        for tensor in tensors:
            self._unpin_memory_fn(tensor)

    def _find_free_slot(self) -> Optional[int]:
        for slot_id, slot in enumerate(self._slots):
            if slot.state == BufferState.free:
                return slot_id
        return None

    def _resize_slot(self, slot: _BufferSlot, required_bytes: int, dtype: torch.dtype) -> None:
        if slot.tensor is not None:
            self._unpin_memory_fn(slot.tensor)
        self._allocated_bytes -= slot.capacity_bytes
        capacity_bytes = max(self._buffer_size, required_bytes)
        element_size = torch.empty((), dtype=dtype).element_size()
        capacity_numel = (capacity_bytes + element_size - 1) // element_size
        tensor = torch.empty(capacity_numel, dtype=dtype, device="cpu")
        slot.tensor = self._pin_memory_fn(tensor)
        slot.capacity_bytes = capacity_numel * element_size
        self._allocated_bytes += slot.capacity_bytes
        self._peak_allocated_bytes = max(self._peak_allocated_bytes, self._allocated_bytes)
        if required_bytes > self._buffer_size:
            self._oversized_allocations += 1

    def _get_state(self, slot_id: int) -> BufferState:
        with self._lock:
            return self._slots[slot_id].state

    def _transition(self, slot_id: int, expected: BufferState, target: BufferState) -> None:
        with self._lock:
            slot = self._slots[slot_id]
            if slot.state != expected:
                raise RuntimeError(f"Invalid pinned buffer transition: {slot.state} -> {target}")
            slot.state = target

    def _transition_from(self, slot_id: int, expected: tuple, target: BufferState) -> None:
        with self._lock:
            slot = self._slots[slot_id]
            if slot.state not in expected:
                raise RuntimeError(f"Invalid pinned buffer transition: {slot.state} -> {target}")
            slot.state = target

    def _release(self, slot_id: int) -> None:
        with self._lock:
            slot = self._slots[slot_id]
            if slot.state not in (BufferState.reserved, BufferState.copy_in_flight, BufferState.filled,
                                  BufferState.in_use):
                raise RuntimeError("Pinned buffer is not leased")
            slot.state = BufferState.free
            self._in_use -= 1
