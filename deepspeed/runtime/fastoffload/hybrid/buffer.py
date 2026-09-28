# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Device-selectable double-buffered gradient accumulation."""

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Mapping, Optional

import torch

from deepspeed.accelerator import get_accelerator


class HybridBufferState(str, Enum):
    free = "free"
    accumulating = "accumulating"
    frozen = "frozen"
    copying_to_cpu = "copying_to_cpu"
    updating = "updating"
    copying_to_gpu = "copying_to_gpu"
    ready_to_commit = "ready_to_commit"


@dataclass
class _AccumulatorSlot:
    state: HybridBufferState = HybridBufferState.free
    gradients: Dict[int, torch.Tensor] = field(default_factory=dict)
    accumulated_steps: int = 0
    version: Optional[int] = None
    result: Optional[Mapping[int, torch.Tensor]] = None


class DoubleBufferedGradientAccumulator:
    """Prevent an asynchronous consumer from mutating the active accumulator."""

    def __init__(self, accumulation_device: str) -> None:
        if accumulation_device not in ("cpu", "gpu"):
            raise ValueError("accumulation_device must be cpu or gpu")
        if accumulation_device == "gpu" and not get_accelerator().is_available():
            raise ValueError("GPU accumulation requires an available accelerator")
        device_name = "cpu" if accumulation_device == "cpu" else get_accelerator().current_device_name()
        self._device = torch.device(device_name)
        self._slots = [_AccumulatorSlot(), _AccumulatorSlot()]
        self._active_id = 0
        self._slots[self._active_id].state = HybridBufferState.accumulating
        self._lock = threading.Lock()

    @property
    def accumulation_device(self) -> torch.device:
        return self._device

    @property
    def active_id(self) -> int:
        with self._lock:
            return self._active_id

    def state(self, buffer_id: int) -> HybridBufferState:
        with self._lock:
            return self._slot(buffer_id).state

    def accumulated_steps(self, buffer_id: int) -> int:
        with self._lock:
            return self._slot(buffer_id).accumulated_steps

    def accumulate(self, gradients: Mapping[int, torch.Tensor]) -> None:
        with self._lock:
            slot = self._slots[self._active_id]
            if slot.state != HybridBufferState.accumulating:
                raise RuntimeError("Active hybrid buffer is not accumulating")
            for parameter_id, gradient in gradients.items():
                source = gradient.detach().to(device=self._device)
                target = slot.gradients.get(parameter_id)
                if target is None:
                    target = torch.zeros_like(source, device=self._device)
                    slot.gradients[parameter_id] = target
                if target.shape != source.shape or target.dtype != source.dtype:
                    raise ValueError(f"Gradient layout changed for parameter {parameter_id}")
                target.add_(source)
            slot.accumulated_steps += 1

    def active_state_dict(self) -> dict:
        with self._lock:
            active = self._slots[self._active_id]
            if active.state != HybridBufferState.accumulating:
                raise RuntimeError("Active hybrid buffer is not accumulating")
            return {
                "accumulated_steps": active.accumulated_steps,
                "gradients": {
                    parameter_id: gradient.detach().cpu().clone()
                    for parameter_id, gradient in active.gradients.items()
                },
            }

    def load_active_state_dict(self, state_dict: dict) -> None:
        with self._lock:
            active = self._slots[self._active_id]
            # Reset slots retain zeroed allocations for reuse; those are not live accumulated gradients.
            if active.state != HybridBufferState.accumulating or active.accumulated_steps:
                raise RuntimeError("Active hybrid buffer must be empty before restore")
            steps = int(state_dict.get("accumulated_steps", 0))
            if steps < 0:
                raise ValueError("Accumulated step count must not be negative")
            gradients = state_dict.get("gradients", {})
            active.gradients = {
                int(parameter_id): gradient.detach().to(device=self._device).clone()
                for parameter_id, gradient in gradients.items()
            }
            active.accumulated_steps = steps

    def discard_active(self) -> None:
        with self._lock:
            active = self._slots[self._active_id]
            if active.state != HybridBufferState.accumulating:
                raise RuntimeError("Cannot discard a non-accumulating hybrid buffer")
            self._reset_slot(active)
            active.state = HybridBufferState.accumulating

    def freeze_and_swap(self, version: int) -> int:
        with self._lock:
            active = self._slots[self._active_id]
            if active.state != HybridBufferState.accumulating:
                raise RuntimeError("Active hybrid buffer cannot be frozen")
            free_id = self._find_free_slot()
            if free_id is None:
                raise RuntimeError("No free hybrid accumulation buffer; asynchronous update is overdue")
            frozen_id = self._active_id
            active.state = HybridBufferState.frozen
            active.version = version
            replacement = self._slots[free_id]
            self._reset_slot(replacement)
            replacement.state = HybridBufferState.accumulating
            self._active_id = free_id
            return frozen_id

    def begin_copy_to_cpu(self, buffer_id: int) -> None:
        self._transition(buffer_id, HybridBufferState.frozen, HybridBufferState.copying_to_cpu)

    def begin_update(self, buffer_id: int) -> None:
        with self._lock:
            slot = self._slot(buffer_id)
            expected = (HybridBufferState.frozen, HybridBufferState.copying_to_cpu)
            if slot.state not in expected:
                raise RuntimeError(f"Cannot update hybrid buffer in state {slot.state.value}")
            slot.state = HybridBufferState.updating

    def frozen_gradients(self, buffer_id: int) -> Mapping[int, torch.Tensor]:
        with self._lock:
            slot = self._slot(buffer_id)
            if slot.state not in (HybridBufferState.frozen, HybridBufferState.copying_to_cpu,
                                  HybridBufferState.updating):
                raise RuntimeError("Hybrid gradients are not frozen")
            return dict(slot.gradients)

    def release_update_buffer(self, buffer_id: int) -> None:
        with self._lock:
            slot = self._slot(buffer_id)
            if slot.state != HybridBufferState.updating:
                raise RuntimeError("Hybrid buffer is not updating")
            self._reset_slot(slot)

    def mark_copying_to_gpu(self, buffer_id: int, result: Mapping[int, torch.Tensor]) -> None:
        with self._lock:
            slot = self._slot(buffer_id)
            if slot.state != HybridBufferState.updating:
                raise RuntimeError("Hybrid buffer is not updating")
            slot.result = dict(result)
            slot.state = HybridBufferState.copying_to_gpu

    def mark_ready(self, buffer_id: int) -> None:
        self._transition(buffer_id, HybridBufferState.copying_to_gpu, HybridBufferState.ready_to_commit)

    def take_ready_result(self, buffer_id: int) -> Mapping[int, torch.Tensor]:
        with self._lock:
            slot = self._slot(buffer_id)
            if slot.state != HybridBufferState.ready_to_commit or slot.result is None:
                raise RuntimeError("Hybrid update result is not ready")
            result = slot.result
            self._reset_slot(slot)
            return result

    def version(self, buffer_id: int) -> int:
        with self._lock:
            version = self._slot(buffer_id).version
            if version is None:
                raise RuntimeError("Hybrid buffer does not own a version")
            return version

    def _transition(self, buffer_id: int, expected: HybridBufferState, target: HybridBufferState) -> None:
        with self._lock:
            slot = self._slot(buffer_id)
            if slot.state != expected:
                raise RuntimeError(f"Expected hybrid buffer state {expected.value}, found {slot.state.value}")
            slot.state = target

    def _find_free_slot(self) -> Optional[int]:
        for buffer_id, slot in enumerate(self._slots):
            if slot.state == HybridBufferState.free:
                return buffer_id
        return None

    def _slot(self, buffer_id: int) -> _AccumulatorSlot:
        if buffer_id < 0 or buffer_id >= len(self._slots):
            raise IndexError("Invalid hybrid buffer id")
        return self._slots[buffer_id]

    @staticmethod
    def _reset_slot(slot: _AccumulatorSlot) -> None:
        for gradient in slot.gradients.values():
            gradient.zero_()
        slot.accumulated_steps = 0
        slot.version = None
        slot.result = None
        slot.state = HybridBufferState.free
