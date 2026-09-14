# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Gradient-accumulation lifecycle for compressed takeover gradients."""

from typing import Dict, Mapping

import torch


class CompressedMicrobatchAccumulator:
    """Accumulate compressed gradients across GAS while tolerating unused parameters."""

    def __init__(self, gradient_accumulation_steps: int, reduction: str = "mean") -> None:
        if gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if reduction not in ("mean", "sum"):
            raise ValueError("Compressed microbatch reduction must be mean or sum")
        self._gas = gradient_accumulation_steps
        self._reduction = reduction
        self._gradients: Dict[int, torch.Tensor] = {}
        self._micro_steps = 0

    @property
    def is_boundary(self) -> bool:
        return self._micro_steps == self._gas

    @property
    def micro_steps(self) -> int:
        return self._micro_steps

    def accumulate(self, gradients: Mapping[int, torch.Tensor], take_ownership: bool = False) -> bool:
        if self.is_boundary:
            raise RuntimeError("Compressed GAS boundary must be consumed before accumulating again")
        if self._gas == 1 and take_ownership:
            # The capture pipeline relinquishes these tensors immediately after this call.
            self._gradients = {parameter_id: gradient.detach() for parameter_id, gradient in gradients.items()}
            self._micro_steps = 1
            return True
        for parameter_id, gradient in gradients.items():
            source = gradient.detach()
            target = self._gradients.get(parameter_id)
            if target is None:
                target = torch.zeros_like(source)
                self._gradients[parameter_id] = target
            if target.shape != source.shape or target.device != source.device or target.dtype != source.dtype:
                raise ValueError(f"Compressed gradient layout changed for parameter {parameter_id}")
            target.add_(source)
        self._micro_steps += 1
        return self.is_boundary

    def take_boundary(self) -> Dict[int, torch.Tensor]:
        if not self.is_boundary:
            raise RuntimeError("Compressed gradients have not reached the GAS boundary")
        scale = 1.0 / self._gas if self._reduction == "mean" else 1.0
        result = self._gradients
        if scale != 1.0:
            for gradient in result.values():
                gradient.mul_(scale)
        self._gradients = {}
        self._micro_steps = 0
        return result

    def discard(self) -> None:
        self._gradients.clear()
        self._micro_steps = 0
