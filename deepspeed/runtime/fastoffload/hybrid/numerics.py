# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Overflow, norm, unscale, and clipping for owner-local compressed gradients."""

from dataclasses import dataclass
from typing import Iterable

import torch

import deepspeed.comm as dist


@dataclass(frozen=True)
class HybridNumericsResult:
    overflow: bool
    global_norm: float
    combined_scale: float


class HybridGradientNumerics:
    """Apply native-style loss unscale and global norm clipping to sparse owner gradients."""

    @staticmethod
    @torch.no_grad()
    def unscale_and_clip(gradients: Iterable[torch.Tensor],
                         loss_scale: float,
                         clip_grad: float,
                         process_group=None) -> HybridNumericsResult:
        tensors = tuple(gradients)
        if loss_scale <= 0.0:
            raise ValueError("loss_scale must be positive")
        if clip_grad < 0.0:
            raise ValueError("clip_grad must not be negative")
        if not tensors:
            return HybridNumericsResult(False, 0.0, loss_scale)
        device = tensors[0].device
        norm_squared = torch.zeros(1, dtype=torch.float32, device=device)
        overflow = torch.zeros(1, dtype=torch.int32, device=device)
        for gradient in tensors:
            if gradient.device != device:
                raise ValueError("Hybrid gradients must use one device for numerics")
            finite = torch.isfinite(gradient).all()
            overflow.copy_(torch.maximum(overflow, (~finite).to(dtype=overflow.dtype).view(1)))
            norm_squared.add_(gradient.detach().float().square().sum())
        if dist.is_initialized():
            dist.all_reduce(norm_squared, op=dist.ReduceOp.SUM, group=process_group)
            dist.all_reduce(overflow, op=dist.ReduceOp.MAX, group=process_group)
        scaled_norm = norm_squared.sqrt().item()
        global_norm = scaled_norm / loss_scale
        clip_factor = 1.0
        if clip_grad > 0.0 and global_norm > clip_grad:
            clip_factor = global_norm / clip_grad
        combined_scale = loss_scale * clip_factor
        if not overflow.item():
            inverse_scale = 1.0 / combined_scale
            for gradient in tensors:
                gradient.mul_(inverse_scale)
        return HybridNumericsResult(bool(overflow.item()), global_norm, combined_scale)
