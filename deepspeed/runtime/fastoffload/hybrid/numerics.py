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
                         process_group=None,
                         communication_device=None) -> HybridNumericsResult:
        tensors = tuple(gradients)
        if loss_scale <= 0.0:
            raise ValueError("loss_scale must be positive")
        if clip_grad < 0.0:
            raise ValueError("clip_grad must not be negative")
        if not tensors:
            return HybridNumericsResult(False, 0.0, loss_scale)
        device = torch.device(communication_device) if communication_device is not None else tensors[0].device
        statistics = {}
        for gradient in tensors:
            if communication_device is None and gradient.device != device:
                raise ValueError("Hybrid gradients must use one device for numerics")
            if gradient.device not in statistics:
                statistics[gradient.device] = (torch.zeros(1, dtype=torch.float32, device=gradient.device),
                                               torch.zeros(1, dtype=torch.int32, device=gradient.device))
            local_norm, local_overflow = statistics[gradient.device]
            finite = torch.isfinite(gradient).all()
            local_overflow.copy_(torch.maximum(local_overflow, (~finite).to(dtype=local_overflow.dtype).view(1)))
            local_norm.add_(gradient.detach().float().square().sum())
        norm_squared = torch.zeros(1, dtype=torch.float32, device=device)
        overflow = torch.zeros(1, dtype=torch.int32, device=device)
        for local_norm, local_overflow in statistics.values():
            # Only scalar statistics cross devices; CPU B gradients and accumulators never return to the GPU.
            norm_squared.add_(local_norm.to(device=device))
            overflow.copy_(torch.maximum(overflow, local_overflow.to(device=device)))
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
