# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Opt-in Linear backward that computes weight gradients for selected columns."""

from types import MethodType
from typing import Tuple

import torch
import torch.nn.functional as F

_SELECTED_COLUMNS_ATTRIBUTE = "_fastoffload_selected_columns"
_WRAPPED_ATTRIBUTE = "_fastoffload_selective_linear_wrapped"
_MIN_TOKENS_ATTRIBUTE = "_fastoffload_sparse_backward_min_tokens"
_MIN_OUTPUT_FEATURES_ATTRIBUTE = "_fastoffload_sparse_backward_min_output_features"
_SPARSE_FORWARD_COUNT_ATTRIBUTE = "_fastoffload_sparse_forward_count"
_NATIVE_FORWARD_COUNT_ATTRIBUTE = "_fastoffload_native_forward_count"
_TAKEOVER_CAPTURE_ATTRIBUTE = "_fastoffload_takeover_capture"
_DENSE_BACKWARD_ATTRIBUTE = "_fastoffload_dense_backward"
_PARAMETER_ID_ATTRIBUTE = "_fastoffload_parameter_id"


class _SelectiveLinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, selected_columns: torch.Tensor,
                parameter_id: int, capture_callback) -> torch.Tensor:
        ctx.save_for_backward(inputs, weight, selected_columns)
        ctx.has_bias = bias is not None
        ctx.parameter_id = parameter_id
        ctx.capture_callback = capture_callback
        return F.linear(inputs, weight, bias)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        inputs, weight, selected_columns = ctx.saved_tensors
        grad_inputs = grad_weight = grad_bias = None

        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_output_2d.matmul(weight).view_as(inputs)
        if ctx.needs_input_grad[1]:
            inputs_2d = inputs.reshape(-1, inputs.shape[-1])
            selected_inputs = inputs_2d.index_select(1, selected_columns)
            selected_grad_weight = grad_output_2d.transpose(0, 1).matmul(selected_inputs)
            if ctx.capture_callback is None:
                grad_weight = torch.zeros_like(weight)
                grad_weight.index_copy_(1, selected_columns, selected_grad_weight)
            else:
                ctx.capture_callback(ctx.parameter_id, selected_columns, selected_grad_weight)
        if ctx.has_bias and ctx.needs_input_grad[2] and ctx.capture_callback is None:
            grad_bias = grad_output_2d.sum(dim=0)
        return grad_inputs, grad_weight, grad_bias, None, None, None


def _selective_linear_forward(module: torch.nn.Linear, inputs: torch.Tensor) -> torch.Tensor:
    selected_columns = getattr(module.weight, _SELECTED_COLUMNS_ATTRIBUTE, None)
    dense_backward = getattr(module.weight, _DENSE_BACKWARD_ATTRIBUTE, False)
    token_count = inputs.numel() // module.in_features
    minimum_tokens = getattr(module, _MIN_TOKENS_ATTRIBUTE, 1024)
    minimum_output_features = getattr(module, _MIN_OUTPUT_FEATURES_ATTRIBUTE, 0)
    if (selected_columns is None or dense_backward or selected_columns.numel() == module.in_features
            or token_count < minimum_tokens or module.out_features < minimum_output_features):
        native_count = getattr(module, _NATIVE_FORWARD_COUNT_ATTRIBUTE, 0)
        setattr(module, _NATIVE_FORWARD_COUNT_ATTRIBUTE, native_count + 1)
        return F.linear(inputs, module.weight, module.bias)
    sparse_count = getattr(module, _SPARSE_FORWARD_COUNT_ATTRIBUTE, 0)
    setattr(module, _SPARSE_FORWARD_COUNT_ATTRIBUTE, sparse_count + 1)
    parameter_id = getattr(module.weight, _PARAMETER_ID_ATTRIBUTE, -1)
    capture_callback = getattr(module.weight, _TAKEOVER_CAPTURE_ATTRIBUTE, None)
    return _SelectiveLinearFunction.apply(inputs, module.weight, module.bias, selected_columns, parameter_id,
                                          capture_callback)


def enable_selective_linear(model: torch.nn.Module,
                            minimum_tokens: int = 1024,
                            minimum_output_features: int = 0) -> int:
    """Wrap Linear forwards without replacing parameters or module classes."""
    if minimum_tokens < 1:
        raise ValueError("minimum_tokens must be greater than zero")
    if minimum_output_features < 0:
        raise ValueError("minimum_output_features must not be negative")
    wrapped_modules = 0
    for module in model.modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if getattr(module, _WRAPPED_ATTRIBUTE, False):
            continue
        module.forward = MethodType(_selective_linear_forward, module)
        setattr(module, _MIN_TOKENS_ATTRIBUTE, minimum_tokens)
        setattr(module, _MIN_OUTPUT_FEATURES_ATTRIBUTE, minimum_output_features)
        setattr(module, _SPARSE_FORWARD_COUNT_ATTRIBUTE, 0)
        setattr(module, _NATIVE_FORWARD_COUNT_ATTRIBUTE, 0)
        setattr(module, _WRAPPED_ATTRIBUTE, True)
        wrapped_modules += 1
    return wrapped_modules


def selective_linear_stats(model: torch.nn.Module) -> dict[str, int]:
    """Collect process-local branch counts for benchmark diagnostics."""
    sparse_count = 0
    native_count = 0
    for module in model.modules():
        sparse_count += getattr(module, _SPARSE_FORWARD_COUNT_ATTRIBUTE, 0)
        native_count += getattr(module, _NATIVE_FORWARD_COUNT_ATTRIBUTE, 0)
    return {"sparse_forward_count": sparse_count, "native_forward_count": native_count}


def clear_takeover_capture(weight: torch.Tensor) -> None:
    """Remove takeover-only state while retaining standalone selected columns."""
    for attribute in (_PARAMETER_ID_ATTRIBUTE, _TAKEOVER_CAPTURE_ATTRIBUTE, _DENSE_BACKWARD_ATTRIBUTE):
        if hasattr(weight, attribute):
            delattr(weight, attribute)


def set_takeover_capture(weight: torch.Tensor, parameter_id: int, capture_callback) -> None:
    """Route packed selected gradients to a takeover runtime instead of materializing a dense gradient."""
    setattr(weight, _PARAMETER_ID_ATTRIBUTE, parameter_id)
    setattr(weight, _TAKEOVER_CAPTURE_ATTRIBUTE, capture_callback)


def set_dense_backward(weight: torch.Tensor, enabled: bool) -> None:
    """Select native dense backward for the graph created by the next forward."""
    setattr(weight, _DENSE_BACKWARD_ATTRIBUTE, enabled)


def set_selected_columns(weight: torch.Tensor, first_indices: torch.Tensor, second_indices: torch.Tensor) -> None:
    """Attach sorted device indices consumed by the next Linear forward."""
    selected_columns = torch.cat((first_indices, second_indices)).sort().values
    selected_columns = selected_columns.to(device=weight.device, dtype=torch.long)
    setattr(weight, _SELECTED_COLUMNS_ATTRIBUTE, selected_columns)
