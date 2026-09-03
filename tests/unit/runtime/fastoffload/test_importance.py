# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import torch

from deepspeed.runtime.fastoffload.importance.context import ParameterImportanceView
from deepspeed.runtime.fastoffload.importance.pretrained_delta import PretrainedDeltaTopK
from deepspeed.runtime.fastoffload.importance.registry import ImportanceRegistry
from deepspeed.runtime.fastoffload.importance.selective_linear import (enable_selective_linear, set_dense_backward,
                                                                       set_selected_columns, set_takeover_capture)
from deepspeed.runtime.fastoffload.importance.selector import StreamingImportanceSelector
from deepspeed.runtime.fastoffload.telemetry.metrics import MetricsRegistry


def test_pretrained_delta_selects_two_column_bands_in_chunks():
    reference = torch.zeros((3, 4), dtype=torch.bfloat16)
    current = torch.tensor([[1, 0, 2, 4], [1, 0, 2, 4], [1, 0, 2, 4]], dtype=torch.bfloat16)
    view = ParameterImportanceView(parameter_id=7, parameter_name="layer.weight", tensor=current)
    algorithm = PretrainedDeltaTopK(topk_ratio=0.25, comparison_chunk_rows=1)

    selection = algorithm.select(view, reference, current)

    assert selection.first_indices.tolist() == [3]
    assert selection.second_indices.tolist() == [2]
    assert selection.first_min_score == 12.0
    assert selection.second_min_score == 6.0


def test_selective_linear_computes_full_input_gradient_and_selected_weight_columns():
    torch.manual_seed(1234)
    native = torch.nn.Linear(6, 4, bias=True, dtype=torch.float64)
    selective = torch.nn.Linear(6, 4, bias=True, dtype=torch.float64)
    selective.load_state_dict(native.state_dict())
    assert enable_selective_linear(selective, minimum_tokens=1) == 1
    set_selected_columns(selective.weight, torch.tensor([1]), torch.tensor([4]))
    native_inputs = torch.randn(2, 3, 6, dtype=torch.float64, requires_grad=True)
    selective_inputs = native_inputs.detach().clone().requires_grad_(True)
    output_gradient = torch.randn(2, 3, 4, dtype=torch.float64)

    native(native_inputs).backward(output_gradient)
    selective(selective_inputs).backward(output_gradient)

    assert torch.allclose(native_inputs.grad, selective_inputs.grad)
    assert torch.allclose(native.bias.grad, selective.bias.grad)
    assert torch.allclose(native.weight.grad[:, [1, 4]], selective.weight.grad[:, [1, 4]])
    assert torch.count_nonzero(selective.weight.grad[:, [0, 2, 3, 5]]) == 0


def test_selective_linear_routes_packed_gradient_without_dense_parameter_gradient():
    torch.manual_seed(1234)
    linear = torch.nn.Linear(6, 4, bias=True, dtype=torch.float64)
    enable_selective_linear(linear, minimum_tokens=1)
    set_selected_columns(linear.weight, torch.tensor([1]), torch.tensor([4]))
    captured = []
    set_takeover_capture(
        linear.weight, 17, lambda parameter_id, columns, gradient: captured.append(
            (parameter_id, columns.clone(), gradient.clone())))
    inputs = torch.randn(2, 3, 6, dtype=torch.float64, requires_grad=True)
    output_gradient = torch.randn(2, 3, 4, dtype=torch.float64)
    expected = output_gradient.reshape(-1, 4).transpose(0, 1).matmul(inputs.detach().reshape(-1, 6)[:, [1, 4]])

    linear(inputs).backward(output_gradient)

    assert inputs.grad is not None
    assert linear.weight.grad is None
    assert linear.bias.grad is None
    assert captured[0][0] == 17
    assert captured[0][1].tolist() == [1, 4]
    assert torch.allclose(captured[0][2], expected)

    set_dense_backward(linear.weight, True)
    linear(inputs.detach()).sum().backward()
    assert linear.weight.grad is not None
    assert linear.bias.grad is not None


def test_selective_linear_uses_native_backward_below_performance_threshold():
    linear = torch.nn.Linear(4, 3, bias=False, dtype=torch.float64)
    enable_selective_linear(linear, minimum_tokens=8)
    set_selected_columns(linear.weight, torch.tensor([1]), torch.tensor([2]))
    inputs = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)

    linear(inputs).sum().backward()

    assert torch.count_nonzero(linear.weight.grad[:, [0, 3]]) > 0


def test_selective_linear_uses_native_backward_below_output_threshold():
    linear = torch.nn.Linear(4, 3, bias=False, dtype=torch.float64)
    enable_selective_linear(linear, minimum_tokens=1, minimum_output_features=4)
    set_selected_columns(linear.weight, torch.tensor([1]), torch.tensor([2]))

    linear(torch.randn(8, 4, dtype=torch.float64)).sum().backward()

    assert torch.count_nonzero(linear.weight.grad[:, [0, 3]]) > 0


def test_streaming_selector_waits_for_warmup_and_keeps_one_dimensional_parameters_dense(tmp_path):
    weight = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.bfloat16)
    bias = torch.zeros(4, dtype=torch.bfloat16)
    views = [
        ParameterImportanceView(parameter_id=1, parameter_name="layer.weight", tensor=weight),
        ParameterImportanceView(parameter_id=2, parameter_name="layer.bias", tensor=bias),
    ]
    metrics = MetricsRegistry()
    registry = ImportanceRegistry()
    selector = StreamingImportanceSelector(views,
                                           PretrainedDeltaTopK(0.25, 1),
                                           registry,
                                           metrics,
                                           warmup_steps=2,
                                           rank=0,
                                           output_path=str(tmp_path / "importance.pt"))
    weight[:, 1] = 3.0
    weight[:, 3] = 2.0

    selector.on_step_end(1)
    assert registry.ready is False
    selector.on_step_end(2)

    assert registry.ready is True
    assert registry.get(1).first_indices.tolist() == [1]
    assert registry.get(1).second_indices.tolist() == [3]
    assert registry.is_dense(2) is True
    snapshot = metrics.snapshot()
    assert snapshot.counters["importance_selected_parameter_count"] == 1
    assert snapshot.counters["importance_dense_parameter_count"] == 1
    assert snapshot.gauges["importance_reference_bytes"] == 0
    assert (tmp_path / "importance.rank0.pt").exists()
    selector.close()
