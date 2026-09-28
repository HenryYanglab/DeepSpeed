# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from deepspeed.runtime.fastoffload.adapters.zero2 import Zero2ObserverAdapter
from deepspeed.runtime.fastoffload.api import create_zero2_controller, install
from deepspeed.runtime.fastoffload.config import FastOffloadConfig, ObserverFailurePolicy
from deepspeed.runtime.fastoffload.context import GradientBucketContext, GradientContext, StepContext
from deepspeed.runtime.fastoffload.controller import FastOffloadController, NullFastOffloadController


class FakeZero2Optimizer:

    def __init__(self):
        self.parameter = torch.nn.Parameter(torch.zeros(16, dtype=torch.float16))
        self.partition_gradients = True
        self.cpu_offload = True
        self.micro_step_id = -1
        self.gradient_accumulation_steps = 2
        self.dp_process_group = None
        self.real_dp_process_group = [None]
        self.grad_position = {3: [0, 8, 0, 8]}
        self.param_names = {self.parameter: "decoder.layer.weight"}
        self.bit16_groups = [[self.parameter]]
        self.round_robin_bit16_groups = [[self.parameter]]
        self.partition_size = [16]
        self.single_partition_of_fp32_groups = [SimpleNamespace(grad=torch.zeros(8, dtype=torch.float32))]
        self.master_weights_and_grads_dtype = torch.float32
        self.overlap_comm = True
        self._is_boundary = False
        self.cleared_parameters = []

    def get_param_id(self, parameter):
        assert parameter is self.parameter
        return 3

    def is_gradient_accumulation_boundary(self):
        return self._is_boundary

    def get_param_gradient_attribute(self, parameter):
        assert parameter is self.parameter
        return parameter.grad

    def clear_grad_attribute(self, parameter):
        self.cleared_parameters.append(parameter)
        parameter.grad = None


def test_cpu_owner_group_rejects_subgroups_before_collective_creation(monkeypatch):
    from deepspeed.runtime.fastoffload.adapters import zero2
    optimizer = FakeZero2Optimizer()
    optimizer.real_dp_process_group = [object()]
    communication = SimpleNamespace(is_initialized=lambda: True,
                                    get_world_size=lambda group=None: 2 if group is None else 1,
                                    get_rank=lambda group=None: 0,
                                    get_global_rank=lambda group, rank: rank,
                                    new_group=Mock())
    monkeypatch.setattr(zero2, "dist", communication)
    adapter = Zero2ObserverAdapter(optimizer)
    with pytest.raises(ValueError, match="full-world data parallel group"):
        adapter.create_cpu_owner_collectives()
    communication.new_group.assert_not_called()


def make_step_context(timestamp_ns=1):
    return StepContext(global_step=0,
                       micro_step=0,
                       rank=0,
                       world_size=1,
                       zero_stage=2,
                       gradient_accumulation_boundary=True,
                       timestamp_ns=timestamp_ns)


def make_gradient_context():
    return GradientContext(parameter_id=3,
                           parameter_name="weight",
                           group_id=0,
                           partition_id=0,
                           parameter_numel=8,
                           shard_numel=8,
                           shard_offset=0,
                           element_size=4,
                           shard_bytes=32,
                           dtype="torch.float32",
                           device="cpu",
                           global_step=0,
                           micro_step=0,
                           rank=0,
                           world_size=1,
                           zero_stage=2,
                           gradient_accumulation_boundary=True,
                           is_local_partition=True,
                           is_cpu_offload=True,
                           timestamp_ns=1)


def make_bucket_context():
    return GradientBucketContext(bucket_id=0,
                                 parameter_count=1,
                                 total_numel=8,
                                 total_bytes=16,
                                 communication_dtype="torch.float16",
                                 overlap_comm=True,
                                 global_step=0,
                                 micro_step=0,
                                 rank=0,
                                 world_size=1,
                                 zero_stage=2,
                                 gradient_accumulation_boundary=True,
                                 timestamp_ns=1)


def test_zero2_adapter_builds_step_and_gradient_contexts():
    optimizer = FakeZero2Optimizer()
    adapter = Zero2ObserverAdapter(optimizer)

    first_backward = adapter.create_backward_begin_context()
    gradient = adapter.create_gradient_context(optimizer.parameter, 0)

    assert first_backward.micro_step == 0
    assert first_backward.gradient_accumulation_boundary is False
    assert gradient.parameter_id == 3
    assert gradient.parameter_name == "decoder.layer.weight"
    assert gradient.shard_offset == 8
    assert gradient.shard_numel == 8
    assert gradient.element_size == 4
    assert gradient.shard_bytes == 32
    assert gradient.dtype == "torch.float32"
    assert gradient.device == "cpu"

    optimizer.micro_step_id = 0
    second_backward = adapter.create_backward_begin_context()
    assert second_backward.micro_step == 1
    assert second_backward.gradient_accumulation_boundary is True


def test_zero2_adapter_reads_live_scheduler_lr_without_caching():
    optimizer = FakeZero2Optimizer()
    optimizer.optimizer = SimpleNamespace(param_groups=[{"lr": 0.1}, {"lr": 0.1}])
    adapter = Zero2ObserverAdapter(optimizer)
    assert adapter.get_learning_rate() == 0.1
    for lr in (0.02, 0.0, 0.05):
        for group in optimizer.optimizer.param_groups:
            group["lr"] = lr
        assert adapter.get_learning_rate() == lr


@pytest.mark.parametrize("rates", [[], [0.1, 0.2], [-0.1], [float("nan")], [float("inf")]])
def test_zero2_adapter_rejects_invalid_or_divergent_learning_rates(rates):
    optimizer = FakeZero2Optimizer()
    optimizer.optimizer = SimpleNamespace(param_groups=[{"lr": lr} for lr in rates])
    with pytest.raises(ValueError):
        Zero2ObserverAdapter(optimizer).get_learning_rate()


def test_zero2_adapter_exposes_matrix_partition_layouts():
    optimizer = FakeZero2Optimizer()
    optimizer.parameter = torch.nn.Parameter(torch.zeros((2, 8), dtype=torch.float16))
    optimizer.bit16_groups = [[optimizer.parameter]]
    optimizer.round_robin_bit16_groups = [[optimizer.parameter]]
    optimizer.param_names = {optimizer.parameter: "decoder.layer.weight"}
    adapter = Zero2ObserverAdapter(optimizer)

    layouts = tuple(adapter.iter_parameter_partition_layouts())

    assert len(layouts) == 1
    assert layouts[0].parameter_id == 3
    assert layouts[0].shape == (2, 8)
    assert layouts[0].group_offset == 0
    assert layouts[0].partition_size == 16
    reducer = adapter.create_compressed_gradient_reducer()
    assert reducer is not None


def test_zero2_adapter_commits_and_publishes_fp32_partition_values():
    optimizer = FakeZero2Optimizer()
    optimizer.single_partition_of_fp32_groups = [torch.zeros(16)]
    optimizer.update_lp_params = Mock()
    adapter = Zero2ObserverAdapter(optimizer)

    adapter.write_fp32_partition_values(0, torch.tensor([2, 5]), torch.tensor([3.0, 7.0]))
    adapter.publish_fp32_partitions([0])

    assert optimizer.single_partition_of_fp32_groups[0][2].item() == 3.0
    assert optimizer.single_partition_of_fp32_groups[0][5].item() == 7.0
    optimizer.update_lp_params.assert_called_once_with()


def test_zero2_adapter_exposes_named_importance_views():
    optimizer = FakeZero2Optimizer()
    adapter = Zero2ObserverAdapter(optimizer)

    views = list(adapter.iter_parameter_importance_views())

    assert len(views) == 1
    assert views[0].parameter_id == 3
    assert views[0].parameter_name == "decoder.layer.weight"
    assert views[0].tensor is optimizer.parameter


def test_zero2_adapter_builds_bucket_context_and_advances_step():
    optimizer = FakeZero2Optimizer()
    adapter = Zero2ObserverAdapter(optimizer)
    bucket = SimpleNamespace(params=[(0, 0, 3)], grads=[optimizer.parameter], elements=16)

    first = adapter.create_bucket_context(torch.float16, bucket)
    second = adapter.create_bucket_context(torch.float16, bucket)
    adapter.complete_step()
    step = adapter.create_step_context()

    assert first.bucket_id == 0
    assert first.parameter_count == 1
    assert first.total_bytes == 32
    assert second.bucket_id == 1
    assert step.global_step == 1


def test_zero2_adapter_builds_transfer_view_and_clears_gradient():
    optimizer = FakeZero2Optimizer()
    optimizer.parameter.grad = torch.arange(16, dtype=torch.float16)
    adapter = Zero2ObserverAdapter(optimizer)

    transfer_view = adapter.create_gradient_transfer_view(optimizer.parameter)

    assert transfer_view.source_offset == 8
    assert transfer_view.destination_offset == 0
    assert transfer_view.numel == 8
    assert transfer_view.source.dtype == torch.float32
    assert torch.equal(transfer_view.source, torch.arange(8, 16, dtype=torch.float32))
    adapter.clear_gradient(optimizer.parameter)
    assert optimizer.parameter.grad is None


def test_zero2_adapter_rejects_unsupported_optimizer_modes():
    optimizer = FakeZero2Optimizer()
    optimizer.partition_gradients = False
    with pytest.raises(ValueError, match="Stage 2"):
        Zero2ObserverAdapter(optimizer)

    optimizer.partition_gradients = True
    optimizer.cpu_offload = False
    with pytest.raises(ValueError, match="CPU"):
        Zero2ObserverAdapter(optimizer)


def test_zero2_adapter_rejects_invalid_group():
    optimizer = FakeZero2Optimizer()
    adapter = Zero2ObserverAdapter(optimizer)

    with pytest.raises(ValueError, match="group"):
        adapter.create_gradient_context(optimizer.parameter, 1)


def make_controller(failure_policy=ObserverFailurePolicy.raise_error):
    adapter = Mock()
    observer = Mock()
    adapter.create_backward_begin_context.return_value = make_step_context()
    adapter.create_step_context.return_value = make_step_context()
    adapter.create_gradient_context.return_value = make_gradient_context()
    adapter.create_bucket_context.return_value = make_bucket_context()
    controller = FastOffloadController(adapter, observer, failure_policy)
    return controller, adapter, observer


def test_controller_forwards_all_events():
    controller, adapter, observer = make_controller()
    parameter = object()
    bucket = object()

    controller.on_backward_begin()
    controller.on_gradient_ready(parameter, 0)
    controller.on_gradient_reduced(parameter, 0)
    controller.on_gradient_bucket(torch.float16, bucket)
    controller.on_backward_end()
    controller.on_step_begin()
    controller.on_step_end()

    observer.on_backward_begin.assert_called_once()
    assert observer.on_gradient_ready.call_args.args[0] == make_gradient_context()
    assert observer.on_gradient_reduced.call_args.args[0] == make_gradient_context()
    observer.on_gradient_bucket.assert_called_once()
    observer.on_backward_end.assert_called_once()
    observer.on_step_begin.assert_called_once()
    observer.on_step_end.assert_called_once()
    adapter.complete_step.assert_called_once()


def test_controller_runs_importance_selection_after_completed_warmup_step():
    adapter = Mock()
    observer = Mock()
    selector = Mock()
    selector.registry = object()
    adapter.create_step_context.return_value = make_step_context()
    controller = FastOffloadController(adapter,
                                       observer,
                                       ObserverFailurePolicy.raise_error,
                                       importance_selector=selector)

    controller.on_step_end()

    selector.on_step_end.assert_called_once_with(1)
    assert controller.importance_registry is selector.registry
    adapter.complete_step.assert_called_once()
    controller.close()
    selector.close.assert_called_once()


def test_controller_round_trips_optional_hybrid_checkpoint_state():
    adapter = Mock()
    observer = Mock()
    runtime = Mock()
    runtime.state_dict.return_value = {"committed_version": 3}
    controller = FastOffloadController(adapter, observer, ObserverFailurePolicy.raise_error, hybrid_runtime=runtime)

    state = controller.hybrid_state_dict()
    controller.load_hybrid_state_dict(state)

    assert state == {"committed_version": 3}
    runtime.load_state_dict.assert_called_once_with(state)


def test_controller_failure_policies():
    raising_controller, _, raising_observer = make_controller(ObserverFailurePolicy.raise_error)
    raising_observer.on_backward_begin.side_effect = RuntimeError("failure")
    with pytest.raises(RuntimeError, match="failure"):
        raising_controller.on_backward_begin()

    warn_controller, _, warn_observer = make_controller(ObserverFailurePolicy.warn)
    warn_observer.on_backward_begin.side_effect = RuntimeError("failure")
    with patch("deepspeed.runtime.fastoffload.controller.logger.warning") as warning:
        warn_controller.on_backward_begin()
    assert warn_controller.disabled is False
    warning.assert_called_once()

    disable_controller, _, disable_observer = make_controller(ObserverFailurePolicy.disable_observer)
    disable_observer.on_backward_begin.side_effect = RuntimeError("failure")
    disable_controller.on_backward_begin()
    assert disable_controller.disabled is True
    disable_observer.on_backward_begin.reset_mock()
    disable_controller.on_backward_begin()
    disable_observer.on_backward_begin.assert_not_called()


def test_disabled_observer_still_flushes_active_scheduler():
    adapter = Mock()
    observer = Mock()
    scheduler = Mock()
    adapter.create_backward_begin_context.return_value = make_step_context()
    observer.on_backward_begin.side_effect = RuntimeError("observer failed")
    controller = FastOffloadController(adapter, observer, ObserverFailurePolicy.disable_observer, scheduler=scheduler)

    controller.on_backward_begin()
    assert controller.disabled is True
    controller.on_backward_end()
    controller.prepare_step()

    scheduler.progress.assert_called_once()
    scheduler.prepare_step.assert_called_once()
    controller.close()
    scheduler.close.assert_called_once()


def test_controller_transfers_gradient_with_scheduler():
    adapter = Mock()
    observer = Mock()
    scheduler = Mock()
    transfer_view = object()
    adapter.create_gradient_transfer_view.return_value = transfer_view
    scheduler.submit.return_value = True
    scheduler.requires_stable_source = True
    controller = FastOffloadController(adapter, observer, ObserverFailurePolicy.raise_error, scheduler=scheduler)
    parameter = object()

    assert controller.transfer_gradient(parameter) is True
    adapter.create_gradient_transfer_view.assert_called_once_with(parameter, stable_source=True)
    scheduler.submit.assert_called_once_with(transfer_view)
    adapter.clear_gradient.assert_called_once_with(parameter)
    controller.close()
    scheduler.close.assert_called_once()


def test_null_controller_accepts_every_event():
    controller = NullFastOffloadController()
    controller.on_backward_begin()
    assert controller.transfer_gradient(object()) is False
    controller.on_gradient_ready(object(), 0)
    controller.on_gradient_reduced(object(), 0)
    controller.on_gradient_bucket(torch.float16, object())
    controller.on_backward_end()
    controller.on_step_begin()
    controller.on_step_end()
    controller.close()


def test_install_disabled_config_creates_null_controller():
    handle = install(FastOffloadConfig(enabled=False))
    try:
        controller = create_zero2_controller(object())
        assert isinstance(controller, NullFastOffloadController)
    finally:
        handle.close()


def test_disabled_hybrid_ignores_takeover_suboptions():
    optimizer = FakeZero2Optimizer()
    handle = install({
        "enabled": True,
        "importance": {
            "enabled": False
        },
        "hybrid_update": {
            "enabled": False,
            "compressed_collective_shadow": True
        }
    })
    try:
        controller = create_zero2_controller(optimizer)
        assert isinstance(controller, FastOffloadController)
    finally:
        handle.close()


def test_install_enabled_config_attaches_and_closes_controller():
    optimizer = FakeZero2Optimizer()
    handle = install({"enabled": True, "telemetry": {"host_timing": False}})
    controller = create_zero2_controller(optimizer)

    assert isinstance(controller, FastOffloadController)
    assert controller.disabled is False
    handle.close()
    assert handle.closed is True
    assert controller.disabled is True


def test_install_rejects_duplicate_active_handle():
    handle = install(FastOffloadConfig())
    try:
        with pytest.raises(RuntimeError, match="already installed"):
            install(FastOffloadConfig())
    finally:
        handle.close()


def test_install_accepts_json_path(tmp_path):
    path = tmp_path / "fastoffload.json"
    path.write_text('{"enabled": false}', encoding="utf-8")

    handle = install(path)
    assert handle.config.enabled is False
    handle.close()


def test_hybrid_runtime_rejects_silent_dense_fallback():
    handle = install({"enabled": True, "importance": {"enabled": True}, "hybrid_update": {"enabled": True}})
    try:
        with pytest.raises(NotImplementedError, match="step takeover"):
            create_zero2_controller(object())
    finally:
        handle.close()


def test_hybrid_shadow_switch_keeps_native_controller_available():
    optimizer = FakeZero2Optimizer()
    optimizer.gradient_accumulation_steps = 1
    handle = install({
        "enabled": True,
        "importance": {
            "enabled": True,
            "warmup_steps": 1
        },
        "hybrid_update": {
            "enabled": True,
            "compressed_collective_shadow": True
        }
    })
    try:
        controller = create_zero2_controller(optimizer)
        assert isinstance(controller, FastOffloadController)
        controller.close()
    finally:
        handle.close()


def test_install_rejects_invalid_input():
    with pytest.raises(TypeError):
        install(1)
