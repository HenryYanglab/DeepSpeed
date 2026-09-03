# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.fastoffload import enable_selective_linear, install
from deepspeed.runtime.fastoffload.controller import FastOffloadController
from unit.common import DistributedTest
from unit.simple_model import SimpleModel


def clone_state(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_state(item) for item in value]
    return value


def assert_state_equal(expected, actual):
    if torch.is_tensor(expected):
        assert torch.equal(expected, actual)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            assert_state_equal(expected[key], actual[key])
    elif isinstance(expected, list):
        assert len(expected) == len(actual)
        for expected_item, actual_item in zip(expected, actual):
            assert_state_equal(expected_item, actual_item)
    else:
        assert expected == actual


def precision_config():
    if get_accelerator().is_bf16_supported():
        return {"bf16": {"enabled": True}}
    if get_accelerator().is_fp16_supported():
        return {"fp16": {"enabled": True}}
    return {}


class TestZero2ObserverIntegration(DistributedTest):
    world_size = 1

    def _run_step(self,
                  mode,
                  gradient_accumulation_steps=1,
                  importance=False,
                  hybrid_shadow=False,
                  takeover=False,
                  optimizer_steps=1):
        hidden_dim = 8
        torch.manual_seed(1234)
        model = SimpleModel(hidden_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        if takeover:
            enable_selective_linear(model, minimum_tokens=1)
        config = {
            "train_batch_size": gradient_accumulation_steps,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True,
                },
            },
            "zero_force_ds_cpu_optimizer": False,
        }
        config.update(precision_config())

        fastoffload_config = {
            "enabled": mode is not None,
            "mode": mode or "observe",
            "transfer": {
                "buffer_count": 1,
                "buffer_size": 4096,
            },
            "telemetry": {
                "host_timing": False,
                "log_interval": 100,
            },
        }
        if importance or hybrid_shadow or takeover:
            fastoffload_config["importance"] = {
                "enabled": True,
                "warmup_steps": 1,
                "topk_ratio": 0.25,
                "comparison_chunk_rows": 2,
                "sparse_backward": takeover,
                "sparse_backward_min_tokens": 1,
            }
        if hybrid_shadow:
            fastoffload_config["hybrid_update"] = {
                "enabled": True,
                "update_interval": 2,
                "compressed_collective_shadow": True,
                "compressed_bucket_bytes": 32,
                "parity_atol": 1e-3,
                "parity_rtol": 1e-3,
            }
        if takeover:
            fastoffload_config["hybrid_update"] = {
                "enabled": True,
                "update_interval": 2,
                "zero2_takeover": True,
                "compressed_bucket_bytes": 32,
            }
        if mode == "async_offload":
            fastoffload_config["scheduler"] = {
                "type": "overlap",
                "max_inflight_tasks": 2,
                "max_inflight_bytes": 8192,
            }
            fastoffload_config["transfer"]["buffer_count"] = 2
        handle = install(fastoffload_config)
        try:
            engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
            torch.manual_seed(5678)
            for _ in range(gradient_accumulation_steps * optimizer_steps):
                inputs = torch.randn(1, hidden_dim, device=engine.device, dtype=engine.module.linears[0].weight.dtype)
                labels = torch.randint(0, hidden_dim, (1, ), device=engine.device)
                loss = engine(inputs, labels)
                engine.backward(loss)
                engine.step()

            state = {name: value.detach().cpu().clone() for name, value in engine.module.state_dict().items()}
            optimizer_state = clone_state(engine.optimizer.optimizer.state_dict())
            metrics = None
            controller = engine.optimizer._fastoffload_controller
            if mode is not None:
                assert isinstance(controller, FastOffloadController)
                metrics = controller.observer.snapshot()
            engine.destroy()
            return loss.detach().float().cpu(), state, optimizer_state, metrics
        finally:
            handle.close()

    def test(self):
        baseline_loss, baseline_state, baseline_optimizer_state, _ = self._run_step(mode=None)
        observed_loss, observed_state, observed_optimizer_state, observer_metrics = self._run_step(mode="observe")
        sync_loss, sync_state, sync_optimizer_state, sync_metrics = self._run_step(mode="sync_offload")
        async_loss, async_state, async_optimizer_state, async_metrics = self._run_step(mode="async_offload")

        assert torch.equal(baseline_loss, observed_loss)
        assert torch.equal(baseline_loss, sync_loss)
        assert torch.equal(baseline_loss, async_loss)
        assert baseline_state.keys() == observed_state.keys()
        assert baseline_state.keys() == sync_state.keys()
        assert baseline_state.keys() == async_state.keys()
        for name in baseline_state:
            assert torch.equal(baseline_state[name], observed_state[name]), name
            assert torch.equal(baseline_state[name], sync_state[name]), name
            assert torch.equal(baseline_state[name], async_state[name]), name
        assert_state_equal(baseline_optimizer_state, observed_optimizer_state)
        assert_state_equal(baseline_optimizer_state, sync_optimizer_state)
        assert_state_equal(baseline_optimizer_state, async_optimizer_state)

        assert observer_metrics.counters["backward_count"] == 1
        assert observer_metrics.counters["step_count"] == 1
        assert observer_metrics.counters["gradient_ready_count"] > 0
        assert observer_metrics.counters["gradient_reduced_count"] > 0
        assert observer_metrics.counters["potential_offload_bytes"] > 0
        assert "actual_offload_bytes" not in observer_metrics.counters

        assert sync_metrics.counters["actual_offload_bytes"] == sync_metrics.counters["potential_offload_bytes"]
        assert sync_metrics.counters["d2h_copy_count"] == sync_metrics.counters["gradient_reduced_count"]
        assert sync_metrics.gauges["buffer_pool_in_use"] == 0
        assert sync_metrics.gauges["pinned_pool_peak_bytes"] == 0
        assert "cpu_staging_copy_count" not in sync_metrics.counters

        assert async_metrics.counters["actual_offload_bytes"] == async_metrics.counters["potential_offload_bytes"]
        assert async_metrics.counters["copy_submit_count"] == async_metrics.counters["copy_complete_count"]
        assert async_metrics.gauges["inflight_tasks"] == 0
        assert async_metrics.gauges["buffer_pool_in_use"] == 0
        assert async_metrics.gauges["pinned_pool_peak_bytes"] == 0
        assert async_metrics.gauges["gpu_staging_pool_peak_bytes"] == 0
        assert async_metrics.counters["producer_stream_submit_count"] == async_metrics.counters[
            "gradient_reduced_count"]
        assert "transfer_hidden_ratio" not in async_metrics.histograms
        assert "cpu_staging_copy_count" not in async_metrics.counters

    def test_importance_selection_preserves_native_update(self):
        baseline_loss, baseline_state, baseline_optimizer_state, _ = self._run_step(mode=None)
        importance_loss, importance_state, importance_optimizer_state, metrics = self._run_step(mode="observe",
                                                                                                importance=True)

        assert torch.equal(baseline_loss, importance_loss)
        assert_state_equal(baseline_state, importance_state)
        assert_state_equal(baseline_optimizer_state, importance_optimizer_state)
        assert metrics.gauges["importance_ready"] == 1
        assert metrics.gauges["importance_reference_bytes"] == 0
        assert metrics.counters["importance_selected_parameter_count"] > 0
        assert metrics.counters["importance_first_column_count"] > 0
        assert metrics.counters["importance_second_column_count"] > 0

    def test_zero2_takeover_bypasses_native_dense_reduction_after_warmup(self):
        loss, state, _, metrics = self._run_step(mode="observe", takeover=True, optimizer_steps=4)

        assert torch.isfinite(loss)
        assert all(torch.isfinite(value).all() for value in state.values())
        assert metrics.counters["takeover_native_boundary_count"] > 0
        assert metrics.counters["takeover_owner_bucket_count"] > 0
        assert metrics.gauges["takeover_owner_bucket_peak_bytes"] > 0

    def test_zero2_takeover_supports_gradient_accumulation(self):
        loss, state, _, metrics = self._run_step(mode="observe",
                                                 gradient_accumulation_steps=2,
                                                 takeover=True,
                                                 optimizer_steps=3)

        assert torch.isfinite(loss)
        assert all(torch.isfinite(value).all() for value in state.values())
        assert metrics.counters["step_count"] == 3

    def test_hybrid_shadow_validates_native_and_owner_values(self):
        _, _, _, metrics = self._run_step(mode="observe", hybrid_shadow=True, optimizer_steps=3)

        assert metrics.counters["hybrid_shadow_parity_check_count"] > 0
        assert metrics.counters.get("hybrid_shadow_parity_mismatch_count", 0) == 0
        assert metrics.counters["hybrid_shadow_norm_check_count"] == 2
        assert metrics.counters["hybrid_shadow_overflow_check_count"] == 2
        assert metrics.counters["hybrid_shadow_owner_write_check_count"] > 0
        assert "hybrid_shadow_owner_write_mismatch_count" not in metrics.counters
        assert metrics.gauges["hybrid_shadow_bucket_peak_bytes"] <= 128

    def test_gradient_accumulation_matches_native_zero(self):
        _, baseline_state, baseline_optimizer_state, _ = self._run_step(mode=None, gradient_accumulation_steps=2)
        _, sync_state, sync_optimizer_state, sync_metrics = self._run_step(mode="sync_offload",
                                                                           gradient_accumulation_steps=2)
        _, async_state, async_optimizer_state, async_metrics = self._run_step(mode="async_offload",
                                                                              gradient_accumulation_steps=2)

        assert baseline_state.keys() == sync_state.keys()
        assert baseline_state.keys() == async_state.keys()
        for name in baseline_state:
            assert torch.equal(baseline_state[name], sync_state[name]), name
            assert torch.equal(baseline_state[name], async_state[name]), name
        assert_state_equal(baseline_optimizer_state, sync_optimizer_state)
        assert_state_equal(baseline_optimizer_state, async_optimizer_state)
        assert sync_metrics.counters["actual_offload_bytes"] > 0
        assert sync_metrics.counters["d2h_copy_count"] < sync_metrics.counters["gradient_reduced_count"]
        assert sync_metrics.gauges["buffer_pool_in_use"] == 0
        assert async_metrics.counters["actual_offload_bytes"] > 0
        assert async_metrics.gauges["inflight_tasks"] == 0
        assert async_metrics.gauges["buffer_pool_in_use"] == 0


class TestZero2TakeoverTwoRanks(DistributedTest):
    world_size = 2

    def test(self):
        hidden_dim = 8
        torch.manual_seed(1234)
        model = SimpleModel(hidden_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        enable_selective_linear(model, minimum_tokens=1)
        config = {
            "train_batch_size": 2,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True,
                },
            },
            "zero_force_ds_cpu_optimizer": False,
        }
        config.update(precision_config())
        handle = install({
            "enabled": True,
            "mode": "observe",
            "importance": {
                "enabled": True,
                "warmup_steps": 1,
                "topk_ratio": 0.25,
                "comparison_chunk_rows": 2,
                "sparse_backward": True,
                "sparse_backward_min_tokens": 1,
            },
            "hybrid_update": {
                "enabled": True,
                "update_interval": 2,
                "zero2_takeover": True,
                "compressed_bucket_bytes": 32,
            },
            "telemetry": {
                "host_timing": False,
                "log_interval": 100,
            },
        })
        engine = None
        try:
            engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
            torch.manual_seed(5678)
            for _ in range(4):
                inputs = torch.randn(1, hidden_dim, device=engine.device, dtype=engine.module.linears[0].weight.dtype)
                labels = torch.randint(0, hidden_dim, (1, ), device=engine.device)
                loss = engine(inputs, labels)
                engine.backward(loss)
                engine.step()
            for parameter in engine.module.parameters():
                assert torch.isfinite(parameter).all()
                mean = parameter.detach().clone()
                dist.all_reduce(mean)
                mean.div_(dist.get_world_size())
                assert torch.allclose(parameter, mean)
        finally:
            if engine is not None:
                engine.destroy()
            handle.close()

    def test_overflow_and_checkpoint_drain(self):
        hidden_dim = 8
        torch.manual_seed(1234)
        model = SimpleModel(hidden_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        enable_selective_linear(model, minimum_tokens=1)
        config = {
            "train_batch_size": 2,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True,
                },
            },
            "zero_force_ds_cpu_optimizer": False,
        }
        config.update(precision_config())
        handle = install({
            "enabled": True,
            "mode": "observe",
            "importance": {
                "enabled": True,
                "warmup_steps": 1,
                "topk_ratio": 0.25,
                "comparison_chunk_rows": 2,
                "sparse_backward": True,
                "sparse_backward_min_tokens": 1,
            },
            "hybrid_update": {
                "enabled": True,
                "update_interval": 2,
                "zero2_takeover": True,
                "compressed_bucket_bytes": 32,
                "max_async_lag": 2,
            },
            "telemetry": {
                "host_timing": False,
                "log_interval": 100,
            },
        })
        engine = None
        try:
            engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
            torch.manual_seed(5678)
            for _ in range(3):
                inputs = torch.randn(1, hidden_dim, device=engine.device, dtype=engine.module.linears[0].weight.dtype)
                labels = torch.randint(0, hidden_dim, (1, ), device=engine.device)
                loss = engine(inputs, labels)
                engine.backward(loss)
                engine.step()

            runtime = engine.optimizer._fastoffload_controller._takeover_runtime
            assert runtime.pending_updates == 1
            checkpoint = engine.optimizer._fastoffload_controller.hybrid_state_dict()
            assert runtime.pending_updates == 0
            engine.optimizer._fastoffload_controller.load_hybrid_state_dict(checkpoint)

            parameters_before_overflow = [parameter.detach().clone() for parameter in engine.module.parameters()]
            inputs = torch.randn(1, hidden_dim, device=engine.device, dtype=engine.module.linears[0].weight.dtype)
            labels = torch.randint(0, hidden_dim, (1, ), device=engine.device)
            loss = engine(inputs, labels) * float("inf")
            engine.backward(loss)
            engine.step()
            assert engine.optimizer.overflow
            for before, parameter in zip(parameters_before_overflow, engine.module.parameters()):
                assert torch.equal(before, parameter)

            for parameter in engine.module.parameters():
                mean = parameter.detach().clone()
                dist.all_reduce(mean)
                mean.div_(dist.get_world_size())
                assert torch.equal(parameter, mean)
        finally:
            if engine is not None:
                engine.destroy()
            handle.close()


class TestZero2HybridShadowTwoRanks(DistributedTest):
    world_size = 2

    def test(self):
        hidden_dim = 8
        torch.manual_seed(1234)
        model = SimpleModel(hidden_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        config = {
            "train_batch_size": 2,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True,
                },
            },
            "zero_force_ds_cpu_optimizer": False,
        }
        config.update(precision_config())
        fastoffload_config = {
            "enabled": True,
            "mode": "observe",
            "importance": {
                "enabled": True,
                "warmup_steps": 1,
                "topk_ratio": 0.25,
                "comparison_chunk_rows": 2,
            },
            "hybrid_update": {
                "enabled": True,
                "update_interval": 2,
                "compressed_collective_shadow": True,
                "compressed_bucket_bytes": 32,
                "parity_atol": 1e-3,
                "parity_rtol": 1e-3,
            },
            "telemetry": {
                "host_timing": False,
                "log_interval": 100,
            },
        }
        handle = install(fastoffload_config)
        engine = None
        try:
            engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
            torch.manual_seed(5678)
            for _ in range(3):
                inputs = torch.randn(1, hidden_dim, device=engine.device, dtype=engine.module.linears[0].weight.dtype)
                labels = torch.randint(0, hidden_dim, (1, ), device=engine.device)
                loss = engine(inputs, labels)
                engine.backward(loss)
                engine.step()

            metrics = engine.optimizer._fastoffload_controller.observer.snapshot()
            assert metrics.counters["hybrid_shadow_parity_check_count"] > 0
            assert metrics.counters.get("hybrid_shadow_parity_mismatch_count", 0) == 0
            assert metrics.counters["hybrid_shadow_owner_write_check_count"] > 0
            assert metrics.counters.get("hybrid_shadow_owner_write_mismatch_count", 0) == 0
            assert metrics.counters["hybrid_shadow_norm_check_count"] == 2
            assert metrics.counters["hybrid_shadow_overflow_check_count"] == 2
        finally:
            if engine is not None:
                engine.destroy()
            handle.close()
