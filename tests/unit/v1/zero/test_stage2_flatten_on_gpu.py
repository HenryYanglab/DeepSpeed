# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Test ZeRO Stage 1/2 flatten placement, including ZenFlow transpose-workspace headroom.
Parametrized over zero_stage (1, 2) and dtype (fp32, fp16, bf16).
"""

import pytest
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import set_log_level_from_string
from unit.common import DistributedTest
from unit.simple_model import SimpleModel, random_dataloader

_DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _apply_dtype_to_config(config_dict, dtype):
    """Set bf16/fp16 in config_dict based on dtype; skip if not supported."""
    if dtype == "bf16":
        if not get_accelerator().is_bf16_supported():
            pytest.skip("bf16 is not supported on this accelerator")
        config_dict["bf16"] = {"enabled": True}
    elif dtype == "fp16":
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported on this accelerator")
        config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
    # fp32: no half-precision block


@pytest.mark.parametrize("zero_stage", [1, 2])
@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"], ids=["fp32", "fp16", "bf16"])
class TestStage2FlattenOnGPU(DistributedTest):
    """ZeRO-1 and ZeRO-2 with small model should flatten on GPU (sufficient VRAM)."""

    world_size = 2  # Run on 2 GPUs when available

    def test_flatten_on_gpu_path_taken(self, monkeypatch, zero_stage, dtype):
        """Assert the GPU flatten path was used (not CPU flatten + move)."""
        if not get_accelerator().is_available():
            pytest.skip("Accelerator not available")
        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {
                "stage": zero_stage
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
        }
        _apply_dtype_to_config(config_dict, dtype)

        set_log_level_from_string("info")
        log_messages = []

        def mock_logger_info(msg, *args, **kwargs):
            log_messages.append(msg if isinstance(msg, str) else str(msg))

        monkeypatch.setattr("deepspeed.utils.logger.info", mock_logger_info)

        hidden_dim = 64
        model = SimpleModel(hidden_dim=hidden_dim, nlayers=2)
        deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=model.parameters(),
        )

        # Small model + no CPU offload => accelerator path logs "Flattening param group ... (sufficient memory)"
        accel_path_logs = [m for m in log_messages if "Flattening param group" in m and "(sufficient memory)" in m]
        assert accel_path_logs, (
            f"Expected accelerator flatten path (log should contain 'Flattening param group' and '(sufficient memory)'). "
            f"Captured messages: {log_messages}")

    def test_flat_buffers_on_accelerator(self, zero_stage, dtype):
        """Regression: flat buffers must end up on the accelerator (not left on CPU)."""
        if not get_accelerator().is_available():
            pytest.skip("Accelerator not available")
        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {
                "stage": zero_stage
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
        }
        _apply_dtype_to_config(config_dict, dtype)

        hidden_dim = 64
        model = SimpleModel(hidden_dim=hidden_dim, nlayers=2)
        engine, _, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=model.parameters(),
        )
        opt = engine.optimizer
        assert hasattr(opt, "bit16_groups_flat"), "ZeRO-1/2 optimizer should have bit16_groups_flat"
        device_type = get_accelerator().device_name()
        for i, flat in enumerate(opt.bit16_groups_flat):
            assert flat.device.type == device_type, (f"Flat buffer {i} must be on {device_type}, got {flat.device}")

    @pytest.mark.world_size(1)
    def test_flatten_on_accelerator_training_step(self, zero_stage, dtype):
        """Regression: flat buffer must be detached so inplace ops during step don't crash."""
        if not get_accelerator().is_available():
            pytest.skip("Accelerator not available")
        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {
                "stage": zero_stage
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
        }
        _apply_dtype_to_config(config_dict, dtype)

        hidden_dim = 64
        model = SimpleModel(hidden_dim=hidden_dim, nlayers=2)
        engine, _, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=model.parameters(),
        )
        for flat in engine.optimizer.bit16_groups_flat:
            assert flat.grad_fn is None, ("Flat buffer must be detached from autograd graph"
                                          " to prevent inplace-modification errors during optimizer step")

        data_loader = random_dataloader(model=engine,
                                        total_samples=8,
                                        hidden_dim=hidden_dim,
                                        device=engine.device,
                                        dtype=_DTYPE_MAP[dtype])
        for batch in data_loader:
            loss = engine(batch[0], batch[1])
            engine.backward(loss)
            engine.step()


@pytest.mark.parametrize("zenflow_offload", [None, False, True], ids=["native", "zenflow", "zenflow_offload"])
@pytest.mark.parametrize("headroom", ["below_flat", "flat_only", "ample"])
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
class TestStage2FlattenMemoryBudget(DistributedTest):
    world_size = 2

    def test_flatten_placement_preserves_parameters(self, monkeypatch, zenflow_offload, headroom, dtype):
        accelerator = get_accelerator()
        if not accelerator.is_available():
            pytest.skip("Accelerator not available")
        config = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
            "zero_optimization": {
                "stage": 2,
                "overlap_comm": True,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                }
            }
        }
        _apply_dtype_to_config(config, dtype)
        use_zenflow = zenflow_offload is not None
        if use_zenflow:
            config["zero_optimization"]["zenflow"] = {
                "topk_ratio": 0.25,
                "update_interval": 4,
                "overlap_step": False,
                "offload": zenflow_offload
            }
        torch.manual_seed(123)
        model = SimpleModel(hidden_dim=64, nlayers=2).to(dtype=_DTYPE_MAP[dtype])
        expected = {name: param.detach().clone() for name, param in model.named_parameters()}
        group_numel = sum(param.numel() for param in model.parameters())
        alignment = 2 * self.world_size
        aligned_numel = (group_numel + alignment - 1) // alignment * alignment
        flat_bytes = aligned_numel * next(model.parameters()).element_size()
        available_bytes = {"below_flat": flat_bytes - 1, "flat_only": flat_bytes, "ample": 2 * flat_bytes}[headroom]
        expect_cpu = headroom == "below_flat" or (use_zenflow and headroom == "flat_only")
        messages = []
        engine = None
        try:
            with monkeypatch.context() as patch:
                # Simulate headroom between the flat output and output-plus-transpose without allocating a huge model.
                patch.setattr(accelerator, "available_memory", lambda: available_bytes)
                patch.setattr("deepspeed.utils.logger.info", lambda msg, *args, **kwargs: messages.append(str(msg)))
                engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
            expected_log = "on CPU (insufficient memory)" if expect_cpu else "(sufficient memory)"
            assert any("Flattening param group" in msg and expected_log in msg for msg in messages)
            for name, param in engine.module.named_parameters():
                assert torch.equal(param.detach().cpu(), expected[name])
                assert param.dtype == _DTYPE_MAP[dtype]
                assert not hasattr(param, "cpu_data")
            for flat in engine.optimizer.bit16_groups_flat:
                assert flat.device.type == accelerator.device_name()
                assert flat.grad_fn is None
            data_loader = random_dataloader(model=engine,
                                            total_samples=16,
                                            hidden_dim=64,
                                            device=engine.device,
                                            dtype=_DTYPE_MAP[dtype])
            for batch in data_loader:
                loss = engine(batch[0], batch[1])
                assert torch.isfinite(loss)
                engine.backward(loss)
                engine.step()
            assert any(not torch.equal(param.detach().cpu(), expected[name])
                       for name, param in engine.module.named_parameters())
        finally:
            if engine is not None:
                engine.destroy()
