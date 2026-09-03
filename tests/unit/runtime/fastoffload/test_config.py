# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import json

import pytest
from pydantic import ValidationError

from deepspeed.runtime.fastoffload.config import (FastOffloadConfig, FastOffloadMode, ObserverFailurePolicy,
                                                  TelemetryConfig, TelemetryRankMode)


def test_default_config_is_disabled_observer():
    config = FastOffloadConfig()

    assert config.enabled is False
    assert config.mode == FastOffloadMode.observe
    assert config.zero_stage == 2
    assert config.failure_policy == ObserverFailurePolicy.raise_error
    assert config.telemetry == TelemetryConfig()


def test_config_accepts_observer_settings():
    config = FastOffloadConfig.from_dict({
        "enabled": True,
        "mode": "observe",
        "zero_stage": 2,
        "failure_policy": "warn",
        "telemetry": {
            "log_interval": 5,
            "parameter_events": False,
            "bucket_events": True,
            "host_timing": True,
            "device_timing": True,
            "memory_metrics": False,
            "distributed_summary": False,
            "rank_mode": "per_rank",
            "debug_event_buffer_size": 64,
            "max_parameter_name_length": 80,
        },
    })

    assert config.enabled is True
    assert config.failure_policy == ObserverFailurePolicy.warn
    assert config.telemetry.log_interval == 5
    assert config.telemetry.parameter_events is False
    assert config.telemetry.device_timing is True
    assert config.telemetry.rank_mode == TelemetryRankMode.per_rank
    assert config.telemetry.debug_event_buffer_size == 64


def test_config_accepts_synchronous_offload_settings():
    config = FastOffloadConfig.from_dict({
        "enabled": True,
        "mode": "sync_offload",
        "policy": {
            "type": "all_offload"
        },
        "scheduler": {
            "type": "synchronous",
            "max_inflight_tasks": 1
        },
        "transfer": {
            "pin_memory": True,
            "buffer_count": 2,
            "buffer_size": 4096
        },
        "worker": {
            "type": "inline"
        },
    })

    assert config.mode == FastOffloadMode.sync_offload
    assert config.transfer.cpu_staging is False
    assert config.transfer.buffer_count == 2
    assert config.transfer.buffer_size == 4096


def test_synchronous_offload_requires_enabled_config():
    with pytest.raises(ValidationError, match="enabled=true"):
        FastOffloadConfig(mode="sync_offload")


def test_config_accepts_asynchronous_offload_settings():
    config = FastOffloadConfig.from_dict({
        "enabled": True,
        "mode": "async_offload",
        "scheduler": {
            "type": "overlap",
            "max_inflight_tasks": 4,
            "max_inflight_bytes": 8192
        },
        "transfer": {
            "buffer_count": 4,
            "buffer_size": 4096
        },
    })

    assert config.mode == FastOffloadMode.async_offload
    assert config.transfer.async_strategy == "producer_stream"
    assert config.scheduler.max_inflight_tasks == 4
    assert config.scheduler.max_inflight_bytes == 8192


@pytest.mark.parametrize("payload", [{
    "enabled": True,
    "mode": "async_offload"
}, {
    "enabled": True,
    "mode": "async_offload",
    "scheduler": {
        "type": "overlap",
        "max_inflight_tasks": 2
    },
    "transfer": {
        "async_strategy": "dedicated_stream",
        "buffer_count": 1
    }
}])
def test_invalid_asynchronous_settings_are_rejected(payload):
    with pytest.raises(ValidationError):
        FastOffloadConfig.from_dict(payload)


def test_config_accepts_importance_selection_settings():
    config = FastOffloadConfig.from_dict({
        "enabled": True,
        "importance": {
            "enabled": True,
            "algorithm": "pretrained_delta_topk",
            "warmup_steps": 4,
            "topk_ratio": 0.2,
            "comparison_chunk_rows": 32,
            "sparse_backward": True,
            "sparse_backward_min_tokens": 512,
            "output_path": "/tmp/importance.pt"
        }
    })

    assert config.importance.enabled is True
    assert config.importance.warmup_steps == 4
    assert config.importance.topk_ratio == 0.2
    assert config.importance.sparse_backward is True
    assert config.importance.sparse_backward_min_tokens == 512


def test_importance_selection_requires_enabled_fastoffload():
    with pytest.raises(ValidationError, match="Importance selection requires enabled=true"):
        FastOffloadConfig.from_dict({"importance": {"enabled": True}})


def test_config_accepts_hybrid_update_settings():
    config = FastOffloadConfig.from_dict({
        "enabled": True,
        "importance": {
            "enabled": True
        },
        "hybrid_update": {
            "enabled": True,
            "update_interval": 4,
            "accumulation_device": "gpu",
            "second_gradient_reduction": "sum",
            "max_async_lag": 2,
            "compressed_collective_shadow": True,
            "compressed_bucket_bytes": 4096,
            "parity_atol": 0.01,
            "parity_rtol": 0.02
        }
    })

    assert config.hybrid_update.enabled is True
    assert config.hybrid_update.update_interval == 4
    assert config.hybrid_update.accumulation_device == "gpu"
    assert config.hybrid_update.second_gradient_reduction == "sum"
    assert config.hybrid_update.compressed_collective_shadow is True
    assert config.hybrid_update.compressed_bucket_bytes == 4096
    assert config.hybrid_update.parity_atol == 0.01
    assert config.hybrid_update.parity_rtol == 0.02


def test_config_accepts_zero2_takeover_and_rejects_shadow_overlap():
    config = FastOffloadConfig.from_dict({
        "enabled": True,
        "importance": {
            "enabled": True
        },
        "hybrid_update": {
            "enabled": True,
            "zero2_takeover": True
        }
    })
    assert config.hybrid_update.zero2_takeover is True

    with pytest.raises(ValidationError, match="cannot be enabled together"):
        FastOffloadConfig.from_dict({
            "enabled": True,
            "importance": {
                "enabled": True
            },
            "hybrid_update": {
                "enabled": True,
                "compressed_collective_shadow": True,
                "zero2_takeover": True
            }
        })


def test_hybrid_update_requires_importance_selection():
    with pytest.raises(ValidationError, match="Hybrid update requires importance selection"):
        FastOffloadConfig.from_dict({"enabled": True, "hybrid_update": {"enabled": True}})


def test_nested_telemetry_defaults_are_applied():
    config = FastOffloadConfig.from_dict({"enabled": True, "telemetry": {"log_interval": 3}})

    assert config.telemetry.log_interval == 3
    assert config.telemetry.parameter_events is True
    assert config.telemetry.device_timing is False
    assert config.telemetry.rank_mode == TelemetryRankMode.rank0_local


def test_config_loads_from_json(tmp_path):
    config_path = tmp_path / "fastoffload.json"
    config_path.write_text(json.dumps({
        "enabled": True,
        "telemetry": {
            "log_interval": 7,
        },
    }), encoding="utf-8")

    config = FastOffloadConfig.from_json(config_path)

    assert config.enabled is True
    assert config.telemetry.log_interval == 7


@pytest.mark.parametrize("zero_stage", [0, 1, 3, "2"])
def test_only_zero_stage_two_is_supported(zero_stage):
    with pytest.raises(ValidationError):
        FastOffloadConfig(zero_stage=zero_stage)


@pytest.mark.parametrize("field", ["log_interval", "max_parameter_name_length"])
def test_positive_telemetry_fields_reject_zero(field):
    with pytest.raises(ValidationError):
        TelemetryConfig(**{field: 0})


@pytest.mark.parametrize("payload", [{"buffer_count": 0}, {"buffer_size": 0}, {"pin_memory": False}])
def test_invalid_synchronous_transfer_settings_are_rejected(payload):
    with pytest.raises(ValidationError):
        FastOffloadConfig(enabled=True, mode="sync_offload", transfer=payload)


def test_debug_event_buffer_size_rejects_negative_value():
    with pytest.raises(ValidationError):
        TelemetryConfig(debug_event_buffer_size=-1)


@pytest.mark.parametrize("payload", [{"unknown": True}, {"telemetry": {"unknown": True}}])
def test_unknown_fields_are_rejected(payload):
    with pytest.raises(ValidationError):
        FastOffloadConfig.from_dict(payload)


def test_invalid_enum_values_are_rejected():
    with pytest.raises(ValidationError):
        FastOffloadConfig(mode="run")

    with pytest.raises(ValidationError):
        TelemetryConfig(rank_mode="all_reduce")


def test_config_is_immutable():
    config = FastOffloadConfig()

    with pytest.raises(ValidationError):
        config.enabled = True


def test_missing_json_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        FastOffloadConfig.from_json(tmp_path / "missing.json")


def test_malformed_json_is_reported(tmp_path):
    config_path = tmp_path / "fastoffload.json"
    config_path.write_text("{", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        FastOffloadConfig.from_json(config_path)
