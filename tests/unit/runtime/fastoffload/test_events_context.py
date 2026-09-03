# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from dataclasses import FrozenInstanceError, asdict, fields

import pytest

from deepspeed.runtime.fastoffload.context import GradientBucketContext, GradientContext, StepContext
from deepspeed.runtime.fastoffload.events import FastOffloadEvent


def make_step_context(**overrides):
    values = {
        "global_step": 3,
        "micro_step": 1,
        "rank": 0,
        "world_size": 2,
        "zero_stage": 2,
        "gradient_accumulation_boundary": True,
        "timestamp_ns": 100,
    }
    values.update(overrides)
    return StepContext(**values)


def make_gradient_context(**overrides):
    values = {
        "parameter_id": 7,
        "parameter_name": "decoder.layer.weight",
        "group_id": 0,
        "partition_id": 1,
        "parameter_numel": 16,
        "shard_numel": 8,
        "shard_offset": 8,
        "element_size": 2,
        "shard_bytes": 16,
        "dtype": "torch.float16",
        "device": "cuda:0",
        "global_step": 3,
        "micro_step": 1,
        "rank": 1,
        "world_size": 2,
        "zero_stage": 2,
        "gradient_accumulation_boundary": True,
        "is_local_partition": True,
        "is_cpu_offload": True,
        "timestamp_ns": 100,
    }
    values.update(overrides)
    return GradientContext(**values)


def make_bucket_context(**overrides):
    values = {
        "bucket_id": 4,
        "parameter_count": 3,
        "total_numel": 32,
        "total_bytes": 64,
        "communication_dtype": "torch.float16",
        "overlap_comm": True,
        "global_step": 3,
        "micro_step": 1,
        "rank": 0,
        "world_size": 2,
        "zero_stage": 2,
        "gradient_accumulation_boundary": True,
        "timestamp_ns": 100,
    }
    values.update(overrides)
    return GradientBucketContext(**values)


def test_all_lifecycle_event_values_are_stable():
    assert [event.value for event in FastOffloadEvent] == [
        "backward_begin",
        "gradient_ready",
        "gradient_reduced",
        "backward_end",
        "step_begin",
        "step_end",
        "report",
        "shutdown",
    ]


def test_events_compare_with_serialized_strings():
    assert FastOffloadEvent.gradient_reduced == "gradient_reduced"
    assert FastOffloadEvent("step_end") is FastOffloadEvent.step_end


def test_step_context_is_immutable_and_serializable():
    context = make_step_context()

    assert asdict(context)["global_step"] == 3
    with pytest.raises(FrozenInstanceError):
        context.global_step = 4


def test_gradient_context_contains_only_metadata():
    context = make_gradient_context()
    values = asdict(context)

    assert values["shard_bytes"] == 16
    assert values["parameter_name"] == "decoder.layer.weight"
    assert all(isinstance(value, (bool, int, str)) for value in values.values())
    assert all("Tensor" not in str(field.type) for field in fields(context))


def test_bucket_context_accepts_empty_bucket():
    context = make_bucket_context(parameter_count=0, total_numel=0, total_bytes=0)

    assert context.parameter_count == 0
    assert context.total_bytes == 0


@pytest.mark.parametrize("rank,world_size", [(-1, 2), (2, 2), (0, 0)])
def test_invalid_rank_metadata_is_rejected(rank, world_size):
    with pytest.raises((TypeError, ValueError)):
        make_step_context(rank=rank, world_size=world_size)


@pytest.mark.parametrize("field", ["global_step", "micro_step", "timestamp_ns"])
def test_negative_step_metadata_is_rejected(field):
    with pytest.raises(ValueError):
        make_step_context(**{field: -1})


def test_only_zero_stage_two_is_supported():
    with pytest.raises(ValueError):
        make_step_context(zero_stage=3)


def test_bool_is_not_accepted_as_an_integer():
    with pytest.raises(TypeError):
        make_step_context(global_step=True)


def test_gradient_shard_must_fit_parameter_bounds():
    with pytest.raises(ValueError, match="exceeds"):
        make_gradient_context(shard_offset=12, shard_numel=8, shard_bytes=16)


def test_gradient_shard_bytes_must_match_shape_metadata():
    with pytest.raises(ValueError, match="shard_bytes"):
        make_gradient_context(shard_bytes=8)


@pytest.mark.parametrize("field", ["parameter_name", "dtype", "device"])
def test_gradient_strings_must_be_non_empty(field):
    with pytest.raises(ValueError):
        make_gradient_context(**{field: " "})


def test_gradient_partition_must_be_in_world():
    with pytest.raises(ValueError, match="partition_id"):
        make_gradient_context(partition_id=2)


@pytest.mark.parametrize("field", ["gradient_accumulation_boundary", "is_local_partition", "is_cpu_offload"])
def test_gradient_flags_must_be_boolean(field):
    with pytest.raises(TypeError):
        make_gradient_context(**{field: 1})


def test_bucket_dtype_must_be_non_empty():
    with pytest.raises(ValueError):
        make_bucket_context(communication_dtype="")
