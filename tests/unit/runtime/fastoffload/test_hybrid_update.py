# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import threading

import pytest
import torch

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.fastoffload.hybrid import (
    CompressedColumnGradient, CompressedGradientCollective, CompressedMicrobatchAccumulator,
    DoubleBufferedGradientAccumulator, HybridBufferState, HybridColumnLayout, HybridCompressedCollectiveShadow,
    HybridGradientNumerics, HybridUpdateCoordinator, HybridUpdateRuntime, OwnerPartitionUpdate,
    PackedCompressedGradients, ParameterPartitionLayout, SelectedColumnAdamW, Zero2CompressedGradientReducer,
    Zero2OwnerCollective, Zero2OwnerCpuUpdater, Zero2OwnerGpuUpdater, Zero2OwnerPartitionCommitter,
    Zero2TakeoverGradientPipeline, Zero2TakeoverRuntime)
from deepspeed.runtime.fastoffload.importance import ImportanceRegistry
from deepspeed.runtime.fastoffload.importance.context import ColumnImportanceSelection, ParameterImportanceView
from deepspeed.runtime.fastoffload.telemetry.metrics import MetricsRegistry
from unit.common import DistributedTest


def test_hybrid_column_layout_packs_disjoint_groups_and_scatters_values():
    layout = HybridColumnLayout(parameter_id=3,
                                column_count=5,
                                first_columns=torch.tensor([3, 1]),
                                second_columns=torch.tensor([4]))
    gradient = torch.arange(10, dtype=torch.float32).view(2, 5)

    first = layout.pack_first(gradient)
    second = layout.pack_second(gradient)
    dense = layout.pack_dense_remainder(gradient)

    assert first.columns.tolist() == [1, 3]
    assert torch.equal(first.values, gradient[:, [1, 3]])
    assert second.columns.tolist() == [4]
    assert dense.columns.tolist() == [0, 2]
    parameter = torch.zeros_like(gradient)
    layout.scatter_values(parameter, first)
    assert torch.equal(parameter[:, [1, 3]], gradient[:, [1, 3]])
    assert torch.count_nonzero(parameter[:, [0, 2, 4]]) == 0


def test_partition_layout_maps_selected_values_to_zero_owners():
    layout = HybridColumnLayout(parameter_id=7,
                                column_count=4,
                                first_columns=torch.tensor([3, 1]),
                                second_columns=torch.tensor([], dtype=torch.long))
    gradient = layout.pack_first(torch.arange(8, dtype=torch.float32).view(2, 4))
    partition = ParameterPartitionLayout(parameter_id=7,
                                         group_id=2,
                                         shape=(2, 4),
                                         group_offset=6,
                                         partition_size=5,
                                         world_size=3)

    owned = partition.map_gradient(gradient)

    assert owned.owner_ranks.tolist() == [1, 1, 2, 2]
    assert owned.local_offsets.tolist() == [2, 4, 1, 3]
    assert partition.owner_counts(gradient.columns) == (0, 2, 2)
    assert partition.local_offsets(gradient.columns, rank=2).tolist() == [1, 3]
    assert partition.extract_local_values(torch.arange(5), gradient.columns, rank=2).tolist() == [1, 3]
    fragment = torch.arange(4)
    assert partition.extract_fragment_values(fragment, gradient.columns, rank=2).tolist() == [1, 3]
    offsets, values = owned.for_rank(2)
    assert offsets.tolist() == [1, 3]
    assert values.tolist() == [5.0, 7.0]
    destination = torch.zeros(5)
    assert owned.scatter_to_partition(destination, rank=2) == 2
    assert destination.tolist() == [0.0, 5.0, 0.0, 7.0, 0.0]


def test_packed_collective_reconstructs_compressed_gradients_without_dense_padding():
    first_layout = HybridColumnLayout(1, 3, torch.tensor([0, 2]), torch.tensor([], dtype=torch.long))
    second_layout = HybridColumnLayout(2, 2, torch.tensor([1]), torch.tensor([], dtype=torch.long))
    gradients = (first_layout.pack_first(torch.arange(6, dtype=torch.float32).view(2, 3)),
                 second_layout.pack_first(torch.arange(4, dtype=torch.float32).view(2, 2)))

    packed = PackedCompressedGradients(gradients)
    reduced = CompressedGradientCollective.all_reduce(gradients)

    assert packed.buffer.numel() == 6
    assert [item.parameter_id for item in reduced] == [1, 2]
    assert torch.equal(reduced[0].values, gradients[0].values)
    assert torch.equal(reduced[1].values, gradients[1].values)


def test_zero2_reducer_batches_gradients_by_partition_group():
    layout = HybridColumnLayout(1, 3, torch.tensor([0, 2]), torch.tensor([], dtype=torch.long))
    gradient = layout.pack_first(torch.arange(6, dtype=torch.float32).view(2, 3))
    partition = ParameterPartitionLayout(parameter_id=1,
                                         group_id=0,
                                         shape=(2, 3),
                                         group_offset=0,
                                         partition_size=6,
                                         world_size=1)
    reducer = Zero2CompressedGradientReducer([partition], {0: None})

    reduced = reducer.reduce([gradient])

    assert torch.equal(reduced[1].values, gradient.values)


def test_hybrid_shadow_switches_from_selected_to_dense_boundary_columns():
    parameter = torch.nn.Parameter(torch.zeros((2, 4)))
    parameter.grad = torch.arange(8, dtype=torch.float32).view(2, 4)
    registry = ImportanceRegistry()
    registry.add(
        ColumnImportanceSelection(parameter_id=1,
                                  parameter_name="weight",
                                  shape=(2, 4),
                                  first_indices=torch.tensor([1]),
                                  second_indices=torch.tensor([3]),
                                  first_min_score=1.0,
                                  second_min_score=0.5,
                                  algorithm="test"))
    registry.finalize()

    class IdentityReducer:

        def __init__(self):
            self.column_history = []

        def reduce(self, gradients):
            gradients = tuple(gradients)
            self.column_history.append(gradients[0].columns.tolist())
            return {gradient.parameter_id: gradient for gradient in gradients}

    reducer = IdentityReducer()
    partition_layout = ParameterPartitionLayout(parameter_id=1,
                                                group_id=0,
                                                shape=(2, 4),
                                                group_offset=0,
                                                partition_size=8,
                                                world_size=1)

    class Adapter:

        @staticmethod
        def create_compressed_gradient_reducer():
            return reducer

        @staticmethod
        def iter_parameter_partition_layouts():
            return (partition_layout, )

        @staticmethod
        def get_partition_rank(_group_id):
            return 0

        @staticmethod
        def get_parameter_id(_parameter):
            return 1

        @staticmethod
        def get_gradient_tensor(_parameter):
            return _parameter.grad

    shadow = HybridCompressedCollectiveShadow(Adapter(),
                                              registry,
                                              MetricsRegistry(),
                                              update_interval=2,
                                              bucket_bytes=1024,
                                              parity_atol=1e-6,
                                              parity_rtol=1e-6)
    shadow.capture(parameter, group_id=0)
    shadow.validate_native_gradient(parameter, group_id=0)
    shadow.reduce()
    assert reducer.column_history[-1] == [1, 3]

    shadow.complete_step()
    shadow.capture(parameter, group_id=0)
    shadow.validate_native_gradient(parameter, group_id=0)
    shadow.reduce()
    assert reducer.column_history[-1] == [0, 1, 2, 3]


def test_hybrid_shadow_chunks_buffers_and_validates_owner_partition():
    parameter = torch.nn.Parameter(torch.zeros((2, 4)))
    parameter.grad = torch.arange(8, dtype=torch.float32).view(2, 4)
    registry = ImportanceRegistry()
    registry.add(
        ColumnImportanceSelection(parameter_id=1,
                                  parameter_name="weight",
                                  shape=(2, 4),
                                  first_indices=torch.tensor([1]),
                                  second_indices=torch.tensor([3]),
                                  first_min_score=1.0,
                                  second_min_score=0.5,
                                  algorithm="test"))
    registry.finalize()
    partition_layout = ParameterPartitionLayout(parameter_id=1,
                                                group_id=0,
                                                shape=(2, 4),
                                                group_offset=0,
                                                partition_size=8,
                                                world_size=1)

    class IdentityReducer:

        @staticmethod
        def reduce(gradients):
            return {gradient.parameter_id: gradient for gradient in gradients}

    class Adapter:

        @staticmethod
        def create_compressed_gradient_reducer():
            return IdentityReducer()

        @staticmethod
        def iter_parameter_partition_layouts():
            return (partition_layout, )

        @staticmethod
        def get_parameter_id(_parameter):
            return 1

        @staticmethod
        def get_gradient_tensor(_parameter):
            return _parameter.grad

        @staticmethod
        def get_partition_rank(_group_id):
            return 0

        @staticmethod
        def synchronize_gradient_transfers():
            pass

        @staticmethod
        def get_fp32_gradient_partition(_group_id):
            return parameter.grad.view(-1)

    metrics = MetricsRegistry()
    shadow = HybridCompressedCollectiveShadow(Adapter(),
                                              registry,
                                              metrics,
                                              update_interval=2,
                                              bucket_bytes=1,
                                              parity_atol=1e-6,
                                              parity_rtol=1e-6)
    shadow.capture(parameter, group_id=0)
    shadow.validate_native_gradient(parameter, group_id=0)
    shadow.validate_owner_partition(parameter, group_id=0)
    shadow.reduce()

    snapshot = metrics.snapshot()
    assert snapshot.counters["hybrid_shadow_bucket_count"] == 1
    assert snapshot.counters["hybrid_shadow_parity_check_count"] == 1
    assert snapshot.counters["hybrid_shadow_norm_check_count"] == 1
    assert snapshot.counters["hybrid_shadow_overflow_check_count"] == 1
    assert snapshot.counters["hybrid_shadow_owner_write_check_count"] == 1
    assert snapshot.counters["hybrid_shadow_owner_write_element_count"] == 4


def test_hybrid_shadow_rejects_compressed_native_mismatch():
    parameter = torch.nn.Parameter(torch.zeros((1, 2)))
    parameter.grad = torch.tensor([[1.0, 2.0]])
    registry = ImportanceRegistry()
    registry.add(
        ColumnImportanceSelection(parameter_id=1,
                                  parameter_name="weight",
                                  shape=(1, 2),
                                  first_indices=torch.tensor([0]),
                                  second_indices=torch.tensor([1]),
                                  first_min_score=1.0,
                                  second_min_score=0.5,
                                  algorithm="test"))
    registry.finalize()

    partition_layout = ParameterPartitionLayout(parameter_id=1,
                                                group_id=0,
                                                shape=(1, 2),
                                                group_offset=0,
                                                partition_size=2,
                                                world_size=1)

    class Adapter:

        @staticmethod
        def create_compressed_gradient_reducer():
            return type("Reducer", (), {"reduce": lambda self, gradients: {1: tuple(gradients)[0]}})()

        @staticmethod
        def iter_parameter_partition_layouts():
            return (partition_layout, )

        @staticmethod
        def get_partition_rank(_group_id):
            return 0

        @staticmethod
        def get_parameter_id(_parameter):
            return 1

        @staticmethod
        def get_gradient_tensor(_parameter):
            return _parameter.grad

    shadow = HybridCompressedCollectiveShadow(Adapter(),
                                              registry,
                                              MetricsRegistry(),
                                              update_interval=2,
                                              bucket_bytes=1,
                                              parity_atol=0.0,
                                              parity_rtol=0.0)
    shadow.capture(parameter, group_id=0)
    parameter.grad.add_(1.0)
    with pytest.raises(RuntimeError, match="Compressed collective mismatch"):
        shadow.validate_native_gradient(parameter, group_id=0)


def test_compressed_microbatch_accumulator_handles_gas_and_unused_parameters():
    accumulator = CompressedMicrobatchAccumulator(gradient_accumulation_steps=2)

    assert accumulator.accumulate({1: torch.tensor([2.0]), 2: torch.tensor([4.0])}) is False
    assert accumulator.accumulate({1: torch.tensor([6.0])}) is True
    boundary = accumulator.take_boundary()

    assert torch.equal(boundary[1], torch.tensor([4.0]))
    assert torch.equal(boundary[2], torch.tensor([2.0]))
    assert accumulator.micro_steps == 0


def test_takeover_pipeline_routes_gas_selected_and_dense_boundary_gradients():
    matrix = torch.nn.Parameter(torch.zeros((2, 3)))
    bias = torch.nn.Parameter(torch.zeros(2))
    parameters = {id(matrix): 1, id(bias): 2}
    layouts = (ParameterPartitionLayout(1, 0, (2, 3), 0, 8, 1), ParameterPartitionLayout(2, 0, (1, 2), 6, 8, 1))
    registry = ImportanceRegistry()
    registry.add(
        ColumnImportanceSelection(parameter_id=1,
                                  parameter_name="weight",
                                  shape=(2, 3),
                                  first_indices=torch.tensor([0]),
                                  second_indices=torch.tensor([1]),
                                  first_min_score=1.0,
                                  second_min_score=0.5,
                                  algorithm="test"))
    registry.add_dense(ParameterImportanceView(2, "bias", bias))
    registry.finalize()

    class Adapter:

        @staticmethod
        def iter_parameter_partition_layouts():
            return layouts

        @staticmethod
        def create_owner_collectives():
            return {0: Zero2OwnerCollective(layouts)}

        @staticmethod
        def get_parameter_id(parameter):
            return parameters[id(parameter)]

        @staticmethod
        def get_gradient_tensor(parameter):
            return parameter.grad

    pipeline = Zero2TakeoverGradientPipeline(Adapter(), registry, update_interval=2, gradient_accumulation_steps=2)
    for micro_step in range(2):
        matrix.grad = torch.full_like(matrix, float(micro_step + 1))
        bias.grad = torch.ones_like(bias)
        assert pipeline.capture(matrix, 0)
        assert pipeline.capture(bias, 0)
        batch = pipeline.finish_microbatch()
    assert batch is not None
    assert batch.dense_boundary is False
    assert torch.equal(batch.first[0].values[1], torch.full((2, ), 3.0))
    assert batch.dense == {}

    pipeline.complete_step(overflow=False)
    for _ in range(2):
        matrix.grad = torch.full_like(matrix, 4.0)
        bias.grad = torch.full_like(bias, 6.0)
        pipeline.capture(matrix, 0)
        pipeline.capture(bias, 0)
        batch = pipeline.finish_microbatch()
    assert batch is not None
    assert batch.dense_boundary is True
    assert torch.equal(batch.dense[0].values[1], torch.full((2, ), 8.0))
    assert torch.equal(batch.dense[0].values[2], torch.full((2, ), 12.0))


def test_takeover_runtime_updates_first_each_step_and_cpu_groups_at_boundary():
    parameter = torch.nn.Parameter(torch.ones((2, 3)))
    fp32_partition = parameter.detach().view(-1).clone()
    layout = ParameterPartitionLayout(1, 0, (2, 3), 0, 6, 1)
    collective = Zero2OwnerCollective([layout])
    registry = ImportanceRegistry()
    registry.add(
        ColumnImportanceSelection(parameter_id=1,
                                  parameter_name="weight",
                                  shape=(2, 3),
                                  first_indices=torch.tensor([0]),
                                  second_indices=torch.tensor([1]),
                                  first_min_score=1.0,
                                  second_min_score=0.5,
                                  algorithm="test"))
    registry.finalize()

    class Adapter:

        @staticmethod
        def iter_parameter_partition_layouts():
            return (layout, )

        @staticmethod
        def create_owner_collectives():
            return {0: collective}

        @staticmethod
        def get_parameter_id(_parameter):
            return 1

        @staticmethod
        def get_gradient_tensor(_parameter):
            return parameter.grad

        @staticmethod
        def get_partition_rank(_group_id):
            return 0

        @staticmethod
        def get_data_parallel_group():
            return None

        @staticmethod
        def get_dense_adam_partition_state(_group_id):
            zeros = torch.zeros_like(fp32_partition)
            return 0, zeros, zeros

        @staticmethod
        def get_fp32_partition(_group_id):
            return fp32_partition

        @staticmethod
        def read_fp32_partition_values(_group_id, offsets, device):
            return fp32_partition.index_select(0, offsets).to(device)

        @staticmethod
        def write_fp32_partition_values(_group_id, offsets, values):
            fp32_partition.index_copy_(0, offsets, values.cpu())

        @staticmethod
        def scatter_parameter_columns(_parameter_id, columns, values):
            parameter.data.index_copy_(1, columns, values)

        @staticmethod
        def all_ranks_ready(ready):
            return ready

    runtime = Zero2TakeoverRuntime(Adapter(), registry, 2, 1, "cpu", "mean", 0.1, (0.0, 0.0), 1e-8, 0.0)
    for step in range(2):
        parameter.grad = torch.ones_like(parameter)
        assert runtime.capture_gradient(parameter, 0)
        batch = runtime.finish_microbatch()
        assert batch is not None
        result = runtime.step(batch, loss_scale=1.0, clip_grad=0.0)
        assert result.numerics.overflow is False
        assert result.first_updated_elements == 2
        if step == 0:
            assert torch.allclose(parameter[:, 0], torch.full((2, ), 0.9))
            assert torch.equal(parameter[:, 1:], torch.ones((2, 2)))
    runtime.close()

    assert torch.allclose(parameter[:, 0], torch.full((2, ), 0.8))
    assert torch.allclose(parameter[:, 1:], torch.full((2, 2), 0.9))


def test_owner_cpu_updater_accumulates_second_and_commits_dense_values():
    parameter = torch.ones((2, 3))
    fp32_partition = parameter.view(-1).clone()
    layout = ParameterPartitionLayout(1, 0, (2, 3), 0, 6, 1)
    collective = Zero2OwnerCollective([layout])
    second_columns = {1: torch.tensor([1])}
    dense_columns = {1: torch.tensor([2])}

    def reduce(columns, value):
        gradient = CompressedColumnGradient(1, columns[1], torch.full((2, 1), value))
        return {0: collective.reduce_scatter([gradient])}

    class Adapter:

        @staticmethod
        def get_partition_rank(_group_id):
            return 0

        @staticmethod
        def get_fp32_partition(_group_id):
            return fp32_partition

        @staticmethod
        def read_fp32_partition_values(_group_id, offsets, device):
            return fp32_partition.index_select(0, offsets).to(device)

        @staticmethod
        def write_fp32_partition_values(_group_id, offsets, values):
            fp32_partition.index_copy_(0, offsets, values.cpu())

        @staticmethod
        def scatter_parameter_columns(_parameter_id, columns, values):
            parameter.index_copy_(1, columns, values)

        @staticmethod
        def all_ranks_ready(ready):
            return ready

    updater = Zero2OwnerCpuUpdater(Adapter(), [layout], {0: collective}, 2, "cpu", "mean", 0.1, (0.0, 0.0), 1e-8, 0.0)
    updater.accumulate_second(reduce(second_columns, 1.0))
    updater.submit_boundary(reduce(second_columns, 3.0), reduce(dense_columns, 2.0), second_columns, dense_columns)

    checkpoint = updater.state_dict()
    assert checkpoint["committed_version"] == 1
    assert updater.pending_updates == 0
    updater.close()

    assert torch.equal(parameter[:, 0], torch.ones(2))
    assert torch.allclose(parameter[:, 1:], torch.full((2, 2), 0.9))
    assert torch.equal(fp32_partition[[1, 2, 4, 5]], torch.ones(4))


def test_owner_gpu_updater_updates_and_publishes_first_columns():
    parameter = torch.ones((2, 2))
    fp32_partition = parameter.view(-1).clone()
    layout = ParameterPartitionLayout(1, 0, (2, 2), 0, 4, 1)
    collective = Zero2OwnerCollective([layout])
    columns = {1: torch.tensor([0])}
    gradient = CompressedColumnGradient(1, columns[1], torch.ones((2, 1)))
    reduced = collective.reduce_scatter([gradient])

    class Adapter:

        @staticmethod
        def get_partition_rank(_group_id):
            return 0

        @staticmethod
        def get_fp32_partition(_group_id):
            return fp32_partition

        @staticmethod
        def read_fp32_partition_values(_group_id, offsets, device):
            return fp32_partition.index_select(0, offsets).to(device)

        @staticmethod
        def write_fp32_partition_values(_group_id, offsets, values):
            fp32_partition.index_copy_(0, offsets, values.cpu())

        @staticmethod
        def scatter_parameter_columns(_parameter_id, selected_columns, values):
            parameter.index_copy_(1, selected_columns, values)

    updater = Zero2OwnerGpuUpdater(Adapter(), [layout], {0: collective}, 0.1, (0.0, 0.0), 1e-8, 0.0)
    zeros = torch.zeros_like(fp32_partition)
    updater.initialize_state(1, columns[1], 0, zeros, zeros, torch.device("cpu"))

    updated = updater.step({0: reduced}, columns)

    assert updated == 2
    assert torch.allclose(parameter[:, 0], torch.full((2, ), 0.9))
    assert torch.equal(parameter[:, 1], torch.ones(2))
    assert torch.equal(fp32_partition[[0, 2]], torch.ones(2))


def test_owner_collective_buckets_parameters_without_splitting_large_parameter():
    layouts = (ParameterPartitionLayout(3, 0, (1, 2), 0, 8, 1), ParameterPartitionLayout(1, 0, (1, 2), 2, 8, 1),
               ParameterPartitionLayout(2, 0, (1, 4), 4, 8, 1))
    collective = Zero2OwnerCollective(layouts)
    gradients = [
        CompressedColumnGradient(layout.parameter_id, torch.arange(layout.shape[1]),
                                 torch.full(layout.shape, float(layout.parameter_id))) for layout in layouts
    ]

    reduced = collective.reduce_scatter_buckets(gradients, bucket_bytes=8)

    assert [bucket.parameter_ids for bucket in reduced.buckets] == [(1, ), (2, ), (3, )]
    assert reduced.buckets[1].padded_numel == 4
    assert reduced.bucket_count == 3
    assert reduced.peak_packed_bytes == 16
    assert reduced.total_packed_bytes == 32
    assert set(reduced.values) == {1, 2, 3}


def test_owner_metadata_buckets_use_global_payload_instead_of_rank_local_values():
    layouts = (ParameterPartitionLayout(1, 0, (1, 4), 0, 4, 2), ParameterPartitionLayout(2, 0, (1, 4), 4, 4, 2))
    collective = Zero2OwnerCollective(layouts)
    columns = {1: torch.arange(4), 2: torch.arange(4)}
    rank_zero_values = {1: torch.ones(4), 2: torch.empty(0)}

    reduced = collective.create_owner_gradient_set(columns, rank_zero_values, bucket_bytes=8)

    assert [bucket.parameter_ids for bucket in reduced.buckets] == [(1, ), (2, )]


def test_owner_partition_committer_writes_local_offsets_and_publishes_version():
    partition = torch.zeros(5)

    class Adapter:

        @staticmethod
        def get_partition_rank(_group_id):
            return 2

        @staticmethod
        def write_fp32_partition_values(_group_id, offsets, values):
            partition.index_copy_(0, offsets, values)

        def __init__(self):
            self.published = []

        def publish_fp32_partitions(self, group_ids):
            self.published.append(set(group_ids))

    adapter = Adapter()
    layout = ParameterPartitionLayout(parameter_id=7,
                                      group_id=2,
                                      shape=(2, 4),
                                      group_offset=6,
                                      partition_size=5,
                                      world_size=3)
    committer = Zero2OwnerPartitionCommitter(adapter, [layout])
    values = torch.arange(8, dtype=torch.float32).view(2, 4)[:, [1, 3]]
    committer.stage(OwnerPartitionUpdate(7, torch.tensor([1, 3]), values, version=1))

    written = committer.commit(expected_version=1)

    assert written == 2
    assert partition.tolist() == [0.0, 5.0, 0.0, 7.0, 0.0]
    assert adapter.published == [{2}]
    assert committer.committed_version == 1
    with pytest.raises(RuntimeError, match="stale"):
        committer.stage(OwnerPartitionUpdate(7, torch.tensor([1, 3]), values, version=1))


def test_hybrid_numerics_unscales_and_clips_owner_gradients():
    first = torch.tensor([6.0])
    second = torch.tensor([8.0])

    result = HybridGradientNumerics.unscale_and_clip([first, second], loss_scale=2.0, clip_grad=2.5)

    assert result.overflow is False
    assert result.global_norm == 5.0
    assert result.combined_scale == 4.0
    assert torch.equal(first, torch.tensor([1.5]))
    assert torch.equal(second, torch.tensor([2.0]))


def test_hybrid_numerics_preserves_nonfinite_gradients_for_overflow_rollback():
    gradient = torch.tensor([float("inf")])

    result = HybridGradientNumerics.unscale_and_clip([gradient], loss_scale=2.0, clip_grad=1.0)

    assert result.overflow is True
    assert torch.isinf(gradient).all()


def test_double_buffer_prevents_frozen_accumulator_reuse():
    accumulator = DoubleBufferedGradientAccumulator("cpu")
    accumulator.accumulate({1: torch.tensor([1.0, 2.0])})
    accumulator.accumulate({1: torch.tensor([3.0, 4.0])})

    frozen_id = accumulator.freeze_and_swap(version=1)

    assert accumulator.state(frozen_id) == HybridBufferState.frozen
    assert accumulator.accumulated_steps(frozen_id) == 2
    assert torch.equal(accumulator.frozen_gradients(frozen_id)[1], torch.tensor([4.0, 6.0]))
    assert accumulator.active_id != frozen_id
    accumulator.accumulate({1: torch.tensor([5.0, 6.0])})
    with pytest.raises(RuntimeError, match="No free hybrid accumulation buffer"):
        accumulator.freeze_and_swap(version=2)

    accumulator.begin_update(frozen_id)
    accumulator.mark_copying_to_gpu(frozen_id, {1: torch.tensor([7.0, 8.0])})
    accumulator.mark_ready(frozen_id)
    assert torch.equal(accumulator.take_ready_result(frozen_id)[1], torch.tensor([7.0, 8.0]))
    assert accumulator.state(frozen_id) == HybridBufferState.free


@pytest.mark.skipif(not get_accelerator().is_available(), reason="requires an accelerator")
def test_gpu_accumulation_keeps_active_gradient_on_gpu():
    accumulator = DoubleBufferedGradientAccumulator("gpu")
    device = get_accelerator().current_device_name()
    accumulator.accumulate({1: torch.tensor([1.0, 2.0], device=device)})

    frozen_id = accumulator.freeze_and_swap(version=1)

    assert accumulator.frozen_gradients(frozen_id)[1].device == torch.device(device)


def test_hybrid_overflow_discards_only_active_accumulation():
    accumulator = DoubleBufferedGradientAccumulator("cpu")
    accumulator.accumulate({1: torch.tensor([3.0])})

    accumulator.discard_active()
    frozen_id = accumulator.freeze_and_swap(version=1)

    assert accumulator.accumulated_steps(frozen_id) == 0
    assert torch.count_nonzero(accumulator.frozen_gradients(frozen_id)[1]) == 0


def test_hybrid_coordinator_accumulates_while_cpu_job_updates():
    worker_started = threading.Event()
    release_worker = threading.Event()
    committed = []

    def update(job):
        worker_started.set()
        assert release_worker.wait(timeout=5)
        assert torch.equal(job.second_gradients[1], torch.tensor([3.0]))
        assert torch.equal(job.dense_gradients[2], torch.tensor([5.0]))
        return {1: job.second_gradients[1] + 10.0, 2: job.dense_gradients[2] + 20.0}

    coordinator = HybridUpdateCoordinator(update_interval=2,
                                          accumulation_device="cpu",
                                          update_function=update,
                                          second_gradient_reduction="mean")
    coordinator.accumulate_second({1: torch.tensor([2.0])})
    assert coordinator.submit_boundary({1: torch.tensor([4.0])}, {2: torch.tensor([5.0])}) == 1
    assert worker_started.wait(timeout=5)

    coordinator.accumulate_second({1: torch.tensor([8.0])})
    assert coordinator.accumulator.accumulated_steps(coordinator.accumulator.active_id) == 1
    release_worker.set()
    coordinator.wait_and_commit_oldest(committed.append)

    assert len(committed) == 1
    assert committed[0].version == 1
    assert torch.equal(committed[0].updated_values[1], torch.tensor([13.0]))
    assert torch.equal(committed[0].updated_values[2], torch.tensor([25.0]))
    assert coordinator.committed_version == 1
    assert coordinator.pending_updates == 0
    coordinator.close()


def test_hybrid_coordinator_waits_for_transfer_in_worker_only():
    transfer_complete = threading.Event()

    class TransferEvent:

        @staticmethod
        def synchronize():
            transfer_complete.set()

    def prepare(second, dense, _steps, _reduction):
        return second, dense, TransferEvent()

    def update(job):
        assert transfer_complete.is_set()
        return {1: job.second_gradients[1]}

    coordinator = HybridUpdateCoordinator(update_interval=2,
                                          accumulation_device="cpu",
                                          update_function=update,
                                          prepare_function=prepare)
    coordinator.accumulate_second({1: torch.tensor([1.0])})
    coordinator.submit_boundary({1: torch.tensor([1.0])}, {})
    coordinator.wait_and_commit_oldest(lambda _result: None)
    coordinator.close()


def test_hybrid_runtime_updates_first_group_each_step_and_cpu_groups_at_boundary():
    parameter = torch.ones((2, 4), dtype=torch.float32)
    layout = HybridColumnLayout(parameter_id=1,
                                column_count=4,
                                first_columns=torch.tensor([0]),
                                second_columns=torch.tensor([1]))
    runtime = HybridUpdateRuntime(parameters={1: parameter},
                                  layouts={1: layout},
                                  update_interval=2,
                                  accumulation_device="cpu",
                                  lr=0.1,
                                  betas=(0.0, 0.0),
                                  eps=1e-8,
                                  weight_decay=0.0)

    runtime.step(first_gradients={1: torch.ones((2, 1))}, second_gradients={1: torch.full((2, 1), 2.0)})
    runtime.step(first_gradients={1: torch.ones((2, 1))},
                 second_gradients={1: torch.full((2, 1), 4.0)},
                 dense_gradients={1: torch.full((2, 2), 5.0)})
    runtime.close()

    assert torch.allclose(parameter[:, 0], torch.full((2, ), 0.8))
    assert torch.allclose(parameter[:, 1:], torch.full((2, 3), 0.9))
    assert runtime.committed_version == 1


class TestCompressedGradientCollective(DistributedTest):
    world_size = 2

    def test_owner_reduce_scatter_and_updated_value_all_gather(self):
        rank = dist.get_rank()
        device = get_accelerator().current_device_name()
        layout = HybridColumnLayout(1, 4, torch.tensor([1, 3]), torch.tensor([], dtype=torch.long))
        partition = ParameterPartitionLayout(1, 0, (2, 4), 0, 4, 2)
        collective = Zero2OwnerCollective([partition])
        dense = torch.full((2, 4), float(rank + 1), device=device)
        gradient = layout.pack_first(dense)

        reduced = collective.reduce_scatter([gradient])

        assert torch.allclose(reduced.values[1], torch.full((2, ), 1.5, device=device))
        reduced.values[1].copy_(torch.tensor([10.0 + rank * 10, 11.0 + rank * 10], device=device))
        gathered = collective.all_gather({1: gradient.columns}, reduced)
        assert torch.equal(gathered[0].values, torch.tensor([[10.0, 11.0], [20.0, 21.0]], device=device))

    def test_two_rank_average(self):
        rank = dist.get_rank()
        device = get_accelerator().current_device_name()
        layout = HybridColumnLayout(1, 2, torch.tensor([1]), torch.tensor([], dtype=torch.long))
        dense_gradient = torch.full((2, 2), float(rank + 1), device=device)

        reduced = CompressedGradientCollective.all_reduce([layout.pack_first(dense_gradient)])

        assert torch.allclose(reduced[0].values, torch.full((2, 1), 1.5, device=device))


def test_hybrid_runtime_updates_bias_and_norm_parameters_only_at_boundary():
    matrix = torch.ones((2, 2), dtype=torch.float32)
    bias = torch.ones(2, dtype=torch.float32)
    layout = HybridColumnLayout(1, 2, torch.tensor([0]), torch.tensor([1]))
    runtime = HybridUpdateRuntime(parameters={
        1: matrix,
        2: bias
    },
                                  layouts={1: layout},
                                  update_interval=2,
                                  accumulation_device="cpu",
                                  lr=0.1,
                                  betas=(0.0, 0.0),
                                  eps=1e-8,
                                  weight_decay=0.0)

    runtime.step(first_gradients={1: torch.ones((2, 1))}, second_gradients={1: torch.ones((2, 1))})
    assert torch.equal(bias, torch.ones(2))
    runtime.step(first_gradients={1: torch.ones((2, 1))},
                 second_gradients={1: torch.ones((2, 1))},
                 dense_gradients={
                     1: torch.empty((2, 0)),
                     2: torch.ones(2)
                 })
    runtime.close()

    assert torch.allclose(bias, torch.full((2, ), 0.9))


def test_hybrid_runtime_migrates_dense_adam_state_and_round_trips_checkpoint():
    parameter = torch.ones((2, 3), dtype=torch.float32)
    layout = HybridColumnLayout(1, 3, torch.tensor([0]), torch.tensor([1]))
    runtime = HybridUpdateRuntime(parameters={1: parameter},
                                  layouts={1: layout},
                                  update_interval=2,
                                  accumulation_device="cpu",
                                  lr=0.1,
                                  betas=(0.9, 0.999),
                                  eps=1e-8,
                                  weight_decay=0.0)
    exp_avg = torch.arange(6, dtype=torch.float32).view(2, 3)
    exp_avg_sq = exp_avg.square().add_(1.0)
    runtime.initialize_from_dense_adam(1, step=7, exp_avg=exp_avg, exp_avg_sq=exp_avg_sq)
    checkpoint = runtime.state_dict()

    restored_parameter = parameter.clone()
    restored = HybridUpdateRuntime(parameters={1: restored_parameter},
                                   layouts={1: layout},
                                   update_interval=2,
                                   accumulation_device="cpu",
                                   lr=0.1,
                                   betas=(0.9, 0.999),
                                   eps=1e-8,
                                   weight_decay=0.0)
    restored.load_state_dict(checkpoint)

    assert restored.state_dict()["gpu_optimizer"]["states"][(1, "first")]["step"] == 7
    assert restored.state_dict()["cpu_optimizer"]["states"][(1, "second")]["step"] == 7
    assert torch.equal(restored.state_dict()["cpu_optimizer"]["states"][(1, "dense")]["exp_avg"], exp_avg[:, [2]])


def test_selected_adam_flattens_state_without_changing_values():
    optimizer = SelectedColumnAdamW(lr=0.1)
    optimizer.initialize_state("a", torch.tensor([1.0, 2.0]), torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4]), 3)
    optimizer.initialize_state("b", torch.tensor([4.0]), torch.tensor([0.5]), torch.tensor([0.6]), 3)

    master, exp_avg, exp_avg_sq, lengths, step = optimizer.flatten_states(["a", "b"])

    assert master.tolist() == [1.0, 2.0, 4.0]
    torch.testing.assert_close(exp_avg, torch.tensor([0.1, 0.2, 0.5]))
    torch.testing.assert_close(exp_avg_sq, torch.tensor([0.3, 0.4, 0.6]))
    assert lengths == [2, 1]
    assert step == 3
    master[0] = 7.0
    assert optimizer.master_values("a")[0].item() == 7.0


def test_selected_column_adam_updates_only_requested_columns():
    parameter = torch.ones((2, 4), dtype=torch.float32)
    original = parameter.clone()
    columns = torch.tensor([1, 3])
    gradient = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    optimizer = SelectedColumnAdamW(lr=0.1, betas=(0.0, 0.0), eps=1e-8, weight_decay=0.0)

    optimizer.step_parameter("weight.first", parameter, columns, gradient)

    assert torch.equal(parameter[:, [0, 2]], original[:, [0, 2]])
    assert torch.allclose(parameter[:, [1, 3]], torch.full((2, 2), 0.9))
    assert optimizer.state_step("weight.first") == 1
