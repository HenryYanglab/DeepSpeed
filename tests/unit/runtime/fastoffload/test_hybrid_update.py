# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import threading
import warnings
from types import SimpleNamespace

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


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_compressed_gas_one_transfers_storage_without_copy(reduction):
    accumulator = CompressedMicrobatchAccumulator(1, reduction)
    source = torch.tensor([2.0, 4.0])
    assert accumulator.accumulate({1: source}, take_ownership=True)
    boundary = accumulator.take_boundary()
    assert boundary[1].data_ptr() == source.data_ptr()
    assert torch.equal(boundary[1], torch.tensor([2.0, 4.0]))
    assert accumulator.micro_steps == 0
    accumulator.discard()
    assert torch.equal(boundary[1], source)


def test_compressed_accumulator_default_preserves_caller_storage():
    accumulator = CompressedMicrobatchAccumulator(1)
    source = torch.tensor([2.0])
    accumulator.accumulate({1: source})
    source.zero_()
    assert torch.equal(accumulator.take_boundary()[1], torch.tensor([2.0]))


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


@pytest.mark.parametrize("runtime_mode", ["hybrid", "gpu_ceiling", "gpu_ceiling_native_boundary"])
def test_takeover_runtime_updates_first_each_step_and_cpu_groups_at_boundary(runtime_mode, monkeypatch):
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
        def get_learning_rate():
            return 0.1

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

    runtime_type = Zero2TakeoverRuntime
    if runtime_mode != "hybrid":
        from deepspeed.runtime.fastoffload.scripts.benchmark_gpu_ceiling import GpuCeilingTakeoverRuntime
        runtime_type = GpuCeilingTakeoverRuntime
    if runtime_mode == "gpu_ceiling_native_boundary":
        Adapter.compute_native_gradient_numerics = staticmethod(lambda *_: (False, 6**0.5, 1.0))
        Adapter.copy_local_reduced_gradient = staticmethod(lambda *_: None)
        Adapter.get_local_reduced_gradient = staticmethod(lambda _: parameter.grad.view(-1))
        Adapter.get_parameter = staticmethod(lambda _: parameter)
        monkeypatch.setattr(get_accelerator(), "synchronize", lambda: None)

    runtime = runtime_type(Adapter(), registry, 2, 1, "cpu", "mean", 0.1, (0.0, 0.0), 1e-8, 0.0)
    for step in range(2):
        parameter.grad = torch.ones_like(parameter)
        if runtime.native_dense_boundary:
            assert runtime.capture_native_reduced_gradient(parameter)
        else:
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
    if runtime_mode == "hybrid":
        assert torch.allclose(parameter[:, 1:], torch.full((2, 2), 0.9))
    else:
        assert torch.equal(parameter[:, 1:], torch.ones((2, 2)))
        runtime.assert_no_cpu_work()
        assert runtime.counts["a_update_steps"] == 2
        assert runtime.counts["dense_steps"] == 1
        assert runtime.counts["ordinary_steps"] == 1
        with pytest.raises(RuntimeError, match="GPU ceiling"):
            runtime._cpu_updater.submit_boundary({}, {}, {}, {})
        with pytest.raises(RuntimeError, match="checkpoints"):
            runtime.state_dict()


@pytest.mark.parametrize("rates", [(None, None), (0.07, 0.02), (0.07, 0.0)])
def test_owner_cpu_updater_accumulates_second_and_commits_dense_values(rates):
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

    updater = Zero2OwnerCpuUpdater(Adapter(), [layout], {0: collective},
                                   2,
                                   "cpu",
                                   "mean",
                                   0.1, (0.0, 0.0),
                                   1e-8,
                                   0.0,
                                   max_async_lag=2)
    started, release = threading.Event(), threading.Event()
    seen_lrs = []

    def blocked_update(job):
        if job.version == 1:
            started.set()
            assert release.wait(timeout=20)
        seen_lrs.append(job.lr)
        return updater._update(job)

    updater._coordinator._update_function = blocked_update
    try:
        for version, rate in enumerate(rates, 1):
            lr = None if rate is None else torch.tensor(rate, dtype=torch.float64)
            updater.accumulate_second(reduce(second_columns, 1.0))
            updater.submit_boundary(reduce(second_columns, 3.0),
                                    reduce(dense_columns, 2.0),
                                    second_columns,
                                    dense_columns,
                                    lr=lr)
            if lr is not None:
                lr.fill_(0.9)
            if version == 1:
                assert started.wait(timeout=20)
        assert updater.pending_updates == 2
        release.set()
        checkpoint = updater.state_dict()
        assert checkpoint["committed_version"] == 2
        assert updater.pending_updates == 0
    finally:
        release.set()
        updater.close()

    expected_lrs = [0.1 if rate is None else rate for rate in rates]
    assert seen_lrs == expected_lrs
    assert torch.equal(parameter[:, 0], torch.ones(2))
    assert torch.allclose(parameter[:, 1:], torch.full((2, 2), 1.0 - sum(expected_lrs)))
    assert torch.equal(fp32_partition[[1, 2, 4, 5]], torch.ones(4))


@pytest.mark.parametrize("codes,error,sync_error,last_error", [
    ((0, 0), None, False, 0),
    ((1, 0, 0), None, False, 1),
    ((0, 1, 0), None, False, 0),
    ((0, 1, 1), "code=1", False, 1),
    ((0, 2), "code=2", False, 0),
    ((0, 1), "earlier async failure", True, 1),
    ((0, 1), "Unexpected CUDA error", False, 700),
])
def test_owner_host_registration_retry_and_rollback(monkeypatch, codes, error, sync_error, last_error):
    calls = []
    unregistered = []
    synchronized = []

    def register(address, nbytes, flags):
        calls.append((address, nbytes, flags))
        return codes[len(calls) - 1]

    def unregister(address):
        unregistered.append(address)
        return 0

    def synchronize():
        synchronized.append(True)
        if sync_error:
            raise RuntimeError("earlier async failure")

    cudart = SimpleNamespace(cudaError=SimpleNamespace(success=0),
                             cudaHostRegister=register,
                             cudaHostUnregister=unregister,
                             cudaGetLastError=lambda: last_error)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)  #ignore-cuda
    monkeypatch.setattr(get_accelerator(), "synchronize", synchronize)
    metrics = MetricsRegistry()
    updater = Zero2OwnerCpuUpdater(None, [], {}, 2, "cpu", "mean", 0.1, (0.9, 0.999), 1e-8, 0.0, metrics=metrics)
    buffers = tuple(torch.empty(8, dtype=torch.bfloat16).share_memory_() for _ in range(2))
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        if error is not None:
            with pytest.raises(RuntimeError, match=error):
                updater._register_transfer_buffers(buffers)
            assert unregistered == [buffers[0].data_ptr()]
        else:
            registered = updater._register_transfer_buffers(buffers)
            assert len(registered) == 2
            assert unregistered == []
            failed_addresses = {call[0] for code, call in zip(codes, calls) if code != 0}
            for index, buffer in enumerate(registered):
                replaced = buffers[index].data_ptr() in failed_addresses
                assert (buffer.data_ptr() != buffers[index].data_ptr()) == replaced
                assert buffer.is_shared()
                assert buffer.dtype == buffers[index].dtype
                assert buffer.numel() == buffers[index].numel()
            updater._native_registered_buffers = registered
    retries = int(1 in codes and not sync_error and last_error in (0, 1))
    assert len(recorded) == retries
    assert metrics.snapshot().counters.get("takeover_host_register_retry_count", 0) == retries
    assert len(synchronized) == int(1 in codes)
    assert len(calls) == len(codes)
    assert all(nbytes == buffers[0].nbytes and flags == 0 for _, nbytes, flags in calls)
    updater.close()
    if error is None:
        assert unregistered == [buffer.data_ptr() for buffer in registered]


@pytest.mark.parametrize("lr", [None, 0.02, 0.0])
def test_owner_gpu_updater_updates_and_publishes_first_columns(lr):
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

    updated = updater.step({0: reduced}, columns, lr=lr)

    assert updated == 2
    expected_lr = 0.1 if lr is None else lr
    assert torch.allclose(parameter[:, 0], torch.full((2, ), 1.0 - expected_lr))
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
        assert job.lr == 0.0
        return {1: job.second_gradients[1]}

    coordinator = HybridUpdateCoordinator(update_interval=2,
                                          accumulation_device="cpu",
                                          update_function=update,
                                          prepare_function=prepare)
    coordinator.accumulate_second({1: torch.tensor([1.0])})
    coordinator.submit_boundary({1: torch.tensor([1.0])}, {}, lr=0.0)
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

    def test_cpu_owner_reduce_scatter_with_uneven_and_empty_owners(self):
        group = dist.new_group([0, 1], backend="gloo")
        try:
            rank = dist.get_rank()
            partitions = [
                ParameterPartitionLayout(1, 0, (2, 4), 0, 8, 2),
                ParameterPartitionLayout(2, 0, (2, 4), 8, 8, 2)
            ]
            collective = Zero2OwnerCollective(partitions, group)
            columns = {1: torch.tensor([1, 3]), 2: torch.tensor([2])}
            gradients = [
                CompressedColumnGradient(pid, cols, torch.full((2, cols.numel()), float(rank + pid)))
                for pid, cols in columns.items()
            ]
            reduced = collective.reduce_scatter_buckets(gradients, bucket_bytes=4)
            assert reduced.bucket_count == 2
            for partition in partitions:
                values = reduced.values[partition.parameter_id]
                count = partition.owner_counts(columns[partition.parameter_id])[rank]
                assert values.device.type == "cpu" and values.numel() == count
                assert torch.equal(values, torch.full((count, ), partition.parameter_id + 0.5))
        finally:
            dist.destroy_process_group(group)

    def test_mixed_device_numerics_share_one_global_clip_scale(self):
        device = get_accelerator().current_device_name()
        first = torch.tensor([3.0], device=device)
        second = torch.tensor([4.0], device="cpu")
        result = HybridGradientNumerics.unscale_and_clip([first, second], 2.0, 1.0, communication_device=device)
        assert result.global_norm == pytest.approx((50.0**0.5) / 2)
        assert result.combined_scale == pytest.approx(50.0**0.5)
        assert first.item() == pytest.approx(3.0 / (50.0**0.5))
        assert second.item() == pytest.approx(4.0 / (50.0**0.5))
        second.fill_(float("inf") if dist.get_rank() == 0 else 1.0)
        before = first.clone()
        result = HybridGradientNumerics.unscale_and_clip([first, second], 2.0, 1.0, communication_device=device)
        assert result.overflow and torch.equal(first, before)

    def test_two_rank_average(self):
        rank = dist.get_rank()
        device = get_accelerator().current_device_name()
        layout = HybridColumnLayout(1, 2, torch.tensor([1]), torch.tensor([], dtype=torch.long))
        dense_gradient = torch.full((2, 2), float(rank + 1), device=device)

        reduced = CompressedGradientCollective.all_reduce([layout.pack_first(dense_gradient)])

        assert torch.allclose(reduced[0].values, torch.full((2, 1), 1.5, device=device))


def test_cpu_B_capture_masks_native_boundary_and_fills_unused_gradients():
    parameter = torch.nn.Parameter(torch.ones(2, 4))
    registry = ImportanceRegistry()
    registry.add(
        ColumnImportanceSelection(parameter_id=1,
                                  parameter_name="weight",
                                  shape=(2, 4),
                                  first_indices=torch.tensor([0]),
                                  second_indices=torch.tensor([1]),
                                  first_min_score=1.0,
                                  second_min_score=0.5,
                                  algorithm="test"))
    registry.finalize()
    collective = Zero2OwnerCollective([ParameterPartitionLayout(1, 0, (2, 4), 0, 8, 1)])
    adapter = SimpleNamespace(
        create_owner_collectives=lambda: {0: collective},
        iter_parameter_partition_layouts=lambda: [ParameterPartitionLayout(1, 0, (2, 4), 0, 8, 1)],
        get_parameter_id=lambda p: 1,
        get_gradient_tensor=lambda p: p.grad)
    pipeline = Zero2TakeoverGradientPipeline(adapter, registry, 2, 1, cpu_second_collectives={0: collective})
    parameter.grad = torch.arange(8, dtype=torch.float32).view(2, 4)
    original = parameter.grad.clone()
    pipeline.capture_native_local_second(parameter)
    assert torch.equal(parameter.grad[:, [0, 2, 3]], original[:, [0, 2, 3]])
    assert torch.count_nonzero(parameter.grad[:, 1]) == 0
    batch = pipeline.finish_microbatch()
    assert torch.equal(batch.second[0].values[1], original[:, 1])
    batch = pipeline.finish_microbatch()
    assert torch.equal(batch.second[0].values[1], torch.zeros(2))


@pytest.mark.parametrize("accumulation_device,gas", [("gpu", 1), ("cpu", 2)])
def test_cpu_B_rejects_incompatible_accumulation_before_creating_groups(accumulation_device, gas):
    with pytest.raises(ValueError, match="CPU accumulation and GAS=1"):
        Zero2TakeoverRuntime(None,
                             None,
                             2,
                             gas,
                             accumulation_device,
                             "mean",
                             0.1, (0.9, 0.99),
                             1e-8,
                             0.0,
                             second_reduce_scatter_device="cpu")


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


def test_selected_adam_per_step_lr_matches_torch_adamw():
    values = torch.tensor([1.0, -2.0, 0.5])
    reference = torch.nn.Parameter(values.clone())
    expected = torch.optim.AdamW([reference], lr=0.5, betas=(0.8, 0.9), eps=1e-6, weight_decay=0.2)
    selected = SelectedColumnAdamW(lr=0.5, betas=(0.8, 0.9), eps=1e-6, weight_decay=0.2)
    for step, lr in enumerate((0.1, 0.025, 0.0, 0.075), 1):
        gradient = torch.tensor([0.2 * step, -0.3, 0.1 / step])
        reference.grad = gradient.clone()
        expected.param_groups[0]["lr"] = lr
        expected.step()
        values = selected.step_values("a", values, gradient, lr=lr)
        torch.testing.assert_close(values, reference)
        master, first, second, actual_step = selected.state_tensors("a")
        torch.testing.assert_close(master, reference)
        torch.testing.assert_close(first, expected.state[reference]["exp_avg"])
        torch.testing.assert_close(second, expected.state[reference]["exp_avg_sq"])
        assert actual_step == step


@pytest.mark.parametrize("lr", [-0.1, float("nan"), float("inf")])
def test_invalid_lr_does_not_advance_adam_or_freeze_b(lr):
    optimizer = SelectedColumnAdamW(lr=0.1)
    with pytest.raises(ValueError, match="finite and non-negative"):
        optimizer.step_values("a", torch.ones(1), torch.ones(1), lr=lr)
    assert optimizer.state_step("a") == 0
    coordinator = HybridUpdateCoordinator(2, "cpu", lambda job: {})
    try:
        coordinator.accumulate_second({1: torch.ones(1)})
        with pytest.raises(ValueError, match="finite and non-negative"):
            coordinator.submit_boundary({1: torch.ones(1)}, {}, lr=lr)
        assert coordinator.active_accumulated_steps == 1
        assert coordinator.next_submitted_version == 1 and coordinator.pending_updates == 0
    finally:
        coordinator.close()
