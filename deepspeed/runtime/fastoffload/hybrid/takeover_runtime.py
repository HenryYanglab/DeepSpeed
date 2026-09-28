# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""End-to-end optimizer-step runtime for owner-sharded ZeRO-2 takeover."""

from dataclasses import dataclass
from typing import Optional

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.fastoffload.importance.selective_linear import (clear_takeover_capture, set_dense_backward,
                                                                       set_selected_columns, set_takeover_capture)

from .numerics import HybridGradientNumerics, HybridNumericsResult
from .owner_cpu_update import Zero2OwnerCpuUpdater
from .owner_update import Zero2OwnerGpuUpdater
from .takeover import TakeoverGradientBatch, Zero2TakeoverGradientPipeline


@dataclass(frozen=True)
class TakeoverStepResult:
    numerics: HybridNumericsResult
    first_updated_elements: int
    submitted_version: Optional[int]


class Zero2TakeoverRuntime:
    """Coordinate compressed capture, numerics, A updates, and asynchronous B/C updates."""

    def __init__(self,
                 adapter,
                 importance_registry,
                 update_interval: int,
                 gradient_accumulation_steps: int,
                 accumulation_device: str,
                 second_gradient_reduction: str,
                 lr: float,
                 betas: tuple[float, float],
                 eps: float,
                 weight_decay: float,
                 max_async_lag: int = 1,
                 compressed_bucket_bytes: int = 134217728,
                 metrics=None,
                 pt_reserved_cores_perc: float = 0.25,
                 second_reduce_scatter_device: str = "gpu") -> None:
        if second_reduce_scatter_device not in ("cpu", "gpu"):
            raise ValueError("B reduce-scatter device must be cpu or gpu")
        self._cpu_second = second_reduce_scatter_device == "cpu"
        if self._cpu_second and (accumulation_device != "cpu" or gradient_accumulation_steps != 1):
            raise ValueError("CPU B reduce-scatter requires CPU accumulation and GAS=1")
        layouts = tuple(adapter.iter_parameter_partition_layouts())
        collectives = adapter.create_owner_collectives()
        self._adapter = adapter
        self._importance_registry = importance_registry
        self._layouts = {layout.parameter_id: layout for layout in layouts}
        self._collectives = collectives
        self._compressed_bucket_bytes = compressed_bucket_bytes
        self._metrics = metrics
        self._supports_native_boundary = (hasattr(adapter, "compute_native_gradient_numerics")
                                          and hasattr(adapter, "copy_local_reduced_gradient"))
        self._native_first = {}
        self._native_second = {}
        self._native_dense = {}
        self._native_columns = ({}, {}, {})
        self._native_transfer_event = None
        self._state_initialized = False
        self._selective_configured = False
        cpu_collectives = adapter.create_cpu_owner_collectives() if self._cpu_second else None
        self._pipeline = Zero2TakeoverGradientPipeline(adapter, importance_registry, update_interval,
                                                       gradient_accumulation_steps, compressed_bucket_bytes, metrics,
                                                       cpu_collectives)
        self._gpu_updater = Zero2OwnerGpuUpdater(adapter, layouts, collectives, lr, betas, eps, weight_decay)
        self._cpu_updater = Zero2OwnerCpuUpdater(adapter, layouts, collectives, update_interval, accumulation_device,
                                                 second_gradient_reduction, lr, betas, eps, weight_decay,
                                                 max_async_lag, metrics, pt_reserved_cores_perc)

    @property
    def active(self) -> bool:
        return self._pipeline.active

    @property
    def pending_updates(self) -> int:
        return self._cpu_updater.pending_updates

    @property
    def native_dense_boundary(self) -> bool:
        return self._supports_native_boundary and self.active and self._pipeline.dense_boundary

    def capture_native_local_second(self, parameter) -> None:
        self._pipeline.capture_native_local_second(parameter)

    def capture_native_reduced_gradient(self, parameter) -> bool:
        if not self.native_dense_boundary:
            return False
        parameter_id = self._adapter.get_parameter_id(parameter)
        layout = self._layouts.get(parameter_id)
        if layout is None:
            return False
        rank = self._adapter.get_partition_rank(layout.group_id)
        fragment = self._adapter.get_local_reduced_gradient(parameter)
        selection = self._importance_registry.get(parameter_id)
        if selection is None:
            columns = (None, None, torch.arange(layout.shape[1], dtype=torch.long))
        else:
            selected = torch.cat((selection.first_indices, selection.second_indices))
            mask = torch.ones(layout.shape[1], dtype=torch.bool)
            mask[selected] = False
            columns = (selection.first_indices, selection.second_indices,
                       torch.arange(layout.shape[1], dtype=torch.long)[mask])
        destinations = (self._native_first, self._native_second, self._native_dense)
        for band, parameter_columns in enumerate(columns):
            if self._cpu_second and band == 1:
                continue
            if parameter_columns is None or parameter_columns.numel() == 0:
                continue
            values = layout.extract_fragment_values(fragment, parameter_columns, rank)
            if band == 2:
                values = self._cpu_updater.stage_native_dense_gradient(parameter_id, values)
            destinations[band].setdefault(layout.group_id, {})[parameter_id] = values
            self._native_columns[band][parameter_id] = parameter_columns
        return True

    def capture_gradient(self, parameter, group_id: int) -> bool:
        gradient = self._adapter.get_gradient_tensor(parameter)
        if gradient is None:
            return False
        self._initialize_from_dense_adam(gradient.device)
        return self._pipeline.capture(parameter, group_id)

    def capture_packed_gradient(self, parameter_id: int, columns: torch.Tensor, values: torch.Tensor) -> None:
        self._initialize_from_dense_adam(values.device)
        self._pipeline.capture_packed(parameter_id, columns, values)

    def finish_microbatch(self) -> Optional[TakeoverGradientBatch]:
        batch = self._pipeline.finish_microbatch()
        if batch is None or not self.native_dense_boundary:
            return batch
        self._native_transfer_event = self._cpu_updater.record_native_transfer_event()
        return TakeoverGradientBatch(first={},
                                     second=batch.second,
                                     dense={},
                                     first_columns={},
                                     second_columns=batch.second_columns,
                                     dense_columns={},
                                     dense_boundary=True,
                                     native_boundary=True)

    def step(self, batch: TakeoverGradientBatch, loss_scale: float, clip_grad: float) -> TakeoverStepResult:
        if batch.native_boundary:
            return self._native_boundary_step(loss_scale, clip_grad, batch)
        gradients = []
        for reduced_by_group in (batch.first, batch.second, batch.dense):
            for reduced in reduced_by_group.values():
                gradients.extend(reduced.values.values())
        numerics = HybridGradientNumerics.unscale_and_clip(
            gradients,
            loss_scale,
            clip_grad,
            process_group=self._adapter.get_data_parallel_group(),
            communication_device=(get_accelerator().current_device_name() if self._cpu_second else None))
        if numerics.overflow:
            self._pipeline.complete_step(overflow=True)
            return TakeoverStepResult(numerics, 0, None)

        lr = self._adapter.get_learning_rate()
        first_updated = self._gpu_updater.step(batch.first, batch.first_columns, lr=lr)
        submitted_version = None
        if batch.dense_boundary:
            if self._cpu_updater.at_capacity:
                self._cpu_updater.wait_and_commit_oldest()
            submitted_version = self._cpu_updater.submit_boundary(batch.second,
                                                                  batch.dense,
                                                                  batch.second_columns,
                                                                  batch.dense_columns,
                                                                  lr=lr)
        else:
            self._cpu_updater.accumulate_second(batch.second)
        self._pipeline.complete_step(overflow=False)
        return TakeoverStepResult(numerics, first_updated, submitted_version)

    def _native_boundary_step(self, loss_scale: float, clip_grad: float,
                              batch: TakeoverGradientBatch) -> TakeoverStepResult:
        if self._metrics is not None:
            self._metrics.increment("takeover_native_boundary_count")
        overflow, global_norm, combined_scale = self._adapter.compute_native_gradient_numerics(loss_scale, clip_grad)
        if self._cpu_second:
            second_gradients = [value for reduced in batch.second.values() for value in reduced.values.values()]
            second_numerics = HybridGradientNumerics.unscale_and_clip(
                second_gradients, 1.0, 0.0, process_group=self._adapter.get_cpu_owner_group())
            # Native sees only A/C; add the globally reduced B norm before choosing ONE clipping scale.
            overflow = overflow or second_numerics.overflow
            global_norm = (global_norm**2 + (second_numerics.global_norm / loss_scale)**2)**0.5
            combined_scale = loss_scale
            if clip_grad > 0.0 and global_norm > clip_grad:
                combined_scale *= global_norm / clip_grad
        if self._native_transfer_event is None:
            raise RuntimeError("Native boundary transfer event is unavailable")
        self._native_transfer_event.synchronize()
        self._native_transfer_event = None
        numerics = HybridNumericsResult(overflow, global_norm, combined_scale)
        if overflow:
            self._cpu_updater.discard_native_staging()
            self._clear_native_capture()
            self._pipeline.complete_step(overflow=True)
            return TakeoverStepResult(numerics, 0, None)

        self._ensure_native_entries()
        inverse_scale = 1.0 / combined_scale
        for values_by_group in (self._native_first, self._native_second):
            for group_values in values_by_group.values():
                for values in group_values.values():
                    values.mul_(inverse_scale)
        if self._cpu_second:
            for reduced in batch.second.values():
                for values in reduced.values.values():
                    values.mul_(inverse_scale)
        values_by_band = (self._native_first, self._native_second, self._native_dense)
        reduced_bands = []
        for band, values_by_group in enumerate(values_by_band):
            if self._cpu_second and band == 1:
                reduced_bands.append(batch.second)
                self._native_columns[band].update(batch.second_columns)
                continue
            reduced_by_group = {}
            for group_id, group_values in values_by_group.items():
                group_columns = {
                    parameter_id: self._native_columns[band][parameter_id]
                    for parameter_id in group_values
                }
                reduced_by_group[group_id] = self._collectives[group_id].create_owner_gradient_set(
                    group_columns, group_values, self._compressed_bucket_bytes)
            reduced_bands.append(reduced_by_group)

        lr = self._adapter.get_learning_rate()
        first_updated = self._gpu_updater.step(reduced_bands[0], self._native_columns[0], lr=lr)
        # The next forward may issue readiness collectives on a different ZeRO path. Finish the boundary publication
        # before allowing ranks with different host-side speeds to enter that collective sequence.
        get_accelerator().synchronize()
        if self._cpu_updater.at_capacity:
            self._cpu_updater.wait_and_commit_oldest()
        submitted_version = self._cpu_updater.submit_boundary(reduced_bands[1],
                                                              reduced_bands[2],
                                                              self._native_columns[1],
                                                              self._native_columns[2],
                                                              dense_scale=combined_scale,
                                                              lr=lr)
        self._clear_native_capture()
        self._pipeline.complete_step(overflow=False)
        return TakeoverStepResult(numerics, first_updated, submitted_version)

    def _ensure_native_entries(self) -> None:
        destinations = (self._native_first, self._native_second, self._native_dense)
        for parameter_id, layout in self._layouts.items():
            selection = self._importance_registry.get(parameter_id)
            if selection is None:
                columns = (None, None, torch.arange(layout.shape[1], dtype=torch.long))
            else:
                selected = torch.cat((selection.first_indices, selection.second_indices))
                mask = torch.ones(layout.shape[1], dtype=torch.bool)
                mask[selected] = False
                columns = (selection.first_indices, selection.second_indices,
                           torch.arange(layout.shape[1], dtype=torch.long)[mask])
            rank = self._adapter.get_partition_rank(layout.group_id)
            for band, parameter_columns in enumerate(columns):
                if self._cpu_second and band == 1:
                    continue
                if parameter_columns is None or parameter_columns.numel() == 0:
                    continue
                destination = destinations[band].setdefault(layout.group_id, {})
                self._native_columns[band][parameter_id] = parameter_columns
                if parameter_id in destination:
                    continue
                local_numel = layout.owner_counts(parameter_columns)[rank]
                if local_numel:
                    raise RuntimeError(f"Native owner gradient is missing for parameter {parameter_id}")
                if band == 2:
                    values = self._cpu_updater.native_dense_gradient_view(parameter_id)
                else:
                    parameter = self._adapter.get_parameter(parameter_id)
                    values = torch.empty(0, dtype=parameter.dtype, device=parameter.device)
                destination[parameter_id] = values

    def _clear_native_capture(self) -> None:
        self._native_first = {}
        self._native_second = {}
        self._native_dense = {}
        self._native_columns = ({}, {}, {})

    def state_dict(self):
        return {
            "second_reduce_scatter_device": "cpu" if self._cpu_second else "gpu",
            "pipeline": self._pipeline.state_dict(),
            "gpu_updater": self._gpu_updater.state_dict(),
            "cpu_updater": self._cpu_updater.state_dict(),
        }

    def load_state_dict(self, state_dict) -> None:
        expected_device = "cpu" if self._cpu_second else "gpu"
        if state_dict.get("second_reduce_scatter_device", "gpu") != expected_device:
            raise ValueError("Cannot change B reduce-scatter device when restoring a takeover checkpoint")
        self._pipeline.load_state_dict(state_dict["pipeline"])
        device = torch.device(get_accelerator().current_device_name())
        self._gpu_updater.load_state_dict(state_dict["gpu_updater"], device)
        self._cpu_updater.load_state_dict(state_dict["cpu_updater"])
        self._cpu_updater.finalize_state_initialization()
        self._state_initialized = True

    def _initialize_from_dense_adam(self, device: torch.device) -> None:
        if self._state_initialized:
            return
        if not self._importance_registry.ready:
            raise RuntimeError("Cannot initialize takeover optimizer before importance selection")
        states = {}
        for parameter_id, layout in self._layouts.items():
            if layout.group_id not in states:
                states[layout.group_id] = self._adapter.get_dense_adam_partition_state(layout.group_id)
            step, exp_avg, exp_avg_sq = states[layout.group_id]
            selection = self._importance_registry.get(parameter_id)
            if selection is None:
                if not self._importance_registry.is_dense(parameter_id):
                    raise RuntimeError(f"Importance result is missing parameter {parameter_id}")
                columns = torch.arange(layout.shape[1], dtype=torch.long)
                self._cpu_updater.initialize_state(parameter_id, columns, True, step, exp_avg, exp_avg_sq)
                continue
            first = selection.first_indices
            second = selection.second_indices
            selected = torch.cat((first, second))
            mask = torch.ones(layout.shape[1], dtype=torch.bool)
            mask[selected] = False
            dense = torch.arange(layout.shape[1], dtype=torch.long)[mask]
            self._gpu_updater.initialize_state(parameter_id, first, step, exp_avg, exp_avg_sq, device)
            self._cpu_updater.initialize_state(parameter_id, second, False, step, exp_avg, exp_avg_sq)
            if dense.numel():
                self._cpu_updater.initialize_state(parameter_id, dense, True, step, exp_avg, exp_avg_sq)
        self._cpu_updater.finalize_state_initialization()
        self._state_initialized = True

    def prepare_forward(self) -> int:
        if not self.active:
            return 0
        self._configure_selective_backward()
        if not self._state_initialized:
            first_parameter_id = next(iter(self._layouts))
            self._initialize_from_dense_adam(self._adapter.get_parameter(first_parameter_id).device)
        committed = 0
        if self._cpu_updater.pending_updates:
            get_accelerator().synchronize()
            committed = self._cpu_updater.commit_if_all_ranks_ready()
        self._cpu_updater.wait_for_visibility()
        dense_backward = self._pipeline.dense_boundary
        if dense_backward:
            self._cpu_updater.prepare_native_boundary()
        for parameter_id in self._layouts:
            selection = self._importance_registry.get(parameter_id)
            if selection is not None:
                set_dense_backward(self._adapter.get_parameter(parameter_id), dense_backward)
        return committed

    def _configure_selective_backward(self) -> None:
        if self._selective_configured:
            return
        for parameter_id in self._layouts:
            selection = self._importance_registry.get(parameter_id)
            if selection is None:
                continue
            parameter = self._adapter.get_parameter(parameter_id)
            selected_columns = torch.cat((selection.first_indices, selection.second_indices)).sort().values
            self._pipeline.configure_packed_columns(parameter_id, selected_columns)
            set_selected_columns(parameter, selection.first_indices, selection.second_indices)
            set_takeover_capture(parameter, parameter_id, self.capture_packed_gradient)
        self._selective_configured = True

    def close(self) -> None:
        self._cpu_updater.close()
        self._pipeline.close()
        if self._cpu_second:
            self._adapter.close_cpu_owner_group()
        if self._selective_configured:
            for parameter_id in self._layouts:
                if self._importance_registry.get(parameter_id) is not None:
                    clear_takeover_capture(self._adapter.get_parameter(parameter_id))
            self._selective_configured = False
