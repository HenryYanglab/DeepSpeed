# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Executable hybrid update runtime for pre-reduced compressed gradients."""

from typing import Any, Dict, Mapping

import torch

from .context import HybridUpdateJob, HybridUpdateResult
from .coordinator import HybridUpdateCoordinator
from .layout import CompressedColumnGradient, HybridColumnLayout
from .selected_adam import SelectedColumnAdamW


class HybridUpdateRuntime:
    """Update A on the live device and B/C through one asynchronous CPU worker."""

    def __init__(self,
                 parameters: Mapping[int, torch.Tensor],
                 layouts: Mapping[int, HybridColumnLayout],
                 update_interval: int,
                 accumulation_device: str,
                 lr: float,
                 betas: tuple[float, float],
                 eps: float,
                 weight_decay: float,
                 second_gradient_reduction: str = "mean") -> None:
        if not layouts.keys() <= parameters.keys():
            raise ValueError("Every hybrid layout must reference a registered parameter")
        self._parameters = dict(parameters)
        self._layouts = dict(layouts)
        self._gpu_optimizer = SelectedColumnAdamW(lr, betas, eps, weight_decay)
        self._cpu_optimizer = SelectedColumnAdamW(lr, betas, eps, weight_decay)
        self._second_values: Dict[int, torch.Tensor] = {}
        self._dense_values: Dict[int, torch.Tensor] = {}
        self._dense_parameter_values: Dict[int, torch.Tensor] = {}
        self._closed = False
        for parameter_id, parameter in self._parameters.items():
            layout = self._layouts.get(parameter_id)
            if layout is None:
                if parameter.dim() == 2:
                    raise ValueError(f"Matrix parameter {parameter_id} requires a hybrid column layout")
                self._dense_parameter_values[parameter_id] = parameter.detach().float().cpu().clone()
                continue
            if parameter.dim() != 2 or parameter.shape[1] != layout.column_count:
                raise ValueError(f"Parameter {parameter_id} does not match its hybrid layout")
            second_columns = layout.second_columns.to(device=parameter.device)
            dense_columns = layout.dense_columns.to(device=parameter.device)
            self._second_values[parameter_id] = parameter.index_select(1, second_columns).detach().float().cpu()
            self._dense_values[parameter_id] = parameter.index_select(1, dense_columns).detach().float().cpu()
        self._coordinator = HybridUpdateCoordinator(update_interval=update_interval,
                                                    accumulation_device=accumulation_device,
                                                    update_function=self._cpu_update,
                                                    second_gradient_reduction=second_gradient_reduction)

    @property
    def committed_version(self) -> int:
        return self._coordinator.committed_version

    @property
    def pending_updates(self) -> int:
        return self._coordinator.pending_updates

    @torch.no_grad()
    def step(self,
             first_gradients: Mapping[int, torch.Tensor],
             second_gradients: Mapping[int, torch.Tensor],
             dense_gradients: Mapping[int, torch.Tensor] | None = None) -> None:
        self._ensure_open()
        self._coordinator.commit_ready(self._commit)
        self._update_first(first_gradients)
        if dense_gradients is None:
            self._coordinator.accumulate_second(second_gradients)
            return
        if self._coordinator.pending_updates:
            self._coordinator.wait_and_commit_oldest(self._commit)
        self._coordinator.submit_boundary(second_gradients, dense_gradients)

    def handle_overflow(self) -> None:
        self._ensure_open()
        self._coordinator.discard_active_gradients()

    def progress(self) -> int:
        self._ensure_open()
        return self._coordinator.commit_ready(self._commit)

    def initialize_from_dense_adam(self, parameter_id: int, step: int, exp_avg: torch.Tensor,
                                   exp_avg_sq: torch.Tensor) -> None:
        self._ensure_open()
        parameter = self._parameter(parameter_id)
        if exp_avg.shape != parameter.shape or exp_avg_sq.shape != parameter.shape:
            raise ValueError("Dense Adam state does not match the hybrid parameter")
        layout = self._layouts.get(parameter_id)
        if layout is None:
            values = parameter.detach().float().cpu()
            self._cpu_optimizer.initialize_state((parameter_id, "dense_parameter"), values, exp_avg.cpu(),
                                                 exp_avg_sq.cpu(), step)
            self._dense_parameter_values[parameter_id] = self._cpu_optimizer.master_values(
                (parameter_id, "dense_parameter"))
            return
        groups = (("first", layout.first_columns, self._gpu_optimizer),
                  ("second", layout.second_columns, self._cpu_optimizer), ("dense", layout.dense_columns,
                                                                           self._cpu_optimizer))
        for group_name, columns, optimizer in groups:
            device = parameter.device if group_name == "first" else torch.device("cpu")
            device_columns = columns.to(device=parameter.device)
            values = parameter.detach().index_select(1, device_columns).to(device=device, dtype=torch.float32)
            moments = exp_avg.detach().index_select(1, columns.to(device=exp_avg.device)).to(device=device)
            variances = exp_avg_sq.detach().index_select(1, columns.to(device=exp_avg_sq.device)).to(device=device)
            optimizer.initialize_state((parameter_id, group_name), values, moments, variances, step)
        self._second_values[parameter_id] = self._cpu_optimizer.master_values((parameter_id, "second"))
        self._dense_values[parameter_id] = self._cpu_optimizer.master_values((parameter_id, "dense"))

    def state_dict(self) -> Dict[str, Any]:
        if self._coordinator.pending_updates:
            raise RuntimeError("Hybrid updates must be committed before checkpointing")
        return {
            "committed_version": self.committed_version,
            "gpu_optimizer": self._gpu_optimizer.state_dict(),
            "cpu_optimizer": self._cpu_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self._ensure_open()
        if self._coordinator.pending_updates:
            raise RuntimeError("Cannot restore a checkpoint with pending hybrid updates")
        gpu_devices = {
            (parameter_id, "first"): parameter.device
            for parameter_id, parameter in self._parameters.items() if parameter_id in self._layouts
        }
        cpu_devices = {}
        for parameter_id in self._layouts:
            cpu_devices[(parameter_id, "second")] = torch.device("cpu")
            cpu_devices[(parameter_id, "dense")] = torch.device("cpu")
        for parameter_id in self._dense_parameter_values:
            cpu_devices[(parameter_id, "dense_parameter")] = torch.device("cpu")
        self._gpu_optimizer.load_state_dict(state_dict["gpu_optimizer"], gpu_devices)
        self._cpu_optimizer.load_state_dict(state_dict["cpu_optimizer"], cpu_devices)
        for parameter_id in self._layouts:
            second_key = (parameter_id, "second")
            dense_key = (parameter_id, "dense")
            if self._cpu_optimizer.state_step(second_key):
                self._second_values[parameter_id] = self._cpu_optimizer.master_values(second_key)
            if self._cpu_optimizer.state_step(dense_key):
                self._dense_values[parameter_id] = self._cpu_optimizer.master_values(dense_key)
        for parameter_id in self._dense_parameter_values:
            key = (parameter_id, "dense_parameter")
            if self._cpu_optimizer.state_step(key):
                self._dense_parameter_values[parameter_id] = self._cpu_optimizer.master_values(key)
        self._coordinator.restore_committed_version(int(state_dict["committed_version"]))

    def close(self) -> None:
        if self._closed:
            return
        self._coordinator.close(self._commit)
        self._closed = True

    def _update_first(self, gradients: Mapping[int, torch.Tensor]) -> None:
        for parameter_id, gradient in gradients.items():
            parameter = self._parameter(parameter_id)
            layout = self._layouts[parameter_id]
            self._gpu_optimizer.step_parameter((parameter_id, "first"), parameter, layout.first_columns, gradient)

    def _cpu_update(self, job: HybridUpdateJob) -> Mapping[int, torch.Tensor]:
        updated: Dict[int, torch.Tensor] = {}
        for parameter_id, layout in self._layouts.items():
            second_gradient = job.second_gradients.get(parameter_id)
            dense_gradient = job.dense_gradients.get(parameter_id)
            if second_gradient is None and dense_gradient is None:
                continue
            if second_gradient is None or dense_gradient is None:
                raise ValueError(f"Hybrid CPU job has incomplete matrix gradients for parameter {parameter_id}")
            second_values = self._cpu_optimizer.step_values((parameter_id, "second"),
                                                            self._second_values[parameter_id], second_gradient)
            dense_values = self._cpu_optimizer.step_values((parameter_id, "dense"), self._dense_values[parameter_id],
                                                           dense_gradient)
            self._second_values[parameter_id] = second_values.float()
            self._dense_values[parameter_id] = dense_values.float()
            updated[parameter_id] = torch.cat((second_values, dense_values), dim=1)
        for parameter_id, values in self._dense_parameter_values.items():
            gradient = job.dense_gradients.get(parameter_id)
            if gradient is None:
                continue
            dense_values = self._cpu_optimizer.step_values((parameter_id, "dense_parameter"), values, gradient)
            self._dense_parameter_values[parameter_id] = dense_values.float()
            updated[parameter_id] = dense_values
        return updated

    @torch.no_grad()
    def _commit(self, result: HybridUpdateResult) -> None:
        for parameter_id, values in result.updated_values.items():
            parameter = self._parameter(parameter_id)
            layout = self._layouts.get(parameter_id)
            if layout is None:
                parameter.copy_(values.to(device=parameter.device, dtype=parameter.dtype))
                continue
            columns = torch.cat((layout.second_columns, layout.dense_columns))
            compressed = CompressedColumnGradient(parameter_id=parameter_id, columns=columns, values=values)
            layout.scatter_values(parameter, compressed)

    def _parameter(self, parameter_id: int) -> torch.Tensor:
        parameter = self._parameters.get(parameter_id)
        if parameter is None:
            raise ValueError(f"Unknown hybrid parameter ID: {parameter_id}")
        return parameter

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Hybrid update runtime is closed")
