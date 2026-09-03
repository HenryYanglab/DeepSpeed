# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""AdamW state and updates for compressed selected-column values."""

from dataclasses import dataclass
from typing import Any, Dict, Hashable, Mapping

import torch


@dataclass
class _AdamState:
    step: int
    master_values: torch.Tensor
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor


class SelectedColumnAdamW:
    """Update exact selected values without scanning a full parameter tensor."""

    def __init__(self,
                 lr: float,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0) -> None:
        if lr < 0.0:
            raise ValueError("lr must be non-negative")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("Adam betas must be in [0, 1)")
        if eps <= 0.0:
            raise ValueError("eps must be greater than zero")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        self._lr = lr
        self._beta1, self._beta2 = betas
        self._eps = eps
        self._weight_decay = weight_decay
        self._states: Dict[Hashable, _AdamState] = {}

    def step_values(self, key: Hashable, values: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
        if values.shape != gradient.shape:
            raise ValueError("Selected values and gradient shapes must match")
        state = self._states.get(key)
        if state is None:
            master_values = values.detach().float().clone()
            state = _AdamState(step=0,
                               master_values=master_values,
                               exp_avg=torch.zeros_like(master_values),
                               exp_avg_sq=torch.zeros_like(master_values))
            self._states[key] = state
        if state.master_values.device != gradient.device:
            raise ValueError("Selected Adam state and gradient must use the same device")

        state.step += 1
        grad = gradient.detach().float()
        if self._weight_decay:
            state.master_values.mul_(1.0 - self._lr * self._weight_decay)
        state.exp_avg.mul_(self._beta1).add_(grad, alpha=1.0 - self._beta1)
        state.exp_avg_sq.mul_(self._beta2).addcmul_(grad, grad, value=1.0 - self._beta2)
        bias_correction1 = 1.0 - self._beta1**state.step
        bias_correction2 = 1.0 - self._beta2**state.step
        step_size = self._lr / bias_correction1
        denominator = state.exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(self._eps)
        state.master_values.addcdiv_(state.exp_avg, denominator, value=-step_size)
        return state.master_values.to(dtype=values.dtype)

    @torch.no_grad()
    def step_parameter(self, key: Hashable, parameter: torch.Tensor, columns: torch.Tensor,
                       gradient: torch.Tensor) -> torch.Tensor:
        if parameter.dim() != 2:
            raise ValueError("Selected-column Adam requires a two-dimensional parameter")
        device_columns = columns.to(device=parameter.device, dtype=torch.long)
        values = parameter.index_select(1, device_columns)
        updated_values = self.step_values(key, values, gradient)
        parameter.index_copy_(1, device_columns, updated_values)
        return updated_values

    def initialize_state(self, key: Hashable, master_values: torch.Tensor, exp_avg: torch.Tensor,
                         exp_avg_sq: torch.Tensor, step: int) -> None:
        if step < 0:
            raise ValueError("Adam step must not be negative")
        if master_values.shape != exp_avg.shape or master_values.shape != exp_avg_sq.shape:
            raise ValueError("Adam state tensor shapes must match")
        master = master_values.detach().float().clone()
        self._states[key] = _AdamState(step=step,
                                       master_values=master,
                                       exp_avg=exp_avg.detach().to(device=master.device, dtype=torch.float32).clone(),
                                       exp_avg_sq=exp_avg_sq.detach().to(device=master.device,
                                                                         dtype=torch.float32).clone())

    def master_values(self, key: Hashable) -> torch.Tensor:
        state = self._states.get(key)
        if state is None:
            raise KeyError(f"Selected Adam state is unavailable for {key}")
        return state.master_values

    def state_keys(self) -> list[Hashable]:
        return list(self._states)

    def state_tensors(self, key: Hashable):
        state = self._states[key]
        return state.master_values, state.exp_avg, state.exp_avg_sq, state.step

    def flatten_states(self, keys: list[Hashable]):
        states = [self._states[key] for key in keys]
        steps = {state.step for state in states}
        if len(steps) != 1:
            raise RuntimeError("Selected Adam states must have identical steps before flattening")
        lengths = [state.master_values.numel() for state in states]
        master = torch.cat([state.master_values.view(-1) for state in states])
        exp_avg = torch.cat([state.exp_avg.view(-1) for state in states])
        exp_avg_sq = torch.cat([state.exp_avg_sq.view(-1) for state in states])
        offset = 0
        for state, length in zip(states, lengths):
            state.master_values = master.narrow(0, offset, length)
            state.exp_avg = exp_avg.narrow(0, offset, length)
            state.exp_avg_sq = exp_avg_sq.narrow(0, offset, length)
            offset += length
        return master, exp_avg, exp_avg_sq, lengths, steps.pop()

    def set_state_steps(self, keys: list[Hashable], step: int) -> None:
        for key in keys:
            self._states[key].step = step

    def state_dict(self) -> Dict[str, Any]:
        return {
            "states": {
                key: {
                    "step": state.step,
                    "master_values": state.master_values.detach().cpu().clone(),
                    "exp_avg": state.exp_avg.detach().cpu().clone(),
                    "exp_avg_sq": state.exp_avg_sq.detach().cpu().clone(),
                }
                for key, state in self._states.items()
            }
        }

    def load_state_dict(self, state_dict: Mapping[str, Any], devices: Mapping[Hashable, torch.device]) -> None:
        states = state_dict.get("states")
        if not isinstance(states, dict):
            raise ValueError("Selected Adam checkpoint is missing states")
        restored = {}
        for key, values in states.items():
            if key not in devices:
                raise ValueError(f"Selected Adam checkpoint contains unknown key: {key}")
            device = devices[key]
            master = values["master_values"].to(device=device, dtype=torch.float32)
            restored[key] = _AdamState(step=int(values["step"]),
                                       master_values=master,
                                       exp_avg=values["exp_avg"].to(device=device, dtype=torch.float32),
                                       exp_avg_sq=values["exp_avg_sq"].to(device=device, dtype=torch.float32))
        self._states = restored

    def state_step(self, key: Hashable) -> int:
        state = self._states.get(key)
        return 0 if state is None else state.step
