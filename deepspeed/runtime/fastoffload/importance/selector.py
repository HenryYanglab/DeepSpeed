# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Streaming lifecycle for pretrained-delta importance selection."""

import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch

from deepspeed.runtime.fastoffload.telemetry.metrics import MetricsRegistry

from .base import ImportanceAlgorithm
from .context import ParameterImportanceView
from .registry import ImportanceRegistry
from .selective_linear import set_selected_columns


class StreamingImportanceSelector:
    """Snapshot on CPU and compare one parameter at a time after warmup."""

    def __init__(self,
                 views: Iterable[ParameterImportanceView],
                 algorithm: ImportanceAlgorithm,
                 registry: ImportanceRegistry,
                 metrics: MetricsRegistry,
                 warmup_steps: int,
                 rank: int,
                 output_path: Optional[str] = None,
                 sparse_backward: bool = False) -> None:
        self._views = tuple(views)
        self._algorithm = algorithm
        self._registry = registry
        self._metrics = metrics
        self._warmup_steps = warmup_steps
        self._rank = rank
        self._output_path = output_path
        self._sparse_backward = sparse_backward
        self._references: Dict[int, torch.Tensor] = {}
        self._selected = False
        self._closed = False
        self._snapshot_references()

    @property
    def registry(self) -> ImportanceRegistry:
        return self._registry

    def on_step_end(self, completed_steps: int) -> None:
        if self._registry.ready and not self._selected:
            self._references.clear()
            self._selected = True
            self._metrics.set_gauge("importance_ready", 1)
            self._metrics.set_gauge("importance_reference_bytes", 0)
        if self._closed or self._selected or completed_steps < self._warmup_steps:
            return
        selection_start_ns = time.perf_counter_ns()
        selected_parameter_count = 0
        first_column_count = 0
        second_column_count = 0
        for view in self._views:
            reference = self._references.pop(view.parameter_id)
            if view.tensor.dim() != 2:
                self._registry.add_dense(view)
                self._metrics.increment("importance_dense_parameter_count")
                continue

            current = view.tensor.detach()
            selection = self._algorithm.select(view, reference, current)
            self._registry.add(selection)
            selected_parameter_count += 1
            first_column_count += selection.first_indices.numel()
            second_column_count += selection.second_indices.numel()
            self._metrics.increment("importance_selected_parameter_count")
            self._metrics.increment("importance_first_column_count", selection.first_indices.numel())
            self._metrics.increment("importance_second_column_count", selection.second_indices.numel())
            if self._sparse_backward:
                set_selected_columns(view.tensor, selection.first_indices, selection.second_indices)

        self._registry.finalize()
        self._selected = True
        elapsed_ms = (time.perf_counter_ns() - selection_start_ns) / 1_000_000.0
        self._metrics.observe("importance_selection_host_ms", elapsed_ms)
        self._metrics.set_gauge("importance_ready", 1)
        self._metrics.set_gauge("importance_reference_bytes", 0)
        if self._output_path:
            self._save()
        if self._rank == 0:
            print(
                f"[FastOffload Importance] algorithm={self._algorithm.name} warmup_steps={completed_steps} "
                f"parameters={selected_parameter_count} first_columns={first_column_count} "
                f"second_columns={second_column_count} selection_ms={elapsed_ms:.2f}",
                flush=True)

    def close(self) -> None:
        self._references.clear()
        self._metrics.set_gauge("importance_reference_bytes", 0)
        self._closed = True

    def _snapshot_references(self) -> None:
        snapshot_start_ns = time.perf_counter_ns()
        reference_bytes = 0
        for view in self._views:
            reference = view.tensor.detach().to(device="cpu", copy=True)
            self._references[view.parameter_id] = reference
            reference_bytes += reference.numel() * reference.element_size()
        elapsed_ms = (time.perf_counter_ns() - snapshot_start_ns) / 1_000_000.0
        self._metrics.observe("importance_snapshot_host_ms", elapsed_ms)
        self._metrics.set_gauge("importance_reference_bytes", reference_bytes)
        self._metrics.set_gauge("importance_ready", 0)

    def _save(self) -> None:
        configured_path = self._output_path.format(rank=self._rank)
        path = Path(configured_path)
        if "{rank}" not in self._output_path:
            path = path.with_name(f"{path.stem}.rank{self._rank}{path.suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._registry.state_dict(), path)
