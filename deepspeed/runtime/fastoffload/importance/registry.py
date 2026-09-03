# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Process-local storage for completed importance selections."""

import threading
from typing import Dict, Optional

from .context import ColumnImportanceSelection, ParameterImportanceView


class ImportanceRegistry:
    """Store immutable per-parameter selections for later policies."""

    def __init__(self) -> None:
        self._selections: Dict[int, ColumnImportanceSelection] = {}
        self._dense_parameters: Dict[int, tuple[str, tuple[int, ...]]] = {}
        self._ready = False
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._ready

    def add(self, selection: ColumnImportanceSelection) -> None:
        with self._lock:
            if self._ready:
                raise RuntimeError("Importance registry is already finalized")
            if selection.parameter_id in self._selections or selection.parameter_id in self._dense_parameters:
                raise ValueError(f"Duplicate importance result for parameter {selection.parameter_id}")
            self._selections[selection.parameter_id] = selection

    def add_dense(self, view: ParameterImportanceView) -> None:
        with self._lock:
            if self._ready:
                raise RuntimeError("Importance registry is already finalized")
            if view.parameter_id in self._selections or view.parameter_id in self._dense_parameters:
                raise ValueError(f"Duplicate importance result for parameter {view.parameter_id}")
            self._dense_parameters[view.parameter_id] = (view.parameter_name, view.shape)

    def finalize(self) -> None:
        with self._lock:
            self._ready = True

    def get(self, parameter_id: int) -> Optional[ColumnImportanceSelection]:
        with self._lock:
            return self._selections.get(parameter_id)

    def is_dense(self, parameter_id: int) -> bool:
        with self._lock:
            return parameter_id in self._dense_parameters

    def load_state_dict(self, state_dict: dict) -> None:
        selections = state_dict.get("selections")
        dense_parameters = state_dict.get("dense_parameters")
        if not isinstance(selections, dict) or not isinstance(dense_parameters, dict):
            raise ValueError("Importance checkpoint is incomplete")
        restored = {}
        for parameter_id, values in selections.items():
            selection_values = dict(values)
            selection_values["parameter_id"] = int(selection_values["parameter_id"])
            selection_values["shape"] = tuple(selection_values["shape"])
            restored[int(parameter_id)] = ColumnImportanceSelection(**selection_values)
        restored_dense = {
            int(parameter_id): (values[0], tuple(values[1]))
            for parameter_id, values in dense_parameters.items()
        }
        with self._lock:
            if self._selections or self._dense_parameters or self._ready:
                raise RuntimeError("Cannot replace a populated importance registry")
            overlap = set(restored).intersection(restored_dense)
            if overlap:
                raise ValueError(f"Importance checkpoint has duplicate parameters: {sorted(overlap)}")
            self._selections = restored
            self._dense_parameters = restored_dense
            self._ready = bool(state_dict.get("ready", False))

    def state_dict(self) -> dict:
        with self._lock:
            return {
                "ready": self._ready,
                "selections": {
                    parameter_id: selection.state_dict()
                    for parameter_id, selection in self._selections.items()
                },
                "dense_parameters": dict(self._dense_parameters),
            }
