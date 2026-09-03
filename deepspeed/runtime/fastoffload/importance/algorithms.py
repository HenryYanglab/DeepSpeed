# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Registration and construction of importance algorithms."""

import threading
from typing import Callable, Dict

from .base import ImportanceAlgorithm
from .pretrained_delta import PretrainedDeltaTopK

AlgorithmFactory = Callable[[float, int], ImportanceAlgorithm]

_ALGORITHMS: Dict[str, AlgorithmFactory] = {}
_LOCK = threading.Lock()


def register_importance_algorithm(name: str, factory: AlgorithmFactory) -> None:
    """Register a process-local algorithm factory before FastOffload initialization."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Importance algorithm name must be non-empty")
    if not callable(factory):
        raise TypeError("Importance algorithm factory must be callable")
    with _LOCK:
        if name in _ALGORITHMS:
            raise ValueError(f"Importance algorithm is already registered: {name}")
        _ALGORITHMS[name] = factory


def create_importance_algorithm(name: str, topk_ratio: float, comparison_chunk_rows: int) -> ImportanceAlgorithm:
    with _LOCK:
        factory = _ALGORITHMS.get(name)
    if factory is None:
        raise ValueError(f"Unknown importance algorithm: {name}")
    return factory(topk_ratio, comparison_chunk_rows)


def _create_pretrained_delta(topk_ratio: float, comparison_chunk_rows: int) -> ImportanceAlgorithm:
    return PretrainedDeltaTopK(topk_ratio, comparison_chunk_rows)


register_importance_algorithm("pretrained_delta_topk", _create_pretrained_delta)
