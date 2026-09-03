# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""DeepSpeed optimizer adapters for FastOffload."""

from .base import ObserverAdapter
from .zero2 import Zero2ObserverAdapter

__all__ = ["ObserverAdapter", "Zero2ObserverAdapter"]
