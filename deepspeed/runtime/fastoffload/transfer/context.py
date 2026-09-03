# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Callback-scoped tensor views used by transfer implementations."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GradientTransferView:
    """Source and destination views valid only during a transfer callback."""

    source: Any
    destination: Any
    parameter_id: int
    group_id: int
    source_offset: int
    destination_offset: int
    numel: int

    @property
    def nbytes(self) -> int:
        return self.numel * self.destination.element_size()
