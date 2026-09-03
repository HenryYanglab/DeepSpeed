# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Column selection based on movement from pretrained parameters."""

import math

import torch

from .base import ImportanceAlgorithm
from .context import ColumnImportanceSelection, ParameterImportanceView


class PretrainedDeltaTopK(ImportanceAlgorithm):
    """Rank columns by their L1 distance from the pretrained checkpoint."""

    def __init__(self, topk_ratio: float, comparison_chunk_rows: int) -> None:
        if not 0.0 < topk_ratio <= 0.5:
            raise ValueError("topk_ratio must be in (0, 0.5]")
        if comparison_chunk_rows < 1:
            raise ValueError("comparison_chunk_rows must be greater than zero")
        self._topk_ratio = topk_ratio
        self._comparison_chunk_rows = comparison_chunk_rows

    @property
    def name(self) -> str:
        return "pretrained_delta_topk"

    def select(self, view: ParameterImportanceView, reference: torch.Tensor,
               current: torch.Tensor) -> ColumnImportanceSelection:
        if reference.device.type != "cpu":
            raise ValueError("Pretrained reference must remain on CPU")
        if reference.shape != current.shape or tuple(current.shape) != view.shape:
            raise ValueError("Reference and current parameter shapes must match")
        if current.dim() != 2:
            raise ValueError("Column selection requires a two-dimensional parameter")

        rows, columns = current.shape
        score_device = current.device
        scores = torch.zeros(columns, dtype=torch.float32, device=score_device)
        for row_start in range(0, rows, self._comparison_chunk_rows):
            row_count = min(self._comparison_chunk_rows, rows - row_start)
            current_chunk = current.narrow(0, row_start, row_count).float()
            reference_chunk = reference.narrow(0, row_start, row_count).to(device=score_device, dtype=torch.float32)
            current_chunk.sub_(reference_chunk).abs_()
            scores.add_(current_chunk.sum(dim=0))

        first_count = max(1, math.ceil(columns * self._topk_ratio))
        selected_count = min(columns, first_count * 2)
        selected_scores, selected_indices = torch.topk(scores, selected_count, largest=True, sorted=True)
        selected_scores = selected_scores.cpu()
        selected_indices = selected_indices.cpu()
        first_indices = selected_indices[:first_count].sort().values
        second_indices = selected_indices[first_count:].sort().values
        first_min_score = float(selected_scores[first_count - 1].item())
        second_min_score = 0.0
        if second_indices.numel():
            second_min_score = float(selected_scores[-1].item())

        return ColumnImportanceSelection(parameter_id=view.parameter_id,
                                         parameter_name=view.parameter_name,
                                         shape=view.shape,
                                         first_indices=first_indices,
                                         second_indices=second_indices,
                                         first_min_score=first_min_score,
                                         second_min_score=second_min_score,
                                         algorithm=self.name)
