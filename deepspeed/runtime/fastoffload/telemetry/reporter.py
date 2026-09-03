# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Periodic human-readable and structured FastOffload reporting."""

import csv
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from deepspeed.runtime.fastoffload.config import TelemetryConfig, TelemetryRankMode

from .metrics import MetricsSnapshot, RunningStatsSnapshot


def _format_bytes(byte_count: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(byte_count)
    unit = units[0]
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            break
        value /= 1024.0
    return f"{value:.2f} {unit}"


def _histogram_mean(snapshot: MetricsSnapshot, name: str) -> Optional[float]:
    stats: Optional[RunningStatsSnapshot] = snapshot.histograms.get(name)
    return None if stats is None else stats.mean


def _print_report(message: str) -> None:
    print(message, flush=True)


class TelemetryReporter:
    """Format metric windows without adding distributed collectives."""

    def __init__(self, config: TelemetryConfig, logging_fn: Optional[Callable[[str], None]] = None) -> None:
        self._config = config
        self._logging_fn = logging_fn or _print_report

    def should_report(self, completed_steps: int) -> bool:
        """Return whether a one-based completed-step count reaches the interval."""
        return completed_steps > 0 and completed_steps % self._config.log_interval == 0

    def report(self, snapshot: MetricsSnapshot, completed_steps: int, rank: int, final: bool = False) -> bool:
        """Emit one local report and return whether output was produced."""
        if self._config.rank_mode == TelemetryRankMode.rank0_local and rank != 0:
            return False

        lines = self.format_report(snapshot, completed_steps, rank, final=final)
        self._logging_fn("\n".join(lines))
        self._write_structured_reports(snapshot, completed_steps, rank, final)
        return True

    def _write_structured_reports(self, snapshot: MetricsSnapshot, completed_steps: int, rank: int,
                                  final: bool) -> None:
        if self._config.jsonl_path:
            path = self._rank_path(self._config.jsonl_path, rank)
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "timestamp_ns": time.time_ns(),
                "completed_steps": completed_steps,
                "rank": rank,
                "final": final,
                "counters": snapshot.counters,
                "gauges": snapshot.gauges,
                "histograms": {
                    name: {
                        "count": stats.count,
                        "total": stats.total,
                        "minimum": stats.minimum,
                        "maximum": stats.maximum,
                        "mean": stats.mean,
                    }
                    for name, stats in snapshot.histograms.items()
                },
            }
            with path.open("a", encoding="utf-8") as output_file:
                output_file.write(json.dumps(record, sort_keys=True) + "\n")
        if self._config.csv_path:
            self._write_csv(snapshot, completed_steps, rank, final)

    def _write_csv(self, snapshot: MetricsSnapshot, completed_steps: int, rank: int, final: bool) -> None:
        path = self._rank_path(self._config.csv_path, rank)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ("timestamp_ns", "completed_steps", "rank", "final", "metric_type", "metric_name", "value",
                      "count", "total", "minimum", "maximum", "mean")
        write_header = not path.exists() or path.stat().st_size == 0
        timestamp_ns = time.time_ns()
        rows: List[Dict[str, Any]] = []
        for metric_type, metrics in (("counter", snapshot.counters), ("gauge", snapshot.gauges)):
            for name, value in metrics.items():
                rows.append({
                    "timestamp_ns": timestamp_ns,
                    "completed_steps": completed_steps,
                    "rank": rank,
                    "final": final,
                    "metric_type": metric_type,
                    "metric_name": name,
                    "value": value,
                })
        for name, stats in snapshot.histograms.items():
            rows.append({
                "timestamp_ns": timestamp_ns,
                "completed_steps": completed_steps,
                "rank": rank,
                "final": final,
                "metric_type": "histogram",
                "metric_name": name,
                "count": stats.count,
                "total": stats.total,
                "minimum": stats.minimum,
                "maximum": stats.maximum,
                "mean": stats.mean,
            })
        with path.open("a", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def _rank_path(self, configured_path: str, rank: int) -> Path:
        path = Path(configured_path)
        if self._config.rank_mode == TelemetryRankMode.per_rank:
            path = path.with_name(f"{path.stem}.rank{rank}{path.suffix}")
        return path

    @staticmethod
    def format_report(snapshot: MetricsSnapshot, completed_steps: int, rank: int, final: bool = False) -> List[str]:
        """Build stable report lines for logging and tests."""
        heading = "[FastOffload Observer Final]" if final else "[FastOffload Observer]"
        lines = [heading, f"rank={rank} completed_steps={completed_steps}"]

        backward_ms = _histogram_mean(snapshot, "backward_host_ms")
        if backward_ms is not None:
            lines.append(f"backward_mean_ms={backward_ms:.2f}")

        optimizer_step_ms = _histogram_mean(snapshot, "optimizer_step_host_ms")
        if optimizer_step_ms is not None:
            lines.append(f"optimizer_step_mean_ms={optimizer_step_ms:.2f}")

        gradient_count = snapshot.counters.get("gradient_reduced_count", 0)
        potential_bytes = snapshot.counters.get("potential_offload_bytes", 0)
        lines.append(f"gradient_shards={gradient_count}")
        lines.append(f"potential_offload={_format_bytes(potential_bytes)}")

        actual_bytes = snapshot.counters.get("actual_offload_bytes")
        if actual_bytes is not None:
            lines.append(f"actual_offload={_format_bytes(actual_bytes)}")
        staging_ms = _histogram_mean(snapshot, "gpu_staging_copy_ms")
        if staging_ms is not None:
            lines.append(f"gpu_staging_copy_mean_ms={staging_ms:.2f}")
        submit_ms = _histogram_mean(snapshot, "copy_submit_host_ms")
        if submit_ms is not None:
            lines.append(f"copy_submit_host_mean_ms={submit_ms:.4f}")
        d2h_ms = _histogram_mean(snapshot, "d2h_copy_ms")
        if d2h_ms is not None:
            lines.append(f"d2h_copy_mean_ms={d2h_ms:.2f}")
        d2h_bandwidth = _histogram_mean(snapshot, "d2h_bandwidth_gbps")
        if d2h_bandwidth is not None:
            lines.append(f"d2h_bandwidth_mean_gbps={d2h_bandwidth:.2f}")
        worker_ms = _histogram_mean(snapshot, "inline_worker_ms")
        if worker_ms is not None:
            lines.append(f"inline_worker_mean_ms={worker_ms:.2f}")
        high_watermark = snapshot.gauges.get("buffer_pool_high_watermark")
        if high_watermark is not None:
            lines.append(f"buffer_pool_high_watermark={int(high_watermark)}")
        queue_high_watermark = snapshot.gauges.get("queue_high_watermark")
        if queue_high_watermark is not None:
            lines.append(f"queue_high_watermark={int(queue_high_watermark)}")
        queue_wait_ms = _histogram_mean(snapshot, "queue_wait_ms")
        if queue_wait_ms is not None:
            lines.append(f"queue_wait_mean_ms={queue_wait_ms:.2f}")
        step_flush_ms = _histogram_mean(snapshot, "step_flush_ms")
        if step_flush_ms is not None:
            lines.append(f"step_flush_mean_ms={step_flush_ms:.2f}")
        hidden_ratio = _histogram_mean(snapshot, "transfer_hidden_ratio")
        if hidden_ratio is not None:
            lines.append(f"transfer_hidden_ratio={hidden_ratio:.4f}")
        source_hold_ms = _histogram_mean(snapshot, "gpu_source_hold_ms")
        if source_hold_ms is not None:
            lines.append(f"gpu_source_hold_mean_ms={source_hold_ms:.2f}")
        pinned_peak_bytes = snapshot.gauges.get("pinned_pool_peak_bytes")
        if pinned_peak_bytes is not None:
            lines.append(f"pinned_pool_peak={_format_bytes(pinned_peak_bytes)}")
        gpu_staging_peak_bytes = snapshot.gauges.get("gpu_staging_pool_peak_bytes")
        if gpu_staging_peak_bytes is not None:
            lines.append(f"gpu_staging_pool_peak={_format_bytes(gpu_staging_peak_bytes)}")
        accelerator_peak_bytes = snapshot.gauges.get("accelerator_peak_allocated_bytes")
        if accelerator_peak_bytes is not None:
            lines.append(f"accelerator_peak_allocated={_format_bytes(accelerator_peak_bytes)}")

        importance_ready = snapshot.gauges.get("importance_ready")
        if importance_ready is not None:
            lines.append(f"importance_ready={int(importance_ready)}")
        reference_bytes = snapshot.gauges.get("importance_reference_bytes")
        if reference_bytes is not None:
            lines.append(f"importance_reference={_format_bytes(reference_bytes)}")
        selected_parameters = snapshot.counters.get("importance_selected_parameter_count")
        if selected_parameters is not None:
            lines.append(f"importance_selected_parameters={selected_parameters}")
            lines.append(f"importance_first_columns={snapshot.counters.get('importance_first_column_count', 0)}")
            lines.append(f"importance_second_columns={snapshot.counters.get('importance_second_column_count', 0)}")
        selection_ms = _histogram_mean(snapshot, "importance_selection_host_ms")
        if selection_ms is not None:
            lines.append(f"importance_selection_ms={selection_ms:.2f}")

        callback_ms = _histogram_mean(snapshot, "observer_callback_ms")
        if callback_ms is not None:
            lines.append(f"observer_callback_mean_ms={callback_ms:.4f}")
        return lines
