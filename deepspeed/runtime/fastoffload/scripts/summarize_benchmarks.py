#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Summarize native, synchronous, and asynchronous FastOffload benchmark runs."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def histogram_mean(records: Iterable[dict], name: str) -> Optional[float]:
    total = 0.0
    count = 0
    for record in records:
        stats = record.get("histograms", {}).get(name)
        if stats:
            total += stats["total"]
            count += stats["count"]
    return total / count if count else None


def maximum_gauge(records: Iterable[dict], name: str) -> Optional[float]:
    values = [record.get("gauges", {}).get(name) for record in records]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def collect_run(path: Path) -> dict:
    benchmark_records = read_jsonl(path)
    if len(benchmark_records) != 1:
        raise ValueError(f"Expected one benchmark record in {path}, found {len(benchmark_records)}")
    record = dict(benchmark_records[0])
    telemetry = read_jsonl(path.parent / "telemetry.jsonl")
    warmup_steps = record.get("warmup_steps", 0)
    telemetry = [entry for entry in telemetry if entry.get("completed_steps", 0) > warmup_steps]
    record["run_dir"] = str(path.parent)
    record["step_time_ms"] = 1000.0 / record["steps_per_second"]
    for metric in ("backward_host_ms", "optimizer_step_host_ms", "gpu_staging_copy_ms", "copy_submit_host_ms",
                   "d2h_copy_ms", "d2h_bandwidth_gbps", "inline_worker_ms", "step_flush_ms", "transfer_hidden_ratio",
                   "gpu_source_hold_ms", "queue_wait_ms"):
        record[metric] = histogram_mean(telemetry, metric)
    for metric in ("pinned_pool_peak_bytes", "gpu_staging_pool_peak_bytes", "queue_high_watermark",
                   "buffer_pool_high_watermark"):
        record[metric] = maximum_gauge(telemetry, metric)
    return record


def numeric_mean(records: List[dict], field: str) -> Optional[float]:
    values = [record[field] for record in records if record.get(field) is not None]
    return statistics.mean(values) if values else None


def numeric_stdev(records: List[dict], field: str) -> Optional[float]:
    values = [record[field] for record in records if record.get(field) is not None]
    return statistics.stdev(values) if len(values) > 1 else 0.0 if values else None


def write_csv(path: Path, records: List[dict], fields: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def format_number(value: Optional[float], digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()

    run_paths = sorted(args.result_dir.glob("**/benchmark.jsonl"))
    if not run_paths:
        raise FileNotFoundError(f"No benchmark.jsonl files found under {args.result_dir}")
    runs = [collect_run(path) for path in run_paths]
    run_fields = [
        "mode", "run_dir", "measured_steps", "elapsed_seconds", "step_time_ms", "steps_per_second",
        "padded_tokens_per_second", "backward_host_ms", "optimizer_step_host_ms", "gpu_staging_copy_ms",
        "copy_submit_host_ms", "d2h_copy_ms", "d2h_bandwidth_gbps", "inline_worker_ms", "step_flush_ms",
        "transfer_hidden_ratio", "gpu_source_hold_ms", "queue_wait_ms",
        "accelerator_baseline_allocated_bytes_per_rank", "accelerator_peak_allocated_bytes_per_rank",
        "accelerator_incremental_peak_bytes_per_rank", "cumulative_pinned_bytes_per_rank", "pinned_pool_peak_bytes",
        "gpu_staging_pool_peak_bytes", "queue_high_watermark", "buffer_pool_high_watermark", "sparse_forward_count",
        "native_forward_count"
    ]
    write_csv(args.result_dir / "runs.csv", runs, run_fields)

    grouped: Dict[str, List[dict]] = defaultdict(list)
    for record in runs:
        grouped[record["mode"]].append(record)
    aggregate_fields = [
        "step_time_ms", "steps_per_second", "padded_tokens_per_second", "backward_host_ms", "optimizer_step_host_ms",
        "gpu_staging_copy_ms", "copy_submit_host_ms", "d2h_copy_ms", "d2h_bandwidth_gbps", "inline_worker_ms",
        "step_flush_ms", "transfer_hidden_ratio", "accelerator_incremental_peak_bytes_per_rank",
        "cumulative_pinned_bytes_per_rank", "pinned_pool_peak_bytes", "gpu_staging_pool_peak_bytes",
        "sparse_forward_count", "native_forward_count"
    ]
    summaries = []
    for mode, records in sorted(grouped.items()):
        summary = {"mode": mode, "runs": len(records)}
        for field in aggregate_fields:
            summary[f"{field}_mean"] = numeric_mean(records, field)
            summary[f"{field}_stdev"] = numeric_stdev(records, field)
        summaries.append(summary)
    summary_fields = ["mode", "runs"]
    for field in aggregate_fields:
        summary_fields.extend((f"{field}_mean", f"{field}_stdev"))
    write_csv(args.result_dir / "summary.csv", summaries, summary_fields)

    lines = [
        "# FastOffload Benchmark Summary", "",
        "| Mode | Runs | Step ms | Steps/s | Padded tokens/s | D2H ms | Hidden ratio | GPU incremental GiB | Pinned cumulative GiB | GPU staging GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    gib = 1024**3
    for summary in summaries:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            summary["mode"], summary["runs"], format_number(summary["step_time_ms_mean"]),
            format_number(summary["steps_per_second_mean"]), format_number(summary["padded_tokens_per_second_mean"],
                                                                           1),
            format_number(summary["d2h_copy_ms_mean"]), format_number(summary["transfer_hidden_ratio_mean"]),
            format_number((summary["accelerator_incremental_peak_bytes_per_rank_mean"] or 0) / gib),
            format_number((summary["cumulative_pinned_bytes_per_rank_mean"] or 0) / gib),
            format_number((summary["gpu_staging_pool_peak_bytes_mean"] or 0) / gib)))
    markdown = "\n".join(lines) + "\n"
    (args.result_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Detailed runs: {args.result_dir / 'runs.csv'}")
    print(f"Aggregate CSV: {args.result_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
