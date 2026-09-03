#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Validate native-versus-shadow losses and compressed-collective telemetry."""

import argparse
import json
import re
from pathlib import Path

LOSS_PATTERN = re.compile(r"\[Alpaca\] step=(\d+).* loss=([+-]?[0-9.eE-]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-log", type=Path, required=True)
    parser.add_argument("--shadow-log", type=Path, required=True)
    parser.add_argument("--native-telemetry", type=Path, required=True)
    parser.add_argument("--shadow-telemetry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--loss-tolerance", type=float, default=1e-5)
    return parser.parse_args()


def read_losses(path: Path) -> dict[int, float]:
    losses = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LOSS_PATTERN.search(line)
        if match:
            losses[int(match.group(1))] = float(match.group(2))
    if not losses:
        raise RuntimeError(f"No Alpaca step losses found in {path}")
    return losses


def aggregate_telemetry(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    counters = {}
    gauges = {}
    with path.open("r", encoding="utf-8") as telemetry_file:
        for line in telemetry_file:
            record = json.loads(line)
            for name, value in record.get("counters", {}).items():
                counters[name] = counters.get(name, 0) + value
            for name, value in record.get("gauges", {}).items():
                gauges[name] = max(gauges.get(name, value), value)
    return counters, gauges


def validate(args: argparse.Namespace) -> dict:
    native_losses = read_losses(args.native_log)
    shadow_losses = read_losses(args.shadow_log)
    if native_losses.keys() != shadow_losses.keys():
        raise RuntimeError("Native and shadow runs reported different optimizer steps")
    loss_differences = {step: abs(native_losses[step] - shadow_losses[step]) for step in native_losses}
    maximum_loss_difference = max(loss_differences.values())
    if maximum_loss_difference > args.loss_tolerance:
        raise RuntimeError(
            f"Shadow changed training loss: maximum difference {maximum_loss_difference} exceeds {args.loss_tolerance}"
        )

    native_counters, _ = aggregate_telemetry(args.native_telemetry)
    shadow_counters, shadow_gauges = aggregate_telemetry(args.shadow_telemetry)
    native_shadow_collectives = native_counters.get("hybrid_shadow_collective_count", 0)
    if native_shadow_collectives:
        raise RuntimeError("Native control unexpectedly executed a hybrid shadow collective")

    collective_count = shadow_counters.get("hybrid_shadow_collective_count", 0)
    selected_count = shadow_counters.get("hybrid_shadow_selected_collective_count", 0)
    dense_count = shadow_counters.get("hybrid_shadow_dense_collective_count", 0)
    selected_bytes = shadow_counters.get("hybrid_shadow_selected_communicated_bytes", 0)
    dense_bytes = shadow_counters.get("hybrid_shadow_dense_communicated_bytes", 0)
    if collective_count < 1 or selected_count < 1 or dense_count < 1:
        raise RuntimeError("Shadow run did not execute both selected and dense-boundary collectives")
    if selected_bytes <= 0 or dense_bytes <= 0:
        raise RuntimeError("Shadow collective byte telemetry is missing")
    if selected_bytes >= dense_bytes:
        raise RuntimeError("Selected-step communication was not smaller than dense-boundary communication")

    required_checks = ("parity", "norm", "overflow", "owner_write")
    for check in required_checks:
        if shadow_counters.get(f"hybrid_shadow_{check}_check_count", 0) < 1:
            raise RuntimeError(f"Shadow run did not execute the {check} check")
        if shadow_counters.get(f"hybrid_shadow_{check}_mismatch_count", 0):
            raise RuntimeError(f"Shadow {check} validation reported a mismatch")
    bucket_peak_bytes = shadow_gauges.get("hybrid_shadow_bucket_peak_bytes", 0)
    reduced_resident_peak_bytes = shadow_gauges.get("hybrid_shadow_reduced_cpu_resident_peak_bytes", 0)
    estimated_peak_live_bytes = shadow_gauges.get("hybrid_shadow_estimated_peak_live_bytes", 0)
    if bucket_peak_bytes <= 0 or estimated_peak_live_bytes <= 0 or estimated_peak_live_bytes >= dense_bytes:
        raise RuntimeError("Chunked shadow buffers were not bounded below the full dense communication size")

    summary = {
        "status": "passed",
        "steps": sorted(native_losses),
        "native_losses": native_losses,
        "shadow_losses": shadow_losses,
        "maximum_loss_difference": maximum_loss_difference,
        "loss_tolerance": args.loss_tolerance,
        "shadow_collective_count": collective_count,
        "selected_collective_count": selected_count,
        "dense_collective_count": dense_count,
        "selected_communicated_bytes": selected_bytes,
        "dense_communicated_bytes": dense_bytes,
        "selected_to_dense_byte_ratio": selected_bytes / dense_bytes,
        "bucket_peak_bytes": bucket_peak_bytes,
        "reduced_cpu_resident_peak_bytes": reduced_resident_peak_bytes,
        "estimated_peak_live_bytes": estimated_peak_live_bytes,
        "parity_check_count": shadow_counters["hybrid_shadow_parity_check_count"],
        "norm_check_count": shadow_counters["hybrid_shadow_norm_check_count"],
        "overflow_check_count": shadow_counters["hybrid_shadow_overflow_check_count"],
        "owner_write_check_count": shadow_counters["hybrid_shadow_owner_write_check_count"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = validate(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
