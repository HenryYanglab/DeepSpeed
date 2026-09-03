# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Summarize a matched Qwen Alpaca epoch comparison and draw its loss curve."""

import argparse
import json
import re
from pathlib import Path

LOSS_PATTERN = re.compile(r"\[Alpaca\] step=(\d+) epoch=\d+ loss=([0-9.]+)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    return parser.parse_args()


def read_benchmark(path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(records) != 1:
        raise ValueError(f"Expected one benchmark record in {path}, found {len(records)}")
    return records[0]


def read_losses(path):
    return [(int(step), float(loss)) for step, loss in LOSS_PATTERN.findall(path.read_text())]


def main():
    args = parse_args()
    fast_dir = args.result_dir / "fastoffload"
    zen_dir = args.result_dir / "zenflow"
    fast_benchmark = read_benchmark(fast_dir / "benchmark.jsonl")
    zen_benchmark = read_benchmark(zen_dir / "benchmark.jsonl")
    fast_losses = read_losses(fast_dir / "train.log")
    zen_losses = read_losses(zen_dir / "train.log")
    paired_count = min(len(fast_losses), len(zen_losses))
    fast_values = [loss for _, loss in fast_losses[:paired_count]]
    zen_values = [loss for _, loss in zen_losses[:paired_count]]
    mean_absolute_loss_difference = sum(abs(fast - zen) for fast, zen in zip(fast_values, zen_values)) / paired_count
    summary = {
        "sequence_length":
        512,
        "warmup_microsteps":
        20,
        "fastoffload":
        fast_benchmark,
        "zenflow":
        zen_benchmark,
        "throughput_speedup":
        fast_benchmark["padded_tokens_per_second"] / zen_benchmark["padded_tokens_per_second"],
        "peak_memory_saved_bytes_per_rank":
        zen_benchmark["accelerator_peak_allocated_bytes_per_rank"] -
        fast_benchmark["accelerator_peak_allocated_bytes_per_rank"],
        "paired_loss_windows":
        paired_count,
        "fastoffload_mean_loss":
        sum(fast_values) / paired_count,
        "zenflow_mean_loss":
        sum(zen_values) / paired_count,
        "mean_absolute_loss_difference":
        mean_absolute_loss_difference,
        "maximum_absolute_loss_difference":
        max(abs(fast - zen) for fast, zen in zip(fast_values, zen_values)),
    }
    (args.result_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    import matplotlib.pyplot as plt

    fast_x = [step for step, _ in fast_losses]
    zen_x = [step * 4 for step, _ in zen_losses]
    figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    axis.plot(fast_x, [loss for _, loss in fast_losses], color="#3B82B4", linewidth=1.5, label="FastOffload")
    axis.plot(zen_x, [loss for _, loss in zen_losses], color="#B77850", linewidth=1.5, label="ZenFlow")
    axis.set_xlabel("Training Microsteps")
    axis.set_ylabel("Mean Loss per 4 Microsteps")
    axis.set_title("Qwen2.5-7B Alpaca One-Epoch Training Loss")
    axis.grid(True, color="#D8DEE6", linewidth=0.7, alpha=0.75)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.savefig(args.result_dir / "loss_curve.png", dpi=300, facecolor="white")
    figure.savefig(args.result_dir / "loss_curve.svg", facecolor="white")
    plt.close(figure)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
