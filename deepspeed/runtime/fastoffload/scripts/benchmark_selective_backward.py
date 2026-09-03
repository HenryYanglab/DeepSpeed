# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Benchmark native and selected-column Linear dW kernels on Qwen-like shapes."""

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

from deepspeed.accelerator import get_accelerator


@dataclass(frozen=True)
class LinearShape:
    name: str
    input_features: int
    output_features: int


@dataclass(frozen=True)
class BenchmarkResult:
    shape: str
    tokens: int
    input_features: int
    output_features: int
    selected_features: int
    selected_ratio: float
    dtype: str
    grad_input_ms: float
    native_dw_ms: float
    gather_ms: float
    selected_dw_ms: float
    selected_total_ms: float
    selected_dw_speedup: float
    estimated_backward_speedup: float
    gather_fraction: float
    maximum_difference: float


QWEN_7B_SHAPES = (
    LinearShape("attention_q", 3584, 3584),
    LinearShape("attention_kv", 3584, 512),
    LinearShape("mlp_gate_up", 3584, 18944),
    LinearShape("mlp_down", 18944, 3584),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[128, 323, 512])
    parser.add_argument("--selected-ratio", type=float, default=0.2)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def timed_ms(operation: Callable[[], None], warmup: int, iterations: int) -> float:
    accelerator = get_accelerator()
    for _ in range(warmup):
        operation()
    accelerator.synchronize()
    samples = []
    for _ in range(iterations):
        start = accelerator.Event(enable_timing=True)
        end = accelerator.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def benchmark_shape(shape: LinearShape, tokens: int, selected_ratio: float, dtype: torch.dtype, warmup: int,
                    iterations: int, seed: int) -> BenchmarkResult:
    if tokens < 1:
        raise ValueError("tokens must be positive")
    if not 0.0 < selected_ratio <= 1.0:
        raise ValueError("selected_ratio must be in (0, 1]")
    accelerator = get_accelerator()
    device = accelerator.current_device_name()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    inputs = torch.randn((tokens, shape.input_features), dtype=dtype, device=device, generator=generator)
    grad_output = torch.randn((tokens, shape.output_features), dtype=dtype, device=device, generator=generator)
    weight = torch.randn((shape.output_features, shape.input_features),
                         dtype=dtype,
                         device=device,
                         generator=generator)
    selected_features = max(1, round(shape.input_features * selected_ratio))
    columns = torch.randperm(shape.input_features, device=device,
                             generator=generator)[:selected_features].sort().values
    gathered_inputs = torch.empty((tokens, selected_features), dtype=dtype, device=device)
    native_output = torch.empty((shape.output_features, shape.input_features), dtype=dtype, device=device)
    selected_output = torch.empty((shape.output_features, selected_features), dtype=dtype, device=device)
    grad_input_output = torch.empty((tokens, shape.input_features), dtype=dtype, device=device)

    def grad_input():
        torch.mm(grad_output, weight, out=grad_input_output)

    def gather():
        torch.index_select(inputs, 1, columns, out=gathered_inputs)

    def native_dw():
        torch.mm(grad_output.transpose(0, 1), inputs, out=native_output)

    def selected_dw():
        torch.mm(grad_output.transpose(0, 1), gathered_inputs, out=selected_output)

    def selected_total():
        gather()
        selected_dw()

    grad_input_ms = timed_ms(grad_input, warmup, iterations)
    native_ms = timed_ms(native_dw, warmup, iterations)
    gather_ms = timed_ms(gather, warmup, iterations)
    selected_ms = timed_ms(selected_dw, warmup, iterations)
    selected_total_ms = timed_ms(selected_total, warmup, iterations)
    native_dw()
    selected_total()
    accelerator.synchronize()
    expected = native_output.index_select(1, columns)
    maximum_difference = (expected.float() - selected_output.float()).abs().max().item()
    speedup = native_ms / selected_total_ms
    estimated_backward_speedup = (grad_input_ms + native_ms) / (grad_input_ms + selected_total_ms)
    gather_fraction = gather_ms / selected_total_ms
    return BenchmarkResult(shape=shape.name,
                           tokens=tokens,
                           input_features=shape.input_features,
                           output_features=shape.output_features,
                           selected_features=selected_features,
                           selected_ratio=selected_features / shape.input_features,
                           dtype=str(dtype).removeprefix("torch."),
                           grad_input_ms=grad_input_ms,
                           native_dw_ms=native_ms,
                           gather_ms=gather_ms,
                           selected_dw_ms=selected_ms,
                           selected_total_ms=selected_total_ms,
                           selected_dw_speedup=speedup,
                           estimated_backward_speedup=estimated_backward_speedup,
                           gather_fraction=gather_fraction,
                           maximum_difference=maximum_difference)


def main():
    args = parse_args()
    if not get_accelerator().is_available():
        raise RuntimeError("Selective backward benchmark requires an accelerator")
    dtype = getattr(torch, args.dtype)
    results = []
    for tokens in args.tokens:
        for index, shape in enumerate(QWEN_7B_SHAPES):
            result = benchmark_shape(shape, tokens, args.selected_ratio, dtype, args.warmup, args.iterations,
                                     args.seed + index)
            results.append(result)
            print(
                f"{result.shape:12s} tokens={result.tokens:4d} native={result.native_dw_ms:8.3f} ms "
                f"gather={result.gather_ms:7.3f} ms selected={result.selected_dw_ms:8.3f} ms "
                f"total={result.selected_total_ms:8.3f} ms dW_speedup={result.selected_dw_speedup:5.2f}x "
                f"backward_speedup={result.estimated_backward_speedup:5.2f}x "
                f"gather_fraction={result.gather_fraction:5.1%} max_diff={result.maximum_difference:.6f}",
                flush=True)
    payload = {
        "device": get_accelerator().device_name(),
        "selected_ratio": args.selected_ratio,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "results": [asdict(result) for result in results],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
