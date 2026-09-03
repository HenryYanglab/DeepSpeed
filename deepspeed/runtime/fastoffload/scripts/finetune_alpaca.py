#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Full-parameter Alpaca fine-tuning with ZeRO-2 CPU offload and FastOffload Observer."""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.fastoffload import (FastOffloadConfig, enable_selective_linear, install, selective_linear_stats)
from deepspeed.utils.pin_memory_tracker import pinned_memory_stats

PROMPT_WITH_INPUT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

PROMPT_WITHOUT_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


class SupervisedDataCollator:
    """Pad tokenized supervised examples without training on padding tokens."""

    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8) -> None:
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_length = max(len(example["input_ids"]) for example in examples)
        if self.pad_to_multiple_of > 1:
            remainder = max_length % self.pad_to_multiple_of
            if remainder:
                max_length += self.pad_to_multiple_of - remainder

        input_ids = []
        attention_masks = []
        labels = []
        for example in examples:
            padding_length = max_length - len(example["input_ids"])
            input_ids.append(example["input_ids"] + [self.pad_token_id] * padding_length)
            attention_masks.append(example["attention_mask"] + [0] * padding_length)
            labels.append(example["labels"] + [-100] * padding_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--dataset_name", default="tatsu-lab/alpaca")
    parser.add_argument("--dataset_path",
                        type=Path,
                        default=None,
                        help="Local JSON/JSONL file or datasets.save_to_disk directory; overrides --dataset_name")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--output_dir", type=Path, default=script_dir / "outputs" / "alpaca-observer")
    parser.add_argument("--deepspeed_config", type=Path, default=script_dir / "deepspeed_zero2_cpu_offload.json")
    parser.add_argument("--fastoffload_config", type=Path, default=script_dir / "fastoffload_observer.json")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--benchmark_jsonl", type=Path, default=None)
    parser.add_argument("--benchmark_warmup_steps", type=int, default=5)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--preprocessing_num_workers", type=int, default=1)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16"), default="auto")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--save_model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "-1")))
    args = parser.parse_args()

    positive_fields = ("max_length", "max_steps", "num_train_epochs", "micro_batch_size",
                       "gradient_accumulation_steps", "log_interval")
    for field in positive_fields:
        if getattr(args, field) < 1:
            parser.error(f"--{field} must be at least 1")
    if args.benchmark_warmup_steps < 0:
        parser.error("--benchmark_warmup_steps must be at least 0")
    if args.benchmark_jsonl is not None and args.benchmark_warmup_steps >= args.max_steps:
        parser.error("--benchmark_warmup_steps must be smaller than --max_steps")
    return args


def load_alpaca_dataset(args: argparse.Namespace) -> Dataset:
    if args.dataset_path is not None:
        if args.dataset_path.is_dir():
            saved_dataset = load_from_disk(str(args.dataset_path))
            if isinstance(saved_dataset, DatasetDict):
                if args.dataset_split not in saved_dataset:
                    available_splits = sorted(saved_dataset.keys())
                    raise ValueError(
                        f"Dataset split '{args.dataset_split}' is unavailable; available splits: {available_splits}")
                dataset = saved_dataset[args.dataset_split]
            else:
                dataset = saved_dataset
        elif args.dataset_path.is_file():
            dataset = load_dataset("json", data_files=str(args.dataset_path), split=args.dataset_split)
        else:
            raise FileNotFoundError(f"Dataset path does not exist: {args.dataset_path}")
    else:
        dataset = load_dataset(args.dataset_name, split=args.dataset_split)

    required_columns = {"instruction", "output"}
    missing_columns = required_columns.difference(dataset.column_names)
    if missing_columns:
        raise ValueError(f"Alpaca dataset is missing columns: {sorted(missing_columns)}")

    if args.max_samples is not None:
        sample_count = min(args.max_samples, len(dataset))
        dataset = dataset.shuffle(seed=args.seed).select(range(sample_count))
    return dataset


def tokenize_dataset(dataset: Dataset, tokenizer, args: argparse.Namespace) -> Dataset:
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define an EOS token")

    def tokenize_example(example):
        instruction = str(example["instruction"]).strip()
        additional_input = str(example.get("input") or "").strip()
        response = str(example["output"]).strip()
        if additional_input:
            prompt = PROMPT_WITH_INPUT.format(instruction=instruction, input=additional_input)
        else:
            prompt = PROMPT_WITHOUT_INPUT.format(instruction=instruction)

        prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False, truncation=False)["input_ids"]
        response_ids.append(tokenizer.eos_token_id)
        input_ids = (prompt_ids + response_ids)[:args.max_length]
        labels = ([-100] * len(prompt_ids) + response_ids)[:args.max_length]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    num_proc = args.preprocessing_num_workers if args.preprocessing_num_workers > 1 else None
    tokenized = dataset.map(tokenize_example,
                            remove_columns=dataset.column_names,
                            num_proc=num_proc,
                            desc="Tokenizing Alpaca")
    return tokenized.filter(lambda example: any(label != -100 for label in example["labels"]),
                            desc="Removing prompt-only examples")


def load_deepspeed_config(args: argparse.Namespace) -> dict:
    with args.deepspeed_config.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    config["train_micro_batch_size_per_gpu"] = args.micro_batch_size
    config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    config["train_batch_size"] = args.micro_batch_size * args.gradient_accumulation_steps * world_size
    config["steps_per_print"] = args.log_interval
    config["optimizer"]["params"]["lr"] = args.learning_rate

    precision = args.precision
    if precision == "auto":
        precision = "bf16" if get_accelerator().is_bf16_supported() else "fp16"
    config.pop("bf16", None)
    config.pop("fp16", None)
    config[precision] = {"enabled": True}
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    get_accelerator().manual_seed_all(seed)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    observer_config = FastOffloadConfig.from_json(args.fastoffload_config)
    observer_handle = install(observer_config)
    engine = None

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                  trust_remote_code=args.trust_remote_code,
                                                  local_files_only=args.local_files_only,
                                                  use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        dataset = load_alpaca_dataset(args)
        dataset = tokenize_dataset(dataset, tokenizer, args)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                     trust_remote_code=args.trust_remote_code,
                                                     local_files_only=args.local_files_only)
        model.config.use_cache = False
        model.config.pad_token_id = tokenizer.pad_token_id
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        if observer_config.importance.enabled and observer_config.importance.sparse_backward:
            wrapped_linears = enable_selective_linear(
                model,
                minimum_tokens=observer_config.importance.sparse_backward_min_tokens,
                minimum_output_features=observer_config.importance.sparse_backward_min_output_features)
            if int(os.getenv("LOCAL_RANK", "0")) == 0:
                print(f"[FastOffload Importance] selective_linear_modules={wrapped_linears}", flush=True)

        ds_config = load_deepspeed_config(args)
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True,
                                     seed=args.seed) if world_size > 1 else None
        data_loader = DataLoader(dataset,
                                 batch_size=args.micro_batch_size,
                                 sampler=sampler,
                                 shuffle=sampler is None,
                                 num_workers=args.dataloader_num_workers,
                                 pin_memory=True,
                                 collate_fn=SupervisedDataCollator(tokenizer.pad_token_id))

        engine.train()
        completed_steps = 0
        loss_since_log = 0.0
        micro_steps_since_log = 0
        measured_padded_tokens = 0
        benchmark_start = None
        benchmark_enabled = args.benchmark_jsonl is not None
        accelerator_baseline_bytes = 0
        if benchmark_enabled:
            accelerator_baseline_bytes = get_accelerator().memory_allocated()
            get_accelerator().reset_peak_memory_stats()
            if args.benchmark_warmup_steps == 0:
                dist.barrier()
                benchmark_start = time.perf_counter()
        for epoch in range(args.num_train_epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            for batch in data_loader:
                if benchmark_enabled and completed_steps >= args.benchmark_warmup_steps:
                    measured_padded_tokens += batch["input_ids"].numel() * world_size
                batch = {name: tensor.to(engine.device, non_blocking=True) for name, tensor in batch.items()}
                outputs = engine(**batch, use_cache=False)
                loss = outputs.loss
                loss_since_log += loss.detach().float().item()
                micro_steps_since_log += 1
                engine.backward(loss)
                is_update_boundary = engine.is_gradient_accumulation_boundary()
                engine.step()

                if is_update_boundary:
                    completed_steps += 1
                    if rank == 0 and completed_steps % args.log_interval == 0:
                        mean_loss = loss_since_log / micro_steps_since_log
                        print(f"[Alpaca] step={completed_steps} epoch={epoch + 1} loss={mean_loss:.6f}", flush=True)
                        loss_since_log = 0.0
                        micro_steps_since_log = 0
                    if benchmark_enabled and completed_steps == args.benchmark_warmup_steps:
                        dist.barrier()
                        benchmark_start = time.perf_counter()
                    if completed_steps >= args.max_steps:
                        break
            if completed_steps >= args.max_steps:
                break

        if benchmark_enabled:
            if benchmark_start is None:
                raise RuntimeError("Training ended before the benchmark warmup completed")
            dist.barrier()
            benchmark_elapsed = time.perf_counter() - benchmark_start
            measured_steps = completed_steps - args.benchmark_warmup_steps
            accelerator_peak_bytes = get_accelerator().max_memory_allocated()
            pinning_stats = pinned_memory_stats()
            benchmark_mode = observer_config.mode.value if observer_config.enabled else "native_zero2"
            if observer_config.importance.sparse_backward:
                benchmark_mode = "selective_backward"
            benchmark_record = {
                "mode": benchmark_mode,
                "model": args.model_name_or_path,
                "world_size": world_size,
                "completed_steps": completed_steps,
                "warmup_steps": args.benchmark_warmup_steps,
                "measured_steps": measured_steps,
                "elapsed_seconds": benchmark_elapsed,
                "steps_per_second": measured_steps / benchmark_elapsed,
                "padded_tokens_per_second": measured_padded_tokens / benchmark_elapsed,
                "accelerator_baseline_allocated_bytes_per_rank": accelerator_baseline_bytes,
                "accelerator_peak_allocated_bytes_per_rank": accelerator_peak_bytes,
                "accelerator_incremental_peak_bytes_per_rank": accelerator_peak_bytes - accelerator_baseline_bytes,
                "pin_memory": ds_config["zero_optimization"]["offload_optimizer"].get("pin_memory", False),
                "overlap_comm": ds_config["zero_optimization"].get("overlap_comm", False),
                "cumulative_pinned_bytes_per_rank": pinning_stats["cumulative_bytes"],
                "pin_allocation_calls_per_rank": pinning_stats["allocation_calls"],
                "timestamp_ns": time.time_ns(),
            }
            controller = getattr(engine.optimizer, "_fastoffload_controller", None)
            observer = getattr(controller, "observer", None)
            if observer is not None:
                gauges = observer.snapshot().gauges
                benchmark_record["pinned_pool_peak_bytes_per_rank"] = gauges.get("pinned_pool_peak_bytes", 0)
                benchmark_record["gpu_staging_pool_peak_bytes_per_rank"] = gauges.get("gpu_staging_pool_peak_bytes", 0)
            if observer_config.importance.sparse_backward:
                benchmark_record.update(selective_linear_stats(engine.module))
            if rank == 0:
                args.benchmark_jsonl.parent.mkdir(parents=True, exist_ok=True)
                with args.benchmark_jsonl.open("a", encoding="utf-8") as output_file:
                    output_file.write(json.dumps(benchmark_record, sort_keys=True) + "\n")
                print(f"[Alpaca Benchmark] {json.dumps(benchmark_record, sort_keys=True)}", flush=True)

        if rank == 0:
            print(f"[Alpaca] training complete: optimizer_steps={completed_steps}", flush=True)
        if args.save_model:
            dist.barrier()
            if rank == 0:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                engine.module.save_pretrained(args.output_dir, safe_serialization=True)
                tokenizer.save_pretrained(args.output_dir)
            dist.barrier()
    finally:
        if engine is not None:
            engine.destroy()
        observer_handle.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
