#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Diagnostic GPU-only ceiling, NOT a training algorithm or checkpoint producer."""

import json
import time
from collections import Counter
from itertools import islice

import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.fastoffload import FastOffloadConfig, enable_selective_linear, install
from deepspeed.runtime.fastoffload import api
from deepspeed.runtime.fastoffload.hybrid.numerics import HybridGradientNumerics, HybridNumericsResult
from deepspeed.runtime.fastoffload.hybrid.takeover import TakeoverGradientBatch
from deepspeed.runtime.fastoffload.hybrid.takeover_runtime import TakeoverStepResult, Zero2TakeoverRuntime


def _forbid_cpu_work(*args, **kwargs):
    raise RuntimeError("GPU ceiling must not initialize, transfer, accumulate, or update B/C on CPU")


class GpuCeilingTakeoverRuntime(Zero2TakeoverRuntime):
    """Retain A updates and O/D gradient computation, discard B/C after reduction.

    Installed only by this benchmark's temporary factory override. Normal
    FastOffload still uses the complete Hybrid A/B/C algorithm.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counts = Counter()
        # Fail loudly if inherited code accidentally reactivates a host path.
        for name in ("initialize_state", "finalize_state_initialization", "accumulate_second", "submit_boundary",
                     "stage_native_dense_gradient", "record_native_transfer_event", "_initialize_native_optimizer",
                     "_prepare_gradients", "_update", "_native_step"):
            setattr(self._cpu_updater, name, _forbid_cpu_work)

    def _initialize_from_dense_adam(self, device):
        if self._state_initialized:
            return
        if not self._importance_registry.ready:
            raise RuntimeError("Importance warmup must finish before the GPU ceiling")
        states = {}
        for parameter_id, layout in self._layouts.items():
            selection = self._importance_registry.get(parameter_id)
            if selection is None:
                continue
            if layout.group_id not in states:
                states[layout.group_id] = self._adapter.get_dense_adam_partition_state(layout.group_id)
            step, exp_avg, exp_avg_sq = states[layout.group_id]
            self._gpu_updater.initialize_state(parameter_id, selection.first_indices, step, exp_avg, exp_avg_sq,
                                               device)
        self._state_initialized = True

    def capture_native_reduced_gradient(self, parameter):
        if not self.native_dense_boundary:
            return False
        parameter_id = self._adapter.get_parameter_id(parameter)
        layout = self._layouts.get(parameter_id)
        if layout is None:
            return False
        selection = self._importance_registry.get(parameter_id)
        if selection is not None:
            rank = self._adapter.get_partition_rank(layout.group_id)
            fragment = self._adapter.get_local_reduced_gradient(parameter)
            columns = selection.first_indices
            values = layout.extract_fragment_values(fragment, columns, rank)
            self._native_first.setdefault(layout.group_id, {})[parameter_id] = values
            self._native_columns[0][parameter_id] = columns
        # Native norm/overflow tracking precedes this hook. C need not be retained
        # on either device when its optimizer and publication are excluded.
        return True

    def finish_microbatch(self):
        batch = self._pipeline.finish_microbatch()
        if batch is None or not self.native_dense_boundary:
            return batch
        return TakeoverGradientBatch(first={},
                                     second={},
                                     dense={},
                                     first_columns={},
                                     second_columns={},
                                     dense_columns={},
                                     dense_boundary=True,
                                     native_boundary=True)

    def _native_first_gradients(self, combined_scale):
        for parameter_id, layout in self._layouts.items():
            selection = self._importance_registry.get(parameter_id)
            if selection is None:
                continue
            columns = selection.first_indices
            values = self._native_first.setdefault(layout.group_id, {})
            self._native_columns[0][parameter_id] = columns
            if parameter_id not in values:
                rank = self._adapter.get_partition_rank(layout.group_id)
                if layout.owner_counts(columns)[rank]:
                    raise RuntimeError(f"Missing native A owner gradient for {parameter_id}")
                parameter = self._adapter.get_parameter(parameter_id)
                values[parameter_id] = torch.empty(0, dtype=parameter.dtype, device=parameter.device)
        reduced = {}
        for group_id, values in sorted(self._native_first.items()):
            for value in values.values():
                value.mul_(1.0 / combined_scale)
            columns = {parameter_id: self._native_columns[0][parameter_id] for parameter_id in values}
            reduced[group_id] = self._collectives[group_id].create_owner_gradient_set(
                columns, values, self._compressed_bucket_bytes)
        return reduced

    def step(self, batch, loss_scale, clip_grad):
        if batch.native_boundary:
            overflow, norm, scale = self._adapter.compute_native_gradient_numerics(loss_scale, clip_grad)
            numerics = HybridNumericsResult(overflow, norm, scale)
        else:
            gradients = []
            for band in (batch.first, batch.second, batch.dense):
                for reduced in band.values():
                    gradients.extend(reduced.values.values())
            numerics = HybridGradientNumerics.unscale_and_clip(gradients,
                                                               loss_scale,
                                                               clip_grad,
                                                               process_group=self._adapter.get_data_parallel_group())
        if numerics.overflow:
            self._clear_native_capture()
            self._pipeline.complete_step(overflow=True)
            self.counts["overflow_steps"] += 1
            return TakeoverStepResult(numerics, 0, None)
        if batch.native_boundary:
            first = self._native_first_gradients(scale)
            columns = self._native_columns[0]
        else:
            first, columns = batch.first, batch.first_columns
        updated = self._gpu_updater.step(first, columns)
        if batch.native_boundary:
            get_accelerator().synchronize()
        self._clear_native_capture()
        self._pipeline.complete_step(overflow=False)
        self.counts["dense_steps" if batch.dense_boundary else "ordinary_steps"] += 1
        self.counts["a_update_steps"] += 1
        return TakeoverStepResult(numerics, updated, None)

    def assert_no_cpu_work(self):
        if self.pending_updates or self._cpu_updater._native_process is not None:
            raise RuntimeError("Unexpected CPU job in GPU ceiling")
        if self._cpu_updater._optimizer.state_keys() or self._cpu_updater._native_registered_buffers:
            raise RuntimeError("Unexpected B/C state or registered transfer buffer in GPU ceiling")

    def state_dict(self):
        raise RuntimeError("GPU ceiling results must not be saved as Hybrid training checkpoints")

    def load_state_dict(self, state_dict):
        raise RuntimeError("GPU ceiling cannot resume Hybrid training checkpoints")


def run(args):
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from deepspeed.runtime.fastoffload.scripts.finetune_alpaca import (SupervisedDataCollator, load_alpaca_dataset,
                                                                       load_deepspeed_config, set_seed,
                                                                       tokenize_dataset)

    config = FastOffloadConfig.from_json(args.fastoffload_config)
    if not (config.enabled and config.hybrid_update.zero2_takeover and config.importance.sparse_backward):
        raise ValueError("GPU ceiling requires enabled Hybrid Takeover and selective backward")
    if args.gradient_accumulation_steps != 1 or args.num_train_epochs != 1:
        raise ValueError("GPU ceiling supports one epoch and GAS=1 only")
    if args.save_model or args.benchmark_jsonl is None:
        raise ValueError("Specify --no-save_model and --benchmark_jsonl for this non-training diagnostic")
    if args.benchmark_warmup_steps <= config.importance.warmup_steps:
        raise ValueError("Benchmark warmup must also exclude A state migration after importance warmup")
    if (args.max_steps - args.benchmark_warmup_steps) % config.hybrid_update.update_interval:
        raise ValueError("Measured window must cover whole Hybrid update intervals")
    ds_config = load_deepspeed_config(args)
    if ds_config.get("scheduler") or ds_config["zero_optimization"].get("zenflow"):
        raise ValueError("Use a constant-LR, non-ZenFlow configuration")
    set_seed(args.seed)
    runtimes = []
    original_factory = api.Zero2TakeoverRuntime

    def factory(*factory_args, **factory_kwargs):
        runtime = GpuCeilingTakeoverRuntime(*factory_args, **factory_kwargs)
        runtimes.append(runtime)
        return runtime

    api.Zero2TakeoverRuntime = factory
    handle = None
    engine = None
    try:
        handle = install(config)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                  trust_remote_code=args.trust_remote_code,
                                                  local_files_only=args.local_files_only)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        dataset = tokenize_dataset(load_alpaca_dataset(args), tokenizer, args)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                     trust_remote_code=args.trust_remote_code,
                                                     local_files_only=args.local_files_only)
        model.config.use_cache = False
        model.config.pad_token_id = tokenizer.pad_token_id
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        enable_selective_linear(model,
                                minimum_tokens=config.importance.sparse_backward_min_tokens,
                                minimum_output_features=config.importance.sparse_backward_min_output_features)
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)
        if len(runtimes) != 1 or engine.gradient_accumulation_steps() != 1:
            raise RuntimeError("Expected exactly one GPU ceiling runtime with GAS=1")
        runtime = runtimes[0]
        # Preserve the training-side CPU affinity of the full Takeover runtime.
        runtime._cpu_updater._configure_cpu_affinity()
        rank, world_size = dist.get_rank(), dist.get_world_size()
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
        sampler.set_epoch(0)
        loader = DataLoader(dataset,
                            batch_size=args.micro_batch_size,
                            sampler=sampler,
                            num_workers=args.dataloader_num_workers,
                            pin_memory=True,
                            collate_fn=SupervisedDataCollator(tokenizer.pad_token_id))
        batches, token_counts = [], []
        for batch in islice(loader, args.max_steps):
            token_counts.append((batch["input_ids"].numel(), int(batch["attention_mask"].sum()),
                                 int((batch["labels"][:, 1:] != -100).sum()), batch["input_ids"].shape[0]))
            batches.append({name: tensor.to(engine.device, non_blocking=True) for name, tensor in batch.items()})
        if len(batches) != args.max_steps:
            raise RuntimeError("Dataset cannot supply the requested complete benchmark window")
        accelerator = get_accelerator()
        accelerator.synchronize()
        cached_bytes = sum(t.numel() * t.element_size() for batch in batches for t in batch.values())
        engine.train()
        timings = []
        start = None
        for index, batch in enumerate(batches):
            measured = index >= args.benchmark_warmup_steps
            if index == args.benchmark_warmup_steps:
                runtime.assert_no_cpu_work()
                accelerator.synchronize()
                dist.barrier()
                accelerator.synchronize()
                accelerator.reset_peak_memory_stats()
                initial_counts = runtime.counts.copy()
                start = time.perf_counter()
            events = [accelerator.Event(enable_timing=True) for _ in range(4)] if measured else []
            if measured:
                events[0].record()
            loss = engine(**batch, use_cache=False).loss
            dense = runtime.native_dense_boundary
            if measured:
                events[1].record()
            engine.backward(loss)
            if measured:
                events[2].record()
            engine.step()
            if measured:
                events[3].record()
                timings.append((index + 1, dense, events))
        accelerator.synchronize()
        elapsed = time.perf_counter() - start
        runtime.assert_no_cpu_work()
        counts = runtime.counts - initial_counts
        if counts["overflow_steps"] or counts["a_update_steps"] != len(timings):
            raise RuntimeError("Ceiling measurement must consist entirely of successful A updates")
        local = torch.tensor(
            [elapsed, accelerator.max_memory_allocated(),
             accelerator.max_memory_reserved(), cached_bytes],
            dtype=torch.float64,
            device=engine.device)
        dist.all_reduce(local, op=dist.ReduceOp.MAX)
        totals = [sum(values[i] for values in token_counts[args.benchmark_warmup_steps:]) for i in range(4)]
        totals = torch.tensor(totals, dtype=torch.long, device=engine.device)
        dist.all_reduce(totals)
        elapsed, allocated, reserved, cached_bytes = local.cpu().tolist()
        padded, real, supervised, samples = totals.cpu().tolist()
        phase_records = []
        for step, dense, events in timings:
            phase_records.append({
                "microstep": step,
                "phase": "dense" if dense else "ordinary",
                "forward_stream_ms": events[0].elapsed_time(events[1]),
                "backward_stream_ms": events[1].elapsed_time(events[2]),
                "a_step_stream_ms": events[2].elapsed_time(events[3]),
                "whole_step_stream_ms": events[0].elapsed_time(events[3]),
            })
        args.benchmark_jsonl.parent.mkdir(parents=True, exist_ok=True)
        phase_path = args.benchmark_jsonl.parent / f"phases_rank{rank}.jsonl"
        phase_path.write_text("".join(json.dumps(record) + "\n" for record in phase_records))
        record = {
            "mode":
            "gpu_only_ceiling_not_training",
            "model":
            args.model_name_or_path,
            "world_size":
            world_size,
            "completed_microsteps":
            len(batches),
            "warmup_microsteps":
            args.benchmark_warmup_steps,
            "measured_microsteps":
            len(timings),
            "elapsed_seconds_max_rank":
            elapsed,
            "padded_tokens":
            padded,
            "input_tokens":
            real,
            "supervised_tokens":
            supervised,
            "padded_tokens_per_second":
            padded / elapsed,
            "input_tokens_per_second":
            real / elapsed,
            "microsteps_per_second":
            len(timings) / elapsed,
            "samples_per_second":
            samples / elapsed,
            "gpu_peak_allocated_bytes_max_rank":
            int(allocated),
            "gpu_peak_reserved_bytes_max_rank":
            int(reserved),
            "cached_input_bytes_max_rank":
            int(cached_bytes),
            "measured_step_counts":
            dict(counts),
            "bc_cpu_jobs":
            0,
            "bc_gradient_d2h_bytes":
            0,
            "bc_parameter_h2d_bytes":
            0,
            "b_accumulation_enabled":
            False,
            "inter_gpu_reduce_and_a_publication_enabled":
            True,
            "inputs_gpu_resident":
            True,
            "importance_warmup_cpu_steps_excluded":
            config.importance.warmup_steps,
            "notes":
            "B/C frozen after importance warmup; CPU BC work guarded against execution. "
            "Startup/native importance warmup and A state migration excluded. Scalar norm/overflow control retained. "
            "CUDA-event phases include stream gaps/waits; not additive kernel-only profiling or a mathematical bound.",
        }
        if rank == 0:
            with args.benchmark_jsonl.open("a") as output:
                output.write(json.dumps(record, sort_keys=True) + "\n")
            print("[GPU Ceiling] " + json.dumps(record, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        try:
            if engine is not None:
                engine.destroy()
            if handle is not None:
                handle.close()
        finally:
            api.Zero2TakeoverRuntime = original_factory
            if dist.is_initialized():
                dist.destroy_process_group()


def main():
    from deepspeed.runtime.fastoffload.scripts.finetune_alpaca import parse_args
    run(parse_args())


if __name__ == "__main__":
    main()
