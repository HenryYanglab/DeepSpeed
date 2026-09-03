#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Run FastOffload paper experiments on an offline H100 node.

The runner is deliberately conservative: it does not change the training
semantics, and all extra validation/profile features are opt-in command-line
flags passed to finetune_llama.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

STEP_RE = re.compile(r"Step\s+(\d+),\s+Loss:\s+([\-+0-9.eE]+),\s+"
                     r"(?:(?:LossAvg20):\s+([\-+0-9.eE]+),\s+)?Time:\s+([0-9.]+)ms")
STEP_METRICS_RE = re.compile(r"\[StepMetrics\]\s+step=(\d+)\s+tokens=([0-9.]+)\s+tokens_per_second=([0-9.eE+-]+)")
EVAL_RE = re.compile(r"\[Eval\]\s+step=(\d+).*validation_loss=([0-9.eE+-]+).*examples=(\d+)")
MC_RE = re.compile(r"\[MCEval\]\s+step=(\d+)\s+task=([A-Za-z0-9_./-]+)\s+accuracy=([0-9.eE+-]+).*total=(\d+)")
GEN_RE = re.compile(r"\[GenEval\]\s+step=(\d+)\s+task=([A-Za-z0-9_./-]+)\s+exact_match=([0-9.eE+-]+).*total=(\d+)")
TRANSFER_RE = re.compile(r"\[FastOffloadTransferProfile\]\s+(.*)")
KV_RE = re.compile(r"([A-Za-z0-9_]+)=([0-9.eE+-]+)")


@dataclass(frozen=True)
class Method:
    name: str
    config: str


@dataclass(frozen=True)
class ModelCase:
    name: str
    path: str
    max_steps: int
    min_gpus: int = 2


@dataclass(frozen=True)
class Experiment:
    run_id: str
    suite: str
    method: str
    config: str
    model: str
    model_path: str
    dataset: str
    dataset_path: str
    seed: int
    max_steps: int
    gpus: str
    eval_interval: int = 0
    mc_eval_interval: int = 0
    mc_eval_tasks: str = ""
    transfer_profile: bool = False
    train_profile: bool = False
    torch_profile: bool = False
    batch_size: int = 8
    max_length: int = 512
    validation_dataset_path: str = ""
    gen_eval_interval: int = 0
    gen_eval_tasks: str = ""
    hardware_label: str = "h100"
    notes: str = ""
    config_overrides: dict[str, Any] = field(default_factory=dict)


METHODS = {
    "zero_offload":
    Method("zero_offload", "zero_offload_config.json"),
    "zero3_offload":
    Method("zero3_offload", "zero3_offload_config.json"),
    "zenflow":
    Method("zenflow", "zf_config.json"),
    "fastoffload":
    Method("fastoffload", "fastoffload_config.json"),
    "fo_fp32grad_fp32return":
    Method("fo_fp32grad_fp32return", "fastoffload_config_ablate_fp32grad_fp32stale.json"),
    "fo_fp32grad_bf16return":
    Method("fo_fp32grad_bf16return", "fastoffload_config_ablate_fp32grad_bit16stale.json"),
    "fo_bf16grad_fp32return":
    Method("fo_bf16grad_fp32return", "fastoffload_config_ablate_bit16grad_fp32stale.json"),
    "fo_cpuaccum":
    Method("fo_cpuaccum", "fastoffload_config_cpuaccum.json"),
    "fo_cpuaccum_asyncgrad":
    Method("fo_cpuaccum_asyncgrad", "fastoffload_config_cpuaccum_async_gradcopy.json"),
    "fo_partial":
    Method("fo_partial", "fastoffload_config_partial_selected_accum.json"),
    "fo_partial_fixed05":
    Method("fo_partial_fixed05", "fastoffload_config_partial_fixed05.json"),
    "fo_partial_fixed05_dense_boundary":
    Method("fo_partial_fixed05_dense_boundary", "fastoffload_config_partial_fixed05_dense_boundary.json"),
    "fo_partial_fixed05_dense_boundary_packed":
    Method(
        "fo_partial_fixed05_dense_boundary_packed",
        "fastoffload_config_partial_fixed05_dense_boundary_packed.json",
    ),
    "fo_partial_fixed07":
    Method("fo_partial_fixed07", "fastoffload_config_partial_fixed07.json"),
    "fo_partial_fixed05_int8grad":
    Method(
        "fo_partial_fixed05_int8gfo_partial_fixed10_dense_boundary_packedfo_partial_fixed10_dense_boundary_packedrad",
        "fastoffload_config_partial_fixed05_int8grad.json"),
    "fo_dynamic_topk":
    Method("fo_dynamic_topk", "fastoffload_config_dynamic_topk10.json"),
}

MODELS = {
    "qwen25_1p5b": ModelCase("qwen25_1p5b", "/home/shared/Qwen2.5-1.5B-Instruct", 120, 1),
    "qwen25_3b": ModelCase("qwen25_3b", "/home/shared/Qwen2.5-3B-Instruct", 120, 1),
    "qwen25_7b": ModelCase("qwen25_7b", "/home/shared/Qwen2.5-7B-Instruct", 160, 2),
    "qwen25_14b": ModelCase("qwen25_14b", "/home/shared/Qwen2.5-14B-Instruct", 120, 4),
    "qwen25_32b": ModelCase("qwen25_32b", "/home/shared/Qwen2.5-32B-Instruct", 80, 4),
    "qwen3_8b": ModelCase("qwen3_8b", "/home/shared/Qwen3-8B", 160, 2),
    "qwen3_14b": ModelCase("qwen3_14b", "/home/shared/Qwen3-14B", 120, 4),
    "qwen3_32b": ModelCase("qwen3_32b", "/home/shared/Qwen3-32B", 80, 8),
    "llama2_7b": ModelCase("llama2_7b", "/home/shared/llama-2/Llama-2-7b-hf", 300, 2),
    "llama2_13b": ModelCase("llama2_13b", "/home/shared/llama-2/Llama-2-13b-hf", 120, 2),
    "llama3_8b": ModelCase("llama3_8b", "/home/shared/Meta-Llama-3-8B-Instruct", 160, 2),
    "llama31_8b": ModelCase("llama31_8b", "/home/shared/Meta-Llama-3.1-8B-Instruct", 160, 2),
    "mistral_nemo_12b": ModelCase("mistral_nemo_12b", "/home/shared/Mistral-Nemo-12B-Instruct-2407", 120, 4),
    "gemma3_12b": ModelCase("gemma3_12b", "/home/shared/gemma-3-12b-it", 120, 4),
    "llama33_70b": ModelCase("llama33_70b", "/home/shared/Llama-3.3-70B-Instruct", 40, 8),
}

DATASETS = {
    "alpaca": "/home/shared/fastoffload_datasets/alpaca/alpaca_data_cleaned.json",
    "dolly": "/home/shared/fastoffload_datasets/dolly/dolly_alpaca.json",
    "gsm8k_sft": "/home/shared/fastoffload_datasets/gsm8k/gsm8k_alpaca.json",
    "openbookqa_sft": "/home/shared/fastoffload_datasets/openbookqa/openbookqa_alpaca.json",
    "xsum_sft": "/home/shared/fastoffload_datasets/xsum/xsum_alpaca.json",
}


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def zenflow_override(*,
                     topk: float | None = None,
                     update_interval: int | None = None,
                     pt_reserved_cores_perc: float | None = None) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if topk is not None:
        values["topk_ratio"] = topk
    if update_interval is not None:
        values["update_interval"] = update_interval
    if pt_reserved_cores_perc is not None:
        values["pt_reserved_cores_perc"] = pt_reserved_cores_perc
    return {"zero_optimization": {"zenflow": values}} if values else {}


def fastoffload_override(*,
                         topk: float | None = None,
                         update_interval: int | None = None,
                         cpuadam_cores_perc: float | None = None,
                         grad_dtype: str | None = None,
                         stale_dtype: str | None = None,
                         layerwise_topk: bool | None = None,
                         layerwise_policy: str | None = None,
                         layerwise_strength: float | None = None) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if topk is not None:
        values["channel_topk_ratio"] = topk
    if update_interval is not None:
        values["noncritical_update_interval"] = update_interval
    if cpuadam_cores_perc is not None:
        values["cpuadam_cores_perc"] = cpuadam_cores_perc
    if grad_dtype is not None:
        values["transfer_lean_grad_transfer_dtype"] = grad_dtype
    if stale_dtype is not None:
        values["transfer_lean_stale_param_dtype"] = stale_dtype
    if layerwise_topk is not None:
        values["transfer_lean_layerwise_topk"] = layerwise_topk
    if layerwise_policy is not None:
        values["transfer_lean_layerwise_topk_policy"] = layerwise_policy
    if layerwise_strength is not None:
        values["transfer_lean_layerwise_topk_strength"] = layerwise_strength
    return {"zero_optimization": {"fast_offload": values}} if values else {}


def update_selected_schedule_override(method: str,
                                      *,
                                      topk: float | None = None,
                                      update_interval: int | None = None) -> dict[str, Any]:
    if method == "zenflow":
        return zenflow_override(topk=topk, update_interval=update_interval)
    if method.startswith("fastoffload") or method.startswith("fo_"):
        return fastoffload_override(topk=topk, update_interval=update_interval)
    return {}


def select_gpus(pool: str, count: int) -> str:
    gpus = parse_csv_list(pool)
    if count > len(gpus):
        raise ValueError(f"Need {count} GPUs but pool only has {pool}")
    return ",".join(gpus[:count])


def prepare_config(exp: Experiment, args: argparse.Namespace, cwd: Path) -> Path:
    base_path = cwd / exp.config
    if not base_path.exists():
        raise FileNotFoundError(str(base_path))
    config = json.loads(base_path.read_text())
    if exp.config_overrides:
        config = deep_update(config, deepcopy(exp.config_overrides))
    config["train_batch_size"] = exp.batch_size
    if args.gradient_accumulation_steps is not None:
        config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    config_dir = args.out_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    generated = config_dir / f"{exp.run_id}.json"
    generated.write_text(json.dumps(config, indent=2, sort_keys=True))
    return generated


def collect_metadata(exp: Experiment, cmd: list[str], config_path: Path, cwd: Path) -> dict[str, Any]:

    def run_text(command: list[str]) -> str:
        try:
            return subprocess.check_output(command, cwd=str(cwd), text=True, stderr=subprocess.STDOUT,
                                           timeout=20).strip()
        except Exception as exc:
            return f"unavailable: {exc}"

    metadata = {
        "run_spec": asdict(exp),
        "cmd": cmd,
        "config_path": str(config_path),
        "git_commit": run_text(["git", "rev-parse", "HEAD"]),
        "git_branch": run_text(["git", "branch", "--show-current"]),
        "git_status_short": run_text(["git", "status", "--short"]),
        "python": sys.version,
        "hostname": run_text(["hostname"]),
        "nvidia_smi": run_text(["nvidia-smi", "-L"]),
    }
    try:
        import torch
        metadata["torch_version"] = torch.__version__
        metadata["cuda_available"] = torch.cuda.is_available()  #ignore-cuda
        metadata["cuda_version"] = torch.version.cuda
    except Exception as exc:
        metadata["torch_version"] = f"unavailable: {exc}"
    try:
        import deepspeed
        metadata["deepspeed_version"] = getattr(deepspeed, "__version__", "unknown")
    except Exception as exc:
        metadata["deepspeed_version"] = f"unavailable: {exc}"
    return metadata


def build_experiments(args: argparse.Namespace) -> list[Experiment]:
    suites = parse_csv_list(args.suite)
    experiments: list[Experiment] = []
    seeds = [int(x) for x in parse_csv_list(args.seeds)]

    def add(suite: str,
            method_names: list[str],
            model_names: list[str],
            *,
            dataset_names: list[str] | None = None,
            seeds_for_suite: list[int] | None = None,
            max_steps_override: int | None = None,
            gpus_override: str | None = None,
            eval_interval: int = 0,
            mc_eval_interval: int = 0,
            mc_eval_tasks: str = "",
            gen_eval_interval: int = 0,
            gen_eval_tasks: str = "",
            transfer_profile: bool = False,
            train_profile: bool = False,
            torch_profile: bool = False,
            batch_size: int | None = None,
            max_length: int | None = None,
            hardware_label: str | None = None,
            notes: str = "",
            overrides_by_method: dict[str, dict[str, Any]] | None = None) -> None:
        for seed in seeds if seeds_for_suite is None else seeds_for_suite:
            for dataset_name in dataset_names or [args.dataset]:
                for model_name in model_names:
                    model = MODELS[model_name]
                    gpus = gpus_override or select_gpus(args.gpu_pool, model.min_gpus)
                    for method_name in method_names:
                        method = METHODS[method_name]
                        max_steps = model.max_steps if max_steps_override is None else max_steps_override
                        method_overrides = deepcopy((overrides_by_method or {}).get(method_name, {}))
                        run_id = (f"{suite}_{method.name}_{model.name}_{dataset_name}_seed{seed}_"
                                  f"g{len(parse_csv_list(gpus))}_steps{max_steps}")
                        experiments.append(
                            Experiment(run_id=run_id,
                                       suite=suite,
                                       method=method.name,
                                       config=method.config,
                                       model=model.name,
                                       model_path=model.path,
                                       dataset=dataset_name,
                                       dataset_path=DATASETS[dataset_name],
                                       seed=seed,
                                       max_steps=max_steps,
                                       gpus=gpus,
                                       eval_interval=eval_interval,
                                       mc_eval_interval=mc_eval_interval,
                                       mc_eval_tasks=mc_eval_tasks,
                                       gen_eval_interval=gen_eval_interval,
                                       gen_eval_tasks=gen_eval_tasks,
                                       transfer_profile=transfer_profile,
                                       train_profile=train_profile,
                                       torch_profile=torch_profile,
                                       batch_size=batch_size or args.batch_size,
                                       max_length=max_length or args.max_length,
                                       validation_dataset_path=args.validation_dataset_path,
                                       hardware_label=hardware_label or args.hardware_label,
                                       notes=notes,
                                       config_overrides=method_overrides))

    if "quality" in suites or "all" in suites:
        add("quality",
            parse_csv_list(args.quality_methods),
            parse_csv_list(args.quality_models),
            dataset_names=parse_csv_list(args.quality_datasets),
            max_steps_override=0 if args.quality_full_epoch else
            (args.quality_steps if args.quality_steps > 0 else None),
            eval_interval=args.eval_interval,
            mc_eval_interval=args.mc_eval_interval,
            mc_eval_tasks=args.mc_eval_tasks,
            gen_eval_interval=args.gen_eval_interval,
            gen_eval_tasks=args.gen_eval_tasks)

    if "scaling" in suites or "all" in suites:
        for world_size in [int(x) for x in parse_csv_list(args.scaling_world_sizes)]:
            add(f"scaling{world_size}g", ["zenflow", "fastoffload"],
                parse_csv_list(args.scaling_models),
                seeds_for_suite=[seeds[0]],
                max_steps_override=args.scaling_steps,
                gpus_override=select_gpus(args.gpu_pool, world_size))

    if "models" in suites or "all" in suites:
        add("models", ["zero_offload", "zenflow", "fastoffload"],
            parse_csv_list(args.model_sweep),
            seeds_for_suite=[seeds[0]])

    if "timeline" in suites or "all" in suites:
        add("timeline", ["zero_offload", "zenflow", "fastoffload"],
            parse_csv_list(args.timeline_models),
            seeds_for_suite=[seeds[0]],
            max_steps_override=args.timeline_steps,
            transfer_profile=True,
            train_profile=True)

    if "hardware" in suites or "all" in suites:
        add(f"hardware_{args.hardware_label}",
            parse_csv_list(args.hardware_methods),
            parse_csv_list(args.hardware_models),
            seeds_for_suite=[seeds[0]],
            max_steps_override=args.hardware_steps,
            eval_interval=args.eval_interval,
            gpus_override=args.hardware_gpus or None,
            transfer_profile=args.hardware_profile,
            hardware_label=args.hardware_label)

    if "baseline" in suites or "all" in suites:
        add("baseline",
            parse_csv_list(args.baseline_methods),
            parse_csv_list(args.baseline_models),
            seeds_for_suite=[seeds[0]],
            max_steps_override=args.baseline_steps)

    if "ablation" in suites or "all" in suites:
        add("ablation", [
            "zenflow",
            "fo_fp32grad_fp32return",
            "fo_fp32grad_bf16return",
            "fo_bf16grad_fp32return",
            "fo_cpuaccum",
            "fo_partial_fixed05",
            "fastoffload",
            "fo_dynamic_topk",
        ], [args.ablation_model],
            seeds_for_suite=[seeds[0]],
            max_steps_override=args.ablation_steps,
            transfer_profile=True)

    if "profile" in suites or "all" in suites:
        add("profile", ["zenflow", "fastoffload"], [args.profile_model],
            seeds_for_suite=[seeds[0]],
            max_steps_override=args.profile_steps,
            transfer_profile=True,
            train_profile=True,
            torch_profile=True)

    if "sensitivity" in suites or "all" in suites:
        sensitivity_model = [args.sensitivity_model]
        sensitivity_seed = [seeds[0]]
        for interval in [int(x) for x in parse_csv_list(args.sensitivity_k_values)]:
            overrides = {
                "zenflow": zenflow_override(update_interval=interval),
                "fastoffload": fastoffload_override(update_interval=interval),
            }
            add(f"sens_k{interval}", ["zenflow", "fastoffload"],
                sensitivity_model,
                seeds_for_suite=sensitivity_seed,
                max_steps_override=args.sensitivity_steps,
                eval_interval=args.sensitivity_eval_interval,
                overrides_by_method=overrides,
                notes=f"update_interval={interval}")
        for topk in [float(x) for x in parse_csv_list(args.sensitivity_topk_values)]:
            overrides = {
                "zenflow": zenflow_override(topk=topk),
                "fastoffload": fastoffload_override(topk=topk),
            }
            add(f"sens_topk{topk:g}", ["zenflow", "fastoffload"],
                sensitivity_model,
                seeds_for_suite=sensitivity_seed,
                max_steps_override=args.sensitivity_steps,
                eval_interval=args.sensitivity_eval_interval,
                overrides_by_method=overrides,
                notes=f"topk={topk:g}")
        for batch_size in [int(x) for x in parse_csv_list(args.sensitivity_batch_sizes)]:
            add(f"sens_bs{batch_size}", ["zenflow", "fastoffload"],
                sensitivity_model,
                seeds_for_suite=sensitivity_seed,
                max_steps_override=args.sensitivity_steps,
                eval_interval=args.sensitivity_eval_interval,
                batch_size=batch_size,
                notes=f"train_batch_size={batch_size}")
        for cores_perc in [float(x) for x in parse_csv_list(args.sensitivity_cpu_cores)]:
            overrides = {
                "zenflow": zenflow_override(pt_reserved_cores_perc=max(0.0, 1.0 - cores_perc)),
                # The transfer-lean path forwards this field into ZenFlow's
                # pt_reserved_cores_perc, so convert worker fraction to reserved fraction.
                "fastoffload": fastoffload_override(cpuadam_cores_perc=max(0.01, 1.0 - cores_perc)),
            }
            add(f"sens_cpu{cores_perc:g}", ["zenflow", "fastoffload"],
                sensitivity_model,
                seeds_for_suite=sensitivity_seed,
                max_steps_override=args.sensitivity_steps,
                transfer_profile=True,
                overrides_by_method=overrides,
                notes=f"cpuadam_cores_perc={cores_perc:g}")
        grad_overrides = {
            "fo_fp32grad_bf16return": {},
            "fastoffload": {},
        }
        add("sens_grad_dtype", ["fo_fp32grad_bf16return", "fastoffload"],
            sensitivity_model,
            seeds_for_suite=sensitivity_seed,
            max_steps_override=args.sensitivity_steps,
            eval_interval=args.sensitivity_eval_interval,
            mc_eval_interval=args.mc_eval_interval,
            mc_eval_tasks=args.mc_eval_tasks,
            overrides_by_method=grad_overrides,
            notes="fp32-grad-vs-bf16-grad")
        for policy, strength in [("uniform", 0.0), ("outlier_grad_ema", 1.5)]:
            overrides = {
                "fastoffload":
                fastoffload_override(topk=0.10,
                                     layerwise_topk=(policy != "uniform"),
                                     layerwise_policy=policy,
                                     layerwise_strength=strength),
            }
            add(f"sens_dyn_topk_{policy}", ["fastoffload"],
                sensitivity_model,
                seeds_for_suite=sensitivity_seed,
                max_steps_override=args.sensitivity_steps,
                eval_interval=args.sensitivity_eval_interval,
                overrides_by_method=overrides,
                notes=f"dynamic_topk_policy={policy}")
    return experiments


def ensure_inputs(exp: Experiment, config_path: Path) -> None:
    missing = []
    for path in [config_path, Path(exp.model_path), Path(exp.dataset_path)]:
        if not Path(path).exists():
            missing.append(str(path))
    if exp.validation_dataset_path and not Path(exp.validation_dataset_path).exists():
        missing.append(exp.validation_dataset_path)
    if missing:
        raise FileNotFoundError(f"{exp.run_id} missing inputs: {missing}")


def sample_gpu(stop_event: threading.Event, out_csv: Path, gpu_indices: str) -> None:
    fields = ["timestamp", "index", "memory.used", "utilization.gpu", "utilization.memory", "power.draw"]
    query = ",".join(fields[1:])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        while not stop_event.is_set():
            try:
                output = subprocess.check_output([
                    "nvidia-smi",
                    f"--id={gpu_indices}",
                    f"--query-gpu={query}",
                    "--format=csv,noheader,nounits",
                ],
                                                 text=True,
                                                 stderr=subprocess.DEVNULL)
                ts = time.time()
                for line in output.splitlines():
                    cols = [col.strip() for col in line.split(",")]
                    if len(cols) == len(fields) - 1:
                        writer.writerow([f"{ts:.3f}", *cols])
                handle.flush()
            except Exception:
                pass
            stop_event.wait(1.0)


def run_one(exp: Experiment, args: argparse.Namespace, cwd: Path, index: int, total: int) -> dict:
    log_path = args.out_dir / "logs" / f"{exp.run_id}.log"
    gpu_path = args.out_dir / "gpu" / f"{exp.run_id}.csv"
    status_path = args.out_dir / "status" / f"{exp.run_id}.json"
    metadata_path = args.out_dir / "metadata" / f"{exp.run_id}.json"
    profile_dir = args.out_dir / "torch_profile" / exp.run_id
    output_dir = args.out_dir / "outputs" / exp.run_id
    for path in [log_path.parent, gpu_path.parent, status_path.parent, metadata_path.parent, output_dir, profile_dir]:
        path.mkdir(parents=True, exist_ok=True)

    try:
        config_path = prepare_config(exp, args, cwd)
        ensure_inputs(exp, config_path)
    except Exception as exc:
        status = {
            "run_id": exp.run_id,
            "returncode": 125,
            "skipped": True,
            "skip_reason": str(exc),
            "started": time.time(),
            "ended": time.time(),
            "elapsed_s": 0.0,
            "log_path": str(log_path),
            "gpu_path": str(gpu_path),
        }
        log_path.write_text("[SKIP_INPUTS] " + json.dumps(asdict(exp), sort_keys=True) + "\n" + str(exc) + "\n")
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True))
        print(f"[SKIP_INPUTS] {index}/{total} {exp.run_id}: {exc}", flush=True)
        return status

    if not args.force and completed_ok(log_path, status_path, exp.max_steps):
        print(f"[SKIP_COMPLETED] {index}/{total} {exp.run_id}", flush=True)
        return {"run_id": exp.run_id, "status": "skipped_completed"}

    env = os.environ.copy()
    env.update({
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "FAST_OFFLOAD_TRANSFER_PROFILE": "1" if exp.transfer_profile else "0",
        "FAST_OFFLOAD_PROFILE": "1" if exp.train_profile else "0",
        "FAST_OFFLOAD_TORCH_PROFILE": "1" if exp.torch_profile else "0",
        "FAST_OFFLOAD_TORCH_PROFILE_DIR": str(profile_dir),
        "FAST_OFFLOAD_TORCH_PROFILE_WAIT": str(args.torch_profile_wait),
        "FAST_OFFLOAD_TORCH_PROFILE_WARMUP": str(args.torch_profile_warmup),
        "FAST_OFFLOAD_TORCH_PROFILE_ACTIVE": str(args.torch_profile_active),
    })
    if Path("/usr/local/lib64/libstdc++.so.6").exists():
        env["LD_PRELOAD"] = "/usr/local/lib64/libstdc++.so.6"

    master_port = args.master_port + (index % 100)
    cmd = ["deepspeed"]
    if args.deepspeed_bind_cores_to_rank or args.deepspeed_bind_core_list:
        cmd.append("--bind_cores_to_rank")
    if args.deepspeed_bind_core_list:
        cmd.append(f"--bind_core_list={args.deepspeed_bind_core_list}")
    cmd.extend([
        f"--include=localhost:{exp.gpus}",
        f"--master_port={master_port}",
        "finetune_llama.py",
        f"--deepspeed_config={config_path}",
        f"--model_name={exp.model_path}",
        f"--dataset_path={exp.dataset_path}",
        "--num_train_epochs=1",
        f"--max_steps={exp.max_steps}",
        f"--lr={args.train_lr}",
        f"--batch_size={exp.batch_size}",
        f"--max_length={exp.max_length}",
        f"--warmup={args.warmup}",
        f"--lr_scheduler_type={args.lr_scheduler_type}",
        "--weight_decay=0.01",
        "--train_on_response_only",
        f"--output_dir={output_dir}",
        f"--seed={exp.seed}",
        "--skip_save",
    ])
    if args.numa_node:
        cmd = ["numactl", "-m", str(args.numa_node), *cmd]
    if args.save_hf_model:
        cmd.append("--save_hf_model")
    if exp.eval_interval > 0:
        cmd.extend([
            f"--eval_interval={exp.eval_interval}",
            f"--eval_max_batches={args.eval_max_batches}",
            "--eval_at_end",
        ])
        if exp.validation_dataset_path:
            cmd.append(f"--validation_dataset_path={exp.validation_dataset_path}")
        else:
            cmd.append(f"--validation_split={args.validation_split}")
        if args.eval_at_start:
            cmd.append("--eval_at_start")
    if exp.mc_eval_interval > 0 and exp.mc_eval_tasks:
        cmd.extend([
            f"--mc_eval_tasks={exp.mc_eval_tasks}",
            f"--mc_eval_root={args.mc_eval_root}",
            f"--mc_eval_interval={exp.mc_eval_interval}",
            f"--mc_eval_max_examples={args.mc_eval_max_examples}",
            "--mc_eval_at_end",
        ])
        if args.mc_eval_at_start:
            cmd.append("--mc_eval_at_start")
    if exp.gen_eval_interval > 0 and exp.gen_eval_tasks:
        cmd.extend([
            f"--gen_eval_tasks={exp.gen_eval_tasks}",
            f"--gen_eval_root={args.gen_eval_root}",
            f"--gen_eval_interval={exp.gen_eval_interval}",
            f"--gen_eval_max_examples={args.gen_eval_max_examples}",
            f"--gen_eval_max_new_tokens={args.gen_eval_max_new_tokens}",
            "--gen_eval_at_end",
        ])
        if args.gen_eval_at_start:
            cmd.append("--gen_eval_at_start")

    metadata_path.write_text(json.dumps(collect_metadata(exp, cmd, config_path, cwd), indent=2, sort_keys=True))

    stop_event = threading.Event()
    sampler = threading.Thread(target=sample_gpu, args=(stop_event, gpu_path, exp.gpus), daemon=True)
    sampler.start()
    started = time.time()
    print(f"[RUN_START] {index}/{total} {exp.run_id}", flush=True)
    print("[CMD] " + " ".join(cmd), flush=True)
    with log_path.open("w", buffering=1) as log_f:
        log_f.write("[RUN_SPEC] " + json.dumps(asdict(exp), sort_keys=True) + "\n")
        log_f.write("[CMD] " + " ".join(cmd) + "\n")
        proc = subprocess.Popen(cmd,
                                cwd=str(cwd),
                                env=env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_f.write(line)
        returncode = proc.wait()
    stop_event.set()
    sampler.join(timeout=5)
    status = {
        "run_id": exp.run_id,
        "returncode": returncode,
        "started": started,
        "ended": time.time(),
        "elapsed_s": time.time() - started,
        "log_path": str(log_path),
        "gpu_path": str(gpu_path),
    }
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True))
    print(f"[RUN_DONE] {exp.run_id} returncode={returncode} elapsed_s={status['elapsed_s']:.1f}", flush=True)
    return status


def parse_log(path: Path) -> dict:
    steps = []
    tokens = []
    eval_rows = []
    mc_rows = []
    gen_rows = []
    transfer_rows = []
    if not path.exists():
        return {
            "steps": steps,
            "tokens": tokens,
            "eval": eval_rows,
            "mc": mc_rows,
            "gen": gen_rows,
            "transfer": transfer_rows,
        }
    for line in path.read_text(errors="replace").splitlines():
        if match := STEP_RE.search(line):
            row = {
                "step": int(match.group(1)),
                "loss": float(match.group(2)),
                "time_ms": float(match.group(4)),
            }
            if match.group(3) is not None:
                row["loss_avg20"] = float(match.group(3))
            steps.append(row)
        if match := STEP_METRICS_RE.search(line):
            tokens.append({
                "step": int(match.group(1)),
                "tokens": float(match.group(2)),
                "tokens_per_second": float(match.group(3)),
            })
        if match := EVAL_RE.search(line):
            eval_rows.append({
                "step": int(match.group(1)),
                "validation_loss": float(match.group(2)),
                "examples": int(match.group(3))
            })
        if match := MC_RE.search(line):
            mc_rows.append({
                "step": int(match.group(1)),
                "task": match.group(2),
                "accuracy": float(match.group(3)),
                "total": int(match.group(4)),
            })
        if match := GEN_RE.search(line):
            gen_rows.append({
                "step": int(match.group(1)),
                "task": match.group(2),
                "exact_match": float(match.group(3)),
                "total": int(match.group(4)),
            })
        if match := TRANSFER_RE.search(line):
            transfer_rows.append({key: float(value) for key, value in KV_RE.findall(match.group(1))})
    return {
        "steps": steps,
        "tokens": tokens,
        "eval": eval_rows,
        "mc": mc_rows,
        "gen": gen_rows,
        "transfer": transfer_rows
    }


def completed_ok(log_path: Path, status_path: Path, max_steps: int) -> bool:
    if not log_path.exists() or not status_path.exists():
        return False
    try:
        status = json.loads(status_path.read_text())
    except Exception:
        return False
    if status.get("returncode") != 0:
        return False
    steps = parse_log(log_path)["steps"]
    return bool(steps) and max(row["step"] for row in steps) >= max_steps


def experiment_update_interval(exp: Experiment) -> int:
    if exp.method in ("zero_offload", "zero3_offload"):
        return 1
    if "update_interval=" in exp.notes:
        try:
            return int(exp.notes.split("update_interval=", 1)[1].split()[0].split(",", 1)[0])
        except Exception:
            pass
    if exp.config_overrides:
        try:
            zf = exp.config_overrides.get("zero_optimization", {}).get("zenflow", {})
            if "update_interval" in zf:
                return int(zf["update_interval"])
        except Exception:
            pass
        try:
            fast_offload = exp.config_overrides.get("zero_optimization", {}).get("fast_offload", {})
            if "noncritical_update_interval" in fast_offload:
                return int(fast_offload["noncritical_update_interval"])
        except Exception:
            pass
    return 4


def parse_gpu(path: Path) -> tuple[float | None, float | None]:
    if not path.exists():
        return None, None
    mem = []
    util = []
    with path.open(newline="", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                mem.append(float(row["memory.used"]))
                util.append(float(row["utilization.gpu"]))
            except Exception:
                pass
    return (max(mem) if mem else None, statistics.mean(util) if util else None)


def summarize(experiments: list[Experiment], args: argparse.Namespace) -> None:
    rows = []
    step_rows = []
    eval_rows = []
    mc_rows = []
    gen_rows = []
    transfer_rows = []
    for exp in experiments:
        log_path = args.out_dir / "logs" / f"{exp.run_id}.log"
        gpu_path = args.out_dir / "gpu" / f"{exp.run_id}.csv"
        status_path = args.out_dir / "status" / f"{exp.run_id}.json"
        parsed = parse_log(log_path)
        steps = parsed["steps"]
        token_by_step = {row["step"]: row for row in parsed["tokens"]}
        update_interval = experiment_update_interval(exp)
        stable = [row for row in steps if row["step"] >= min(args.stable_start, exp.max_steps)]
        target = stable or steps
        boundary_target = [row for row in target if update_interval > 0 and row["step"] % update_interval == 0]
        nonboundary_target = [
            row for row in target if not (update_interval > 0 and row["step"] % update_interval == 0)
        ]
        gpu_peak, gpu_util = parse_gpu(gpu_path)
        status = {}
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text())
            except Exception:
                status = {}
        token_target = [row for row in parsed["tokens"] if row["step"] >= min(args.stable_start, exp.max_steps)]
        transfer_target = parsed["transfer"]
        row = {
            **asdict(exp),
            "returncode":
            status.get("returncode"),
            "elapsed_s":
            status.get("elapsed_s"),
            "logged_steps":
            len(steps),
            "last_step":
            max((row["step"] for row in steps), default=0),
            "stable_time_s":
            sum(row["time_ms"] for row in target) / 1000.0 if target else None,
            "stable_avg_step_ms":
            statistics.mean(row["time_ms"] for row in target) if target else None,
            "stable_avg_loss":
            statistics.mean(row["loss"] for row in target) if target else None,
            "boundary_time_s":
            sum(row["time_ms"] for row in boundary_target) / 1000.0 if boundary_target else None,
            "nonboundary_time_s":
            sum(row["time_ms"] for row in nonboundary_target) / 1000.0 if nonboundary_target else None,
            "boundary_avg_step_ms":
            statistics.mean(row["time_ms"] for row in boundary_target) if boundary_target else None,
            "nonboundary_avg_step_ms":
            statistics.mean(row["time_ms"] for row in nonboundary_target) if nonboundary_target else None,
            "update_interval":
            update_interval,
            "last_loss":
            steps[-1]["loss"] if steps else None,
            "stable_tokens_per_second":
            statistics.mean(row["tokens_per_second"] for row in token_target) if token_target else None,
            "peak_mem_gib":
            gpu_peak / 1024.0 if gpu_peak is not None else None,
            "avg_gpu_util_pct":
            gpu_util,
            "last_validation_loss":
            parsed["eval"][-1]["validation_loss"] if parsed["eval"] else None,
            "transfer_boundary_total_ms":
            statistics.mean(row["boundary_total_ms"] for row in transfer_target
                            if "boundary_total_ms" in row) if transfer_target else None,
            "transfer_grad_d2h_ms":
            statistics.mean(row["grad_d2h_ms"] for row in transfer_target
                            if "grad_d2h_ms" in row) if transfer_target else None,
            "transfer_param_h2d_ms":
            statistics.mean(row["param_h2d_ms"] for row in transfer_target
                            if "param_h2d_ms" in row) if transfer_target else None,
        }
        rows.append(row)
        cumulative_time_ms = 0.0
        for step in steps:
            cumulative_time_ms += step["time_ms"]
            token_row = token_by_step.get(step["step"], {})
            step_rows.append({
                **asdict(exp),
                **step,
                "cumulative_time_s": cumulative_time_ms / 1000.0,
                "is_boundary_step": int(update_interval > 0 and step["step"] % update_interval == 0),
                "update_interval": update_interval,
                "tokens": token_row.get("tokens"),
                "tokens_per_second": token_row.get("tokens_per_second"),
            })
        for mc in parsed["mc"]:
            mc_rows.append({**asdict(exp), **mc})
        for gen in parsed["gen"]:
            gen_rows.append({**asdict(exp), **gen})
        for eval_row in parsed["eval"]:
            eval_rows.append({**asdict(exp), **eval_row})
        for profile in parsed["transfer"]:
            transfer_rows.append({**asdict(exp), **profile})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "summary.csv", rows)
    write_csv(args.out_dir / "steps.csv", step_rows)
    write_csv(args.out_dir / "eval.csv", eval_rows)
    write_csv(args.out_dir / "mc_eval.csv", mc_rows)
    write_csv(args.out_dir / "gen_eval.csv", gen_rows)
    write_csv(args.out_dir / "transfer_profile.csv", transfer_rows)
    (args.out_dir / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
    write_report(args.out_dir / "REPORT.md", rows, mc_rows, gen_rows)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=3):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(path: Path, rows: list[dict], mc_rows: list[dict], gen_rows: list[dict]) -> None:
    lines = [
        "# FastOffload Paper Experiment Suite",
        "",
        "Stable timing is the end-to-end sum over logged steps after the configured stable-start.",
        "",
        "| suite | method | model | seed | gpus | steps | time s | bnd s | non-bnd s | tok/s | train loss | val loss | peak GiB | util % | hw | rc |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['suite']} | {row['method']} | {row['model']} | {row['seed']} | {row['gpus']} | "
                     f"{row['last_step']}/{row['max_steps']} | {fmt(row['stable_time_s'])} | "
                     f"{fmt(row['boundary_time_s'])} | {fmt(row['nonboundary_time_s'])} | "
                     f"{fmt(row['stable_tokens_per_second'], 1)} | {fmt(row['stable_avg_loss'], 4)} | "
                     f"{fmt(row['last_validation_loss'], 4)} | {fmt(row['peak_mem_gib'], 1)} | "
                     f"{fmt(row['avg_gpu_util_pct'], 1)} | {row.get('hardware_label')} | {row.get('returncode')} |")
    if mc_rows:
        lines.extend([
            "", "## Multiple-choice evaluation", "", "| suite | method | model | seed | step | task | acc | total |",
            "|---|---|---:|---:|---:|---|---:|---:|"
        ])
        for row in mc_rows:
            lines.append(f"| {row['suite']} | {row['method']} | {row['model']} | {row['seed']} | {row['step']} | "
                         f"{row['task']} | {row['accuracy']:.4f} | {row['total']} |")
    if gen_rows:
        lines.extend([
            "", "## Generation evaluation", "",
            "| suite | method | model | seed | step | task | exact match | total |",
            "|---|---|---:|---:|---:|---|---:|---:|"
        ])
        for row in gen_rows:
            lines.append(f"| {row['suite']} | {row['method']} | {row['model']} | {row['seed']} | {row['step']} | "
                         f"{row['task']} | {row['exact_match']:.4f} | {row['total']} |")
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="quality")
    parser.add_argument("--seeds", default="42,1234,2026")
    parser.add_argument("--dataset", default="alpaca", choices=sorted(DATASETS))
    parser.add_argument("--gpu-pool", default="4,5,6,7")
    parser.add_argument("--master-port", type=int, default=29710)
    parser.add_argument("--out-dir", type=Path, default=Path("logs/paper_experiments_20260518"))
    parser.add_argument("--deepspeed-bind-cores-to-rank", action="store_true")
    parser.add_argument("--deepspeed-bind-core-list", default="")
    parser.add_argument("--numa-node", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-runs", action="store_true")
    parser.add_argument("--stable-start", type=int, default=21)
    parser.add_argument("--hardware-label", default="h100")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--train-lr", type=float, default=5e-6)
    parser.add_argument("--warmup", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="cosine", choices=["none", "cosine", "linear"])
    parser.add_argument("--validation-dataset-path", default="")
    parser.add_argument("--validation-split", type=float, default=0.02)
    parser.add_argument("--quality-methods", default="zero_offload,zenflow,fastoffload,fo_fp32grad_bf16return")
    parser.add_argument("--quality-models", default="qwen25_7b,llama31_8b")
    parser.add_argument("--quality-datasets", default="alpaca")
    parser.add_argument("--quality-steps", type=int, default=0)
    parser.add_argument("--quality-full-epoch",
                        action="store_true",
                        help="Run quality experiments for one full epoch by passing --max_steps=0.")
    parser.add_argument("--save-hf-model", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-at-start", action="store_true")
    parser.add_argument("--eval-max-batches", type=int, default=8)
    parser.add_argument("--mc-eval-interval", type=int, default=150)
    parser.add_argument("--mc-eval-tasks", default="mmlu,arc_challenge,hellaswag,boolq,openbookqa")
    parser.add_argument("--mc-eval-root", default="/home/shared/fastoffload_datasets")
    parser.add_argument("--mc-eval-max-examples", type=int, default=32)
    parser.add_argument("--mc-eval-at-start", action="store_true")
    parser.add_argument("--gen-eval-interval", type=int, default=150)
    parser.add_argument("--gen-eval-tasks", default="gsm8k")
    parser.add_argument("--gen-eval-root", default="/home/shared/fastoffload_datasets")
    parser.add_argument("--gen-eval-max-examples", type=int, default=32)
    parser.add_argument("--gen-eval-max-new-tokens", type=int, default=96)
    parser.add_argument("--gen-eval-at-start", action="store_true")
    parser.add_argument("--scaling-models", default="qwen25_3b,llama31_8b")
    parser.add_argument("--scaling-world-sizes", default="1,2,4,8")
    parser.add_argument("--scaling-steps", type=int, default=120)
    parser.add_argument("--model-sweep",
                        default="qwen25_1p5b,qwen25_3b,qwen3_8b,llama31_8b,mistral_nemo_12b,qwen25_14b,qwen3_32b")
    parser.add_argument("--timeline-models", default="llama31_8b,qwen25_7b")
    parser.add_argument("--timeline-steps", type=int, default=120)
    parser.add_argument("--hardware-methods", default="zero_offload,zenflow,fastoffload")
    parser.add_argument("--hardware-models", default="qwen25_3b,llama31_8b")
    parser.add_argument("--hardware-steps", type=int, default=120)
    parser.add_argument("--hardware-gpus", default="")
    parser.add_argument("--hardware-profile", action="store_true")
    parser.add_argument("--baseline-methods", default="zero_offload,zero3_offload,zenflow,fastoffload")
    parser.add_argument("--baseline-models", default="qwen25_3b,llama31_8b")
    parser.add_argument("--baseline-steps", type=int, default=80)
    parser.add_argument("--ablation-model", default="llama31_8b")
    parser.add_argument("--ablation-steps", type=int, default=120)
    parser.add_argument("--profile-model", default="llama31_8b")
    parser.add_argument("--profile-steps", type=int, default=60)
    parser.add_argument("--sensitivity-model", default="llama31_8b")
    parser.add_argument("--sensitivity-steps", type=int, default=120)
    parser.add_argument("--sensitivity-eval-interval", type=int, default=60)
    parser.add_argument("--sensitivity-k-values", default="2,4,8")
    parser.add_argument("--sensitivity-topk-values", default="0.05,0.10,0.15,0.20")
    parser.add_argument("--sensitivity-batch-sizes", default="4,8,16")
    parser.add_argument("--sensitivity-cpu-cores", default="0.4,0.6,0.8,1.0")
    parser.add_argument("--torch-profile-wait", type=int, default=20)
    parser.add_argument("--torch-profile-warmup", type=int, default=2)
    parser.add_argument("--torch-profile-active", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cwd = Path(__file__).resolve().parent
    experiments = build_experiments(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "specs.json").write_text(json.dumps([asdict(exp) for exp in experiments], indent=2,
                                                        sort_keys=True))
    if args.dry_run:
        for exp in experiments:
            print(json.dumps(asdict(exp), sort_keys=True))
        return
    if not args.skip_runs:
        for index, exp in enumerate(experiments, start=1):
            run_one(exp, args, cwd, index, len(experiments))
            summarize(experiments, args)
    summarize(experiments, args)
    print(f"[SUMMARY] {args.out_dir / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
