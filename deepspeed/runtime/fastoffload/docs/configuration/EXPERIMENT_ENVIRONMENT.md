<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- DeepSpeed Team -->

# Qwen2.5-7B Alpaca Experiment Environment

This file records the environment used by the matched one-epoch FastOffload and ZenFlow experiment launched on
September 3, 2026. The experiment output directory is
`/tmp/qwen_alpaca_epoch_comparison_20260903_011949` on the original machine.

## Source checkout

| Item | Value |
|---|---|
| FastOffload repository | `git@github.com:HenryYanglab/FastOffload.git` |
| Outer commit | `b5bab4dfd26f8e0ff112a4073b0b98c049b34a7b` |
| DeepSpeed fork | `git@github.com:HenryYanglab/DeepSpeed.git` |
| DeepSpeed branch | `fastoffload` |
| FastOffload commit | `2ce7e1005d454a9e55b2ae4e3e27656ed0966bc0` |
| Active source path | `/data/hangyu/ResearchHub/FastOffload/src/DeepSpeed` |

The process sets `PYTHONPATH` to the active source path. The imported source reports DeepSpeed
`0.19.6+24402c3be`; the installed distribution metadata reports `0.19.3+6498adc5`. The source path and Git commit
above are authoritative because they override the installed package at import time.

## Operating system and hardware

| Item | Value |
|---|---|
| Operating system | Red Hat Enterprise Linux 8.4 (Ootpa) |
| Kernel | `4.18.0-305.el8.x86_64` |
| glibc | 2.28 |
| System GCC | 8.5.0 |
| CPU | 2 sockets × Intel Xeon Platinum 8358P at 2.60 GHz |
| CPU cores | 64 physical cores, one thread per core |
| NUMA | 2 nodes: CPUs 0–31 and 32–63 |
| Host memory | 2.0 TiB |
| GPUs installed | 8 × NVIDIA A800-SXM4-80GB |
| GPUs used | Physical GPUs 6 and 7 |
| GPU memory | 81,920 MiB per GPU |
| GPU interconnect | NVLink NV8 |
| NVIDIA driver | 595.71.05 |
| Driver-supported CUDA | 13.2, as displayed by `nvidia-smi` |

The driver-supported CUDA version is not the CUDA runtime used by PyTorch. The experiment uses the CUDA 12.1
runtime and toolkit listed below.

## Conda and Python

| Item | Value |
|---|---|
| Conda environment | `deepspeed` |
| Environment path | `/home/hangyuan/.conda/envs/deepspeed` |
| Python | 3.10.20 |
| Python build compiler | GCC 14.3.0 |
| Python executable | `/home/hangyuan/.conda/envs/deepspeed/bin/python` |

Activation:

```bash
conda activate deepspeed
export PYTHONPATH=/data/hangyu/ResearchHub/FastOffload/src/DeepSpeed
```

## CUDA and PyTorch stack

| Component | Version |
|---|---|
| PyTorch | `2.5.1+cu121` |
| PyTorch CUDA runtime | 12.1 |
| Local CUDA toolkit / NVCC | 12.1 / V12.1.66 |
| cuDNN | 9.1.0 |
| NCCL | 2.21.5 |
| Triton | 3.1.0 |
| PyTorch build C++ standard | C++17 |
| PyTorch CPU math | Intel oneAPI MKL 2024.2 and oneDNN 3.5.3 |
| PyTorch CPU capability | AVX512 |

The relevant CUDA wheel packages are:

```text
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.0.2.54
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu12==12.1.0.106
nvidia-nccl-cu12==2.21.5
nvidia-nvtx-cu12==12.1.105
```

## Python packages

| Package | Version |
|---|---|
| `transformers` | 5.14.1 |
| `datasets` | 5.0.1 |
| `tokenizers` | 0.22.2 |
| `safetensors` | 0.8.0 |
| `numpy` | 2.2.6 |
| `pydantic` | 2.13.4 |
| `pyarrow` | 25.0.0 |
| `matplotlib` | 3.10.9 |
| `ninja` | 1.13.0 |
| `packaging` | 26.0 |
| `psutil` | 7.2.2 |
| `py-cpuinfo` | 9.0.0 |
| `accelerate` | not installed |

## Experiment workload

| Item | Value |
|---|---|
| Model | `/data/Qwen2.5-7B-Instruct` |
| Dataset | `/data/hangyu/datasets/alpaca` |
| Dataset size | 52,002 examples |
| Sequence cutoff | 512 tokens |
| Epochs | 1 |
| Precision | BF16 |
| ZeRO stage | 2 |
| Optimizer offload | CPU with pinned memory |
| Microbatch size | 1 per GPU |
| World size | 2 |
| Seed | 42 |
| Timing warmup | 20 microsteps |
| Gradient checkpointing | enabled |
| AdamW learning rate | `2e-5` |
| AdamW betas | `(0.9, 0.95)` |
| AdamW epsilon | `1e-8` |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |

FastOffload uses A/B/C ratios of approximately 10%/10%/80%, `update_interval=4`, GPU B accumulation,
`max_async_lag=2`, 128 MiB compressed buckets, BF16 gradient/return transfer, and
`pt_reserved_cores_perc=0.25`.

ZenFlow uses `topk_ratio=0.1`, `update_interval=4`, `overlap_step=true`, `full_warm_up_rounds=0`, and
`pt_reserved_cores_perc=0.25`. ZenFlow maps its update interval to GAS=4, so five ZenFlow warmup boundaries equal
the same 20 microsteps used by FastOffload.

The experiment was launched with:

```bash
CUDA_VISIBLE_DEVICES=6,7 \
CONDA_ENV=deepspeed \
bash deepspeed/runtime/fastoffload/scripts/run_qwen7b_alpaca_epoch_comparison.sh
```

Configuration files:

```text
deepspeed/runtime/fastoffload/scripts/fastoffload_qwen7b_alpaca_epoch.json
deepspeed/runtime/fastoffload/scripts/deepspeed_qwen7b_alpaca_zenflow_epoch.json
deepspeed/runtime/fastoffload/scripts/deepspeed_zero2_cpu_offload.json
```
