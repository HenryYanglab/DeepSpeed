<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- DeepSpeed Team -->

# Alpaca 微调与 FastOffload

本目录提供完整参数微调样例，可验证 Phase 1 Observer、Phase 2 同步 Offload 或 Phase 3 Backward/D2H 重叠。三个模式都在每一步更新所有梯度。

> Phase 3 执行异步 D2H，但尚未选择 Top-K、延迟参数更新或启动独立 CPU optimizer 进程。

## 文件

```text
scripts/
├── finetune_alpaca.py                  # Alpaca causal-LM 全参数微调
├── deepspeed_zero2_cpu_offload.json    # ZeRO-2 CPU optimizer offload
├── fastoffload_observer.json           # Phase 1 Observer
├── fastoffload_sync.json               # Phase 2 同步 Offload Baseline
├── fastoffload_async.json              # Phase 3 Backward/D2H 重叠
├── fastoffload_native.json             # 原生 ZeRO-2 对照配置
├── run_fastoffload_benchmarks.sh       # native/sync/async 吞吐与内存对比
├── run_phase3_stability.sh             # Phase 3 长时间稳定性测试
├── run_hybrid_shadow_validation.sh     # Native/Shadow 双 GPU 正确性验证
├── validate_hybrid_shadow.py           # loss 和 compressed telemetry 校验
├── fastoffload_hybrid_shadow.json      # Shadow 模式配置模板
├── fastoffload_cpu_b.json              # Takeover：B在CPU归约与累积（GAS=1）
├── prepare_benchmark_config.py         # 生成 run-specific telemetry 配置
├── summarize_benchmarks.py             # 生成 CSV 和 Markdown 汇总
├── benchmark_selective_backward.py     # Qwen shape selected-dW GPU microbenchmark
├── run_alpaca.sh                       # DeepSpeed 启动入口
├── requirements.txt                    # 样例额外依赖
└── README.md
```

## CPU B归约与累积

`fastoffload_cpu_b.json` 同时启用 `second_reduce_scatter_device=cpu` 和 `accumulation_device=cpu`。
需配合GAS=1的ZeRO-2 CPU optimizer offload主配置，以及覆盖整个world的数据并行组。
普通步和密集边界的真实B梯度都在CPU经Gloo Reduce-Scatter归约，再跨步累积；A/C仍在GPU归约。
它不是Native全量更新，也不是Shadow。未开启新开关的现有配置不改变归约位置。
首版B下传和CPU归约同步等待，CPUAdam仍异步；保留Native密集桶中的零B占位，不宣称密集通信量减少。
详细流程、精度和Checkpoint限制见[配置说明](../docs/configuration/CONFIGURATION.md)的“B在CPU归约与累积”。

## 安装

在 DeepSpeed 仓库根目录执行：

```bash
python -m pip install -r deepspeed/runtime/fastoffload/scripts/requirements.txt
python -m pip install -e .
```

CPU optimizer offload 会使用 DeepSpeed CPUAdam。首次运行可能需要编译对应 op，因此系统需要可用的 C++ 编译环境。

先运行不需要模型和 GPU 的脚本自测：

```bash
python deepspeed/runtime/fastoffload/scripts/smoke_test_alpaca.py
```

## 填写本地模型和数据集路径

编辑同目录文件：

```text
deepspeed/runtime/fastoffload/scripts/local_paths.sh
```

只需要填写：

```bash
LOCAL_MODEL_NAME_OR_PATH="/data/models/Llama-3.2-1B"
LOCAL_DATASET_PATH="/data/datasets/alpaca_data.json"
LOCAL_OUTPUT_DIR="/data/checkpoints/alpaca-observer"
LOCAL_HF_HOME="/data/huggingface-cache"
```

其中模型路径和数据集路径留空时会使用默认 Hugging Face 模型和 `tatsu-lab/alpaca`。填写以后直接执行：

```bash
bash deepspeed/runtime/fastoffload/scripts/run_alpaca.sh
```

临时环境变量的优先级高于 `local_paths.sh`，因此仍然可以在命令行覆盖路径。

## Phase 2 同步 Offload

在普通启动参数后覆盖 FastOffload 配置：

```bash
FASTOFFLOAD_CONFIG=deepspeed/runtime/fastoffload/scripts/fastoffload_sync.json \
bash deepspeed/runtime/fastoffload/scripts/run_alpaca.sh
```

周期报告会额外输出 `actual_offload`、`d2h_copy_mean_ms`、`d2h_bandwidth_mean_gbps`、`inline_worker_mean_ms` 和 `buffer_pool_high_watermark`。

## Phase 3 异步重叠

```bash
FASTOFFLOAD_CONFIG=deepspeed/runtime/fastoffload/scripts/fastoffload_async.json \
bash deepspeed/runtime/fastoffload/scripts/run_alpaca.sh
```

异步报告会额外输出 `queue_high_watermark`、`queue_wait_mean_ms`、`step_flush_mean_ms`、`gpu_source_hold_mean_ms` 和 `transfer_hidden_ratio`。

## Selective backward GPU microbenchmark

使用 Qwen2.5-7B 常见 attention/MLP shape 分别测量 grad-input、native dW、input gather 和 selected dW：

```bash
python deepspeed/runtime/fastoffload/scripts/benchmark_selective_backward.py \
  --tokens 128 323 512 1024 \
  --selected-ratio 0.2 \
  --warmup 5 \
  --iterations 20 \
  --output selective_backward.json
```

A800、BF16、323 tokens 的样例结果显示：attention Q、MLP gate/up、MLP down 的估算 Linear backward speedup 分别约为 1.06x、1.10x、1.17x；KV projection 因 output features 较小而约为 0.75x。默认 `sparse_backward_min_output_features=1024` 因此跳过小 output projection。该 benchmark 只用于 kernel 选择，不代表端到端训练性能。

## GPU-only ceiling diagnostic

`benchmark_gpu_ceiling.py` measures the GPU-side lower-work reference, **not a valid training method or a mathematical
throughput bound**. It temporarily installs a diagnostic runtime; the production Hybrid A/B/C runtime is unchanged.

It retains full forward, ordinary packed A/B backward, full dense-boundary backward, gradient reduction, global numerics,
owner A GPUAdam, and A all-gather. After the initial native importance warmup it disables B/C state migration, CPU jobs,
B accumulation, gradient D2H, parameter H2D, and B/C publication. Guard functions fail if a disabled CPU path is invoked.
B/C remain frozen, so loss/convergence and model export are not supported. Omitting B accumulation/publication makes this
an optimistic ceiling diagnostic, not an isolated semantics-preserving CPU-overlap ablation.

The script preloads the finite input window to GPU, removing batch H2D as well. Native importance warmup and initial A
state migration still use CPU/GPU transfers, but must complete before the timed window. Scalar norm/overflow control,
host kernel-launch work, and inter-GPU collectives are retained. It reports all-rank padded/input/supervised token sums,
maximum-rank synchronized wall time, maximum-rank measured-window allocated/reserved peaks, and per-rank CUDA-event
forward/backward/A-step stream elapsed times (including gaps/waits, not pure kernel time). No per-step synchronization is
added except the existing dense-boundary publication synchronization.

From the DeepSpeed repository root, using a run-specific FastOffload config with distinct telemetry output paths:

```bash
CUDA_VISIBLE_DEVICES=6,7 deepspeed --master_port 29618 \
  deepspeed/runtime/fastoffload/scripts/benchmark_gpu_ceiling.py \
  --model_name_or_path /data/Qwen2.5-7B-Instruct \
  --dataset_path /data/hangyu/datasets/alpaca \
  --deepspeed_config deepspeed/runtime/fastoffload/scripts/deepspeed_zero2_cpu_offload.json \
  --fastoffload_config /path/to/run/fastoffload.json \
  --max_steps 100 --benchmark_warmup_steps 20 --max_length 512 \
  --gradient_accumulation_steps 1 --learning_rate 5e-6 --no-save_model \
  --benchmark_jsonl /path/to/run/benchmark.jsonl
```

Use the takeover settings in `fastoffload_qwen7b_alpaca_epoch.json`; only GAS=1, constant LR and one epoch are supported.
The measured window must contain complete update intervals. Runtime monkeypatching is scoped to this standalone process
and restored on exit; never enable this diagnostic in a production fine-tuning run or export it as a Hybrid checkpoint.

## Hybrid Shadow 验证

Shadow 验证脚本使用相同 seed、模型、数据和训练参数依次运行 Native control 与 Hybrid shadow。它验证：

- 两次运行逐 step loss 在容差内一致，确认 shadow 不参与参数更新；
- Native control 没有执行 compressed collective；
- Shadow 同时执行了 A+B selected collective 和 interval dense-boundary collective；
- selected collective 的通信字节数小于 dense boundary；
- compressed owner values 与 native reduced gradient 一致；
- compressed/native overflow 和 L2 norm 一致；
- D2H 后 owner offset/value 与 ZeRO FP32 partition 一致；
- chunked bucket peak 小于整模型 dense packed buffer。

```bash
conda activate fastoffload2
cd /data/hangyu/ResearchHub/FastOffload/src/DeepSpeed/deepspeed/runtime

bash fastoffload/scripts/run_hybrid_shadow_validation.sh
```

默认使用两张空闲 GPU、GAS=1、4 个 optimizer steps、2-step importance warmup 和 interval=2。可以覆盖：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
MODEL_NAME_OR_PATH=/data/Qwen2.5-7B-Instruct \
DATASET_PATH=/data/hangyu/datasets/alpaca \
MAX_STEPS=6 \
IMPORTANCE_WARMUP_STEPS=2 \
UPDATE_INTERVAL=2 \
MAX_LENGTH=128 \
COMPRESSED_BUCKET_BYTES=134217728 \
bash fastoffload/scripts/run_hybrid_shadow_validation.sh
```

结果写入：

```text
shadow_validation_results/<timestamp>/
├── experiment.env
├── native/{fastoffload.json,train.log,telemetry.jsonl,benchmark.jsonl}
├── shadow/{fastoffload.json,train.log,telemetry.jsonl,benchmark.jsonl}
└── validation_summary.json
```

Shadow 会额外执行通信，因此该脚本只验证正确性和压缩比例，不能用于吞吐性能结论。

## Qwen2.5-7B Alpaca 单 epoch 对比

下面的匹配实验使用 Alpaca 微调中常见的 `512` token cutoff，训练一个完整 epoch，并从计时中排除前
20 个 microsteps：

```bash
CUDA_VISIBLE_DEVICES=6,7 CONDA_ENV=deepspeed \
  bash run_qwen7b_alpaca_epoch_comparison.sh
```

FastOffload 每四个 microsteps 输出一个平均 loss。ZenFlow 会把 `update_interval=4` 映射为 GAS=4，
因此 ZenFlow 的五个 warmup boundaries 等于同样的 20 个 warmup microsteps。脚本按相同数据窗口比较，
并在 `$RESULT_ROOT` 下生成 `summary.json`、`loss_curve.{png,svg}`、各方法的 benchmark JSONL、Telemetry
和训练日志。可通过 `RESULT_ROOT`、`MODEL_NAME_OR_PATH` 和 `DATASET_PATH` 覆盖本地默认路径。

## Benchmark 与长期稳定性

运行 native ZeRO-2、Phase 2 sync 和 Phase 3 async 三组对比，每组先 warmup 10 steps，再测量 100 steps：

```bash
conda activate fastoffload2
cd /data/hangyu/ResearchHub/FastOffload/src/DeepSpeed/deepspeed/runtime

REPEATS=3 \
MEASURE_STEPS=100 \
WARMUP_STEPS=10 \
MAX_LENGTH=512 \
bash fastoffload/scripts/run_fastoffload_benchmarks.sh
```

默认结果目录：

```text
fastoffload/scripts/benchmark_results/<timestamp>/
├── experiment.env
├── native/repeat_*/
├── sync/repeat_*/
├── async/repeat_*/
├── runs.csv
├── summary.csv
└── summary.md
```

每次运行目录包含：

```text
benchmark.jsonl    # 端到端吞吐、step time、GPU/pinned memory
telemetry.jsonl    # 每个报告窗口的完整 Metrics snapshot
telemetry.csv      # long-format 指标表
train.log
fastoffload.json   # 本次实际配置
```

只比较指定模式：

```bash
MODES="native async" REPEATS=3 MEASURE_STEPS=100 \
bash fastoffload/scripts/run_fastoffload_benchmarks.sh
```

Phase 3 长跑默认执行 10-step warmup 和 1000 个测量 steps，并在结束后扫描 NCCL、Traceback 和 RuntimeError：

```bash
MEASURE_STEPS=1000 LOG_INTERVAL=50 \
bash fastoffload/scripts/run_phase3_stability.sh
```

常用环境变量：

- `RESULT_DIR`：自定义结果目录；应使用空目录。
- `MODES`：空格分隔的 `native sync async selective`；`selective` 使用 importance 配置和带 token 阈值的列选择 backward。
- `REPEATS`：每种模式重复次数。
- `MEASURE_STEPS`、`WARMUP_STEPS`：测量和预热 step。
- `NUM_GPUS`、`MICRO_BATCH_SIZE`、`GRADIENT_ACCUMULATION_STEPS`；benchmark 默认 `MICRO_BATCH_SIZE=2`、GAS=1，在两张 GPU 上对应 global batch 4。
- `MAX_LENGTH`、`MAX_SAMPLES`、`LOG_INTERVAL`。
- `MODEL_NAME_OR_PATH`、`DATASET_PATH`：覆盖本地路径。

`accelerator_incremental_peak_bytes_per_rank` 是模型初始化后重置 peak statistics 得到的训练增量；`cumulative_pinned_bytes_per_rank` 是本进程通过 DeepSpeed accelerator 执行 page-lock 的累计字节；`pinned_pool_peak_bytes_per_rank` 和 `gpu_staging_pool_peak_bytes_per_rank` 是 FastOffload pool 的精确峰值容量。默认 direct-destination 路径中 `pinned_pool_peak_bytes_per_rank=0`，因为直接复用 ZeRO 最终 pinned gradient partition；设置 `transfer.cpu_staging=true` 才会分配额外 CPU pool。

正式比较建议至少 `REPEATS=3`，并确保其他 GPU/CPU/NUMA 负载一致。短 smoke benchmark 的结果不能用于性能结论。

## 快速训练 Smoke Test

下面只使用 16 条样本、执行 2 个 optimizer steps：

```bash
MODEL_NAME_OR_PATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
NUM_GPUS=1 \
bash deepspeed/runtime/fastoffload/scripts/run_alpaca.sh \
  --max_samples 16 \
  --max_steps 2 \
  --gradient_accumulation_steps 2 \
  --max_length 256 \
  --no-save_model
```

默认数据集为 Hugging Face `tatsu-lab/alpaca`，默认模型为 TinyLlama 1.1B。模型和数据集需要网络访问或已经存在于本地缓存。

## 正式训练示例

```bash
MODEL_NAME_OR_PATH=/models/Llama-3.2-1B \
NUM_GPUS=2 \
OUTPUT_DIR=/checkpoints/alpaca-fastoffload-observer \
bash deepspeed/runtime/fastoffload/scripts/run_alpaca.sh \
  --max_steps 1000 \
  --num_train_epochs 3 \
  --micro_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_length 1024 \
  --learning_rate 2e-5
```

脚本会按照实际 `WORLD_SIZE` 自动计算：

```text
train_batch_size = micro_batch_size × gradient_accumulation_steps × world_size
```

## 使用本地 Alpaca 数据

支持 JSON、JSONL 文件，以及 Hugging Face `Dataset.save_to_disk()`/`DatasetDict.save_to_disk()` 目录：

```bash
bash deepspeed/runtime/fastoffload/scripts/run_alpaca.sh \
  --model_name_or_path /models/TinyLlama-1.1B \
  --dataset_path /datasets/alpaca_data.json \
  --local_files_only \
  --max_steps 100
```

数据至少包含：

```json
[
  {
    "instruction": "解释什么是梯度累积。",
    "input": "",
    "output": "梯度累积是将多个 micro-batch 的梯度累加后再更新参数。"
  }
]
```

其中 `input` 可以省略或为空，`instruction` 和 `output` 必须存在。

## Precision

默认 `--precision auto`：支持 BF16 时使用 BF16，否则使用 FP16。也可以显式指定：

```bash
--precision bf16
```

或：

```bash
--precision fp16
```

## 当前 Observer 如何使用

关键顺序是：先安装 Observer，再调用 `deepspeed.initialize()`。

```python
import deepspeed
from deepspeed.runtime.fastoffload import FastOffloadConfig, install

observer_config = FastOffloadConfig.from_json(
    "deepspeed/runtime/fastoffload/scripts/fastoffload_observer.json"
)
handle = install(observer_config)
engine = None

try:
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=deepspeed_config,
    )

    loss = engine(**batch).loss
    engine.backward(loss)
    engine.step()
finally:
    if engine is not None:
        engine.destroy()
    handle.close()
```

必须满足：

```text
zero_optimization.stage = 2
offload_optimizer.device = cpu
```

如果没有调用 `install()`，或配置中 `enabled=false`，ZeRO-2 使用 Null Controller，Observer 不会采集指标。

## 当前采集的指标

每隔 `telemetry.log_interval` 个 optimizer steps，rank 0 输出类似：

```text
[FastOffload Observer]
rank=0 completed_steps=10
backward_mean_ms=842.13
optimizer_step_mean_ms=311.40
gradient_shards=1870
potential_offload=18.20 GiB
observer_callback_mean_ms=0.0021
```

主要指标包括：

- `backward_count`、`backward_host_ms`
- `gradient_ready_count`
- `gradient_reduced_count`
- `gradient_local_shard_numel`
- `gradient_local_shard_bytes`
- `potential_offload_bytes`
- `gradient_bucket_count`、`gradient_bucket_bytes`
- `optimizer_step_host_ms`
- `observer_callback_ms`

`potential_offload_bytes` 是当前本地 ZeRO partition 理论上可 offload 的数据量，不代表 FastOffload 已经执行了额外传输。

## 在代码中读取 Metrics

在报告窗口重置前，可以通过 Controller 获取快照：

```python
controller = engine.optimizer._fastoffload_controller
snapshot = controller.observer.snapshot(reset=False)

print(snapshot.counters)
print(snapshot.gauges)
print(snapshot.histograms)
```

`_fastoffload_controller` 当前是 DeepSpeed 内部集成属性，适合实验和诊断，不应作为长期稳定的用户 API。常规使用应依赖周期日志。

## 关闭和错误处理

- `engine.destroy()` 会关闭绑定到 optimizer 的 Controller。
- `handle.close()` 会注销进程内 FastOffload 配置，并关闭仍然存活的 Controller。
- 两者都可以重复调用。
- 同一进程同时只能存在一个活动的 FastOffload installation。
- 默认 `failure_policy=raise`，Observer 内部错误会立即终止训练，避免输出不可信指标。

## 当前限制

- 只支持 ZeRO Stage 2 CPU optimizer offload。
- 不支持 ZeRO-1、ZeRO-3、NVMe offload 和 ZenFlow 组合验证。
- 只实现 host timer；`device_timing` 尚未产生 accelerator event 指标。
- `distributed_summary=false`；不会额外发起指标 all-reduce。
- `memory_metrics` 配置字段已预留，但当前 Observer 尚未输出 CPU/GPU 内存指标。
- 当前训练样例是全参数 SFT，不包含 LoRA/QLoRA。
