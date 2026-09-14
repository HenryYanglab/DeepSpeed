<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- DeepSpeed Team -->

# FastOffload 配置

配置模块支持 Phase 1 Observer、Phase 2 同步 Offload 和 Phase 3 异步重叠，独立于 DeepSpeed 的主配置。

## 示例

```json
{
  "enabled": true,
  "mode": "observe",
  "zero_stage": 2,
  "failure_policy": "raise",
  "telemetry": {
    "log_interval": 10,
    "parameter_events": true,
    "bucket_events": true,
    "host_timing": true,
    "device_timing": false,
    "memory_metrics": false,
    "distributed_summary": false,
    "rank_mode": "rank0_local",
    "debug_event_buffer_size": 0,
    "max_parameter_name_length": 128
  }
}
```

加载方式：

```python
from deepspeed.runtime.fastoffload.config import FastOffloadConfig

config = FastOffloadConfig.from_json("fastoffload.json")
```

也可以从字典构造：

```python
config = FastOffloadConfig.from_dict({
    "enabled": True,
    "telemetry": {"log_interval": 5},
})
```

同步模式示例见 `scripts/fastoffload_sync.json`：

```json
{
  "enabled": true,
  "mode": "sync_offload",
  "policy": {"type": "all_offload"},
  "scheduler": {"type": "synchronous", "max_inflight_tasks": 1},
  "transfer": {"pin_memory": true, "cpu_staging": false, "buffer_count": 1, "buffer_size": 134217728},
  "worker": {"type": "inline"}
}
```

异步模式示例见 `scripts/fastoffload_async.json`：

```json
{
  "enabled": true,
  "mode": "async_offload",
  "scheduler": {
    "type": "overlap",
    "max_inflight_tasks": 4,
    "max_inflight_bytes": 2147483648
  },
  "transfer": {
    "pin_memory": true,
    "cpu_staging": false,
    "async_strategy": "producer_stream",
    "buffer_count": 4,
    "buffer_size": 134217728
  }
}
```

异步模式要求 `buffer_count >= max_inflight_tasks`。单个 oversized shard 可以超过 `max_inflight_bytes`，但只有在队列为空时提交，避免永久阻塞。

## 字段

### 顶层字段

- `enabled`：是否启用，默认 `false`。
- `mode`：`observe`、`sync_offload` 或 `async_offload`；Offload 模式要求 `enabled=true`。
- `zero_stage`：当前只允许 ZeRO Stage 2。
- `failure_policy`：Observer 回调失败时采用 `raise`、`disable_observer` 或 `warn`。
- `policy`：Phase 2 当前只支持 `all_offload`。
- `scheduler`：同步模式使用 `synchronous` 且 `max_inflight_tasks=1`；异步模式使用 `overlap`，通过 `max_inflight_tasks` 和 `max_inflight_bytes` 限制队列。
- `transfer`：传输 buffer 配置；`pin_memory` 必须为 `true`。`cpu_staging=false` 默认直接写 ZeRO 最终 pinned CPU partition；设置为 `true` 仅用于旧 CPU staging 路径消融。Async 默认使用 `async_strategy=producer_stream`，直接在梯度 producer stream 提交 non-blocking D2H，不分配 GPU staging；`dedicated_stream` 保留独立 copy stream + GPU staging 路径用于 overlap 消融。buffer 数量和字节容量必须为正数，且 dedicated strategy 要求 `buffer_count >= max_inflight_tasks`。
- `worker`：当前只支持 `inline`。
- `telemetry`：Telemetry 子配置。

### Telemetry 字段

- `log_interval`：报告间隔，必须大于等于 1。
- `parameter_events`：是否采集参数级事件。
- `bucket_events`：是否采集 bucket 级事件。
- `host_timing`：是否采集 host timer。
- `device_timing`：预留字段，accelerator event timer 尚未实现，当前应保持关闭。
- `memory_metrics`：在 step 边界采集 accelerator current/peak allocated/reserved memory。
- `distributed_summary`：预留字段，分布式指标汇总尚未实现，当前应保持关闭。
- `rank_mode`：`rank0_local` 或 `per_rank`。
- `debug_event_buffer_size`：调试 ring buffer 容量，0 表示关闭。
- `max_parameter_name_length`：日志中的最大参数名称长度。
- `jsonl_path`：可选 JSONL snapshot 输出路径。
- `csv_path`：可选 long-format CSV 输出路径。

## Column importance selection

重要性选择默认关闭。以下配置在初始化时逐参数保存普通 CPU reference，并在第 10 个 optimizer step 后逐参数比较；二维权重按列的 L1 delta 排名，第一组为 `[0, K)`，第二组为 `[K, 2K)`：

```json
{
  "importance": {
    "enabled": true,
    "algorithm": "pretrained_delta_topk",
    "warmup_steps": 10,
    "topk_ratio": 0.1,
    "comparison_chunk_rows": 4096,
    "sparse_backward": true,
    "sparse_backward_min_tokens": 1024,
    "sparse_backward_min_output_features": 1024,
    "output_path": "/results/importance.pt"
  }
}
```

`K = ceil(column_count * topk_ratio)`，因此 `topk_ratio` 最大为 0.5。一维参数保持 dense。预训练 reference 始终位于普通 CPU 内存；`comparison_chunk_rows` 限制每次传入 GPU 的 reference chunk 和 FP32 delta scratch，不构造整层或全模型 delta。未包含 `{rank}` 的输出文件名会自动转换为 `importance.rank0.pt` 等 rank-local 文件。

`sparse_backward=true` 为 `torch.nn.Linear` 安装 opt-in backward：完整计算 `grad_input`，但 dW GEMM 只计算第一、第二重要组的列。独立 selective-validation 模式仍返回 zero-filled full-shape gradient；`zero2_takeover=true` 时则通过 packed-gradient side channel 直接把 selected dW 交给 Takeover runtime，weight 和 bias 的 autograd gradient 返回 `None`，不再分配、清零或 scatter dense dW。interval boundary 会动态切回原生 Linear backward 以生成 A+B+C。任意列 gather 在小 token 或小 output shape 上可能慢于 full cuBLAS GEMM，因此只有 `token_count >= sparse_backward_min_tokens` 且 `out_features >= sparse_backward_min_output_features` 时启用，否则自动走原生 backward。通用模型需要在 DeepSpeed 初始化前调用 `enable_selective_linear(model, minimum_tokens=...)`；随附 Alpaca launcher 会根据配置自动调用。

自定义算法可在安装 FastOffload 前调用 `register_importance_algorithm(name, factory)`；factory 接收 `topk_ratio` 和 `comparison_chunk_rows`，返回 `ImportanceAlgorithm`。

## Hybrid sparse/dense update

完整语义和状态机见 [HYBRID_UPDATE_DESIGN.md](HYBRID_UPDATE_DESIGN.md)。核心配置为：

```json
{
  "hybrid_update": {
    "enabled": true,
    "update_interval": 8,
    "accumulation_device": "cpu",
    "dense_boundary_enabled": true,
    "double_buffer": true,
    "second_gradient_reduction": "mean",
    "max_async_lag": 2,
    "pt_reserved_cores_perc": 0.25,
    "overdue_policy": "wait",
    "compressed_collective_shadow": false,
    "zero2_takeover": false,
    "compressed_bucket_bytes": 134217728,
    "parity_atol": 0.001,
    "parity_rtol": 0.001
  }
}
```

`accumulation_device` 支持 `cpu` 和 `gpu`，默认 CPU。CPU 模式节省显存，但会在普通步骤产生逐参数 D2H 和 host accumulation；以吞吐为首要目标且显存允许时应显式使用 `gpu`。Qwen2.5-7B、双 A800 的短程结果中，GPU accumulation 将 ordinary Takeover optimizer 从约 4.30 秒降至约 1.33 秒，而 boundary peak 仅从约 47.99 GiB/rank 增至约 48.70 GiB/rank。Hybrid 模式要求同时启用 importance selection。

`compressed_collective_shadow` 是实验开关，默认 `false`。关闭时原 ZeRO-2 reduction、D2H、CPU optimizer 和 all-gather 路径完全不变。设为 `true` 后，importance warmup 完成后的 A/B 梯度和 interval boundary 的完整矩阵梯度会额外经过 packed compressed all-reduce；结果仅用于验证并记录 telemetry，原 ZeRO optimizer 仍负责实际参数更新。当前 shadow 模式要求 `mode=observe` 和 GAS=1。

`zero2_takeover` 默认 `false`，并且不能与 `compressed_collective_shadow` 同时启用。启用后，importance warmup 仍走原生 ZeRO-2；选择完成后，A/B/C 梯度改用 owner-padded reduce-scatter，A 由 owner-local GPU Adam 每步更新，B/C 和 bias/norm 由异步 CPU Adam 在 interval boundary 更新，更新值通过 compressed all-gather 发布。Takeover 支持 GAS 和 unused parameters，并迁移原生 owner Adam moments。当前要求所有 optimizer parameter group 使用相同 Adam 超参数。设置 `importance.sparse_backward=true` 后，受支持的 Linear 在普通步骤使用 packed-gradient side channel；未包装模块仍生成 dense autograd gradient，但只通信和更新 A+B。DeepSpeed 已经按 GAS 缩放各 microbatch loss，因此 Takeover 对 compressed microbatch gradients 使用 sum，避免重复除以 GAS。

Takeover checkpoint 保存 importance layout、A/B/C optimizer state、successful-step/version 和活跃 B accumulator；保存前排空已提交CPU job并等待参数可见，不为末尾部分B额外生成C更新。LR scheduler状态仍由DeepSpeed管理，不包含在单独的Hybrid state字典中。

`compressed_bucket_bytes` 同时限制 Shadow 和 Takeover 待打包参数集合的目标大小。Takeover 按 parameter ID 确定性分桶，对每个桶独立执行 owner-padded reduce-scatter、更新和 all-gather，并在下一个桶前释放临时 packed tensor。单个参数不会被拆分，因此参数数据的目标上限是 `max(compressed_bucket_bytes, largest_compressed_parameter_bytes)`；owner padding 可能使实际 collective allocation 更高，实际值记录在 `takeover_owner_bucket_peak_bytes`。Shadow 归约结果在完成 native parity 和 owner partition 校验后立即释放，不再保留整模型 packed buffer。`parity_atol` 和 `parity_rtol` 控制 compressed/native、norm 和 owner-value 比对容差。

Shadow 目前校验 owner-rank 对应元素的 compressed collective 数值、有限性/overflow 决策、L2 norm，以及 D2H 后 ZeRO FP32 flat partition 的 owner offset/value。尚未用压缩结果替换 ZeRO-2 optimizer step，因此该开关用于分布式正确性和通信量验证，而不是性能训练。

## Takeover学习率scheduler

LR scheduler配置写在 **DeepSpeed主配置** 中，不是FastOffload配置里的传输 `scheduler` 字段。
标准DeepSpeed LR scheduler更新Native optimizer的LR；每个成功Takeover step经Adapter读取当前值给A，dense boundary的B/C任务固定使用该步A的同一LR。
CPU即使排队到后续step才执行，也不会改用后来的LR。构造时的 `_lr` / `_native_hyperparameters` 不再代表当前调度LR，审计应读取实际step/job参数。

例如，在主配置已有optimizer（峰值LR=2e-6）、GAS1的情况下，可以添加以下配置片段：

```json
{
  "scheduler": {
    "type": "WarmupCosineLR",
    "params": {
      "total_num_steps": 1024,
      "warmup_num_steps": 32,
      "warmup_min_ratio": 0.1,
      "cos_min_ratio": 0.1
    }
  }
}
```

这是接线示例，不是已完成的1024步LLM调参结果。`total_num_steps`以scheduler/成功optimizer更新为单位；GAS大于1时不能直接当作microstep数。
LR warmup与importance warmup不同；如果Native importance预热期间LR一直为0，pretrained-delta无法观察学习引起的权重变化，应联合检查两种warmup设置。
不改变B的mean/sum累积方式或C的边界梯度语义，不将LR额外乘/除update_interval。LR=0合法，负值/NaN/Inf和组间不同LR会被拒绝。
当前只同步LR，不能使用同时改变betas、eps或weight decay的调度配置。Overflow不推进scheduler；旧的有效CPU任务仍用提交时LR完成。
恢复完整训练还需恢复Native optimizer和scheduler状态；真实大模型optimizer-state resume仍未完成验证。

## 约束

配置模型采用严格且不可变的 Pydantic 模型：

- 未知字段会被拒绝，避免配置拼写错误被静默忽略；
- 默认值也会被校验；
- 构造后不能修改，运行时状态不得写入配置；
- JSON 文件不存在或格式错误时直接报告原始异常。

## 尚未实现的配置

`device_timing` 和 `distributed_summary` 已进入配置模型，但当前 Observer 尚未消费。`memory_metrics`、JSONL 和 CSV 已实现。

后续实现要求：

- Device Timing 必须延迟解析 accelerator event，不能在梯度热路径同步设备；
- Distributed Summary 只能在所有 rank 一致的报告边界执行，并需要处理异常和事件数量不一致，避免 collective 死锁。
