<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- DeepSpeed Team -->

# FastOffload 当前实现状态

本文档汇总当前 `fastoffload` 分支已经落地的功能、运行路径、验证结果和已知限制。FastOffload 当前的生产目标是
**DeepSpeed ZeRO Stage 2 + CPU optimizer offload**；关闭 FastOffload 时保持原生 ZeRO-2 行为。

## 1. 总体架构

FastOffload 通过少量生命周期入口接入 DeepSpeed，并将具体逻辑隔离在以下组件中：

| 组件 | 职责 |
|---|---|
| Adapter | 隔离 ZeRO-2 私有状态和 partition 布局 |
| Controller | 生命周期调度、模式选择和失败策略 |
| Policy | 判断梯度是否 offload、何时执行动作 |
| Scheduler | 同步或异步任务排序与反压 |
| Transfer Engine | D2H/H2D 传输、stream、event 和 buffer pool |
| Worker | CPU 侧处理与 optimizer 执行 |
| Importance | 重要性计算、选择和状态恢复 |
| Hybrid Runtime | A/B/C 梯度、更新和发布语义 |
| Telemetry | 时间、显存、传输和参数级统计 |

`Zero2ObserverAdapter` 是 FastOffload 中唯一允许读取 ZeRO-2 私有字段的组件。主要集成点位于：

```text
deepspeed/runtime/zero/stage_1_and_2.py
deepspeed/runtime/fastoffload/
```

## 2. 配置和运行模式

FastOffload 使用独立 JSON 配置，不改变原有 DeepSpeed JSON 的含义。当前支持：

| 模式 | 状态 | 说明 |
|---|---|---|
| Disabled | 已实现 | 完全使用 Native ZeRO-2 路径 |
| Observer | 已实现 | 默认只采集 metadata，不改变训练 |
| Synchronous Offload | 已实现 | 直接写入目标 CPU tensor 的同步 D2H |
| Asynchronous Offload | 已实现 | producer-stream 或独立 copy-stream D2H |
| Importance | 已实现 | pretrained-to-warmup input-column selection |
| Hybrid Shadow | 已实现 | 验证压缩结果，不修改 live parameter |
| Hybrid Takeover | 已实现 | A/B/C 路径接管梯度、optimizer 和 publication |

Shadow 和 Takeover 互斥，默认均不启用。Takeover 当前只支持 ZeRO-2 CPU optimizer offload。

## 3. Native 行为保护

FastOffload disabled 时保留以下原生行为：

```text
ZeRO-2 gradient reduction
CPU optimizer D2H/H2D
parameter all-gather
loss scaling
GAS
unused parameters
overflow detection
global norm and clipping
optimizer state
checkpoint
```

Observer 默认不读取 gradient values、不增加 collective，也不持有额外 gradient tensor。

## 4. Observer 和 Telemetry

当前已实现的观测能力包括：

- gradient-ready、gradient-reduced、optimizer-step 等生命周期事件；
- 参数名、shape、dtype、device、partition 和 bucket metadata；
- host timer 和可选 accelerator timer；
- 当前显存、峰值 allocated/reserved memory；
- gradient shard 和潜在 offload 字节数；
- pinned-memory allocation 次数和累计字节；
- owner bucket 数量、packed bytes 和 peak bucket bytes；
- Native dense boundary 次数；
- CPU D2H completion wait 和 CPUAdam 时间；
- stdout、JSONL、CSV 和 benchmark JSONL 输出；
- rank-0 local 或 distributed summary 策略。

Observer callback 使用有界 debug event buffer，默认不保存参数级历史事件。

## 5. 重要性选择

当前生产算法为 `PretrainedDeltaTopK`：

```python
score = abs(warmup_weight - pretrained_weight).sum(dim=0)
```

它沿二维权重 `[out_features, in_features]` 的 `dim=1` 选择输入列：

```text
A：排名 [0, K) 的 first-important columns
B：排名 [K, 2K) 的 second-important columns
C：其余 columns
```

当前功能包括：

- pretrained CPU reference snapshot；
- warmup 后一次选择；
- GPU 分块比较，避免完整 FP32 comparison tensor；
- CPU reference 和 bounded GPU row chunks；
- `ImportanceRegistry` checkpoint state；
- 自定义算法注册接口 `register_importance_algorithm()`；
- Qwen2.5-7B 198 个二维参数选择；
- 重要性选择耗时从早期约 669 秒降低至约 12 秒。

## 6. Packed Selective Backward

普通步骤对满足条件的 Linear 启用 packed selective backward：

```text
Forward：保持原始 Linear 语义
Backward grad_input：完整计算
Backward grad_weight：只计算 A+B columns
```

普通步骤不会构造 dense zero-filled weight gradient：

```python
grad_weight = None
grad_bias = None
```

A/B packed gradients 通过 side channel 交给 Takeover runtime。当前还包括：

- minimum token 和 output-feature 阈值；
- adaptive selective fallback；
- unsupported projection 回退到 Native backward；
- selective GEMM microbenchmark；
- GAS microbatch gradient sum；
- unused parameter 处理。

Dense boundary 仍然计算完整梯度，因此不会丢失 C 或 dense parameter 的梯度覆盖。

## 7. Hybrid A/B/C 更新语义

默认 `update_interval=4` 时：

```text
普通成功步骤：
  A：GPU Adam，每步更新
  B：GPU 累积
  C：不生成 packed weight gradient

Dense boundary：
  A：GPU Adam
  B：合并 ordinary + boundary gradients 后异步 CPUAdam
  C：本次完整 gradient 异步 CPUAdam
  Bias/Norm/1D：在 boundary 异步 CPUAdam
```

当 `second_gradient_reduction="mean"` 时：

```text
B gradient = interval 内所有成功步骤 B gradient 的和 / 成功步骤数
```

Overflow step 不会更新 A、累积 B、提交 C，也不会推进 successful-step scheduler。

## 8. Owner-Sharded GPU 更新和通信

A 使用 owner-local GPU optimizer state：

```text
compressed A gradient
→ owner-aware reduce-scatter
→ owner-local FP32 Adam
→ compressed all-gather
→ 所有 rank scatter A columns
```

已实现：

- `ParameterPartitionLayout` contiguous owner segments；
- owner count、local offset 和 fragment extraction；
- 不创建 per-element owner tensor 或 boolean mask；
- deterministic parameter-level bucket packing；
- bucket 内不拆分单个参数；
- oversized parameter 和 owner padding；
- 根据 global logical payload 确定 bucket membership；
- 所有 rank 使用相同 collective 数量和顺序；
- compressed reduce-scatter 和 updated-value all-gather；
- owner-only write validation。

普通步骤的 A+B gradient coverage 约 20%；每四步一次 dense boundary 时，平均 gradient coverage 为：

```text
(3 × 20% + 100%) / 4 = 40%
```

## 9. Native ZeRO-2 Dense-Boundary Streaming

Dense boundary 不保留额外 full-model GPU gradient。当前生命周期为：

```text
full layer/bucket gradient
→ Native ZeRO-2 reduce-scatter
→ owner-local A/B/C split
→ A/B 保留在 GPU
→ C 直接复制到 shared+pinned CPU slot
→ 立即释放 dense owner fragment
```

该路径包括：

- Native gradient-ready bucket capture；
- missing-owner zero-length entry 对齐；
- deterministic cross-rank entry ordering；
- direct-to-final-offset C staging；
- small-model staging fallback；
- overflow discard 和 Native capture cleanup；
- 不分配额外 full-size GPU staging tensor。

Qwen2.5-7B dense-boundary peak 已从早期约 49–60 GiB/rank 降至约 30–31 GiB/rank。

## 10. 异步 CPUAdam

B/C 使用独立 spawned process 执行 Native `DeepSpeedCPUAdam`：

```text
GPU BF16 owner gradient
→ version-owned shared+pinned BF16 gradient slot
→ CPU FP32 workspace
→ DeepSpeedCPUAdam
→ canonical FP32 master/moments
→ persistent shared+pinned BF16 return buffer
```

已实现：

- CPU optimizer 独立进程，不占用训练线程；
- shared-memory master、moments、gradient 和 return storage；
- `cudaHostRegister`/`cudaHostUnregister`；
- 按 rank 划分 CPU physical cores；
- `pt_reserved_cores_perc` 配置；
- bounded async lag 和 overdue backpressure；
- gradient slots 与 version 一一对应；
- slot 未释放前禁止复用；
- CPUAdam 完成后再发布 version；
- CPUAdam 和 D2H completion wait 遥测。

`max_async_lag=2` 时使用两个 gradient slots。参数 return buffer 由 visibility event 保护，在上一个 version 对 GPU
可见前不会被下一个 CPU version 覆盖。

## 11. Parameter Publication 和 Visibility

CPU update 完成后的发布路径为：

```text
pinned BF16 return values
→ nonblocking H2D
→ compressed all-gather
→ scatter selected columns
→ record CUDA visibility event
→ next Forward wait_event
```

当前保证：

- Forward 不读取部分发布的 version；
- publication 按 version 顺序执行；
- 每次跨 rank 只提交一个共同 ready version；
- H2D 完成前 return buffer 不会被覆盖；
- checkpoint 和 shutdown 会同步最后一个 visibility event。

## 12. Numerics、Overflow 和 Clipping

当前支持：

- DeepSpeed loss scale；
- BF16/FP16 overflow 检测；
- owner-local FP32 norm accumulation；
- distributed norm SUM；
- distributed overflow MAX；
- global gradient clipping；
- ordinary packed path 和 Native dense-boundary numerics；
- overflow 时清空 active accumulation 和 Native staging；
- overflow step 参数不更新、scheduler 不推进。

当前完整支持 CPU job 提交前的 overflow rollback。已经进入 queued/running CPUAdam 的任意时刻取消仍属于后续增强项。

## 13. Checkpoint 和 Shutdown

Checkpoint 当前保存：

- Importance registry；
- successful-step scheduler；
- A GPU optimizer state；
- B/C CPU optimizer FP32 master 和 moments；
- committed version；
- active B accumulator；
- dense parameter optimizer state。

`state_dict()` 和 shutdown 执行：

```text
等待 pending CPU jobs
→ 全 rank 检查相同 next-ready version
→ 每次 commit 一个 version
→ 等待 visibility
→ 保存状态或关闭 worker
```

已实现 checkpoint drain、load 后继续 interval、最终 worker join 和 pinned buffer unregister。

## 14. 测试和模型验证

当前 Hybrid unit suite：

```text
27 passed, 2 skipped
```

双 GPU integration 覆盖：

- owner reduce-scatter/all-gather；
- rank 参数一致性；
- Native dense boundary；
- pending-update checkpoint drain；
- Hybrid state reload；
- overflow 不更新参数；
- shutdown drain。

Qwen2.5-7B Shadow 两 GPU验证曾得到：

```text
maximum_loss_difference：0.0
parity_check_count：     200
owner_write_check_count：200
status：                 passed
```

Qwen2.5-7B、Alpaca、sequence cutoff 512 的 FastOffload 单 epoch 已完成：

```text
microsteps：              26001
warmup：                  20 microsteps
measured time：           33050.26 s（9.18 h）
throughput：              168.72 padded tokens/s
peak allocated memory：   30.74 GiB/rank
```

完整环境见 [EXPERIMENT_ENVIRONMENT.md](EXPERIMENT_ENVIRONMENT.md)。完整 ZenFlow 单 epoch 对比仍应以对应实验结束后
生成的最终 summary 为准。

## 15. 脚本和资产

当前提供：

- Alpaca smoke test 和正式训练脚本；
- Native、Observer、同步和异步配置；
- Importance 和 Hybrid Shadow 配置；
- selective backward benchmark；
- FastOffload/ZenFlow 单 epoch 对比 runner；
- JSONL/CSV benchmark summary；
- Qwen2.5-7B 实际权重和 A/B/C selection thumbnails；
- Overview figure brief 和绘图 prompt；
- FastOffload/ZenFlow loss curve PNG/SVG。

## 16. 当前限制和后续工作

尚未完成或不属于当前生产范围的项目：

1. ZeRO Stage 3 Takeover；
2. 多 optimizer group 使用独立 Adam hyperparameters 和 authority；
3. queued/running CPUAdam job 的任意时刻 overflow cancellation；
4. gradient D2H 和 parameter H2D 独立 dtype 配置；
5. 所有 unsupported dtype/layout、embedding 和 output head 的 bucket-local fallback；
6. persistent packed GPU workspace 和完整 fragmentation cost model；
7. pack、cast、D2H、CPUAdam、CPU cast、H2D、visibility wait 的完整 CUDA-event 字节/带宽统计；
8. output-row importance 作为可选 selection axis；
9. 多 seed、完整 validation set 和多 epoch 收敛评估。

FastOffload 不会把 Hybrid A/B/C 替换为论文中的 selected-only、ordinary-no-update 算法。当前 A 每步更新、B 累积、
B/C boundary 更新和 dense boundary 完整梯度语义是保留的核心算法。
