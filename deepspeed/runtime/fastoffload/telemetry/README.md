<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- DeepSpeed Team -->

# Telemetry

Telemetry 提供性能观测、诊断和策略反馈。第一阶段 FastOffload 应只实现该模块和空操作 hook，以验证接入点不会改变训练结果。

详细设计参见 [Observer 与 Telemetry 实现设计](OBSERVER_DESIGN.md)。

## 职责

- 记录 backward、gradient-ready、copy、queue、CPU optimizer 和 commit 时间。
- 统计 D2H/H2D 字节数、有效带宽、队列深度和 pinned-memory 使用量。
- 记录 Policy 决策分布和参数版本等待。
- 输出 rank-local 指标，并支持 rank 0 聚合摘要。
- 为 Adaptive Policy 提供只读指标快照。

## 已实现文件

```text
telemetry/
├── README.md
├── metrics.py   # counter/gauge/running statistics
├── timers.py    # 不同步 accelerator 的 host timer
├── observer.py  # 生命周期指标聚合
└── reporter.py  # 周期性 rank-local 摘要
```

## 使用样例

```python
with telemetry.timer("gradient_d2h"):
    ticket = transfer_engine.copy_to_cpu(...)

telemetry.increment("offload_bytes", context.nbytes)
telemetry.observe("worker_queue_depth", worker.queue_depth)
```

异步 accelerator 操作不能只使用普通 CPU wall-clock timer；需要通过 event 在操作完成后计算真实耗时。

## 推荐指标

```text
backward_ms
optimizer_step_ms
gradient_ready_to_copy_ms
d2h_copy_ms
d2h_bandwidth_gbps
worker_queue_wait_ms
cpu_optimizer_ms
parameter_commit_wait_ms
inflight_bytes
pinned_memory_bytes
keep_gpu_ratio
offload_cpu_ratio
stale_update_count
```

## Observer 模式

```json
{
  "enabled": true,
  "mode": "observe",
  "telemetry": {
    "log_interval": 10,
    "rank_mode": "rank0_local"
  }
}
```

Observer 模式只记录事件，必须与未启用 FastOffload 的损失、梯度和参数更新结果一致。

## 当前实现状态

| 能力 | 状态 | 说明 |
|---|---|---|
| 参数和 gradient shard 计数 | 已实现 | 只读取 shape、dtype、device、offset 等元数据 |
| Gradient bucket 计数 | 已实现 | 在 ZeRO-2 bucket 清理前记录元素数和字节数 |
| Backward/optimizer host timing | 已实现 | 使用 `time.perf_counter_ns()`，不执行设备同步 |
| Rank-local 周期报告 | 已实现 | `rank0_local` 或 `per_rank`，不新增 collective |
| 固定容量 debug event buffer | 已实现 | 只保存不可变元数据，满后覆盖最旧事件 |
| 通用 Device event timing | 未实现 | Phase 3 只对异步 D2H 使用 timing event；Observer 的任意设备阶段计时仍未实现 |
| GPU memory metrics | 已实现 | `memory_metrics=true` 时在 step 边界读取 accelerator current/peak allocated/reserved |
| Distributed metric summary | 未实现 | 属于 Phase 1 增强项；新增 collective 可能改变时序或因 rank 事件不一致而死锁 |
| JSONL/CSV 输出 | 已实现 | `jsonl_path` 保存嵌套 snapshot，`csv_path` 保存 long-format 指标 |
| 参数级聚合报告 | 未实现 | Context 已包含参数 ID/名称，但 Reporter 尚未按参数汇总 |
| 实际 D2H 字节/时间/带宽 | 已实现 | Phase 2/3 记录已完成的实际传输，不再只是 potential bytes |
| 异步队列和隐藏比例 | 已实现 | Phase 3 记录 inflight、queue wait、step flush、source hold 和近似 hidden ratio |

在上述功能实现前，建议配置保持：

```json
{
  "device_timing": false,
  "memory_metrics": true,
  "distributed_summary": false
}
```

通用 device timing 和 distributed summary 仍是可选增强项，不阻塞 Offload 数据路径。Phase 2 已实现同步 pinned D2H，Phase 3 已实现 copy stream/event 与 backward 重叠；策略化更新属于 Phase 4。
