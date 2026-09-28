<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- DeepSpeed Team -->

# FastOffload 总体架构设计

## 1. 背景与目标

FastOffload 用于研究和扩展 DeepSpeed ZeRO-Offload，重点控制 forward/backward 期间的梯度归约、GPU/CPU 传输、梯度累积和 optimizer 更新，例如实现与 ZenFlow 相似的选择性更新与计算重叠能力。

目标如下：

- 将实验算法集中在独立模块中，避免散落到 DeepSpeed Engine 和 ZeRO Optimizer。
- 支持 Policy、Scheduler、Transfer Engine 和 Worker 分别演进。
- 通过稳定 Context 隔离 ZeRO-2/ZeRO-3 私有数据结构。
- 支持逐模块实现、测试和性能验证。
- 未启用 FastOffload 时不改变 DeepSpeed 原有行为。

## 2. 为什么不能完全零修改

控制 ZeRO-Offload 的内部数据通路需要介入以下关键时刻：

1. backward 开始；
2. gradient partition/reduce 完成；
3. GPU gradient buffer 被复用或释放之前；
4. optimizer step 前后；
5. ZeRO-3 parameter fetch/release 前后。

这些时刻无法仅通过训练脚本的公开 API 完整覆盖。完全依赖 monkey patch 或复制整个 ZeRO Optimizer 会高度依赖私有实现，升级成本更高。

因此推荐采用“最小通用扩展点”方案：DeepSpeed 内部只触发生命周期事件，具体算法全部位于 FastOffload。扩展点应提供默认空实现，从而保持现有路径不变。

## 3. 总体分层

```text
Training Application
        │
        ▼
FastOffload Bootstrap/API
        │
        ▼
DeepSpeed Adapter
        │
        ▼
FastOffload Controller
   ├── Policy
   ├── Scheduler
   ├── Transfer Engine
   ├── Optimizer Worker
   ├── State Manager
   └── Telemetry
```

### 3.1 Bootstrap/API

负责读取独立 FastOffload 配置、构造组件、注册扩展并在训练结束时清理资源。

### 3.2 Adapter

唯一了解 DeepSpeed 私有数据结构的模块。ZeRO 内部版本变化原则上只影响 Adapter。

### 3.3 Controller

接收生命周期事件并协调其他组件。Controller 不直接执行 copy，也不直接操作 ZeRO 私有字段。

### 3.4 Policy

根据梯度重要性、层类型、历史状态或系统负载产生处理决策。

### 3.5 Scheduler

将决策转为有依赖关系的 copy、accumulate、update 和 commit 任务，并管理反压与同步边界。

### 3.6 Transfer Engine

使用 pinned memory、可选 producer/dedicated stream 和 event 管理 GPU/CPU 数据传输。

### 3.7 Worker

执行 CPU 梯度累积和 optimizer 更新，可先实现同步 inline worker，再实现独立进程。

### 3.8 State Manager

统一维护 gradient、optimizer、parameter 和 committed version，是异步正确性的核心。

### 3.9 Telemetry

记录关键路径耗时、带宽、队列和策略分布，并为自适应策略提供输入。

## 4. 生命周期事件

建议定义以下通用事件：

```text
BACKWARD_BEGIN
GRADIENT_READY
GRADIENT_REDUCED
GRADIENT_OFFLOAD_SUBMITTED
BACKWARD_END
STEP_BEGIN
STEP_END
PARAMETER_FETCH_BEGIN
PARAMETER_FETCH_END
PARAMETER_RELEASE
CHECKPOINT_BEGIN
CHECKPOINT_END
SHUTDOWN
```

第一阶段只需要支持：

```text
BACKWARD_BEGIN
    → GRADIENT_REDUCED
    → BACKWARD_END
    → STEP_BEGIN
    → STEP_END
```

`GRADIENT_REDUCED` 必须位于梯度归约/分片完成之后、相应 GPU buffer 被覆盖或释放之前。

## 5. 稳定 Context

DeepSpeed 扩展点不应把整个 optimizer 直接暴露给所有模块，而应通过 Adapter 构造最小 Context：

```python
@dataclass(frozen=True)
class GradientContext:
    parameter_id: int
    partition_id: int
    step_id: int
    micro_step_id: int
    gradient_version: int
    numel: int
    dtype: torch.dtype
    device: torch.device
    gradient_shard: torch.Tensor
    process_group: object
```

实际实现时需要评估哪些字段可以冻结、哪些 tensor 只能在事件回调期间有效。长期异步任务应保存稳定 ID、offset 和 buffer handle，而不是保存完整 Parameter。

## 6. Policy 决策

建议统一返回结构化决策：

```python
class OffloadAction(Enum):
    KEEP_GPU = "keep_gpu"
    OFFLOAD_CPU = "offload_cpu"
    ACCUMULATE_CPU = "accumulate_cpu"
    DEFER_UPDATE = "defer_update"

@dataclass(frozen=True)
class OffloadDecision:
    action: OffloadAction
    priority: int = 0
    update_interval: int = 1
    accumulate: bool = False
```

Policy 不得执行设备传输、等待 event 或调用 optimizer。

## 7. 异步数据通路

```text
Reduced GPU Gradient
       │
       ▼
Policy Decision
       │
       ▼
Pinned CPU Buffer Pool
       │  dedicated copy stream
       ▼
Asynchronous D2H Copy
       │  completion event
       ▼
Bounded Worker Queue
       │
       ▼
CPU Gradient Accumulation / Optimizer
       │
       ▼
Version Validation
       │
       ▼
Parameter Commit
```

### 7.1 Buffer Pool

Pinned buffer 必须预分配并复用，避免每个梯度动态 page-lock。Pool 必须有容量限制和明确状态机。

### 7.2 Stream 与 Event

Transfer Engine 默认在 producer stream 直接提交 D2H，利用 stream 顺序保护源 buffer；可选 dedicated accelerator stream 使用稳定 GPU staging 保护源数据。CPU Worker 只能在 copy event 完成后读取目标 buffer，ZeRO 只能在源 buffer 不再被任务引用后复用它。

### 7.3 有界队列

当 pinned buffer 或 worker queue 达到上限时，Scheduler 必须反压、降级为同步路径或调整 Policy，不能无限增长内存。

### 7.4 参数版本

每个 shard 至少维护：

```text
gradient_version → optimizer_version → parameter_version → committed_version
```

旧版本更新不得覆盖新参数，overflow 或取消的版本不得提交。

## 8. DeepSpeed 最小接入方案

推荐新增一个通用 extension 协议和默认空实现，而不是继续增加特性专用条件分支：

```python
class NullOffloadExtension:
    def on_backward_begin(self, context):
        pass

    def on_gradient_reduced(self, context):
        pass

    def on_backward_end(self, context):
        pass

    def on_step_begin(self, context):
        pass

    def on_step_end(self, context):
        pass
```

潜在接入文件：

```text
deepspeed/runtime/engine.py
deepspeed/runtime/zero/stage_1_and_2.py
deepspeed/runtime/zero/stage3.py
```

ZeRO-2 重点位置包括 backward prologue、gradient partition epilogue、CPU gradient copy 和 optimizer step。ZeRO-3 还需要 subgroup update、parameter fetch/release 等位置。

第一轮接入应仅触发 Observer，不改变 tensor、stream、通信或更新行为。

## 9. 独立配置

初期不修改 DeepSpeed Pydantic 配置，FastOffload 使用独立 JSON：

```json
{
  "enabled": true,
  "mode": "observe",
  "zero_stage": 2,
  "policy": {
    "type": "all_offload"
  },
  "scheduler": {
    "type": "synchronous",
    "max_inflight_tasks": 1
  },
  "transfer": {
    "pin_memory": true,
    "buffer_count": 4,
    "buffer_size": 134217728
  },
  "worker": {
    "type": "inline",
    "num_workers": 1
  },
  "telemetry": {
    "log_interval": 10
  }
}
```

目标启动接口：

```python
config = FastOffloadConfig.from_json("fastoffload.json")
handle = install(config)
engine, optimizer, _, _ = deepspeed.initialize(...)
```

实现稳定后再评估是否加入 `zero_optimization.fastoffload`。

## 10. 分阶段实施计划

### Phase 1：Observer

- 建立配置、事件、Context 和 ZeRO-2 Adapter。
- 只采集 gradient-ready、backward、step 和传输量信息。
- 验证损失、梯度、参数和未启用路径完全一致。

### Phase 2：同步 Offload Baseline（已实现）

- 已实现有界 pinned buffer pool、同步 D2H 和 Inline Worker。
- 所有梯度每步更新，不改变 optimizer、overflow、clipping 或 gradient accumulation 语义。
- 已建立原生 ZeRO-2 数值对照和真实传输指标 baseline。

### Phase 3：传输与 Backward 重叠（已实现）

- 已增加 producer/dedicated stream 策略、source-ready/copy-complete event 和 task/byte 双重有界队列。
- 默认 producer-stream direct D2H 避免额外 GPU copy 和并发资源争用；可选 dedicated-stream 策略通过稳定 GPU staging tensor 避免 ZeRO reduction bucket 复用竞争。
- optimizer step 前 flush，记录 queue wait、step flush、source hold 和传输隐藏比例。

### Phase 4：策略化更新

- 实现固定 interval 和 Norm Top-K Policy。
- 重要梯度高频更新，其他梯度在 CPU 累积后低频更新。
- 明确 gradient clipping、overflow 和 scheduler 语义。

### Phase 5：异步 CPU Optimizer

- 增加独立 worker process。
- 实现 CPU core/NUMA binding、错误传播和优雅退出。
- 通过 State Manager 完成版本验证与提交。

### Phase 6：自适应调度

- 根据传输带宽、队列深度和 optimizer 时长调整 top-k ratio、update interval 和并发数。

### Phase 7：ZeRO-3

- 增加 parameter shard fetch/release 和 subgroup 支持。
- 最后再评估 NVMe、Pipeline Parallel 和复杂模型结构。

## 11. 第一版范围

建议第一版严格限定为：

```text
ZeRO Stage 2
CPU optimizer offload
FP16/BF16
AdamW
单个 Inline/CPU Worker
固定 update interval
Norm Top-K Policy
不支持 NVMe
不支持 Pipeline Parallel
不支持 ZeRO Stage 3
```

## 12. 正确性约束

1. Overflow 检测必须覆盖所有应参与当前 step 的梯度。
2. Gradient clipping 必须基于定义明确且一致的 global norm。
3. 所有 rank 必须具有一致的选择与更新语义。
4. 不得破坏 gradient accumulation boundary。
5. 通信完成前不能消费梯度，传输完成前不能复用源 buffer。
6. 参数更新提交前必须确认版本和依赖。
7. unused parameter 不能造成永久等待。
8. checkpoint 前必须排空或可恢复所有异步任务。
9. worker 异常必须传播到训练主进程。
10. shutdown 必须释放 stream、event、queue、worker 和 pinned memory。

## 13. 测试规划

每个阶段至少包含：

- 单元测试：Policy、状态机、buffer pool 和配置校验。
- Adapter 测试：验证 shard、offset、dtype 和 process group。
- 单卡一致性测试：与原生 ZeRO-Offload 比较 loss、grad 和 parameter。
- 多卡一致性测试：验证 rank 间选择与版本一致。
- gradient accumulation、unused parameter、overflow 和 checkpoint 测试。
- 性能测试：backward、D2H、CPU optimizer、queue wait 和 step wall time。

## 14. 模块实现顺序

```text
config/events/context
    → telemetry
    → adapters/base
    → adapters/zero2
    → policies/base/all_offload
    → controller
    → synchronous scheduler
    → transfer buffer pool/engine
    → inline worker
    → state manager
    → overlap scheduler
    → norm_topk policy
    → process worker
    → adaptive policy
    → zero3 adapter
```

该顺序保证每一步都可以独立验证，避免在尚未建立正确性 baseline 时同时引入异步传输、选择性更新和多进程 optimizer。
