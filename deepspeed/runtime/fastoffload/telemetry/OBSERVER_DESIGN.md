<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- DeepSpeed Team -->

# Observer 与 Telemetry 实现设计

## 1. 目标

Phase 1 只观测 ZeRO-2 CPU optimizer offload，不改变梯度、通信、传输和 optimizer 更新行为。该阶段需要回答：

- 参数梯度何时 ready，何时完成 reduce/partition 的提交；
- 每个 rank 持有多少梯度分片，理论可 offload 多少字节；
- backward、gradient reduction 和 optimizer step 的耗时分布；
- bucket 大小、参数数量以及 CPU/GPU 内存使用情况；
- gradient accumulation 和 unused parameter 对数据通路的影响。

开启 Observer 前后的 loss、gradient、parameter、optimizer state、overflow 和 global grad norm 必须一致。

## 2. 第一阶段范围

```text
ZeRO Stage 2
CPU optimizer offload
FP16/BF16
Adam/AdamW
单卡和数据并行
支持 gradient accumulation
只记录元数据
不计算每个梯度的 norm
不支持 ZeRO-3 和 NVMe
```

默认禁止在热路径调用 `Tensor.item()`、`Tensor.cpu()`、`Tensor.numpy()`、`Tensor.norm()`、accelerator synchronize 或新增 distributed collective。

## 3. 规划文件

```text
fastoffload/
├── __init__.py
├── api.py
├── config.py
├── events.py
├── context.py
├── controller.py
├── adapters/
│   ├── __init__.py
│   ├── base.py
│   └── zero2.py
└── telemetry/
    ├── __init__.py
    ├── metrics.py
    ├── timers.py
    ├── observer.py
    └── reporter.py
```

实现顺序为 Metrics、Config、Events/Context、Observer/Reporter、Adapter Base、ZeRO-2 Adapter、Controller、注册 API，最后才添加 DeepSpeed hook。

## 4. 生命周期事件

```python
class FastOffloadEvent(Enum):
    BACKWARD_BEGIN = "backward_begin"
    GRADIENT_READY = "gradient_ready"
    GRADIENT_REDUCED = "gradient_reduced"
    BACKWARD_END = "backward_end"
    STEP_BEGIN = "step_begin"
    STEP_END = "step_end"
    REPORT = "report"
    SHUTDOWN = "shutdown"
```

典型顺序：

```text
BACKWARD_BEGIN
    ├── GRADIENT_READY × N
    ├── GRADIENT_REDUCED × local_N
BACKWARD_END
STEP_BEGIN
STEP_END
```

使用 gradient accumulation 时，每个 micro step 都有 backward 事件，只有 accumulation boundary 才执行对应 optimizer step。

## 5. Context

基础上下文建议使用不可变 dataclass：

```python
@dataclass(frozen=True)
class StepContext:
    global_step: int
    micro_step: int
    rank: int
    world_size: int
    zero_stage: int
    gradient_accumulation_boundary: bool
    timestamp_ns: int
```

梯度上下文只保存元数据：

```python
@dataclass(frozen=True)
class GradientContext:
    parameter_id: int
    parameter_name: str
    group_id: int
    partition_id: int
    parameter_numel: int
    shard_numel: int
    shard_offset: int
    element_size: int
    shard_bytes: int
    dtype: str
    device: str
    global_step: int
    micro_step: int
    gradient_accumulation_boundary: bool
    is_local_partition: bool
    is_cpu_offload: bool
    timestamp_ns: int
```

Context 不得长期保存 Parameter、Tensor 或 process group，避免延长 buffer 生命周期。异步模块后续应使用稳定 ID、offset 和 buffer handle。

建议另设 bucket 上下文，以较低开销采集 bucket 数量、参数数量、numel、字节数、通信 dtype 和 overlap 状态。

## 6. ZeRO-2 Adapter

`adapters/zero2.py` 是唯一允许访问 `grad_position`、`params_in_partition`、`bit16_groups`、`param_id`、`param_names`、`is_param_in_current_partition` 和 `micro_step_id` 等 ZeRO 私有字段的模块。

目标接口：

```python
class Zero2ObserverAdapter:
    def create_step_context(self) -> StepContext: ...
    def create_gradient_context(self, parameter, group_id) -> GradientContext: ...
    def create_bucket_context(self, communication_dtype, bucket) -> GradientBucketContext: ...
```

参数 ID 到名称的映射必须在初始化时构建一次，热路径不能遍历 `named_parameters()`。

## 7. Observer 与 Controller

Observer 目标接口：

```python
class FastOffloadObserver:
    def on_backward_begin(self, context): ...
    def on_gradient_ready(self, context): ...
    def on_gradient_reduced(self, context): ...
    def on_backward_end(self, context): ...
    def on_step_begin(self, context): ...
    def on_step_end(self, context): ...
    def report(self): ...
    def close(self): ...
```

Controller 只负责通过 Adapter 构造 Context 并分发事件：

```python
class FastOffloadController:
    def on_gradient_reduced(self, parameter, group_id):
        context = self.adapter.create_gradient_context(parameter, group_id)
        self.observer.on_gradient_reduced(context)
```

Observer 不得直接接触 DeepSpeed optimizer。

## 8. ZeRO-2 Hook 位置

### 8.1 Backward 边界

在 ZeRO optimizer 的 `backward_prologue()` 和 `backward_epilogue()` 触发 backward begin/end。默认 extension 必须是空实现。

### 8.2 Gradient Ready

在 `reduce_independent_p_g_buckets_and_remove_grads()` 中获取有效 `grad_reduc` 后触发。该事件只记录参数元数据和 host timestamp，不读取梯度值。

### 8.3 Gradient Reduced

ZeRO-2 路径为：

```text
reduce_ipg_grads()
    → average_tensor()
    → copy_grads_in_partition()
```

对本 rank 持有的 partition，在 `copy_grads_in_partition()` 返回后触发。此时可以通过 `grad_position` 准确描述 shard。

当 `overlap_comm=True` 时，事件表示归约和分片操作已提交并建立 stream 依赖，不表示 CPU 已经可以立即读取。未来 Transfer Engine 仍需通过 stream/event 管理依赖。

### 8.4 Step 边界

在 ZeRO optimizer `step()` 外围触发 step begin/end，只记录耗时，不改变 overflow、gradient clipping、optimizer step 或 parameter all-gather。

## 9. Metrics

Metrics 模块提供：

- Counter：单调累加事件数量和字节数；
- Gauge：保存最新 step、队列深度或内存值；
- RunningStats：以固定内存统计 count、total、minimum、maximum 和 mean；
- MetricsRegistry：线程安全地注册、更新、快照和重置指标。

推荐指标：

```text
backward_count
backward_host_ms
gradient_ready_count
gradient_reduced_count
gradient_local_shard_numel
gradient_local_shard_bytes
gradient_bucket_count
gradient_bucket_bytes
potential_offload_bytes
unused_parameter_count
optimizer_step_host_ms
gpu_memory_allocated
cpu_rss_bytes
observer_callback_ms
observer_dropped_events
```

Phase 1 未执行 offload，因此必须使用 `potential_offload_bytes`，不能将其称为实际 `offload_bytes`。

Metrics 必须进行 O(1) 聚合，不永久保存全部样本。调试事件可使用默认关闭的固定容量 ring buffer。

## 10. Timer

Host timer 使用 `time.perf_counter_ns()`。Device timer 默认关闭，因为读取 accelerator event 耗时可能引入设备同步。

```json
{
  "telemetry": {
    "host_timing": true,
    "device_timing": false
  }
}
```

如后续开启 device timing，应按 bucket 或整个 backward 记录 event，并在报告边界解析，不能为每个参数创建一对 event。

## 11. Reporter

Reporter 按固定 step 间隔读取 Metrics snapshot，不在梯度热路径写日志或文件。默认只输出 rank 0 的本地统计，不新增 collective：

```text
[FastOffload Observer]
step=100
backward_ms=842.13
optimizer_step_ms=311.40
gradient_shards=187
potential_offload=1.82 GiB
observer_overhead_ms=0.31
```

必须明确 rank 0 local shard 不等于全局总量。分布式汇总后续作为显式配置，仅允许在报告边界执行。

## 12. 初始配置

```json
{
  "enabled": true,
  "mode": "observe",
  "zero_stage": 2,
  "telemetry": {
    "log_interval": 10,
    "parameter_events": true,
    "bucket_events": true,
    "host_timing": true,
    "device_timing": false,
    "memory_metrics": true,
    "distributed_summary": false,
    "rank_mode": "rank0_local"
  }
}
```

禁用时应在初始化阶段选择 `NullOffloadExtension`，避免在每个热路径事件中重复执行复杂配置判断。

## 13. 正确性与性能测试

测试包括：

1. Counter、Gauge、RunningStats 和 snapshot/reset 单元测试；
2. Context shard numel、offset、bytes 和无 Tensor 引用测试；
3. 生命周期顺序测试；
4. 原生 ZeRO-2 与 Observer 模式的 loss、gradient、parameter、optimizer state 对比；
5. unused parameter、gradient accumulation、overflow、超大参数和不同通信配置；
6. 单卡与两卡、FP16 与 BF16；
7. 训练 step wall-time 增幅、CPU 内存稳定性及无新增同步/collective 检查。

Phase 1 验收目标：

```text
step wall-time 增幅 < 1%
额外 GPU 显存接近 0
额外 CPU 内存不随 step 增长
不新增 distributed collective
不新增 accelerator synchronize
```

## 14. 最终数据流

```text
ZeRO-2 lifecycle hook
        │
        ▼
Zero2ObserverAdapter
        │  转为不可变元数据
        ▼
FastOffloadController
        │
        ▼
FastOffloadObserver
        ├── MetricsRegistry
        ├── HostTimer
        └── TelemetryReporter
```

Observer 数据将用于确定后续 D2H hook、pinned buffer 大小、最大 inflight task 数量、选择策略和 CPU optimizer 重叠空间。
