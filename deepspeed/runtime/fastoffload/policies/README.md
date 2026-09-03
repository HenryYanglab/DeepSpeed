<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- DeepSpeed Team -->

# Policies

Policy 负责回答“某个梯度或参数应该如何处理”，但不执行数据传输和 optimizer 计算。

## 职责

- 根据 norm、层类型、历史统计或带宽压力计算决策。
- 选择保留在 GPU、offload 到 CPU、CPU 累积或延迟更新。
- 为 Scheduler 提供优先级和建议更新间隔。
- 在需要时维护轻量的策略统计状态。

## 当前实现

`all_offload.py` 已实现 Phase 2 baseline policy，对每个本 rank reduced gradient shard 返回 `OFFLOAD_CPU`。它不读取梯度值，也不执行传输。

## 规划文件

```text
policies/
├── README.md
├── base.py          # OffloadPolicy 接口
├── all_offload.py   # 所有梯度 offload，用作正确性 baseline（已实现）
├── norm_topk.py     # 按梯度 norm/列重要性选择
├── interval.py      # 固定更新间隔
└── adaptive.py      # 根据运行时指标动态调节
```

## 决策模型样例

```python
class OffloadAction(Enum):
    KEEP_GPU = "keep_gpu"
    OFFLOAD_CPU = "offload_cpu"
    ACCUMULATE_CPU = "accumulate_cpu"
    DEFER_UPDATE = "defer_update"

@dataclass
class OffloadDecision:
    action: OffloadAction
    priority: int = 0
    update_interval: int = 1
```

## Policy 样例

```python
class NormTopKPolicy:
    def decide(self, context):
        score = context.gradient_norm
        if score >= self.threshold:
            return OffloadDecision(action=OffloadAction.KEEP_GPU, priority=10)
        return OffloadDecision(
            action=OffloadAction.ACCUMULATE_CPU,
            update_interval=4,
        )
```

以上代码仅用于说明接口。实际实现需要保证不同 rank 的选择语义一致，并明确评分发生在归约前还是归约后。

## 扩展方式

新增策略时只增加一个 Policy 文件，通过配置注册：

```json
{
  "policy": {
    "type": "norm_topk",
    "topk_ratio": 0.1,
    "selection_interval": 4
  }
}
```

Policy 不得直接访问 CUDA stream、CPU queue 或 DeepSpeed optimizer 私有字段。
