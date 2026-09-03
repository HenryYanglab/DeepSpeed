<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- DeepSpeed Team -->

# Adapters

Adapter 是 FastOffload 与 DeepSpeed ZeRO 内部实现之间唯一允许存在的强依赖边界。

## 职责

- 将 ZeRO-2/ZeRO-3 私有数据结构转换成稳定的 FastOffload Context。
- 获取本 rank 的 gradient、parameter 和 optimizer-state shard。
- 提供 process group、partition offset、dtype、device 等元数据。
- 封装必要的同步、参数提交和 buffer 生命周期操作。
- 隔离 DeepSpeed 版本变化，禁止其他模块直接读取 ZeRO 私有字段。

## 不负责

- 不决定梯度是否 offload。
- 不创建 CPU worker。
- 不实现重要性评分算法。
- 不自行改变 optimizer 更新频率。

## 文件

```text
adapters/
├── README.md
├── base.py       # ObserverAdapter 抽象接口
├── zero2.py      # 已实现的 stage_1_and_2.py Observer 适配
└── zero3.py      # 后续阶段规划
```

## ZeRO-2 使用样例

```python
adapter = Zero2ObserverAdapter(optimizer)
step_context = adapter.create_backward_begin_context()
gradient_context = adapter.create_gradient_context(parameter, group_id=0)
```

ZeRO-2 Adapter 封装 `grad_position`、参数分组、process group 和 FP32 gradient partition 等内部结构。生成的 Context 只包含标量与字符串元数据，不持有 Tensor。

## ZeRO-3 使用样例

```python
with zero3_adapter.acquire_parameter_shard(parameter_id) as shard:
    controller.on_parameter_available(shard)
```

ZeRO-3 Adapter 后续还需要处理 subgroup、parameter fetch/release、partition persistence 和 NVMe 状态，但不属于第一阶段实现范围。
