<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- DeepSpeed Team -->

# FastOffload

FastOffload 是面向 DeepSpeed ZeRO-Offload 的实验性扩展框架，目标是在 forward/backward、梯度归约、GPU/CPU 传输和 optimizer step 过程中提供低耦合、可组合的控制能力。

当前已完成 Phase 1 Observer、Phase 2 同步 Offload Baseline 和 Phase 3 event 驱动异步传输：异步模式默认在 producer stream 直接提交 D2H，也可切换到独立 copy stream + GPU staging 做 overlap 消融；两种策略都使用 accelerator event、有界 inflight 队列并在 optimizer step 前 flush，所有梯度仍在每个 step 更新。

## 设计原则

1. DeepSpeed 只暴露少量通用生命周期扩展点，不感知具体 FastOffload 算法。
2. FastOffload 核心模块不直接访问 DeepSpeed 私有字段。
3. 只有 `adapters/` 可以了解 ZeRO-2、ZeRO-3 的内部数据结构。
4. Policy 负责决策，Scheduler 负责任务时序，Transfer 负责数据移动，Worker 负责计算。
5. 优先完成 ZeRO Stage 2，再扩展到 ZeRO Stage 3。
6. 所有异步更新都必须经过版本检查和显式提交。

## 文档

- [分类文档索引](docs/README.md)

论文、图片、工作簿及原始实验输出不随本次代码与文档更新发布。

- [FastOffload方法与实现详解（中文；A/B/C累积位置、dtype与完整生命周期）](docs/design/FASTOFFLOAD_METHODOLOGY_AND_IMPLEMENTATION_ZH.md)
- [完整消融实验方案（中文；排除CPU累积与CPU归约）](docs/experiments/FASTOFFLOAD_ABLATION_EXPERIMENT_PLAN_ZH.md)
- [当前实现状态与功能清单](docs/history/IMPLEMENTATION_STATUS.md)
- [总体架构设计](docs/design/DESIGN.md)
- [Hybrid sparse/dense update 设计](docs/design/HYBRID_UPDATE_DESIGN.md)
- [论文与当前实现差距分析](docs/design/PAPER_IMPLEMENTATION_GAP_ANALYSIS.md)
- [Overview 图绘制需求简述](docs/paper/OVERVIEW_FIGURE_BRIEF.md)
- [Overview 图详细提示词与规范](docs/paper/OVERVIEW_FIGURE_PROMPT.md)
- [实验软硬件环境与版本](docs/configuration/EXPERIMENT_ENVIRONMENT.md)
- [Observer 配置](docs/configuration/CONFIGURATION.md)
- [DeepSpeed 适配层](adapters/README.md)
- [策略模块](policies/README.md)
- [重要性选择配置](docs/configuration/CONFIGURATION.md#column-importance-selection)
- [调度模块](schedulers/README.md)
- [传输模块](transfer/README.md)
- [CPU Worker 模块](workers/README.md)
- [状态管理模块](state/README.md)
- [遥测模块](telemetry/README.md)

## 规划结构

```text
fastoffload/
├── README.md
├── docs/                   # 设计、配置、实验、论文与历史文档
├── api.py                  # Observer 安装、卸载和 Null Controller
├── config.py               # 独立且不可变的配置模型
├── controller.py           # 生命周期协调与失败策略
├── context.py              # 不持有 Tensor 的稳定元数据
├── events.py               # 生命周期事件
├── actions.py              # Policy 决策类型
├── adapters/               # ZeRO-2 私有结构隔离层
├── policies/               # 梯度和参数处理策略
├── importance/             # 可插拔、逐层的两级列重要性选择
├── hybrid/                 # 双 buffer、异步 CPU job 和 SelectedAdam 核心
├── schedulers/             # 任务排序、反压和同步边界
├── transfer/               # GPU/CPU 异步传输
├── workers/                # CPU optimizer 执行单元
├── state/                  # 参数、梯度和 optimizer 版本管理
└── telemetry/              # 指标、计时和诊断
```

## Observer 使用方式

```python
from deepspeed.runtime.fastoffload import FastOffloadConfig, install

config = FastOffloadConfig.from_json("fastoffload.json")
handle = install(config)

engine, optimizer, _, _ = deepspeed.initialize(...)

# 训练结束后输出最后一个指标窗口并注销 Observer。
handle.close()
```

## 推荐开发顺序

1. Observer 和 telemetry，只观测、不改变训练行为。（已完成）
2. ZeRO-2 Adapter 与同步 CPU offload baseline。（已完成）
3. 可选 producer/dedicated stream、event 和异步 D2H copy。（已完成）
4. 固定间隔的梯度累积与更新。
5. 预训练参数 delta Top-K 两级列选择和带性能阈值的 SelectiveLinear backward。（已完成实验实现；ZeRO 传输和 optimizer 仍为 dense）
6. 独立 CPU optimizer worker 和版本提交。
7. 自适应调度。
8. ZeRO-3 参数 fetch/release 支持。
