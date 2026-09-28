<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- DeepSpeed Team -->

# FastOffload 文档索引

本索引覆盖随代码发布的设计、配置、实现说明和实验计划。
历史文档保留编写时的状态与限制，不代表后续实验已完成或当前功能均已验证。

论文源文件、PDF、图片、工作簿、原始实验输出和本地备份不随本次代码更新发布。
完整工作区索引在本地 `LOCAL_INDEX.md` 中保留，该文件不属于公开文档集。

## 设计与实现

- [方法与实现详解：梯度累积位置、dtype与生命周期](design/FASTOFFLOAD_METHODOLOGY_AND_IMPLEMENTATION_ZH.md)
- [总体架构设计](design/DESIGN.md)
- [Hybrid A/B/C更新设计](design/HYBRID_UPDATE_DESIGN.md)
- [历史论文与实现差距分析](design/PAPER_IMPLEMENTATION_GAP_ANALYSIS.md)

## 配置与环境

- [配置说明与CPU B归约路线](configuration/CONFIGURATION.md)
- [实验软硬件环境](configuration/EXPERIMENT_ENVIRONMENT.md)
- [训练脚本与配置模板](../scripts/README.md)

## 实验计划

下列文档是方案或历史执行记录；其中的本地数据路径不保证在其他机器上存在。
计划中的变体不等于生产配置中已经支持的开关，也不构成继续运行旧队列的指令。

- [完整消融实验方案：排除CPU累积与CPU归约](experiments/FASTOFFLOAD_ABLATION_EXPERIMENT_PLAN_ZH.md)
- [五小时最小实验方案](experiments/FIVE_HOUR_MINIMAL_EXPERIMENT_PLAN.md)
- [主性能实验设计](experiments/MAIN_PERFORMANCE_EXPERIMENT_DESIGN.md)
- [模型与数据集矩阵](experiments/PAPER_MODEL_DATASET_MATRIX.md)
- [系统论文实验方案](experiments/SYSTEMS_PAPER_EXPERIMENT_PLAN.md)
- [工作簿后续实验计划](experiments/WORKBOOK_NEXT_EXPERIMENT_PLAN.md)

## 图示设计说明

这些是文字设计说明，不包含论文源文件或图像附件。
历史CPU路线图示不应被用来替代当前GPU累积主路径的描述。

- [CPU累积路线图示说明](paper/CPU_ACCUMULATION_FIGURE_GUIDE_ZH.md)
- [多GPU总览图提示词](paper/MULTI_GPU_OVERVIEW_FIGURE_PROMPT.md)
- [总览图需求](paper/OVERVIEW_FIGURE_BRIEF.md)
- [总览图详细规范](paper/OVERVIEW_FIGURE_PROMPT.md)

## 历史状态

- [实现状态与功能清单](history/IMPLEMENTATION_STATUS.md)
- [大模型调试状态](history/LARGE_MODEL_DEBUG_STATUS.md)

## 模块说明

- [适配层](../adapters/README.md)
- [策略](../policies/README.md)
- [调度](../schedulers/README.md)
- [状态](../state/README.md)
- [遥测](../telemetry/README.md)
- [传输](../transfer/README.md)
- [Worker](../workers/README.md)
