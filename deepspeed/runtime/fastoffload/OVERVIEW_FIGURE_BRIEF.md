<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- DeepSpeed Team -->

# FastOffload Overview 绘图需求简述

本文档用于向绘图设计人员简单说明 FastOffload Overview 图需要表达的内容。图片不需要展示代码和复杂实现，重点是让读者快速看懂输入、处理流程和输出。

## 1. 这张图要说明什么

FastOffload 是一个 GPU 和 CPU 协同训练系统。

它根据参数列的重要性，将二维模型权重分成 A、B、C 三部分：

- **A：最重要的参数列**，每一步都在 GPU 上更新；
- **B：第二重要的参数列**，每一步计算并在 GPU 上累积，定期交给 CPU 更新；
- **C：其余参数列**，只在周期边界计算完整梯度并交给 CPU 更新；
- **Bias、Norm 等一维参数**与 C 一样，在周期边界更新。

这样可以减少普通训练步骤的梯度计算和通信，同时保留周期性的完整梯度更新。

## 2. 输入

图片左侧建议画三个输入：

1. **Training Batch**
   - 输入数据，例如 tokens、labels；

2. **Current Model**
   - 当前版本的模型参数；

3. **Importance Information**
   - 根据预训练参数和 warmup 后参数的变化，判断哪些参数列更重要；
   - 输出 A、B、C 参数划分。

可以用一个按列着色的权重矩阵表示 A/B/C：

- A 使用红色；
- B 使用橙色；
- C 使用蓝色。

## 3. 核心处理流程

图片中间分成上下两个区域：上方是 GPU，下方是 CPU。

### 3.1 GPU 区域

主要流程为：

```text
Forward
→ Backward
→ ZeRO-2 分桶通信
→ 将当前 GPU 负责的梯度拆分为 A/B/C
```

需要表现两种训练步骤：

- **Ordinary Step**：只计算 A+B 梯度；
- **Dense Boundary**：计算完整 A+B+C 梯度。

A/B/C 拆分后：

- A 在 GPU 上执行 Adam 更新；
- B 在 GPU 上累积；
- C 在 Dense Boundary 中异步传输到 CPU；
- 完整梯度分片使用完后立即释放，避免所有层的完整梯度同时占用 GPU 显存。

### 3.2 CPU 区域

主要流程为：

```text
接收 B/C 梯度
→ 转换为 FP32
→ CPUAdam 更新
→ 生成新的参数版本
```

CPU 优化器在独立进程中运行，可以和后续 GPU 训练重叠。

CPU 更新完成后，不会立即修改正在使用的模型，而是在安全的训练边界发布新版本。

## 4. GPU 与 CPU 之间的数据传输

GPU 和 CPU 之间画一条明显的连接箭头：

```text
Asynchronous BF16 D2H
GPU → CPU
```

需要表达：

- B/C 梯度使用 BF16 从 GPU 传到 CPU；
- 传输是异步的；
- 不需要额外的完整 GPU staging buffer；
- CPU 更新可以和下一步 GPU 训练重叠。

GPU 之间的 ZeRO-2 通信可以使用紫色箭头，GPU 到 CPU 的传输使用蓝色箭头，以免混淆。

## 5. 输出

图片右侧只需要两个主要输出：

1. **Training Loss**
   - Forward 产生的训练损失；

2. **Updated Model**
   - GPU 更新 A，CPU 定期更新 B/C；
   - 最终得到安全发布的新版本模型参数。

可以从 Updated Model 画一条回到下一次 Forward 的箭头，表示训练循环。

## 6. 推荐的简单布局

```text
输入                    GPU                         CPU                    输出

Training Batch ──▶ Forward ───────────────────────────────────────────▶ Loss
                     │
Current Model ───────┘
                     ▼
                  Backward
                     │
Importance ──▶ A/B/C 参数划分
                     │
              ZeRO-2 分桶通信
                     │
              GPU 梯度拆分 A/B/C
                 │       │
                 │       └──── B/C 异步传输 ────▶ CPUAdam
                 │                                  │
                 ▼                                  ▼
              A GPU 更新                     B/C 新参数版本
                 │                                  │
                 └────────── 安全版本发布 ──────────┘
                                    │
                                    ▼
                              Updated Model
                                    │
                                    └────▶ 下一次 Forward
```

## 7. 视觉建议

- 整体采用从左到右的横向布局；
- GPU 和 CPU 使用不同的浅色背景区域；
- A、B、C 始终使用相同颜色：红、橙、蓝；
- 实线表示当前步骤必须执行的流程；
- 虚线表示异步传输或并行执行；
- 尽量少写文字，使用短标签和清晰箭头；
- 图片应突出 A/B/C、GPU/CPU 分工和异步更新。

## 8. 不需要画的内容

Overview 图中不需要展示：

- 代码类名或文件名；
- Telemetry；
- benchmark 性能数字；
- 配置文件和底层 API；
- CUDA Event、进程队列等实现细节；
- 复杂的数学公式；
- checkpoint 细节。

## 9. 最重要的一句话

整张图最需要表达的是：

> 普通步骤只处理重要的 A+B 参数；周期边界仍计算完整 A+B+C 梯度，但梯度按 ZeRO-2 bucket 流式拆分和异步传到 CPU，因此可以降低 GPU 计算、通信和边界显存压力，同时让 CPU 更新与后续 GPU 训练重叠。
