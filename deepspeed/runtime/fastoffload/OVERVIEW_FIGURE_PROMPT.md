<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- DeepSpeed Team -->

# FastOffload Overview 图绘制提示词与规范

本文档用于绘制论文或技术报告中的 FastOffload Overview 图。目标是通过一张图清楚说明系统输入输出、A/B/C 参数划分、Ordinary 与 Dense Boundary 的不同执行路径、ZeRO-2 bucket streaming、GPU/CPU 分工、异步 CPU 优化和版本化发布。

本图只表达训练语义和核心系统数据流，不包含 Telemetry、benchmark 数值以及低层工程实现细节。

## 1. 图的核心叙事

读者应能在不阅读正文的情况下，从图中回答以下问题：

1. FastOffload 接收什么输入，产生什么输出？
2. 二维权重如何按照重要性划分为 A、B、C 三组？
3. Ordinary step 和 Dense Boundary 分别计算哪些梯度？
4. Native ZeRO-2 gradient-ready bucket 在系统中承担什么作用？
5. A、B、C 分别在哪里累积和更新？
6. GPU 到 CPU 传输哪些数据，使用什么精度？
7. CPUAdam 如何与后续 GPU 训练重叠？
8. 异步更新如何安全地发布回模型参数？
9. 为什么该路径不会保留全模型 dense gradient，也不需要完整 GPU staging buffer？

图中需要传达的中心结论是：

> FastOffload 在 Ordinary step 中只计算和通信重要的 A+B 列；在 Dense Boundary 中仍计算数学上完整的 A+B+C 梯度，但复用 Native ZeRO-2 的逐层 bucket 生命周期，将 owner-local fragment 立即拆分，并把 C 直接异步写入 CPU buffer，从而避免全模型 dense gradient 同时驻留 GPU。

## 2. 推荐画布和整体布局

推荐使用横向画布，比例为 16:9 或论文双栏宽图：

- 推荐尺寸：`1920 × 1080`，或等比例 SVG/PDF；
- 阅读方向：从左到右；
- 主体分为三个区域：
  - 左侧 22%：输入、重要性选择和 A/B/C 划分；
  - 中间 58%：GPU/CPU Hybrid Update Pipeline；
  - 右侧 20%：Loss 与 Updated Model 输出；
- 中间区域按上下分层：上半部分为 GPU，下半部分为 CPU；
- GPU 与 CPU 之间明确画出 PCIe 边界；
- 不要把图画成代码模块依赖图，应画成训练数据流与执行时序图。

推荐标题：

```text
FastOffload: Importance-Guided Hybrid GPU–CPU Training on ZeRO-2
```

也可以使用更短标题：

```text
FastOffload System Overview
```

## 3. 系统输入

图的最左侧放置三个主要输入框。

### 3.1 Training Batch

框内文字：

```text
Training Batch
Tokens x, Labels y, Attention Mask
```

该输入连接到 GPU Forward。

### 3.2 Current Model

框内文字：

```text
Current Model
Parameters θᵥ
```

该输入连接到 GPU Forward，同时表示当前安全发布的参数版本。

### 3.3 Importance Information

建议画成一个小型两阶段流程：

```text
Pretrained Parameters θ₀
          +
Warmup Parameters θw
          │
          ▼
Column Importance Ranking
|θw − θ₀|
```

Importance Ranking 的输出连接到 A/B/C Partition。若版面有限，可以合并为：

```text
Column Importance
Pretrained-to-Warmup Δ
```

### 3.4 可选控制输入

FastOffload Configuration 不是核心数据输入，可以画成较小的控制框，并使用细虚线连接到 Importance Partition 和 Step Scheduler：

```text
FastOffload Configuration
A/B Ratios · Update Interval
```

不要展开 JSON 字段、路径、bucket 字节数或 CPU affinity 参数。

## 4. A/B/C 参数划分

在左侧输入之后放置三色权重矩阵。矩阵按列划分，不要按元素随机着色。

框内标题：

```text
Importance-Guided Column Partition
```

三组列的标签：

```text
A: First-Important Columns
Top-K · GPU update every successful step

B: Second-Important Columns
Next-K · GPU accumulation, CPU boundary update

C: Remaining Columns
Dense-boundary gradient and CPU update
```

如果需要展示典型比例，可使用：

```text
A: 10%    B: 10%    C: 80%
```

但需标注为示例：

```text
Example ratio
```

避免让读者误以为比例不可配置。

一维 Bias/Norm 参数不适合按列划分。在矩阵下方单独放置：

```text
Bias / Norm / 1D Parameters
Dense boundary update
```

## 5. Step Scheduler

在 A/B/C Partition 下方画一条简洁的 step 时间轴：

```text
Step             1       2       3       4
Type             O       O       O       D
A update         ✓       ✓       ✓       ✓
B accumulation   +       +       +       CPU update
C update                                 CPU update
```

其中：

```text
O = Ordinary selective step
D = Dense boundary
```

时间轴应突出：

- A 每个成功 step 都更新；
- B 在 Ordinary step 中持续累积；
- B/C 在 interval boundary 提交异步 CPU update；
- overflow step 不推进 successful-step scheduler，但 Overview 中不需要展开 overflow rollback 流程。

## 6. GPU 主流程

中间上半部分使用浅绿色或浅灰绿色背景，标题为：

```text
GPU Runtime
```

GPU 主流程从上到下或从左到右依次包含以下模块。

### 6.1 Forward

```text
Forward
Model θᵥ + Training Batch
```

Forward 分出一条箭头到右侧 Training Loss：

```text
Training Loss L(θᵥ; x, y)
```

Loss 同时连接回 Backward，形成训练依赖。

### 6.2 Ordinary / Dense Backward

使用一个有两条路径的框：

```text
Backward

Ordinary Step:
Packed A+B gradients only

Dense Boundary:
Full A+B+C gradients
```

必须明确标注：

```text
Dense boundary preserves full-gradient semantics
```

不能把 Dense Boundary 画成只计算 C。Boundary 中 A、B、C 都要重新计算完整梯度。

Ordinary 路径旁边可增加一条简短说明：

```text
No dense zero-filled grad_weight
```

### 6.3 Native ZeRO-2 Gradient-Ready Buckets

Backward 后连接：

```text
Native ZeRO-2
Gradient-Ready Buckets
```

框下方可加小字：

```text
Layer/bucket streaming lifecycle
```

该模块是 dense boundary 显存优化的关键，应比普通中间框更突出。

### 6.4 NCCL Reduce-Scatter

接下来画紫色实线箭头：

```text
BF16 NCCL Reduce-Scatter
Deterministic Bucket Order
```

箭头终点为：

```text
Owner-Local Gradient Fragment
```

需要表达 owner-sharded 语义，但不必画出每个 rank 的完整拓扑。如果希望展示双 GPU，可在框旁画两个小 GPU 图标：

```text
Rank 0 Owner      Rank 1 Owner
```

### 6.5 Owner-Local A/B/C Split

这是 Overview 的核心节点，建议使用加粗边框：

```text
Owner-Local Fragment Split
A / B / C
```

从该节点分出三条颜色对应的路径。

#### A 路径

红色实线：

```text
A Owner Gradient
      │
      ▼
Unscale / Clip
      │
      ▼
GPU Adam
      │
      ▼
Publish A Every Successful Step
```

A 在 GPU 更新，需要保留 owner-local A gradient，直到 global overflow、norm 和 clip scale 已知。

#### B 路径

橙色路径：

Ordinary step：

```text
B Owner Gradient
      │
      ▼
GPU B Accumulator
```

Dense Boundary：

```text
Frozen B Accumulation
      │
      ▼
Asynchronous BF16 D2H
```

B 的 reduction 语义可以在小字中表示：

```text
Mean over successful interval steps
```

不要在主图中展开求和公式，公式适合放在正文。

#### C 路径

蓝色虚线或蓝色粗箭头：

```text
C Owner Gradient
      │
      ▼
Direct BF16 D2H to CPU Offset
```

在箭头旁必须标注：

```text
No full-model GPU gradient retention
No full-size GPU staging buffer
```

C 在 Ordinary step 中不存在，因此可以从 Dense Boundary 路径开始连接，并在路径上标记：

```text
Dense Boundary Only
```

### 6.6 Dense Fragment Release

从 Owner-Local Fragment Split 画一条灰色短箭头到释放符号：

```text
Release Dense Fragment
Immediately After D2H Submission
```

更准确的生命周期说明：

```text
Full layer gradient
→ Bucket reduce-scatter
→ Owner-local split
→ Async C offload
→ Fragment release
```

该说明可放在 GPU 区域底部，作为一条高亮横向注释。

## 7. GPU 与 CPU 之间的 PCIe 数据流

GPU 和 CPU 区域之间画一条明显的横向边界：

```text
PCIe / Host Interface
```

从 GPU B/C 路径到 CPU buffer 使用蓝色粗虚线箭头，标签为：

```text
Asynchronous BF16 D2H
B + C Owner Gradients
```

需要区分通信类型：

- NCCL reduce-scatter：GPU 间通信，使用紫色；
- BF16 D2H：GPU 到 CPU 的 PCIe 传输，使用蓝色；
- 不要把 NCCL 和 PCIe 画成同一种箭头。

Overview 中不写具体 PCIe 利用率，因为当前系统图描述机制而不是实验结果。

## 8. CPU 主流程

中间下半部分使用浅蓝色背景，标题为：

```text
CPU Runtime
```

### 8.1 Shared and Pinned Gradient Buffer

GPU B/C D2H 箭头连接到：

```text
Shared + Pinned BF16 Gradient Buffer
Precomputed Selected-Flat Offsets
```

需表达：

- 使用单份 shared CPU allocation；
- 该 allocation 被 host-register 为 pinned；
- C 直接写入对应 flat offset；
- 不创建额外完整 CPU staging copy。

但 Overview 中不需要出现 `cudaHostRegister()` API 名称。

### 8.2 Independent Optimizer Process

Buffer 连接到：

```text
Independent CPU Optimizer Process
```

框内画出：

```text
BF16 Gradient
      │
      ▼
FP32 Workspace
      │
      ▼
DeepSpeedCPUAdam
```

框旁标注：

```text
Overlaps Subsequent GPU Training
```

使用一条长虚线箭头从 CPUAdam 指向下一次 GPU Forward/Backward，表示 CPU update 与后续 GPU step 并行执行，而不是依赖箭头。

### 8.3 CPU Optimizer State

在 CPUAdam 旁边放置持久状态框：

```text
Canonical CPU State
FP32 Master Parameters
Adam Moments m, v
Optimizer Step
```

CPUAdam 从该状态读写。

### 8.4 Bounded Queue

如果版面允许，在 Optimizer Process 前画两个小 job slot：

```text
Version v+1     Version v+2
```

并标注：

```text
Bounded Asynchronous Lag
```

不需要展示线程池、Python Queue 或 multiprocessing API。

## 9. Versioned Publication

CPUAdam 输出连接到：

```text
Completed CPU Update
Version v+1
```

随后连接：

```text
Safe Forward Boundary
Versioned Publication
```

最后连接右侧模型输出：

```text
Updated Model
Parameters θᵥ₊₁
```

需要通过箭头表达：

1. CPUAdam 完成不等于立即修改正在执行 Forward 的参数；
2. publication 发生在安全的 forward boundary；
3. 只有完整版本才对训练可见；
4. A 可以每步发布，B/C 通过异步版本发布。

从 Updated Model 画一条回环箭头到下一次 Forward：

```text
Next Training Iteration
```

## 10. 系统输出

图的右侧只保留两个主要输出。

### 10.1 Training Loss

```text
Training Loss
L(θᵥ; x, y)
```

由 Forward 输出，并作为 Backward 输入。

### 10.2 Updated Model

```text
Updated Model
Parameters θᵥ₊₁
```

由 A 的同步 GPU publication 和 B/C 的 versioned CPU publication共同形成。

Overview 中不画 Telemetry。除非论文正文专门讨论 checkpoint，否则也不画 Checkpoint。

## 11. 颜色、箭头和图例

建议固定以下视觉编码：

| 元素 | 推荐颜色 | 含义 |
|---|---|---|
| A / First-important | 红色或深粉色 | 高频 GPU 更新 |
| B / Second-important | 橙色 | GPU 累积、CPU boundary 更新 |
| C / Dense remainder | 蓝色 | Dense boundary CPU 更新 |
| GPU Runtime 背景 | 浅绿色 | GPU 计算和状态 |
| CPU Runtime 背景 | 浅蓝色 | CPU buffer、CPUAdam 和状态 |
| NCCL 通信 | 紫色 | GPU 间 collective |
| PCIe D2H | 蓝色粗箭头 | GPU 到 CPU 传输 |
| 同步依赖 | 实线箭头 | 当前操作必须等待前驱 |
| 异步操作/重叠 | 虚线箭头 | 可与后续 GPU 工作重叠 |
| 释放 | 灰色 | tensor 生命周期结束 |
| Native fallback | 灰色细线 | FastOffload 禁用时的原生路径 |

图例建议放在右下角，但不要占据系统输出区域：

```text
Solid Arrow: Synchronous Dependency
Dashed Arrow: Asynchronous Execution
Purple: Inter-GPU NCCL
Blue: GPU-to-CPU PCIe Transfer
```

## 12. 建议放入图中的关键短语

以下短语建议直接出现在图内：

```text
Importance-Guided Column Partition
Packed A+B Gradient
Full A+B+C Gradient
Native ZeRO-2 Gradient-Ready Bucket
Deterministic BF16 Reduce-Scatter
Owner-Local Fragment Split
Immediate Dense-Fragment Release
Asynchronous BF16 D2H
No Full-Size GPU Staging Buffer
Independent CPU Optimizer Process
FP32 Master + Adam Moments
Bounded Asynchronous Lag
Versioned Publication
Safe Forward Boundary
```

## 13. 不应放入 Overview 的内容

为避免图过于复杂，不要绘制以下内容：

- Telemetry、JSONL、CSV 或 metrics reporter；
- benchmark 数字、steps/s、tokens/s 或显存结果；
- Python 类名和源码文件名；
- Adapter、Controller、Policy 等软件包依赖关系；
- CUDA Event pool、buffer pool 和 pin-memory tracker；
- `cudaHostRegister()`、IPC queue 等具体 API；
- CPU affinity 和 reserved-core 参数；
- checkpoint 字段和序列化格式；
- 每个参数的 owner offset 数值；
- 每个 NCCL bucket 的字节级布局；
- 逐元素 mask 或索引实现；
- Qwen、Alpaca 等特定模型和数据集名称；
- PCIe 利用率估计值；
- Native 与 Takeover 的性能柱状图。

性能对比应放在单独的 Evaluation 图中，而不是 Overview 中。

## 14. 推荐的最终线框图

```text
INPUTS                     GPU RUNTIME                         OUTPUTS

Training Batch ───────▶ Forward ───────────────────────────▶ Training Loss
x, y, mask                 │                                  L(θᵥ; x, y)
                           ▼
Current Model θᵥ ─────▶ Ordinary / Dense Backward
                           │
Importance                 │ Ordinary: packed A+B
θ₀ + θw                    │ Boundary: full A+B+C
    │                      ▼
    ▼              Native ZeRO-2 Gradient-Ready Buckets
Column Ranking             │
    │                      ▼
    ▼              BF16 NCCL Reduce-Scatter
A / B / C                  │
Partition                  ▼
                   Owner-Local Fragment Split
                    │          │          │
                    │ A        │ B        │ C
                    ▼          ▼          ▼
                 GPU Adam   GPU B Acc.  Async BF16 D2H
                    │          │          │
                    │          └────┬─────┘
                    │               ▼
                    │    ─────── PCIe Boundary ───────
                    │               ▼
                    │    Shared + Pinned BF16 Buffer
                    │               │
                    │               ▼
                    │    Independent CPU Optimizer
                    │    BF16 → FP32 → CPUAdam
                    │               │
                    │               ▼
                    │    FP32 Master + Adam Moments
                    │               │
                    └───────────────┴────▶ Versioned Publication
                                             │
                                             ▼
                                    Updated Model θᵥ₊₁
                                             │
                                             └────▶ Next Forward
```

在 Owner-Local Fragment Split 下方增加一行高亮说明：

```text
Full layer gradient → bucket reduce-scatter → asynchronous CPU offload → immediate GPU release
```

## 15. 可直接交给绘图模型的中文提示词

```text
请绘制一张横向、论文风格、矢量化的深度学习系统架构图，标题为“FastOffload System Overview”。画面比例为16:9，白色背景，简洁、专业、适合计算机系统或机器学习论文，使用统一字体、圆角矩形、清晰箭头和平面化配色，不使用3D效果。

整张图从左到右分为输入、GPU/CPU混合训练流水线和输出三个区域。左侧包含三个输入：Training Batch，标注tokens x、labels y和attention mask；Current Model，标注Parameters θ_v；Importance Information，展示Pretrained Parameters θ_0与Warmup Parameters θ_w通过pretrained-to-warmup parameter delta生成Column Importance Ranking。

在左侧中部绘制一个按列着色的二维权重矩阵，标题为Importance-Guided Column Partition。将矩阵列划分为三组：红色A表示First-Important Top-K Columns，橙色B表示Second-Important Next-K Columns，蓝色C表示Remaining Columns。标注A在每个successful step上进行GPU update，B在GPU上累积并在boundary进行CPU update，C只在dense boundary计算和进行CPU update。在矩阵下方单独标注Bias、Norm和1D Parameters在dense boundary更新。增加一个简短时间轴：Step 1、2、3为Ordinary，Step 4为Dense Boundary；A每步更新，B前三步累积并在第4步更新，C在第4步更新。

中间上半部分绘制浅绿色GPU Runtime区域。依次包含Forward、Backward、Native ZeRO-2 Gradient-Ready Buckets、BF16 NCCL Reduce-Scatter和Owner-Local Fragment Split。Forward接收Training Batch和Current Model，并向右输出Training Loss L(θ_v; x,y)。Backward框内明确区分两条语义：Ordinary Step只产生packed A+B gradients，并注明No dense zero-filled grad_weight；Dense Boundary产生Full A+B+C gradients，并明确注明Full-gradient semantics preserved。

Backward连接Native ZeRO-2 Gradient-Ready Buckets，强调layer/bucket streaming lifecycle。随后用紫色实线箭头表示Deterministic BF16 NCCL Reduce-Scatter，输出Owner-Local Gradient Fragment。将Owner-Local Fragment Split画成核心加粗节点，从中分出红色A、橙色B和蓝色C三条路径。

红色A路径为A Owner Gradient、Unscale/Clip、GPU Adam、Publish A Every Successful Step。橙色B路径为B Owner Gradient进入GPU B Accumulator，并在dense boundary冻结后执行Asynchronous BF16 D2H。蓝色C路径只在dense boundary出现，标注Dense Boundary Only，并从C Owner Gradient直接执行Direct BF16 D2H to CPU Offset。在C路径附近突出标注No full-model GPU gradient retention和No full-size GPU staging buffer。从Owner-Local Fragment Split画一条灰色箭头到Release Dense Fragment，标注Immediately After D2H Submission。

中间上下区域之间绘制清晰的PCIe / Host Interface边界。GPU到CPU的B和C传输使用蓝色粗虚线箭头，标注Asynchronous BF16 D2H和B+C Owner Gradients。紫色只表示GPU间NCCL通信，蓝色只表示GPU到CPU的PCIe传输。

中间下半部分绘制浅蓝色CPU Runtime区域。首先是Shared + Pinned BF16 Gradient Buffer，标注Precomputed Selected-Flat Offsets。之后连接Independent CPU Optimizer Process，内部流程为BF16 Gradient到FP32 Workspace，再到DeepSpeedCPUAdam。旁边放置Canonical CPU State，包含FP32 Master Parameters、Adam Moments m和v、Optimizer Step。使用虚线箭头表示CPU optimizer与后续GPU Forward/Backward重叠，标注Overlaps Subsequent GPU Training。可绘制两个小型Version Job Slot表示Bounded Asynchronous Lag，但不要绘制具体queue API。

CPUAdam完成后连接Completed CPU Update Version v+1，再连接Safe Forward Boundary和Versioned Publication。Versioned Publication输出到右侧Updated Model Parameters θ_(v+1)，并从Updated Model画回环箭头连接下一次Forward。强调CPU完成更新后不会在任意时刻修改正在使用的模型参数，而是在安全forward boundary发布完整版本。

右侧只保留两个核心输出：Training Loss L(θ_v; x,y)和Updated Model Parameters θ_(v+1)。不要绘制Telemetry、benchmark数字、checkpoint、源码类名、配置文件路径、CUDA Event Pool、CPU affinity、pin-memory tracker或PCIe利用率。

颜色规范：A使用红色或深粉色，B使用橙色，C使用蓝色；GPU区域使用浅绿色背景，CPU区域使用浅蓝色背景；NCCL通信使用紫色实线箭头；PCIe D2H使用蓝色粗虚线箭头；同步依赖使用实线，异步执行使用虚线，tensor释放使用灰色。图的底部加入一句核心生命周期说明：Full layer gradient → bucket reduce-scatter → asynchronous CPU offload → immediate GPU release。
```

## 16. 可直接交给绘图模型的英文提示词

```text
Create a clean, publication-quality vector system diagram titled “FastOffload System Overview” in a horizontal 16:9 layout. Use a white background, flat colors, consistent typography, rounded rectangles, clear arrows, and no 3D effects. The figure must explain an importance-guided hybrid GPU–CPU training system built on DeepSpeed ZeRO-2.

Organize the figure from left to right into Inputs, a GPU/CPU Hybrid Training Pipeline, and Outputs. On the left, show three inputs: Training Batch with tokens x, labels y, and attention mask; Current Model with parameters theta_v; and Importance Information, where pretrained parameters theta_0 and warmup parameters theta_w produce a pretrained-to-warmup column importance ranking.

Show a two-dimensional weight matrix partitioned by columns under the title “Importance-Guided Column Partition.” Color the first-important Top-K columns red and label them A; color the second-important Next-K columns orange and label them B; color the remaining columns blue and label them C. State that A is updated on the GPU every successful step, B is accumulated on the GPU and updated on the CPU at an interval boundary, and C is computed and updated on the CPU only at a dense boundary. Show bias, normalization, and one-dimensional parameters separately as dense-boundary parameters. Add a compact timeline with Ordinary steps 1, 2, and 3 followed by Dense Boundary step 4. Show A updated at every step, B accumulated during ordinary steps and updated at step 4, and C updated at step 4.

Use a light-green container titled “GPU Runtime” in the upper center. Show Forward, Backward, Native ZeRO-2 Gradient-Ready Buckets, BF16 NCCL Reduce-Scatter, and Owner-Local Fragment Split. Forward consumes the training batch and current model and emits Training Loss L(theta_v; x,y). In the Backward block, explicitly distinguish “Ordinary Step: Packed A+B Gradients Only” from “Dense Boundary: Full A+B+C Gradients.” Add “No dense zero-filled grad_weight” beside the ordinary path and “Full-gradient semantics preserved” beside the dense-boundary path.

Connect Backward to “Native ZeRO-2 Gradient-Ready Buckets” and label it as a layer/bucket streaming lifecycle. Use a purple solid arrow for “Deterministic BF16 NCCL Reduce-Scatter,” producing an “Owner-Local Gradient Fragment.” Make “Owner-Local Fragment Split” a visually emphasized central node and branch it into red A, orange B, and blue C paths.

The red A path should show A Owner Gradient, Unscale/Clip, GPU Adam, and Publish A Every Successful Step. The orange B path should show B Owner Gradient entering a GPU B Accumulator and, at a dense boundary, a Frozen B Accumulation submitted through Asynchronous BF16 D2H. The blue C path should appear only for dense boundaries and show C Owner Gradient going directly through BF16 D2H into its CPU buffer offset. Place the labels “No full-model GPU gradient retention” and “No full-size GPU staging buffer” beside the C path. From the owner-local split, draw a gray arrow to “Release Dense Fragment Immediately After D2H Submission.”

Separate the GPU and CPU areas with a clearly labeled “PCIe / Host Interface” boundary. Use thick blue dashed arrows for asynchronous BF16 GPU-to-CPU transfers of B and C owner gradients. Keep purple exclusively for inter-GPU NCCL communication and blue exclusively for GPU-to-CPU PCIe transfers.

Use a light-blue container titled “CPU Runtime” in the lower center. First show a “Shared + Pinned BF16 Gradient Buffer” with “Precomputed Selected-Flat Offsets.” Connect it to an “Independent CPU Optimizer Process” containing the flow BF16 Gradient to FP32 Workspace to DeepSpeedCPUAdam. Beside it, show “Canonical CPU State” containing FP32 Master Parameters, Adam Moments m and v, and Optimizer Step. Use a long dashed arrow to indicate that the CPU optimizer overlaps subsequent GPU forward and backward execution. Optionally show two compact version job slots to represent bounded asynchronous lag, without showing implementation-level queue APIs.

After CPUAdam, show “Completed CPU Update, Version v+1,” followed by “Safe Forward Boundary” and “Versioned Publication.” The publication output goes to “Updated Model Parameters theta_(v+1)” on the right. Draw a feedback arrow from the updated model to the next Forward operation. Make clear that a completed CPU update is exposed only as a complete version at a safe forward boundary.

The right side must contain only two primary outputs: “Training Loss L(theta_v; x,y)” and “Updated Model Parameters theta_(v+1).” Do not include telemetry, benchmark values, checkpoints, source-code class names, configuration paths, CUDA event pools, CPU affinity, pin-memory trackers, or PCIe utilization numbers.

Use red or dark pink for A, orange for B, and blue for C. Use a light-green GPU background, a light-blue CPU background, purple solid arrows for NCCL communication, thick blue dashed arrows for asynchronous PCIe D2H, solid arrows for synchronous dependencies, dashed arrows for asynchronous overlap, and gray for released tensors. Add the following highlighted lifecycle statement along the bottom: “Full layer gradient → bucket reduce-scatter → asynchronous CPU offload → immediate GPU release.”
```

## 17. Negative Prompt

如果绘图工具支持 negative prompt，使用：

```text
Do not create a photorealistic image, 3D rendering, isometric data center, decorative GPU hardware illustration, dense source-code diagram, UML class diagram, telemetry dashboard, benchmark chart, or overly colorful infographic. Do not show C-only dense backward, random element-wise sparsity, full-model GPU gradient retention, full-size GPU staging buffers, immediate unsafe CPU parameter writes, or synchronous CPUAdam on every ordinary step. Avoid tiny unreadable text, crossing arrows, gradients, shadows, excessive icons, and ambiguous communication directions.
```

## 18. 绘制完成后的检查清单

发布前逐项检查：

- [ ] 图中存在明确的 Training Batch 和 Current Model 输入。
- [ ] 图中输出只突出 Training Loss 和 Updated Model。
- [ ] 没有 Telemetry 模块。
- [ ] A/B/C 按列划分，而不是逐元素稀疏。
- [ ] A、B、C 使用稳定且一致的三种颜色。
- [ ] Ordinary step 明确只计算 packed A+B。
- [ ] Dense Boundary 明确计算完整 A+B+C。
- [ ] Bias/Norm/1D 参数在 boundary 更新。
- [ ] Native ZeRO-2 gradient-ready bucket 被明确画出。
- [ ] NCCL reduce-scatter 与 PCIe D2H 使用不同颜色。
- [ ] Owner-local fragment split 位于 reduce-scatter 之后。
- [ ] A 在 GPU 每个成功 step 更新。
- [ ] B 在 GPU 累积，并在 boundary 提交 CPU update。
- [ ] C 只在 dense boundary 出现。
- [ ] B/C 使用异步 BF16 D2H。
- [ ] 图中明确没有完整 GPU staging buffer。
- [ ] 图中明确表达 dense fragment 的及时释放。
- [ ] CPUAdam 位于独立 CPU process 中。
- [ ] CPU 使用 FP32 master parameters 和 Adam moments。
- [ ] CPU update 与后续 GPU 训练使用异步重叠箭头。
- [ ] B/C 通过 safe forward boundary 进行 versioned publication。
- [ ] Updated Model 回连到下一次 Forward。
- [ ] 图中没有 benchmark、Telemetry 或实现 API 细节。
- [ ] 所有字体在论文双栏缩放后仍然可读。
- [ ] 箭头方向无歧义，没有不必要的交叉。
