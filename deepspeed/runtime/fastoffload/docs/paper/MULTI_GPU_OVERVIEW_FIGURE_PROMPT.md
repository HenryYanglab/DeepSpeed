<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- DeepSpeed Team -->

# FastOffload 多 GPU 系统 Overview：科研绘图提示词

## 1. 使用目的

绘制一张用于计算机系统论文的 FastOffload 技术 overview：扁平化、模块化、具象图标、明确输入输出、清晰的数据流与控制流。重点展示多 GPU 数据并行场景下，梯度在哪里计算、何时跨卡归约、哪些梯度 offload 到 CPU、哪些更新参数加载回 GPU，以及异步计算与版本发布如何连接成训练闭环。

图中表达当前 Hybrid A/B/C 实现，不画理想化的未来系统，不混入 ZenFlow、学习率 scheduler 或实验性能曲线。

最终交付：可编辑 SVG/PDF 和 300–600 DPI PNG。图中文字使用英文；本文中文解释用于指导布局和语义。

## 2. 必须传达的技术事实

1. 输入是训练数据集与预训练模型，输出是微调后的完整模型。
2. 使用 ZeRO-2 数据并行：每个 GPU 保留完整 BF16 模型副本，各 rank 读取不同 minibatch；梯度/优化器权威状态按 owner 分片。不是把网络层分给不同 GPU 的流水线并行。
3. A/B/C 是二维权重的输入列集合，权重形状为 `[out_features, in_features]`，选择沿 `dim=1`。
4. 重要性来自 pretrained-to-warmup movement：`score[j] = sum_i |W_warmup[i,j] - W_pretrained[i,j]|`。A 为第一重要列，B 为第二重要列，C 为其余列。示例比例 A=10%、B=10%、C=80%，不是固定限制。
5. Ordinary step：完整模型 forward，activation/input-gradient backward 仍继续传播；受支持 Linear 的 weight-gradient GEMM 只产生 packed A+B，不构造 dense zero-filled grad_weight。不能把图画成只运行 20% forward 或完全不计算 activation gradients。
6. Dense boundary：本步计算完整 A+B+C weight gradients，非选择参数走相应 dense 路径；不是只计算 C，也不是补算前三步缺失的 C。
7. Ordinary 的 packed A/B 通过 compressed reduce-scatter 跨 rank 归约到 owner；dense boundary 复用 Native ZeRO-2 gradient-ready bucket reduction 生命周期，逐 bucket 归约并取 owner-local fragment。
8. 每个成功步骤：A 在 owner GPU 上进行 FP32 Adam 更新，再通过 compressed all-gather 发布 A 模型值到所有 GPU 副本。
9. B 在 owner GPU 累积。Boundary 包含本步 B 后冻结 interval 累积，并 offload 到 CPU。配置为 mean 时按 interval 成功步骤数取平均。
10. C 仅在 boundary 计算本步梯度，owner-local C fragment 流式写入 CPU BF16 staging。Bias/Norm/1D 及其他 dense-only 参数单独标识，不能全部硬画成矩阵列。
11. B/C gradient D2H 和 updated B/C model-value H2D 均使用 BF16；CPU canonical master 和 Adam moments 为 FP32。Adam moments 不在每步 GPU/CPU 往返。
12. 每个 rank 有对应独立 CPU optimizer process，处理本 rank 的 owner B/C 状态；不是所有 GPU 共用一个全模型 CPUAdam 中央瓶颈。
13. CPU 更新完成后，B/C 返回本 rank GPU，再跨 rank compressed all-gather/scatter 到完整模型副本。必须画出 H2D 和 all-gather 两段，不能只画 CPU 到最终输出。
14. 异步版本按序、跨 rank 协调提交；下一次 forward 等待参数 publication visibility event。CPU 完成不等于参数已经对 forward 可见。
15. 没有额外 full-model GPU gradient staging 或 full-model GPU parameter-return staging。局部临时 buffer/collective workspace 仍存在。释放/复用必须满足 CUDA stream/event 生命周期，不能画成 D2H 读完前不安全复用。
16. 训练结束，drain 所有 pending CPU versions、完成参数发布与 visibility，再导出微调后的模型。

## 3. 构图方案：主架构 + 两条时序展开

建议横向宽图，画布约 2800×1700，允许扩展为 16:10。论文中作为双栏宽图，避免将全部内容挤进普通 16:9 小图。

### 总体空间分配

- 顶部 15%：输入与初始化/重要性选择。
- 中间 60%：主架构，左到右排列 Data Pipeline、Multi-GPU Runtime、Publication / Output；CPU 区域置于 GPU 区域下方。
- 底部 25%：Ordinary / Dense Boundary 两条编号执行路径，以及一条异步重叠时间轴。

主图体现模块和物理归属；底部展开“什么时候执行”。用共享编号连接两者，避免在每张 GPU 卡内重复放大量文字。

### 建议区域标题

```text
Inputs & Initialization
Data Pipeline
Multi-GPU Data-Parallel Runtime (ZeRO-2)
Per-Rank CPU Optimizer Processes
Versioned Publication & Final Export
Execution Schedule
```

## 4. 输入、图标和初始化

### 数据输入

画数据库圆柱或叠放文档 icon：

```text
Training Dataset
Instructions · Inputs · Responses
```

连接：

```text
Tokenize & Collate
Tokens · Labels · Attention Mask
        → Distributed Sampler
        → Rank-local Minibatches
```

用两束不同编号的 sample 卡片表示数据分片：`Batch b₀`、`Batch b₁`。经细灰色 H2D 箭头进入对应 GPU。

这条箭头标注：

```text
Batch H2D · Every microbatch
```

与 B/C 模型参数 H2D 返回区分，不能只标一个泛化的 “Load Data”。不要暗示所有 rank 读取相同 batch。

### 模型输入

画模型文件/网络节点 icon：

```text
Pretrained Model W₀
```

分成两条路径：

- `Initial model load / replicate` → 每个 GPU 的完整 BF16 model replica；仅初始化。
- `Reference W₀ + Warmup Ww` → `Column Importance Selector`。

重要性选择模块用柱状排序 icon 和三色矩阵 icon：

```text
Pretrained-to-Warmup Δ
score[j] = Σᵢ |Ww[i,j] − W₀[i,j]|
```

输出：

```text
Shared A/B/C Column Plan
A: Top-K · B: Next-K · C: Rest
```

用细点线控制箭头把计划发给所有 GPU 的 selective backward 和 owner mapping。列计划相同不等于 owner 相同。

矩阵绘制约 16–20 列，A/B/C 沿竖向列着色，可用离散分布的同色列加抽取箭头展示 packed representation。不要沿行划分，不要做棋盘状逐元素稀疏。

初始化 warmup 与底部 interval 时间轴分离，时间轴标题写 `After importance warmup`，避免读者认为初始化第一步已经 selective。

## 5. 多 GPU 主体：完整副本 + 不同 owner

至少显式画 GPU Rank 0 和 GPU Rank 1 两个等尺寸卡片，用省略号和 `… Rank N−1` 表示可扩展。GPU 使用扁平芯片 icon，不画真实硬件照片。

每个卡片含四个紧凑功能模块：

### ① Model Replica & Forward

```text
Full BF16 Model Replica
Forward → Loss
```

两个 rank 的模型 icon 相同，输入 batch icon 不同。Forward 是 dense model execution。Loss 用小曲线/标量 icon 展示，并用箭头连接 backward。

### ② Selective / Dense Backward

在卡片内并列两小行：

```text
Ordinary: Packed ∇W[A,B]
Boundary: Full ∇W[A,B,C]
```

小字：

```text
Activation-gradient propagation retained
```

### ③ Owner Gradient Shard

归约之后显示短的彩色分片条：

```text
Rank-r Owner Fragments
Aᵣ | Bᵣ | Cᵣ
```

rank 0、rank 1 的分片条用不同 owner 边框/编号，内部仍采用 A/B/C 三色。旁注：

```text
Ownership follows ZeRO partitions,
not whole layers or entire A/B/C bands
```

不能画成 Rank 0 只负责 A、Rank 1 只负责 B/C。

### ④ Local Update & Accumulation

画两个模块：

```text
Aᵣ GPU Adam
FP32 master + m + v
```

```text
Bᵣ GPU Accumulator
Across successful interval steps
```

C 的箭头从 boundary owner fragment 直接向下通往本 rank CPU，不经过一个完整 GPU C accumulator。

## 6. 多 GPU Collective：明确两次不同方向的操作

在 GPU 卡片之间或上方设置两条分开的紫色共享通信轨道。

### 梯度归约轨道

标题：

```text
Gradient Reduce-Scatter
All ranks contribute → Owner shards
```

用两边输入、分别输出的分片箭头表示：所有 rank 的同一逻辑坐标梯度被归约，结果仅落到 owner。图例微型示意：

```text
Rank 0 local gradients ─┐
                       ├─ Reduce-Scatter → g_owner0, g_owner1
Rank 1 local gradients ─┘
```

两个模式标签：

```text
Ordinary: compressed A+B buckets
Boundary: Native ZeRO-2 dense gradient-ready buckets
```

明确 reduce 在 backward gradient ready 后、owner optimizer update 前。Dense 路径强调 bucket-by-bucket，而不是等全模型 backward 结束后一起构造 staging。

边注：

```text
Deterministic bucket membership & collective order
```

### 参数发布轨道

标题：

```text
Updated-Value All-Gather / Scatter
Owner updates → All model replicas
```

两条入口：

- `A updated values · Every successful step`，来自 owner GPU Adam。
- `B/C updated values · On committed CPU version`，来自各 rank H2D return。

该轨道的输出回到所有 GPU 的完整 model replica。必须把它与 gradient reduce-scatter 分开，不可用一个 “NCCL Sync” 双向箭头代替。

小型灰色控制轨道：

```text
Scalar All-Reduce
Overflow / Norm · Version Readiness
```

Norm/overflow 决定本步能否更新；readiness 决定哪个异步版本可以全 rank 提交。不要把它画成再次归约大张量。

## 7. GPU–CPU 边界：传输方向、内容、时机、精度

画清晰横向边界，标签：

```text
GPU ↔ Host Transfers
```

每个 rank 各有向下 D2H、向上 H2D 两条箭头，避开交叉。方向本身与箭头标签必须同时明确。

### 向下：B 梯度

```text
BF16 D2H
Frozen Bᵣ interval gradient
Boundary submission only
```

从 GPU B accumulator 发出。旁注：

```text
Includes boundary B gradient
Mean over successful steps (configured)
```

### 向下：C 和 dense-only 梯度

```text
BF16 D2H · Streamed per bucket
Cᵣ + dense-only owner gradients
Dense boundary only
```

从 dense owner fragment split 发出，落到预计算 CPU buffer offset。

强调：

```text
No C gradient on ordinary steps
No additional full-model GPU staging
```

### 向上：更新后参数值

```text
BF16 H2D
Updated owner Bᵣ/Cᵣ + dense-only values
After CPU completion & coordinated commit
```

从 CPU BF16 return mirror 返回本 rank GPU bounded collective workspace，之后进入 all-gather/scatter。

旁注：

```text
Return updated values — not gradients or Adam moments
```

不要将整份 FP32 master 或 moments 从 CPU 加载回 GPU。不要画所有模型参数每步从 CPU reload。GPU 保留完整 BF16 模型副本，仅更新对应坐标。

## 8. 每 rank 独立 CPU optimizer 模块

CPU 区域使用浅蓝灰底，在每个 GPU 下方放对应 CPU worker lane：

```text
CPU Worker for Rank 0
CPU Worker for Rank 1
…
```

每个 lane 从左到右：

### Shared + Pinned Gradient Slots

图标用两三个带版本编号的扁平 buffer 卡片：

```text
BF16 Gradient Slots
Version-owned · Bounded lag
```

C streaming arrow 和 B frozen arrow 都落入本 rank 对应版本 slot。不能把 version slot 理解为复制完整 GPU 模型。

### Independent CPUAdam Process

CPU 芯片 + 小齿轮 icon：

```text
Wait D2H Completion
BF16 → FP32 Workspace
DeepSpeedCPUAdam
```

旁边连接持久状态抽屉 icon：

```text
Owner B/C Canonical State
FP32 Master + Adam m,v
```

dense-only 参数也在该 CPU authority 内，用小标签说明。

### BF16 Return Mirror

CPUAdam 输出连接：

```text
Updated BF16 Return Values
Protected Until GPU Publication Completes
```

再沿 H2D 返回对应 GPU。

在整个 CPU 区域画横向 bracket：

```text
Independent CPU processes overlap subsequent GPU training
```

不要用 CPUAdam → next forward 的实线依赖表示重叠；用下方时间条的并行跨度表达。

## 9. 两条可读的执行时序展开

### Ordinary Step：O

一条左到右编号路径：

```text
① Batch H2D + Full Forward
 → ② Packed A/B Weight Gradients
 → ③ Compressed A/B Reduce-Scatter
 → ④ Global Overflow/Norm + Clip
 → ⑤ A Owner GPU Adam + B Owner Accumulate
 → ⑥ A Value All-Gather
 → Next Step
```

附注：

```text
No new B/C gradient offload on ordinary steps.
Previously submitted CPU jobs may still execute or publish.
```

不能标成普通步骤完全没有 PCIe 流量，因为还有 batch H2D 和可能的旧版本 B/C 参数返回。

### Dense Boundary：D

一条左到右编号路径，②–④用 streaming bracket 包围：

```text
① Full Forward / Backward
 → ② Full A+B+C Gradient-Ready Bucket
 → ③ Native ZeRO-2 Bucket Reduction to Owners
 → ④ Split Owner A/B/C; Stream C to CPU; Release Safely
 → ⑤ Final Overflow/Norm Decision + Clip/Scale
 → ⑥ A GPU Update; Freeze B; Submit B/C CPU Job
```

随后分叉：

- A 走 `A All-Gather → Continue GPU Training`；
- CPU 走 `Wait D2H → FP32 CPUAdam → BF16 H2D → B/C All-Gather → Visibility`。

说明：C 可以在最终 norm/overflow 决策前先完成 staging，但 CPU optimizer job 只在成功步骤确认后提交；图中把 speculative transfer 和 optimizer update 区分开。若空间不足，写 `Stage first; update only after finite-step approval`。

不得声称 boundary C 累积了前几个 ordinary steps。可在图下注释：

```text
B uses interval accumulation; C uses the current dense-boundary gradient.
```

## 10. 异步重叠与安全发布

使用底部三条泳道：

```text
GPU compute:       O₁ | O₂ | O₃ | D₄ | O₅ | O₆ | …
CPU optimizer:                       [ Job v: wait / Adam ]
Publication:                                            [H2D → AG → Visible]
```

这是时序示意，不固定 CPU job 一定在 O₆ 完成。标签：

```text
Illustrative timing; commit when all ranks are ready
```

标明：

- `A update` 在每个成功步骤下方出现小勾；
- `B accumulation` 在 O 和 D 下方出现小加号；
- `B/C job submission` 只在 D 处画信封/任务 icon；
- `Bounded lag: wait at capacity` 用小闸门 icon；
- publication 末尾使用锁/眼睛 icon：`Record visibility → Next forward waits`；
- return mirror 的复用用回环小箭头：`Reuse only after publication completes`。

CPU version 标签只指 B/C job 的版本，不要暗示 A/B/C 共享一个每步同步的全局 optimizer version。

## 11. 最终输出与训练闭环

右侧分为上下两个节点：

### 训练中闭环

```text
Updated Replicated Model
A: every successful step
B/C: ordered asynchronous publication
```

回环到下一次 forward，回环线沿主图外围走，不穿过 CPU buffer。

### 训练结束输出

大模型文件 + 勾选 icon：

```text
Drain Pending Updates
→ Complete Publication / Visibility
→ Fine-Tuned Model W*
```

这是整个图最明确的终点。Training Loss 作为训练内辅助标量，不应与最终微调模型争夺视觉焦点。

输出可以标注 `Export`，但不要宣称当前某次 benchmark 已保存 checkpoint。图表达正确的最终导出语义。

## 12. 视觉编码与图例

| 类型 | 颜色/样式 | 用途 |
|---|---|---|
| A | 红粉 `#D95F76` | 每步 GPU 更新列 |
| B | 琥珀 `#E6A23C` | GPU interval 累积列 |
| C | 蓝 `#4C91C7` | boundary-only 其余列 |
| GPU 区域 | 极浅绿灰 `#F0F6F2` | GPU compute/state |
| CPU 区域 | 极浅蓝灰 `#EFF4F9` | CPU buffers/process/state |
| Inter-GPU collective | 紫 `#8064A2` 粗实线 | RS / AG |
| Gradient D2H | 青 `#249D9F` 向下箭头 | 梯度 offload |
| Parameter H2D | 深绿 `#41856B` 向上箭头 | 更新参数返回 |
| Control | 灰色点线 | schedule / readiness / visibility |
| Local data dependency | 深灰细实线 | 同设备模块连接 |

A/B/C 颜色属于 tensor 内容；传输箭头颜色属于通信种类。可在传输箭头旁放 A/B/C 彩色小条说明 payload，避免颜色含义混淆。

异步不应仅靠虚线表达：明确标注 `Async` 并用并行时间条展示。每种箭头都有图例，避免“虚线既是控制又是数据”的歧义。

所有关键传输箭头回答四个问题：`What / When / Direction / Dtype`。例如：

```text
B interval gradient | Boundary | GPU→CPU | BF16
Updated B/C values | Commit | CPU→GPU | BF16
```

## 13. 可直接交给绘图模型的完整提示词

> 绘制一张计算机系统论文风格的 FastOffload 技术 overview。采用扁平化、模块化的可编辑矢量设计，横向宽画布，白底、统一无衬线字体、浅色物理区域背景、清晰有方向的连接线。不要做照片、3D、等距服务器机房、营销海报或软件类继承图。使用适量数据库、模型文件、GPU芯片、CPU芯片、buffer卡片、任务信封、锁和导出勾选图标，使数据和硬件分工形象可读。图中文字使用英文，标题为 “FastOffload: Importance-Guided Multi-GPU CPU Offload”。
>
> 图分为输入与初始化、多GPU运行区、每rank CPU优化器区、版本发布与最终输出，以及底部执行时序。最左侧明确画 Training Dataset 和 Pretrained Model W₀。Dataset 经过 Tokenize & Collate、Distributed Sampler，产生不同的 Rank-local Minibatches，通过每microbatch的 Batch H2D 分别进入 GPU Rank 0 和 Rank 1。预训练模型初始化加载到每个GPU，形成相同的 Full BF16 Model Replica。至少画两个GPU，并用省略号表示 Rank N−1。必须体现 ZeRO-2 数据并行：模型完整复制，owner梯度和优化器状态分片，不是按层切分模型。
>
> 预训练参考权重 W₀ 和 warmup 权重 Ww 连接 Column Importance Selector，标注 score[j]=Σᵢ|Ww[i,j]−W₀[i,j]|。输出同一套 A/B/C Column Plan 给所有rank。画形状为[out,in]的权重矩阵，沿输入维的竖列着色：红粉A为Top-K，橙B为Next-K，蓝C为Rest，可标示Example 10%/10%/80%。Bias/Norm/1D和dense-only参数单独画小条，不强行按列切分。列计划通过灰色点线控制 selective backward 和 owner mapping。
>
> 每个GPU卡片内部依次画 Full Model Forward→Loss→Selective/Dense Backward。Forward始终使用完整模型；普通步骤只为受支持Linear计算packed A+B weight gradients，activation-gradient传播仍继续；dense boundary计算本步完整A+B+C梯度，绝不能只画C。普通路径旁标No dense zero-filled grad_weight。两个rank的梯度汇入紫色 Gradient Reduce-Scatter 轨道。普通步骤标Compressed A+B Buckets；boundary标Native ZeRO-2 Gradient-Ready Buckets，逐bucket streaming。collective从所有rank局部梯度归约到不同owner fragments，然后每个rank得到自己的Aᵣ/Bᵣ/Cᵣ。不要把一个rank分配成只负责A、另一个rank只负责C。
>
> Owner fragment之后，每个GPU上画Aᵣ GPU Adam，内部FP32 master+m+v；A每个成功step更新，之后经另一条独立紫色 Updated-Value All-Gather/Scatter 轨道将updated A广播式重建到所有模型副本。Bᵣ进入GPU B Accumulator，在成功步骤累积，在dense boundary加入本步B梯度后冻结。配置mean时用成功步骤数取平均。Cᵣ只在boundary存在，由owner dense fragment按bucket直接写入CPU目标offset，不画GPU全模型C累积buffer。旁注No additional full-model GPU staging，dense fragment仅在流和事件生命周期安全时释放/复用。
>
> GPU下方画清晰Host Transfer边界，每个GPU对应一个独立CPU worker lane。向下青色箭头分两种：Frozen B interval gradient，Boundary only，BF16 D2H；C+dense-only owner gradients，Dense boundary per-bucket streaming，BF16 D2H。普通步骤不新提交B/C梯度offload，但允许先前CPU job继续执行和返回。不可把普通步骤画成完全无PCIe活动。
>
> 每个CPU worker从Shared+Pinned BF16 Gradient Slots开始，用版本卡片表示version-owned bounded buffers；连接Wait D2H Completion→BF16-to-FP32 Workspace→Independent DeepSpeedCPUAdam Process。CPUAdam旁边画Owner B/C Canonical State，包含FP32 master和Adam moments m,v。两个rank的CPU状态分开，不是一个集中全模型优化器。CPUAdam输出Updated BF16 Return Mirror，标Protected until publication completes。
>
> 从CPU return mirror向上画深绿色BF16 H2D箭头，标Updated owner B/C+dense-only values，After CPU completion and coordinated commit。返回本rank GPU后进入Updated-Value All-Gather/Scatter轨道，将各owner更新参数合成为所有GPU上的完整模型副本。必须清晰画出CPU→本rank GPU→所有GPU两段路径。不要画梯度H2D，不要画Adam moments每步返回，不要画每步全模型从CPU reload，不要增加full-model GPU return staging。
>
> 在旁边用细灰色控制轨道画Scalar All-Reduce：Overflow/Norm与Version Readiness。A update和B/C job submission必须经成功step决策；boundary C可先staging，但不能在最终finite检查前进行CPU optimizer update。跨rank按相同版本顺序commit，H2D/all-gather/scatter完成后Record Visibility，Next Forward Waits，用锁或闸门icon表达。CPU job完成不等于forward立即可见。异步CPU工作用底部平行时间泳道展示，而非画成每个GPU step都等待CPU。
>
> 底部画After Importance Warmup的示例interval=4时间线O₁,O₂,O₃,D₄,O₅,…，A每成功step更新，B持续累积，D提交B/C CPU job。CPU job条与后续GPU计算重叠，完成时间不固定；标Bounded lag: wait at capacity。另用两条紧凑编号流程明确时序：Ordinary是Batch H2D/Forward→Packed A/B Backward→Compressed RS→Overflow/Norm/Clip→A GPU Adam+B Accumulate→A AG；Dense是Full A+B+C Gradient-Ready Bucket→Native ZeRO-2 Owner Reduction→Split/Stream C/Safe Release→Finite-Step Approval→A Update+Freeze B+Submit CPU Job→CPUAdam→BF16 H2D→B/C AG→Visibility。说明B使用interval累积，C使用当前boundary梯度，不补算前三步C。
>
> 最右侧画训练闭环Updated Replicated Model回到Next Forward；最终出口单独画Drain Pending Updates→Complete Publication/Visibility→Fine-Tuned Model W*，使用模型文件和勾选icon，确保输入数据集与模型、输出微调模型一目了然。Loss只是forward/backward之间的小标量节点，不是主要最终产物。
>
> 所有模块通过明确箭头连通，尽量少交叉，跨卡collective、D2H、H2D和控制线使用不同视觉编码并附图例。tensor A/B/C保持红粉/橙/蓝，collective为紫色，D2H为青色向下，H2D为深绿向上，控制为灰色点线。GPU和CPU容器使用很浅的底色；icon仅辅助，关键数据必须文字标注。每条重要传输线注明内容、时机、方向和精度。信息丰富但层级清楚，主图看模块和归属，底部看执行顺序。不要放benchmark数字、loss曲线、scheduler、源码文件名或大段API代码。输出出版级SVG/PDF与高分辨率PNG。

## 14. Negative prompt

```text
No photorealism, no 3D hardware, no isometric datacenter, no gradients or shadows,
no decorative neural-network background, no benchmark dashboard, no LR scheduler,
no UML class hierarchy, no tiny paragraphs inside GPU cards, no tangled arrows.
Do not depict model/pipeline parallelism instead of ZeRO-2 data parallelism.
Do not assign entire A/B/C bands to different ranks.
Do not show only A/B forward computation or omit activation-gradient propagation.
Do not show C-only dense backward or interval accumulation of uncomputed C gradients.
Do not merge reduce-scatter and all-gather into an unlabeled synchronization arrow.
Do not omit parameter H2D or post-H2D inter-GPU all-gather.
Do not transfer full FP32 master/moments back to GPU every step.
Do not show full-model GPU gradient/return staging or unsafe buffer reuse.
Do not depict instantaneous globally visible CPU updates or discard pending final updates.
Do not invent performance claims, exact overlap latency, or fixed CPU completion steps.
```

## 15. 最终审图清单

- [ ] Dataset、Pretrained Model 输入与 Fine-Tuned Model 输出明确。
- [ ] 两张以上GPU的视角清楚；每rank完整模型、不同batch、不同owner。
- [ ] 数据batch H2D、初始化模型加载、更新参数H2D三者区分。
- [ ] A/B/C按输入列划分，dense-only参数另列。
- [ ] Ordinary packed A/B与boundary full A/B/C路径区分。
- [ ] Full forward和activation-gradient传播没有被误画成稀疏。
- [ ] Reduce-scatter在owner更新之前；all-gather在owner更新之后。
- [ ] 普通compressed归约与Native dense bucket生命周期分开。
- [ ] A每成功step GPU更新，B GPU累积，C boundary-only。
- [ ] B mean包含boundary本步；C不声称跨ordinary累积。
- [ ] 每rank独立CPU process、BF16梯度和FP32状态都画出。
- [ ] D2H与H2D payload、dtype、触发时间明确。
- [ ] B/C H2D后仍有跨GPU参数all-gather。
- [ ] 未新增full-model GPU staging；局部buffer生命周期安全。
- [ ] Overflow/norm和version readiness属于标量控制collective。
- [ ] CPU overlap、bounded lag、visibility和return-buffer保护可见。
- [ ] 最终导出前drain pending updates，不遗漏B/C版本。
- [ ] 无伪造性能、未来功能、源码级杂项或歧义箭头。
