<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- DeepSpeed Team -->

# FastOffload：方法与实现详解

> 核对日期：2026-09-17。
> 本文以当前工作树的 **ZeRO-2 owner-sharded Takeover** 为主要实现依据，包含已实现的三种B路径。
> CPU B Reduce-Scatter属于当前本地新增实现；不能仅用此前已发布的基线commit代表本文全部功能。
> 全文仅说明FastOffload自身的方法、组件、作用与边界，不展开其他方法的设计、实验对比或训练数据集配方。

**阅读约定：**第一章回答“为什么这样设计、数学上更新什么”；第二章回答“代码怎么执行、张量放在哪里、谁读写、何时可以复用”。
“释放”必须区分**不再承载有效数据、允许复用、删除引用、底层存储真正归还**，它们不是同一时刻。
文中BF16表示主要验证配置；凡代码继承输入或模型dtype的地方会单独注明，不将整条路径一概称为BF16或FP32。

**目录**

- 第一章 Methodology：目标与术语、重要性选择、A/B/C语义、三层累积、三种B路线、owner分片、数值规则、异步发布与方法边界。
- 第二章 Implementation：接入点、数据结构、逐步执行、三种B的实际复制路径、缓冲状态机、CPUAdam、通信、精度、checkpoint、辅助组件与验证范围。

## 第一章 Methodology：FastOffload的方法

### 1.1 问题定义与设计目标

FastOffload面向 **DeepSpeed ZeRO Stage 2 + CPU optimizer offload** 的全参数微调。
模型计算仍在GPU进行，优化器工作在GPU与CPU之间重新分配；“全参数”表示所有参数都有更新机会，不表示每一步都更新全部参数。

方法希望同时降低以下成本：

1. 普通步骤中不必要的Linear权重梯度计算。
2. 每一步完整梯度归约与完整参数发布的有效载荷。
3. 每一步在CPU上处理全部优化器状态的工作量。
4. GPU等待CPU更新完成的暴露时间。
5. 密集更新边界的GPU临时张量峰值。

基本取舍是：**少量重要参数高频更新，其余参数以不同方式低频更新；允许受约束的异步陈旧，但不能允许缓冲覆盖、乱序发布或rank副本分歧。**
这会改变优化过程，不能把它描述为“只换实现、与每步完整AdamW严格等价”。

### 1.2 五套不能混淆的时间与状态

| 名称 | 含义 | 何时变化 |
|---|---|---|
| microstep | 一次训练microbatch的forward/backward | 每个microbatch推进 |
| GAS边界 | DeepSpeed完成一组microbatch累积，允许optimizer step | 每`G`个microbatch到达 |
| 成功Hybrid step | 完成数值检查、没有overflow的Takeover更新 | 仅成功更新后推进 |
| B/C任务version | 某个密集边界冻结并提交的CPU任务编号 | 每次成功提交B/C任务推进 |
| committed/visible version | 已按序发布的B/C版本，以及其GPU写入真正可见的状态 | CPU任务完成后，在安全发布边界推进；可见性还由event保证 |

另有Adapter/Observer的调用计数，用于事件记录和重要性预热触发；它不能直接代替Hybrid的成功步计数。
`max_async_lag=2`限制未完成提交生命周期的B/C版本数量，不是“只落后两个microstep”。

### 1.3 预热与重要性选择

对二维参数`W`，形状为`[out_features, in_features]`，定义输入列分数：

```text
score[j] = sum_i abs(W_after_warmup[i,j] - W_reference[i,j])
```

`W_reference`是安装选择器时保存的参数快照；标准训练入口应在训练前保存预训练权重，所以称为pretrained-delta。
若安装发生在其他时点，代码不会自动替用户重新加载预训练checkpoint。

设输入列数为`D`，比例为`r`：

```text
K = max(1, ceil(D * r))
A = 分数最高的前K列
B = 排名[K, min(2K,D))的列
C = 其余列
```

- A/B/C互不重叠。
- 对小矩阵，取整可能使B不足K列，或者C为空。
- 比例是逐矩阵输入列比例，不是严格的全模型元素配额。
- 当前生产选择是预热后一次选定，不是每一步重新排序。
- 选择的是输入列，不是输出行、整个layer或attention head。
- 非二维参数不套用列评分，按dense参数处理，常见对象是bias和Norm权重。

**作用：**让高频计算与更新优先覆盖对预热训练发生较大响应的参数方向。
这是一种可执行的启发式，不是重要性最优性或收敛性的证明。

### 1.4 A/B/C的完整更新定义

| 分组 | 普通成功步骤 | 密集边界成功步骤 | 优化器状态位置 |
|---|---|---|---|
| A | 当前梯度归约、统一数值处理后更新 | 同样更新 | owner GPU |
| B | 当前梯度归约、统一数值处理后加入周期累积 | 先加入本次梯度，再提交整个周期的B | owner CPU；周期累积设备可选 |
| C | 不执行C参数更新 | 使用当前边界的C梯度更新 | owner CPU |
| 非二维dense参数 | 不更新 | 使用当前边界梯度更新 | owner CPU |

这里“普通步骤不需要C梯度”指方法需求和受支持SelectiveLinear路径；未包装模块或Native回退可能仍计算完整autograd梯度，然后仅消费需要的部分。
前向计算、Attention与输入梯度传播并未因此稀疏化。

### 1.5 三层“累积”必须分别理解

#### 1.5.1 同一次backward内的贡献合并

同一参数可能在一次backward中收到多次packed capture。对应贡献先合并成该microbatch的本地梯度。
它只覆盖本次backward，不是跨Hybrid周期的B累积。

#### 1.5.2 GAS层的microbatch累积

在一次optimizer step内，合并`G`个microbatch的梯度。
DeepSpeed已在loss侧执行GAS缩放，因此当前Takeover采用**求和**，不再除一次`G`。

- 普通packed步骤：本地压缩A/B由microbatch accumulator累积，之后再归约到owner。
- 密集Native边界：沿用Native的GAS处理和CPU-offload暂存机制，不能把它全部画成同一个GPU压缩buffer。
- CPU B Reduce-Scatter路线当前只允许GAS=1，不存在该路线下跨多个microbatch的受支持用法。

#### 1.5.3 Hybrid层的B跨成功step累积

GAS边界完成、梯度已归约并通过统一数值检查后，B才进入跨成功Hybrid step的累积器。
这个累积器保存的是**owner负责的B梯度**，不是每rank未归约的完整本地B。

**A没有这一层周期累积；C也没有这一层周期累积。**
C在GAS>1时仍可能有microbatch累积，这不等于C跨多个Hybrid step累积。

### 1.6 B的周期更新与C的边界更新

设周期为`N`，`g̃_B,t`表示第`t`个成功Hybrid step在归约、unscale与裁剪之后的B梯度。
一个周期内：

```text
S_B ← S_B + g̃_B,t              # 每个成功step执行，包括边界
n_B ← n_B + 1
到达n_B=N：
    g_B,job = S_B / N           # mean模式
    g_C,job = g̃_C,current      # 只用当前边界C
    冻结该周期，换一个B累积buffer
```

配置也支持`sum`，但mean与sum是不同算法设置，不能混用结论。
当前不是“先把未裁剪的整周期梯度加完，再对B/C另做一次统一裁剪”。
每个成功step先按该步参与梯度得到clip scale，B再把这些处理后的梯度累加。

A不进入CPU B/C任务，避免同一边界被GPU和CPU重复更新。
C在普通步骤不推进自身Adam计数，也不暗中执行weight decay。

### 1.7 一个周期的逐步示例

以下例子假定GAS=1、Native重要性预热1步、周期`N=4`，没有overflow。
`A_t/B_t/C_t`表示经过对应归约与数值处理的本步梯度，不是模型参数。

| 训练步 | 阶段 | A | B活跃buffer内容 | C | B/C版本动作 |
|---|---|---|---|---|---|
| 1 | Native预热 | 完整Native更新 | 尚未开始Hybrid累积 | 完整Native更新 | 选择重要性；之后迁移状态 |
| 2 | Hybrid 1 | 用`A_2`更新 | `B_2` | 不更新 | 无 |
| 3 | Hybrid 2 | 用`A_3`更新 | `B_2+B_3` | 不更新 | 无 |
| 4 | Hybrid 3 | 用`A_4`更新 | `B_2+B_3+B_4` | 不更新 | 无 |
| 5 | Hybrid 4，密集边界 | 用`A_5`更新 | 加入`B_5`后冻结 | 使用`C_5` | 提交version 1，B取4步mean |
| 6～8 | 下一周期普通步 | 每步更新 | 在另一个buffer累积`B_6..B_8` | 不更新 | version 1若共同ready则在安全点发布 |
| 9 | 下一密集边界 | 更新 | 加入`B_9`后冻结 | 使用`C_9` | 提交version 2；必要时等待容量 |

version 1不保证在第6步前已经发布，取决于CPU完成情况和全rank readiness。
B/C模型参数可以保持上一已发布版本，但不能出现某个rank已使用新版本、其他rank仍使用旧版本的任意分叉。

### 1.8 三种B归约/累积位置

| 路线 | B跨rank归约 | B周期累积 | B跨GPU/CPU边界的时机 | 主要取舍 |
|---|---|---|---|---|
| GPU RS + GPU accumulation | GPU | owner GPU | 周期边界传累计和 | 降低普通步D2H/CPU累加，但增加GPU常驻buffer |
| GPU RS + CPU accumulation | GPU | owner CPU | 每个成功step传owner B | 节省GPU周期buffer，但每步有CPU复制/累加 |
| CPU RS + CPU accumulation | CPU Gloo | owner CPU FP32 | 每步在归约前传本地B | GPU不处理真实B归约值，但增加host传输、转换和CPU通信 |

`second_reduce_scatter_device`控制归约位置；`accumulation_device`控制周期累积位置。
只设置CPU累积不会自动切换到CPU归约。

第一条路线的GPU累积器保存**和**。当前大owner实现把它送到CPU版本槽后，再在CPU侧完成mean。
第二条路线的旧累积入口保持输入dtype，不保证FP32。
第三条路线显式把本地B转成CPU FP32，再执行CPU归约和FP32周期累积；但边界提交仍可能量化到模型dtype槽。
因此三条路线在有限精度、通信顺序和发布时序上不能假定逐位等价。

### 1.9 owner-sharded通信与优化器

每个rank保有用于forward/backward的完整低精度模型副本，但优化器权威状态按ZeRO owner分片。

```text
本地梯度
  → 按参数列和owner组织的Reduce-Scatter
  → 每个owner只更新自己负责的参数元素
  → All-Gather更新值
  → scatter到所有rank的模型副本
```

A的优化器状态也为owner-local GPU状态，而不是每个rank复制完整A状态。
B/C优化器状态为owner-local CPU状态。

**作用：**同时分摊优化器存储和计算，并在普通步骤只发布需要更新的A。
密集版本发布涉及B和C，可能覆盖模型的大部分参数；“压缩发布”不表示所有时刻通信都只有A的大小。

### 1.10 Forward与Backward的计算范围

FastOffload的SelectiveLinear保持：

```text
Forward:    Y = X Wᵀ + bias               # 完整
grad_input: dX = dY W                      # 完整
grad_weight_selected: dW[:,A∪B] = dYᵀ X[:,A∪B]
```

普通步骤减少的是受支持Linear的dW计算、物化和后处理。
Attention、完整前向、activation和输入梯度仍然存在。
密集边界切回原生Linear backward，以取得完整A/B/C和bias梯度。

小token或小output shape可能更适合完整GEMM，因而保留按shape阈值的Native回退。
回退只改变如何产生梯度，不应把Hybrid更新规则改回每步全参数更新。

### 1.11 数值检查、裁剪与overflow

归约之后，owner分片共同决定当前step的global norm与overflow：

```text
norm² = sum_over_owners(sum(current_reduced_gradients.float()²))
global_norm = sqrt(norm²) / loss_scale
clip_factor = max(1, global_norm / clip_grad)     # clip_grad>0时
combined_scale = loss_scale * clip_factor
processed_gradient = reduced_gradient / combined_scale
```

普通步骤的检查域是本步实际参与的A/B；密集边界包含A/B/C及dense参数。
CPU B路线中，CPU B与GPU A/C共享一个裁剪决定，只移动统计标量，不把整份CPU B搬回GPU计算norm。

Overflow的当前Takeover语义：

1. 当前A不更新。
2. 当前B不加入有效周期buffer。
3. 当前C不提交CPUAdam。
4. Hybrid成功计数不推进，因此下一次仍面对同一个逻辑边界。
5. 丢弃当前无效microbatch/Native staging，不无故丢弃之前成功步骤的B累积。
6. 之前已提交的有效任务仍按自己的version/LR完成。

这是当前step提交前的保护，不等于已经实现任意queued/running CPUAdam的撤销和回滚。

### 1.12 Adam状态与学习率规则

- A与B/C各自维护FP32 master、`exp_avg`、`exp_avg_sq`及step计数。
- 从Native预热状态迁移，不在切换时重新把moments清零。
- A每个成功step推进自身Adam计数；B/C每个实际CPU更新推进自身计数。
- Weight decay只在该组真正更新时应用。
- Native optimizer的当前group LR是LR authority；构造时缓存不是live LR。
- 边界B/C任务使用**提交时A使用的同一LR**，CPU晚执行也不改用新LR。
- 不按周期内各步LR对B梯度重新加权，也不额外乘除周期来修正LR。
- LR=0合法：参数更新量为0，但moments和计数仍按实际更新推进。
- 当前要求各optimizer group超参数一致，仅同步LR；不同group LR、动态betas/eps/weight decay没有完整支持。

对一个实际被更新的分组，AdamW的逻辑为：

```text
n ← n + 1
m ← beta1*m + (1-beta1)*g
v ← beta2*v + (1-beta2)*g²
m_hat = m / (1-beta1^n)
v_hat = v / (1-beta2^n)
master ← (1-lr*weight_decay)*master - lr*m_hat/(sqrt(v_hat)+eps)
```

A的`g`是本步A梯度；B的`g`是本周期mean或sum；C的`g`是当前边界C梯度。每组使用自己的实际更新计数，不能统一用训练microstep号作bias correction。上式表示逻辑运算，实际cast/归一化顺序见第二章。

重要性warmup和LR warmup互相独立。重要性预热期间LR为0、更新溢出或没有生成可迁移Adam状态，都需要单独处理和验证，不能默认选择仍然有效。

### 1.13 异步执行与安全发布

FastOffload区分四件事：

1. GPU产生了梯度。
2. CPU已读到完整梯度并更新master。
3. 更新值已经提交到GPU传输/collective。
4. 下一次forward可以安全读取全部更新值。

只有第4件事满足，才算GPU使用新参数版本的可见性条件成立。
后台worker不直接修改live GPU模型；训练线程在安全边界组织H2D、更新值All-Gather和scatter。

允许的重叠是：上一周期B/C在CPU更新，同时GPU计算下一周期并累积新B。
禁止的重叠是：下一周期覆盖旧梯度槽、下一CPU任务覆盖尚在H2D的返回镜像、forward读一半旧值一半新值。

### 1.14 缓冲所有权与有界陈旧

方法需要分别管理：

- 当前周期B累积buffer。
- 已冻结周期的B输入。
- 某version拥有的CPU梯度槽。
- CPU FP32计算工作区。
- 某version的返回参数存储。
- 正在发布且尚未完成的GPU写入。

双缓冲解决新旧B周期互不覆盖；版本槽解决传输和CPU读取互不覆盖；visibility event解决返回值与GPU读取互不覆盖。
这些机制不能用一句“用了双buffer所以安全”替代。

当任务或buffer达到容量时执行反压，等待最旧可发布版本；不丢梯度、不覆盖buffer、不无限增加队列。

### 1.15 Checkpoint、尾部与关闭语义

正常checkpoint/shutdown先完成已经提交的CPU任务，逐个共同version发布并等待可见性。
不足一个周期的B可以作为active accumulator保存，但不为它虚构一个C梯度或额外密集更新。

例如预热1步、周期4、总200步且均成功：

```text
Native预热更新：1次
A更新：199次
B/C任务：49个
尾部尚未提交的B：3步
```

模型权重导出不等于optimizer-state resume。
用户强制中止进程也不等于完成了正常drain或保存了可恢复checkpoint。

### 1.16 方法的作用与明确边界

若忽略一维参数、取整、padding、warmup及fallback，A/B/C元素比例为`a/b/c`，周期为`N`：

```text
每步平均需要的逻辑权重梯度覆盖 = a + b + c/N
每步平均CPU optimizer元素更新量 = (b+c)/N
每步平均全部optimizer元素更新量 = a + (b+c)/N
```

若B逻辑元素总量为`E_B`、DP数为`P`，忽略不均衡与额外copy，并假设同一传输dtype：

```text
GPU RS + GPU accumulation：每rank每步平均B D2H元素量 ≈ E_B/(P*N)
GPU RS + CPU accumulation：每rank每步平均B D2H元素量 ≈ E_B/P
CPU RS + CPU accumulation：每rank每步平均B D2H元素量 ≈ E_B
```

这些是工作量和数据流模型，不是端到端加速保证。实际还包括dense forward、dX、通信padding、不同dtype、CPU内存带宽、host同步、版本等待和临时分配。
当前不把CPU归约路线描述为必然更快，也不把GPU累积路线描述为必然更省显存。

## 第二章 Implementation：FastOffload的实际实现

### 2.1 组件边界与真实入口

下列路径均相对`deepspeed/runtime/fastoffload/`，ZeRO接入文件另行注明。

| 组件 | 当前职责 | 主要文件 |
|---|---|---|
| 配置/API | 严格配置、安装handle、建立controller | `config.py`、`api.py` |
| Adapter | ZeRO参数ID、partition、梯度视图、Native norm/LR、模型scatter | `adapters/zero2.py` |
| Controller | 接收DeepSpeed生命周期并路由到相应组件 | `controller.py` |
| Importance | CPU reference、分块评分、选择registry、Linear包装 | `importance/` |
| 梯度pipeline | packed/dense捕获、microbatch累积、归约 | `hybrid/takeover.py` |
| Takeover runtime | 数值决定、A更新、B/C提交、前向发布 | `hybrid/takeover_runtime.py` |
| owner布局/collective | 列到owner的映射、RS与更新值AG | `hybrid/partition.py`、`hybrid/owner_collective.py` |
| GPU A updater | owner-local FP32 SelectedAdamW与A发布 | `hybrid/owner_update.py` |
| CPU B/C updater | Native staging、共享槽、CPU进程与版本发布 | `hybrid/owner_cpu_update.py` |
| 累积与协调 | 两个周期buffer、future、ready结果、提交顺序 | `hybrid/buffer.py`、`hybrid/coordinator.py` |
| Numerics | owner norm、overflow、unscale和clip | `hybrid/numerics.py` |
| 基础D2H框架 | 非Takeover的同步/异步传输、pool与反压 | `schedulers/`、`transfer/`、`workers/inline.py` |
| Observer | metadata、host统计与输出 | `telemetry/` |

主生产入口是`Zero2TakeoverRuntime`，不是独立的`HybridUpdateRuntime`辅助执行器。
API对enabled Hybrid要求明确选择Shadow或Takeover，防止配置写了Hybrid却仍悄悄运行完整dense optimizer。

### 2.2 安装、关闭与模式选择

1. 应用在DeepSpeed初始化前调用`install(config)`。
2. 一个进程只允许一个尚未关闭的active handle。
3. ZeRO-2构造时调用`create_zero2_controller()`；未安装或disabled时得到Null controller。
4. Handle用弱引用集合管理其controller；关闭时逐个清理。
5. Controller/handle的正常close有重复调用保护；不应把它与任意内部对象都可重复初始化混为一谈。

FastOffload模式需要区分：

| 路径 | 是否改变实际更新语义 | 是否是本文主要A/B/C训练路径 |
|---|---|---|
| Disabled | 否 | 否 |
| 纯Observer | 否，仅metadata | 否 |
| sync_offload / async_offload | 接管D2H，但仍由Native完成每步更新 | 否 |
| Hybrid Shadow | 验证压缩通信，不用shadow结果接管参数更新 | 否 |
| Hybrid Takeover | 是，执行A/B/C更新 | 是 |

当前Hybrid要求配置`mode="observe"`，这是组件装配入口，并不表示启用Takeover后“只观察、不更新”。
`worker.type="inline"`属于基础传输框架，也不能据此推断Takeover没有spawned CPUAdam。

### 2.3 配置实际约束与默认值

配置采用严格、不可变Pydantic模型：未知字段拒绝、默认值也校验、运行时状态不得写入配置。

| 字段 | 配置类默认值 | 本文常用说明 |
|---|---|---|
| `importance.warmup_steps` | 10 | 示例可显式设为1 |
| `importance.topk_ratio` | 0.1 | A/B各一个排序区间，受取整影响 |
| `importance.comparison_chunk_rows` | 4096 | reference比较的行块大小 |
| `importance.sparse_backward` | false | 需显式启用并包装受支持Linear |
| `sparse_backward_min_tokens` | 1024 | 小shape回退阈值，不是序列截断长度 |
| `sparse_backward_min_output_features` | 1024 | 配置层阈值；包装函数自身默认值不一定相同 |
| `hybrid_update.update_interval` | 8 | 周期4必须显式配置 |
| `accumulation_device` | cpu | 主要GPU累积路线必须显式设gpu |
| `second_reduce_scatter_device` | gpu | 新CPU RS必须显式设cpu |
| `second_gradient_reduction` | mean | 也允许sum |
| `max_async_lag` | 2 | 未提交完成的B/C版本上限 |
| `compressed_bucket_bytes` | 134217728 | 目标128MiB，不是所有临时显存的硬上限 |
| `pt_reserved_cores_perc` | 0.25 | 训练侧CPU核预留比例 |
| `overdue_policy` | wait | 当前不是丢弃或无限扩容策略 |

`dense_boundary_enabled`和`double_buffer`当前被限制为true，不是已实现任意关闭语义的消融开关。
Shadow与Takeover互斥；Shadow要求GAS=1。
CPU RS要求enabled ZeRO-2 Takeover、CPU accumulation、GAS=1与完整world DP。

一个明确的GPU RS/GPU accumulation示例：

```json
{
  "enabled": true,
  "mode": "observe",
  "zero_stage": 2,
  "failure_policy": "raise",
  "importance": {
    "enabled": true,
    "algorithm": "pretrained_delta_topk",
    "warmup_steps": 1,
    "topk_ratio": 0.1,
    "sparse_backward": true,
    "sparse_backward_min_tokens": 1,
    "sparse_backward_min_output_features": 1024
  },
  "hybrid_update": {
    "enabled": true,
    "zero2_takeover": true,
    "update_interval": 4,
    "accumulation_device": "gpu",
    "second_reduce_scatter_device": "gpu",
    "second_gradient_reduction": "mean",
    "compressed_bucket_bytes": 134217728,
    "max_async_lag": 2,
    "pt_reserved_cores_perc": 0.25
  }
}
```

改`accumulation_device=cpu`得到GPU RS/CPU accumulation；再改`second_reduce_scatter_device=cpu`得到CPU RS/CPU accumulation。
还需要DeepSpeed侧ZeRO-2 CPU optimizer offload及相应pin-memory设置；这段不是完整训练配置。
通用模型需在DeepSpeed初始化前调用`enable_selective_linear()`，将配置中的阈值传给包装器；单独加载JSON不会自动修改任意模型的Linear。

### 2.4 ZeRO生命周期接入点

接入文件：`deepspeed/runtime/zero/stage_1_and_2.py`。

| 时点 | FastOffload动作 | 所有权变化 |
|---|---|---|
| 初始化 | 建立controller、reference、layout与runtime | CPU reference及metadata开始持有 |
| 前向边界 | `prepare_forward()` | 初始化Hybrid状态；检查共同ready；保证返回值可见；设置本次dense/packed选择 |
| backward开始 | `on_backward_begin()` | 记录当前microstep metadata |
| 梯度ready | `on_gradient_ready()` | 普通步接管梯度；CPU RS密集边界预先取出B并清零原B列 |
| Native owner梯度形成 | `transfer_gradient()` | 密集边界提取owner A/B，将C写入最终CPU槽 |
| backward结束 | `finish_microbatch()` | 完成压缩GAS累积，必要时生成本次`TakeoverGradientBatch` |
| optimizer step | `takeover_step()` | 统一数值处理；A更新；B累积或B/C提交 |
| step结束 | Observer/importance回调 | 更新统计、触发一次性importance选择 |
| 保存/恢复 | Hybrid state接入ZeRO state字典 | drain、序列化、校验与恢复 |
| 正常关闭 | controller/runtime close | drain、等待visibility、关闭worker、注销mapping与CPU group |

Takeover返回结果后，ZeRO更新loss scale/overflow状态，清理grad和Native临时buffer，然后直接返回，不再执行第二次完整Native optimizer step和完整参数发布。

### 2.5 初始化与importance对象生命周期

#### 2.5.1 Reference

`StreamingImportanceSelector`遍历参数，`detach().to(device="cpu", copy=True)`保存reference。
它是普通CPU张量，dtype继承原参数，不是额外FP32全模型GPU副本。

```text
安装/初始化
  → CPU reference字典
  → Native预热
  → 逐参数pop reference并分块比较
  → registry保存列索引和元信息
  → reference字典清空
```

被pop的reference仅在当前参数评分期间仍由局部变量引用；并非直到整个训练结束才释放。
恢复时若registry已ready，选择器清空无用reference，不再重新选择。

#### 2.5.2 评分与索引

- 每个参数只建立长度为输入列数的FP32 score。
- `comparison_chunk_rows`控制当前参数的FP32比较chunk，不表示reference总CPU存储只有一个chunk。
- `torch.topk`先按分数选A/B，再将各组列号排序以适配owner映射和gather。
- 当前没有额外定义跨不同硬件的同分数tie-break协议，不能由“桶顺序确定”推导出任意环境top-k逐位一致。
- registry常驻CPU索引/metadata；参数属性和pipeline缓存建立所需GPU索引。
- `output_path`未含`{rank}`时自动加入rank，避免多rank写同一选择文件。

`importance/algorithms.py`提供进程本地的`register_importance_algorithm(name, factory)`。自定义factory在初始化前注册，接收`topk_ratio`和`comparison_chunk_rows`并返回`ImportanceAlgorithm`。空名字、重复注册、不可调用factory和未知算法会明确报错。这个扩展接口已实现，不表示已经实现了多种在线自适应重要性算法。

#### 2.5.3 从Native迁移Adam

第一次激活Takeover时，`_initialize_from_dense_adam()`抽取Native owner FP32 master、moments及step。
A迁移到owner GPU SelectedAdamW，B/C迁移到owner CPU状态；dense参数走C分支。
Hybrid master成为后续更新的权威状态，不能假设旧Native flat master在每个Takeover step都同步刷新。
Native optimizer的group LR仍继续作为scheduler authority读取。

### 2.6 先看张量总账：在哪里产生、在哪里积累、何时结束

| 对象 | 范围/位置 | dtype | 产生与消费者 | 逻辑生命周期/复用条件 |
|---|---|---|---|---|
| live model参数 | 每rank完整GPU副本 | 模型dtype，主配置BF16 | forward/backward读取；A/BC发布写入 | 整个模型生命周期 |
| importance reference | 每rank CPU快照 | 继承模型dtype | 初始化保存，预热后评分读取 | 逐参数评分后删除引用 |
| importance scores | 当前参数GPU列向量 | FP32 | chunk比较累加 | 该参数top-k完成后结束 |
| 比较chunk | 当前参数GPU行块 | FP32 | 当前/reference差值计算 | 每块复用/释放临时引用 |
| packed dW | 本rank GPU，A∪B列 | 实际GEMM输出dtype | SelectiveLinear产生，capture读取 | 分拆/本地累加接管后结束 |
| `_micro_first/second/dense` | 本rank本次backward | 随路径继承；CPU RS的B为CPU FP32 | capture累加，同一次backward贡献合并 | `finish_microbatch()`交给GAS accumulator后清空字典 |
| 压缩GAS accumulator | 本rank；普通A/B通常GPU | 继承输入 | 合并一个optimizer step内的microbatch | GAS边界取走映射并重置计数 |
| RS packed输入 | 当前桶、按owner补齐 | 继承该band输入；CPU RS为FP32 | collective读取 | 单次RS结束后不再持有局部packed输入 |
| owner归约输出 | 本rank owner shard | RS输出dtype | norm、A更新或B累积 | 当前step消费完成后结束；可能跨多个桶同时存活 |
| B周期buffer 0/1 | owner GPU或CPU | 继承输入；CPU RS路线明确FP32 | 成功step原地add，边界freeze | 输入安全转移后清零并变FREE，分配保留以便复用 |
| CPU RS B scratch | CPU pinned，单rank可复用 | BF16 | 每块D2H写入，CPU转换读取 | 每块event完成且复制进FP32结果后可复用 |
| CPU RS本地B结果 | CPU，尚未owner归约的本地B | FP32 | 捕获时形成，Gloo输入读取 | 持有至本step CPU归约消费完；不是只有一块scratch的内存 |
| Native边界owner fragment | GPU，当前Native参数/bucket片段 | Native梯度dtype | 拆A/B/C、记录norm | 完成提取/安全传输安排后沿Native生命周期释放 |
| Native A/B提取结果 | owner GPU | 继承片段dtype | step统一缩放、A更新/B提交 | 本边界消费后清空引用 |
| Native C暂存视图 | 大owner为CPU版本槽的view | 模型dtype | GPU D2H写，CPU任务读取 | 随对应version梯度槽拥有；不是独立整份C副本 |
| CPU版本梯度槽 | CPU shared+pinned，共L个 | 首个相关模型参数dtype | B和C写入；协调线程归一化；CPU进程读取 | 对应version安全提交完成后允许后续版本复用 |
| CPU FP32 gradient workspace | CPU shared，一份 | FP32 | 子进程从版本槽copy/cast，供CPUAdam读取 | 顺序CPU任务间复用 |
| A master/m/v | owner GPU | FP32 | SelectedAdamW原地更新 | 跨训练step常驻 |
| B/C master/m/v | owner CPU；大路径shared | FP32 | CPUAdam原地更新 | 跨CPU任务常驻 |
| CPU return mirror | CPU shared+pinned，一份 | 模型dtype | CPU进程写；训练线程H2D读 | 前一个version的return visibility完成后才能覆盖 |
| CPU result映射 | CPU view与version metadata | 大路径继承return mirror | future→ready→commit | commit及相关传输使用完成后不再需要该映射 |
| 更新值AG工作区 | GPU、当前发布桶 | A当前为FP32；大CPU路径模型dtype | All-Gather与scatter | 该桶发布消费后结束 |
| 任务metadata | CPU/Python对象 | ID、列、version、LR等 | 协调与发布查询 | 对应version commit后移除 |
| stream/event/group | runtime资源 | 非张量 | 建立跨设备与跨rank依赖 | 对应操作完成；runtime正常关闭后清理其持有关系 |

**两个重要结论：**

1. `FREE`不等于`torch`存储已经释放；B双buffer会保留清零后的allocation。
2. `cudaHostUnregister`只解除注册，不自动删除所有shared张量引用；底层存储最终回收还依赖引用与进程生命周期。

### 2.7 Packed SelectiveLinear：一次backward内发生什么

实现：`importance/selective_linear.py`。

1. 包装器不替换Parameter，也不改模型类，只替换受支持`torch.nn.Linear`的forward方法。
2. Forward仍调用完整`F.linear`，保存inputs、weight与列索引供backward使用。
3. Backward完整计算`grad_input`。
4. 用`inputs.index_select(1, selected_columns)`建立选中输入，并计算A∪B的dW。
5. Takeover capture通过cached位置再分为A和B，分别合并到`_micro_first/_micro_second`。
6. 启用Takeover side channel时，不返回完整weight/bias autograd gradient；bias在dense边界覆盖。
7. 仅使用独立SelectiveLinear验证而没有capture callback时，才会把选中dW写进zero-filled完整gradient；不要把该验证行为画到Takeover性能路径中。
8. 未选中、到达dense边界、全列被选中、token/output shape低于阈值时，返回Native `F.linear`路径。

生存期：保存的activation/weight引用由autograd图管理；selected inputs和packed dW是该次backward临时对象；capture接管的A/B值会比局部GEMM临时变量活得更久，直到microbatch/GAS/归约消费者完成。
当前没有已完成的通用persistent packed workspace，不应把所有index_select临时分配视为已经消除。

### 2.8 普通步骤的microbatch/GAS累积

实现：`CompressedMicrobatchAccumulator`与`Zero2TakeoverGradientPipeline`。

```text
capture本地A/B
  → 本次backward的_micro_*映射
  → finish_microbatch()
  → A/B/C各自的CompressedMicrobatchAccumulator
  → 到GAS边界后take_boundary()
  → _reduce_band()
```

- Takeover显式使用`reduction="sum"`。
- GAS=1且允许接管所有权时，只detach并接管映射中的张量，不为累积再建一套zeros+copy。
- GAS>1时按首次输入dtype/device分配目标，后续贡献原地add；shape/device/dtype变化会报错。
- `take_boundary()`返回当前映射，清空内部映射并把microstep计数归零，不等于立即释放返回张量。
- 同一次packed backward重复贡献在`_accumulate_micro_gradient()`合并，之后才进入GAS层。

支持的是已测试布局与unused场景；特别是CPU B的全局补零机制不能自动证明GPU A/C任意rank条件分支缺梯度都已完备。
不同rank必须维持兼容的参数参与集合和collective顺序。

### 2.9 owner布局、分桶与真实内存边界

实现：`ParameterPartitionLayout`、`Zero2OwnerCollective`。

对row-major参数，若在ZeRO组中的起点是`O`，列数是`D`，partition大小是`Q`：

```text
group_flat_offset(i,j) = O + i*D + j
owner(i,j) = floor(group_flat_offset / Q)
local_offset(i,j) = group_flat_offset mod Q
```

主路径利用排序列和区间计数建立owner片段；参数跨partition时处理完整行与头尾部分行，不要求参数整块属于一个owner。

RS实际过程：

1. 按parameter ID排序。
2. 构造每个owner的元素计数。
3. 每个owner补齐到该桶最大owner元素数。
4. 分配`world_size × padded_numel`的packed输入。
5. 把相应值copy到owner段，padding保持零。
6. `reduce_scatter_fn()`得到本rank输出，并按world size求平均。
7. 以输出的view构建各参数的owner梯度，不为每个参数再次复制一份结果。

桶成员按全局逻辑载荷决定，不能按本rank owner大小决定。空owner仍有统一metadata与正确的空view。
`create_owner_gradient_set()`可在Native已归约后仅构造发布所需metadata，不重复做一次梯度归约。

**内存边界：**RS的packed输入是桶级临时张量，但多个桶的owner输出会保存在`OwnerReducedGradientSet`中，直到当前step统一norm和更新完成。
所以“按桶通信”不等于“GPU同一时刻只有一个桶的所有梯度”。单参数不拆分、owner padding、A/B同时存活和AG dtype都会影响峰值。

### 2.10 A的完整生命周期

```text
GPU本地A梯度
  → 同backward贡献合并 / 普通步骤GAS求和
  → GPU owner Reduce-Scatter
  → FP32统计norm，原梯度按统一scale缩放
  → owner GPU SelectedColumnAdamW
  → 更新值All-Gather
  → 所有rank scatter A到模型
  → 本步A梯度与发布临时tensor结束
```

A不进入`DoubleBufferedGradientAccumulator`，不等待N步再更新，也不传给CPU B/C任务。
跨step长期保存的是A的FP32 master/moments和计数，不是A梯度的周期和。

**实际dtype细节：**`Zero2OwnerGpuUpdater.step()`把FP32 master作为`step_values()`的values传入，返回值按values.dtype保持FP32。
因此当前A更新值All-Gather可使用FP32，最后`scatter_parameter_columns()`再转换成模型dtype。
不能把“A发布”也笼统写成BF16通信；这与大owner CPU返回镜像的模型dtype路径不同。

### 2.11 GPU RS + GPU accumulation：B的完整生命周期

```text
GPU本地B
  → GPU压缩GAS累积（如需要）
  → GPU RS，得到owner B
  → unscale/clip成功
  → owner GPU活跃周期buffer += B
  → 普通step结束，当前owner B输入可释放
  → 密集边界加入当前B，freeze-and-swap
  → 复制冻结的B和到CPU版本槽的B区间
  → transfer event完成
  → GPU周期buffer清零、允许复用
  → CPU对版本槽B区间做mean
  → 子进程转FP32，CPUAdam
```

- 两个周期buffer只按需要懒分配，每个buffer的各参数张量继承首次输入dtype，不自动提升到FP32。
- GPU周期buffer保存的是owner B，不是每rank完整B。
- 周期buffer不能在D2H仍读取它时清零；`_transfer_sources`与event保护读取窗口。
- CPUAdam开始前即可释放周期buffer的逻辑所有权，因为CPU版本槽已经接管输入。
- 清零后allocation保留，下次周期复用，因而仍影响GPU常驻/峰值。

### 2.12 GPU RS + CPU accumulation：B的完整生命周期

```text
GPU本地B
  → GPU RS得到owner B
  → GPU侧统一数值处理
  → gradient.detach().to(device="cpu")
  → CPU活跃周期buffer原地add
  → 边界加入当前B，freeze-and-swap
  → CPU周期buffer复制进CPU版本槽B区间
  → CPU侧mean，FP32 workspace，CPUAdam
```

这里必须明确两个不同buffer：

- **CPU周期buffer：**普通CPU tensor，保存跨成功step的B和。
- **CPU版本槽：**大owner路径预分配的shared+pinned tensor，供子进程读取。

`DoubleBufferedGradientAccumulator.accumulate()`本身使用普通`.to(device=...)`，没有把这一路自动改造成专用有界BF16 pinned scratch，也没有按B参数组成完整异步D2H流水线。
它继承归约梯度dtype；实际输入为BF16时不能称为CPU FP32累积。

作用是移除GPU周期buffer，并在下传前用GPU RS缩小到owner shard。
代价是每个成功step都有owner B的D2H和CPU累加，以及边界CPU buffer到shared槽的复制。
`takeover_cpu_B_d2h_bytes=0`在此路线中仅说明没有进入CPU-RS专用计数路径，不说明没有B下传。

### 2.13 CPU RS + CPU accumulation：B的完整生命周期

#### 2.13.1 本地捕获与D2H

普通packed路径在capture阶段分出B，调用`_stage_second()`：

1. 将当前B flatten。
2. 在普通CPU内存分配该梯度的FP32结果张量。
3. 准备可复用的CPU pinned BF16 scratch。
4. 按`min(numel, max(1, bucket_bytes//2))`个BF16元素分块。
5. 当前GPU块必要时转BF16，然后nonblocking copy到scratch。
6. 在当前producer stream记录event，并等待该event完成。
7. 将scratch转换/copy进CPU FP32结果对应区间。
8. 当前块完成后复用scratch，返回完整CPU FP32结果。

这里`non_blocking=True`不意味着训练线程完全不等；代码随后等待每块event。
CPU输入则直接转为FP32 clone。

#### 2.13.2 本地B集合与CPU归约

CPU FP32结果进入`_micro_second`。如果同参数有多次capture，先在CPU合并该次backward贡献。
CPU RS目前GAS=1；到`finish_microbatch()`时：

- 遍历全局固定的B参数/列集合。
- 本rank缺失的B用CPU FP32零补齐。
- 通过私有Gloo group执行真正的CPU RS。
- 输出按DP world size求平均，得到owner CPU FP32 B。

**不是**每层下传完就立即完成那层的CPU RS/Adam；当前归约集中在microbatch收尾。
也不是CPU All-Reduce后切片。
本地CPU FP32 B集合、CPU packed输入、owner输出可能同时占用host内存，scratch有界不等于全部host内存有界在同一数值。

#### 2.13.3 数值检查与周期累积

普通步骤将CPU B和GPU A的统计标量合并，再使用共同scale原地缩放各自梯度。
成功后，CPU owner B加入CPU FP32周期buffer；失败则不加入。
边界同样加入当前B，然后进入版本槽和CPUAdam。

**精度边界：**FP32 CPU累积和仍会复制/cast进模型dtype的CPU版本槽；当前并未实现“从CPU FP32累积一路保持FP32到Native CPUAdam输入”的独立精度开关。

### 2.14 Native密集边界：A/B/C到底在哪里存活

#### 2.14.1 边界之前

`prepare_forward()`根据`(_successful_steps+1) % update_interval == 0`判定边界。
它在需要时等待版本容量、预留本次C将写入的CPU slot，然后为相应Linear设置dense backward标志。
GAS>1时，Native边界的microbatch梯度由Native机制累积；CPU RS路线不允许这种GAS配置。

#### 2.14.2 GPU RS路线

```text
Native完整dW
  → Native归约/GAS处理
  → owner fragment
  → A提取到GPU _native_first
  → B提取到GPU _native_second
  → C提取后D2H到版本槽C区间
  → 原dense fragment沿Native bucket生命周期释放
```

无论B周期累积在GPU还是CPU，这里的Native归约先在GPU完成。
C已经在CPU版本槽，不再等待所有参数完成后拼一份额外全模型GPU C。
A/B提取结果会持有到边界统一数值决定和更新/提交，不能称为每个片段形成后全部立即销毁。

#### 2.14.3 CPU RS路线

在Native归约之前，`capture_native_local_second()`先取本地B下传到CPU，再把原GPU gradient的B列清零。
之后Native路径看到的是A/C有效值和B零占位；CPU B走本章2.13的Gloo归约。
Native owner capture跳过band B，从而不把那些零值误当作真正的B再次累积。

这保留了Native密集布局的安全性，但没有删除GPU通信buffer里的B位置，不能声称已经形成纯A/C压缩密集通信。

#### 2.14.4 Norm、缩放与C的提前暂存

Native在owner片段传输前记录norm/overflow统计。
GPU RS路线使用Native完整A/B/C统计；CPU RS路线组合Native A/C和Gloo B统计：

```text
global_norm² = native_AC_norm² + (cpu_B_scaled_norm/loss_scale)²
```

选出一个`combined_scale`。在做最终决定前等待本次Native transfer event，避免仍在写入的C槽被提前丢弃或复用。

- 成功：GPU A/B按scale缩放；CPU RS的B也按同一scale缩放；C的scale作为`dense_scale`跟随version交给CPU处理。
- Overflow：撤销本次Native staging version标记、清空capture引用，不更新A、不提交C、不增加本步B；既有有效B周期内容保留。

**C被提前D2H，不等于C已经被优化器消费。**
它在全局数值决定之前只是版本槽中的候选输入。

#### 2.14.5 C与非二维dense参数的完整生命周期

```text
普通Hybrid step：
    模型中的C仍参与dense forward和输入梯度传播
    不推进C优化器状态，不保存跨Hybrid step的C梯度和
    Native fallback若产生C梯度，不等于本步会更新C

密集Hybrid step：
    完整backward产生C（GAS>1时先按Native机制合并本步microbatch）
      → Native GPU归约得到owner C
      → 提取并写入本version的CPU梯度槽C后缀
      → 等待本步全部norm/overflow决定
      → 成功时固定dense_scale；失败时撤销本次暂存，不提交任务
      → CPU按dense_scale缩放C，不对C除以Hybrid周期N
      → FP32 CPUAdam更新C master/moments
      → 返回镜像C片段
      → H2D / owner AG / scatter / visibility
      → 下一forward开始使用已发布的C
```

Bias、Norm等非二维参数通过`(1, numel)`布局表示，其全部元素都走这条dense路径，没有独立A/B列子集。
C没有自己的“两周期累积buffer”；常驻的是C的CPU master/moments，候选梯度属于当前版本槽。
GPU模型中的旧C在CPU计算及尚未发布期间仍被使用。CPU master更新完成与GPU模型C被替换不是同一时刻。

### 2.15 B双缓冲的实际状态机

`DoubleBufferedGradientAccumulator`固定创建两个slot，只有一个slot处于`accumulating`。
普通累积要求shape/dtype不变；每次合法调用将`accumulated_steps`加1。
边界由`HybridUpdateCoordinator.submit_boundary()`先加入当前B，再检查步数恰为N，然后freeze-and-swap。

owner Takeover主要经过：

```text
FREE
  → ACCUMULATING
  → FROZEN
  → UPDATING（输入D2H由prepare_function和event跟踪）
  → 输入完成安全转移
  → 清零梯度 / steps=0 / version=None / FREE
```

当前CUDA owner路径不能依赖`copying_to_cpu`标记：coordinator的该分支比较设备type为`"gpu"`，而PyTorch CUDA device.type为`"cuda"`，所以实际可从FROZEN直接进入UPDATING。DMA依赖由prepare函数和event保障，不由该枚举证明；本文记录现状，不修改实现。
enum中还存在`copying_to_gpu`、`ready_to_commit`及相应方法，但**不能把它们画成当前owner Takeover的必经链条**。
真实CPU输出与发布由coordinator的future/ready映射和CPU updater的version metadata、return event管理，GPU周期buffer不必一直占用到H2D结束。

实现中`updating`是slot所有权状态，不表示此时CPUAdam已开始执行。
`_reset_slot()`原地zero已有梯度，不删除`slot.gradients`里的张量，所以FREE只表示可复用。
没有free slot时抛出错误；runtime通过等待最旧version的反压尽量避免进入这种非法提交。

### 2.16 B/C提交：版本输入如何冻结

`Zero2OwnerCpuUpdater.submit_boundary()`与coordinator分工如下：

1. 校验/准备B/C optimizer state及本次Native staging version。
2. 固定本次`dense_scale`。
3. 从Adapter读取的live LR由runtime显式传入，固定为job LR。
4. Coordinator将当前边界B加入active accumulator。
5. 检查累计步数等于N，分配新的version，freeze-and-swap。
6. `_prepare_gradients()`将冻结B和C准备成CPU输入。
7. 保存source引用，提交一个协调线程future。
8. CPU updater保存该version的参数列、owner bucket与发布metadata；`_metadata_only()`不保留原owner梯度值。

`HybridUpdateJob`保存version、buffer ID、interval steps、B/C映射、event和LR。
`dataclass(frozen=True)`只禁止字段重新绑定，不会让其中Tensor底层存储自动不可修改；真实安全性来自槽所有权和event协议。

### 2.17 CPU梯度槽、工作区与返回镜像的布局

#### 2.17.1 大owner路径何时启用

`finalize_state_initialization()`统计本rank B/C selected state元素数；达到`1_000_000`时走Native spawned CPUAdam。
这是一条当前代码阈值，不是“所有模型都一定有CPU子进程”。

#### 2.17.2 常驻对象

设本owner B/C总元素数为`M`，lag为`L`，模型传输元素字节数为`s`：

| 对象 | 数量 | 近似数据字节 | 是否shared | 是否显式host registered |
|---|---:|---:|---|---|
| FP32 master | 1 | `4M` | 是 | 不是本注册集合 |
| FP32 exp_avg | 1 | `4M` | 是 | 不是本注册集合 |
| FP32 exp_avg_sq | 1 | `4M` | 是 | 不是本注册集合 |
| 版本梯度槽 | L | `L*s*M` | 是 | 是 |
| FP32 gradient workspace | 1 | `4M` | 是 | 不是本注册集合 |
| return mirror | 1 | `s*M` | 是 | 是 |

B周期buffer、CPU RS本地结果/packed临时空间、Native遗留状态、框架开销等不在上表中。
因此上表不是整个进程CPU内存峰值公式。

#### 2.17.3 槽内排列

```text
CPU version gradient slot:
[ B(parameter_id升序) | C/dense(parameter_id升序) ]

CPU FP32 workspace / master / moments / return mirror:
与上面对应的同一flat key顺序与offset
```

`_native_offsets[(parameter_id, tag)]`给出每段offset/length。
B与C使用不同tag，因此同一个矩阵的两组状态不会冲突。
零长度owner条目仍保留布局一致性。

#### 2.17.4 槽与version绑定

```text
slot(version) = (version - 1) % max_async_lag
```

`prepare_native_boundary()`检查该slot的旧version已安全commit后，才允许新边界占用。
槽不是因为“CPUAdam貌似读完了”就可以任意提前覆盖；当前规则与提交进度及backpressure共同约束。

### 2.18 大owner输入准备、mean与精度顺序

`_prepare_gradients()`先校验B/C布局与初始化flat keys一致。

- B：从冻结的周期buffer复制到当前CPU版本槽前缀。
- C：若Native已direct-stage，不再复制一遍；否则从传入的dense梯度准备C区间。
- 必要时将source cast为槽dtype。
- copy stream先等待producer stream；准备完成后记录transfer event。
- 返回flat槽、B元素数metadata、event及需保活的source引用。

协调线程`_run_update()`：

```text
等待transfer event
  → 删除transfer-source保活引用
  → 释放/清零旧B周期slot
  → _update(job)
```

`_update(job)`在CPU版本槽上执行：

```text
B前缀 /= interval_steps          # mean配置时
C后缀 /= dense_scale             # 若Native阶段尚未做最终缩放
```

然后才进入子进程的FP32 workspace转换与CPUAdam。
所以主BF16配置下，大owner实际顺序是：

```text
B周期和（输入dtype或CPU RS的FP32）
  → cast/copy到BF16版本槽
  → 在CPU上的BF16槽执行mean
  → copy/cast到CPU FP32 workspace
  → FP32 Adam
```

不能把它写成“FP32 mean之后一次性量化”，也不能假定CPU FP32累积自动消除了全部后续BF16舍入。
小owner路径没有这一套flat槽，使用CPU clone后mean，其dtype行为要另看输入。

### 2.19 CPUAdam进程与顺序执行

实现入口：`_run_native_cpu_adam_process()`。

1. 主进程将FP32 master/moments flatten为shared storage；各参数状态重新绑定成flat storage的view，避免继续维护独立的旧拷贝作为另一套authority。`flatten_states()`要求参与的B/C状态具有相同初始Adam step，否则报错；一个flat CPUAdam调用不能暗中代表不同step计数的独立优化器。
2. 使用`multiprocessing.get_context("spawn")`创建CPU进程。
3. 建立容量为1的command queue与result queue，以及ready event。
4. 子进程构造`DeepSpeedCPUAdam(adamw_mode=True)`，直接绑定共享moments和初始step。
5. 收到`("step", slot, lr)`后设置真实optimizer LR。
6. 将版本槽copy/cast到FP32 gradient workspace，设置`parameter.grad`。
7. 调用一次flat CPUAdam step，更新canonical FP32 master/moments。
8. 清掉parameter.grad，把更新后的master cast/copy到唯一return mirror。
9. 返回已完成的optimizer step计数，供主进程同步状态metadata。

协调器本身另有一个`ThreadPoolExecutor(max_workers=1)`。线程负责等待D2H、归一化、发送命令和接收结果，真正的大块CPUAdam在spawned进程中。
因此“有异步线程”与“优化器计算是独立进程”同时成立，不能只看其中一个类推断全貌。

进程启动等待有显式超时与失败处理；每次发送Native更新命令前检查进程存活。结果接收使用阻塞queue读取，不能据此宣称已经有完整的运行中心跳、任务超时或worker崩溃恢复协议。任意故障位置恢复/事务回滚不是当前已完成能力。

### 2.20 CPU affinity与小owner执行路径

CPU affinity根据当前允许核集合、physical core数、`LOCAL_WORLD_SIZE/LOCAL_RANK`划分rank资源，再按`pt_reserved_cores_perc`划分训练侧与worker侧。
协调线程和CPUAdam进程可使用worker核集合。

作用是降低本作业内线程争用，但这不是完整NUMA memory placement，也不是根据实时带宽自适应线程数。
CPU B主线程归约/累加也会消耗训练侧CPU资源，不能假定它免费使用独立后台核。

小owner路径：

- 不创建上述Native flat共享槽与spawned CPUAdam进程。
- 将冻结输入clone到CPU，按配置mean。
- 在协调线程内调用`SelectedColumnAdamW`更新CPU state并返回结果映射。
- 数值语义测试可使用它，但它的性能、结果存储和复用方式不能替代大owner shared-return-mirror协议的验证。

### 2.21 Future、ready、publication与return buffer保护

#### 2.21.1 任务状态

```text
提交job
  → _futures[version]
  → future完成，progress()取得结果
  → _ready[version]
  → 全rank对同一个next version都ready
  → commit一个version
  → 从ready删除，committed_version推进
```

`pending_updates`包括future和ready结果；CPU算完但还未发布的结果仍然算pending。
`future.result()`传播执行异常，不把失败结果标记为成功ready。

#### 2.21.2 全rank一致性

Adapter用ready flag的跨rankMIN判断共同ready。
owner updater每次只commit一个共同的`committed_version+1`，不是某rank将本地所有ready一次性发布。
等待drain时循环执行相同规则，以免不同rank的CPU完成速度影响collective序列。

#### 2.21.3 大路径返回镜像为何不会提前被覆盖

`_native_step()`在发送下一CPU命令前执行两道保护：

1. 前一version至少已进入按序publication，不允许CPU无限向前覆盖唯一镜像。
2. 若保存了`_return_visibility_event`，等待它完成，确认GPU不再读取前一返回镜像。

CPU返回结果映射中的各tensor是mirror的view，不是每个version一份独立完整参数拷贝。
当前owner updater设置`clone_update_results=False`以避免额外大复制，所以这两道保护是必要条件，不是可选优化。

`L`个输入槽不表示同时运行`L`个CPUAdam，也不表示有`L`份返回镜像。当前只有一个协调worker、一个Native CPUAdam进程和一份return mirror。下一CPU命令还受上一版本publication/visibility约束，这会限制CPU推进速度，但GPU可在容量允许时继续产生下一周期梯度。

### 2.22 H2D、All-Gather、scatter与下一次forward

大owner B/C发布的实际调用：

```text
CPU return mirror的owner views
  → values.to(device=GPU, non_blocking=True)
  → owner updated-value All-Gather
  → adapter.scatter_parameter_columns()
  → 在当前publication stream记录visibility event
  → 下一次forward的stream等待event
```

CPU侧B/C先commit B，再commit C，按排序后的group、bucket和参数metadata组织。
`scatter_parameter_columns()`对二维参数按列写；非二维参数经flatten视图写回。

**stream细节：**当前C/D2H准备使用`_transfer_stream`并等待producer；B/C发布函数中的H2D、AG、scatter按其当前调用stream执行。
不能仅因为updater拥有一个transfer stream，就宣称所有H2D也已显式放到那条独立stream。

`_visibility_event`用于下一forward的`wait_event`；`_return_visibility_event`用于CPU复用mirror前的等待。
两个引用可以指向同一event，但保护的消费者不同。
`wait_event`建立GPU执行依赖，不必然阻塞host；checkpoint/close则使用必要的event synchronize。

`prepare_forward()`在有pending时还会做必要的accelerator同步再检查ready；Native边界A发布后也有同步。
因此实际实现保留安全同步，不是“所有普通step零等待”的理想流水线。

### 2.23 全路径dtype账本

| 环节 | 当前实际规则 | 不应误写成 |
|---|---|---|
| 模型权重 | 使用模型dtype，主要验证BF16 | 模型全部FP32 |
| 普通packed A/B | 继承GEMM输出；GAS累积保持dtype | 强制FP32梯度 |
| GPU RS | packed输入dtype决定，当前没有独立传输dtype开关 | 所有通信统一BF16或统一FP32 |
| GPU B周期累积 | 保持输入dtype | 必定FP32 |
| GPU RS后的CPU B周期累积 | `.to(cpu)`不改dtype，保持输入 | CPU就自动FP32 |
| CPU RS前B D2H scratch | 显式BF16 | 随任意模型dtype自由切换 |
| CPU RS与该路线B周期累积 | CPU FP32 | BF16 CPU归约 |
| norm统计 | FP32；overflow flag为整数 | 改变所有梯度存储为FP32 |
| Native大owner B/C梯度槽 | 首个相关模型参数dtype，BF16配置为BF16 | CPU FP32累积后永不再量化 |
| Native CPUAdam工作区、master、moments | FP32 | 低精度优化器状态 |
| 大owner CPU返回镜像和对应发布值 | 模型dtype | 所有返回都FP32 |
| A更新值AG | 当前从FP32 master返回FP32值 | A也固定BF16 AG |
| 最终scatter模型 | cast成目标参数dtype | 改变模型dtype |

独立控制gradient D2H和parameter H2D dtype、并做完整精度消融，仍是未完成项。

### 2.24 Overflow与异常：哪些东西清，哪些保留

| 状态/对象 | 当前step overflow时 | 理由 |
|---|---|---|
| 当前A梯度/更新 | 不执行本次A更新，当前batch消费后结束 | 不写入坏梯度 |
| 当前B输入 | 不加入周期buffer | 周期仅包含有效step |
| 已有效累积的B | 保留 | 不丢弃此前成功step |
| 当前C staging | 等待写入安全结束后撤销本次staging所有权 | 防止复用仍在DMA的槽 |
| Hybrid成功步数 | 不推进 | 下次仍处于相同逻辑周期位置 |
| 此前已提交有效CPU版本 | 按原version/LR继续处理 | 当前坏step不自动否定旧有效任务 |
| Scheduler | 遵循DeepSpeed成功optimizer step规则 | 不将overflow当作有效更新 |

源码中通用辅助runtime还存在主动`discard_active_gradients()`/`handle_overflow()`入口，它们不能替代上表所述生产Takeover的当前step分支。
特别是独立`HybridUpdateRuntime.handle_overflow()`会主动清active buffer，不能把该辅助类行为误写成所有Takeover overflow都清整个B周期。

尚无完整“CPU已经更新master后，任意撤销该version并恢复master/moments”的事务协议。
异常关闭也不能承诺正常checkpoint的可恢复性；需保留失败证据而不是静默当作成功训练。

### 2.25 学习率在实际调用链中的生命周期

```text
DeepSpeed scheduler
  → Native optimizer.param_groups[*]["lr"]
  → adapter.get_learning_rate()
  → runtime当前成功step
       ├─ GPU A step(..., lr=lr)
       └─ 若边界：submit_boundary(..., lr=lr)
             → HybridUpdateJob.lr
             → 等D2H / 排队
             → ("step", slot, lr)
             → CPUAdam.param_groups[0]["lr"]
```

Adapter要求所有group LR相同、有限且非负。初始化还要求betas、eps和weight decay一致。
coordinator等待传输后重新构造job时保留LR；小owner`_step()`也显式接收job LR。

构造时的`_lr`或`_native_hyperparameters`仅可作为独立调用默认值，不用于证明当前scheduler已生效。
审计应读取实际GPU step参数、job和CPU命令。

LR=0时SelectedAdamW仍推进step/moments，weight decay乘子为1，参数更新量为0。
周期B不按早期各步LR加权，C不乘除周期，changing LR也不额外强制清空有效队列。

### 2.26 Checkpoint与恢复的实际调用链

ZeRO state字典中加入`fastoffload_hybrid_state`，controller调用Takeover `state_dict()`。

保存内容包括：

- B归约设备标记。
- pipeline成功步计数与importance registry。
- owner GPU A optimizer state。
- owner CPU B/C optimizer state与committed version。
- active B累积步数及对应CPU序列化梯度值。

CPU updater保存之前：

```text
while pending_updates:
    等待所有rank相同next version ready
    commit一个version
等待最终visibility
序列化canonical state和active B
```

**不保存仍在DMA中的slot让恢复后接着读，也不把队列中的任务再次重放。**
已经提交的任务在保存前完成，因此没有需要恢复的queued LR；后续LR由恢复后的Native optimizer/scheduler控制。

恢复时：

1. 检查checkpoint中的CPU/GPU B归约位置，旧checkpoint无字段时按GPU解释。
2. 拒绝存在pending update或活跃累积的冲突恢复。
3. 恢复pipeline、registry和A/B/C状态。
4. 恢复committed/submitted起点与active B。
5. 按需要重新初始化Native大owner执行设施。

`load_active_state_dict()`允许替换已清零的缓存allocation；仅有allocation不表示有有效未处理梯度。
不允许中途改变布局或把CPU-RS checkpoint静默恢复到GPU-RS。

**验证边界：**小模型部分周期恢复及继续执行已有覆盖；真实LLM、大owner完整optimizer-state resume仍未完成验证。
不能把“state_dict有这些字段”“模型导出成功”或“两个rank参数一致”单独当作完整resume证明。

### 2.27 正常关闭与存储的最终寿命

主关闭顺序：

1. CPU updater按共同version drain已提交任务。
2. 关闭协调executor。
3. 等待最终GPU visibility。
4. 向Native CPU进程发送`close`并join；不正常退出会报错并执行必要的子进程清理。
5. 对显式注册的梯度槽/return mirror调用`cudaHostUnregister`。
6. Pipeline丢弃microbatch状态和CPU B scratch引用。
7. CPU RS路线销毁私有Gloo group。
8. 移除Takeover专用parameter capture/dense切换属性。
9. Selector清理reference，Observer输出最后不足一个报告周期的统计。

注销mapping、join进程或把slot标记FREE，不意味着所有Python张量马上归还OS/GPU allocator。
部分optimizer数组、清零周期buffer和索引缓存仍可能被runtime对象引用；真正物理回收取决于后续对象销毁与allocator策略。
正常close也不会为尾部不足N步的B额外提交C。

### 2.28 shared+pinned注册与一次性恢复

`_register_transfer_buffers()`把每个版本梯度槽和return mirror作为完整shared allocation注册一次。
不对同一DMA可能跨越的存储切成独立注册slice。

遇到`cudaErrorInvalidValue`对应数值1时：

1. 同步CUDA，先暴露可能更早的异步错误。
2. 读取并清理预期的last error；不同错误不伪装成注册问题。
3. 旧PyTorch缺少对应绑定时，从PyTorch已加载扩展解析相同runtime入口，不另找任意toolkit库。
4. 保持失败映射存活，同时分配新的shared storage，减少立即复用同一地址的可能。
5. 对新allocation仅重试一次。
6. 成功后将替代槽同时接入主进程slot表与spawned进程参数。
7. 再失败、非预期错误或同步失败则抛错，并注销此前已成功注册的mapping。

这只是一种初始化阶段的有界恢复，不证明底层驱动问题已定位，也不是训练失败自动重跑。
健康路径不额外同步或重新分配；重试时可能短暂多占一个CPU allocation。

### 2.29 CPU Gloo group的生命周期与兼容性

`Zero2ObserverAdapter.create_cpu_owner_collectives()`先检查所有相关DP group成员和顺序是否等于完整world。
通过检查后才创建一个私有Gloo group，并供各参数组的CPU owner collective复用。

```text
配置/运行时合法性检查
  → 全rank一致创建Gloo group
  → 重复执行CPU B RS及必要CPU统计归约
  → drain CPU任务与发布
  → destroy_process_group(cpu_group)
```

`deepspeed.comm.new_group(ranks, backend=None)`保留旧调用方式，显式backend时才向下传递；对应Torch backend同样支持参数。
不使用直接导入torch distributed的旁路。
当前子组/model-parallel创建顺序没有完整全局规划，因此明确拒绝，不能声称任意并行组合都支持。

### 2.30 基础同步/异步D2H组件：与Takeover的关系

这些组件属于FastOffload，但不是上面B/C版本流水线的别名。
API对Hybrid要求`mode=observe`，不会同时自动把`OverlapScheduler`套到owner updater的每次B复制上。
Policy与Scheduler通过`OffloadDecision`解耦，但当前公开配置中的实际policy是`AllOffloadPolicy`，为收到的归约后本地shard返回offload动作。接口分层不等于已实现任意层级优先级、自适应负载决策或更多可配置policy。

#### 2.30.1 同步直接D2H

Adapter提供owner gradient view与最终CPU partition destination。
`SynchronousTransferEngine`用blocking copy直接写destination；可选CPU staging pool则先写pinned lease，再由InlineWorker复制到最终目标。
它不改变Native每步optimizer语义，也不是完整同步Hybrid参考。

#### 2.30.2 producer-stream异步D2H

在producer stream提交nonblocking copy、记录copy完成event。
任务保留source view直到完成；stream顺序保护后续对源存储的复用。
默认不额外分配GPU staging，降低额外copy和显存，但不能因此自动声称copy与同stream backward kernel并发。

#### 2.30.3 dedicated-stream异步D2H

```text
producer stream把source复制到GPU staging lease
  → source-ready event
  → dedicated stream等待该event
  → D2H到最终destination或CPU staging lease
  → copy-complete event
  → worker消费（若启用CPU staging）
  → 释放lease与event bundle
```

GPU staging保护Native源bucket复用，但增加GPU存储和一次copy。
独立copy stream不是无条件更快，需要实际测量。

#### 2.30.4 Pool与FIFO反压

Pinned pool的lease状态：

```text
FREE → RESERVED → COPY_IN_FLIGHT → FILLED → IN_USE（若worker消费）→ FREE
```

slot懒分配并按需要扩容，FREE后保留allocation，close且无active lease时才释放pool持有并调用unpin。
GPU staging pool和event pool也通过lease/完成事件控制复用。

`OverlapScheduler`同时限制task数和inflight字节；空队列允许一个oversized shard单独进入，避免永远无法满足阈值。
`progress()`只推进已完成任务；`prepare_step()`在Native optimizer读取CPU梯度之前flush剩余任务。
producer/dedicated路径的计时和隐藏比例属于这个Transfer Engine，不自动覆盖全部Takeover操作。

### 2.31 Shadow与独立Hybrid辅助执行器

#### 2.31.1 HybridCompressedCollectiveShadow

Shadow保留Native实际更新，同时额外对普通步A/B或边界完整矩阵列做压缩All-Reduce验证。
它比较：

- Native归约值与压缩值。
- owner offset与CPU FP32 partition中的对应值。
- 有限性/overflow决定。
- 对应梯度norm。

捕获值先进入有界参数桶，完成Native/owner两类校验后删除相关临时记录；不长期保存整模型packed副本。
Shadow可能主动同步并增加CPU/GPU数据访问，是验证模式，不是仅metadata的Observer，也不是Takeover性能路径。

#### 2.31.2 HybridUpdateRuntime

这是接受预归约压缩梯度的独立执行器，使用SelectedColumnAdamW与coordinator演示A/B/C调度。
它不负责完整ZeRO owner通信，也不是当前API装配的主Takeover runtime。
其overflow、checkpoint、commit及small-state行为不能逐字套用到`Zero2TakeoverRuntime`；生产生命周期以本章前述owner实现为准。

### 2.32 Observer、Telemetry与指标生命周期

纯Observer的Context只持有稳定metadata，不长期持有梯度Tensor。
可记录参数/桶事件、host阶段耗时、当前和峰值allocated/reserved、潜在offload字节等。
debug事件环有固定容量，默认关闭；JSONL/CSV按rank-local策略和报告周期输出，close输出最后一个部分窗口。

主Takeover另外记录或更新：

- owner bucket数量与packed payload/peak gauge。
- Native dense boundary次数。
- CPU D2H等待host耗时。
- CPU updater host耗时。
- host registration重试次数。
- pinned allocation累计计数/字节。
- `takeover_cpu_B_d2h_bytes`：该rank经CPU-RS专用BF16 staging的实际B元素字节数。
- `takeover_cpu_B_reduce_scatter_buckets`：CPU B归约桶数。

解释限制：

1. `takeover_cpu_adam_host_ms`覆盖调用路径中的等待/通信等host时间，不必然等于纯CPUAdam kernel时间。
2. CPU-RS字节计数为0不表示GPU-RS/CPU-accumulation没有D2H。
3. Owner payload gauge不等于全部H2D/D2H流量，尤其A AG与梯度RS可能dtype不同。
4. 累计pinned allocation不是当前live residency；unregister也不是自动从累计计数扣除。
5. Reporter窗口可能reset计数，外部审计需使用明确的累计/窗口口径。
6. 通用Observer的`device_timing`和`distributed_summary`仍有未消费的预留配置；不能因为Transfer Engine有event计时就宣称全系统细粒度计时已完成。
7. 重叠阶段耗时不可简单相加得出训练总时长。

`failure_policy`支持raise、disable_observer、warn，用于相应回调处理；Takeover核心更新不能依靠“忽略异常”证明正确。
方法验证和正式路径应优先保持默认raise，避免把部分失效的选择/观测状态误认为正常执行。

### 2.33 已实现技术点、作用与代价对照

| 技术点 | 实际作用 | 仍存在的代价/边界 |
|---|---|---|
| Native预热后的列重要性 | 有依据地区分高频/低频参数 | 一次性快照与选择开销，固定mask启发式 |
| CPU reference + GPU chunk评分 | 限制比较GPU scratch | 预热前仍有整份CPU reference |
| A/B/C差异化频率 | 降低普通步与CPU optimizer工作量 | 改变优化数学，不是无损调度 |
| Packed SelectiveLinear | 缩减dW GEMM，避免dense zero-fill/scatter | gather/临时tensor成本；不缩减dense forward/dX |
| shape阈值fallback | 小shape避免稀疏路径反而更慢 | 不等于自动成本模型 |
| 压缩GAS求和 | 保持已缩放loss的正确梯度尺度 | 不同路径的GAS设施不同 |
| 紧凑owner布局 | 减少重复optimizer状态与全量owner metadata | 特殊布局并未全部支持 |
| 确定性owner RS/AG | 分摊梯度/更新并保持通信序列一致 | padding、桶临时分配和实际dtype成本 |
| Native密集边界流式C | 不额外收集整模型GPU C梯度 | 边界仍有完整backward和Native通信 |
| GPU B周期累积 | 避免每步B下传/CPU累加 | 两个GPU周期buffer可能常驻 |
| GPU RS + CPU累积 | 下传已分片B、移除GPU周期buffer | 每步同步复制/CPU加法；不强制FP32 |
| CPU RS + CPU FP32累积 | 真实把B归约与累积移到CPU | 每步未归约B下传；当前Gloo/D2H同步 |
| 独立spawned CPUAdam | 减少训练线程直接承受大块optimizer调用 | 共享内存、进程与host带宽成本 |
| Flat master/moments/workspace | 减少逐参数调度和额外状态拷贝 | 大块CPU内存仍常驻 |
| 模型dtype版本槽/返回镜像 | 降低传输与共享槽字节 | cast和量化，A AG仍可能FP32 |
| 双周期buffer + 版本梯度槽 | 防止新旧梯度覆盖，可提前复用累积器 | FREE不等于allocation释放 |
| live LR + submission LR | scheduler对GPU/CPU任务语义一致 | 不支持任意组间/非LR调度 |
| 全局numerics与CPU/GPU标量合并 | 一致overflow/clip，不搬回整份CPU B | 统计与同步不是零成本 |
| 共同ready、顺序commit、visibility | 避免版本分叉与H2D返回覆盖 | 有显式同步和反压等待 |
| drain + partial B checkpoint | 保存有效进度，不虚构尾部C | LLM大owner完整resume待验证 |
| 完整shared注册与单次新存储重试 | 提升初始化可靠性 | 不代表驱动原因完全解决或任意失败可恢复 |
| Observer/Shadow/基础D2H模式 | 支持分层验证与诊断 | 不是所有模式都运行同一A/B/C主路径 |

### 2.34 验证状态与不得扩大的结论

当前CPU B改动相关测试集合已通过118项CPU测试、21项GPU测试，包含真实CPU collective、空/不均匀owner、Native B masking、混合numerics、实际spawned CPUAdam、提交LR、CPU-B-only overflow、部分B恢复与进程/group清理。
真实模型短检查也已覆盖CPU RS路线；但通过这些检查不等于所有规模、所有队列时序、所有布局都已验证。

以下仍不是当前已完成能力：

- CPU B的完全异步分块接收/归约流水线。
- 消除Native密集GPU桶中的零B占位。
- CPU RS的GAS>1、DP subgroup/model-parallel路由。
- 任意queued/running CPUAdam的撤销与状态回滚。
- 大owner/真实LLM完整optimizer-state resume。
- 不同optimizer authority、不同group超参数与非LR调度。
- 全部unsupported dtype/layout的bucket-local fallback。
- 独立梯度/参数传输dtype控制及完整精度消融。
- 通用persistent packed GPU workspace和完整碎片化模型。
- 全阶段传输带宽、exposed等待、CPU PSS/live-pinned峰值的完备观测。
- 在线自适应重要性、周期和并发度。
- ZeRO-3、NVMe及任意复杂并行组合。

当前实现中的CPU core划分不构成完整NUMA优化；小模型checkpoint通过不构成大模型resume证明；压缩有效元素比例也不是实际GPU通信字节或端到端加速倍率。
本文不把未完成项写进已实现收益。

### 2.35 代码阅读索引与最终核对清单

建议按以下顺序对应本文阅读：

1. [`config.py`](../../config.py)、[`api.py`](../../api.py)：哪些配置能真正装配到运行时。
2. [`importance/selector.py`](../../importance/selector.py)、[`importance/pretrained_delta.py`](../../importance/pretrained_delta.py)：reference到固定mask。
3. [`importance/selective_linear.py`](../../importance/selective_linear.py)：dense forward/dX与packed dW。
4. [`controller.py`](../../controller.py)、[`adapters/zero2.py`](../../adapters/zero2.py)：接管时点与Native私有布局隔离。
5. [`hybrid/takeover.py`](../../hybrid/takeover.py)、[`hybrid/microbatch.py`](../../hybrid/microbatch.py)：本地capture、GAS、CPU B staging与RS。
6. [`hybrid/partition.py`](../../hybrid/partition.py)、[`hybrid/owner_collective.py`](../../hybrid/owner_collective.py)：owner计数、padding、RS与AG。
7. [`hybrid/takeover_runtime.py`](../../hybrid/takeover_runtime.py)、[`hybrid/numerics.py`](../../hybrid/numerics.py)：本步数值决定、Native边界与forward安全点。
8. [`hybrid/owner_update.py`](../../hybrid/owner_update.py)、[`hybrid/selected_adam.py`](../../hybrid/selected_adam.py)：A权威状态、实际更新值dtype。
9. [`hybrid/buffer.py`](../../hybrid/buffer.py)、[`hybrid/coordinator.py`](../../hybrid/coordinator.py)：周期输入所有权、future与ready队列。
10. [`hybrid/owner_cpu_update.py`](../../hybrid/owner_cpu_update.py)：版本槽、C direct staging、CPUAdam、return mirror和checkpoint drain。
11. [`transfer/asynchronous.py`](../../transfer/asynchronous.py)、[`schedulers/overlap.py`](../../schedulers/overlap.py)：独立基础D2H框架，而不是CPU B流水线的自动实现。
12. [`hybrid/shadow.py`](../../hybrid/shadow.py)、[`telemetry/observer.py`](../../telemetry/observer.py)：数值验证与metadata观测的不同职责。

最后用六个问题检查任一张FastOffload流程图是否准确：

- 画的是local gradient、owner gradient，还是更新后的parameter value？
- “累积”指同backward、GAS，还是B跨成功step周期？
- CPU张量是普通allocation、pinned scratch、shared版本槽，还是canonical FP32 state？
- 当前event保护source读取、CPU输入完成，还是GPU返回值可见？
- 该buffer是逻辑FREE可复用，还是底层存储真的被释放？
- 写的是方法理想公式、当前生产大owner路径，还是辅助验证/小模型路径？

只要这六点没有区分，图中即使都标着“GPU/CPU、A/B/C、async”，也不足以描述当前FastOffload的实现。
