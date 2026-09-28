<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- DeepSpeed Team -->

# 主性能实验设计（当前执行：200步双卡短测）

本方案最初响应用户“先设计主实验”，前8节保留设计依据。用户随后批准先执行双卡可运行的主性能场景，并把预算改为200步；当前执行以第9节为准。质量评估与调参暂停，历史消融/长测及其冻结文件不改写。本文不沿用旧方案中的三seed、质量通过后才能启动性能或自动下载新模型等建议。

## 1. 主实验回答什么

在相同硬件、模型、输入工作量和精度下，完整FastOffload相对Native ZeRO-2 CPU Offload及明确版本的ZenFlow，能否提高持续训练吞吐？显存代价是多少？收益在另一模型架构、不同真实输入分布和更大模型上是否仍出现？

主实验是端到端系统比较，不隔离单个机制，不证明同质量训练加速，也不以“所有方法数学等价”为前提。机制归因交给独立消融与profile；参数/GPU数量变化交给敏感性/扩展实验。

## 2. 核心矩阵：固定2GPU、每卡microbatch1

每行比较Native / FastOffload / ZenFlow三种方法，不做模型×数据×长度的全笛卡尔积。

| ID | 模型 | 真实训练数据 | cutoff | GPU | 主要观察 |
| --- | --- | --- | ---: | ---: | --- |
| W1 | Qwen2.5-7B-Instruct | Alpaca | 512 | 2 | 已测锚点：短输入offload场景 |
| W2 | Mistral-7B-Instruct-v0.1 | 相同Alpaca原始划分 | 512 | 2 | 第二架构及hidden/vocab形状覆盖 |
| W3 | Qwen2.5-7B-Instruct | GSM8K | 1024 | 2 | 问题/完整解答负载，不评测答案准确率 |
| W4 | Qwen2.5-7B-Instruct | XSum | 2048 | 2 | 真实长文档/摘要，计算与激活占比变化 |

cutoff不是实际平均长度；每行必须报告输入/回答长度分布、padding及截断比例。W3/W4之间不作单独序列长度因果归因。禁止把Alpaca大量补pad当作长上下文结果。

核心主表共12个目标结果。W1已有RUN074–076三项验收结果，先保留为已测锚点；W2–W4缺9项性能及9个对应gate。W1不能以消融控制RUN093单独替换RUN075并继续沿用旧Native/ZenFlow分母。

### W1来源复用边界

W1原三方法成组结果与新实验的输入预算/性能配方一致，但早于当前LR接线和新审计版本。复用时必须逐组标明源码/审计版本，不混入重复均值。若最终要求整张主表严格统一为一个新源码版本，应另行冻结完整三方法W1版本复验组，而不是默认重跑或只替换最快的一根柱。本方案不自动启动该复验。

## 3. 大模型扩展：与2GPU核心表分组

| ID | 模型/数据 | cutoff | GPU | 执行边界 |
| --- | --- | ---: | ---: | --- |
| L1 | Qwen2.5-14B-Instruct / Alpaca | 512 | 4 | 三方法容量gate，通过后再安排完整性能窗口 |
| L2 | Gemma-2-27B / Alpaca | 512 | 8 | 探索性容量gate；完整FO/ZF有历史风险 |

7B/2GPU、14B/4GPU、27B/8GPU是不同资源规模点，不是固定资源比较或GPU强扩展曲线。不承诺所有方法成功，失败/unsupported保留且吞吐留空。

ZenFlow主线仍关闭初始化修复、selective offload=false，并记录实际版本；14B该版本曾初始化OOM。若需要修复版完整三方法性能，应单列“ZenFlow + initialization fix”扩展组，不能混进未修复主基线或暗开offload=true。此处只列设计，不新增该版本授权或启动。

W4若2GPU失败，保留原2GPU结果；另建4GPU比较组时三方法统一资源，不只给失败方法加卡。受外部占用干扰的OOM必须记录当时资源/失败阶段，不能直接解释为模型的固有容量上限。

## 4. 固定方法配置

| 方法 | 实际GAS | GPU更新 | CPU更新 | 2GPU/microbatch1的框架batch |
| --- | ---: | --- | --- | ---: |
| Native ZeRO-Offload | 1 | 无独立选中更新 | 每microstep全参数更新 | 2 |
| FastOffload | 1 | A每成功步 | B/C每4成功Hybrid步，前置1步Native importance warmup | 2 |
| ZenFlow | 4 | selected部分每microstep | 每4microsteps | 8 |

这是method-native、相同输入工作量的比较，不是相同effective batch/梯度缩放/优化器数学。不得改Native GAS来人为对齐表格。

共同：BF16、全参数、ZeRO-2 CPU optimizer offload、pin/overlap、activation checkpointing、use_cache=false、动态padding到8、response-only、不跨样本packing；同组相同attention backend与数据处理。

性能配方固定原W1：constant LR5e-6、AdamW betas(.9,.95)、eps1e-8、weight decay.1、clip1、无scheduler。不是最新RUN088质量调参配方，不按方法分别调LR。FO固定A/B/C约10/10/80%、GPU B mean累积、周期4、lag2、128MiB桶、packed selective backward；ZenFlow固定top-k10%、interval4、warmup0、overlap、reserved cores比例.25、selective offload=false。

主表先固定microbatch1，避免把批次优化与方法收益混在一起。每卡2/4另作单因素batch性能表，保持各方法GAS不变；不在主表中悄悄使用各方法不同的“最大能跑batch”，也不把更大batch的样本预算当作原预算。

## 5. 数据及来源冻结

- W1沿用当前冻结split与Sampler seed42/epoch0，不重排样本。W2沿用原始记录划分，但用Mistral自己的tokenizer，冻结其过滤结果及实际token计数。
- GSM8K使用官方train派生训练划分，预留validation，test不参与本阶段；保留完整解答及最终答案标记，截断不得产生无监督样本。明确记录实际保留/过滤ID。
- XSum使用官方train的固定子集，预留summary监督预算、优先截断document；在正式测量前冻结模板、ID、过滤/截断策略，不能根据性能或loss更换样本。
- 每组冻结模型权重/config/tokenizer SHA256、数据版本/ID、实际每rank消费清单、生产源码快照、initializer版本、optimizer配置、启动命令及软件版本。
- Mistral本地只有`.bin`分片，当前训练环境PyTorch2.5.1受到Transformers安全加载检查限制。先获得经验证的safetensors，或用兼容的隔离环境做安全转换；不得关闭安全检查、把weights_only改为false、暗换模型或升级整个基准训练环境。

## 6. Gate与正式预算

每个新模型/数据/方法组合先做36 microsteps gate，跨多个dense边界。校验：

1. 实际BF16、GAS、offload、模型参数形状和路径覆盖；不静默更换后端/算法。
2. 真实输入/监督/padding计数、全rank消费ID、有限loss；loss仅作数值健全性检查，不作质量结论。
3. 实际Native/GPU A/CPU B-C或ZenFlow更新次数；保持B含当前边界梯度，不补造末尾C。
4. 已提交工作drain、完整rank副本一致、worker退出0及注册缓冲释放。

通过后每配置仅1次、seed42：1,200 microsteps，排除前200，测量后1,000。两卡每卡microbatch1时共消费2,400训练样本，窗口2,000样本；不同tokenizer/任务的token量据实统计。窗口含FO/ZF各250个CPU周期，Native含1,000次CPU更新。

若gate发现200步不足以进入稳定阶段，须在正式性能运行前冻结该组统一的新warmup，不在结果出来后裁剪最快窗口。同组方法串行，不并发启动本项目其他CPU-offload任务争抢CPU/内存带宽。外部竞争仍按观察-only政策记录，而非空闲门禁或自动停训条件。

性能运行无窗口内保存/验证，也不做终点模型导出/质量评估；保留上述完整正确性与清理审计。不从既有model-only导出resume。失败不自动重试，不删除历史模型/日志，不以未完成benchmark JSON填成功值。

## 7. 指标与主表

主指标：所有rank非padding input tokens总和 / MAX-rank测量窗口秒数。supervised/padded tokens另外报告，不能混用。加速比分别报告FO相对Native和ZenFlow；仅在同一完整组内计算。

主表字段：

| 工作负载 | GPU | 方法/版本 | 状态 | input tok/s | FastOffload/Native或FastOffload/ZenFlow | allocated GiB | reserved GiB | window s | loop+drain s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 每个workload每方法一行 | 实际值 | 源码/initializer明确 | 完成/OOM/失败/未运行 | 仅已验收 | 同组分母 | max rank | max rank | 排除warmup | 独立时间口径 |

时间必须分开：窗口排除最后drain；window+tail包含最后drain但不保证零队列起点；完整loop+drain包含warmup、不含加载/初始化/终点审计/teardown。端到端进程wall可另列，不能代替训练循环时间。

为兼容已有W1，主allocated/reserved峰值定义为engine初始化之后、含warmup至循环结束的最大rank值；不当作冷启动总峰值。NVML冷启动峰值、进程树PSS和详细时间线另做资源/profile运行，不把周期GPU进程查询当作内存采样，不用累计pinned申请量充当live peak。

只有单seed/单次结果，不画统计误差条，不推断稳定置信区间。先逐组展示原始结果，不通过删除OOM/失败组生成看起来更好的总体均值。无质量评估时不能声称同质量加速或收敛收益。

## 8. 推荐图与执行顺序

主图：四个核心workload的input吞吐分组柱状图；附FO/Native与FO/ZenFlow比值和实际GAS脚注。
第二图：相同workload/硬件的allocated峰值；reserved可作附表，不与NVML混画为同一指标。
大模型扩展表：L1/L2逐方法状态、失败阶段及成功者吞吐/显存，不从7B外推。

建议按现有结果核对 → W3适配/gate/三方法性能 → W4适配/gate/三方法性能推进；W2在安全权重加载准备好后完成第二架构组。L1/L2作为后续资源批次，不要求等待所有质量实验。当前消融队列保持原状，本文件本身不启动以上工作。

## 9. 用户批准执行：双卡主性能200步（2026-09-14）

用户要求“先跑2卡可以跑的，只跑性能，200个step就够”。已新建并冻结：
`/data/hangyu/ResearchHub/FastOffload/benchmark_results/workbook_main200_20260914_145637/`。
本次授权包含W1三方法的新短预算组，不把旧1200步或消融中的单独FastOffload结果拼入新分母。旧长测与RUN089–096八项已完成消融/gate全部保留。

| 场景 | 新数据gate（36/20） | 主性能（200/40） | 状态/边界 |
| --- | --- | --- | --- |
| W1 Qwen7B/Alpaca512 | 复用已验收W1三方法兼容性与相同输入前缀 | RUN097–099 | Native→FastOffload→ZenFlow |
| W3 Qwen7B/GSM8K1024 | RUN100–102 | RUN103–105 | FastOffload→ZenFlow→Native |
| W4 Qwen7B/XSum2048 | RUN106–108 | RUN109–111 | ZenFlow→Native→FastOffload |
| W2 Mistral7B/Alpaca512 | 未运行 | 未运行 | 本地只有bin分片，安全加载前提未满足，不绕过检查、不换模型 |

每项性能固定前40步排除、后160步计时，单seed42/单次、GPU6/7/world2/microbatch1；Native/FastOffload/ZenFlow保持GAS1/1/4和CPU周期1/4/4。FO继续完整A1/B-C4，原W1 constant5e-6/Adam(.9,.95)/wd.1不变。共9项200步性能、6项新数据gate，gate失败仅跳过对应长测，不自动重试。
W1训练集49,920；GSM8K从官方7,473 train按question-group hash预留512 validation后对齐保留6,960；XSum取官方train前50,000，过滤/对齐后保留49,994。新输入保留完整解答/摘要和EOS及提示后缀，仅优先截断question/document body；所有保留/过滤ID、模板与token文件在启动前冻结。未加载test，未tokenize/evaluate预留validation，原数据缓存不改动。
本批400个实际样本的平均/max输入长度：Alpaca约101/341，GSM8K约190/444，XSum约539/2048；不把cutoff当作平均有效长度。

12项CPU检查通过，覆盖响应/EOS/提示后缀保留、问题分组、collator及三类token计数、实际方法更新节奏、观察故障不停止、历史工作簿/公式/质量数据保持。初次预检误指P0目录下不存在的dataset hash manifest，改为已完成消融组中的原始manifest；初次单测把ZenFlow配置层级误写为顶层，改为zero_optimization.zenflow。初始日志均保留，修正发生在训练启动前，不是重试训练。
ENV11、CFG077–091、RUN097–111已登记原工作簿 `30_双卡主性能200步`，监督队列已启动。没有新质量评估、快照或模型导出；drain、全参数副本、实际更新与清理审计仍必需。显存仍按post-init含warmup口径，160步只是短性能窗口，不宣称长稳或与历史长测合并。外部进程/观察失败/时长警告不终止训练；4/8卡大模型、batch扫描、Mistral安全转换均未在此队列启动。
