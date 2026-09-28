<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- DeepSpeed Team -->

# 系统论文：模型与数据集选择、实验矩阵

本文是 `SYSTEMS_PAPER_EXPERIMENT_PLAN.md` 的具体工作负载计划。下面是建议方案，不代表这些训练或兼容性测试已经完成。先确认矩阵，再实现数据适配和启动长实验。

## 1. 推荐结论

**主实验用两种架构：Qwen2.5-7B-Instruct、Mistral-7B-Instruct-v0.1；三类训练任务：Alpaca、GSM8K、XSum。**

另新增同家族 Qwen2.5-1.5B-Instruct / 14B-Instruct 做规模点，具体取决于下载与计算预算。现有 Qwen3-8B 可作第三架构补充，但它不是“大模型容量扩展”的替代。

不做所有模型×所有数据×所有长度×所有GPU数的全组合。采用4个主workload + 独立规模/扩展矩阵，控制成本。

## 2. 本地资源核查

已检查目录与所列模型的config；**目录存在不代表权重完整、许可证已核对或训练兼容性已通过**。

| 模型/资源 | 当前本地状态 | 建议 |
|---|---|---|
| `/data/Qwen2.5-7B-Instruct` | config为Qwen2ForCausalLM；已有三方法pilot | 主模型 |
| `/data/Mistral-7B-Instruct-v0.1` | config为MistralForCausalLM | 第二主架构，先做compatibility smoke |
| `/data/Mistral-7B-v0.3` | base模型，MistralForCausalLM | 可替代Mistral主模型，但不能与Instruct结果混用 |
| `/data/Qwen3-8B` | config为Qwen3ForCausalLM | 可选第三架构；固定thinking模式 |
| `/data/Qwen3___5-4B`、`/data/Qwen3.5-9B` | config为Qwen3_5ForConditionalGeneration | 不直接作为纯文本decoder-only主基线，需另审模型路径 |
| Qwen2.5-1.5B-Instruct / 14B-Instruct | 本轮未发现对应本地目录 | 规模实验建议新增，先确认下载与许可证 |
| Alpaca | `/data/hangyu/datasets/alpaca`，已有训练 | 主指令任务 |
| GSM8K | `/data/hangyu/datasets/gsm8k_dataset`，存在train/test | 主数学任务 |
| XSum | `/data/hangyu/datasets/xsum_dataset`，存在train/validation/test | 主长输入摘要任务 |
| MMLU | `/data/hangyu/datasets/mmlu_dataset` | 仅能力保持评价，先核实版本/schema |
| OpenBookQA | `/data/hangyu/datasets/openbookqa_dataset` | 可选辅助评价，不作主训练任务 |

不要为了“新模型”优先引入VL、多模态、MoE、embedding或尚未验证的新算子路径；当前贡献范围是dense text causal LM的ZeRO-2 CPU offload。

## 3. 核心主实验矩阵

每行均比较 Native ZeRO-2 CPU Offload、ZenFlow、FastOffload。共同数据顺序/预算，方法native实际GAS与A/B/C语义显式披露。

| ID | 模型 | 训练数据 | Cutoff | 主GPU数 | 目的 | 最终质量 |
|---|---|---|---:|---:|---|---|
| W1 | Qwen2.5-7B-Instruct | Alpaca | 512 | 2 | 延续pilot、短序列offload基准 | held-out response NLL；辅助能力保持 |
| W2 | Mistral-7B-Instruct-v0.1 | 同一Alpaca split | 512 | 2 | 跨架构、hidden/vocab形状变化 | 同模型内三方法NLL对比 |
| W3 | Qwen2.5-7B-Instruct | GSM8K | 1024 | 2 | 有明确正确答案的任务质量 | 官方test答案exact match、NLL |
| W4 | Qwen2.5-7B-Instruct | XSum | 2048 | 2，若统一OOM则4 | 长输入、activation与带宽权衡 | ROUGE-1/2/L、NLL |

W4的主GPU数在试运行后对所有方法统一锁定；固定2GPU容量实验仍应保留OOM结果，不因基线OOM只给某方法增加GPU。

优先完成W1；通过质量门槛后做W2/W3/W4。W1三seed是最小起点，最终用于核心质量主张的workload应尽量都做3seed；单seed结果只能标探索性，不宣称统计稳定。

说明：Alpaca适合系统负载，但单靠Alpaca train loss不能证明SFT质量；W3和W4提供下游任务指标。不同模型tokenizer的NLL/token不可直接跨模型横向比高低，质量比较只在同一模型、同一任务内进行。

## 4. 数据划分与预处理

### D1：Alpaca

- 原始约52,002条；以实际加载与清洗后的数量为准。
- 首先规范化文本、去除完全重复记录，对近重复划分风险做检查；按稳定ID划分，不能先随机划分再让重复答案泄漏。
- 建议train约50,000、validation约1,000、test约1,000；小余数处理记录在manifest，不承诺清洗后仍恰好52,002。
- 为所有1/2/4/8卡、interval=4、microbatch=1的选定实验使训练样本数可整除32。例如50,000向下取49,984，其余记录明确保留/不使用，不移入test进行事后挑选。
- 现有无held-out的完整epoch是pilot；正式split后需要重跑，不把旧结果混入正式表。
- W1/W2使用相同原始样本ID split和顺序；各自tokenizer导致token数不同，分别报告。
- 采用固定Alpaca prompt可延续现有实现；所有方法一致。若改成model-native chat template，三方法重跑，不能只改变某个方法。
- 只监督response与终止标记，不监督prompt/pad。

### D2：GSM8K

- 标准版本通常有7,473 train、1,319 test；本地版本/数量/schema须核实。
- 从官方train固定划出约512 validation；test完全不参与调参。
- 训练输入是问题，目标是数据集提供的解答及最终答案；保留最终答案标记（标准数据通常为`####`）。
- 必须报告cutoff导致解答/最终答案丢失的比例；必要时将cutoff升至2048，对三方法统一修改。
- 初始预算3 epochs；每epoch所有方法消费相同样本。若按GPU/interval对齐减小train，保存准确ID，不修改官方test。
- 测试使用greedy decoding，统一max_new_tokens（先建议512，pilot检查截断率后锁定）、统一答案提取脚本与数值规范化。
- 主指标最终数值答案exact match；不以生成文本字面完全匹配评分，不使用多次采样的best-of-N作为单次准确率。
- Qwen-Instruct可能接触过该任务；报告step-0准确率与训练增益，不能声称完全无预训练污染。

### D3：XSum

- 使用官方train/validation/test；常见原版约204k训练记录，实际本地数量与版本需核实。
- 为成本可控，主实验预先固定约49,984个train、1,024 validation、2,048 test，全部来自各自官方split，保存ID/hash。
- 输入document，目标summary；模板固定。不是把XSum强制改成Alpaca原始schema而丢失正文。
- cutoff2048：预留目标summary token预算（如128，先检查目标长度分布），优先截断document，避免长prompt把所有监督标签挤掉。
- train/eval使用相同固定截断策略；报告input长度、response长度、padding比例和被截断样本比例。
- 初始预算1 epoch；若质量仍在改善，可统一扩展到2 epochs，不能只延长某方法来隐藏速度成本。
- 解码建议greedy，max_new_tokens=128或经pilot锁定值，三方法完全一致。
- 指标ROUGE-1/2/L，固定分词/stemming/实现版本，声明返回尺度是0–1还是0–100；辅以NLL。
- 2048 cutoff不等于每个样本2048有效tokens，必须测实际分布。

### D4：MMLU / OpenBookQA（可选能力保持）

仅用于微调前后比较，不加入主训练。使用固定官方评测协议和few-shot样例，对三种方法使用相同prompt、答案打分逻辑、seed与预算。

MMLU报告全科目加权/宏平均具体定义；选择题可以使用候选答案likelihood评分以减小decode噪声。不能用test选择LR。它们提供能力保持信息，不代替Alpaca任务质量或无污染泛化证明。

## 5. 模型规模矩阵：与跨架构实验分开

### 推荐同家族规模点

| 点 | 模型 | 主要用途 | GPU建议 |
|---|---|---|---|
| S1 | Qwen2.5-1.5B-Instruct（待准备） | 正确性、低成本长训、CPU offload收益边界 | 1/2 |
| S2 | Qwen2.5-7B-Instruct（已有） | 主实验与扩展性 | 1/2/4/8 |
| S3 | Qwen2.5-14B-Instruct（待准备） | 大参数量offload价值与容量 | 2/4/8，OOM点保留 |

统一用Alpaca作为性能负载，cutoff512/2048，测稳态500–1000 microsteps或预注册等量tokens窗口。先不要求所有规模点3seed完整训练；质量主张限定在已测完整训练模型上。

若14B无法准备，可先投稿最小矩阵但明确大模型规模证据不足；不要用另一个7B替代14B并声称容量扩展。

### 可选第三架构

Qwen3-8B + Alpaca，cutoff512/2048，做500–1000 microsteps性能、compatibility检查。若做质量必须固定thinking/non-thinking模式、chat template、decode预算，先确认当前Transformers支持。它作为额外泛化结果，不挤占GSM8K/XSum质量实验预算。

## 6. 序列长度与batch敏感性

- 用Qwen2.5-7B + XSum或适当长文本数据测512/1024/2048/4096；先测实际有效长度分布。
- 相同原始document ID、不同cutoff会改变监督上下文，所以这是workload敏感性，不是数学等价训练消融。
- 不通过大量无效padding把Alpaca短样本伪装成长上下文任务。
- 固定长度synthetic tensor可用于kernel/transfer microbenchmark，但不得替代真实SFT端到端主结果。
- microbatch/GPU建议1/2/4做容量扫描；方法在相同配置下OOM如实标记。
- 质量实验保持选定batch预算；性能容量扫描允许batch变动，但图中说明。

## 7. 各模型先跑兼容性门槛

Mistral、Qwen3及新增模型均先完成：

1. 读取架构与权重完整性，记录revision/hash、license。
2. 检查Linear/embedding/output head/tied weight结构与importance coverage；不能默认所有模块都支持packed路径。
3. 2GPU、至少16–32 microsteps，跨多个dense boundaries，确认无deadlock/NaN。
4. Native、ZenFlow、FastOffload配置都能启动；实际GAS、offload路径与model dtype正确。
5. rank副本参数一致、CPU job drain、保存/重新加载后eval一致。
6. 覆盖率/unsupported模块比例写进workload表。若主要模块不支持，不能静默fallback后仍声称同样稀疏计算覆盖。

初始checkpoint precision、RMSNorm、GQA和模型家族差异均由真实模型实现处理；不为让某方法更快而单独替换attention或Linear实现。

## 8. 最小运行数量与预算控制

### 第一轮：用户确认后优先执行

1. W1加入固定validation、step-0 eval与准确计数。
2. 三方法constant LR：`2e-6 / 5e-6 / 1e-5`，每候选1000 microsteps，seed42，共9个短run。
3. 同预算选LR后，W1三方法各3seed完整训练，共9个长run。
4. Mistral smoke通过后做W2；并行完成GSM8K/XSum数据适配与step-0评估。

### 第二轮：主论文矩阵

- W1–W4 × 3方法 × 3seed = **36个质量run**；W1上述9个已包含，不重复计数。
- 其中GSM8K可能3epochs，不可把“run数”直接当等量GPU-hours。
- 独立性能重复：4 workloads × 3方法 × 3 repeats = **36个稳态run**；详细profiling另跑，避免eval频繁drain污染steady-state。
- scaling/消融只选W1或W4代表点，避免全矩阵扩展。

若预算不足，先保证W1+W3的3seed质量证据，W2/W4先做单seed探索和性能；正式文中明确其证据强度，后续再补齐。

## 9. 推荐主结果表结构

| Model / Dataset | Cutoff | Method | Actual GAS | Real tokens/s | GPU peak | Host peak | Val NLL | Task score | Time-to-target |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B / Alpaca | 512 | 三方法 | 实测 | 全rank | max rank | 去重口径 | response-token加权 | 可选 | 未达则NR |
| Mistral-7B / Alpaca | 512 | 三方法 | 实测 | 同上 | 同上 | 同上 | 同模型内比较 | 可选 | 同上 |
| Qwen2.5-7B / GSM8K | 1024 | 三方法 | 实测 | 同上 | 同上 | 同上 | NLL | Answer EM | 同上 |
| Qwen2.5-7B / XSum | 2048 | 三方法 | 实测 | 同上 | 同上 | 同上 | NLL | ROUGE-L等 | 同上 |

不要跨任务平均原始Loss；吞吐几何平均只对可比、预先选定且所有方法有结果的workload计算，OOM/未达质量点单独披露。

## 10. 需要用户确认的选择

- 是否同意以 **Qwen2.5-7B-Instruct + Mistral-7B-Instruct-v0.1** 作为两种主架构？
- 是否同意 **Alpaca + GSM8K + XSum** 三任务，先不用新增外部数据？
- 是否允许准备 **Qwen2.5-1.5B-Instruct / 14B-Instruct**，作为小/大规模点？
- 是否按W1优先、质量通过后再展开其余矩阵，避免低质量配置消耗大量长训预算？

推荐批准顺序：主矩阵先确认；新增模型下载与大型容量run单独确认。当前本文仅制定计划，不启动训练或下载。

## 11. 扩展核查：本地较大模型与修订建议

进一步扫描了 `/data/` 顶层模型config、权重index引用文件与部分safetensors header。这里只验证元数据/文件存在，不是权重checksum或实际训练兼容性认证。

### 优先选择：Gemma 2系列

| 本地模型 | Header中存储tensor元素 | 磁盘权重 | 当前dtype | 用途 |
|---|---:|---:|---|---|
| `/data/gemma-2-2b` | 约2.614B | 9.74 GiB | FP32 | 小规模点 |
| `/data/gemma-2-9b` | 约9.242B | 34.43 GiB | FP32 | 中规模/跨架构点 |
| `/data/gemma-2-27b` | 约27.227B | 101.43 GiB | FP32 | 首选大模型点 |

三者config为Gemma2ForCausalLM，所检查index没有缺失shard，存在tokenizer文件。存储tensor元素不一定等于独立可训练参数数，最终以实际模型去重统计为准。

**修订建议：无需优先新增Qwen 1.5B/14B下载，可先采用本地Gemma-2 2B/9B/27B作为同家族规模矩阵。** Qwen7B保留主任务与已有pilot，Mistral7B保留第二家族验证；成本紧张时Gemma9B可承担新增架构证据，减少重复7B workload，但主矩阵修改应在正式结果前冻结。

Gemma2需要检查attention softcapping、sliding-window实现、tied embedding/head、attention backend与ZenFlow selective模块覆盖。磁盘FP32并不意味着训练要用FP32；正式三方法保持相同BF16配置。不可仅为某方法更省初始化显存而改变加载dtype/路径，须统一验证与记录。

### 大模型GPU与内存规划

Gemma-2-27B约27.2B存储元素，BF16权重理论量级约50.7 GiB。ZeRO-2仍在每rank保留完整BF16模型副本，增加GPU主要减小owner optimizer/gradient状态，**不会把该模型副本除以GPU数**。

- 先在8×A800、microbatch/GPU=1、cutoff512、activation checkpointing上做三方法16–32 microsteps的兼容性和内存测试。
- 通过后扩展到500–1000 microsteps稳态，再考虑4GPU与cutoff1024/2048容量点。
- 2GPU只作可行性探索，不能预先保证fit；4/8GPU也须实测，临时workspace/activation可能触发OOM。
- 大模型主数据先用Alpaca；长序列容量用XSum。性能点无需全部完整epoch，若声称27B同质量加速则必须补对应质量训练。
- 分别记录初始化peak和steady training peak；全模型FP32加载、importance reference、CPU共享state可能带来很高host启动峰值，先核算内存再启动。

### 其他本地候选

| 模型 | 检查结果 | 当前判断 |
|---|---|---|
| `/data/Yi-9B-200K` | LlamaForCausalLM，约8.829B BF16元素，index无缺shard | 可用候选，架构较直接；不是20B+规模点 |
| `/data/Mistral-Small-3.1-24B-Instruct-2503` | 约24.011B BF16元素，Mistral3ForConditionalGeneration，带视觉结构 | 可选24B扩展；必须适配text-only加载/训练与统计scope，不是当前AutoModelForCausalLM runner即插即用 |
| `/data/Mixtral-8x7B-v0.1` | MoE，约87 GiB权重，index无缺shard | BF16全模型副本已超80GiB量级，不适合当前ZeRO-2完整副本路径；另有expert/unused-gradient语义工作 |
| `/data/Qwen2-57B-A14B` | MoE，约107 GiB权重 | active 14B不等于只需存14B；不适合当前范围 |
| `/data/Phi-3.5-MoE-instruct` | MoE，约78 GiB权重 | 副本几乎占满80GiB，无正常训练空间；不优先 |
| `/data/deepseek-moe-16b-base` | 自定义MoE结构，约30.5 GiB权重 | 容量可能可行，但需专门expert与remote-code适配，放后续 |
| `/data/gpt-oss-20b` | MoE + MXFP4量化，header含大量U8 packed tensors | 不是BF16全参数Adam训练直接基线，暂不选 |
| `/data/Llama-3.2-11B-Vision-Instruct` | 多模态Mllama | 不属于当前纯文本主范围 |

本次检查还发现：`Llama-2-7B-32K-Instruct`的index引用缺少`pytorch_model-00001-of-00002.bin`；`Qwen3.5-9B`缺少index引用的第1和第3个safetensors shard。它们不能仅凭目录存在就视为可加载模型。

### 建议最终覆盖

- **主质量**：Qwen2.5-7B + Alpaca/GSM8K/XSum；第二架构选Mistral7B或Gemma9B。
- **规模曲线**：Gemma-2 2B / 9B / 27B，同家族，同数据/长度协议。
- **大模型重点**：Gemma-2-27B，先8GPU smoke，再4/8GPU性能/容量。
- **可选**：Yi9B；Mistral24B仅在有适配预算时加入。

本轮核查未启动训练。

## 12. 用户确认的新增规模点

用户已确认同时加入以下两个规模点，不互相替代：

- **Qwen2.5-14B-Instruct**：官方仓库`Qwen/Qwen2.5-14B-Instruct`，目标目录`/data/Qwen2.5-14B-Instruct`。已启动下载，完成后检查index引用的全部权重分片；尚未完成训练兼容性验证。
- **Gemma-2-27B**：使用现有`/data/gemma-2-27b`，作为更大规模、不同架构的容量与性能点。

修订后的重点规模矩阵：Qwen2.5-7B/14B用于同家族对比，Gemma-2-27B用于大模型扩展；Gemma-2-2B/9B保留为可选同家族辅助点。

Qwen14B先做2/4GPU、cutoff512的三方法兼容性测试，再展开性能与质量；Gemma27B先做8GPU smoke，再探索4/8GPU容量。所有可行性以实际峰值和运行结果为准。

下载过程日志：`/tmp/download_qwen25_14b.log`；当前下载可使用同一snapshot_download脚本重试并复用已有缓存，不删除已完成分片。
