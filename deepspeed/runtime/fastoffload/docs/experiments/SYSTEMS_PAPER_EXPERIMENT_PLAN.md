<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- DeepSpeed Team -->

# FastOffload 系统论文实验设计与执行计划

## 1. 论文定位与核心证据链

建议定位：面向多 GPU ZeRO-2 CPU optimizer offload 的重要性感知混合更新系统。贡献包括 packed selective backward、owner-sharded A/B/C 更新、Native dense-boundary streaming、独立 CPUAdam process、BF16 双向传输与版本安全发布。

**不能只用 tokens/s 证明系统有效。** FastOffload 改变参数更新频率与梯度覆盖，和 Native/ZenFlow 不是数学等价的同一优化器。因此论文必须同时回答：

1. 同等模型、数据和硬件下，端到端速度与资源开销如何？
2. 达到同一验证质量需要多少时间？最终质量是否可接受？
3. 加速来自哪些机制，而不是更少训练数据、不匹配 GAS 或较差质量？
4. 增加 GPU 数、改变主机带宽和 batch 后还能否受益？
5. 异步版本、溢出、checkpoint、shutdown 是否正确？

推荐组织为三类结论，分别呈现，不能互相替代：

- **Method-native comparison**：各方法按原生算法运行，相同输入工作量，比较性能和质量。
- **Quality-matched comparison**：各方法在同等调参预算下达到相同预注册质量目标的时间。
- **Semantics-preserving ablation**：固定 FastOffload A/B/C 算法，隔离系统实现机制的收益。

不预先写“无损”“收敛等价”“线性扩展”等结论；由数据决定表述。

## 2. 当前证据与缺口

### 已有结果（仅作 pilot，不替代正式评测）

- Qwen2.5-7B-Instruct / Alpaca，2×A800，cutoff=512，1 epoch，LR=2e-5。
- FastOffload / ZenFlow：168.72 / 81.98 padded tokens/s，约2.06×。
- GPU allocated peak：30.74 / 36.65 GiB/rank。
- 排除20 microsteps后平均 online training loss：1.54921 / 1.48287；后半程差距扩大。
- LR=5e-6 的200-microstep pilot：三种方法后40步 rank-0 train loss 均约0.95–0.98。

### 必须修正的评测缺口

- 现有 Loss 是 rank-0、按 microbatch 等权的 online training loss，不是全 rank/token 加权验证指标。
- 200步短测只用于排查和调参，不能证明长期收敛。
- 不同方法 actual GAS、更新频率不同，不能按日志 step 横轴直接比较。
- 旧 benchmark 的 padded token 计数使用本 rank token 数乘 world size；动态 padding 下其他 rank 长度未必相同。正式实验必须累计所有 rank 的真实计数。
- 累计 pinned allocation 不是当前 pinned residency 或 peak pinned residency。
- 旧计时边界在最终 engine destroy/drain 之前，可能未包含全部异步尾部工作。正式端到端指标必须覆盖最终 drain/visibility。
- 不应从 changing-minibatch loss 趋势直接推断学习率过大或模型退化；需要固定验证集。
- 旧完整epoch末尾存在partial interval；正式主实验使用可整除样本数或已验证的partial flush。
- 现有完整epoch没有最终模型质量评测与多seed重复。
- 节点外部CPU/内存竞争未完全控制，不能作为最终性能结果的唯一依据。

## 3. 实验前必须落地的基础设施（Gate 0）

### 3.1 统一 runner 与配置快照

每次运行保存：CLI、解析后DeepSpeed配置、FastOffload配置、运行时actual GAS/global batch、optimizer各band LR、随机种子、数据顺序摘要、软件commit、模型revision、硬件拓扑、CPU affinity、NCCL/CUDA版本和环境变量白名单。

避免保存认证token。仓库dirty时保存必要patch或文件hash，不能仅记录commit。

启动前校验：FastOffload disabled的Native配置没有ZenFlow，ZenFlow的FastOffload takeover关闭。保存实际模式，不用`native_zero2`默认字符串掩盖ZenFlow运行身份。

### 3.2 质量评测

- 在训练前固定train/validation/test划分及样本ID，保存hash。
- 使用固定验证集，在`eval()`、`no_grad()`下计算有效response-token加权NLL。
- 跨rank累积`sum(NLL)`和`count(valid shifted labels)`，最终全局相除；忽略prompt/pad的`-100`标签。
- 确保分布式eval不因sampler补齐重复统计样本；样本只计一次。
- step 0先评测预训练模型，再按相同训练样本/token进度评测。
- 每次评测前安全drain异步更新并完成visibility；报告drain与eval开销。
- 固定prompt模板、EOS、truncation和label masking；检查有效response token数、空监督样本和模板边界。
- validation用于选择超参数；test只用于最终选定模型，不用于挑LR。

### 3.3 两套时间口径

**Steady-state throughput**：排除预注册warmup，统计固定microsteps/样本窗口。起止做必要GPU同步和跨rank协调，使用最慢rank elapsed。单独列出尾部drain，避免隐藏尚未完成的CPU工作。

**End-to-end / time-to-quality**：包括importance初始化、算法warmup、训练、相同频率eval、必要drain；明确是否包含模型加载/数据预处理/JIT编译。建议同时报告train-ready-to-finish与cold-start全程时间，首次编译单列。

统计：

- real input tokens/s（attention mask有效token）；
- supervised response tokens/s（shift后有效label）；
- padded tokens/s（所有rank实际tensor元素之和）；
- samples/s、microsteps/s；
- optimizer boundaries/s只作辅助。

计数按rank本地累计，在计时窗口外做一次归约，避免逐步统计collective干扰性能。

### 3.4 动态LR支持：未完成前禁止宣称scheduler实验

当前短测使用constant LR。若加入warmup+cosine，先验证：

- Native、A GPU Adam、B/C CPUAdam都能读取正确LR。
- B/C每个job在提交时冻结LR；排队或执行期间不被后续scheduler更新污染。
- schedule以公共样本/token训练进度对齐，不使用不同方法的原始logged step。
- checkpoint恢复进度和LR；overflow不错误推进成功更新调度。
- 小模型数值测试比较预期更新；动态LR和queued jobs专项测试通过。

在此之前可以用constant LR做主实验，三种方法使用同等LR搜索预算；无需为了论文强行增加未验证scheduler。

## 4. 公平对比协议

### 4.1 主基线

| 方法 | 角色 | 要求 |
|---|---|---|
| Native DeepSpeed ZeRO-2 CPU Offload | 必选标准基线 | FastOffload关闭；AdamW/精度/数据一致 |
| ZenFlow | 必选近邻方法 | 公布actual GAS、top-k、warmup与更新语义 |
| FastOffload | 本方法 | A/B/C配置、异步lag和transfer dtype明确 |
| GPU-only ZeRO-2或DDP | 可选性能参考 | 仅能fit时；注明不具备相同offload资源约束 |
| 其他CPU-offload系统 | 视投稿相关工作选择 | 仅加入可复现、支持目标硬件/训练模式的实现 |

相关工作调研后决定是否加入其他基线，不为凑数量比较不兼容的ZeRO-3推理/offload路径。

### 4.2 共同固定项

模型checkpoint、全参数训练范围、tokenizer/模板、训练样本顺序、cutoff、动态padding规则、activation checkpointing、precision、Adam betas/eps/weight decay、clipping、seed、GPU/CPU资源、eval频率、token预算均固定。

不同方法GAS不强行伪装相同：

- method-native保留ZenFlow内部GAS=interval并明确披露；
- compute-matched按相同microsteps/tokens比较；
- batch-matched为补充实验，需要重新确认FastOffload interval是按optimizer step还是microstep推进；
- 不能简单把三种方法CLI GAS全部写1就宣称实际GAS相同。

### 4.3 两种超参数公平性

1. **Shared-config**：固定同一组超参数，便于隔离系统行为。
2. **Equal-budget tuning**：各方法同样候选LR数、训练token预算、seed数，用validation挑最佳；展示各自最优质量/性能。

在方法native比较中不能用FastOffload调优结果对比未调优基线。短测速度不计入正式训练速度，但披露调参预算。

## 5. 实验矩阵与优先级

### E1：正确性和长稳（P0，性能前置）

小模型、可控张量，覆盖1/2/4 ranks：

- Native disabled路径对照参数、梯度、optimizer states。
- FastOffload对照**同A/B/C规则的同步参考实现**，不是要求与dense Adam数值相同。
- owner梯度归约、A发布、B interval mean、C boundary-only、unused参数、1D/dense-only、bucket跨owner与oversized参数。
- lag=1/2情况下顺序发布、buffer不提前复用、跨rank参数副本一致。
- 注入overflow：本次A/B/C不更新、不推进successful schedule；检查active累积和staging清理。
- 对已提交/运行job明确约定：历史成功step的合法job不自动等同于当前overflow job。若需要取消/失效语义，必须实现并单测后再宣称支持。
- 有pending jobs时checkpoint/drain、恢复后的参数和moments、最终导出与安全shutdown。
- 稳态压力测试至少1–2小时，检查GPU/host内存不持续增长、queue lag有界、无collective deadlock。

产物：正确性矩阵、数值误差容限及失败条件；不是只列“测试passed”。

### E2：小规模收敛与LR选择（P0）

首轮用Qwen2.5-7B-Instruct/Alpaca，2GPU，cutoff512：

- 方法3种；LR候选`2e-6, 5e-6, 1e-5`，`2e-5`保留历史参照或补同协议测试。
- constant LR先控制变量；候选每个1000–2000 microsteps，而非只200步。
- 固定validation 512条左右，step 0及每400 microsteps评测；具体规模用吞吐预估后锁定。
- 第一阶段seed42筛选；每种方法前两名用额外seed复验。
- 所有候选相同token预算，避免选择起步较快但长期退化的LR。
- 小模型可额外做固定小数据集重复训练的overfit sanity check，确认loss/梯度链路有效。

产物：validation NLL vs samples/time；LR选择表。200步pilot仅作参考。

### E3：端到端主结果（P0）

最小主矩阵：

| 维度 | 建议 |
|---|---|
| 模型 | 至少2种架构/家族，约3B与7–8B；模型须本地可用或许可证允许 |
| 数据 | Alpaca + 另一种较长响应/不同分布的公开SFT数据，固定split |
| GPU | 主结果2GPU；扩展性单列 |
| 序列 | 512和2048；显存不足按预注册规则记OOM |
| 方法 | Native / ZenFlow / FastOffload |
| 训练预算 | 同样训练examples/tokens；主质量结果至少完整1 epoch，必要时更长 |

为了控制成本，不必所有维度全笛卡尔积。建议4个代表workloads：小模型短序列、7B短序列、7B长序列、第二架构/第二数据集。

每workload记录吞吐、完成时间、allocated/reserved peak、host RSS/PSS峰值、pinned live峰值、最终validation/test指标。

模型权重/数据集具体名称在运行前锁定；不能用同一家族多个大小作为唯一的架构泛化证据。

### E4：同质量时间（P0，核心）

- 用独立pilot和预训练eval确定可解释的validation NLL目标，正式运行前锁定，不能事后挑最有利阈值。
- 用相同eval频率，画validation NLL vs elapsed time与vs consumed tokens两套图。
- 目标可定义为经调优Native质量的预注册容忍区间；同时展示完整轨迹，避免阈值依赖。
- 若方法未达到目标，标`Not reached within budget`；不得外推速度。
- 记录首次评测达到阈值的时间；评测间隔造成的时间区间明确说明。
- 至少3 training seeds；最终test NLL和至少一种适合任务的公开下游评价使用同一模板与预算。
- 若SFT使用开放生成评价，固定decoding参数，尽量使用可复现自动评分；LLM judge时报告版本、成本与潜在偏差。

不能把train loss均值接近视为quality-matched。若FastOffload最终质量落后，报告Pareto trade-off，而不是无损加速。

### E5：多GPU扩展性（P1，核心系统证据）

单节点1/2/4/8 GPU，主模型7B、cutoff512和一个较长序列点：

- **Weak scaling**：每GPU microbatch固定；报告tokens/s、吞吐/GPU、与1GPU相对效率。优化质量可能随global batch变化，不据此宣称相同收敛。
- **Strong scaling**：固定全局训练工作量和effective batch；通过合法GAS/微批组合配置。若ZenFlow耦合GAS使部分点不可比，标N/A并解释，不暗改interval。
- 每rank CPU core预算/总预算两种策略选主一种，另一种做敏感性；明确避免GPU变多时悄悄增加总host资源。
- 记录实际NUMA/NVLink/PCIe拓扑、cross-socket配置、CPUAdam时间、collective时间和rank straggler。

1GPU OOM是有效容量结果，不能省略；无法据1GPU求效率时用最小可运行卡数作基准并注明。

多节点不是当前必选：只有实现/硬件支持并在摘要声称跨节点扩展时才必须补充。

### E6：内存与最大可训练规模（P1）

固定硬件下测试模型大小/sequence/microbatch增长：

- 同配置allocated与reserved GPU peak；采集所有rank，报告max和分布。
- CPU训练进程+worker的RSS/PSS；shared memory不要简单sum RSS导致重复计数。
- pinned live/peak与cumulative allocation分别测量，统计coverage说明。
- A master/m/v、B accumulator、collective workspace、gradient slots和return镜像分项估计，与实测peak交叉核对。
- 按相同搜索流程找最大可行microbatch/sequence；OOM点明确画出。

主图建议：GPU/host资源双图，避免只展示省GPU而隐藏CPU增长。

### E7：关键系统消融（P1）

固定相同A/B/C选择、训练token窗口、超参数；每次只改一个机制。

| 消融 | 验证贡献 | 是否需先实现/验证 |
|---|---|---|
| 同步CPU job + 同步发布 vs async | CPUAdam overlap收益 | 安全同步开关/参考路径 |
| dense grad_weight后选列 vs packed backward | 避免普通dense weight-gradient GEMM | 保持相同gradient语义的受控开关 |
| FP32 vs BF16 D2H / return | 转移字节、数值和时间 | 独立dtype配置当前待补 |
| Native bucket streaming vs保留dense fragments | dense峰值优化 | 仅小模型安全对照，不能在7B强造OOM buffer |
| owner-sharded A状态 vs replicated reference | 状态/通信收益 | 小模型受控参考，语义相同 |
| version slot/lag配置 | buffer和overlap权衡 | 只能使用合法安全配置，禁止关闭安全保护作基线 |

不要把Observer sync/async模式直接当作Hybrid sync/async消融：它们接管范围不同。

不必恢复已删除的昂贵生产实现；小型受控reference/microbenchmark可证明对应机制。每个消融标注实现范围和是否保持优化轨迹。

### E8：算法参数敏感性（P1，与系统消融分开）

- interval：`1,2,4,8`（先验证interval=1语义；不等同Native自动成立）。
- A/B比例：示例`5/5,10/10,20/20`；C为余量。
- lag：`1,2,4`，先确认内存和版本支持。
- importance：pretrained delta与random同覆盖率作对照；周期刷新为未来机制，未实现前不列已支持。
- bucket：`32,64,128,256 MiB`，记录实际oversized/padded字节而非只看配置值。

避免全组合爆炸：代表workload one-factor-at-a-time初筛，挑3–5个Pareto点做完整质量验证。改变interval/覆盖率会改变训练算法，不能仅用速度判断最优。

### E9：系统瓶颈与重叠解释（P1）

独立profiling run，使用Nsight Systems/CUDA events与host时间线：

- ordinary vs boundary forward/backward、RS、A Adam、AG、B freeze、D2H、CPUAdam、H2D、publication wait。
- 区分issued work和exposed stall；重叠任务时间总和可以大于wall time，不能堆成不守恒的“耗时占比”。
- D2H/H2D分别记录payload bytes、active ms、effective GB/s；不要用CPU host wait推算PCIe利用率。
- 实际collective bytes和padding；owner imbalance、CPU job queue lag分布、p50/p95 boundary latency。
- importance初始化时间与整轮训练摊销占比。

产物：一张可读trace、critical-path分解、普通/边界延迟CDF。详细telemetry开关对吞吐影响单测；正式性能run只开必要采集。

### E10：资源敏感性（P2，加分项）

固定GPU配置，调整允许使用的CPU cores/NUMA绑定；有真实设备时比较不同host interconnect。报告CPU带宽/NUMA放置对吞吐与lag的影响。

不建议通过后台竞争任务模拟正式弱硬件并宣称普适结果。若做干扰实验，作为单独robustness测试，清楚说明干扰负载。

## 6. 推荐论文图表（约8图+3表）

1. **System overview**：多GPU owner-sharded路径与CPU异步更新。
2. **Main throughput**：多workload grouped bars，附最终质量或对应表。
3. **Time-to-quality**：validation NLL vs时间；旁图vs tokens。
4. **Memory trade-off**：GPU peak与host/pinned footprint，OOM/capacity。
5. **Scaling**：1/2/4/8GPU weak/strong scaling。
6. **Mechanism ablations**：语义不变的速度/内存消融。
7. **Timeline**：ordinary/boundary、CPUAdam重叠与发布等待。
8. **Sensitivity Pareto**：质量–吞吐，点颜色区分interval/覆盖率。

表：硬件软件环境与workload；runtime actual配置；最终quality/seed统计及正确性覆盖。

正文篇幅有限时把完整LR sweep、额外trace、boundary C语义验证、更多OOM点放附录，但主质量结果不可只放附录。

## 7. 重复、统计与失败处理

- 性能：每点至少3次独立进程重复，exclusive node，顺序随机或交替，固定预热长度。报告各次值、均值和标准差；3次CI较弱，避免过度解释小差异。
- 质量：关键主workload至少3 seeds，不把同一seed的3次测速当3次独立收敛实验。
- 运行顺序按方法轮换，记录CPU load、温度/时钟和其他GPU占用。
- 性能窗口建议至少500–1000 microsteps并覆盖足够boundary；具体用pilot验证稳态后预注册。
- OOM、deadlock、NaN、未达目标均保留记录，不能只挑成功/最快run。
- 异常节点干扰的排除规则事先写明，保留原始结果和rerun原因。
- 主张速度提升时用相同工作量时间比或全rank token吞吐比；不要用Fast step / Zen logged step。

## 8. 分阶段执行与计算预算

### Phase A：一到数天工程准备

实现真实token/内存/时间计数、validation、final drain/export、runtime配置快照；通过小模型正确性测试。此阶段不启动大规模长训。

### Phase B：调参与小规模机制验证

7B、2GPU，3方法×3LR×1000–2000 microsteps；按同等预算筛选。Native慢，可以先统一1000步，不能只缩短Native预算。并行跑在不同GPU对会共享CPU/内存，调参可容忍但不作正式性能结果。

### Phase C：主质量实验

先选一个代表workload，3方法×3seeds，完成预算与eval。若质量明显不达标，优先修正算法/数值语义，暂停扩展大型性能矩阵。

现有pilot大致速度仅用于预算：200 microsteps Native约17分钟、FastOffload约7分钟、ZenFlow约10分钟（含启动）；全epoch Fast约9.2小时、Zen约18.9小时。Native整轮时间需实测，不能把短run外推当结果。

预算公式：`GPU-hours = Σ(wall hours × allocated GPU count)`；独占8卡节点只用2卡时同时记录实际节点占用小时。验证、checkpoint和初始化另计，预留20–30%失败/重跑预算。

### Phase D：性能、扩展与消融

质量通过后再跑多workload短稳态性能、1/2/4/8卡扩展、机制消融与profiling。代表点做长稳，其余点无需全部完整epoch。

### Phase E：冻结与复现

冻结代码/配置/数据split，抽查重新clone后能运行。生成图表的脚本从原始JSONL自动计算，避免手工粘贴数字。

## 9. 最小投稿实验包与扩展包

### 最小但可信的包

- 3基线，至少2个模型家族/架构，4个代表workload。
- 主workload完整训练与3seeds、固定validation/test、time-to-quality。
- 所有workload端到端吞吐与GPU/host内存。
- 1/2/4/8 GPU扩展或明确scope受限的可运行点。
- 至少3项关键系统机制消融。
- 一张真实异步timeline和一组interval/coverage Pareto。
- 异步正确性、checkpoint/overflow/长期稳定证据。

### 资源允许再补

更大模型、长上下文、第二种主机平台、多节点（若支持）、更多任务与seeds、能耗。能耗只有在可可靠测量GPU和CPU能量时报告，不能仅GPU功率代表整机能效。

## 10. 产物目录与复现清单

建议所有正式结果存持久目录，不放`/tmp`：

```text
benchmark_results/paper_v1/
  protocol.md
  environment/
  splits/
  tuning/<method>/<lr>/<seed>/
  quality/<workload>/<method>/<seed>/
  performance/<workload>/<method>/<repeat>/
  scaling/<gpu_count>/<method>/<repeat>/
  ablations/<variant>/<repeat>/
  profiles/
  figures/
  summary.json
```

每run包含manifest、resolved configs、train/eval JSONL、benchmark、memory、stdout/stderr、exit status、checkpoint路径/hash（如适用）。大权重与原始trace放独立存储；Git仅提交必要脚本、配置、汇总小表与图，不提交认证、模型权重或海量日志。

## 11. 下一步具体任务（按顺序）

1. 给`finetune_alpaca.py`加入固定validation和跨rank token-weighted NLL，先验算小数据。
2. 修正token计数、计时末尾drain和内存口径；记录实际GAS与各band LR。
3. 保存/恢复时验证最终模型已包含pending B/C更新。
4. 采用constant LR做1000步的3方法×3LR验证实验，起始候选5e-6但不预设它最佳。
5. 每种方法用validation选配置，预算一致；预注册完整训练质量阈值。
6. 跑代表workload完整训练，多seed，评估是否能支持质量可接受的加速主张。
7. 再展开多模型、扩展、消融与trace。
8. scheduler作为独立工程任务，只有A与版本owned CPU LR传递/恢复测试完成后才加入候选配置。

**决策原则：先测量正确，再验证质量，再扩大性能证据。FastOffload的核心主张应是可量化的性能–质量–资源收益，而不是仅凭稀疏计算更快就推断训练更好。**
