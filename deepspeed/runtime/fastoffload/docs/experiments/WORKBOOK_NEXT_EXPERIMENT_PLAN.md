<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- DeepSpeed Team -->

# 按原Excel推进的实验计划（当前：性能与消融优先）

## 1. 依据与范围

依据 `assets/FastOffload_实验数据填写模板.xlsx` 的 `00_填写指南`，按P0主性能/质量、P1消融/容量/分解、P2扩展性执行。
主配置以 `19_Native默认频率` 的RUN071–RUN073、CFG051–CFG053为准，不以早先Native GAS4的批次代替。
用户已批准启动第一批W1的6个训练run。其余场景仍按分批验收推进，不自动下载模型或改变算法；原Excel历史结果保留。

按用户最新要求：**统一seed=42，每个实验配置只运行一次，不做性能重复或多seed质量实验。**
下文不同训练预算、兼容性gate和独立profile属于不同实验目的，不是同配置的统计重复。
中止后的重试必须另行确认；用户已于2026-09-12先后批准Native质量补跑RUN080及第3次尝试RUN081，均从基座重新起跑，RUN079/080中止记录保留。
旧计划中的“14B尚未准备”等描述是历史状态；Qwen14B目前已下载并完成过实际训练。旧P0 GAS4质量结果不并入本配置结果。

## 2. 冻结配置

| 方法 | 实际GAS | GPU更新 | CPU更新 | 备注 |
| --- | ---: | --- | --- | --- |
| Native ZeRO-Offload |1 | 无独立选中更新 | 每microstep全参数更新 | 保留原Native更新节奏，不改成GAS4 |
| FastOffload |1 | A每成功microstep | B/C每4成功microsteps提交 | 完整Hybrid A/B/C，不使用GPU ceiling或selected-only替代 |
| ZenFlow |4 | 选中部分每microstep | 每4microsteps提交 | 关闭初始化修复，selective offload=false |

“默认Native”指已经确认的基线配置与更新节奏，不代表所有超参数均采用框架出厂默认值。
两种选择性方法只对齐更新周期，不要求其GAS、优化器数学或启动相位相同。
FastOffload保留1步Native importance预热；B累积普通步和当前边界梯度，C只用当前边界梯度。
停止时只drain已经提交的工作，不为最后不完整B区间补造一次C更新。

共同固定：

- ZeRO-2 CPU optimizer offload，BF16，全参数微调；不切换ZeRO-3、量化或冻结参数。
- 每GPU microbatch=1作为主实验默认；LR5e-6 constant，无scheduler；AdamW betas=(0.9,0.95)、eps1e-8、weight decay0.1、clip1。
- activation checkpointing开启，use_cache=false，动态padding到8，response-only监督，无跨样本packing。
- FastOffload：A/B/C约10/10/80%，pretrained-delta选择，packed selective backward，GPU B累积，mean reduction，lag2，compressed bucket128MiB。
- ZenFlow：top-k10%，interval4，full warmup0，overlap开启，pt_reserved_cores_perc=.25；基础CPU optimizer offload保持开启。
- 保留当前FastOffload必要telemetry；主吞吐运行关闭外部PSS/NVML高频资源采样，详细profile另跑。
- 同组固定模型版本、数据ID顺序、tokenizer/模板、GPU集合、工作量、计时窗口；记录实际CPU线程和NUMA绑定。

模板写有“同组batch固定”，但用户已明确要求保留方法原生频率。本计划显式采用method-native比较：
在N卡、microbatch1时，Native/FastOffload有效batch=N，ZenFlow有效batch=4N。
配置表如实填写，比较组注明“同输入工作量、非同有效batch”，不再通过改Native GAS来满足旧批次的batch条件。

## 3. 模型、数据与场景矩阵

每行内部比较三种方法；不做所有模型×数据×长度×GPU的全组合。

| ID | 优先级 | 模型 | 数据/长度上限 | GPU | 场景与主要问题 |
| --- | --- | --- | --- | ---: | --- |
| W1 | P0，先做 | Qwen2.5-7B-Instruct | Alpaca /512 |2 | 主基准：稳态吞吐、资源、真实验证质量、长稳 |
| W2 | P0 | Mistral-7B-Instruct-v0.1 | Alpaca /512 |2 | 第二架构：收益是否依赖Qwen的层/vocab形状 |
| W3 | P0 | Qwen2.5-7B-Instruct | GSM8K /1024 |2 | 数学任务：除了NLL，能否保持最终答案准确率 |
| W4 | P0 | Qwen2.5-7B-Instruct | XSum /2048 |2 | 长输入摘要：激活/计算占比变化及ROUGE质量 |
| L1 | P1，容量先行 | Qwen2.5-14B-Instruct | Alpaca /512 |4 | 同家族规模点；初始化与完整训练容量 |
| L2 | P1，探索 | Gemma-2-27B | Alpaca /512 |8 | 更大模型、另一架构、容量极限与失败位置 |

以上模型和数据目录本地均存在。Mistral、GSM8K/XSum适配和当前27B完整路径仍需gate；目录存在不等于兼容性通过。
Gemma-2-9B可作为后续中间规模点，不自动替换失败的27B记录，也不作为首轮必跑项。
不优先新增MoE、多模态模型或模型下载任务。

### 大模型的已知限制

- Qwen14B在关闭初始化修复的ZenFlow上，历史上4/8GPU均出现过初始化workspace OOM。
  主线保持关闭修复：只做一次有明确版本的容量复验，复现相同失败则不继续启动相同的长run。
- 该OOM反映当前未修复初始化路径，不等于硬件在任何实现下都无法训练14B。
  如果后续需要完整三方法14B性能比较，须另行批准并单列“ZenFlow+初始化修复”扩展组，不暗开修复或offload=true。
- Gemma27B历史上Native成功，FastOffload/ZenFlow完整训练仍有内存/兼容性风险。GPU-only ceiling成功不算完整训练成功。
- 增加DP卡数不会减少ZeRO-2每卡完整模型权重。7B/2卡、14B/4卡、27B/8卡是不同资源规模点，不是强扩展曲线。
- W4先保持2GPU；OOM原样记录。若另设4GPU组，三方法都使用4GPU，不给某一个方法单独增加硬件。

## 4. 数据与质量协议

### Alpaca

沿用现有分组去重/冻结ID划分：train49,920、validation1,023、test1,024。
Qwen7B快速验证继续使用固定512条validation、29,706个shifted response tokens；最终质量可评完整validation和锁定test。
不重新切分来挑更有利样本。Mistral沿用原始记录ID划分，但使用自己的tokenizer；如产生无监督样本，先统一过滤并冻结新模型manifest。
只在同模型、同tokenizer、同验证子集内比较NLL，不横比不同模型的绝对token NLL。

### GSM8K

核实本地官方train/test版本，从train固定划validation，test不调参。
保留解答与最终答案；检查1024截断是否丢失最终答案，必要时在正式比较前统一调整长度并新建配置。
主指标为统一greedy decoding、统一答案提取/数值规范化后的answer exact match，模板中填写0–1；辅助response NLL。
报告基座成绩与训练增益，不声称排除了预训练污染。

### XSum

沿用官方split，预注册约5万训练样本、1,024 validation、2,048 test的固定子集，实际数量以清洗/分布式对齐后manifest为准。
优先截断document并预留summary预算，避免没有有效监督；记录真实输入长度、padding和截断比例。
主指标ROUGE-1/2/L，固定实现、尺度和解码参数；辅助response NLL。
不要用Alpaca的大量无效padding替代真实长文本。

## 5. P0：主性能与质量

### Gate：正式运行前

1. 参数化现有run-local runner的模型/数据输入，测试label masking、真实token计数、非重复评估分片与模型导出/reload。
2. 新模型/数据组合先各方法36 microsteps，跨多个CPU边界；检查实际GAS、A/B/C或选中更新调用次数、有限loss、覆盖率/回退模块。
3. 验证全rank副本一致、已提交CPU任务drain、模型导出和worker退出；失败不记成完整性能run。
4. 记录模型/源码hash、offload开关、CPU主/子进程affinity、默认allocator及资源快照。

W1已完成100步三方法gate；W2–W4各需3个新gate，共9次。L1/L2另有最多6次初始容量尝试。

### 单次性能实验

- 每个通过gate的W1–W4：**3方法各运行1次**，seed统一42。
- 每次**1,200 microsteps：前200排除，后1,000计时**。窗口覆盖250个选择性CPU周期；Native窗口执行1,000次CPU更新。
- 通过pilot确认200步已覆盖初始化；若不足，正式测量前统一调整warmup并冻结，不能事后裁剪最快窗口。
- 性能窗口内不插入validation或checkpoint；最终drain、导出/评估在窗口外记录。
- 三方法串行，不同workload轮换方法顺序，不在其他GPU同时跑CPU-offload任务污染同机内存带宽。
- 总计最多12个核心性能run，先执行W1的3个，不一次展开全部模型/任务。

### 真实质量与训练预算

- 第一批：W1三方法各用**seed42运行1次、4,096 microsteps**，BASE与终点普通HF重载评估，共3个质量pilot。
- 这是不同于1,200步性能实验的较长训练预算，不增加统计重复；4,096步仍不代表完整收敛。
- 质量无异常后：W1/W2建议1个完整epoch；W3建议3 epochs；W4建议冻结子集1 epoch。
  各组方法消费同样样本预算，均只用seed42、从同一基座起跑；不声称多seed稳定性。
- 周期性validation需先验证不会推进训练GAS/A/B/C调度或改变B累积、不会增加全模型GPU副本，并与终点HF重载结果交叉核验。
  该gate通过后才加入验证曲线和time-to-quality；训练online loss曲线不能代替它。
- time-to-quality目标从独立pilot确定后锁定；未达到目标记NR，不外推。
- 本轮保持LR5e-6，不自动调LR、scheduler、interval或选中比例。若将来做等预算调参，另列实验方案。
- 现有导出是model-only；4,096步pilot不能直接当作已验证的optimizer-resume checkpoint。完整质量训练可重新起跑，或先完成真实续训测试。

## 6. P1：容量、分解与独立消融

| 场景 | 代表工作负载 | 执行内容 | 边界 |
| --- | --- | --- | --- |
| 大模型容量 | L1/L2 |36步gate；通过者再做100步确认，之后才安排单次长窗口 | 每方法单独记录成功、初始化OOM、backward OOM、超时/unsupported |
| 内存容量扫描 | Qwen7B + XSum | 固定2GPU，长度512/2048/4096；另组microbatch1/2/4 | 每次只改长度或microbatch，不改更新频率 |
| 普通/边界分解 | W1，必要时W4 | 预热后约64步专用profile；D2H/H2D字节、活动时间、CPUAdam、RS/AG、pack/cast、等待与lag | profile运行不混入主吞吐平均 |
| 资源占用 | W1及成功的大模型点 | 单独资源运行，采集初始化/稳态/边界/导出时点GPU与进程树PSS | PSS成本和共享页口径明确；未实现pinned-live峰值则留空 |
| 系统消融候选 | W1 | packed selective backward开启/关闭；compressed bucket64/128/256MiB | 单因素、新CFG；先同语义正确性gate，不替换默认配置 |

消融是模板要求的独立变体，不是把主实验配置改掉。性能/消融按用户2026-09-13的新授权优先单列推进，见第14节。
CPU累积、独立FP32/BF16传输、同步Hybrid等路径，未经对应完整Takeover验证不排作已支持的开关。
不把Observer sync/async当Hybrid消融，不关闭版本/visibility保护，也不为填表临时增加算法。

## 7. P2：扩展与敏感性

- **GPU weak scaling**：Qwen7B/Alpaca512，1/2/4/8GPU，每卡microbatch1，Native/FastOffload GAS1、ZenFlow GAS4不变。
  记录总吞吐、每卡吞吐、峰值显存和相对最小可行卡数的效率；global batch随卡数变化，如实说明。
- 所有点固定同一总CPU core预算，按rank划分/记录实际策略，不随GPU数悄悄增加总CPU资源。
- **长度敏感性**：Qwen7B/XSum，固定2GPU，cutoff512/2048/4096，避免把空padding当真实长上下文。
- **microbatch敏感性**：固定模型/长度/GPU，microbatch1/2/4；GAS与CPU周期保持不变。
- **可选strong scaling**：固定全局microbatch8，通过1/2/4/8GPU对应每GPU microbatch8/4/2/1实现，不修改GAS。
  单个方法内effective batch固定，但方法间仍不同；OOM点保留。该组不属于第一批必跑。
- interval1/2/4/8、A/B比例、LR、ZenFlow状态offload等会改变已冻结配置，本轮不自动扫描。

## 8. 计量、填表与失败规则

| 原模板工作表 | 本计划填入内容 |
| --- | --- |
|01_环境 | 拓扑、软件、CPU预算/NUMA、环境白名单；实际带宽未知则不编造 |
|02_配置 | 模型/数据版本、实际GAS/batch、更新周期、初始化版本；每种变化建新CFG |
|03_运行结果 | 一次独立运行一行；全rank有效input tokens、MAX-rank时间/显存、状态和日志 |
|04_质量评估 | BASE/各checkpoint的NLL、EM或ROUGE；每项指标/seed一行 |
|05_性能与内存分解 | 普通/边界、活动/暴露时间、字节、CPU/GPU内存；注明独立profile |
|06_曲线文件 | 在线loss、验证指标、吞吐/延迟、资源trace的路径和明确横纵轴 |

- input、supervised、padded tokens分别统计，不将旧padded吞吐抄作模板的有效input吞吐。
- 主吞吐窗口不含最终drain；另报window+tail和完整loop+drain。完整loop仍不等于含模型加载/评估/teardown的cold start。
- GPU allocated/reserved/NVML分列取最大rank；CPU为含worker的采样进程树PSS，不简单sum RSS；累计pin申请不当作live peak。
- 稳态窗口可能从有pending工作的状态开始；尾部drain不伪装成零队列实验。重叠组件活动时间不能简单相加当总耗时。
- 性能与质量均报告seed42单次原始结果；不画跨运行误差条、不计算重复实验标准差/置信区间、不声称跨seed稳定性。
  OOM/失败/未运行留空对应指标，不填0，不挑最快run。
- 按用户最新要求，GPU进程监测改为**只观察、不自动停止**：任何GPU（包括当前使用的GPU）上的外部进程均不触发停训，也不作为启动等待门槛。
  可选监测/日志失败不终止训练；取消监督器的硬时长终止，原时长阈值只提示。只有新的明确停止指令才由监督器终止本实验进程组，不得停止其他用户进程。
  记录外部进程、检查成本和观察错误；有共享负载时如实披露，不把该吞吐标作独占硬件结果，也不能据此声称排除了CPU-only竞争。
  训练程序本身的OOM/异常退出仍记失败；旧中止/受干扰记录保留，不自动重复，必要重试另行确认。
- 模型与trace体积较大，按批核算磁盘预算，确认新中间checkpoint保留策略；不自动删除任何旧实验产物。
- 不预设任何方法必须达到某个倍率，不用7B比例推算大模型性能。

## 9. 建议批准的执行顺序

1. **第一批：W1单次性能3个 + W1单次质量pilot3个**，共6个新训练run，全部seed42；两种训练预算分别每方法1次，不做统计重复。
2. **第二批：W2/W3/W4适配和gate → 各组单次性能 → 分任务单seed质量**，优先补第二架构与GSM8K真实答案指标。
3. **第三批：L1/L2容量 + W1代表性profile/资源运行**，大模型失败如实保留；修复版扩展另行确认。
4. **第四批：1/2/4/8卡扩展、长度/microbatch敏感性、批准的单机制消融**。
5. 质量pilot通过并确认预算后再启动完整epoch与time-to-quality，不直接铺开全部长训练。

W1以外的耗时须先用对应gate校准，不给出未经实测的大模型/长序列GPU-hours承诺。
第一批已于2026-09-12登记并启动，CFG054–CFG059、RUN074–RUN079，进度见原工作簿 `20_W1单seed长预算`。
执行目录：`/data/hangyu/ResearchHub/FastOffload/benchmark_results/workbook_w1_single_seed_20260912_012436/`。

首批启动前16项CPU检查通过，重新核验完整模型SHA256、冻结数据/BASE验证子集、1,200/4,096步输入ID与token计数。
三项性能及FastOffload/ZenFlow质量共5项完成并通过审计。Native质量RUN079最后记录2,312/4,096步；12:42:47检测到GPU0外部作业后触发旧全节点守卫，主动中止，并非证明了GPU6/7已受干扰。

用户随后批准Native重启，并要求其他GPU上的无关作业不得触发停止。新尝试为CFG060/RUN080，进度见 `21_Native质量重启`：
`/data/hangyu/ResearchHub/FastOffload/benchmark_results/workbook_w1_native_restart_20260912_132413/`。
该次RUN080从同一基座起跑，seed42、GAS1、有效batch2不变；19:20:22在GPU6出现外部进程后中止，最后记录4,024/4,096步，未完成终点导出/验证。
当时的选中GPU守卫通过21项CPU检查，但其自动中止策略已被用户随后明确取消；旧源码/manifest保持原样。

第3次尝试CFG061/RUN081已于2026-09-13 04:32完成，仍为4,096步、seed42、GAS1，结果见 `22_Native不中断重跑`：
`/data/hangyu/ResearchHub/FastOffload/benchmark_results/workbook_w1_native_observe_only_20260912_224540/`。
这次不从4,024步续训，不增加算法/optimizer-resume/窗口内保存功能；RUN079/080及5个已完成结果均不覆盖。
27项CPU检查通过，覆盖同卡外部进程、监测查询/写日志失败、监督观察器异常及超过时长提示阈值时均继续等待训练自然退出。
监督器只记录GPU进程，不因外部作业或时长而杀进程；无PSS或周期内存采样，不承诺共享环境吞吐等于独占环境吞吐。
状态以各目录 `status.json`、`summary.json`、`completion.json` 或 `failure.json` 为准，启动不代表完成；原生错误/用户停止后仍不自动再重试。
RUN081训练/评估均exit0，两个rank各核验4,096次CPU更新、7,615,616,512个参数逐位副本一致，339个导出tensor检查及清理通过。
Native验证NLL=1.4188440372249553，perplexity=4.132340846075287；窗口input吞吐42.78748231350617 tok/s，最大rank allocated峰值23.78528928756714 GiB，loop+drain=19,982.193192秒。
窗口为排除200后的3,896 microsteps，不与1,200步性能组混合；332次进程采样未发现同卡外部进程，不代表排除了所有GPU/CPU竞争。
同为4,096步/seed42的FastOffload、ZenFlow验证NLL分别为1.298853848503297、1.3430591939517669；这是固定预算单seed pilot，不是收敛或调优后质量等价证明。
第一批现有6个完整目标结果：RUN074–078及RUN081。RUN079/080仍保留中止，原批次失败文件不改写为成功。
监督进程已退出；后续模型/场景未启动，第一批结果验收后再安排。

### W1训练loss曲线

已生成 `assets/qwen2_5_7b_w1_4096_training_loss.png` 和同名SVG，并登记原工作簿 `23_4096步Loss曲线` / `06_曲线文件`。
数据仅来自完整RUN081/077/078，按双rank有效回答token数加权；上图逐microstep原始loss，下图完整128步后向滑动窗口，前127个平滑值留空。
不额外按GAS缩放，不把改变中的训练样本loss当作验证轨迹；RUN079/080中止数据不纳入。
全精度CSV、统计、来源SHA256及脚本位于：
`/data/hangyu/ResearchHub/FastOffload/benchmark_results/workbook_w1_loss_curves_20260913_110255/`。
14项CPU聚合/滑窗检查通过；绘图未重新训练或评估，原工作簿02/03/04结果及全部已有公式保持不变。

## 10. 用户新增：FastOffload 1024步LR调参

2026-09-13用户明确批准调整学习率或scheduler，使用FastOffload每项仅训练1,024步，寻找更稳定的下降趋势。
该新增任务仅覆盖FastOffload调参，不改写此前三方法基准，也不自动更改Native/ZenFlow配置。

首轮固定LR候选顺序：5e-6新对照、2e-6、1e-6、3e-6；均seed42、GAS1、GPU6/7，从同一基座与相同数据顺序起跑。
当前实现的GPU SelectedColumnAdamW和CPUAdam子进程缓存初始化LR，CPU任务未携带版本专属LR；普通DeepSpeed scheduler不能自动协调三处LR，因此首轮不放置无效的名义scheduler，也不修改生产算法。
模型在256/512/768步排空已提交CPU工作后做快照，保留活动B与成功步调度，1024步用原终点导出；训练结束后在独立普通HF进程评估四个快照。
暂停可能改变异步就绪/发布时序，所有候选采用同一快照协议，本轮计时包含中间操作，不作主吞吐比较。
LR也可能改变一步Native importance预热后的选择集合；这是完整训练配置调参，不是固定mask的LR消融。

预先冻结筛选条件：256→1024验证NLL下降至少2%；相邻验证点回升不超过0.005；终点不高于5e-6对照+0.005；训练首/末256步token加权loss下降至少2%，8个128步非重叠分段斜率为负。
这些是工程筛选阈值，不是统计置信区间或每个batch单调的承诺；验证集用于调参，test不使用。不通过就如实报告，不能调整数据顺序或平滑窗口来制造下降。
对照之后第一个满足全部条件的候选完成后结束首轮；若4项均不满足，再进入下一阶段，不宣称已找到方案。

执行目录：`/data/hangyu/ResearchHub/FastOffload/benchmark_results/workbook_fo_lr1024_20260913_122050/`。
已登记CFG062–065 / RUN082–085，进度见 `24_FO_1024步LR调参`。
5e-6对照RUN082已完整训练/评估：四个快照验证NLL为1.281621、1.275722、1.279903、1.281959；试验成功，但未达到预设下降标准。
3项CPU测试及双rank 4.5M参数Qwen快照gate通过，覆盖spawned CPUAdam、8/12步快照后继续到16步、partial B=3及worker退出。
Gate首次因启动PATH未包含已有conda ninja而在训练前失败，日志保留；修正为与正式启动一致的PATH后通过，未重跑任何LLM调参候选。
保持GPU进程仅观察、不因外部作业/监测失败/时长提示停训；不自动重试原生失败，不删除已有模型或输出。

### 用户追加4卡资源：两组2卡并行

用户随后允许使用4卡加快调参。实际采用2+2并行：原GPU6/7队列及其脚本/manifest完全保留、不暂停、不重跑，GPU4/5新增一个1.5e-6插值候选。
每项仍是world2、microbatch1/GPU、GAS1、global microbatch2、1,024步/seed42，不是把同一试验改成DP4/global batch4。
新增槽只跑该一个候选；已启动的试验不会因为另一槽达标而被截断。并发共享64个CPU核和主存带宽，不修改既有运行的affinity，不声称2倍加速或严格隔离LR因果。
启动观察已记录GPU4/5也有外部作业；继续执行仅观察策略，真实OOM/运行错误仍如实记为失败，不自动重试。

新增目录：`/data/hangyu/ResearchHub/FastOffload/benchmark_results/workbook_fo_lr1024_parallel_20260913_132510/`。
复用验收通过的RUN082对照，训练/快照/源模块helper逐字节一致，模型/数据/预期ID重新核验；筛选条件不变。
8项CPU检查通过，覆盖GPU分组/不重复候选、筛选一致性、原写入者PID复用、延迟Excel登记、可选GPU查询失败和旧结果/公式保留。
首次Excel测试误把预格式化空白模板行当成已填结果，失败日志保留；修正为检查全部已用行及所有原公式，允许向空模板行新增记录。
原工作簿写入者未使用跨进程锁，因此新增槽先记录本地状态，待原supervisor退出后再登记 `25_FO四卡并行调参` 并分配新CFG/RUN/EVAL ID，避免并发覆盖旧结果。
该等待发生在新增训练/评估结束后，不是GPU空闲门禁；`workbook_ids.json`出现前不宣称ID已经登记。

新增1.5e-6槽实际在第5步dense backward发生CUDA OOM，双rank仅完成4条microstep记录：物理GPU4外部进程占46.32GiB，本rank含非PyTorch占32.55GiB，请求520MiB时仅剩256.06MiB空闲。
报错点为Native offload overflow检查的 `float_x.isinf()`；同时存在2.25GiB PyTorch reserved-but-unallocated，不能将所有因素简单归结为单一因果。
这是真实运行错误后由DeepSpeed清理同作业rank，不是监控因为外部进程主动停训。没有快照/终点评估，完整性能/质量指标留空，不自动重试或切换allocator。
失败保存在新增目录 `FAILURE_NOTE.md` / `failure.json` / `candidate_1p5e6/fastoffload/train.log`；新增GPU作业已退出，仅supervisor等待安全登记OOM记录。
原GPU6/7队列未被改动或中断。2e-6（RUN083）已完整训练/评估：验证NLL依次为1.272596、1.264874、1.261611、1.259486，方向持续下降，但256→1024降幅约1.03%，未过预设2%门槛；其余四项筛选检查通过，阈值未改。
原队列后续1e-6（RUN084）和3e-6（RUN085）也已各完成1,024步及四个快照评估，终点NLL分别1.260399和1.264498，256→1024降幅分别1.68%和0.74%。
首轮4项完成，但没有候选满足全部预设条件；1.5e-6失败已登记为RUN086/CFG066，`25_FO四卡并行调参`已回填OOM，两个实验supervisor均退出。
四卡并行尝试因新增槽OOM未能持续，不声称已经提速。

## 11. Scheduler接线：B/C使用同一步A的LR

用户明确同意B/C同步当前A使用的LR后，已修改生产Takeover路径：Adapter从Native optimizer读取当前统一LR，GPU A显式使用它；dense boundary提交的B/C任务保存相同scalar LR，传输等待及CPUAdam进程执行都保留该值。
因此第9步提交的B/C使用A第9步的LR，即使第12步才执行，也不会改用第12步的新LR。B梯度仍按原规则mean/sum累积，C仍只用边界梯度，更新频率和forward publication不变。
这是对第10节当时“缓存LR未同步”限制的功能修复；旧实验配置、脚本及结果不被改写。现有冻结实验harness会因生产源hash变化拒绝复用，需要为scheduler实验创建新来源快照。

已有78项CPU/gloo检查和14项GPU集成检查通过，包括：可变/零LR的AdamW数值对照、运行/排队任务独立LR、D2H事件后保留LR、真实spawned CPUAdam的FP32 master/moments对照、GAS、overflow不推进scheduler、Hybrid drain及Native optimizer/scheduler状态恢复、禁用Takeover路径的一致性。
测试/源码前后快照目录：`/data/hangyu/ResearchHub/FastOffload/benchmark_results/fo_lr_sync_20260913_161945/`。
主配置可使用DeepSpeed `WarmupCosineLR`，参见 `CONFIGURATION.md`；FastOffload配置中的 `scheduler` 仍是传输调度器，不能混用。
当前只同步LR，不支持同时改变betas/eps/weight decay的调度或不同optimizer group LR；构造缓存不再代表当前实际step/job LR。
**这轮完成的是接线和小模型正确性验证，尚未新跑1,024步Qwen scheduler调参，也不代表真实大模型optimizer-state resume已验证。**

## 12. 用户批准启动首项scheduler试验

新目录：`/data/hangyu/ResearchHub/FastOffload/benchmark_results/workbook_fo_scheduler1024_20260913_173857/`。
只运行一个新的FastOffload候选：GPU6/7、world2/GAS1/global microbatch2、seed42、1,024步；峰值LR2e-6、DeepSpeed `WarmupCosineLR` 的log warmup32、warmup/cos最低比例均0.1。A1/B-C4、快照位置及五项筛选标准不变，不追加其他候选。
发现该scheduler默认index=-1会将首次更新LR置0，即使warmup_min_ratio>0也是如此；run-local harness在训练前显式调用 `step(0)`，记录global_steps仍0。首个Native importance预热LR为2e-7，第32步LR2e-6，第1,024步LR2e-7；没有多做训练更新。
采用新的生产源码快照，LR改动7文件与92项回归所测版本一致；父/CPU子进程冻结导入关键模块，逐step/提交版本/真实CPUAdam group LR审计。快照还检查scheduler计数及LR不变。
8项CPU检查和两rank16步Qwen快照/继续训练gate通过，gate实际spawned CPUAdam各完成3个不同LR任务并退出0；模型/数据/ID及旧harness hash重验通过。
初次preflight因参考日志后缀误写为 `.json` 而失败，修正为 `.jsonl` 后通过，失败日志保留；这不是LLM训练重试。
复用RUN082（5e-6）筛选对照及RUN083（constant2e-6）参考，不改写或重跑旧实验。LR warmup会改变importance选择，新审计和共享主机时序也可能影响异步发布，因此不声称固定mask、严格隔离scheduler因果或主吞吐收益。
沿用观察-only政策：外部进程/可选查询失败/时间警告不自动停训；自然失败不自动重试，保留历史文件且不删除旧快照。结果登记在原工作簿新页 `26_FO_LR调度器1024步`；实际启动与完成状态以新目录的进程/manifest/status/审计文件为准。

RUN087/CFG067已完成1,024步及四次评估：验证NLL为1.274877、1.266892、1.261264、1.259408，连续下降1.21%，未过预设2%后段下降门槛，其余四项通过；在线首末256下降4.28%。终点与RUN083 constant2e-6的1.259486基本持平，不据此声称scheduler明显收益。每rank255个B/C版本实际LR、全副本/导出/清理审计通过；44次训练观察中35次有选中GPU外部作业，未因此中断。原表26已回填完成，筛选为否。

## 13. 用户批准旧optimizer/scheduler配方迁移

新目录：`/data/hangyu/ResearchHub/FastOffload/benchmark_results/workbook_fo_legacy_recipe1024_20260913_205753/`。
只新增一个FastOffload 1,024步候选：峰值5e-6、betas(.9,.999)、eps1e-8、weight decay.01、clip1；严格采用归档H100脚本的线性warmup3%+cosine到0公式。3%按旧脚本向下取整为30步，首步LR5e-6/30，第30步到5e-6，第1,024步到0。
采用run-local LambdaLR callback交给DeepSpeed：constructor的index0对应旧公式t=1，之后由engine每次成功更新后推进，无额外 `step(0)`，不在DS JSON里填不可执行的scheduler条目。独立 `lr_schedule.json` 和manifest记录实际方案；单测提取归档中已审查的纯LR函数逐步比对，不执行旧训练入口。
当前生产实现与RUN087一致，新增的H100脚本仅作为归档/来源快照，不作为训练后端。保持GPU6/7/world2/GAS1/global microbatch2、seed42、当前冻结数据/Prompt/response-only/token加权512例验证、A1/B-C4、快照256/512/768/1024及原五项筛选标准。
8项CPU检查及两rank16步小Qwen gate通过，含实际CPUAdam的betas/eps/weight decay、版本所属LR、最终A的零LR、快照后继续训练和清理；完整模型/数据/旧harness hash预检通过。运行采用新root和原工作簿新页 `27_FO旧配方1024步`，不自动重试或追加候选。
这是多个optimizer/scheduler因子一并改变的配方对照，不是单因素因果消融、固定mask研究或旧H100实验的等价复现；不迁移旧后端的selected-only变体、旧验证小样本/重复计数、旧GAS改写或旧JSON/CLI LR不一致。复用RUN082筛选对照及RUN083/RUN087参考，旧结果全部保留。观察-only政策和不删除历史快照的约束不变。

RUN088/CFG068随后已完成1,024步及四次评估，旧结果与模型保留；用户现已要求暂停质量工作，优先补性能和消融，不继续此调参队列。

## 14. 用户改为性能与消融优先：首批系统实验已启动

2026-09-13用户要求先不关注质量数据，补齐原工作簿其他实验，优先性能与消融。本节优先于此前“质量后再扩展”的顺序：新性能工作仍先做正确性gate，但不要求追加held-out评估或质量调参。已测失败、质量结果及缺测字段不改写。

首批目录：`/data/hangyu/ResearchHub/FastOffload/benchmark_results/workbook_perf_ablation_20260913_224705/`。
在GPU6/7串行运行Qwen7B/Alpaca512、microbatch1/GAS1、seed42；采用原W1性能配方constant5e-6、AdamW(.9,.95)、wd.1，无scheduler，不继承RUN088质量配方。
生产源码与RUN088相同；配置/来源均冻结。先4个36步gate（排除20），再对gate通过的变体各做1次1,200步性能（排除200、测量1,000）：

| 变体 | gate | 性能 | 唯一变化 |
| --- | --- | --- | --- |
| 当前完整FO对照 | RUN089/CFG069 | RUN093/CFG073 | 同源码及审计的128MiB控制 |
| dense Linear backward | RUN090/CFG070 | RUN094/CFG074 | 不安装selective Linear wrapper；保留选中梯度提取、owner压缩RS/AG和Hybrid |
| bucket64 | RUN091/CFG071 | RUN095/CFG075 | compressed bucket目标64MiB |
| bucket256 | RUN092/CFG072 | RUN096/CFG076 | compressed bucket目标256MiB |

关闭packed selective backward不等于关闭所有packing；桶大小是单因素敏感性而不是移除某个机制，参数不拆桶/owner padding可使实际payload峰值超过目标。A1/B-C4、GPU B mean累积、当前边界C、版本/visibility保护保持不变。
新完整FO控制共用当前LR接线源码与路径计数审计，不把历史RUN075或带快照/LR逐任务日志的调参时间当作本批消融分母；不是重复多次后择优。Gate与长测是不同目的和预算，各只尝试一次。

10项CPU检查通过：单因素差异、dense/packed选中权重梯度和输入梯度对照、真实token/padding计数、partial B、观察故障/外部进程/时长不终止，以及工作簿旧数据和全部公式保持。初次检查发现启动前GPU映射查询回退未接入，已在新runner补齐并通过；初始失败日志保留，未重跑训练。
完整模型/数据/历史harness及当前源码SHA256预检通过；真实模型gate还核验实际捕获路径、wrapper数量、实际桶设置、跨rank/变体的相同importance mask、全部更新次数、drain后全参数副本及worker退出/解除注册。即使mask相同，异步就绪/发布时序仍可改变，不声称跨运行轨迹逐位相同。

纯性能批次不做窗口内快照/验证，终点也不导出模型或评估，仍保留完整正确性/清理审计；不删除已有模型。显存峰值从engine初始化后开始统计、含warmup，不当作冷启动NVML或live pinned峰值。Owner payload gauge不当作D2H/H2D总流量。独立profile仍在后续批次，不把重叠计时相加。
新ENV10、CFG069–076、RUN089–096已登记；工作簿 `28_性能消融待补清单` 跟踪缺口，`29_FO系统消融性能` 跟踪当前8项。只回填实际通过完整审计链的结果；失败gate对应长测保持未运行，独立变体可继续，无自动重试。源码完整性或本作业清理异常会推迟后续启动，不杀任何外部作业。
沿用观察-only政策，无外部进程空闲门禁、无硬超时杀进程，查询/日志/绘图失败不终止正在运行的训练。后续按W2/W3/W4数据/架构适配与gate→主性能→独立profile/容量→microbatch/长度/GPU扩展推进；这些尚未启动，不把待补清单视作已完成。未经完整Takeover验证的CPU累积/独立传输精度/同步Hybrid仍注明未支持，不为填表新增伪消融。

上述RUN089–096现已全部完成并验收，4个36步gate和4个1,200步性能结果均保留在原表29；不把桶敏感性当作多项独立机制消融。

## 15. 用户批准双卡主性能200步

2026-09-14用户要求先跑2卡可运行的主实验，仅性能、预算200步。已按 `MAIN_PERFORMANCE_EXPERIMENT_DESIGN.md` 第9节冻结新组：GPU6/7，200步中排除40、测量160；相同seed42、microbatch1，三方法保留GAS1/1/4和原生更新节奏，constant5e-6/Adam(.9,.95)/wd.1。
新目录：`/data/hangyu/ResearchHub/FastOffload/benchmark_results/workbook_main200_20260914_145637/`。
W1三方法RUN097–099；W3三方法36步gate RUN100–102与200步性能RUN103–105；W4 gate RUN106–108与性能RUN109–111，共15个已登记尝试（9性能/6gate），ENV11、CFG077–091，原工作簿 `30_双卡主性能200步` 自动更新。Mistral本地bin安全加载前提未满足，明确未运行，不绕过检查或更换模型；不展开4/8卡大模型和batch扫描。
W1沿用原输入前缀；GSM8K/XSum新增run-local适配分别保留6,960/49,994训练记录，保留完整response/EOS/提示后缀，仅截断正文，过滤及ID在启动前冻结。test不加载，validation不评估，原数据缓存不改。12项CPU检查及完整来源/计数预检通过，初始检查中的hash-manifest路径和ZenFlow配置层级错误均已修正，失败日志保留，未因此重试训练。
本批不导出模型、不做质量或快照；更新次数、drain、全副本及CPU清理仍验收。200/40为独立短测，不池化旧200/20、100/20、1200/200或质量调参计时，也不声称160步已证明长期稳态。外部作业/观察失败/时间警告不自动停训，自然失败保留且不自动重试。

## 16. CPU B归约与累积实现及性能测试

当前稳定基线已推送：内层 `6388b7564`，外层 `c20c8b0`。随后新增可选 `second_reduce_scatter_device=cpu` 配合 `accumulation_device=cpu`：本地B逐步BF16 D2H，CPU Gloo Reduce-Scatter、FP32累积；包含当前边界B。A/C继续GPU归约，Native密集布局中的B先取出再清零，不增加全模型GPU暂存。B D2H和Gloo为同步参考路径，CPUAdam仍异步，尚未实现完全异步B通信或删除GPU密集桶零B占位。GAS=1/完整world DP限制显式校验；118项CPU与21项GPU检查通过，覆盖真实spawned CPUAdam、混合范数、CPU-B-only边界溢出、部分B保存恢复及清理；新实现暂未提交。

用户要求测性能后，新建并冻结：`/data/hangyu/ResearchHub/FastOffload/benchmark_results/workbook_cpu_b_perf200_20260914_232759/`。
比较同一当前源码的GPU归约+GPU累积与CPU归约+CPU累积，Qwen7B/Alpaca512、GPU6/7、seed42、GAS1、microbatch1、constant5e-6/Adam(.9,.95)/wd.1、A1/B-C4、packed backward、128MiB桶、lag2。两处位置开关同时变化，是完整CPU B方案对照，不单独归因。两方案各36/20gate，再各200/40性能，失败不重试，不使用历史吞吐分母。实际CPU归约计数、BF16 B下传字节、缓冲位置及相同mask均验收。
9项流程测试和完整来源/数据预检通过。ENV12、CFG092–095、RUN112–115登记原工作簿 `31_B_CPU归约累积性能`；备份位于 `backup/xlsx_before_cpu_b_perf200_20260914_233527/`。队列已启动，RUN112 GPU gate已验收；其余结果以本批 `status.json` / `completion.json` 为准，不把启动或早期benchmark当作性能完成。
保留原200/40计时及post-init含warmup显存口径，不评估/保存模型；完整更新、drain、全参数副本、CPU group/worker清理仍必需。CPU/PSS/live-pinned峰值未测留空，GPU密集通信字节不宣称减少；外部作业、观测失败、耗时警告只记录不停止。

## 17. 补测GPU归约＋CPU累积（等待前一组结束）

用户随后要求补测中间路径：`second_reduce_scatter_device=gpu`、`accumulation_device=cpu`。新目录：`/data/hangyu/ResearchHub/FastOffload/benchmark_results/workbook_gpu_rs_cpu_accum200_20260915_001850/`。
仅改变相对同源GPU对照的B累积位置；GPU先将B归约到owner，每步复制owner B到CPU并累加。此旧累积入口保留来梯度dtype，并非强制FP32，也不使用CPU-RS专用BF16暂存/Gloo路径；终点记录真实累积dtype。CPU-RS专用字节计数为0不表示没有B的D2H。
冻结相同Qwen7B/Alpaca512、GPU6/7、GAS1/microbatch1、seed42、constant5e-6/Adam(.9,.95)/wd.1、A1/B-C4、packed backward/128MiB/lag2，先36/20gate，后200/40性能，各一次。显式共享前一组已验收RUN114 GPU/GPU200步对照及RUN112 gate，不重跑对照；CPU/CPU200步结果仅在验收后作为第三条参考展示，不混算gate和性能。
预检与11项CPU流程测试通过，源码、模型/输入及对照配置一致。监督进程已启动，但等待已核验PID、创建时间、UID和命令的前一组自有监督进程结束且退出，并验证自有训练组清理记录；不启动并发GPU训练，也不并发写工作簿。`queue_manifest.json` 在等待前冻结脚本，依赖释放后再次检查来源。
原工作簿 `32_GPU归约CPU累积性能` 将在前一写入者退出后备份、登记两个新RUN/CFG；等待时尚未分配ID，不预填成功或性能值。外部作业不构成空闲门禁；不停止/改动前一冻结队列，无自动重试。等待状态见本批 `waiting.json`，训练启动与完成以 `status.json` / `completion.json` 为准。
