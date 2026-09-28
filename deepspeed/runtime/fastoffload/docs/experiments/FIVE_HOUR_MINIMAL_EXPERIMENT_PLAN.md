<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- DeepSpeed Team -->

# 五小时最小实验计划

## 1. 目标与结论边界

目标是在**独占节点、最长5小时wall-clock**内获得三种方法的短程性能、GPU内存、初期Loss及7B→14B规模变化数据，并尽可能补充27B可行性。不是缩小版完整收敛实验：不声称最终质量相同、time-to-quality或论文最终统计显著性。

本文件仅制定计划，不自动启动实验。

优先级：

1. Qwen2.5-7B：Native / ZenFlow / FastOffload，100 microsteps。
2. Qwen2.5-14B：同三种方法，各100 microsteps。
3. Gemma-2-27B：同三种方法，各100 microsteps；保留五小时硬截止，尽力完成。

用户确认：全部9个run统一目标100 microsteps，不再将大模型缩短为16/64步。若超出预算，标记未完成，不将截短run作为100步结果。

**五小时是执行截止，不是对所有模型都能成功跑完的保证。** 未完成、OOM、超时也作为容量/兼容性结果保留。下载、缺权重、代码适配不纳入训练预算，若用户按从现在起5小时计，则它们占用同一预算并按降级方案跳过未就绪模型。

## 2. 实验矩阵

| 组 | 模型 | GPU | 方法 | microsteps/方法 | 排除warmup | 主要产物 |
|---|---|---:|---|---:|---:|---|
| A，必选 | Qwen2.5-7B-Instruct | 固定同一对2GPU | 三方法 | 100 | 20 | 稳态短窗口吞吐、内存、Loss |
| B，优先 | Qwen2.5-14B-Instruct | 固定同一组4GPU | 三方法 | 100 | 20 | 较大模型性能、内存、短程Loss |
| C，探索 | Gemma-2-27B | 全部8GPU | 三方法 | 100 | 20 | 大模型短窗口性能、初始化/训练peak、Loss |

三组同时改变模型大小和GPU数，**不能用它们直接计算强扩展效率**；各组内部方法间比较才是主要结论。各run排除前20 microsteps，测量后80 microsteps；短窗口是否达到稳态仍须检查，不能仅凭统一warmup就认定稳态。

所有方法串行，不在不同GPU组并发跑CPU-offload任务，避免CPU内存带宽竞争污染比较。

### 实际step换算

| 组 | Native/FastOffload `max_steps` | ZenFlow `max_steps` | Native/Fast benchmark warmup | Zen benchmark warmup |
|---|---:|---:|---:|---:|
| A | 100 | 25 | 20 | 5 |
| B | 100 | 25 | 20 | 5 |
| C | 100 | 25 | 20 | 5 |

前提是ZenFlow初始化actual GAS=4、另外两种actual GAS=1。启动时必须记录实际值；不满足则停止该run检查，不能继续按表错误换算。

## 3. 统一训练配置

- 数据：本地Alpaca，所有方法相同样本顺序，seed42。
- 模板：沿用现有Alpaca prompt与response-only监督，不在此次短测中更换模板。
- cutoff512，动态padding保持一致。
- BF16，microbatch/GPU=1，activation checkpointing开启。
- LR **5e-6 constant**；不加尚未验证的scheduler。
- AdamW：betas=(0.9,0.95)，eps=1e-8，weight_decay=0.1；clip=1.0。
- ZeRO-2 CPU optimizer offload，pin_memory和overlap_comm开启。
- FastOffload：A/B=10%/10%，interval4，GPU B accumulation，max_async_lag2，compressed bucket128MiB，原importance warmup1步。
- ZenFlow：top-k10%，interval4，保留原生算法warmup等配置，完整记录其差异。
- 正式计时run关闭详细parameter/bucket telemetry；保留必要benchmark。若没有验证关闭开销/行为，沿用已有配置但明确记录。
- 不导出数十GB模型，不做完整checkpoint；结束时仍必须drain pending jobs并完成publication，不能靠丢弃CPU尾部工作获得速度。
- 同模型三方法使用同一加载dtype/backend。Gemma磁盘FP32，要监控加载峰值，不能仅看BF16训练峰值。

这是method-native比较。相同LR并不意味着GAS、importance规则或A/B/C优化轨迹相同，结果表必须披露。

## 4. 五小时调度与截止

以下是预算，不是预测实测耗时。首次加载/JIT、下载速度和模型兼容性可能改变完成数量。

| 时间段 | 最大预算 | 工作 |
|---|---:|---|
| 00:00–00:20 | 20 min | 核查14B下载完整性、GPU空闲、配置快照、token/计时口径验证 |
| 00:20–01:00 | 40 min | 组A，3个7B短run |
| 01:00–02:30 | 90 min | 组B，3个14B的100-microstep run，含cleanup |
| 02:30–04:35 | 125 min | 组C，3个27B的100-microstep run，含加载与cleanup；单run规划约40min |
| 04:35–05:00 | 25 min | 停止新训练、drain/cleanup、汇总统计与绘图 |

组A依据已有200步总耗时Native约17min、Fast约7min、Zen约10min，100步三run预留40min比较保守，但仍需要硬截止。

组B/组C尚无实测速度，不承诺上述run数均能完成；绝不能把线性外推当结果。4:35后不再启动新训练。预先记录总deadline，所有launcher遵守。

### 执行顺序

各组开始前记录方法顺序，可轮换：

- A：Native → FastOffload → ZenFlow；
- B：ZenFlow → Native → FastOffload；
- C：FastOffload → Native → ZenFlow。

C先验证本方法有助于尽早识别27B兼容性，但必须披露不完整三方法组，不能只展示成功方法暗示其他方法失败。

## 5. 前置检查与失败降级

### Qwen14B

- 下载日志成功结束，检查index全部shard、tokenizer/config可读取。
- 20min准备窗口内没有就绪：不长时间等下载；组B标`weights not ready`。
- 可用本地Gemma-2-9B替代组B做第二架构，但必须改结果标签，不叫14B规模结果。用户优先14B时可直接保留预算给其余已就绪模型。

### Gemma27B

- 确认8GPU独占；先检查模型类、ZenFlow/selective覆盖和加载路径。
- 第一个probe如遇确定性架构异常，不现场大改生产代码耗尽预算；记录unsupported，跳过后续相同失败条件。
- Native/Fast/Zen各自OOM应分别记录，不由一个方法OOM推断全部OOM。
- 不在本轮临时迁移ZeRO-3、改量化或冻结模型来“跑通”，那会改变论文比较范围。

### 超时处理

- 由父runner管理独立进程组与deadline；优先请求正常停止并给予drain宽限，再在确认仅属于本run时清理残留进程。
- 强制终止run标`TIMEOUT/INCOMPLETE`，不使用其未drain的throughput作成功结果。
- 不使用宽泛`pkill python`，不影响下载或其他用户任务。
- 不自动无限重试。每组最多用预留复验时间做一次解释明确的重跑。

## 6. 本轮必须收集的指标

### 性能与资源

1. 固定测量窗口elapsed；所有rank GPU同步，报告最慢rank时间。
2. **全rank真实padded/input/supervised tokens**和samples累计；不要rank0 token数乘world size。
3. tokens/s和microsteps/s；ZenFlow logged boundary/s仅辅助。
4. initialization GPU allocated/reserved peak、training allocated/reserved peak，所有rank的max。
5. CPU训练/worker进程内存可用时记录RSS/PSS，注明共享页口径；pinned cumulative不得叫pinned peak。
6. final drain时间、train-ready到完成时间、冷启动总时间分别列出。
7. run状态、实际完成microsteps、实际GAS、参与GPU数、权重/代码版本。

**计量门槛**：目前旧脚本token计数和最终drain计时有已知局限。准备阶段若不能完成并验证修正，本轮仍可跑Loss/内存/elapsed，但旧tokens/s只标legacy estimate，不作为正式跨方法主结果。也可以在训练外按相同tokenizer/sampler/collator预计算本次每rank窗口token数，要求严格对齐实际已消费batch。

### Loss

- 每4 microsteps一个窗口；三组每方法均25点，排除warmup后20点。
- 全rank、有效response-token加权NLL优先；若沿用旧logger，只标`rank-0 mean online loss`。
- 分别列全窗口mean、warmup后mean、前/后等长窗口mean。不使用不同点数平滑造成视觉误导。
- 全部100步仅观察初期趋势，不声称收敛；跨模型不比较绝对Loss优劣。

## 7. 固定验证集：可选，不牺牲时间上限

若已有通过验证的eval代码，可以固定约64条未参与本次训练的Alpaca样本，保存ID，评测step0与最终drain后NLL：

- 同模型step0只需计算一次，但所有方法应加载完全相同起点。
- eval使用全rank有效label-token加权NLL，不把训练的最后一个batch当validation。
- 每次模型内三方法使用相同eval样本、模板、precision。
- evaluation latency计入预算，单独列出。

若eval功能尚未验证，本轮不临时写大量未测试代码并将结果称为可靠质量；先输出online loss，后续补正式验证。小验证集也不足以确认泛化或time-to-quality。

## 8. 最终图表

### 图1：7B / 14B / 27B的三方法Loss

三面板分别为7B、14B、27B，横轴microsteps，纵轴明确标注Loss统计口径。

- 原始四步窗口细线；5-window moving average粗线，相当于20 microsteps；平滑不向前补值。
- 7B、14B分别画，不把不同tokenizer/模型的绝对Loss横向评价成优劣。
- Native绿色、Fast蓝色、Zen橙色。
- 27B同样展示25个四步窗口和20-microstep平滑曲线。未完成run明确标注实际步数，不补齐或外推到100步。

### 图2：短程性能

每模型3根柱，单位real tokens/s或samples/s，旁注GPU数与测量窗口。标明single run/no error bars。若token计量未修复，用同步elapsed而非伪精确token吞吐。

### 图3：内存/容量

各模型三方法GPU allocated/reserved峰值；初始化峰值单列。27B标OK/OOM/unsupported/timeout，并显示成功run的峰值。未完成不画0GiB。

### 汇总表

```text
Model | GPUs | Method | Completed microsteps | Actual GAS
Measured elapsed | Final drain | End-to-end time
Tokens/s (measurement definition) | GPU peak | Loss mean
Status | Notes
```

无多repeat不画置信区间；任何speedup仅针对同模型同GPU同窗口的成功完整run。

## 9. 输出目录与最小产物

```text
benchmark_results/five_hour_<timestamp>/
  protocol.json                 # 总deadline、预注册顺序和配置
  environment.json
  qwen7b/{native,fastoffload,zenflow}/
  qwen14b/{native,fastoffload,zenflow}/
  gemma27b/{native,fastoffload,zenflow}/
  summary.json
  summary.csv
  loss_curves.png/.svg
  performance.png/.svg
  memory_capacity.png/.svg
  REPORT.md
```

每run保存resolved configs、command、train log、benchmark、exit status、start/end时间。原始结果置持久目录，不仅保留在/tmp。无论某组成功与否都输出REPORT，解释实际完成范围与下一步。

## 10. 五小时后能回答什么

可以回答：

- 相同LR=5e-6下三方法在7B与14B的初期Loss是否正常。
- 同模型同GPU的短窗口速度与显存差异。
- 14B和27B能否在当前实现/资源下运行，瓶颈是容量还是兼容性。
- 哪个大模型配置值得投入后续完整训练。

不能回答：完整收敛、最终质量等价、多seed稳定性、强/弱扩展效率、长上下文泛化、所有系统机制的因果贡献。

**优先保证7B/14B结果完整且口径可信，27B是有硬截止的探索项；宁可清楚报告一个未完成点，也不把超时/未drain的run包装成成功加速。**
