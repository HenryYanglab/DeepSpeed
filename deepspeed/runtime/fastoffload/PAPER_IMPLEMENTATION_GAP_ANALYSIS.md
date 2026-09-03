<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- DeepSpeed Team -->

# FastOffload 论文与当前实现差距分析

本文档对照论文《Accelerating LLM Full Fine-Tuning via Offloading and Periodic Transfer》检查当前 `deepspeed/runtime/fastoffload/` 实现。检查基于论文 Design（第 3 节）和 Implementation（第 4 节）。

## 1. 结论

当前实现与论文不是简单的“部分功能尚未完成”，而是存在两套不同的更新算法：

- **论文算法**：Selected/Unselected 两组；普通步骤仍执行 dense backward，把 selected output-row ranges 传到 CPU 累积；周期边界执行一次完整 CPUAdam，并把完整低精度模型分区返回 GPU。
- **当前 Hybrid Takeover**：A/B/C 三组；普通步骤可以执行 packed selective backward；A 每步在 GPU Adam 更新，B 在 GPU 累积，B/C 在周期边界由 CPUAdam 更新；只发布发生变化的 selected columns。

因此，不能把论文缺失点直接混入当前 Takeover，否则会改变现有 A/B/C 数学语义。

**当前项目决策**：保留 `hybrid_abc` 作为唯一核心更新语义，并继续实现论文中与 A/B/C 兼容的传输、异步、buffer、fallback、checkpoint 和 instrumentation 技术。论文中与 A/B/C 直接冲突的 selected-only CPU accumulation、ordinary-step no-update 和 single full-partition CPUAdam 不并入当前模式；对应论文文字和 Algorithm 1 需要改写为 A/B/C 语义。Output-row range packing可以作为可选选择轴研究，但不能替换当前默认 input-column 语义。

## 2. 已实现或基本实现的技术点

| 论文技术点 | 当前状态 | 说明 |
|---|---|---|
| ZeRO-2 CPU optimizer offload 集成 | 已实现 | Takeover 复用 ZeRO-2 partition ownership 和 gradient-ready 生命周期。 |
| 周期性 dense boundary | 已实现 | `update_interval` 控制完整 A+B+C boundary。 |
| BF16 gradient D2H | 已实现 | Boundary owner gradients 使用模型 dtype，Qwen BF16。 |
| CPU FP32 master parameters | 已实现 | CPU 侧 master values 保持 FP32。 |
| CPU FP32 Adam moments | 已实现 | `exp_avg`、`exp_avg_sq` 保持 FP32。 |
| Native DeepSpeedCPUAdam | 已实现 | 大模型状态通过独立进程调用 `DeepSpeedCPUAdam`。 |
| 独立 CPU optimizer process | 已实现 | 避免 Python GIL 和 CPUAdam/OpenMP 与训练线程竞争。 |
| Shared CPU storage | 已实现 | master、moments 和 gradient 使用 shared memory。 |
| Pinned CPU gradient storage | 已实现 | shared allocation 使用 host registration，避免重复 CPU buffer。 |
| CUDA stream/event D2H | 已实现 | Boundary C 可在 transfer stream 上直接写入 CPU flat offset。 |
| ZeRO owner-local partition clipping | 已实现 | 支持 parameter fragment 首尾非完整 row。 |
| Safe version publication | 基本实现 | CPU result 通过版本号在安全 forward boundary 发布。 |
| Bounded asynchronous lag | 已实现 | `max_async_lag` 限制未提交 CPU versions。 |
| Dense boundary bucket streaming | 已实现 | 复用 Native ZeRO-2 gradient-ready buckets，避免全模型 dense gradient 常驻 GPU。 |
| Overflow、norm 和 clipping | 基本实现 | Takeover 使用分布式 overflow/global norm/combined scale。 |
| Disabled mode 保持 Native 行为 | 已实现并测试 | FastOffload/Hybrid 关闭时不接管 Native ZeRO-2。 |

## 3. 关键算法差异

### 3.1 选择维度不一致

论文第 2.3、3.3 和图 3/4 使用 **output rows**：

```text
W.shape = [out_features, in_features]
selected payload = dW[selected_rows, :]
selection axis = dim 0
```

当前实现使用 **input columns**：

```text
selected payload = dW[:, selected_columns]
selection axis = dim 1
```

当前重要性分数为 pretrained-to-warmup L1 column movement：

```python
abs(current - reference).sum(dim=0)
```

论文中的测量依据为 gradient squared row norm：

```python
squared_gradient.sum(dim=1)
```

这是最高优先级差异。论文使用 row selection 的一个系统原因是 row-major 权重中的完整 output row 是连续 range，适合 offset-length packing；当前 column selection 在 row-major layout 中通常是大量不连续位置。

### 3.2 更新语义不一致

论文 Algorithm 1：

```text
Ordinary step:
    dense backward
    pack selected rows
    BF16 D2H
    CPU FP32 selected accumulator += gradient
    GPU model parameters unchanged

Dense reconciliation:
    dense gradient BF16 D2H
    dense_gradient += selected_accumulator
    one full CPUAdam update
    full BF16 model partition H2D
```

当前 Takeover：

```text
Ordinary step:
    packed A+B backward（可选）
    A owner GPU Adam update
    B owner GPU accumulation

Dense boundary:
    full A+B+C backward
    A GPU Adam update
    B/C owner CPUAdam update
    compressed selected-value publication
```

论文明确说明不会在 sparse step 对 selected coordinates 独立执行 AdamW；当前 A 正是在每个 successful step 上执行独立 GPU AdamW。因此两者不能视为同一算法的实现细节差异。

### 3.3 CPU accumulation 位置不一致

论文默认：

```text
selected sparse gradients → CPU FP32 accumulator
```

当前高吞吐配置默认实际使用：

```text
B gradients → GPU accumulator
```

当前配置支持 `accumulation_device="cpu"`，但它仍属于 A/B/C Takeover，并不是论文所描述的单一 selected accumulator + full dense CPUAdam 路径。

## 4. 尚未实现的论文技术点

### P0：论文兼容算法语义

1. **独立 paper-compatible runtime policy**
   - 新增 `paper_sparse_dense`，不得修改当前 `hybrid_abc` 默认语义；
   - Selected/Unselected 两组，而不是 A/B/C 三组 optimizer；
   - ordinary step 不更新 GPU 参数；
   - boundary 使用同一个完整 CPUAdam state 更新所有 owner-local coordinates。

2. **Output-row selection**
   - 支持 `selection_axis="output_row"`；
   - `W[selected_rows, :]` 映射为连续 flat ranges；
   - 正确处理 ZeRO partition 首尾截断；
   - embeddings、output heads、1D 参数和不支持布局走 dense fallback。

3. **CPU-side selected-gradient accumulation**
   - ordinary step 的 selected row ranges 使用 BF16 D2H；
   - CPU 收到后转为 FP32；
   - 累积到 owner-local selected accumulator；
   - boundary 将 accumulator 加入当前 dense gradient；
   - successful boundary 后只清除 selected ranges。

4. **Single full-partition CPUAdam boundary update**
   - CPUAdam 的输入为完整 owner-local dense gradient；
   - selected coordinates 同时包含 boundary gradient 和此前 accumulated gradient；
   - master、moments 和 weight decay 由同一个 CPUAdam authority 管理；
   - 不为 selected coordinates 维护第二套 GPU optimizer state。

### P1：传输路径

5. **Packed contiguous sparse D2H**
   - 将 visible selected rows 转成 `(source_offset, length, packed_offset)`；
   - 合并相邻 ranges；
   - 使用少量 contiguous GPU pack buffers；
   - 使用 persistent GPU pack buffer 和 pinned CPU receive buffer；
   - 避免每个 range 单独发起 D2H。

6. **Low-precision full parameter return（部分完成）**
   - Hybrid A/B/C 已增加 persistent shared+pinned model-dtype return buffer；
   - CPU optimizer process 将更新后的 FP32 selected masters cast 到该低精度 mirror；
   - publication 从低精度 pinned mirror 执行 selected-value H2D 和 compressed all-gather；
   - Hybrid A/B/C selected-value publication 已实现独立 visibility event；论文模式的 full-partition H2D 不属于当前算法范围。

7. **完整的三类事件（已完成 Hybrid A/B/C 路径）**
   - Ready event：gradient producer → transfer stream；
   - Complete event：D2H completion → CPU accumulation/CPUAdam；
   - Visibility event：H2D/compressed publication completion → next GPU model read；
   - return buffer 在 visibility 完成前不会被下一 CPU version 覆盖。

8. **Transfer precision configuration**
   - 分别配置 gradient D2H dtype 和 parameter H2D dtype；
   - 支持 `G32/R16`、`G16/R16` 等论文消融配置；
   - unsupported dtype 自动回退。

### P1：正确性与回退

9. **Bucket-level conservative fallback**
   - unsupported layout/dtype；
   - embeddings/output heads；
   - non-contiguous gather 代价高于 dense；
   - overflow recovery；
   - checkpoint/debug path；
   - fallback 应只影响当前 bucket，不应关闭整个 FastOffload run。

10. **Overflow fallback 行为**
    - 论文要求受影响 bucket 回到 dense FP32 movement；
    - 当前 Takeover overflow 路径丢弃本次 staging/accumulation并重试，不是 bucket-level dense FP32 fallback。

11. **Checkpoint drain-and-commit（已完成核心路径）**
    - `state_dict()` 和 shutdown 都会等待 outstanding CPU jobs；
    - 每个 version 必须经过全 rank readiness 后单独 commit，避免各 rank 一次提交不同数量的 ready versions；
    - checkpoint 会同步 visibility event，并保存 active B accumulator；
    - 两 GPU pending-update drain、reload 和 overflow rollback integration test 已通过。

12. **Optimizer group 支持**
    - 当前 Takeover 要求 optimizer groups 使用相同 Adam hyperparameters；
    - 论文语义应按 ZeRO optimizer group 分别维护 CPUAdam authority。

### P2：策略与实验能力

13. **FastOffload\* aggressive mode**
    - ordinary step 跳过 selected sparse D2H；
    - boundary 仍执行完整 dense reconciliation；
    - 配置和结果中必须与默认 quality-oriented mode 明确区分。

14. **Eligibility cost model**
    - 根据 selected ratio、range fragmentation、pack cost 和 dense transfer cost选择 sparse 或 dense route；
    - 当前 selective backward 有计算 fallback，但没有论文所述 transfer-range fallback cost model。

15. **论文级 route instrumentation**
    - pack、cast、D2H、CPU accumulation、CPUAdam、CPU cast、H2D、visibility wait；
    - issued route time 与 exposed critical-path time分开；
    - dense/selected/fallback element counts；
    - 当前 telemetry 只有其中一部分。

16. **持久 buffer 重用与 reallocation 统计**
    - selected payload 稳定时复用 GPU pack 和 pinned CPU receive buffer；
    - selected size 或 fallback 改变时扩容；
    - 记录 reallocation 次数和峰值。

## 5. 论文文字与当前实现需要统一的地方

如果论文描述的是当前 A/B/C Takeover，则以下论文内容必须修改，而不是继续补代码：

1. `selected output rows` 应改为 `selected input columns`；
2. `GPU still runs dense backward on every step` 应改为 Ordinary packed selective backward；
3. `selected gradients accumulate directly on CPU` 应改为 B 默认在 GPU 累积；
4. `no independent selected AdamW update` 与 A 每步 GPU Adam直接冲突；
5. `full model partition BF16 H2D` 应改为 compressed selected-value publication；
6. Algorithm 1 应重写为 A/B/C scheduler；
7. Figure 2、Figure 4、Table 1 的数据流和字节模型均需相应修改。

当前已确定以 Hybrid A/B/C 为实现规范，因此论文 Algorithm 1、图 2/4 和字节模型需要按当前更新语义修订，不能继续使用 selected-only CPU accumulation 的描述。

## 6. 推荐实现顺序

### Phase 1：保持 Hybrid A/B/C 语义

- 默认和生产路径继续使用 `hybrid_abc`；
- 不增加会静默切换 optimizer 数学语义的策略；
- 优先移植论文中兼容的低精度 return、persistent buffers、visibility、fallback、checkpoint drain 和 route instrumentation。

### Phase 2：低精度 publication 和持久 buffer

- persistent shared+pinned model-dtype return buffer（已完成）；
- CPU FP32 master 到低精度 return mirror（已完成）；
- selected-value nonblocking H2D（已完成）；
- 显式 visibility event 和 transfer-buffer version ownership；
- gradient/return dtype 独立配置。

### Phase 3：正确性与回退

- checkpoint 自动 drain pending updates（已完成基础路径）；
- bucket-level unsupported layout/dtype fallback；
- overflow 与正在执行的 D2H/CPU job 回滚；
- 多 optimizer group 的独立 CPUAdam state；
- unused parameter 和 checkpoint/backward 互斥测试。

### Phase 4：Range packing 和可选选择轴

- 为当前 input-column payload 优化 deterministic packing 和 buffer reuse；
- 根据 fragmentation/copy cost 自动退回 dense route；
- 可选实现 output-row selection、ZeRO local clipped ranges 和 adjacent-range merge；
- output-row 仅作为 Hybrid A/B/C 的可选 selection axis，不改变 A/B/C optimizer schedule。

### Phase 5：Instrumentation 和实验模式

- pack、cast、D2H、CPUAdam、CPU cast、H2D 和 visibility wait；
- issued time 与 exposed time分开；
- dense/selected/fallback element counts；
- persistent buffer reallocation 统计；
- 如需 aggressive mode，必须保持 A/B/C 边界语义并单独报告质量结果。

### Phase 6：长程验证

- GAS=1/2、overflow、checkpoint、unused parameters；
- 1/2/4 GPU deterministic collective order；
- Qwen/Llama 长程 loss 和 convergence；
- Native、Hybrid A/B/C 及各传输消融使用一致数据顺序。

## 7. 必须增加的测试

- output-row score 和 top-k CPU reference；
- row range clipping：fully local、partial、absent；
- adjacent range merge 和 deterministic packing；
- BF16 sparse D2H → FP32 CPU accumulation；
- GAS=1/2 selected accumulation；
- dense gradient + selected accumulator 数值；
- single CPUAdam 与 CPU reference parity；
- overflow 不污染 accumulator/master/moments；
- visibility event 阻止 partial model version；
- checkpoint with pending D2H/CPUAdam；
- unsupported bucket local fallback；
- multiple optimizer groups；
- FastOffload\* ordinary no-transfer assertion；
- 1/2/4 GPU deterministic collective order；
- Native disabled-mode parity。

## 8. 验收标准

Hybrid A/B/C 完整路径应满足：

```text
Ordinary:
packed A+B backward
→ deterministic owner reduction
→ A GPU Adam and publication
→ B accumulation

Boundary:
full A+B+C backward
→ Native ZeRO-2 bucket streaming
→ owner-local A/B/C split
→ C direct BF16 D2H and immediate dense-fragment release
→ A GPU Adam
→ asynchronous B/C CPUAdam
→ pinned low-precision selected-value H2D
→ safe version publication
```

同时必须确保：

- CPU FP32 master parameters 和 Adam moments 始终是 B/C canonical state；
- pending CPU updates 可在 checkpoint 前 drain；
- buffer reuse、fallback 和 instrumentation 不改变 A/B/C 数学语义；
- `hybrid_abc` 已有性能不回归；
- FastOffload 全部关闭时 Native ZeRO-2 行为不变；
- 不再把 selected-only 论文算法的 benchmark 或质量结论与 A/B/C 结果混用。
