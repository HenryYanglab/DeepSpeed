# FastOffload 配置说明

本文对应当前仓库实现，主要参考：

- `deepspeed/runtime/fast_offload/config.py`：字段、默认值和组合校验
- `runtime.py`：预热、层选择、异步 CPU 更新与预算控制
- `layer_selection.py`：Transformer block 分组和关键层选择
- `channel_selection.py`：按梯度范数选择通道
- `scheduler.py`：tile 能量、陈旧度及覆盖率调度
- `async_cpu_optimizer.py`：后台 CPU AdamW 队列
- `runtime/zero/stage_1_and_2.py`、`runtime/engine.py`：ZeRO-2/Transfer-Lean 实际接入点

> **重要：本文件主要解释 109 个字段的声明含义，不代表字段在所有 mode 中都会生效。**
> `quality_mode` 的真实路由、逐 mode 生效/失效矩阵及当前配置审计请以
> [`QUALITY_MODE_CONFIG_MATRIX.md`](./QUALITY_MODE_CONFIG_MATRIX.md) 为准。
> 特别是 `quality_mode=transfer_lean` 会被转换到 ZenFlow runtime，原生 critical-layer、proxy、tile、residual
> 等大量字段不会传入实际 optimizer。

## 1. 本目录中的配置

| 文件 | 对应 paper runner 方法 | 说明 |
|---|---|---|
| `fastoffload_config.json` | `fastoffload` | 与上级 `fastoffload_config.json` 内容一致 |
| `zenflow_config.json` | `zenflow` | 与 paper runner 使用的上级 `zf_config.json` 内容一致 |

`run_fastoffload_paper_experiments.py` 中的映射是：

```python
"zenflow": Method("zenflow", "zf_config.json")
"fastoffload": Method("fastoffload", "fastoffload_config.json")
```

独立脚本顶部设置 `METHOD=fastoffload` 或 `METHOD=zenflow` 即可切换。脚本会直接从所选 JSON 读取 `train_batch_size`、优化器学习率和 weight decay，避免维护两份配置。基准 JSON 不会被修改；脚本会在本次输出目录生成 `*.runtime.json`，只按 `OFFLOAD_PIN_MEMORY` 应用本机兼容覆盖。

## 2. 工作方式概览

当前代码存在两条互斥的顶层 runtime 路由：

1. **`quality_mode=transfer_lean`**：`engine.py` 将部分字段转换为 `ZenFlowConfig`，使用
   `ZenFlowZeroOptimizer`，并传入 `fast_offload_zero_config=None`。该路由不启用原生 critical-layer GPU AdamW。
2. **其余五种 mode**：进入原生 `FastOffloadRuntime`，将参数分成 critical GPU AdamW 路径与
   noncritical CPU/proxy/residual 路径。

默认配置使用 `quality_mode=transfer_lean`：在非边界 step 可仅同步 selected channels，在更新边界处理
ZenFlow CPU optimizer 结果，从而降低通信和 CPU optimizer 停顿。详细差异见
[`QUALITY_MODE_CONFIG_MATRIX.md`](./QUALITY_MODE_CONFIG_MATRIX.md)。

## 3. 所有 FastOffload 专有字段

以下字段位于：

```json
"zero_optimization": {
  "stage": 2,
  "offload_optimizer": {"device": "cpu"},
  "fast_offload": { ... }
}
```

### 3.1 总开关、层分组与关键层选择

| 字段 | 含义与功能 |
|---|---|
| `enabled` | FastOffload 总开关。存在配置段时默认开启。 |
| `warmup_steps` | 初始完整同步更新步数；预热期间收集梯度统计，尚不稀疏化层更新。 |
| `critical_layer_ratio` | 被判为关键 Transformer blocks 的比例。比例越高，质量通常更稳，但 CPU/传输节省越少。 |
| `layer_grouping` | 层分组策略；当前实现只支持 `transformer_block`，通过参数名识别 `model.layers.N`、`h.N` 等。 |
| `layer_importance_metric` | 关键层评分：`grad_norm_mean` 使用预热期平均梯度范数；`owlore_outlier`/`ows_outlier`/`weight_outlier` 使用权重离群值评分。 |
| `layer_outlier_threshold` | 离群值阈值，以参数样本 RMS 的倍数表示。 |
| `layer_outlier_sample_size` | 每个参数计算离群值评分时最多采样的元素数，限制初始化开销。 |
| `layer_update_mode` | 非关键层 CPU 更新方式：`interval` 按间隔更新，`immediate` 尽快提交。 |
| `skip_grad_offload_while_update_pending` | 上一次 CPU 更新尚未返回时，是否跳过该非关键层的新梯度 offload，避免队列和版本积压。 |
| `always_critical` | 参数名包含这些 token 时强制走关键路径；默认覆盖 embedding、lm_head、norm、bias 等。 |
| `always_critical_layers` | 指定始终关键的 Transformer block 编号。 |
| `critical_update_device` | 关键层更新设备；当前实现要求 `gpu`。 |
| `noncritical_update_device` | 非关键层更新设备；当前实现要求 `cpu`。 |
| `migrate_optimizer_state` | 在可用时迁移关键参数的 optimizer state，以支持关键路径 GPU 更新。 |

`layer_selection.py` 的选择并非单纯取最高分：一部分预算取最高重要度层，另一部分覆盖网络前中后位置，减少关键层集中在局部深度的风险。

### 3.2 更新周期、异步队列与回传

| 字段 | 含义与功能 |
|---|---|
| `noncritical_update_interval` | 每隔多少 optimizer steps 提交一次非关键更新。Transfer-Lean 中也常作为 update boundary 周期。 |
| `noncritical_gradient_accumulation` | 跳过完整更新期间的梯度策略：`selected_average` 累积选中梯度并取平均；`disabled` 不进行该通用累积。 |
| `noncritical_cpu_update` | 非关键 CPU 更新实现；当前只支持 `async_adamw`。 |
| `async_max_pending_per_group` | 每个 ZeRO 参数组最多允许的后台异步更新数；达到上限会等待，值越大越能重叠但占用更多内存。 |
| `max_async_lag_steps` | CPU 精确结果最多允许落后 GPU 当前权重多少步；超过后需要等待或优先应用，防止修正过旧。 |
| `layer_return_bucket_numel` | layer 模式下一个 CPUAdam 返回 bucket 最多包含的参数元素数。大 bucket 吞吐高，小 bucket 首包返回快。 |
| `layer_max_pending_buckets` | layer 模式允许同时在途的返回 bucket 数，超过后强制等待。 |
| `streaming_return_enabled` | 启用实验性的 streaming-return 路径；使用复制的 `fast_offload_streaming` 实现。 |
| `streaming_return_max_pending` | streaming-return 在 GPU 上安全提交前允许暂存的最大 bucket 数。 |
| `cpuadam_cores_perc` | CPU Adam worker 可使用的 CPU 核比例。过高可能挤占 dataloader/通信线程。 |
| `cpu_worker_backend` | CPU worker 后端：`cpp` 优先调用原生 fused op，不可用时回退；`torch` 使用 PyTorch 实现。 |

### 3.3 GPU proxy、CPU 精确修正与 rebase

| 字段 | 含义与功能 |
|---|---|
| `quality_mode` | 顶层算法模式：`transfer_lean`、`reshadow`、`residual_refresh`、`persistent_topk`、`layer_interval`、`layer_immediate`。 |
| `noncritical_quality_policy` | 非关键层质量策略：`residual_refresh` 或 `persistent_topk`；部分 `quality_mode` 会自动设置它。 |
| `gpu_proxy_update` | CPU 精确更新尚在后台运行时，GPU 的近似更新：`momentum_sgd` 或 `normalized_momentum`。 |
| `gpu_proxy_lr_scale` | proxy update 相对基础学习率的倍率。过大可能造成 correction 增大或不稳定。 |
| `proxy_momentum` | GPU proxy momentum 系数。 |
| `proxy_normalization_eps` | normalized momentum 分母的数值稳定项。 |
| `proxy_normalization_scope` | 归一化范围：`range` 按调度范围，`chunk` 按执行 chunk。 |
| `rebase_policy` | CPU 精确权重返回后的合并方式；当前只支持 `cpu_exact_plus_gpu_delta`，即保留 CPU 计算期间 GPU 已产生的增量。 |
| `max_proxy_update_numel_per_group` | 每组一次最多进行 proxy update 的元素数；`0` 表示不设上限。 |
| `proxy_update_chunk_numel` | 一个 CUDA proxy kernel chunk 的最大元素数，用于控制峰值和 kernel 粒度。 |

### 3.4 tile 调度和 residual refresh

| 字段 | 含义与功能 |
|---|---|
| `tile_size` | 将非关键 ZeRO partition 切成 tile 时每块的元素数。 |
| `tile_scheduler` | tile 选择算法；当前只支持 `quality_cost`。优先级综合梯度能量、residual 能量、陈旧度，再除以传输成本。 |
| `min_energy_coverage` | 稀疏 CPU 精确更新希望覆盖的最小能量比例。越高选中的 tile 越多。 |
| `tile_coverage_window` | tile 最长目标刷新间隔；超过该步数的 tile 会进入防饥饿候选。 |
| `tile_round_robin_ratio` | 稀疏更新预算中，为最陈旧 tile 保留的比例。 |
| `tile_staleness_weight` | 陈旧度在 tile 优先级中的权重。 |
| `residual_capture_ratio` | 将候选非关键通道写入 residual buffer 的比例；为 `0` 时实现会根据 `channel_topk_ratio` 推导。 |
| `dense_refresh_interval` | 每隔多少步对非关键范围执行一次 dense refresh；`0` 禁用。 |
| `dense_refresh_ratio` | dense refresh 覆盖比例；当前第一版只允许 `1.0`。 |
| `residual_decay` | 每次加入新 residual 梯度前，对旧 residual 的衰减系数。 |
| `residual_gradient_reduction` | residual 梯度提交 CPU 前使用 `mean` 还是 `sum`。 |
| `max_noncritical_update_numel_per_group` | 每组提交 CPU 精确更新的非关键元素上限；`0` 表示不限制。 |
| `target_loss_gap` | 轻量质量控制器相对 dense/offload reference 的目标 loss gap。 |

`scheduler.py` 会先从预算中保留 round-robin 部分给超龄 tile，再按“能量/传输成本”排序，直到达到 `min_energy_coverage` 或元素预算。

### 3.5 通道选择

| 字段 | 含义与功能 |
|---|---|
| `channel_topk_ratio` | 非关键二维参数选择的通道比例。`channel_selection.py` 沿第 0 维按梯度 L2 norm 取 top-k。 |
| `selected_channel_policy` | 通道策略；当前只接受 `persistent_topk`。 |

对于 ZeRO partition，只选择完全落在本 rank partition 内的完整通道；无法按通道处理的一维参数会走完整/关键路径。

### 3.6 Transfer-Lean：选择、传输与同步

| 字段 | 含义与功能 |
|---|---|
| `transfer_lean_select_strategy` | 选择重算策略：`auto`、`step` 或 `epoch`。 |
| `transfer_lean_select_interval` | 重新选择 top-k 的间隔；可为整数或 `auto`。 |
| `transfer_lean_overlap_step` | 将 CPU optimizer 与后续训练 step 重叠，是减少边界停顿的核心开关。 |
| `transfer_lean_offload_selective_optimizer` | 是否将选中通道对应的 optimizer states 也 offload。 |
| `transfer_lean_warmup_steps` | Transfer-Lean 自己的完整更新预热轮数，区别于通用 `warmup_steps`。 |
| `transfer_lean_direct_pinned_copy` | CPU 更新完成后，直接从 pinned master weight 返回 fp32 partition，减少中间复制。与 int8 stale return 不兼容。 |
| `transfer_lean_stale_param_dtype` | CPU 更新参数返回 GPU 的传输类型：`fp32`、`bit16` 或 `int8`。`bit16` 降低一半流量；int8 更省但有量化误差。 |
| `transfer_lean_stale_param_quant_block_size` | int8 stale parameter 分块量化的 block size。 |
| `transfer_lean_pinned_h2d_staging` | H2D 返回时使用持久化 pinned CPU staging buffer。 |
| `transfer_lean_grad_transfer_dtype` | 梯度 D2H 类型：`fp32`、`bit16` 或 `int8`。 |
| `transfer_lean_grad_quant_block_size` | int8 梯度分块量化 block size。 |
| `transfer_lean_grad_transfer_warmup_steps` | 开始压缩前先使用 fp32 梯度传输的步数。 |
| `transfer_lean_async_grad_copy` | 用独立 CUDA stream 和 GPU clone buffer 异步执行梯度 D2H。 |
| `transfer_lean_async_grad_copy_max_pending` | 异步梯度复制最大在途数；达到后强制完成一部分。 |
| `transfer_lean_pack_selected_grad_d2h` | 先在 GPU 将多个选中范围打包成连续 buffer，再执行 D2H，降低小拷贝开销。 |
| `transfer_lean_async_selected_grad_accum` | 在后台 CPU worker 中用原生 fused op 累积已打包的选中梯度。 |
| `transfer_lean_async_selected_grad_accum_workers` | 上述后台累积 worker 数量。 |
| `transfer_lean_selective_backend` | 选中通道 optimizer 后端：`python` 或 `fused_cuda`。 |
| `transfer_lean_fused_selected_grad` | 使用原生 CUDA kernel 打包/解包选中通道梯度。 |
| `transfer_lean_selected_grad_sync` | rank 间选中梯度同步方式：`broadcast` 或 `all_reduce`。 |
| `transfer_lean_selected_only_nonboundary` | 非更新边界 step 只 reduce 选中梯度，而不是完整梯度。 |
| `transfer_lean_skip_nonboundary_cpu_grad_accum` | 非边界 selected-only step 跳过完整 CPU 梯度累积，进一步减少 D2H/CPU 工作。 |
| `transfer_lean_accumulate_selected_nonboundary` | 在非边界 step 累积选中的非关键范围。 |
| `transfer_lean_dense_boundary_update` | 在 update boundary 提交所有非关键范围，并合并此前选中范围的累积梯度。 |

### 3.7 Transfer-Lean：动态 layerwise top-k

| 字段 | 含义与功能 |
|---|---|
| `transfer_lean_layerwise_topk` | 不再给每层统一比例，而是按层重要度分配 top-k 预算。 |
| `transfer_lean_layerwise_topk_policy` | `weight_outlier` 使用静态权重离群度；`outlier_grad_ema` 混合离群度和梯度 EMA；`task_grad_outlier` 再加入任务敏感度。 |
| `transfer_lean_layerwise_topk_strength` | 层重要度对预算偏斜的强度；`0` 退化为均匀 top-k。 |
| `transfer_lean_layerwise_topk_min_ratio` | 单层 top-k 比例下限。 |
| `transfer_lean_layerwise_topk_max_ratio` | 单层 top-k 比例上限，必须不小于 min。 |
| `transfer_lean_layerwise_topk_grad_ema_beta` | 动态层梯度能量 EMA 的 beta。 |
| `transfer_lean_layerwise_topk_grad_weight` | 动态梯度能量相对静态权重离群分数的权重。 |
| `transfer_lean_global_topk_budget` | 保持所有层选中元素总数等于全局 `channel_topk_ratio` 的预算，避免动态分配意外增加总传输量。 |

### 3.8 Transfer-Lean：任务/Token 敏感度

| 字段 | 含义与功能 |
|---|---|
| `transfer_lean_task_sensitivity` | 启用任务、通道和 token 条件敏感度，影响 layerwise top-k 分配。 |
| `transfer_lean_task_metadata_mode` | `auto_grad` 从梯度自动推断任务 bucket；`engine_metadata` 使用训练程序通过 engine 注入的 metadata。 |
| `transfer_lean_task_bucket_count` | 自动任务 bucket 数量。 |
| `transfer_lean_channel_sensitivity_bins` | 每个参数最多维护的通道敏感度 bins。 |
| `transfer_lean_task_ema_beta` | 任务/通道敏感度统计的 EMA beta。 |
| `transfer_lean_task_priority_weight` | 任务条件层敏感度在动态 top-k score 中的权重。 |
| `transfer_lean_channel_priority_weight` | 参数内部选择通道时，channel/bin 敏感度的权重。 |
| `transfer_lean_token_priority_weight` | 序列长度/token context 信号的权重。 |

默认 paper 配置中该功能关闭，三个 priority weight 也为 `0.0`。

### 3.9 Transfer-Lean：异步 boundary 与 correction cache

| 字段 | 含义与功能 |
|---|---|
| `transfer_lean_async_boundary` | update boundary 不等待完整 CPU offload 结果，结果就绪后异步应用。要求 `transfer_lean_overlap_step=true`。 |
| `transfer_lean_async_boundary_bucket_numel` | 将完整 CPUAdam boundary 结果切成 bucket 时，每个 bucket 的元素数。 |
| `transfer_lean_async_boundary_grad_buffers` | 为异步 boundary 准备的重叠 CPU gradient buffers 数，至少为 2。 |
| `transfer_lean_async_boundary_max_pending` | 每 rank 最多排队/运行的完整 boundary CPUAdam 任务数，不能超过 grad buffers 数。 |
| `transfer_lean_async_boundary_apply_interval` | 每多少个 boundary 应用一次 ready 结果，可合并通知。 |
| `transfer_lean_async_boundary_force_wait_on_reuse` | 仅当即将覆盖仍被 pending task 使用的 gradient buffer 时强制等待。 |
| `transfer_lean_correction_cache` | 对 ready boundary bucket 使用带优先级和预算的 correction cache；要求 async boundary。 |
| `transfer_lean_apply_budget_mb` | 每个 boundary 最多应用多少 MB 的 H2D correction。 |
| `transfer_lean_force_apply_staleness` | ready correction 等待达到该 boundary 数后，无视普通预算强制应用。 |
| `transfer_lean_loss_spike_apply_boost` | 发生质量异常时用于提高 correction apply budget 的倍率预留值。 |

当前校验禁止 async boundary 与 int8 gradient/stale-parameter transfer 组合。

### 3.10 轻量质量控制器

| 字段 | 含义与功能 |
|---|---|
| `quality_controller_enabled` | 启用根据质量信号调整 GPU proxy 元素预算的轻量控制器。 |
| `quality_controller_interval` | 每多少 optimizer steps 调整一次预算。 |
| `proxy_budget_adjust_numel` | 每次提高或降低 proxy budget 的元素数。 |
| `proxy_budget_min_numel` | proxy budget 下限；`0` 表示没有显式下限。 |
| `proxy_budget_max_numel` | proxy budget 上限；`0` 时使用 `max_proxy_update_numel_per_group`。 |

### 3.11 Kernel、内存池与日志

| 字段 | 含义与功能 |
|---|---|
| `cuda_kernels` | 可用时使用 FastOffload 原生 CUDA kernels 执行 proxy update、选中梯度打包/解包和量化。 |
| `pinned_transfer_pool_mb` | FastOffload transfer staging 预留的 pinned host memory 预算（MB）。 |
| `log_profile` | 输出 FastOffload 内部路径的细粒度 timing。独立脚本另外设置 `FAST_OFFLOAD_PROFILE=1` 记录训练阶段 timing。 |
| `log_stats` | 输出层/通道选择、异步任务和传输统计。 |

## 4. 当前 `fastoffload_config.json` 的关键选择

当前配置进入 `quality_mode=transfer_lean` 的 ZenFlow bridge。实际关键项为：

- **生效**：更新周期 `noncritical_update_interval=4`
- **生效**：全局通道预算 `channel_topk_ratio=0.1`
- **生效**：gradient D2H 和 stale parameter H2D 使用 `bit16`
- **生效**：`transfer_lean_selected_only_nonboundary=true`
- **生效**：layerwise top-k，策略为 `outlier_grad_ema`
- **条件生效**：`transfer_lean_skip_nonboundary_cpu_grad_accum=true`；当前
  `gradient_accumulation_steps=1` 时没有实际效果
- **不生效**：通用 `warmup_steps=20`；该路由实际读取 `transfer_lean_warmup_steps=0`
- **不生效**：`critical_layer_ratio=0.5`、`noncritical_gradient_accumulation`、proxy/tile/residual 配置族
- **不生效**：原生 `cpu_worker_backend`、`cuda_kernels`、`pinned_transfer_pool_mb` 配置族

完整字段状态见 [`QUALITY_MODE_CONFIG_MATRIX.md`](./QUALITY_MODE_CONFIG_MATRIX.md)。

## 5. A800 迁移说明

当前 A800 节点上，两个 rank 同时注册多 GB CPU master partition 时，PyTorch/CUDA 在 `Tensor.pin_memory()` 返回 `CUDA error: invalid argument`。这不是数据集错误，也不是模型 OOM。独立脚本默认：

```bash
CONDA_ENV=fastoffload_codex
TORCH_CUDA_ARCH_LIST=8.0
OFFLOAD_PIN_MEMORY=false
```

同时，ZeRO-2 初始化已修正为在 `offload_optimizer.pin_memory=false` 时不再无条件 pin fp32 master partition。H100 环境若确认支持大块 pinned allocation，可将 `OFFLOAD_PIN_MEMORY=true`，并把架构改为 `9.0`。

## 6. 常见调参影响

| 目标 | 建议方向 | 代价/风险 |
|---|---|---|
| 更少传输、更快 | 降低 `channel_topk_ratio`，使用 `bit16`，开启 selected-only | 可能降低收敛质量 |
| 更稳质量 | Transfer-Lean 中提高 top-k ratio 或缩短 update interval；原生 mode 中才可提高 `critical_layer_ratio` | step 时间和传输增加 |
| 减少 boundary 停顿 | 开启 overlap/async boundary，增加 buffer | 内存增加，调度和版本管理更复杂 |
| 避免 tile 长期不更新 | 缩短 coverage window 或提高 round-robin ratio | 重要 tile 的预算占比下降 |
| 降低 pinned 内存 | 减少 pool/bucket/pending 数 | 可能降低复制吞吐或增加等待 |

修改组合前应先查看 `config.py` 的 validator；非法组合会在 DeepSpeed 初始化阶段直接报错。
