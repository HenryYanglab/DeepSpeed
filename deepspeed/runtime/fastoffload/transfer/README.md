<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- DeepSpeed Team -->

# Transfer

Transfer 模块只负责 GPU/CPU 数据移动和传输资源，不参与策略选择。

## Phase 2 已实现

```text
transfer/
├── context.py       # 回调期间有效的 source/destination tensor view
├── buffer_pool.py       # 有界、可复用的 pinned CPU buffer
├── gpu_buffer_pool.py   # Phase 3 稳定 GPU staging buffer
├── event_pool.py        # 有界、可复用的 accelerator event bundles
└── synchronous.py       # 同步 D2H Transfer Engine
```

默认同步数据路径直接写最终目标，避免重复 CPU copy：

```text
reduced local gradient
  → blocking D2H to ZeRO pinned FP32 CPU gradient partition
```

`transfer.cpu_staging=true` 时仍可启用旧的 pinned staging + Inline Worker 路径，用于消融实验，不建议用于性能实验。

Buffer 状态机：

```text
FREE → RESERVED → FILLED → IN_USE → FREE
```

可选 CPU staging pool 延迟分配槽位。遇到超过 `buffer_size` 的 shard 时槽位增长并保留供后续复用，因此不会在每个 step 重复 page-lock；同时槽位总数受 `buffer_count` 限制。异常路径也必须释放 lease，关闭时若仍有活跃 lease 会直接报错。

同步模式记录：

```text
actual_offload_bytes
d2h_copy_count
d2h_copy_ms
d2h_bandwidth_gbps
buffer_pool_in_use
buffer_pool_high_watermark
oversized_buffer_allocations
```

## Phase 3 已实现

`asynchronous.py` 和 `task.py` 提供两种 event 驱动路径。默认 `producer_stream` 在梯度 producer stream 上直接提交 non-blocking D2H，并由 task 持有 source tensor 直到 completion event 完成；同一 stream 的顺序保证 D2H 先于后续 buffer 复用，不需要 GPU staging。`dedicated_stream` 使用 source-ready/copy-complete event 和独立 accelerator copy stream；`gpu_buffer_pool.py` 为该策略复用稳定 GPU staging shard，使原始 ZeRO bucket 可以安全复用。两种路径默认都直接写 ZeRO 最终 pinned CPU partition，启用 CPU staging 消融选项后才额外持有 CPU lease。

Scheduler 通过 event `query()` 非阻塞推进；只有队列反压和 optimizer step flush 会等待单个 completion event，从不调用全局 accelerator synchronize。Event bundle 在任务完成后归还 pool，warmup 后不再为每个 shard 新建 event。`producer_stream` 优先避免 staging 和并发资源争用，但不报告 overlap hidden ratio；`dedicated_stream` 提供真正的 backward/D2H overlap，但可能因 HBM、PCIe 和 NCCL 竞争而降低端到端吞吐。CPU Worker 仅在可选 CPU staging 路径中使用，并且只能在 copy-complete event 完成后读取 buffer。
