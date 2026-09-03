<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- DeepSpeed Team -->

# Workers

Worker 执行 CPU 侧梯度累积和 optimizer 更新。它不决定哪些梯度重要，也不直接读取 DeepSpeed ZeRO 对象。

## 职责

- 等待 TransferTask 对应的 copy event 完成。
- 在 CPU 上累积梯度或执行 optimizer step。
- 维护 worker 生命周期、错误传播和优雅退出。
- 将完成结果和版本通知 State Manager/Scheduler。
- 后续支持 CPU core 与 NUMA 绑定。

## 当前实现

`inline.py` 已实现 Phase 2 Inline Worker：它在训练主线程中消费已经完成同步 D2H 的 staging buffer，并复制到 ZeRO 的 FP32 CPU gradient partition。该过程记录 `inline_worker_ms`，不创建线程或进程，也不改变原始 CPU optimizer step。

## 规划文件

```text
workers/
├── README.md
├── base.py      # OptimizerWorker 接口
├── inline.py    # 同进程同步 baseline（已实现）
└── process.py   # 独立 CPU worker process
```

## 目标接口样例

```python
class OptimizerWorker:
    def start(self): ...
    def submit(self, task): ...
    def flush(self): ...
    def close(self): ...
```

## Inline Worker 样例

```python
worker.submit(update_task)
worker.flush()  # 返回时任务已经完成
```

Inline Worker 用于最早期正确性验证，不提供计算重叠。

## Process Worker 样例

```python
worker.start()
worker.submit(update_task)
result = worker.poll_completed()
state_manager.mark_optimizer_complete(result.parameter_id, result.version)
worker.close()
```

## 进程模型约束

- 不在 fork 后重新初始化 accelerator runtime。
- 队列传递稳定句柄，不复制大型 tensor。
- worker 异常必须传播到训练主进程。
- 每个 rank 的 worker 应绑定到独立物理核心，并考虑 NUMA locality。
- `close()` 必须停止接收任务、排空或取消队列并回收共享资源。
