<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- DeepSpeed Team -->

# State

State 模块是梯度、optimizer state 和参数版本的唯一真值来源，用于保证异步更新的正确性。

## 职责

- 跟踪每个 parameter shard 的梯度版本、optimizer 版本和参数版本。
- 管理 pending、ready、committed、aborted 状态转换。
- 防止旧更新覆盖新参数。
- 为 checkpoint 提供可序列化状态。
- 在 overflow、取消或 worker 失败时回滚未提交任务。

## 规划文件

```text
state/
├── README.md
├── manager.py     # StateManager 和状态转换
├── parameter.py   # ParameterState/Version 数据结构
└── checkpoint.py  # save/load 与 drain 协议
```

## 版本模型

```text
gradient_version
       ↓
optimizer_version
       ↓
parameter_version
       ↓
committed_version
```

只有依赖满足且数值检查通过的版本才能成为 `committed_version`。

## 状态样例

```python
state_manager.register_gradient(parameter_id=17, version=8)
state_manager.mark_transfer_complete(parameter_id=17, version=8)
state_manager.mark_optimizer_complete(parameter_id=17, version=8)
state_manager.commit(parameter_id=17, version=8)
```

## Checkpoint 样例

```python
scheduler.pause()
scheduler.drain()
state = state_manager.state_dict()
checkpoint_writer.save(state)
scheduler.resume()
```

第一版建议 checkpoint 前排空所有异步任务，不保存 inflight task。稳定后再考虑恢复未完成任务。

## 不变量

- `committed_version <= parameter_version <= optimizer_version <= gradient_version`。
- 同一个 parameter shard 的提交顺序必须单调递增。
- aborted 版本永远不能再次提交。
- 所有 rank 在 checkpoint 边界必须具有一致的训练 step 语义。
