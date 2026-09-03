<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- DeepSpeed Team -->

# Schedulers

Scheduler 将 Policy 的决策转换为具有明确依赖关系的传输、累积、更新和提交任务。

## 职责

- 排列 D2H copy、CPU optimizer 和参数提交任务。
- 管理最大并发任务数和队列反压。
- 处理 gradient accumulation boundary 与 optimizer update boundary。
- 在 backward 结束、step 开始和 checkpoint 前执行必要的 flush。
- 保证 buffer 在通信或传输完成前不会被复用。

## 当前实现

`synchronous.py` 已实现 Phase 2 单任务调度器。`overlap.py` 已实现 Phase 3 event-driven FIFO：提交前非阻塞 progress，按 task 数和 bytes 双重反压，并在 optimizer step 前 flush 当前梯度版本。

## 规划文件

```text
schedulers/
├── README.md
├── base.py          # OffloadScheduler 抽象接口
├── synchronous.py   # 同步 baseline（已实现）
└── overlap.py       # backward/copy 重叠（已实现）
```

## 同步 Scheduler 样例

```python
scheduler.submit(context, decision)
scheduler.flush_backward()
scheduler.prepare_step()
scheduler.commit_step()
```

同步版本每次等待传输与更新完成，主要用于建立数值正确性基线。

## Overlap Scheduler 样例

```python
for context in reduced_gradients:
    decision = policy.decide(context)
    scheduler.submit(context, decision)

# backward 结束只提交已满足依赖的任务，不一定等待 CPU optimizer。
scheduler.finish_backward()

# step 前只等待当前参数版本所必需的任务。
scheduler.prepare_step(required_version=step_id)
```

## 关键约束

- 队列必须有界，不能无限占用 pinned memory。
- Overflow 或 global norm 尚未确认时，不能不可逆地提交更新。
- checkpoint、异常退出和训练结束必须执行 drain/close。
- Scheduler 只能通过 State Manager 查询版本，不能自行维护第二套版本真值。
