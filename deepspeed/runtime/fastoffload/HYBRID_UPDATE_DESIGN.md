# FastOffload Hybrid Sparse/Dense Update Design

## 1. Scope

This design targets ZeRO Stage 2 with CPU optimizer offload. It combines high-frequency sparse updates with a low-frequency dense refresh after a fixed warmup and importance-selection phase.

For every two-dimensional parameter, pretrained-delta column selection creates three disjoint groups:

- **A / first-important**: rank `[0, K)`, updated on GPU every optimizer step.
- **B / second-important**: rank `[K, 2K)`, accumulated for `update_interval` steps and updated asynchronously on CPU.
- **C / dense remainder**: all other columns, evaluated and updated only at a dense boundary.

One-dimensional parameters are treated as dense C parameters in the first implementation.

DeepSpeed gradient accumulation and the hybrid update interval are separate concepts. The first implementation requires DeepSpeed GAS=1. A hybrid dense boundary occurs every `update_interval` completed optimizer steps.

## 2. Step semantics

Let the hybrid interval be `N`.

### 2.1 Non-boundary step

```text
selective backward computes A + B weight gradients
  A → distributed reduction → GPU SelectedAdam update
  B → distributed reduction → active accumulation buffer
  C → not computed
```

### 2.2 Dense boundary step

```text
full backward computes A + B + C
  A → distributed reduction → GPU SelectedAdam update
  B → add to active B accumulator
  C → boundary-only CPU gradient

freeze active B buffer
swap to the other accumulation buffer
submit mean(B[1:N]) + C[N] to asynchronous CPU updater
```

The B update uses the mean of all N B gradients. The C update uses only the current boundary gradient. A must be excluded from the CPU job to prevent a duplicate update.

## 3. Accumulation device

`accumulation_device` is configurable and defaults to `cpu`.

### 3.1 CPU accumulation

```text
compressed B gradient
  → pinned D2H staging
  → wait for copy-complete event
  → CPU active_buffer[B] += gradient
```

This uses little GPU memory but transfers B every step.

### 3.2 GPU accumulation

```text
GPU active_buffer[B] += gradient
```

At a dense boundary, the frozen sum is copied to CPU once. This reduces D2H submissions but consumes GPU memory and creates a concentrated boundary transfer.

## 4. Double-buffer state machine

Each slot follows this state machine:

```text
FREE
  → ACCUMULATING
  → FROZEN
  → COPYING_TO_CPU
  → UPDATING
  → COPYING_TO_GPU
  → READY_TO_COMMIT
  → FREE
```

Only one slot may be `ACCUMULATING`. A slot being updated or copied must never be reused for accumulation. If no free slot exists at a boundary, the default `overdue_policy=wait` applies backpressure until the oldest update reaches a safe commit point.

## 5. Asynchronous CPU update

A frozen job owns:

```text
HybridUpdateJob
  version
  interval_steps
  B accumulated gradients
  C boundary gradients
  optimizer hyperparameters
  source buffer id
```

The CPU worker updates CPU parameter shadows and optimizer state without modifying live GPU model tensors. Completion produces an immutable result with `version + 1`.

## 6. Safe GPU commit

A worker must not mutate GPU parameters from its background thread. Results are committed by the training thread at a safe pre-forward boundary:

```text
worker result ready
  → selected parameter H2D on transfer stream
  → record completion event
  → training stream waits for event
  → scatter B/C columns into live GPU parameters
  → publish committed version
```

Training may use stale B/C values while a CPU job is running. At most `max_async_lag` uncommitted versions are allowed. Dense boundaries reuse Native ZeRO-2 gradient-ready buckets: every layer still computes its full gradient, but each reduced owner fragment is split immediately. A/B remain on GPU, while C is copied directly into its version-owned shared+pinned BF16 gradient slot and the dense fragment is released with the Native bucket. No full-model GPU gradient or GPU staging tensor is created. A final CUDA event gates one native DeepSpeedCPUAdam call in the worker. Large CPUAdam jobs cast updated FP32 masters once into a persistent shared+pinned model-dtype return buffer. Publication copies selected values nonblocking from this low-precision mirror and records an explicit visibility event; the next forward waits on that event, and the CPU process cannot reuse the return buffer until visibility completes. The FP32 master and moments remain canonical CPU state. Bucket membership is computed from the global logical payload rather than rank-local owner counts so every rank issues exactly the same collective sequence. Checkpoint and shutdown paths drain and publish one globally ready version at a time. The training thread never waits for CPUAdam during ordinary steps unless bounded-lag backpressure is reached.

## 7. Optimizer ownership

A and B/C use separate optimizer state:

- A uses GPU FP32 selected master values, `exp_avg`, and `exp_avg_sq`.
- B/C use CPU FP32 selected master values, `exp_avg`, and `exp_avg_sq`.
- Weight decay and momentum are applied only when a group is updated.
- Unselected C columns do not change between dense boundaries.

The optimizer step counters for A and B/C are independent because A updates every step while B/C update every N steps.

## 8. Distributed mapping

The compressed representation uses global parameter column indices. Before communication, the ZeRO-2 adapter maps selected elements to partition owners. Communication must operate on packed owner-contiguous buffers and must not recreate a full zero-filled gradient.

A GPU update needs identical reduced A gradients on all data-parallel ranks, or a sharded A optimizer followed by a compressed parameter all-gather. The initial design uses replicated A optimizer state for simplicity and treats sharded A state as a later memory optimization.

B/C CPU updates are partitioned by the existing ZeRO owner mapping. Updated values are packed, exchanged, and scattered into replicated model parameters.

## 9. Overflow, norm, and clipping

Hybrid updates define two logical clipping domains:

- A: reduced and clipped every step.
- B/C: B mean plus C boundary gradient, reduced and clipped when the CPU job is submitted.

Any overflow invalidates the affected logical update. A buffer containing invalid gradients must be cleared rather than submitted. No rank may commit a version rejected by another rank.

## 10. Configuration

```json
{
  "hybrid_update": {
    "enabled": true,
    "update_interval": 8,
    "accumulation_device": "cpu",
    "dense_boundary_enabled": true,
    "double_buffer": true,
    "second_gradient_reduction": "mean",
    "max_async_lag": 2,
    "pt_reserved_cores_perc": 0.25,
    "overdue_policy": "wait"
  }
}
```

The feature requires importance selection. The first end-to-end version also requires ZeRO-2, CPU optimizer offload, fixed selected indices, GAS=1, one CPU worker, and one pending CPU update.

## 11. Implementation layers

1. **Core state machine**: device-selectable double accumulator, immutable jobs/results, async worker, versioned safe commit.
2. **Selected optimizers**: exact selected-column AdamW for GPU and CPU shadows.
3. **Gradient capture**: compressed A/B side channel and dense-boundary switching.
4. **ZeRO adapter**: owner mapping and packed collectives.
5. **Step takeover**: norm/overflow, GPU A update, CPU job submission, and safe commit.
6. **Performance path**: pinned pools, transfer streams, bucket packing, and telemetry.

Layers 1 and 2 are independently testable and do not depend on ZeRO private fields. `HybridUpdateRuntime` executes the complete A/B/C schedule when supplied with pre-reduced compressed gradients: A SelectedAdam runs on the live parameter device, B uses the configured double accumulator, and B/C update through the asynchronous CPU worker before a versioned training-thread commit. Layers 3–5 are enabled only after distributed parity tests pass; there must be no silent fallback that claims hybrid mode while running the original dense optimizer.

## 12. Current implementation status

Implemented and tested:

- `HybridColumnLayout` and `CompressedColumnGradient` packing/scatter.
- `ParameterPartitionLayout` mapping each selected matrix element to its ZeRO group, owner rank, and local flat-partition offset, including parameters crossing partition boundaries.
- `PackedCompressedGradients` and `Zero2CompressedGradientReducer`, which concatenate selected values by ZeRO process group and all-reduce only the compressed buffers.
- A disabled-by-default `compressed_collective_shadow` hook at ZeRO's existing gradient-ready event. It communicates A+B on ordinary hybrid steps and all matrix columns on interval boundaries, while deliberately retaining the native ZeRO reduction and optimizer path for rollback and parity validation.
- Parameter-bounded compressed buckets. A bucket flushes when it reaches `compressed_bucket_bytes`; completed values are released after native and owner validation instead of retaining a full-model packed gradient.
- Direct owner-value parity against native reduced gradients before D2H and against the ZeRO FP32 flat partition after D2H.
- Per-backward compressed/native finite-value, overflow-decision, and L2-norm parity checks.
- CPU and GPU `DoubleBufferedGradientAccumulator`.
- Strict buffer-state transitions and overwrite prevention.
- `HybridUpdateCoordinator` with one background CPU worker, B mean/sum reduction, version ordering, async lag checks, and training-thread commits.
- `SelectedColumnAdamW` for live-device A values and CPU B/C shadows.
- `HybridUpdateRuntime` executing non-boundary and boundary schedules from pre-reduced compressed gradients.
- Configuration validation and an explicit runtime error that prevents a hybrid config from silently using the original dense ZeRO optimizer.

Not yet connected to the live ZeRO-2 path:

- Capturing compressed A/B gradients without returning a full dense autograd gradient.
- Dense-boundary switching for all parameter types.
- Replacing the existing dense ZeRO gradient hooks with the implemented compressed collective and owner mapping.
- Overflow/global norm/clipping collectives.
- Stream/event-based pinned CPU accumulation and H2D commits.
- Replacing the original `ZeroOptimizer.step()` with `HybridUpdateRuntime.step()`.

Until these items pass distributed parity tests, `hybrid_update.enabled=true` intentionally raises during ZeRO controller creation rather than running with misleading dense semantics.
