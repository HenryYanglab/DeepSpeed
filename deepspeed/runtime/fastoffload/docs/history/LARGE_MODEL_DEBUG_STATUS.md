<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- DeepSpeed Team -->

# Large-model FastOffload debugging status

## Retained changes

- The compressed microbatch accumulator accepts explicit ownership transfer for GAS=1. The capture pipeline relinquishes its tensor maps immediately afterward, avoiding zeros_like/add copies of captured gradients.
- Boundary extraction returns owned storage and applies mean scaling in place when necessary, avoiding a second result allocation. Default accumulation still isolates caller storage.
- CPU host-registration errors now report numeric CUDA status, allocation bytes, slot, and address.
- Hybrid unit tests: 32 passed, including new sum/mean ownership and caller-isolation coverage. This does not establish large-model end-to-end success.

## Large-model probes

Qwen14B (4 ranks) and Gemma27B (8 ranks) were retried with 16-microstep targets, original LR=5e-6 and cutoff512. Neither completed. Whole-buffer registration returned CUDA error code 1 (invalid value), for example 6,646,476,470-byte Qwen buffers and 6,125,845,866-byte Gemma buffers.

Standalone single-GPU registration of the same allocation sizes succeeded. Thus these observations do not establish a simple allocation-size limit or insufficient host RAM as the cause. Multi-rank/runtime allocation history remains to be isolated.

An experimental 1-GiB registration-chunk variant let Qwen reach logged step 4, but a later dense-boundary D2H copy failed with invalid argument. Gemma still failed registration. This variant was reverted: changing registration boundaries alone is not a validated fix, and transfers spanning independently registered regions need careful investigation.

## Remaining work

1. Reproduce registration and D2H with the same shared-buffer lifecycle across 4/8 ranks in isolation, including copy ranges and registration boundaries.
2. Capture immediate CUDA error state before registration and distinguish prior asynchronous errors from registration failure.
3. Validate any segmented allocation/registration solution with boundary-crossing D2H/H2D, ordered publication, cleanup, and error rollback tests.
4. Re-measure Gemma peak after host transfer failures are resolved. The GAS=1 optimization removes redundant copies but has not yet demonstrated a successful 27B training run or quantified peak saving.

No A/B/C scheduling, importance ratio, learning rate, precision, or sequence length was changed in the retained patch. Do not label the large-model issue as fixed yet.

## GPU-only isolation diagnostic

The separate `scripts/benchmark_gpu_ceiling.py` diagnostic has now completed 100 microsteps on Qwen7B (2 ranks),
Qwen14B (4 ranks), and Gemma27B (8 ranks, `expandable_segments:True`). It freezes B/C after importance warmup,
disables their CPU state/jobs/transfers/publication and B accumulation, and preloads input batches to GPU.
This is not valid Hybrid training and is never installed by the production path.

- Qwen7B: 40.61 s for the measured 80 microsteps, 427.34 padded tokens/s.
- Qwen14B: 83.96 s, 402.49 padded tokens/s. This isolates a working GPU-side path, not a host registration fix.
- Gemma27B default allocator still failed in owner-padded reduce-scatter workspace allocation (866 MiB request).
  The allocator-only retry completed: 350.72 s, 205.73 padded tokens/s, maximum-rank allocated peak 73.30 GiB.

All successful windows include 60 ordinary and 20 dense steps. Runtime guards confirm zero B/C CPU jobs and transfers
in the timed window; initial native importance warmup and A state migration are excluded. At the time of this GPU-only probe, CPU registration and full
B/C publication were still unverified; the subsequent Qwen14B full-path checks are recorded below.

Raw records: `/data/hangyu/ResearchHub/FastOffload/benchmark_results/gpu_ceiling_20260907_140050/`.

## Qwen14B full-path revalidation and registration recovery

The current full Hybrid path completed two 16-microstep probes and an audited 100-microstep run on four A800s.
No CPU work was disabled. Settings remained BF16, GAS=1, cutoff512, LR=5e-6, seed42, A/B ratios 0.1/0.1,
interval4, and `max_async_lag=2`. These runs did **not** reproduce the original registration error, including a
run without extra registration synchronization. Therefore the initial code-1 failure's underlying cause remains unconfirmed.

The audited 100-step run measured 80 steps in 604.08 s (55.94 all-rank padded tokens/s), with rank0 allocated peak
46.96 GiB. It performed 99 successful Hybrid A steps after one Native warmup step. All ranks submitted and committed
24 B/C CPU versions, drained their last pending job, exited the CPU process with code 0, and unregistered all transfer
buffers. End-only, bounded-scratch comparison checked all 14,770,033,664 model elements bitwise across ranks.
The final partial B interval is not converted into an extra dense update. Timing excludes final drain and replica checking;
these numbers are debugging measurements, not sustained performance or quality evidence.

Independent four-rank probes also passed whole-buffer registration and D2H/H2D checks at the start, around 1/2/4 GiB,
and at the tail of 6,646,476,470-byte allocations, both with and without Native pinned/shared allocation history.
A three-rank large-buffer probe subsequently exercised the recovery using real CUDA errors while only three GPUs were idle.

`owner_cpu_update.py` now retries code 1 once using a fresh **whole** shared mapping, retaining the failed mapping until
replacement allocation. It synchronizes first and explicitly consumes the expected CUDA last error (0/1); other errors
abort with cleanup. A real zero-byte host-registration probe confirmed that merely retrying successfully leaves code 1
pending, causing the next otherwise valid GPU kernel to fail. This fixes a concrete error-recovery hazard, but does not
establish why the historical full-size registration initially failed. No registration chunks or full-model GPU staging are
introduced. `takeover_host_register_retry_count` and a warning expose recovery rather than silently masking it.

Regression coverage now includes real CUDA registration-error injection in the spawned CPUAdam path, two version slots,
three committed B/C versions, dtype/cleanup assertions, bounded retry/rollback, rejection of unrelated/asynchronous errors,
and the existing overflow/checkpoint tests: 45 passed. Full-model checkpoint restore is not covered by this 100-step audit.

Artifacts: `/data/hangyu/ResearchHub/FastOffload/benchmark_results/qwen14b_repair_20260907_162651/`.
The return-code-only `fault_audit_100` probe was deliberately stopped after finding an audit that read a reset interval
metric; it is not a successful 100-step result. The corrected `real_cuda_fault_audit_100` uses a real zero-length CUDA
registration and a persistent initialization record. It subsequently completed all 100 steps on available GPUs 1,2,3,4:
each rank recovered once, committed all 24 CPU versions, passed the full-model bitwise replica check, and exited with zero
pending jobs/registered buffers. Its 80-step window was 515.82 s (65.51 padded tokens/s). The different GPU placement and
shared-node load mean the difference from the ordinary run must not be attributed to an initialization-only retry change.
This validates recovery from injected real CUDA errors, not a reproduction of the historical initial failure's cause.

## Qwen7B performance check after registration recovery

A subsequent two-A800 comparison ran current FastOffload, a control reverting only registration strategy, and ZenFlow.
Each completed three 100-microstep repetitions, with 20 excluded warmup microsteps and balanced method order.
All six FastOffload runs registered normally (zero recovery attempts). Full CPU work and normal input transfers remained
active; no GPU-only diagnostic was used.

| Method | Pooled padded tokens/s | Per-run range | Maximum-rank allocated peak |
| --- | ---: | ---: | ---: |
| Current FastOffload | 169.89 | 165.69–174.46 | 30.60 GiB |
| Legacy-registration control | 166.89 | 161.51–172.88 | 30.60 GiB |
| ZenFlow | 83.97 | 82.36–86.69 | 36.50 GiB |

Current FastOffload achieved 2.023x ZenFlow's steady-window throughput, saving 5.907 GiB/rank (16.18%) allocated memory.
Paired current-versus-control window-time changes were -7.42%, +0.59%, and +1.85%; this is not a consistent regression
or evidence that an initialization-only change accelerates the hot path. Shared-node short runs do not resolve tiny effects.
Including algorithm startup and final publication drain, the full 100-step training loops averaged 218.84 s versus
293.68 s for ZenFlow (1.342x), excluding model/engine initialization and process teardown. ZenFlow's actual GAS=4 was
normalized to microsteps/tokens; these measurements do not establish algorithmic or training-quality equivalence.

Records and plots: `/data/hangyu/ResearchHub/FastOffload/benchmark_results/qwen7b_recovery_comparison_20260908_111425/`.
The original workbook contains all nine records and editable charts in `11_7B恢复后对比`.

## Subsequent Qwen14B method remeasurement

On four idle A800s (physical4–7), the current full FastOffload path completed three more 100-microstep runs,
with 20 excluded benchmark-warmup microsteps, BF16, cutoff512, LR5e-6, seed42 and the default allocator.
All-rank padded throughput was 75.14, 77.02 and 76.31 tokens/s (pooled76.15); pooled input throughput was73.78 tokens/s.
The maximum-rank80-step windows averaged443.77 s and allocated GPU peak was46.87 GiB/rank. Full100-step
training loops including algorithm startup and final publication drain averaged724.57 s, excluding model/engine
initialization and process teardown. All12 rank/run audits recorded99 successful Hybrid steps,24 committed B/C
versions, zero registration retries, no pending work or registered buffers after close, and CPU worker exit0.
These repeats did not perform held-out validation, full-model bitwise replica checking, or checkpoint restore.

ZenFlow was retried with the same four-card workload and again failed before training during parameter flattening:
27.51 GiB requested with only22.14–22.33 GiB free (55.31 GiB already allocated). Further identical repetitions were
skipped. Its transposed-contiguous temporaries add memory beyond the base flatten-output budget; no workaround was
applied. ZenFlow throughput and the FastOffload/ZenFlow speedup remain unmeasured, not extrapolated from7B or historical Native runs.

Artifacts: `/data/hangyu/ResearchHub/FastOffload/benchmark_results/qwen14b_method_comparison_20260908_134119/`.
The original workbook preserves the failed attempt and three successes, with charts in `12_14B完整路径复测`.

### ZenFlow selective-state offload enabled

A separate same-GPU14B attempt changed only `zero_optimization.zenflow.offload` from its default false to true.
Config validation confirmed the flag alongside the existing `offload_optimizer.device=cpu` and `overlap_comm=true`.
It still failed before training at the same parameter-flatten allocation:27.51 GiB requested,22.14–22.33 GiB free.
This switch pages selective Adam moments between CPU and GPU during updates; it does not offload the full model
weights or alter the earlier base-constructor flattening peak. The selective-state paging path was not reached, so
its training-memory saving and throughput remain unmeasured. No initialization workaround was applied.

Artifacts: `/data/hangyu/ResearchHub/FastOffload/benchmark_results/qwen14b_zenflow_state_offload_20260908_143503/`.
Workbook RUN043 records this flag-enabled failure separately from the default-mode failure.

An eight-A800 retry (physical0–7) retained selective-state offload=true and microbatch1 per GPU, targeting GAS4
and effective batch32. All eight ranks again failed in initialization:27.51 GiB requested with22.14–22.32 GiB free
and55.31 GiB already allocated. No training steps completed. ZeRO-2 keeps full model weights per rank, so increasing
DP ranks does not reduce this full-weight transpose/flatten allocation. This was not fixed-global-batch scaling,
and no initialization or algorithm workaround was applied. Workbook RUN044 and `13_ZenFlow卸载与卡数` retain the result.
Artifacts: `/data/hangyu/ResearchHub/FastOffload/benchmark_results/qwen14b_zenflow_state_offload_8gpu_20260908_145047/`.

## ZenFlow initialization budget corrected: 8 and 4 cards now train

The preceding failures were an implementation peak, not a hardware inability to fine-tune14B. ZenFlow's initialization
kept original BF16 weights, contiguous transposed weights and the concatenated output live together (roughly82.5 GiB
for14B), but the placement check budgeted only the output. `zero/stage_1_and_2.py` now also budgets the necessary
2D transpose copies. Insufficient headroom selects the existing CPU flatten-and-copy path. Native's budget is unchanged;
no training algorithm, precision, optimizer, ratio or allocator workaround was introduced.

The final regression suite passed42 tests:18 existing Native flatten checks,18 new memory-budget cases and6 config tests.
A two-rank BF16 offload fixture failed the placement assertion before the patch and passed afterward. New coverage checks
Native/ZenFlow, selective-state offload on/off, low/flat-only/ample headroom, FP32/BF16, exact initialized parameters,
detached GPU flats, host-placeholder cleanup and subsequent parameter updates.

| Corrected ZenFlow, offload=true | Effective batch | Completed microsteps | Padded tokens/s | Maximum-rank allocated peak |
| --- | ---: | ---: | ---: | ---: |
| 8 A800s, physical0–7 | 32 | 100 | 139.79 | 46.53 GiB |
| 4 A800s, physical4–7 | 16 | 100 | 69.35 | 49.99 GiB |

Each run excludes20 warmup microsteps and uses GAS4, microbatch/GPU1, BF16, cutoff512, LR5e-6 and seed42.
Maximum-rank80-step windows were490.96 s and487.29 s; full100-loop+drain times were665.81 s and669.55 s.
The latter exclude model/engine initialization (including CPU flattening), process teardown and the end-only audit.
These are single performance probes, not fixed-global-batch scaling or held-out quality measurements.

Both runs verified pinned CPU selective-moment buffers (about5.50 GiB/rank), released GPU moment references after updates,
all14,770,033,664 model elements bitwise equal across ranks after publication, and all CPU workers exiting0.
The full-replica audit uses bounded4M-element scratch outside benchmark timing. No checkpoint restore was tested.

Earlier four-card FastOffload's three-run pooled throughput was76.15 padded tokens/s versus69.35 for this one corrected
ZenFlow run (about1.10x). This is a sequential reference comparison, not an interleaved repeated benchmark, and their GAS
and selective-state placements differ. Do not use the eight-card result against four-card FastOffload as a method speedup.

Artifacts: `/data/hangyu/ResearchHub/FastOffload/benchmark_results/qwen14b_zenflow_flatten_fix_20260908_151203/`.
Workbook RUN045/RUN046 and `14_ZenFlow初始化修复` record these successes separately; original failed attempts remain intact.

## Qwen7B ZenFlow initialization-patch performance control

Seven additional100-microstep runs on GPUs6,7 separated initialization budgeting from selective-state residency.
Three before/after offload=false repeats used frozen copies of the single ZeRO module, interleaved without rolling back
the working tree. Per-rank audits verified the loaded source and GPU flattening in both versions. One extra offload=true
run was a separate state-residency diagnostic. Settings stayed BF16, cutoff512, LR5e-6, seed42, microbatch1 and actual GAS4;
20 microsteps were excluded and all-rank token counts matched the earlier seven-billion-parameter-model window.

| ZenFlow condition | Repetitions | Pooled padded tokens/s | Range | Maximum-rank allocated peak |
| --- | ---: | ---: | ---: | ---: |
| Earlier offload=false reference | 3 | 83.97 | 82.36–86.69 | 36.50 GiB |
| Before patch, offload=false, new control | 3 | 84.99 | 81.73–87.12 | 36.50 GiB |
| After patch, offload=false | 3 | 84.15 | 82.52–85.75 | 36.50 GiB |
| After patch, offload=true diagnostic | 1 | 68.52 | Single run | 32.24 GiB |

After-patch throughput was+0.21% versus the historical reference and-0.99% versus the contemporaneous before-patch control.
Paired window-time changes were-4.69%,+4.61%,+3.42%; these short shared-node runs do not resolve tiny causal effects.
There is no added training operation or different initialization placement for7B, and allocated peaks were identical.
The separate offload=true probe reduced allocated peak by4.27 GiB but measured18.57% lower throughput than the
three-run after/offload=false result. That changes training transfers and cannot be attributed to the initialization fix;
it also needs repetitions before a sustained tradeoff claim. CPU/GPU moment residency and worker exit0 were verified.

Artifacts: `/data/hangyu/ResearchHub/FastOffload/benchmark_results/qwen7b_zenflow_init_patch_comparison_20260908_162607/`.
Workbook RUN047–RUN053 and `15_7B_ZF补丁影响` preserve the measurements, separately from earlier results.

## Workbook-directed P0 batch started on2026-09-09

The user requested execution according to the original workbook. Its sheet00 prioritizes real quality together with
performance and requires a fixed batch within comparison groups. A new batch-matched Qwen7B/Alpaca pilot therefore uses
actual GAS4/global batch8 for all three methods, rather than relabeling historical FastOffload GAS1 versus ZenFlow GAS4.
FastOffload retains Hybrid A/B/C10/10/80%; its interval4 now spans16 microsteps. ZenFlow is explicitly the initialization-memory
fix version with selective-state offload=false. No production algorithm was changed for this experiment batch.

Artifacts and the live supervisor are under:
`/data/hangyu/ResearchHub/FastOffload/benchmark_results/workbook_p0_20260909_144320/`.
`PROTOCOL.md` and `protocol.json` define the frozen split, timing, source/model hashes, sampling, gates and run order.

- Frozen tokenized split:49,920 train,1,023 validation,1,024 test; normalized identical prompts cannot cross splits.
  Evaluate the same first512 validation records/29,706 response tokens; test remains unused. Near duplicates and
  pretraining contamination have not been exhaustively excluded. Historical all-train runs are not held-out results.
- Four run-local CPU tests passed: split/group isolation, nonduplicating eval sharding, shifted response-token NLL,
  and HF loss/model-only BF16 column-major export/reload. Preliminary harness failures and fixes remain in `checks/`.
- BASE response-token validation NLL was2.0746921265. Three36-microstep gates completed training, final CPU publication,
  bounded full-replica bitwise checks, CPU-contiguous model export with every saved tensor checked, and fresh-process
  ordinary HF evaluation. Gate NLL: Native1.3285775206, FO1.3940274219, ZenFlow1.3780680009. These are short-budget
  observations, not convergence or equivalence claims. Optimizer-state checkpoint resume was not tested.
- The gated next batch has3 methods × training seeds42/43/44,1028 microsteps each,132 excluded and896 measured.
  FastOffload has one Native optimizer warmup plus256 Hybrid steps, so all64 B/C intervals finish without an invented partial
  flush. This is a shared-LR5e-6 pilot, not LR-tuned full-epoch/time-to-quality evidence or3 same-seed timing repetitions.
- `03_运行结果` RUN054–RUN056 are successful gates; RUN057–RUN065 are the registered main pilot runs, updated only
  as they execute. `04_质量评估` now contains actual BASE/exported-model validation values. Sheet16 tracks this batch.
  `summary.json`, `completion.json` and the workbook, not this start-time note, determine subsequent completion status.

The serial supervisor uses idle GPUs6,7, measures all-rank input tokens/max-rank GPU/time and sampled process-tree PSS,
checks cleanup, and updates the original workbook after each result. PSS includes initialization/export and apportions
shared pages; it is not a pinned-live peak. Final-loop/drain, export and fresh-model eval are separate timings.
P1 capacity/profiling/available ablations and P2 scaling remain subsequent stages, not implicitly completed by this batch.

## User-requested CPU-period4 short comparison on2026-09-11

The user requested FastOffload A updates every microstep and B/C updates every4 microsteps, rather than the P0 GAS4
configuration. Native keeps GAS4/CPU period4; ZenFlow keeps its native per-microstep selective updates and CPU period4.
FastOffload therefore uses GAS1. Effective batches are2 versus8, explicitly not a batch-matched comparison.

Three new100-microstep runs completed on GPUs6,7, seed42, the same frozen Alpaca split, BF16/cutoff512/LR5e-6.
The first20 microsteps were excluded; each measured window processed16,041 real input tokens. External PSS/NVML
sampling was disabled. Other jobs were present on other GPUs, so shared-host interference remains a limitation.

| Method | Input tokens/s | Maximum-rank allocated GiB |100-loop+drain seconds | Validation response NLL |
| --- | ---: | ---: | ---: | ---: |
| Native GAS4 |77.547 |23.726 |273.787 |1.294144 |
| FastOffload GAS1 |152.302 |30.595 |224.066 |1.286321 |
| ZenFlow + initialization fix, offload=false |73.820 |36.502 |308.679 |1.305682 |

Successful method-call audits on both ranks verified99 GPU A updates at microsteps2–100 after one Native importance
warmup, and24 B/C submissions at5,9,...97. Native/ZenFlow CPU submissions occurred at4,8,...100. In the measured
window all three methods submitted20 CPU updates, and FastOffload/ZenFlow each updated selected GPU parameters on
all80 microsteps. Periods agree but the warmup phase offset is retained; the last3 accumulated B steps do not invent
an extra C update. No production algorithm change was needed.

FastOffload window throughput ratios were1.964× Native and2.063× ZenFlow. Full-loop+drain ratios were1.222× and1.378×;
these exclude model/engine initialization, auditing, export/evaluation and process teardown. All ranks passed complete
replica bitwise checks and cleanup; every exported model tensor was checked, followed by fresh HF evaluation on the
same512 validation samples. This remains a single-seed short pilot, not convergence or optimizer-resume evidence.

Artifacts: `/data/hangyu/ResearchHub/FastOffload/benchmark_results/cpu_period4_short_20260911_160116/`.
Workbook RUN066–RUN068, CFG046–CFG048 and `17_CPU周期4短测` preserve these results separately from P0.
The preceding resource-monitor diagnostic retained two successful arms; its final repeat was interrupted to prioritize
this user request. That incomplete diagnostic is not a finished three-arm comparison; its `STOP_REASON.md` preserves why.

## ZenFlow initialization fix disabled/enabled on2026-09-11

The user questioned whether the initializer repair explained the difference from the historical approximately2×
ZenFlow/Native throughput ratio. Workbook RUN007/RUN009 indeed report43.256/87.954 padded tokens/s, but Native used
GAS1/CPU updates every microstep there. The recent CPU-period4 comparison used Native GAS4, measuring80.365 padded
tokens/s (RUN066). These are different reference configurations, not evidence that the initializer repair halved ZenFlow speed.

Two additional100-microstep ZenFlow runs,20 excluded, reused the recent frozen input sequence, GPUs6,7, seed42,
GAS4/batch8, LR5e-6 and selective offload=false, without PSS/NVML sampling. Only the initializer budget differed:

| ZenFlow condition | Input tokens/s | Padded tokens/s | Maximum-rank allocated GiB | Validation NLL |
| --- | ---: | ---: | ---: | ---: |
| Initialization fix disabled |71.342 |73.935 |36.502 |1.305950 |
| Initialization fix enabled |71.439 |74.036 |36.502 |1.305886 |

The enabled/disabled throughput difference was+0.136% in this single pair. Both versions actually flattened on GPU
on both ranks. Constructor source and spawned CPU-worker source hashes were audited; only the frozen single module
was selected, with no working-tree rollback. The bootstrap imports the training entry point normally so that it remains
the multiprocessing main module and spawned workers reinstall the same source selector.

Each run observed100 selected GPU update microsteps and25 CPU submissions, and passed final publication drain,
full replica equality, saved-tensor verification, ordinary HF reload evaluation and worker exit0. Exported checkpoint
shards were not identical between the two runs; no cross-run bitwise determinism is claimed. The tiny NLL difference
is not a convergence comparison, and its cause was not isolated.

Together with the earlier three-pair initializer control on worksheet15, these observations do not support a large
initializer-induced slowdown. They do not fully explain the remaining absolute-throughput difference from historical
ZenFlow runs: data order/window and resource conditions also differ. Native GAS1 was not rerun in this pair; the latest
CPU-period4 comparison remains intact.

Artifacts: `/data/hangyu/ResearchHub/FastOffload/benchmark_results/zenflow_init_toggle_short_20260911_165356/`.
Workbook RUN069–RUN070, CFG049–CFG050 and `18_7B_ZF关闭修复` keep these controls and reference distinctions explicit.
The disabled repair is a variant of this project implementation, not a claim of an otherwise unmodified upstream checkout.

## Corrected baseline: Native default cadence on2026-09-11

The user explicitly clarified the intended method-native comparison: **keep Native ZeRO-Offload at GAS1, with a full
CPU update every microstep; match only FastOffload/ZenFlow GPU period1 and CPU period4.** Subsequent comparisons
using this requested baseline must not silently use Native GAS4. Earlier Native GAS4 runs remain valid separate
accumulation/CPU-period diagnostics, not the default Native denominator.

Three fresh100-microstep runs completed,20 excluded, on GPUs6,7 with the same frozen Qwen7B/Alpaca sequence,
BF16/cutoff512/LR5e-6 and seed42. Native optimizer/offload settings match the historical GAS1 JSON except common
logging frequency. "Default" here specifies the established Native baseline, not framework defaults for every hyperparameter.
ZenFlow again disabled only the initializer repair; FastOffload retained the complete A/B/C algorithm.

| Method | Actual GAS | Input tokens/s | Padded tokens/s | Max allocated GiB |100-loop+drain seconds | Validation NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native default cadence |1 |42.169 |43.702 |23.749 |477.435 |1.320807 |
| FastOffload |1 |157.576 |163.303 |30.595 |222.751 |1.285603 |
| ZenFlow, initialization repair disabled |4 |73.899 |76.584 |36.502 |310.803 |1.305886 |

Actual calls on both ranks verified Native100 CPU updates, FastOffload one Native importance warmup plus99 GPU A
updates/24 B/C submissions, and ZenFlow100 selected GPU update microsteps/25 CPU submissions. Measured windows
contained80 Native CPU updates, versus20 CPU submissions and80 GPU selected update microsteps for each selective
method. Startup phase offsets and the final partial B accumulation were retained without inventing another C update.

Window throughput ratios were FastOffload/Native3.737×, ZenFlow/Native1.752× and FastOffload/ZenFlow2.132×.
Full100-loop+drain FastOffload ratios were2.143× Native and1.395× ZenFlow. Loop+drain excludes model/engine
initialization, auditing, export/evaluation and teardown. All three runs used16,041 measured input tokens and passed
full replica equality, saved-model tensor checks, fresh HF validation on512 examples, source/worker audits and cleanup.

These remain single-seed short-budget observations, not convergence or quality-equivalent speedups. Native/FastOffload
have effective batch2, ZenFlow batch8; only the selective methods' update periods match, not their optimizer mathematics.
PSS/NVML sampling stayed disabled. No production algorithm change was made for this corrected comparison.

Artifacts: `/data/hangyu/ResearchHub/FastOffload/benchmark_results/native_default_fo_zf_period4_20260911_184950/`.
Workbook RUN071–RUN073, CFG051–CFG053 and `19_Native默认频率` record the corrected baseline separately.
