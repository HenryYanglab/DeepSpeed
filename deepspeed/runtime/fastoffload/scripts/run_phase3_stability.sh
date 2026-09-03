#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export RESULT_DIR=${RESULT_DIR:-"${SCRIPT_DIR}/benchmark_results/stability_${TIMESTAMP}"}
export MODES=async
export REPEATS=${REPEATS:-1}
export MEASURE_STEPS=${MEASURE_STEPS:-1000}
export WARMUP_STEPS=${WARMUP_STEPS:-10}
export LOG_INTERVAL=${LOG_INTERVAL:-50}

bash "${SCRIPT_DIR}/run_fastoffload_benchmarks.sh"

if grep -R -E "NCCL.*(error|Error)|Traceback \(most recent call last\)|RuntimeError:" "${RESULT_DIR}"/*/repeat_*/train.log; then
  echo "Stability run completed with error signatures in its logs" >&2
  exit 1
fi

echo "Phase 3 stability run passed: ${RESULT_DIR}"
