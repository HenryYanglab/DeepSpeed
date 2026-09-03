#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR=${RESULT_DIR:-"${SCRIPT_DIR}/benchmark_results/${TIMESTAMP}"}
MODES=${MODES:-"native sync async"}
REPEATS=${REPEATS:-1}
MEASURE_STEPS=${MEASURE_STEPS:-100}
WARMUP_STEPS=${WARMUP_STEPS:-10}
TOTAL_STEPS=$((MEASURE_STEPS + WARMUP_STEPS))
NUM_GPUS=${NUM_GPUS:-2}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-16}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
MAX_LENGTH=${MAX_LENGTH:-512}
LOG_INTERVAL=${LOG_INTERVAL:-10}
MAX_SAMPLES=${MAX_SAMPLES:-$((TOTAL_STEPS * NUM_GPUS * MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))}
ROTATE_MODES=${ROTATE_MODES:-true}
read -r -a MODE_LIST <<<"${MODES}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "fastoffload2" ]]; then
  echo "Activate the fastoffload2 conda environment before benchmarking" >&2
  exit 1
fi
if ! command -v deepspeed >/dev/null 2>&1; then
  echo "deepspeed executable not found in the fastoffload2 environment" >&2
  exit 1
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "CUDA_VISIBLE_DEVICES is unset and nvidia-smi is unavailable" >&2
    exit 1
  fi
  GPU_MAX_USED_MEMORY_MB=${GPU_MAX_USED_MEMORY_MB:-1024}
  GPU_MAX_UTILIZATION=${GPU_MAX_UTILIZATION:-5}
  mapfile -t idle_gpus < <(
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
      awk -F, -v max_memory="${GPU_MAX_USED_MEMORY_MB}" -v max_util="${GPU_MAX_UTILIZATION}" '
        {
          gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3)
          if ($2 <= max_memory && $3 <= max_util) print $1
        }' |
      head -n "${NUM_GPUS}"
  )
  if [[ "${#idle_gpus[@]}" -lt "${NUM_GPUS}" ]]; then
    echo "Need ${NUM_GPUS} idle GPUs but found ${#idle_gpus[@]}; set CUDA_VISIBLE_DEVICES explicitly to override" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${idle_gpus[*]}")
  export CUDA_VISIBLE_DEVICES
  echo "[Benchmark] auto-selected idle physical GPUs: ${CUDA_VISIBLE_DEVICES}"
fi
if [[ -e "${RESULT_DIR}/runs.csv" || -e "${RESULT_DIR}/summary.csv" ]]; then
  echo "Result directory already contains a completed benchmark: ${RESULT_DIR}" >&2
  exit 1
fi
mkdir -p "${RESULT_DIR}"

cat >"${RESULT_DIR}/experiment.env" <<EOF
MODES=${MODES}
REPEATS=${REPEATS}
MEASURE_STEPS=${MEASURE_STEPS}
WARMUP_STEPS=${WARMUP_STEPS}
TOTAL_STEPS=${TOTAL_STEPS}
NUM_GPUS=${NUM_GPUS}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
MAX_LENGTH=${MAX_LENGTH}
MAX_SAMPLES=${MAX_SAMPLES}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-}
DATASET_PATH=${DATASET_PATH:-}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
ROTATE_MODES=${ROTATE_MODES}
EOF

config_for_mode() {
  case "$1" in
    native) echo "${SCRIPT_DIR}/fastoffload_native.json" ;;
    sync) echo "${SCRIPT_DIR}/fastoffload_sync.json" ;;
    async) echo "${SCRIPT_DIR}/fastoffload_async.json" ;;
    selective) echo "${SCRIPT_DIR}/fastoffload_importance.json" ;;
    *) echo "Unknown benchmark mode: $1" >&2; return 1 ;;
  esac
}

for repeat in $(seq 1 "${REPEATS}"); do
  start_index=0
  if [[ "${ROTATE_MODES}" == "true" ]]; then
    start_index=$(((repeat - 1) % ${#MODE_LIST[@]}))
  fi
  for offset in "${!MODE_LIST[@]}"; do
    mode_index=$(((start_index + offset) % ${#MODE_LIST[@]}))
    mode=${MODE_LIST[mode_index]}
    run_dir="${RESULT_DIR}/${mode}/repeat_${repeat}"
    mkdir -p "${run_dir}"
    source_config=$(config_for_mode "${mode}")
    run_config="${run_dir}/fastoffload.json"
    python "${SCRIPT_DIR}/prepare_benchmark_config.py" \
      --input "${source_config}" \
      --output "${run_config}" \
      --run_dir "${run_dir}" \
      --log_interval "${LOG_INTERVAL}"

    echo "[Benchmark] mode=${mode} repeat=${repeat} steps=${TOTAL_STEPS} result=${run_dir}"
    FASTOFFLOAD_CONFIG="${run_config}" \
    NUM_GPUS="${NUM_GPUS}" \
    bash "${SCRIPT_DIR}/run_alpaca.sh" \
      --max_samples "${MAX_SAMPLES}" \
      --max_steps "${TOTAL_STEPS}" \
      --num_train_epochs 1 \
      --micro_batch_size "${MICRO_BATCH_SIZE}" \
      --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
      --max_length "${MAX_LENGTH}" \
      --log_interval "${LOG_INTERVAL}" \
      --benchmark_warmup_steps "${WARMUP_STEPS}" \
      --benchmark_jsonl "${run_dir}/benchmark.jsonl" \
      --no-save_model \
      2>&1 | tee "${run_dir}/train.log"
  done
done

python "${SCRIPT_DIR}/summarize_benchmarks.py" "${RESULT_DIR}"
echo "Benchmark results: ${RESULT_DIR}"
