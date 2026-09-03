#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR=${RESULT_DIR:-"${SCRIPT_DIR}/shadow_validation_results/${TIMESTAMP}"}
NUM_GPUS=${NUM_GPUS:-2}
MAX_STEPS=${MAX_STEPS:-4}
IMPORTANCE_WARMUP_STEPS=${IMPORTANCE_WARMUP_STEPS:-2}
UPDATE_INTERVAL=${UPDATE_INTERVAL:-2}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
MAX_LENGTH=${MAX_LENGTH:-128}
MAX_SAMPLES=${MAX_SAMPLES:-$((MAX_STEPS * NUM_GPUS * MICRO_BATCH_SIZE * 2))}
LOSS_TOLERANCE=${LOSS_TOLERANCE:-1e-5}
COMPRESSED_BUCKET_BYTES=${COMPRESSED_BUCKET_BYTES:-134217728}

if [[ "${CONDA_DEFAULT_ENV:-}" != "fastoffload2" ]]; then
  echo "Activate the fastoffload2 conda environment before validation" >&2
  exit 1
fi
if (( MAX_STEPS < IMPORTANCE_WARMUP_STEPS + UPDATE_INTERVAL )); then
  echo "MAX_STEPS must cover importance warmup and at least one full hybrid interval" >&2
  exit 1
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
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
    echo "Need ${NUM_GPUS} idle GPUs but found ${#idle_gpus[@]}" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${idle_gpus[*]}")
  export CUDA_VISIBLE_DEVICES
fi
if [[ -e "${RESULT_DIR}" ]]; then
  echo "Result directory already exists: ${RESULT_DIR}" >&2
  exit 1
fi
mkdir -p "${RESULT_DIR}/native" "${RESULT_DIR}/shadow"

python - "${SCRIPT_DIR}/fastoffload_hybrid_shadow.json" "${RESULT_DIR}" \
  "${IMPORTANCE_WARMUP_STEPS}" "${UPDATE_INTERVAL}" "${COMPRESSED_BUCKET_BYTES}" <<'PY'
import json
import sys
from pathlib import Path

template_path = Path(sys.argv[1])
result_dir = Path(sys.argv[2])
warmup_steps = int(sys.argv[3])
update_interval = int(sys.argv[4])
bucket_bytes = int(sys.argv[5])
base = json.loads(template_path.read_text(encoding="utf-8"))
base["importance"]["warmup_steps"] = warmup_steps
base["hybrid_update"]["update_interval"] = update_interval
base["hybrid_update"]["compressed_bucket_bytes"] = bucket_bytes
for mode in ("native", "shadow"):
    config = json.loads(json.dumps(base))
    config["hybrid_update"]["enabled"] = mode == "shadow"
    config["hybrid_update"]["compressed_collective_shadow"] = mode == "shadow"
    config["telemetry"]["jsonl_path"] = str(result_dir / mode / "telemetry.jsonl")
    config["telemetry"]["csv_path"] = str(result_dir / mode / "telemetry.csv")
    (result_dir / mode / "fastoffload.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY

cat >"${RESULT_DIR}/experiment.env" <<EOF
NUM_GPUS=${NUM_GPUS}
MAX_STEPS=${MAX_STEPS}
IMPORTANCE_WARMUP_STEPS=${IMPORTANCE_WARMUP_STEPS}
UPDATE_INTERVAL=${UPDATE_INTERVAL}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}
MAX_LENGTH=${MAX_LENGTH}
MAX_SAMPLES=${MAX_SAMPLES}
LOSS_TOLERANCE=${LOSS_TOLERANCE}
COMPRESSED_BUCKET_BYTES=${COMPRESSED_BUCKET_BYTES}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-}
DATASET_PATH=${DATASET_PATH:-}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
EOF

for mode in native shadow; do
  run_dir="${RESULT_DIR}/${mode}"
  echo "[Shadow Validation] mode=${mode} result=${run_dir}"
  FASTOFFLOAD_CONFIG="${run_dir}/fastoffload.json" \
  OUTPUT_DIR="${run_dir}/model" \
  NUM_GPUS="${NUM_GPUS}" \
  bash "${SCRIPT_DIR}/run_alpaca.sh" \
    --max_samples "${MAX_SAMPLES}" \
    --max_steps "${MAX_STEPS}" \
    --num_train_epochs 1 \
    --micro_batch_size "${MICRO_BATCH_SIZE}" \
    --gradient_accumulation_steps 1 \
    --max_length "${MAX_LENGTH}" \
    --log_interval 1 \
    --benchmark_warmup_steps 0 \
    --benchmark_jsonl "${run_dir}/benchmark.jsonl" \
    --no-save_model \
    2>&1 | tee "${run_dir}/train.log"
done

python "${SCRIPT_DIR}/validate_hybrid_shadow.py" \
  --native-log "${RESULT_DIR}/native/train.log" \
  --shadow-log "${RESULT_DIR}/shadow/train.log" \
  --native-telemetry "${RESULT_DIR}/native/telemetry.jsonl" \
  --shadow-telemetry "${RESULT_DIR}/shadow/telemetry.jsonl" \
  --output "${RESULT_DIR}/validation_summary.json" \
  --loss-tolerance "${LOSS_TOLERANCE}"

echo "Shadow validation passed: ${RESULT_DIR}/validation_summary.json"
