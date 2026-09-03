#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)
CONDA_ENV=${CONDA_ENV:-deepspeed}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6,7}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-/data/Qwen2.5-7B-Instruct}
DATASET_PATH=${DATASET_PATH:-/data/hangyu/datasets/alpaca}
RESULT_ROOT=${RESULT_ROOT:-/tmp/qwen_alpaca_epoch_comparison_$(date +%Y%m%d_%H%M%S)}
MAX_STEPS=100000
MASTER_PORT=${MASTER_PORT:-29617}

mkdir -p "${RESULT_ROOT}/fastoffload" "${RESULT_ROOT}/zenflow"
export CUDA_VISIBLE_DEVICES
export MASTER_PORT
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python - "${SCRIPT_DIR}" "${RESULT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
result_root = Path(sys.argv[2])
fast = json.loads((script_dir / "fastoffload_qwen7b_alpaca_epoch.json").read_text())
fast["telemetry"]["jsonl_path"] = str(result_root / "fastoffload" / "telemetry.jsonl")
fast["telemetry"]["csv_path"] = str(result_root / "fastoffload" / "telemetry.csv")
(result_root / "fastoffload" / "fastoffload.json").write_text(json.dumps(fast, indent=2) + "\n")

native = json.loads((script_dir / "fastoffload_native.json").read_text())
native["telemetry"]["jsonl_path"] = str(result_root / "zenflow" / "telemetry.jsonl")
native["telemetry"]["csv_path"] = str(result_root / "zenflow" / "telemetry.csv")
(result_root / "zenflow" / "fastoffload.json").write_text(json.dumps(native, indent=2) + "\n")
PY

cat >"${RESULT_ROOT}/experiment.env" <<EOF
CONDA_ENV=${CONDA_ENV}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}
DATASET_PATH=${DATASET_PATH}
SEQUENCE_LENGTH=512
EPOCHS=1
WARMUP_MICROSTEPS=20
MICRO_BATCH_SIZE_PER_GPU=1
WORLD_SIZE=2
MASTER_PORT=${MASTER_PORT}
FASTOFFLOAD_UPDATE_INTERVAL=4
ZENFLOW_UPDATE_INTERVAL=4
EOF

COMMON_ARGS=(
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --dataset_path "${DATASET_PATH}"
  --max_length 512
  --max_steps "${MAX_STEPS}"
  --num_train_epochs 1
  --micro_batch_size 1
  --gradient_accumulation_steps 1
  --learning_rate 2e-5
  --seed 42
  --no-save_model
)

conda run --no-capture-output -n "${CONDA_ENV}" deepspeed "${SCRIPT_DIR}/finetune_alpaca.py" \
  "${COMMON_ARGS[@]}" \
  --output_dir "${RESULT_ROOT}/fastoffload/model" \
  --deepspeed_config "${SCRIPT_DIR}/deepspeed_zero2_cpu_offload.json" \
  --fastoffload_config "${RESULT_ROOT}/fastoffload/fastoffload.json" \
  --log_interval 4 \
  --benchmark_warmup_steps 20 \
  --benchmark_jsonl "${RESULT_ROOT}/fastoffload/benchmark.jsonl" \
  >"${RESULT_ROOT}/fastoffload/train.log" 2>&1

# ZenFlow maps update_interval=4 to GAS=4. Five ZenFlow boundaries therefore equal the same 20 warmup microsteps.
conda run --no-capture-output -n "${CONDA_ENV}" deepspeed "${SCRIPT_DIR}/finetune_alpaca.py" \
  "${COMMON_ARGS[@]}" \
  --output_dir "${RESULT_ROOT}/zenflow/model" \
  --deepspeed_config "${SCRIPT_DIR}/deepspeed_qwen7b_alpaca_zenflow_epoch.json" \
  --fastoffload_config "${RESULT_ROOT}/zenflow/fastoffload.json" \
  --log_interval 1 \
  --benchmark_warmup_steps 5 \
  --benchmark_jsonl "${RESULT_ROOT}/zenflow/benchmark.jsonl" \
  >"${RESULT_ROOT}/zenflow/train.log" 2>&1

conda run --no-capture-output -n "${CONDA_ENV}" python \
  "${SCRIPT_DIR}/summarize_qwen_epoch_comparison.py" "${RESULT_ROOT}" \
  | tee "${RESULT_ROOT}/summary.txt"

echo "Results: ${RESULT_ROOT}"
