#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# LOCAL_MODEL_NAME_OR_PATH="/data/qwen/Qwen2___5-14B-Instruct"
LOCAL_MODEL_NAME_OR_PATH="/data/Qwen2.5-7B-Instruct"
LOCAL_DATASET_PATH="/data/hangyu/datasets/alpaca"
LOCAL_OUTPUT_DIR="/data/hangyu/ResearchHub/FastOffload/src/DeepSpeed/deepspeed/runtime/fastoffload/scripts/output"
LOCAL_HF_HOME=""



# 默认读取同目录的本地路径配置；也可以通过 PATH_CONFIG 指定另一个文件。
PATH_CONFIG=${PATH_CONFIG:-"${SCRIPT_DIR}/local_paths.sh"}
if [[ -f "${PATH_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${PATH_CONFIG}"
fi

# 环境变量优先于 local_paths.sh，方便临时覆盖。
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-${LOCAL_MODEL_NAME_OR_PATH:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}}
DATASET_PATH=${DATASET_PATH:-${LOCAL_DATASET_PATH:-}}
OUTPUT_DIR=${OUTPUT_DIR:-${LOCAL_OUTPUT_DIR:-"${SCRIPT_DIR}/outputs/alpaca-observer"}}
NUM_GPUS=${NUM_GPUS:-2}
FASTOFFLOAD_CONFIG=${FASTOFFLOAD_CONFIG:-"${SCRIPT_DIR}/fastoffload_observer.json"}
if [[ -n "${LOCAL_HF_HOME}" ]]; then
  export HF_HOME=${HF_HOME:-${LOCAL_HF_HOME}}
fi

LAUNCH_ARGS=()
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  LAUNCH_ARGS+=(--num_gpus "${NUM_GPUS}")
fi

TRAIN_ARGS=(
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --deepspeed_config "${SCRIPT_DIR}/deepspeed_zero2_cpu_offload.json"
  --fastoffload_config "${FASTOFFLOAD_CONFIG}"
)
if [[ -n "${DATASET_PATH}" ]]; then
  TRAIN_ARGS+=(--dataset_path "${DATASET_PATH}")
fi

exec deepspeed "${LAUNCH_ARGS[@]}" "${SCRIPT_DIR}/finetune_alpaca.py" "${TRAIN_ARGS[@]}" "$@"
