#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
LOG_FILE="${REPO_ROOT}/log/deploy_glm5_2_awq_int4_dataset_128k_p90.log"
VLLM_BIN="${REPO_ROOT}/venv-vllm/bin/vllm"
MODEL_PATH="/home/coder/workspace/model/GLM-5.2-AWQ-INT4"
PORT="${PORT:-8000}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
MAX_NUM_BATCHED_TOKENS=16384
GPU_MEMORY_UTILIZATION=0.82
DEPLOY_ARGS=(
  --disable-custom-all-reduce
  --enable-expert-parallel
  --performance-mode throughput
  --speculative-config.method mtp
  --speculative-config.num_speculative_tokens 2
  --kv-cache-dtype fp8
)
if [[ "${DEPLOY_PROFILE:-}" == "huawei_cmp" ]]; then
  MAX_NUM_BATCHED_TOKENS=8192
  GPU_MEMORY_UTILIZATION=0.90
  DEPLOY_ARGS=()
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${REPO_ROOT}/log"

exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --served-model-name "glm5_2_awq_int4_dataset_128k_p90" \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  "${DEPLOY_ARGS[@]}" \
  --max-model-len 132096 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${MAX_RUNNING_REQUESTS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --enable-prefix-caching > "${LOG_FILE}" 2>&1
