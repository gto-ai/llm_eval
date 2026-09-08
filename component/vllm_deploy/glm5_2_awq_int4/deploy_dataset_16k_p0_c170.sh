#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
LOG_FILE="${REPO_ROOT}/log/deploy_glm5_2_awq_int4_dataset_16k_p0_c170.log"
VLLM_BIN="${REPO_ROOT}/venv-vllm/bin/vllm"
MODEL_PATH="/home/coder/workspace/model/GLM-5.2-AWQ-INT4"
PORT="${PORT:-8000}"
MAX_RUNNING_REQUESTS=40
MAX_NUM_BATCHED_TOKENS=16384

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${REPO_ROOT}/log"

exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --served-model-name "glm5_2_awq_int4_dataset_16k_p0" \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --disable-custom-all-reduce \
  --enable-expert-parallel \
  --performance-mode throughput \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 2 \
  --kv-cache-dtype fp8 \
  --max-model-len 17408 \
  --gpu-memory-utilization 0.82 \
  --max-num-seqs "${MAX_RUNNING_REQUESTS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --no-enable-prefix-caching > "${LOG_FILE}" 2>&1
