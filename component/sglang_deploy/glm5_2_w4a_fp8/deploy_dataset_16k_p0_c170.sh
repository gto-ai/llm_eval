#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
LOG_FILE="${REPO_ROOT}/log/deploy_glm5_2_w4a_fp8_dataset_16k_p0_c170.log"
SGLANG_PYTHON="${REPO_ROOT}/venv-slg/bin/python"
MODEL_PATH="/home/coder/workspace/model/GLM-5.2-W4AFP8"
PORT="${PORT:-8000}"
MAX_RUNNING_REQUESTS=40
CHUNKED_PREFILL_SIZE=32768

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PATH="${REPO_ROOT}/venv-slg/bin:${PATH}"
mkdir -p "${REPO_ROOT}/log"

exec "${SGLANG_PYTHON}" -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --served-model-name "glm5_2_w4a_fp8_dataset_16k_p0" \
  --trust-remote-code \
  --tp 8 \
  --dsa-prefill-backend flashmla_sparse_q8 \
  --dsa-decode-backend flashmla_kv \
  --attn-cp-size 8 \
  --enable-dsa-prefill-context-parallel \
  --dsa-prefill-cp-mode round-robin-split \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 2 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 3 \
  --context-length 17408 \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
  --disable-radix-cache \
  --quantization w4afp8 \
  --disable-shared-experts-fusion \
  --kv-cache-dtype fp8_e4m3 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --mem-fraction-static 0.80 > "${LOG_FILE}" 2>&1
