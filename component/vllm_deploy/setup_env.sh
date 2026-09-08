#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VLLM_PYTHON="${REPO_ROOT}/venv-vllm/bin/python"

if [[ ! -x "${VLLM_PYTHON}" ]]; then
  uv venv "${REPO_ROOT}/venv-vllm" --python 3.12.12
fi

uv pip sync --python "${VLLM_PYTHON}" "${SCRIPT_DIR}/requirements.lock"
"${REPO_ROOT}/venv-vllm/bin/vllm" --version
"${VLLM_PYTHON}" -c \
  "from vllm.model_executor.models import ModelRegistry; assert 'GlmMoeDsaForCausalLM' in ModelRegistry.get_supported_archs(); print('GLM architecture registered')"
