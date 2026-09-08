#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SGLANG_PYTHON="${REPO_ROOT}/venv-slg/bin/python"

if [[ ! -x "${SGLANG_PYTHON}" ]]; then
  uv venv "${REPO_ROOT}/venv-slg" --python 3.12.12
fi

uv pip sync --python "${SGLANG_PYTHON}" "${SCRIPT_DIR}/requirements.lock"
"${SGLANG_PYTHON}" "${SCRIPT_DIR}/patch_cutlass.py"
"${SGLANG_PYTHON}" -c "import sglang; print('SGLang', sglang.__version__)"
