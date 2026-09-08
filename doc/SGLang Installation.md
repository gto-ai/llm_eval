# SGLang Installation

This project uses `venv-slg` only for SGLang model serving. Keep it separate
from the default `.venv` and `venv-vllm` environments.

The verified serving stack is Python 3.12.12 and SGLang 0.5.13.post1. The
environment is controlled by:

- `component/sglang_deploy/requirements.in`
- `component/sglang_deploy/requirements.lock`
- `component/sglang_deploy/setup_env.sh`
- `component/sglang_deploy/patch_cutlass.py`

## Install

From the repository root:

```bash
component/sglang_deploy/setup_env.sh
```

The script creates `venv-slg`, synchronizes the locked packages, applies the
idempotent CUTLASS compatibility patch, and prints the SGLang version.

Manual equivalent:

```bash
uv venv venv-slg --python 3.12.12
uv pip sync \
  --python venv-slg/bin/python \
  component/sglang_deploy/requirements.lock
venv-slg/bin/python component/sglang_deploy/patch_cutlass.py
```

Activation is unnecessary because the deployment scripts call
`venv-slg/bin/python` directly.

## Verify

```bash
venv-slg/bin/python -c \
  "import sglang; print(sglang.__version__)"
venv-slg/bin/ninja --version
uv pip check --python venv-slg/bin/python
venv-slg/bin/python component/sglang_deploy/patch_cutlass.py
```

Expected primary version:

```text
SGLang 0.5.13.post1
```

The CUTLASS check should report either `applied` or `already present`.

## Deploy

For example:

```bash
component/sglang_deploy/glm5_2_w4a_fp8/deploy_dataset_128k_p90.sh
```

The W4AFP8 deployment scripts use eight GPUs and tensor parallel size 8. They
include the model-required options:

```text
--quantization w4afp8
--disable-shared-experts-fusion
--kv-cache-dtype fp8_e4m3
--reasoning-parser glm45
--tool-call-parser glm47
--trust-remote-code
```

The `dataset_128k_p90` case keeps SGLang's RadixAttention prefix cache enabled.
The p0 cases pass `--disable-radix-cache`.

Check readiness with:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
```

Server output is written to the matching file under `log/`. Each new run
overwrites the previous log for that case.

## Errors encountered

### `venv-slg/bin/python: No such file or directory`

The deployment script was started before the dedicated environment existed.
Create and synchronize it with:

```bash
component/sglang_deploy/setup_env.sh
```

Do not point this project at the old Huawei environment or copy an existing
virtual environment, because its launchers can contain absolute paths.

### `uv pip sync`: unexpected `--prerelease` argument

We initially passed `--prerelease allow` to `uv pip sync` and received:

```text
error: unexpected argument '--prerelease' found
```

Prerelease selection belongs to dependency resolution with `uv pip compile`.
It is not a `uv pip sync` option. The lockfile already contains exact versions,
so installation must use:

```bash
uv pip sync \
  --python venv-slg/bin/python \
  component/sglang_deploy/requirements.lock
```

### CUTLASS MLIR destructor API incompatibility

The first SGLang compilation can fail because CUTLASS DSL's
`llvm.mlir_global_dtors` operation has different schemas across bundled MLIR
versions. Some versions require a `data` attribute, while the locked environment
exposes only `dtors` and `priorities`. Passing `data=[]` to that binding fails
with:

```text
mlir_global_dtors() got an unexpected keyword argument 'data'
```

Apply the project patch after every environment synchronization:

```bash
venv-slg/bin/python component/sglang_deploy/patch_cutlass.py
```

The patch detects the installed operation signature and applies the matching
form. It is safe to run repeatedly.

`setup_env.sh` performs this step automatically. If the patch reports an
unexpected CUTLASS layout, do not edit `site-packages` manually; restore the
locked environment and investigate the dependency change.

### Ninja installed but not found

CUTLASS and FlashInfer launch compilation subprocesses. Installation alone is
not enough when `venv-slg/bin` is absent from `PATH`.

Every SGLang deployment script therefore exports:

```bash
export PATH="${REPO_ROOT}/venv-slg/bin:${PATH}"
```

Verify Ninja before deployment:

```bash
venv-slg/bin/ninja --version
```

### Long pause during the first startup

The first W4AFP8 launch compiles nvcc/Ninja, FlashInfer, DeepGEMM, and CUDA
graph artifacts. The log may remain quiet for a long time while GPU workers are
still active. This is not automatically a failure. Wait for `/health` and
inspect whether the server process is still running.

Later launches should reuse the compilation cache and start faster.

### OOM with the general 1M context configuration

The 1M context configuration can run out of memory while allocating static KV
cache or CUDA graph buffers. Use the tested case settings:

| Deployment case | Context length | Static memory fraction |
| --- | ---: | ---: |
| `dataset_128k_p90` | 132096 | 0.85 |
| `dataset_16k_p0` | 17408 | 0.85 |
| `dataset_3_5k_p0` | 5120 | 0.85 |

If startup still fails, check context length, maximum running requests, static
memory fraction, and whether another process is using the GPUs.

### `torchao` invalid escape-sequence warning

The first import may print a Python `SyntaxWarning` from a `torchao` regular
expression. In the verified environment this warning does not prevent SGLang
from importing or serving. Treat it as non-fatal unless it is followed by an
exception or a nonzero process exit.

## Updating packages

Edit `component/sglang_deploy/requirements.in`, then resolve prerelease
dependencies into a new lockfile:

```bash
uv pip compile \
  component/sglang_deploy/requirements.in \
  --python-version 3.12 \
  --prerelease allow \
  --output-file component/sglang_deploy/requirements.lock
component/sglang_deploy/setup_env.sh
```

After an update, repeat package compatibility, CUTLASS, Ninja, import, health,
and minimal request checks before benchmarking.

## References

- [SGLang installation](https://github.com/sgl-project/sglang/blob/main/docs/docs/get-started/install.mdx)
- [SGLang server arguments](https://docs.sglang.ai/advanced_features/server_arguments.html)
