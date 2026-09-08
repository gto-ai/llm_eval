# vLLM Installation

This project uses a dedicated uv environment for model serving:

- `.venv`: Gradio, dataset generation, and AISBench
- `venv-vllm`: vLLM serving only

The verified serving stack is Python 3.12.12, vLLM 0.23.0, and Transformers
5.16.1. Do not install vLLM into `.venv`.

## Install

From the repository root:

```bash
component/vllm_deploy/setup_env.sh
```

The script creates `venv-vllm`, synchronizes the exact versions from
`component/vllm_deploy/requirements.lock`, prints the vLLM version, and checks
that `GlmMoeDsaForCausalLM` is registered.

Manual equivalent:

```bash
uv venv venv-vllm --python 3.12.12
uv pip sync \
  --python venv-vllm/bin/python \
  component/vllm_deploy/requirements.lock
```

Activation is unnecessary because every script calls the environment's
executable directly.

## Verify

```bash
venv-vllm/bin/vllm --version
uv pip check --python venv-vllm/bin/python
venv-vllm/bin/python -c \
  "import transformers, vllm; print(vllm.__version__, transformers.__version__)"
venv-vllm/bin/python -c \
  "from vllm.model_executor.models import ModelRegistry; assert 'GlmMoeDsaForCausalLM' in ModelRegistry.get_supported_archs()"
```

Expected versions:

```text
vllm 0.23.0
transformers 5.16.1
```

## Deploy

For example:

```bash
component/vllm_deploy/glm5_2_awq_int4/deploy_dataset_128k_p90.sh
```

The deployment scripts use eight GPUs, tensor parallel size 8, and
`MAX_NUM_BATCHED_TOKENS=8192`. The 128K prefix case enables prefix caching;
the p0 cases disable it.

Check readiness with:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
```

Server output is written to the matching file under `log/`. Each new run
overwrites the previous log for that case.

## Errors encountered

### `venv-vllm/bin/vllm: No such file or directory`

The deployment script was started before `venv-vllm` existed. Run:

```bash
component/vllm_deploy/setup_env.sh
```

Do not point the new project at the old Huawei environment or copy a virtual
environment, because virtual-environment launchers can contain absolute paths.

### Transformers v4 deprecation warning

An initial dependency resolution installed Transformers 4.57.6 and vLLM
reported that its v4 code path was deprecated. The known working GLM stack uses
Transformers 5.16.1. Both versions are now pinned in `requirements.in` and the
resolved lockfile.

Restore the correct environment with:

```bash
component/vllm_deploy/setup_env.sh
```

Avoid running an unpinned `uv pip install vllm`, because it can replace the
verified dependency set.

### Engine startup timeout after 600 seconds

The AWQ checkpoint is about 441 GiB and is loaded from NFS. Loading its shards
can take roughly 20 minutes. The default engine timeout can expire while the
workers are still healthy, followed by `BrokenPipe` errors when the API process
exits.

Every AWQ deployment script therefore sets:

```bash
export VLLM_ENGINE_READY_TIMEOUT_S=3600
```

Keep monitoring the log while the model loads. Do not restart merely because
the service is not ready within the first several minutes.

### OOM during CUDA graph initialization

The model's general 1M context configuration is too large for this serving
profile. Use the case-specific limits already present in the scripts:

| Deployment case | Maximum model length |
| --- | ---: |
| `dataset_128k_p90` | 132096 |
| `dataset_16k_p0` | 17408 |
| `dataset_3_5k_p0` | 5120 |

If startup still runs out of memory, check context length, `max-num-seqs`,
`max-num-batched-tokens`, and GPU memory utilization before retrying.

### Incorrect `--quantization awq` assumption

Do not force the legacy AWQ quantization path. vLLM recognizes this checkpoint
as `compressed-tensors` and selects its WNA16/Marlin backend automatically.

## Updating packages

Edit `component/vllm_deploy/requirements.in`, then regenerate and apply the
lockfile:

```bash
uv pip compile \
  component/vllm_deploy/requirements.in \
  --python-version 3.12 \
  --output-file component/vllm_deploy/requirements.lock
component/vllm_deploy/setup_env.sh
```

After an update, repeat the version, package compatibility, and GLM registry
checks before starting a benchmark.

## References

- [vLLM quickstart](https://docs.vllm.ai/en/stable/getting_started/quickstart/)
- [vLLM serve arguments](https://docs.vllm.ai/en/stable/cli/serve.html)
