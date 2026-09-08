# LLM Evaluation

This project deploys GLM-5.2 models with vLLM or SGLang, benchmarks the
OpenAI-compatible endpoint with AISBench, and generates a Markdown performance
report through a Gradio UI.

## Environments

The project intentionally uses three separate Python environments:

| Environment | Purpose | Setup command |
| --- | --- | --- |
| `.venv` | Gradio UI, AISBench, dataset generation, and reporting | `uv sync --locked` |
| `venv-vllm` | vLLM serving for `GLM-5.2-AWQ-INT4` | `./component/vllm_deploy/setup_env.sh` |
| `venv-slg` | SGLang serving for `GLM-5.2-W4AFP8` | `./component/sglang_deploy/setup_env.sh` |

Keeping the serving engines separate avoids dependency conflicts between
AISBench, vLLM, and SGLang.

## Prerequisites

- Linux with `bash`
- `git`, [Git LFS](https://git-lfs.com/), and
  [uv](https://docs.astral.sh/uv/)
- Python 3.12 (the serving setup scripts request Python 3.12.12)
- CUDA and NVIDIA drivers compatible with the locked serving packages
- Eight visible NVIDIA GPUs
- Enough disk space for the environments, model weights, datasets, and results

The included deployment scripts use GPUs `0,1,2,3,4,5,6,7` with tensor
parallelism 8. Adjust the scripts before running on a different GPU layout.

## 1. Clone the repository and download datasets

The generated JSONL datasets are stored with Git LFS. Install LFS before
pulling them:

```bash
git lfs install
git clone git@github.com:gto-ai/llm_eval.git
cd llm_eval
git lfs pull
```

Verify that the large files are present rather than LFS pointer files:

```bash
git lfs ls-files --size
```

## 2. Install the AISBench and UI environment

From the repository root, create `.venv` and install the locked project
dependencies:

```bash
uv sync --locked
```

This environment contains AISBench, Gradio, Hugging Face Hub, Transformers, and
the `llm-eval-ui` command. Verify it with:

```bash
./.venv/bin/ais_bench --help
./.venv/bin/python -c "import gradio; import ais_bench; print(gradio.__version__)"
```

## 3. Install the vLLM environment

Create `venv-vllm`, install the versions in
`component/vllm_deploy/requirements.lock`, and verify GLM architecture support:

```bash
./component/vllm_deploy/setup_env.sh
```

The setup script performs the equivalent of:

```bash
uv venv venv-vllm --python 3.12.12
uv pip sync --python venv-vllm/bin/python component/vllm_deploy/requirements.lock
```

Verify the installation at any time with:

```bash
./venv-vllm/bin/vllm --version
```

## 4. Install the SGLang environment

Create `venv-slg`, install the locked SGLang dependencies, apply the local
CUTLASS compatibility patch, and verify the installation:

```bash
./component/sglang_deploy/setup_env.sh
```

The setup script performs the equivalent of:

```bash
uv venv venv-slg --python 3.12.12
uv pip sync --python venv-slg/bin/python component/sglang_deploy/requirements.lock
./venv-slg/bin/python component/sglang_deploy/patch_cutlass.py
```

Verify the installation at any time with:

```bash
./venv-slg/bin/python -c "import sglang; print(sglang.__version__)"
```

## 5. Prepare model weights

The deployment scripts expect these exact directories:

```text
/home/coder/workspace/model/GLM-5.2-AWQ-INT4
/home/coder/workspace/model/GLM-5.2-W4AFP8
```

If the paths do not match your machine, update `MODEL_PATH` in the scripts
under:

```text
component/vllm_deploy/glm5_2_awq_int4/
component/sglang_deploy/glm5_2_w4a_fp8/
```

On the expected `/home/coder/workspace` layout, both supported models can be
downloaded with:

```bash
./.venv/bin/python model/download_model.py
```

The model downloads are large. Confirm the destination and available disk space
before starting them.

## 6. Start the UI

Run the combined deployment and evaluation UI from the repository root:

```bash
./.venv/bin/llm-eval-ui
```

The application binds to `0.0.0.0`. Open the URL printed in the terminal,
normally:

```text
http://127.0.0.1:7860
```

When running on a remote machine, use your IDE/Coder port forwarding or an SSH
tunnel to expose the printed Gradio port. Model serving uses port `8000`, which
must be free.

## 7. Deploy a model from the UI

Open the **Deploy** tab:

1. Select `GLM-5.2-AWQ-INT4` to use vLLM, or `GLM-5.2-W4AFP8` to use SGLang.
2. Select the deployment case that matches the dataset you will evaluate.
3. Click **Deploy**.
4. Watch **Status** and **Deployment log** until the status changes from
   `Starting` to `Running`.

Regular deployment cases are used for concurrency 4 through 64. The `_c170`
cases are tuned for the special concurrency-170 run.

Use **Stop** before switching to a different model or deployment case. Only one
managed model server can use port 8000 at a time.

## 8. Run AISBench from the UI

Open the **Evaluate** tab and select the same model and base dataset used in the
Deploy tab:

- `dataset_128k_p90`: 128K input workload with 90% shared prefix
- `dataset_16k_p0`: 16K input workload without a shared prefix
- `dataset_3_5k_p0`: 3.5K input workload without a shared prefix

Available actions:

- **Run** evaluates one concurrency. Supported values are 4, 8, ..., 64, and
  170. The prompt count is automatically set to `concurrency × 4`.
- **Run All (4–64)** evaluates the selected model and dataset at every
  concurrency from 4 through 64. Deploy the matching regular case first.
- **Auto Run** deploys each required model/case, runs missing concurrency points
  from 4 through 64 plus 170, updates the report, and stops each deployment.
- **Stop** requests that the active evaluation, auto run, and managed deployment
  stop.
- **Generate Report** rebuilds the Markdown report from the latest rows in the
  summary CSV.

The **AISBench log** and **Results** table update while an evaluation is running.
Do not start a manual deployment or evaluation while Auto Run is active.

## 9. View reports and artifacts

Each successful evaluation automatically updates both files below:

```text
report/summary.csv
report/report.md
```

- `report/summary.csv` is the complete machine-readable result history.
- `report/report.md` is the latest comparison table across model, dataset, and
  concurrency. Each result shows TTFT, TPOT, and output throughput per GPU.

Open the Markdown report directly:

```bash
less report/report.md
```

Or click **Generate Report** in the Evaluate tab and open the path displayed in
the **Report** field.

Additional runtime artifacts are written to:

```text
output/<model>/<dataset>/<timestamp>_c<concurrency>/
log/evaluation/<model>/<dataset>/evaluation.log
log/deploy_<model>_<dataset>.log
log/auto_run.log
```

Each output directory contains the generated AISBench configuration, run
manifest, client log, normalized `metrics.json`, and raw AISBench results. The
`output/` and `log/` directories are intentionally ignored by Git; generated
reports under `report/` are not ignored.

## Troubleshooting

- **AISBench is not installed**: run `uv sync --locked` and confirm
  `.venv/bin/ais_bench` exists.
- **Deployment exits immediately**: check the Deployment log and verify the
  selected engine environment, model path, CUDA setup, and all eight GPUs.
- **Model endpoint is not ready**: wait for deployment status to become
  `Running` and verify `curl http://127.0.0.1:8000/v1/models`.
- **Port 8000 is occupied**: stop the other model server before using Deploy or
  Auto Run.
- **Dataset appears to contain only an LFS pointer**: install Git LFS and run
  `git lfs pull`.
- **SGLang build reports that `ninja` is missing**: rerun the SGLang setup script;
  `ninja` is included in its locked environment.
