from collections import deque
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from component.aisbench_request.evaluation import EvaluationRequest
from component.common.evaluation import EvaluationCase
from component.reporting import EvaluationReport


class AisBenchEvaluator:
    def __init__(self, request_rate: float = 0) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.aisbench_bin = self.repo_root / ".venv/bin/ais_bench"
        self.output_root = self.repo_root / "output"
        self.report_root = self.repo_root / "report"
        self.summary_path = self.report_root / "summary.csv"
        self.report = EvaluationReport(
            self.summary_path,
            self.report_root / "report.md",
        )
        self.log_root = self.repo_root / "log/evaluation"
        self.log_path: Path | None = None
        self.host = "127.0.0.1"
        self.port = 8000
        self.gpu_num = 8
        self.request_rate = request_rate
        self.all_concurrencies = tuple(range(4, 65, 4))
        self.deployment_configs = {
            "GLM-5.2-AWQ-INT4": "vllm_ep_fp8kv_bt16384_mtp2_nccl_mr16",
            "GLM-5.2-W4AFP8": "sglang_tp_fp8dsa_cp32768_mtp2",
        }
        self.request = EvaluationRequest()
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self.status = "Stopped"
        self.stop_requested = False
        self.lock = threading.Lock()
        self.cases = self._create_cases()

    def get_models(self) -> tuple[str, ...]:
        models = tuple(dict.fromkeys(case.model for case in self.cases))
        return models

    def get_cases(self) -> tuple[str, ...]:
        cases = tuple(dict.fromkeys(case.dataset_name for case in self.cases))
        return cases

    def get_data_num(self, concurrency: int | float) -> int:
        data_num = self.request.get_data_num(concurrency)
        return data_num

    def get_deployment_config(self, model: str) -> str:
        deployment_config = self.deployment_configs[model]
        return deployment_config

    def is_running(self) -> bool:
        with self.lock:
            thread = self.thread
        running = thread is not None and thread.is_alive()
        return running

    def is_model_ready(self, model: str, dataset_name: str) -> bool:
        evaluation_case = self._get_case(model, dataset_name)
        try:
            served_models = self._get_served_models()
        except RuntimeError:
            ready = False
        else:
            ready = evaluation_case.served_model_name in served_models
        return ready

    def start(self, model: str, dataset_name: str, concurrency: int | float) -> str:
        self.get_data_num(concurrency)
        normalized_concurrency = int(concurrency)
        status = self._start(model, dataset_name, (normalized_concurrency,))
        return status

    def start_all(self, model: str, dataset_name: str) -> str:
        status = self._start(model, dataset_name, self.all_concurrencies)
        return status

    def stop(self) -> str:
        with self.lock:
            self.stop_requested = True
            process = self.process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        status = "Stopping evaluation."
        return status

    def monitor(self) -> tuple[str, str, list[list[Any]]]:
        with self.lock:
            status = self.status
        state = (status, self._read_log(), self._read_summary())
        return state

    def _start(
        self,
        model: str,
        dataset_name: str,
        concurrencies: tuple[int, ...],
    ) -> str:
        evaluation_case = self._get_case(model, dataset_name)
        self.request.validate_dataset(evaluation_case.dataset_path, max(concurrencies))
        if not self.aisbench_bin.exists():
            raise FileNotFoundError(
                f"AISBench is not installed: {self.aisbench_bin}. Run uv sync."
            )
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                status = "An evaluation is already running."
                return status
            self.stop_requested = False
            self.status = f"Starting: {model} / {dataset_name}"

        log_path = (
            self.log_root
            / evaluation_case.dataset_path.parent.name
            / dataset_name
            / "evaluation.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
        self.log_path = log_path
        thread = threading.Thread(
            target=self._run,
            args=(evaluation_case, concurrencies),
            daemon=True,
        )
        with self.lock:
            self.thread = thread
        thread.start()
        status = self.status
        return status

    def _run(
        self,
        evaluation_case: EvaluationCase,
        concurrencies: tuple[int, ...],
    ) -> None:
        try:
            served_models = self._get_served_models()
            if evaluation_case.served_model_name not in served_models:
                raise RuntimeError(
                    f"expected {evaluation_case.served_model_name}, "
                    f"found {served_models}"
                )
        except Exception as error:
            self._append_log(f"ERROR: {error}\n")
            with self.lock:
                self.status = "Evaluation failed."
            return

        had_error = False
        for concurrency in concurrencies:
            with self.lock:
                if self.stop_requested:
                    break
                self.status = (
                    f"Running: {evaluation_case.model} / "
                    f"{evaluation_case.dataset_name} / concurrency {concurrency}"
                )
            try:
                self._run_one(evaluation_case, concurrency)
            except Exception as error:
                had_error = True
                self._append_log(f"ERROR concurrency {concurrency}: {error}\n")

        with self.lock:
            if self.stop_requested:
                self.status = "Evaluation stopped."
            elif had_error:
                self.status = "Evaluation completed with errors."
            else:
                self.status = "Evaluation completed."
            self.process = None

    def _run_one(self, evaluation_case: EvaluationCase, concurrency: int) -> None:
        data_num = self.request.validate_dataset(
            evaluation_case.dataset_path,
            concurrency,
        )
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = (
            self.output_root
            / self._slug(evaluation_case.model)
            / evaluation_case.dataset_name
            / f"{run_id}_c{concurrency}"
        )
        run_dir.mkdir(parents=True)
        benchmark_dataset_path = self._prepare_dataset(
            evaluation_case,
            run_dir,
        )
        config_path = run_dir / "aisbench_config.py"
        self._write_config(
            evaluation_case, concurrency, benchmark_dataset_path, config_path
        )
        self._write_manifest(evaluation_case, concurrency, data_num, run_dir)
        if evaluation_case.prefix_ratio > 0:
            self._warm_prefix(evaluation_case, run_dir)

        command = [
            str(self.aisbench_bin),
            "--mode", "perf",
            "--num-prompts", str(data_num),
            "--num-warmups", "0",
            "--work-dir", str(run_dir / "aisbench"),
            str(config_path),
        ]
        self._append_log(
            f"\nStarting concurrency {concurrency}, prompts {data_num}\n"
        )
        self._run_process(command, run_dir / "client.log")
        result = self._collect_result(evaluation_case, concurrency, data_num, run_dir)
        self._append_summary(result)
        self.report.update()
        self._append_log(
            f"Finished concurrency {concurrency}: {result['sla_status']}, "
            f"TTFT={result['ttft_ms']} ms, TPOT={result['tpot_ms']} ms, "
            f"throughput/GPU={result['throughput_per_gpu_token_s']} token/s\n"
        )

    def _run_process(self, command: list[str], client_log_path: Path) -> None:
        environment = os.environ.copy()
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        with client_log_path.open("w", encoding="utf-8") as client_log:
            process = subprocess.Popen(
                command,
                cwd=self.repo_root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            with self.lock:
                self.process = process
            if process.stdout is not None:
                for line in process.stdout:
                    client_log.write(line)
                    client_log.flush()
                    self._append_log(line)
            return_code = process.wait()
            with self.lock:
                self.process = None

        if self.stop_requested:
            raise RuntimeError("evaluation was stopped")
        if return_code != 0:
            raise RuntimeError(f"AISBench exited with code {return_code}")

    def _collect_result(
        self,
        evaluation_case: EvaluationCase,
        concurrency: int,
        data_num: int,
        run_dir: Path,
    ) -> dict[str, Any]:
        result_path = self._find_file(run_dir / "aisbench", "json")
        metrics_path = self._find_file(run_dir / "aisbench", "csv")
        raw_result = json.loads(result_path.read_text(encoding="utf-8"))
        metric_rows = self._read_metric_rows(metrics_path)
        ttft_ms = self._number(metric_rows["TTFT"]["Average"])
        tpot_ms = self._number(metric_rows["TPOT"]["Average"])
        total_throughput = self._number(
            raw_result["Output Token Throughput"]["total"]
        )
        observed_concurrency = int(raw_result["Max Concurrency"]["total"])
        result = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": evaluation_case.model,
            "dataset": evaluation_case.dataset_name,
            "concurrency": concurrency,
            "data_num": data_num,
            "ttft_ms": round(ttft_ms, 4),
            "tpot_ms": round(tpot_ms, 4),
            "total_output_throughput_token_s": round(total_throughput, 4),
            "throughput_per_gpu_token_s": round(total_throughput / self.gpu_num, 4),
            "ttft_sla_ms": evaluation_case.ttft_sla_ms,
            "tpot_sla_ms": evaluation_case.tpot_sla_ms,
            "sla_status": self._get_sla_status(
                ttft_ms,
                tpot_ms,
                evaluation_case.ttft_sla_ms,
                evaluation_case.tpot_sla_ms,
            ),
            "observed_max_concurrency": observed_concurrency,
            "concurrency_matched": observed_concurrency == concurrency,
            "success_requests": int(raw_result["Success Requests"]["total"]),
            "failed_requests": int(raw_result["Failed Requests"]["total"]),
            "run_dir": str(run_dir),
        }
        (run_dir / "metrics.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return result

    def _write_config(
        self,
        evaluation_case: EvaluationCase,
        concurrency: int,
        dataset_path: Path,
        path: Path,
    ) -> None:
        config = f'''from ais_bench.benchmark.models import VLLMCustomAPI
from ais_bench.benchmark.datasets import GSM8KDataset
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator
from ais_bench.benchmark.summarizers import DefaultPerfSummarizer
from ais_bench.benchmark.calculators import DefaultPerfMetricCalculator

models = [dict(
    attr="service", type=VLLMCustomAPI,
    abbr={evaluation_case.served_model_name!r},
    path={str(evaluation_case.model_path)!r},
    model={evaluation_case.served_model_name!r},
    stream=True, request_rate={self.request_rate!r}, use_timestamp=False, retry=2,
    host_ip={self.host!r}, host_port={self.port},
    max_out_len={evaluation_case.output_tokens}, batch_size={concurrency},
    trust_remote_code=True,
    generation_kwargs=dict(temperature=0.0, ignore_eos=True),
)]

reader_cfg = dict(input_columns=["question"], output_column="answer")
infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template="{{question}}"),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
eval_cfg = dict(
    evaluator=dict(type=Gsm8kEvaluator), pred_role="BOT",
    pred_postprocessor=dict(type=gsm8k_postprocess),
    dataset_postprocessor=dict(type=gsm8k_dataset_postprocess),
)
datasets = [dict(
    abbr={evaluation_case.served_model_name!r}, type=GSM8KDataset,
    path={str(dataset_path)!r}, reader_cfg=reader_cfg,
    infer_cfg=infer_cfg, eval_cfg=eval_cfg,
)]
summarizer = dict(
    attr="performance", type=DefaultPerfSummarizer,
    calculator=dict(
        type=DefaultPerfMetricCalculator,
        stats_list=["Average", "Min", "Max", "Median", "P75", "P90", "P99"],
    ),
)
'''
        path.write_text(config, encoding="utf-8")

    def _prepare_dataset(
        self,
        evaluation_case: EvaluationCase,
        run_dir: Path,
    ) -> Path:
        dataset_path = run_dir / "dataset"
        dataset_path.mkdir()
        train_path = self.repo_root / "data/source/gsm8k/train.jsonl"
        test_path = evaluation_case.dataset_path / "test.jsonl"
        (dataset_path / "train.jsonl").symlink_to(train_path)
        (dataset_path / "test.jsonl").symlink_to(test_path)
        return dataset_path

    def _write_manifest(
        self,
        evaluation_case: EvaluationCase,
        concurrency: int,
        data_num: int,
        run_dir: Path,
    ) -> None:
        manifest = {
            "model": evaluation_case.model,
            "dataset": evaluation_case.dataset_name,
            "served_model_name": evaluation_case.served_model_name,
            "deployment_config": self.get_deployment_config(evaluation_case.model),
            "concurrency": concurrency,
            "data_num": data_num,
            "gpu_num": self.gpu_num,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

    def _warm_prefix(self, evaluation_case: EvaluationCase, run_dir: Path) -> None:
        prefix_path = evaluation_case.dataset_path / "prefix.jsonl"
        with prefix_path.open(encoding="utf-8") as prefix_file:
            prefix = json.loads(next(prefix_file))["question"]
        body = json.dumps({
            "model": evaluation_case.served_model_name,
            "prompt": prefix,
            "max_tokens": 1,
            "temperature": 0.0,
            "ignore_eos": True,
            "stream": False,
        }).encode("utf-8")
        request = Request(
            f"http://{self.host}:{self.port}/v1/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=900) as response:
            response.read()
        (run_dir / "prefix-warmup.status").write_text("success\n", encoding="utf-8")

    def _get_served_models(self) -> list[str]:
        try:
            with urlopen(
                f"http://{self.host}:{self.port}/v1/models", timeout=5
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, URLError) as error:
            raise RuntimeError("model endpoint is not ready") from error
        served_models = [item["id"] for item in payload.get("data", [])]
        return served_models

    def _get_case(self, model: str, dataset_name: str) -> EvaluationCase:
        for evaluation_case in self.cases:
            if evaluation_case.model == model and evaluation_case.dataset_name == dataset_name:
                return evaluation_case
        raise ValueError(f"unsupported evaluation case: {model} / {dataset_name}")

    def _create_cases(self) -> tuple[EvaluationCase, ...]:
        models = {
            "GLM-5.2-AWQ-INT4": "glm5_2_awq_int4",
            "GLM-5.2-W4AFP8": "glm5_2_w4a_fp8",
        }
        datasets = {
            "dataset_128k_p90": (1024, 0.9, 20_000.0),
            "dataset_16k_p0": (1024, 0.0, 10_000.0),
            "dataset_3_5k_p0": (1536, 0.0, 4_000.0),
        }
        cases = []
        for model, model_directory in models.items():
            for dataset_name, values in datasets.items():
                output_tokens, prefix_ratio, ttft_sla_ms = values
                cases.append(EvaluationCase(
                    model=model,
                    model_path=Path("/home/coder/workspace/model") / model,
                    dataset_name=dataset_name,
                    dataset_path=(
                        self.repo_root / "data/generated" / model_directory / dataset_name
                    ),
                    served_model_name=f"{model_directory}_{dataset_name}",
                    output_tokens=output_tokens,
                    prefix_ratio=prefix_ratio,
                    ttft_sla_ms=ttft_sla_ms,
                ))
        return tuple(cases)

    def _append_summary(self, result: dict[str, Any]) -> None:
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.summary_path.exists()
        with self.summary_path.open("a", encoding="utf-8", newline="") as summary_file:
            writer = csv.DictWriter(summary_file, fieldnames=tuple(result))
            if not file_exists:
                writer.writeheader()
            writer.writerow(result)

    def _read_summary(self) -> list[list[Any]]:
        if not self.summary_path.exists():
            return []
        with self.summary_path.open(encoding="utf-8", newline="") as summary_file:
            rows = list(csv.DictReader(summary_file))
        results = [
            [
                row["model"], row["dataset"], int(row["concurrency"]),
                int(row["data_num"]), float(row["ttft_ms"]),
                float(row["tpot_ms"]),
                float(row["throughput_per_gpu_token_s"]), row["sla_status"],
            ]
            for row in reversed(rows[-100:])
        ]
        return results

    def _read_log(self, max_lines: int = 300) -> str:
        log_path = self.log_path
        if log_path is None or not log_path.exists():
            return ""
        with log_path.open(encoding="utf-8", errors="replace") as log_file:
            lines = deque(log_file, maxlen=max_lines)
        log_text = "".join(lines)
        return log_text

    def _append_log(self, text: str) -> None:
        log_path = self.log_path
        if log_path is None:
            raise RuntimeError("evaluation log path is not initialized")
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(text)

    @staticmethod
    def _read_metric_rows(path: Path) -> dict[str, dict[str, str]]:
        with path.open(encoding="utf-8", newline="") as metrics_file:
            rows = {
                row["Performance Parameters"]: row
                for row in csv.DictReader(metrics_file)
            }
        return rows

    @staticmethod
    def _find_file(root: Path, suffix: str) -> Path:
        for path in root.rglob(f"*.{suffix}"):
            text = path.read_text(encoding="utf-8", errors="replace")
            if suffix == "json" and '"Output Token Throughput"' in text:
                return path
            if suffix == "csv" and text.startswith("Performance Parameters"):
                return path
        raise RuntimeError(f"AISBench result {suffix.upper()} was not found")

    @staticmethod
    def _number(value: Any) -> float:
        number = float(str(value).split()[0])
        return number

    @staticmethod
    def _get_sla_status(
        ttft_ms: float,
        tpot_ms: float,
        ttft_sla_ms: float,
        tpot_sla_ms: float,
    ) -> str:
        ttft_failed = ttft_ms > ttft_sla_ms
        tpot_failed = tpot_ms > tpot_sla_ms
        if ttft_failed and tpot_failed:
            return "FAIL_BOTH"
        if ttft_failed:
            return "FAIL_TTFT"
        if tpot_failed:
            return "FAIL_TPOT"
        return "PASS"

    @staticmethod
    def _slug(value: str) -> str:
        slug = value.lower().replace("-", "_").replace(".", "_")
        return slug
