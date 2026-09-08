from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import time
from typing import Any

from component.aisbench_request.runner import AisBenchEvaluator
from component.common.deployment import DeploymentManager
from component.common.evaluation import EvaluationCase


class TuningReport:
    def __init__(
        self,
        summary_path: Path,
        report_path: Path,
        datasets: tuple[str, ...],
        max_requests: tuple[int, ...],
        concurrencies: tuple[int, ...],
    ) -> None:
        self.summary_path = summary_path
        self.report_path = report_path
        self.datasets = datasets
        self.max_requests = max_requests
        self.concurrencies = concurrencies

    def update(self) -> Path:
        latest = self._read_latest()
        lines = [
            "| Dataset | Best MAX_RUNNING_REQUESTS | AISBench concurrency | Peak throughput/GPU |",
            "| --- | ---: | ---: | ---: |",
        ]
        for dataset in self.datasets:
            results = [
                value
                for (name, max_requests, _), value in latest.items()
                if name == dataset and max_requests in self.max_requests
            ]
            if results:
                best = max(
                    results,
                    key=lambda row: float(row["throughput_per_gpu_token_s"]),
                )
                lines.append(
                    f'| {dataset} | {best["max_running_requests"]} | '
                    f'{best["concurrency"]} | '
                    f'{float(best["throughput_per_gpu_token_s"]):.2f} tok/s/GPU |'
                )
            else:
                lines.append(f"| {dataset} | - | - | - |")

        for dataset in self.datasets:
            lines.extend([
                "",
                f"## {dataset}",
                "",
                "Cell: TTFT ms / TPOT ms / tok/s/GPU / SLA",
                "",
                "| AISBench concurrency | "
                + " | ".join(f"MR={value}" for value in self.max_requests)
                + " |",
                "| ---: | " + " | ".join("---" for _ in self.max_requests) + " |",
            ])
            for concurrency in self.concurrencies:
                cells = [
                    self._format(latest.get((dataset, value, concurrency)))
                    for value in self.max_requests
                ]
                lines.append(f"| {concurrency} | " + " | ".join(cells) + " |")

            peaks = []
            for value in self.max_requests:
                results = [
                    row
                    for (name, max_requests, _), row in latest.items()
                    if name == dataset and max_requests == value
                ]
                if results:
                    peak = max(
                        results,
                        key=lambda row: float(
                            row["throughput_per_gpu_token_s"]
                        ),
                    )
                    peaks.append(
                        f'{float(peak["throughput_per_gpu_token_s"]):.2f} '
                        f'@ c{peak["concurrency"]}'
                    )
                else:
                    peaks.append("-")
            lines.append("| **Peak** | " + " | ".join(peaks) + " |")

        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return self.report_path

    def _read_latest(
        self,
    ) -> dict[tuple[str, int, int], dict[str, str]]:
        latest: dict[tuple[str, int, int], dict[str, str]] = {}
        if not self.summary_path.exists():
            return latest
        with self.summary_path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                try:
                    key = (
                        row["dataset"],
                        int(row["max_running_requests"]),
                        int(row["concurrency"]),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                latest[key] = row
        return latest

    @staticmethod
    def _format(row: dict[str, str] | None) -> str:
        if row is None:
            return "-"
        return (
            f'{float(row["ttft_ms"]):.1f} / '
            f'{float(row["tpot_ms"]):.1f} / '
            f'{float(row["throughput_per_gpu_token_s"]):.2f} / '
            f'{row["sla_status"]}'
        )


class TuningEvaluator(AisBenchEvaluator):
    def __init__(
        self,
        model: str,
        max_requests: int,
        request_rate: float,
        summary_path: Path,
        output_root: Path,
        log_root: Path,
        report: TuningReport,
        deployment_config: str,
    ) -> None:
        super().__init__(request_rate=request_rate)
        self.max_requests = max_requests
        self.summary_path = summary_path
        self.output_root = output_root / f"mr_{max_requests}"
        self.log_root = log_root
        self.report = report
        self.deployment_configs[model] = deployment_config

    def _collect_result(
        self,
        evaluation_case: EvaluationCase,
        concurrency: int,
        data_num: int,
        run_dir: Path,
    ) -> dict[str, Any]:
        result = super()._collect_result(
            evaluation_case, concurrency, data_num, run_dir
        )
        result["request_rate"] = self.request_rate
        result["max_running_requests"] = self.max_requests
        (run_dir / "metrics.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return result

    def _write_manifest(
        self,
        evaluation_case: EvaluationCase,
        concurrency: int,
        data_num: int,
        run_dir: Path,
    ) -> None:
        super()._write_manifest(
            evaluation_case, concurrency, data_num, run_dir
        )
        path = run_dir / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["request_rate"] = self.request_rate
        manifest["max_running_requests"] = self.max_requests
        path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )


class MaxRunningRequestsBenchmark:
    def __init__(self) -> None:
        self.repo_root = Path(__file__).resolve().parent
        self.model = "GLM-5.2-AWQ-INT4"
        self.datasets = (
            "dataset_3_5k_p0",
            "dataset_16k_p0",
            "dataset_128k_p90",
        )
        self.max_requests = (8, 16, 32, 64)
        self.concurrencies = tuple(range(4, 65, 4))
        self.request_rate = 8
        self.deployment_config = "vllm_ep_fp8kv_bt16384_mtp2_nccl"
        self.poll_interval_s = 2
        self.deployment_timeout_s = 3600

        experiment = f"request_rate_{self.request_rate}"
        self.report_root = self.repo_root / "report/max_running_requests" / experiment
        self.summary_path = self.report_root / "summary.csv"
        self.output_root = self.repo_root / "output/max_running_requests" / experiment
        self.log_root = self.repo_root / "log/max_running_requests" / experiment
        self.run_log_path = self.log_root / "benchmark.log"
        self.report = TuningReport(
            self.summary_path,
            self.report_root / "report.md",
            self.datasets,
            self.max_requests,
            self.concurrencies,
        )
        self.deployment: DeploymentManager | None = None
        self.evaluator: TuningEvaluator | None = None
        self.active_dataset = ""
        self.active_max_requests = 0

    def run(self) -> Path:
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.run_log_path.write_text("", encoding="utf-8")
        completed = self._read_completed()
        total = len(self.datasets) * len(self.max_requests) * len(self.concurrencies)
        self._log(f"Completed {len(completed)}/{total}; starting benchmark.")

        if DeploymentManager().is_server_ready():
            raise RuntimeError(
                "Port 8000 is occupied. Stop the existing model service first."
            )

        try:
            for dataset in self.datasets:
                for max_requests in self.max_requests:
                    pending = tuple(
                        concurrency
                        for concurrency in self.concurrencies
                        if (dataset, max_requests, concurrency) not in completed
                    )
                    if not pending:
                        self._log(f"Skip completed: {dataset} / MR={max_requests}")
                        continue
                    self._run_stage(
                        dataset, max_requests, pending, completed
                    )
        finally:
            self.stop()

        report_path = self.report.update()
        self._log(f"Benchmark completed. Report: {report_path}")
        return report_path

    def stop(self) -> None:
        if self.evaluator is not None and self.evaluator.is_running():
            self.evaluator.stop()
            while self.evaluator.is_running():
                time.sleep(self.poll_interval_s)
        if self.deployment is not None:
            self.deployment.stop()
            self._save_deploy_log()
        self.evaluator = None
        self.deployment = None

    def _run_stage(
        self,
        dataset: str,
        max_requests: int,
        pending: tuple[int, ...],
        completed: set[tuple[str, int, int]],
    ) -> None:
        self.active_dataset = dataset
        self.active_max_requests = max_requests
        self._log(
            f"Deploying {dataset} with MAX_RUNNING_REQUESTS={max_requests}"
        )
        deployment = DeploymentManager()
        self.deployment = deployment

        previous = os.environ.get("MAX_RUNNING_REQUESTS")
        os.environ["MAX_RUNNING_REQUESTS"] = str(max_requests)
        try:
            status = deployment.start(self.model, dataset)
        finally:
            if previous is None:
                os.environ.pop("MAX_RUNNING_REQUESTS", None)
            else:
                os.environ["MAX_RUNNING_REQUESTS"] = previous
        self._log(status)

        evaluator = TuningEvaluator(
            self.model,
            max_requests,
            self.request_rate,
            self.summary_path,
            self.output_root,
            self.log_root / "_current",
            self.report,
            f"{self.deployment_config}_mr{max_requests}",
        )
        self.evaluator = evaluator

        try:
            self._wait_for_deployment(evaluator, dataset)
            for concurrency in pending:
                self._run_evaluation(
                    evaluator, dataset, max_requests, concurrency
                )
                completed.add((dataset, max_requests, concurrency))
        finally:
            if evaluator.is_running():
                evaluator.stop()
                while evaluator.is_running():
                    time.sleep(self.poll_interval_s)
            deployment.stop()
            self._save_deploy_log()
            self.evaluator = None
            self.deployment = None

    def _wait_for_deployment(
        self,
        evaluator: TuningEvaluator,
        dataset: str,
    ) -> None:
        started_at = time.monotonic()
        last_update = -30.0
        while not evaluator.is_model_ready(self.model, dataset):
            if self.deployment is None:
                raise RuntimeError("deployment state was lost")
            status = self.deployment.get_status()
            if status.startswith("Stopped (exit code"):
                raise RuntimeError(status)
            elapsed = time.monotonic() - started_at
            if elapsed >= self.deployment_timeout_s:
                raise TimeoutError("model deployment timed out")
            if elapsed - last_update >= 30:
                self._log(f"{status}; elapsed={elapsed:.0f}s")
                last_update = elapsed
            time.sleep(self.poll_interval_s)
        self._log("Model is ready.")

    def _run_evaluation(
        self,
        evaluator: TuningEvaluator,
        dataset: str,
        max_requests: int,
        concurrency: int,
    ) -> None:
        data_num = evaluator.get_data_num(concurrency)
        self._log(
            f"Evaluating {dataset} / MR={max_requests} / "
            f"request_rate={self.request_rate} / concurrency={concurrency} / "
            f"prompts={data_num}"
        )
        self._log(evaluator.start(self.model, dataset, concurrency))
        try:
            while evaluator.is_running():
                time.sleep(self.poll_interval_s)
            status, _, _ = evaluator.monitor()
            if status != "Evaluation completed.":
                raise RuntimeError(status)
        finally:
            self._save_evaluation_log(
                evaluator, dataset, max_requests, concurrency
            )
        self.report.update()
        self._log("Evaluation completed.")

    def _save_deploy_log(self) -> None:
        deployment = self.deployment
        if deployment is None or deployment.active_target is None:
            return
        source = deployment.active_target.log_path
        if not source.exists():
            return
        destination = (
            self.log_root
            / self.active_dataset
            / f"mr_{self.active_max_requests}"
            / "deploy.log"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    def _save_evaluation_log(
        self,
        evaluator: TuningEvaluator,
        dataset: str,
        max_requests: int,
        concurrency: int,
    ) -> None:
        source = evaluator.log_path
        if source is None or not source.exists():
            return
        destination = (
            self.log_root
            / dataset
            / f"mr_{max_requests}"
            / f"c_{concurrency}"
            / "evaluation.log"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    def _read_completed(self) -> set[tuple[str, int, int]]:
        completed: set[tuple[str, int, int]] = set()
        if not self.summary_path.exists():
            return completed
        with self.summary_path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                try:
                    dataset = row["dataset"]
                    max_requests = int(row["max_running_requests"])
                    request_rate = float(row["request_rate"])
                    concurrency = int(row["concurrency"])
                    data_num = int(row["data_num"])
                    success = int(row["success_requests"])
                    failed = int(row["failed_requests"])
                except (KeyError, TypeError, ValueError):
                    continue
                if (
                    row.get("model") == self.model
                    and dataset in self.datasets
                    and max_requests in self.max_requests
                    and request_rate == self.request_rate
                    and concurrency in self.concurrencies
                    and data_num == concurrency * 4
                    and success == data_num
                    and failed == 0
                ):
                    completed.add((dataset, max_requests, concurrency))
        return completed

    def _log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with self.run_log_path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")


def _stop_on_sigterm(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt


def main() -> None:
    benchmark = MaxRunningRequestsBenchmark()
    signal.signal(signal.SIGTERM, _stop_on_sigterm)
    try:
        report_path = benchmark.run()
        print(f"Report generated: {report_path}")
    except KeyboardInterrupt:
        benchmark.stop()
        print("Benchmark stopped. Run this file again to resume.", flush=True)


if __name__ == "__main__":
    main()
