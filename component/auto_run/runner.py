from collections import deque
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time

from component.aisbench_request import AisBenchEvaluator
from component.common.deployment import DeploymentManager


class AutoRunStopped(Exception):
    pass


class AutoRunManager:
    def __init__(
        self,
        deployment: DeploymentManager,
        evaluation: AisBenchEvaluator,
        deployment_timeout_s: float = 3600,
        poll_interval_s: float = 2,
    ) -> None:
        self.deployment = deployment
        self.evaluation = evaluation
        self.deployment_timeout_s = deployment_timeout_s
        self.poll_interval_s = poll_interval_s
        self.concurrencies = tuple(range(4, 65, 4))
        self.special_concurrency = 170
        self.summary_path = evaluation.summary_path
        self.log_path = evaluation.repo_root / "log/auto_run.log"
        self.legacy_deployment_configs = {
            "GLM-5.2-AWQ-INT4": "vllm_tp",
            "GLM-5.2-W4AFP8": "sglang_tp",
        }
        self.thread: threading.Thread | None = None
        self.status = "Stopped"
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    def start(self) -> str:
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                status = "Auto run is already running."
                return status
            if self.evaluation.is_running():
                status = "Cannot start auto run while an evaluation is running."
                return status

            self.stop_event.clear()
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("", encoding="utf-8")
            self.status = "Starting auto run."
            thread = threading.Thread(target=self._run, daemon=True)
            self.thread = thread
            thread.start()
            status = self.status
            return status

    def stop(self) -> str:
        self.stop_event.set()
        self.evaluation.stop()
        self.deployment.stop()
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                self.status = "Stopping auto run."
            else:
                self.status = "Auto run is not running."
            status = self.status
        return status

    def is_running(self) -> bool:
        with self.lock:
            thread = self.thread
        running = thread is not None and thread.is_alive()
        return running

    def monitor(self) -> tuple[str, str]:
        with self.lock:
            status = self.status
        log_text = self._read_log()
        state = (status, log_text)
        return state

    def _run(self) -> None:
        try:
            completed = self._read_completed_experiments()
            pending_count = self._count_pending(completed)
            self._append_log(f"Pending experiments: {pending_count}\n")
            if pending_count == 0:
                self.evaluation.report.update()
                self._set_status("Auto run completed. Nothing is missing.")
                self._append_log("Nothing is missing.\n")
                return

            self.deployment.stop()
            if self.deployment.is_server_ready():
                raise RuntimeError("port 8000 is occupied by an unmanaged model server")

            for model in self.evaluation.get_models():
                for dataset in self.evaluation.get_cases():
                    self._check_stopped()
                    regular = tuple(
                        concurrency
                        for concurrency in self.concurrencies
                        if (model, dataset, concurrency) not in completed
                    )
                    if regular:
                        self._run_stage(model, dataset, dataset, regular, completed)

                    special_key = (model, dataset, self.special_concurrency)
                    if special_key not in completed:
                        deployment_case = f"{dataset}_c{self.special_concurrency}"
                        self._run_stage(
                            model,
                            dataset,
                            deployment_case,
                            (self.special_concurrency,),
                            completed,
                        )

            self.evaluation.report.update()
            self._set_status("Auto run completed.")
            self._append_log("Auto run completed.\n")
        except AutoRunStopped:
            self._set_status("Auto run stopped.")
            self._append_log("Auto run stopped.\n")
        except Exception as error:
            self._set_status(f"Auto run failed: {error}")
            self._append_log(f"ERROR: {error}\n")

    def _run_stage(
        self,
        model: str,
        dataset: str,
        deployment_case: str,
        concurrencies: tuple[int, ...],
        completed: set[tuple[str, str, int]],
    ) -> None:
        self._check_stopped()
        self._set_status(f"Deploying: {model} / {deployment_case}")
        self._append_log(f"\nDeploying {model} / {deployment_case}\n")
        deployment_status = self.deployment.start(model, deployment_case)
        self._append_log(f"{deployment_status}\n")

        try:
            self._wait_for_deployment(model, dataset)
            for concurrency in concurrencies:
                self._check_stopped()
                self._set_status(
                    f"Evaluating: {model} / {dataset} / concurrency {concurrency}"
                )
                self._append_log(
                    f"Evaluating {model} / {dataset} / concurrency {concurrency}\n"
                )
                evaluation_status = self.evaluation.start(
                    model,
                    dataset,
                    concurrency,
                )
                self._append_log(f"{evaluation_status}\n")
                self._wait_for_evaluation(model, dataset, concurrency)
                completed.add((model, dataset, concurrency))
        finally:
            stop_status = self.deployment.stop()
            self._append_log(f"{stop_status}\n")

    def _wait_for_deployment(self, model: str, dataset: str) -> None:
        started_at = time.monotonic()
        while True:
            self._check_stopped()
            deployment_status = self.deployment.get_status()
            if deployment_status.startswith("Stopped (exit code"):
                raise RuntimeError(deployment_status)
            if self.evaluation.is_model_ready(model, dataset):
                self._append_log("Model is ready.\n")
                return
            if time.monotonic() - started_at >= self.deployment_timeout_s:
                raise TimeoutError("model deployment timed out")
            self.stop_event.wait(self.poll_interval_s)

    def _wait_for_evaluation(
        self,
        model: str,
        dataset: str,
        concurrency: int,
    ) -> None:
        while self.evaluation.is_running():
            self._check_stopped()
            self.stop_event.wait(self.poll_interval_s)

        self._check_stopped()
        evaluation_status, _, _ = self.evaluation.monitor()
        if evaluation_status != "Evaluation completed.":
            raise RuntimeError(evaluation_status)

        completed = self._read_completed_experiments()
        experiment = (model, dataset, concurrency)
        if experiment not in completed:
            raise RuntimeError(
                f"incomplete result: {model} / {dataset} / concurrency {concurrency}"
            )
        self._append_log("Evaluation completed.\n")

    def _read_completed_experiments(self) -> set[tuple[str, str, int]]:
        completed: set[tuple[str, str, int]] = set()
        if not self.summary_path.exists():
            return completed

        with self.summary_path.open(encoding="utf-8", newline="") as summary_file:
            for result in csv.DictReader(summary_file):
                try:
                    concurrency = int(result["concurrency"])
                    data_num = int(result["data_num"])
                    success_requests = int(result["success_requests"])
                    failed_requests = int(result["failed_requests"])
                    concurrency_matched = (
                        result["concurrency_matched"].strip().lower() == "true"
                    )
                    deployment_config = self._read_deployment_config(result)
                    expected_config = self.evaluation.get_deployment_config(
                        result["model"]
                    )
                except (KeyError, TypeError, ValueError):
                    continue

                expected_data_num = concurrency * 4
                if (
                    concurrency_matched
                    and data_num == expected_data_num
                    and success_requests == data_num
                    and failed_requests == 0
                    and deployment_config == expected_config
                ):
                    completed.add((result["model"], result["dataset"], concurrency))
        return completed

    def _read_deployment_config(self, result: dict[str, str]) -> str:
        manifest_path = Path(result["run_dir"]) / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            deployment_config = manifest.get("deployment_config")
        except (OSError, json.JSONDecodeError):
            deployment_config = None
        if deployment_config is None:
            deployment_config = self.legacy_deployment_configs[result["model"]]
        return str(deployment_config)

    def _count_pending(self, completed: set[tuple[str, str, int]]) -> int:
        expected_concurrencies = (*self.concurrencies, self.special_concurrency)
        pending_count = sum(
            (model, dataset, concurrency) not in completed
            for model in self.evaluation.get_models()
            for dataset in self.evaluation.get_cases()
            for concurrency in expected_concurrencies
        )
        return pending_count

    def _check_stopped(self) -> None:
        if self.stop_event.is_set():
            raise AutoRunStopped

    def _set_status(self, status: str) -> None:
        with self.lock:
            self.status = status

    def _read_log(self, max_lines: int = 300) -> str:
        if not self.log_path.exists():
            log_text = ""
            return log_text
        with self.log_path.open(encoding="utf-8", errors="replace") as log_file:
            lines = deque(log_file, maxlen=max_lines)
        log_text = "".join(lines)
        return log_text

    def _append_log(self, text: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"[{timestamp}] {text}")
