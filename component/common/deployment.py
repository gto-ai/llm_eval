from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import threading
from urllib.error import URLError
from urllib.request import urlopen


@dataclass(frozen=True)
class DeploymentTarget:
    model: str
    deployment_case: str
    script_path: Path
    log_path: Path


class DeploymentManager:
    def __init__(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.port = 8000
        self.models = (
            "GLM-5.2-AWQ-INT4",
            "GLM-5.2-W4AFP8",
        )
        self.deployment_cases = (
            "dataset_128k_p90",
            "dataset_16k_p0",
            "dataset_3_5k_p0",
            "dataset_128k_p90_c170",
            "dataset_16k_p0_c170",
            "dataset_3_5k_p0_c170",
        )
        self.model_directories = {
            "GLM-5.2-AWQ-INT4": ("vllm_deploy", "glm5_2_awq_int4"),
            "GLM-5.2-W4AFP8": ("sglang_deploy", "glm5_2_w4a_fp8"),
        }
        self.process: subprocess.Popen[bytes] | None = None
        self.active_target: DeploymentTarget | None = None
        self.is_ready = False
        self.lock = threading.Lock()

    def get_models(self) -> tuple[str, ...]:
        models = self.models
        return models

    def get_deployment_cases(self) -> tuple[str, ...]:
        deployment_cases = self.deployment_cases
        return deployment_cases

    def start(self, model: str, deployment_case: str) -> str:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                status = "A deployment is already running. Stop it first."
                return status

            target = self._get_target(model, deployment_case)
            target.log_path.parent.mkdir(parents=True, exist_ok=True)
            target.log_path.write_text("", encoding="utf-8")

            environment = os.environ.copy()
            environment["PORT"] = str(self.port)
            process = subprocess.Popen(
                [str(target.script_path)],
                cwd=self.repo_root,
                env=environment,
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.process = process
            self.active_target = target
            self.is_ready = False
            status = f"Starting {model} with {deployment_case}."
            return status

    def stop(self) -> str:
        with self.lock:
            process = self.process
            if process is None or process.poll() is not None:
                self.process = None
                self.is_ready = False
                status = "No deployment is running."
                return status

            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                pass

            self.process = None
            self.is_ready = False
            status = "Deployment stopped."
            return status

    def get_status(self) -> str:
        with self.lock:
            process = self.process
            target = self.active_target
            is_ready = self.is_ready

        if process is None or target is None:
            status = "Stopped"
            return status

        exit_code = process.poll()
        if exit_code is not None:
            with self.lock:
                self.is_ready = False
            status = f"Stopped (exit code {exit_code})"
            return status

        if is_ready or self._server_is_ready():
            with self.lock:
                self.is_ready = True
            status = f"Running: {target.model} / {target.deployment_case}"
            return status

        status = f"Starting: {target.model} / {target.deployment_case}"
        return status

    def read_log(self, max_lines: int = 200) -> str:
        with self.lock:
            target = self.active_target

        if target is None or not target.log_path.exists():
            log_text = ""
            return log_text

        with target.log_path.open("r", encoding="utf-8", errors="replace") as log_file:
            log_lines = deque(log_file, maxlen=max_lines)

        log_text = "".join(log_lines)
        return log_text

    def monitor(self) -> tuple[str, str]:
        status = self.get_status()
        log_text = self.read_log()
        result = (status, log_text)
        return result

    def is_server_ready(self) -> bool:
        ready = self._server_is_ready()
        return ready

    def _get_target(self, model: str, deployment_case: str) -> DeploymentTarget:
        if model not in self.model_directories:
            raise ValueError(f"Unsupported model: {model}")
        if deployment_case not in self.deployment_cases:
            raise ValueError(f"Unsupported deployment case: {deployment_case}")

        deploy_directory, model_directory = self.model_directories[model]
        script_path = (
            self.repo_root
            / "component"
            / deploy_directory
            / model_directory
            / f"deploy_{deployment_case}.sh"
        )
        log_path = self.repo_root / "log" / f"deploy_{model_directory}_{deployment_case}.log"
        target = DeploymentTarget(model, deployment_case, script_path, log_path)
        return target

    def _server_is_ready(self) -> bool:
        health_url = f"http://127.0.0.1:{self.port}/v1/models"
        try:
            with urlopen(health_url, timeout=0.5) as response:
                is_ready = response.status == 200
        except (OSError, URLError):
            is_ready = False
        return is_ready
