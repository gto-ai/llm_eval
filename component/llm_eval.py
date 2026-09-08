from typing import Any

from component.aisbench_request import AisBenchEvaluator
from component.auto_run import AutoRunManager
from component.common.deployment import DeploymentManager


class LlmEval:
    def __init__(self) -> None:
        self.deployment = DeploymentManager()
        self.evaluation = AisBenchEvaluator()
        self.auto_run = AutoRunManager(self.deployment, self.evaluation)

    def get_models(self) -> tuple[str, ...]:
        models = self.deployment.get_models()
        return models

    def get_deployment_cases(self) -> tuple[str, ...]:
        deployment_cases = self.deployment.get_deployment_cases()
        return deployment_cases

    def start_deployment(self, model: str, deployment_case: str) -> str:
        if self.auto_run.is_running():
            status = "Cannot deploy manually while auto run is running."
            return status
        status = self.deployment.start(model, deployment_case)
        return status

    def stop_deployment(self) -> str:
        status = self.deployment.stop()
        return status

    def monitor_deployment(self) -> tuple[str, str]:
        deployment_state = self.deployment.monitor()
        return deployment_state

    def get_evaluation_data_num(self, concurrency: int | float) -> int:
        data_num = self.evaluation.get_data_num(concurrency)
        return data_num

    def get_evaluation_models(self) -> tuple[str, ...]:
        models = self.evaluation.get_models()
        return models

    def get_evaluation_cases(self) -> tuple[str, ...]:
        evaluation_cases = self.evaluation.get_cases()
        return evaluation_cases

    def start_evaluation(
        self,
        model: str,
        evaluation_case: str,
        concurrency: int | float,
    ) -> str:
        if self.auto_run.is_running():
            status = "Cannot evaluate manually while auto run is running."
            return status
        status = self.evaluation.start(model, evaluation_case, concurrency)
        return status

    def start_all_evaluations(self, model: str, evaluation_case: str) -> str:
        if self.auto_run.is_running():
            status = "Cannot evaluate manually while auto run is running."
            return status
        status = self.evaluation.start_all(model, evaluation_case)
        return status

    def stop_evaluation(self) -> str:
        status = self.evaluation.stop()
        return status

    def monitor_evaluation(self) -> tuple[str, str, list[list[Any]]]:
        evaluation_state = self.evaluation.monitor()
        return evaluation_state

    def generate_report(self) -> str:
        report_path = self.evaluation.report.update()
        message = f"Report generated: {report_path}"
        return message

    def start_auto_run(self) -> str:
        status = self.auto_run.start()
        return status

    def stop_all(self) -> str:
        auto_status = self.auto_run.stop()
        return auto_status

    def monitor_auto_run(self) -> tuple[str, str]:
        state = self.auto_run.monitor()
        return state
