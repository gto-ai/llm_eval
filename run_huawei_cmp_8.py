from __future__ import annotations

import os
from pathlib import Path
import signal

from component.common.deployment import DeploymentManager
from run_max_running_requests import MaxRunningRequestsBenchmark, TuningReport


class HuaweiComparisonRate8Benchmark(MaxRunningRequestsBenchmark):
    def __init__(self) -> None:
        super().__init__()
        self.model = "GLM-5.2-W4AFP8"
        self.dataset_max_requests = (
            ("dataset_128k_p90", 16),
            ("dataset_128k_p90", 32),
            ("dataset_128k_p90", 64),
            ("dataset_16k_p0", 16),
            ("dataset_16k_p0", 32),
            ("dataset_16k_p0", 64),
            ("dataset_3_5k_p0", 16),
            ("dataset_3_5k_p0", 32),
            ("dataset_3_5k_p0", 64),
        )
        self.datasets = tuple(
            dict.fromkeys(
                dataset for dataset, _max_requests in self.dataset_max_requests
            )
        )
        self.max_requests = (16, 32, 64)
        self.concurrencies = (170, *range(4, 65, 4))
        self.request_rate = 8
        self.deployment_config = "huawei_sglang_baseline_tp8"

        campaign_root = "huawei_cmp_8"
        self.report_root = self.repo_root / "report" / campaign_root
        self.summary_path = self.report_root / "summary.csv"
        self.output_root = self.repo_root / "output" / campaign_root
        self.log_root = self.repo_root / "log" / campaign_root
        self.run_log_path = self.log_root / "benchmark.log"
        self.report = TuningReport(
            self.summary_path,
            self.report_root / "report.md",
            self.datasets,
            self.max_requests,
            self.concurrencies,
        )

    def run(self) -> Path:
        previous = os.environ.get("DEPLOY_PROFILE")
        os.environ["DEPLOY_PROFILE"] = "huawei_cmp"
        try:
            report_path = self._run_campaign()
        finally:
            if previous is None:
                os.environ.pop("DEPLOY_PROFILE", None)
            else:
                os.environ["DEPLOY_PROFILE"] = previous
        return report_path

    def _run_campaign(self) -> Path:
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.run_log_path.write_text("", encoding="utf-8")
        completed = self._read_completed()
        total = len(self.dataset_max_requests) * len(self.concurrencies)
        completed_count = sum(
            (dataset, max_requests, concurrency) in completed
            for dataset, max_requests in self.dataset_max_requests
            for concurrency in self.concurrencies
        )
        self._log(f"Completed {completed_count}/{total}; starting benchmark.")

        if DeploymentManager().is_server_ready():
            raise RuntimeError(
                "Port 8000 is occupied. Stop the existing model service first."
            )

        try:
            concurrency_phases = ((170,), tuple(range(4, 65, 4)))
            for phase in concurrency_phases:
                self._log(f"Starting concurrency phase: {phase}")
                for dataset, max_requests in self.dataset_max_requests:
                    pending = tuple(
                        concurrency
                        for concurrency in phase
                        if (dataset, max_requests, concurrency) not in completed
                    )
                    if not pending:
                        self._log(
                            f"Skip completed: {dataset} / "
                            f"MR={max_requests} / phase={phase}"
                        )
                        continue
                    self._run_stage(
                        dataset,
                        max_requests,
                        pending,
                        completed,
                    )
        finally:
            self.stop()

        report_path = self.report.update()
        self._log(f"Benchmark completed. Report: {report_path}")
        return report_path


def _stop_on_sigterm(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def main() -> None:
    benchmark = HuaweiComparisonRate8Benchmark()
    signal.signal(signal.SIGTERM, _stop_on_sigterm)
    try:
        report_path = benchmark.run()
        print(f"Report generated: {report_path}")
    except KeyboardInterrupt:
        benchmark.stop()
        print("Benchmark stopped. Run this file again to resume.", flush=True)


if __name__ == "__main__":
    main()
