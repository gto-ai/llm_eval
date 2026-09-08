from __future__ import annotations

import os
from pathlib import Path
import signal

from run_max_running_requests import MaxRunningRequestsBenchmark, TuningReport


class HuaweiComparison3_5KBenchmark(MaxRunningRequestsBenchmark):
    def __init__(self, request_rate: int) -> None:
        super().__init__()
        self.model = "GLM-5.2-W4AFP8"
        self.datasets = ("dataset_3_5k_p0",)
        self.max_requests = (16, 32, 64)
        self.concurrencies = (*range(4, 65, 4), 170)
        self.request_rate = request_rate
        self.deployment_config = "huawei_sglang_baseline_tp8"

        experiment = f"request_rate_{request_rate}"
        campaign_root = "huawei_cmp_3_5k"
        self.report_root = self.repo_root / "report" / campaign_root / experiment
        self.summary_path = self.report_root / "summary.csv"
        self.output_root = self.repo_root / "output" / campaign_root / experiment
        self.log_root = self.repo_root / "log" / campaign_root / experiment
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
            report_path = super().run()
        finally:
            if previous is None:
                os.environ.pop("DEPLOY_PROFILE", None)
            else:
                os.environ["DEPLOY_PROFILE"] = previous
        return report_path


def _stop_on_sigterm(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def main() -> None:
    benchmark: HuaweiComparison3_5KBenchmark | None = None
    signal.signal(signal.SIGTERM, _stop_on_sigterm)
    try:
        for request_rate in (0, 8):
            benchmark = HuaweiComparison3_5KBenchmark(request_rate)
            print(f"Starting request_rate={request_rate} benchmark.", flush=True)
            report_path = benchmark.run()
            print(
                f"request_rate={request_rate} report generated: {report_path}",
                flush=True,
            )
    except KeyboardInterrupt:
        if benchmark is not None:
            benchmark.stop()
        print("Benchmark stopped. Run this file again to resume.", flush=True)


if __name__ == "__main__":
    main()
