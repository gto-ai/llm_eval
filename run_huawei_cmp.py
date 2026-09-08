from __future__ import annotations

import os
from pathlib import Path
import signal

from run_max_running_requests import MaxRunningRequestsBenchmark, TuningReport


class HuaweiComparisonBenchmark(MaxRunningRequestsBenchmark):
    def __init__(self) -> None:
        super().__init__()
        self.model = "GLM-5.2-W4AFP8"
        self.max_requests = (64,)
        self.deployment_config = "huawei_sglang_baseline_tp8"

        self.report_root = self.repo_root / "report/huawei_cmp"
        self.summary_path = self.report_root / "summary.csv"
        self.output_root = self.repo_root / "output/huawei_cmp"
        self.log_root = self.repo_root / "log/huawei_cmp"
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
    benchmark = HuaweiComparisonBenchmark()
    signal.signal(signal.SIGTERM, _stop_on_sigterm)
    try:
        report_path = benchmark.run()
        print(f"Report generated: {report_path}")
    except KeyboardInterrupt:
        benchmark.stop()
        print("Benchmark stopped. Run this file again to resume.", flush=True)


if __name__ == "__main__":
    main()
