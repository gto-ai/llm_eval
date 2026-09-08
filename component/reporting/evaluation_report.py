import csv
from pathlib import Path


class EvaluationReport:
    def __init__(self, summary_path: Path, report_path: Path) -> None:
        self.summary_path = summary_path
        self.report_path = report_path
        self.concurrencies = (*range(4, 65, 4), 170)
        self.columns = (
            ("AWQ INT4 · 128K P90", "GLM-5.2-AWQ-INT4", "dataset_128k_p90"),
            ("AWQ INT4 · 16K P0", "GLM-5.2-AWQ-INT4", "dataset_16k_p0"),
            ("AWQ INT4 · 3.5K P0", "GLM-5.2-AWQ-INT4", "dataset_3_5k_p0"),
            ("W4AFP8 · 128K P90", "GLM-5.2-W4AFP8", "dataset_128k_p90"),
            ("W4AFP8 · 16K P0", "GLM-5.2-W4AFP8", "dataset_16k_p0"),
            ("W4AFP8 · 3.5K P0", "GLM-5.2-W4AFP8", "dataset_3_5k_p0"),
        )

    def update(self) -> Path:
        latest_results = self._read_latest_results()
        lines = [
            "| Concurrency | "
            + " | ".join(label for label, _, _ in self.columns)
            + " |",
            "| ---: | " + " | ".join("---" for _ in self.columns) + " |",
        ]

        for concurrency in self.concurrencies:
            cells = []
            for _, model, dataset in self.columns:
                result = latest_results.get((model, dataset, concurrency))
                cell = self._format_result(result)
                cells.append(cell)
            lines.append(f"| {concurrency} | " + " | ".join(cells) + " |")

        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        updated_report_path = self.report_path
        return updated_report_path

    def _read_latest_results(self) -> dict[tuple[str, str, int], dict[str, str]]:
        latest_results: dict[tuple[str, str, int], dict[str, str]] = {}
        if not self.summary_path.exists():
            return latest_results

        with self.summary_path.open(encoding="utf-8", newline="") as summary_file:
            for result in csv.DictReader(summary_file):
                key = (
                    result["model"],
                    result["dataset"],
                    int(result["concurrency"]),
                )
                latest_results[key] = result
        return latest_results

    def _format_result(self, result: dict[str, str] | None) -> str:
        if result is None:
            return "—"

        ttft_ms = float(result["ttft_ms"])
        tpot_ms = float(result["tpot_ms"])
        throughput = float(result["throughput_per_gpu_token_s"])
        value = (
            f"{ttft_ms:.1f} ms / {tpot_ms:.1f} ms / "
            f"{throughput:.2f} tok/s/GPU"
        )
        return value


def demo_gen_report() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    summary_path = repo_root / "report/summary.csv"
    report_path = repo_root / "report/report.md"
    report = EvaluationReport(summary_path, report_path)
    generated_report_path = report.update()
    print(f"Report generated: {generated_report_path}")
    return generated_report_path


def main() -> None:
    demo_gen_report()


if __name__ == "__main__":
    main()
