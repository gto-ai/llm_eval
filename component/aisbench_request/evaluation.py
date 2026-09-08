import json
from pathlib import Path


class EvaluationRequest:
    def __init__(self) -> None:
        self.data_points_per_concurrency = 4
        self.valid_concurrencies = (*range(4, 65, 4), 170)

    def get_data_num(self, concurrency: int | float) -> int:
        normalized_concurrency = int(concurrency)
        if normalized_concurrency != concurrency:
            raise ValueError("concurrency must be an integer")
        if normalized_concurrency not in self.valid_concurrencies:
            raise ValueError("concurrency must be 4, 8, ..., 64, or 170")

        data_num = normalized_concurrency * self.data_points_per_concurrency
        return data_num

    def validate_dataset(self, dataset_path: Path, concurrency: int) -> int:
        data_num = self.get_data_num(concurrency)
        manifest_path = dataset_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        available_data_num = int(manifest["data_num"])

        if data_num > available_data_num:
            raise ValueError(
                f"dataset has {available_data_num} rows, but {data_num} are required"
            )

        validated_data_num = data_num
        return validated_data_num
