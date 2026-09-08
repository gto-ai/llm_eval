from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvaluationCase:
    model: str
    model_path: Path
    dataset_name: str
    dataset_path: Path
    served_model_name: str
    output_tokens: int
    prefix_ratio: float
    ttft_sla_ms: float
    tpot_sla_ms: float = 20.0
