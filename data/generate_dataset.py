"""Generate tokenizer-exact GSM8K performance datasets."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path("~/workspace/model").expanduser()
SOURCE_PATH = PROJECT_ROOT / "data/source/gsm8k/test.jsonl"
OUTPUT_ROOT = PROJECT_ROOT / "data/generated"
MAX_EVAL_CONCURRENCY = 170
DATA_POINTS_PER_CONCURRENCY = 4
MAX_DATA_NUM = MAX_EVAL_CONCURRENCY * DATA_POINTS_PER_CONCURRENCY


class DatasetGenerator:
    def __init__(
        self,
        tokenizer_path: Path,
        dataset_name: str,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("trust_remote_code", True)
        kwargs.setdefault("local_files_only", True)

        self.tokenizer_path = tokenizer_path
        self.dataset_name = dataset_name
        self.source_path = SOURCE_PATH
        self.output_root = OUTPUT_ROOT
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **kwargs)
        self.source_rows = self._read_source()
        self.safe_tokens: list[int] | None = None

    def generate(
        self,
        case_name: str,
        input_tokens: int,
        output_tokens: int,
        data_num: int = MAX_DATA_NUM,
        prefix_ratio: float = 0.0,
        seed: int = 1,
        **kwargs: Any,
    ) -> Path:
        overwrite = kwargs.get("overwrite", False)
        dataset_path = self.output_root / self.dataset_name / case_name

        if dataset_path.exists():
            if not overwrite:
                raise FileExistsError(f"dataset already exists: {dataset_path}")
            shutil.rmtree(dataset_path)

        if prefix_ratio:
            rows, prefix_rows, common_prefix_tokens = self._prefix_rows(
                input_tokens,
                data_num,
                prefix_ratio,
                seed,
            )
        else:
            rows = self._normal_rows(input_tokens, data_num, seed)
            prefix_rows = []
            common_prefix_tokens = 0

        dataset_path.mkdir(parents=True)
        test_path = dataset_path / "test.jsonl"
        self._write_jsonl(test_path, rows)

        files = {
            "test.jsonl": {
                "rows": len(rows),
                "sha256": self._sha256(test_path),
            }
        }
        if prefix_rows:
            prefix_path = dataset_path / "prefix.jsonl"
            self._write_jsonl(prefix_path, prefix_rows)
            files["prefix.jsonl"] = {
                "rows": len(prefix_rows),
                "sha256": self._sha256(prefix_path),
            }

        manifest = {
            "case_name": case_name,
            "tokenizer_path": str(self.tokenizer_path),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "data_num": data_num,
            "prefix_ratio": prefix_ratio,
            "common_prefix_tokens": common_prefix_tokens,
            "seed": seed,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "files": files,
        }
        manifest_path = dataset_path / "manifest.json"
        manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False)
        manifest_path.write_text(f"{manifest_text}\n", encoding="utf-8")

        generated_dataset_path = dataset_path
        return generated_dataset_path

    def _normal_rows(
        self,
        input_tokens: int,
        data_num: int,
        seed: int,
    ) -> list[dict[str, str]]:
        random_generator = random.Random(seed)
        rows = []
        for _ in range(data_num):
            source_row = random_generator.choice(self.source_rows)
            source_tokens = self._encode(source_row["question"])
            prompt, _ = self._exact_text(source_tokens, input_tokens)
            rows.append({"question": prompt, "answer": "none"})

        generated_rows = rows
        return generated_rows

    def _prefix_rows(
        self,
        input_tokens: int,
        data_num: int,
        prefix_ratio: float,
        seed: int,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]], int]:
        random_generator = random.Random(seed)
        prefix_length = int(input_tokens * prefix_ratio)
        suffix_length = input_tokens - prefix_length - 1
        if prefix_length < 1 or suffix_length < 1:
            raise ValueError("invalid prefix ratio")

        prefix_source = random_generator.choice(self.source_rows)
        prefix_source_tokens = self._encode(prefix_source["question"])
        prefix_text, prefix_tokens = self._exact_text(
            prefix_source_tokens,
            prefix_length,
        )

        safe_tokens = list(self._safe_boundary_tokens())
        if len(safe_tokens) < data_num:
            raise RuntimeError(
                f"need {data_num} tokenizer-safe boundary tokens, "
                f"found {len(safe_tokens)}"
            )
        random_generator.shuffle(safe_tokens)
        rows = []
        full_token_rows = []
        used_boundary_tokens = set()

        for row_index in range(data_num):
            source_row = random_generator.choice(self.source_rows)
            suffix_source_tokens = self._encode(source_row["question"])
            _, suffix_tokens = self._exact_text(
                suffix_source_tokens,
                suffix_length,
            )

            prompt = None
            prompt_tokens = None
            for boundary_token in safe_tokens:
                if boundary_token in used_boundary_tokens:
                    continue

                candidate_tokens = prefix_tokens + [boundary_token] + suffix_tokens
                candidate_text = self._decode(candidate_tokens)
                encoded_candidate = self._encode(candidate_text)
                if encoded_candidate != candidate_tokens:
                    continue

                prompt = candidate_text
                prompt_tokens = encoded_candidate
                used_boundary_tokens.add(boundary_token)
                break

            if prompt is None or prompt_tokens is None:
                raise RuntimeError(f"could not generate prefix row {row_index}")

            rows.append({"question": prompt, "answer": "none"})
            full_token_rows.append(prompt_tokens)

        common_prefix_tokens = self._common_prefix_length(full_token_rows)
        if common_prefix_tokens != prefix_length:
            raise ValueError(
                f"common prefix is {common_prefix_tokens}, expected {prefix_length}"
            )

        prefix_rows = [{"question": prefix_text, "answer": "none"}]
        generated_result = (rows, prefix_rows, common_prefix_tokens)
        return generated_result

    def _exact_text(
        self,
        source_tokens: list[int],
        target_length: int,
    ) -> tuple[str, list[int]]:
        candidate_tokens = self._repeat_tokens(source_tokens, target_length)

        for _ in range(12):
            candidate_text = self._decode(candidate_tokens)
            encoded_candidate = self._encode(candidate_text)
            if len(encoded_candidate) == target_length:
                exact_result = (candidate_text, encoded_candidate)
                return exact_result

            if len(encoded_candidate) > target_length:
                candidate_tokens = encoded_candidate[:target_length]
            else:
                missing_length = target_length - len(encoded_candidate)
                filler_tokens = self._repeat_tokens(source_tokens, missing_length)
                candidate_tokens = encoded_candidate + filler_tokens

        raise RuntimeError(f"could not generate exactly {target_length} tokens")

    def _safe_boundary_tokens(self) -> list[int]:
        if self.safe_tokens is not None:
            cached_safe_tokens = self.safe_tokens
            return cached_safe_tokens

        special_tokens = set(self.tokenizer.all_special_ids)
        candidate_tokens = set()
        for row in self.source_rows:
            candidate_tokens.update(self._encode(row["question"]))

        safe_tokens = []
        for token_id in sorted(candidate_tokens):
            if token_id in special_tokens:
                continue
            token_text = self._decode([token_id])
            if token_text and self._encode(token_text) == [token_id]:
                safe_tokens.append(token_id)

        self.safe_tokens = safe_tokens
        generated_safe_tokens = safe_tokens
        return generated_safe_tokens

    def _read_source(self) -> list[dict[str, str]]:
        rows = []
        with self.source_path.open(encoding="utf-8") as source_file:
            for line in source_file:
                row = json.loads(line)
                if not isinstance(row.get("question"), str):
                    raise ValueError("GSM8K row is missing a question")
                rows.append(row)

        if not rows:
            raise ValueError("GSM8K source dataset is empty")

        source_rows = rows
        return source_rows

    def _encode(self, text: str) -> list[int]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return token_ids

    def _decode(self, token_ids: list[int]) -> str:
        text = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return text

    @staticmethod
    def _repeat_tokens(token_ids: list[int], target_length: int) -> list[int]:
        if not token_ids:
            raise ValueError("cannot repeat an empty token sequence")

        repeat_count = (target_length + len(token_ids) - 1) // len(token_ids)
        repeated_tokens = (token_ids * repeat_count)[:target_length]
        return repeated_tokens

    @staticmethod
    def _common_prefix_length(token_rows: list[list[int]]) -> int:
        first_row = token_rows[0]
        common_length = len(first_row)

        for token_row in token_rows[1:]:
            row_common_length = 0
            for first_token, current_token in zip(first_row, token_row):
                if first_token != current_token:
                    break
                row_common_length += 1
            common_length = min(common_length, row_common_length)

        measured_common_length = common_length
        return measured_common_length

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8") as output_file:
            for row in rows:
                row_text = json.dumps(row, ensure_ascii=False)
                output_file.write(f"{row_text}\n")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)

        file_hash = digest.hexdigest()
        return file_hash


def gen_data_glm5_2_awq_int4(*, overwrite: bool = False) -> list[Path]:
    generator = DatasetGenerator(
        tokenizer_path=MODEL_ROOT / "GLM-5.2-AWQ-INT4",
        dataset_name="glm5_2_awq_int4",
    )
    dataset_128k_p90 = generator.generate(
        "dataset_128k_p90",
        131_072,
        1_024,
        data_num=MAX_DATA_NUM,
        prefix_ratio=0.9,
        overwrite=overwrite,
    )
    dataset_16k_p0 = generator.generate(
        "dataset_16k_p0",
        16_384,
        1_024,
        data_num=MAX_DATA_NUM,
        prefix_ratio=0.0,
        overwrite=overwrite,
    )
    dataset_3_5k_p0 = generator.generate(
        "dataset_3_5k_p0",
        3_584,
        1_536,
        data_num=MAX_DATA_NUM,
        prefix_ratio=0.0,
        overwrite=overwrite,
    )

    generated_datasets = [dataset_128k_p90, dataset_16k_p0, dataset_3_5k_p0]
    return generated_datasets


def gen_data_glm5_2_w4a_fp8(*, overwrite: bool = False) -> list[Path]:
    generator = DatasetGenerator(
        tokenizer_path=MODEL_ROOT / "GLM-5.2-W4AFP8",
        dataset_name="glm5_2_w4a_fp8",
    )
    dataset_128k_p90 = generator.generate(
        "dataset_128k_p90",
        131_072,
        1_024,
        data_num=MAX_DATA_NUM,
        prefix_ratio=0.9,
        overwrite=overwrite,
    )
    dataset_16k_p0 = generator.generate(
        "dataset_16k_p0",
        16_384,
        1_024,
        data_num=MAX_DATA_NUM,
        prefix_ratio=0.0,
        overwrite=overwrite,
    )
    dataset_3_5k_p0 = generator.generate(
        "dataset_3_5k_p0",
        3_584,
        1_536,
        data_num=MAX_DATA_NUM,
        prefix_ratio=0.0,
        overwrite=overwrite,
    )

    generated_datasets = [dataset_128k_p90, dataset_16k_p0, dataset_3_5k_p0]
    return generated_datasets


def main() -> None:
    gen_data_glm5_2_awq_int4()
    gen_data_glm5_2_w4a_fp8()


if __name__ == "__main__":
    main()
