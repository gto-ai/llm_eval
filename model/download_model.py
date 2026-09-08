"""Download supported models from Hugging Face."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


MODEL_ROOT = Path("~/workspace/model").expanduser()


class ModelDownloader:
    def __init__(self, repo_id: str, model_name: str) -> None:
        self.repo_id = repo_id
        self.model_name = model_name
        self.model_path = MODEL_ROOT / model_name

    def download(self, revision: str = "main", **kwargs: Any) -> Path:
        kwargs.setdefault("repo_type", "model")
        kwargs.setdefault("max_workers", 8)
        downloaded_path = snapshot_download(
            repo_id=self.repo_id,
            revision=revision,
            local_dir=self.model_path,
            **kwargs,
        )
        downloaded_model_path = Path(downloaded_path)
        return downloaded_model_path


def download_glm5_2_awq_int4(**kwargs: Any) -> Path:
    downloader = ModelDownloader(
        repo_id="cyankiwi/GLM-5.2-AWQ-INT4",
        model_name="GLM-5.2-AWQ-INT4",
    )
    downloaded_model_path = downloader.download(**kwargs)
    return downloaded_model_path


def download_glm5_2_w4a_fp8(**kwargs: Any) -> Path:
    downloader = ModelDownloader(
        repo_id="PhalaCloud/GLM-5.2-W4AFP8",
        model_name="GLM-5.2-W4AFP8",
    )
    downloaded_model_path = downloader.download(**kwargs)
    return downloaded_model_path


def main() -> None:
    download_glm5_2_awq_int4()
    download_glm5_2_w4a_fp8()


if __name__ == "__main__":
    main()
