#!/usr/bin/env python3
from huggingface_hub import snapshot_download

from innerthink.config import get_settings


def main() -> None:
    settings = get_settings()
    cache_dir = str(settings.cache_dir) if settings.cache_dir else None
    print(f"Downloading CODI checkpoint: {settings.model_id}")
    snapshot_download(
        repo_id=settings.model_id,
        cache_dir=cache_dir,
        token=settings.hf_token,
        allow_patterns=["*.bin", "*.json", "*.jinja"],
    )
    print(f"Downloading base model: {settings.base_model_id}")
    snapshot_download(
        repo_id=settings.base_model_id,
        cache_dir=cache_dir,
        token=settings.hf_token,
    )
    print("Model assets are cached.")


if __name__ == "__main__":
    main()
