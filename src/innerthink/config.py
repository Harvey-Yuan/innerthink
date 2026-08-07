from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INNERTHINK_",
        env_file=".env",
        extra="ignore",
    )

    model_id: str = "cds-jb/codi_qwen3-8b-answer_only"
    base_model_id: str = "Qwen/Qwen3-8B"
    device: Literal["mps", "cpu"] = "mps"
    dtype: Literal["float16", "float32"] = "float16"
    cache_dir: Path | None = None
    hf_token: str | None = Field(default=None, repr=False)
    max_prompt_tokens: int = Field(default=1024, ge=32, le=8192)
    max_new_tokens: int = Field(default=128, ge=1, le=512)
    latent_iterations: int = Field(default=6, ge=0, le=32)
    preload_model: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
