"""Configuration from the environment (12-factor, rule III): INDEX_PATH, LOG_LEVEL, PORT."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    index_path: Path = Path("index.bin")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    port: int = 8000
    max_k: int = 100


@lru_cache
def get_settings() -> Settings:
    return Settings()
