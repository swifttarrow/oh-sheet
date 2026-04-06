"""Application settings.

All values can be overridden via environment variables prefixed with
``OHSHEET_`` (e.g. ``OHSHEET_BLOB_ROOT=/var/lib/ohsheet/blob``).
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OHSHEET_",
        env_file=".env",
        extra="ignore",
    )

    # Where the LocalBlobStore writes its files. Returned URIs are file:// based.
    blob_root: Path = Path("./blob")

    # CORS — wide open for dev; tighten in deployment.
    cors_origins: list[str] = ["*"]

    # Worker timeout used by OrchestratorCommand envelopes.
    job_timeout_sec: int = 600


settings = Settings()
