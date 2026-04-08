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

    # ---- Basic Pitch transcription -----------------------------------------
    # Tunable knobs passed through to basic_pitch.inference.predict(). Defaults
    # mirror upstream (basic_pitch.constants.DEFAULT_*). The ONNX model ships
    # inside the basic-pitch wheel, so there's no checkpoint path to configure.
    basic_pitch_onset_threshold: float = 0.5
    basic_pitch_frame_threshold: float = 0.3
    basic_pitch_minimum_note_length_ms: float = 127.7


settings = Settings()
