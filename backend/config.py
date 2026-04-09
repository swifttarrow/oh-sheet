"""Application settings.

All values can be overridden via environment variables prefixed with
``OHSHEET_`` (e.g. ``OHSHEET_BLOB_ROOT=/var/lib/ohsheet/blob``).
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.contracts import ScorePipelineMode

# Default MT3 checkpoint lives next to the vendored source tree at
# backend/vendor/mr_mt3/pretrained/mt3.pth (tracked via git-lfs).
_VENDORED_MT3_CHECKPOINT = (
    Path(__file__).resolve().parent
    / "vendor" / "mr_mt3" / "pretrained" / "mt3.pth"
)


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

    # ---- MT3 baseline transcription ----------------------------------------
    # Path to the pretrained MT3 checkpoint. Defaults to the vendored copy
    # at backend/vendor/mr_mt3/pretrained/mt3.pth (git-lfs). Override via
    # OHSHEET_MT3_CHECKPOINT_PATH to point at a fine-tuned checkpoint.
    mt3_checkpoint_path: Path | None = _VENDORED_MT3_CHECKPOINT
    # Inference knobs.
    mt3_batch_size: int = 4

    # Score path after transcription (or MIDI-derived TranscriptionResult).
    # Env: ``OHSHEET_SCORE_PIPELINE`` — ``arrange`` (default) or ``condense_transform``.
    score_pipeline: ScorePipelineMode = "arrange"


settings = Settings()
