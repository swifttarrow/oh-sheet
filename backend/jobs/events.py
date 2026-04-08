"""JobEvent — single message broadcast over the WebSocket and stored on the record."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

JobStatus = Literal["pending", "running", "succeeded", "failed", "cancelled"]

EventType = Literal[
    "job_created",
    "job_started",
    "stage_started",
    "stage_progress",
    "stage_completed",
    "stage_failed",
    "job_succeeded",
    "job_failed",
]


def _utcnow_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


class JobEvent(BaseModel):
    job_id: str
    type: EventType
    stage: str | None = None
    message: str | None = None
    progress: float | None = None         # 0.0 - 1.0
    data: dict[str, Any] | None = None
    timestamp: str = Field(default_factory=_utcnow_iso)
