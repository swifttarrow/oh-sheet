"""Job submission and inspection."""
from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ohsheet.api.deps import get_job_manager
from ohsheet.contracts import (
    SCHEMA_VERSION,
    Difficulty,
    EngravedOutput,
    InputBundle,
    InputMetadata,
    PipelineConfig,
    PipelineVariant,
    RemoteAudioFile,
    RemoteMidiFile,
)
from ohsheet.jobs.events import JobEvent
from ohsheet.jobs.manager import JobManager, JobRecord

router = APIRouter()


class JobCreateRequest(BaseModel):
    """Submit a pipeline job.

    Source signal (pick one):
      * ``audio``  — RemoteAudioFile from /v1/uploads/audio (variant: audio_upload)
      * ``midi``   — RemoteMidiFile  from /v1/uploads/midi  (variant: midi_upload)
      * neither, but ``title`` set    — title-lookup        (variant: full)

    ``title`` and ``artist`` are metadata — they can be supplied alongside any
    source.
    """

    audio: Optional[RemoteAudioFile] = None
    midi: Optional[RemoteMidiFile] = None
    title: Optional[str] = None
    artist: Optional[str] = None

    skip_humanizer: bool = False
    difficulty: Difficulty = "intermediate"


class JobSummary(BaseModel):
    job_id: str
    status: str
    variant: str
    title: Optional[str] = None
    artist: Optional[str] = None
    error: Optional[str] = None
    result: Optional[EngravedOutput] = None


def _record_to_summary(record: JobRecord) -> JobSummary:
    return JobSummary(
        job_id=record.job_id,
        status=record.status,
        variant=record.config.variant,
        title=record.bundle.metadata.title,
        artist=record.bundle.metadata.artist,
        error=record.error,
        result=record.result,
    )


@router.post("/jobs", response_model=JobSummary, status_code=202)
async def create_job(
    body: JobCreateRequest,
    manager: Annotated[JobManager, Depends(get_job_manager)],
) -> JobSummary:
    # Source signal: audio xor midi; if neither, fall back to title-lookup.
    if body.audio is not None and body.midi is not None:
        raise HTTPException(
            status_code=400,
            detail="Provide audio OR midi, not both.",
        )
    if body.audio is None and body.midi is None and not body.title:
        raise HTTPException(
            status_code=400,
            detail="Provide one of: audio, midi, or title (for title-lookup).",
        )

    if body.audio is not None:
        bundle = InputBundle(
            schema_version=SCHEMA_VERSION,
            audio=body.audio,
            midi=None,
            metadata=InputMetadata(title=body.title, artist=body.artist, source="audio_upload"),
        )
        variant: PipelineVariant = "audio_upload"
    elif body.midi is not None:
        bundle = InputBundle(
            schema_version=SCHEMA_VERSION,
            audio=None,
            midi=body.midi,
            metadata=InputMetadata(title=body.title, artist=body.artist, source="midi_upload"),
        )
        variant = "midi_upload"
    else:
        assert body.title is not None
        bundle = InputBundle(
            schema_version=SCHEMA_VERSION,
            audio=None,
            midi=None,
            metadata=InputMetadata(title=body.title, artist=body.artist, source="title_lookup"),
        )
        variant = "full"

    config = PipelineConfig(variant=variant, skip_humanizer=body.skip_humanizer)
    record = await manager.submit(bundle, config)
    return _record_to_summary(record)


@router.get("/jobs", response_model=list[JobSummary])
async def list_jobs(
    manager: Annotated[JobManager, Depends(get_job_manager)],
) -> list[JobSummary]:
    return [_record_to_summary(r) for r in manager.list()]


@router.get("/jobs/{job_id}", response_model=JobSummary)
async def get_job(
    job_id: str,
    manager: Annotated[JobManager, Depends(get_job_manager)],
) -> JobSummary:
    record = manager.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return _record_to_summary(record)


@router.get("/jobs/{job_id}/events", response_model=list[JobEvent])
async def get_job_events(
    job_id: str,
    manager: Annotated[JobManager, Depends(get_job_manager)],
) -> list[JobEvent]:
    record = manager.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return list(record.events)
