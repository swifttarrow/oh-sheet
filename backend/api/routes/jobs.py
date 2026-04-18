"""Job submission and inspection."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.api.deps import get_blob_store, get_job_manager
from backend.config import settings
from backend.contracts import (
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
from backend.jobs.events import JobEvent
from backend.jobs.manager import JobManager, JobRecord
from backend.storage.local import LocalBlobStore

router = APIRouter()


class JobCreateRequest(BaseModel):
    """Submit a pipeline job.

    Source signal (pick one):
      * ``audio``  — RemoteAudioFile from /v1/uploads/audio (variant: audio_upload)
      * ``midi``   — RemoteMidiFile  from /v1/uploads/midi  (variant: midi_upload)
      * neither, but ``title`` set    — title-lookup        (variant: full)

    ``title`` and ``artist`` are metadata — they can be supplied alongside any
    source.

    ``prefer_clean_source`` is the user's opt-in to the cover-search fast
    path: when True, the ingest stage will try to find a clean piano
    cover of the song and transcribe that instead of the user's original
    YouTube URL. Only meaningful for title-lookup / YouTube inputs; the
    audio_upload and midi_upload variants ignore it because the user is
    providing the source directly. See ``backend.services.cover_search``
    for the matching policy. Defaults to False so existing clients keep
    working unchanged.
    """

    audio: RemoteAudioFile | None = None
    midi: RemoteMidiFile | None = None
    title: str | None = None
    artist: str | None = None
    prefer_clean_source: bool = False

    skip_humanizer: bool = False
    difficulty: Difficulty = "intermediate"


class JobSummary(BaseModel):
    job_id: str
    status: str
    variant: str
    title: str | None = None
    artist: str | None = None
    source_url: str | None = None
    error: str | None = None
    result: EngravedOutput | None = None


def _record_to_summary(record: JobRecord) -> JobSummary:
    return JobSummary(
        job_id=record.job_id,
        status=record.status,
        variant=record.config.variant,
        title=record.bundle.metadata.title,
        artist=record.bundle.metadata.artist,
        source_url=record.bundle.metadata.source_url,
        error=record.error,
        result=record.result,
    )


@router.post("/jobs", response_model=JobSummary, status_code=202)
async def create_job(
    body: JobCreateRequest,
    manager: Annotated[JobManager, Depends(get_job_manager)],
    blob: Annotated[LocalBlobStore, Depends(get_blob_store)],
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

    # title_lookup jobs resolve through TuneChat upstream — they never
    # reach the ML engraver. When TuneChat is disabled there's no path
    # that can produce a score for these jobs, so reject at creation
    # time rather than burning ~1 minute of ingest/transcribe/arrange/
    # humanize only to hard-fail at engrave.
    # Explicit title check (not just "neither audio nor midi") so a
    # future source type doesn't silently classify as title_lookup.
    is_title_lookup = (
        body.audio is None and body.midi is None and body.title is not None
    )
    if is_title_lookup and not settings.tunechat_enabled:
        raise HTTPException(
            status_code=400,
            detail=(
                "title-lookup jobs require TuneChat, which is currently disabled. "
                "Upload the audio or MIDI file directly."
            ),
        )

    # Integrity: the audio / midi URI must point to a real blob in
    # storage. Without this check a client could forge a Remote*File
    # with an arbitrary URI and the pipeline would "succeed" by
    # running stub stages over nothing — a silent failure mode that
    # masked real upload bugs during development.
    if body.audio is not None and not blob.exists(body.audio.uri):
        raise HTTPException(
            status_code=400,
            detail=f"Audio URI does not resolve to a stored blob: {body.audio.uri!r}",
        )
    if body.midi is not None and not blob.exists(body.midi.uri):
        raise HTTPException(
            status_code=400,
            detail=f"MIDI URI does not resolve to a stored blob: {body.midi.uri!r}",
        )

    # Build InputMetadata once — prefer_clean_source is threaded through
    # every variant so uploads can theoretically opt in too (the ingest
    # stage just ignores it when audio is already present).
    source_filename = None
    if body.audio is not None:
        source_filename = body.audio.source_filename
    elif body.midi is not None:
        source_filename = body.midi.source_filename
    metadata_kwargs = {
        "title": body.title,
        "artist": body.artist,
        "source_filename": source_filename,
        "prefer_clean_source": body.prefer_clean_source,
    }

    if body.audio is not None:
        bundle = InputBundle(
            schema_version=SCHEMA_VERSION,
            audio=body.audio,
            midi=None,
            metadata=InputMetadata(source="audio_upload", **metadata_kwargs),
        )
        variant: PipelineVariant = "audio_upload"
    elif body.midi is not None:
        bundle = InputBundle(
            schema_version=SCHEMA_VERSION,
            audio=None,
            midi=body.midi,
            metadata=InputMetadata(source="midi_upload", **metadata_kwargs),
        )
        variant = "midi_upload"
    else:
        assert body.title is not None
        bundle = InputBundle(
            schema_version=SCHEMA_VERSION,
            audio=None,
            midi=None,
            metadata=InputMetadata(source="title_lookup", **metadata_kwargs),
        )
        variant = "full"

    config = PipelineConfig(
        variant=variant,
        skip_humanizer=body.skip_humanizer,
        enable_refine=settings.refine_active,
        score_pipeline=settings.score_pipeline,
    )
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
