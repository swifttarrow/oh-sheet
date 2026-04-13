"""Job submission and inspection."""
from __future__ import annotations

import logging
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

log = logging.getLogger(__name__)


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

    ``enable_refine`` opts the job into the LLM refine stage between humanize
    and engrave. Defaults to False; requires OHSHEET_ANTHROPIC_API_KEY to be
    set (returns 400 otherwise) and respects OHSHEET_REFINE_KILL_SWITCH.
    """

    audio: RemoteAudioFile | None = None
    midi: RemoteMidiFile | None = None
    title: str | None = None
    artist: str | None = None
    prefer_clean_source: bool = False

    skip_humanizer: bool = False
    difficulty: Difficulty = "intermediate"

    # Phase 1 (CFG-02): opt-in to the LLM refine stage. Default False
    # preserves backward compatibility. When True, the pipeline runs an
    # additional refine step between humanize and engrave (or between
    # arrange/transform and engrave for sheet_only). Requires
    # OHSHEET_ANTHROPIC_API_KEY to be set; the route returns HTTP 400
    # otherwise (CFG-04). If OHSHEET_REFINE_KILL_SWITCH is true, the
    # route silently coerces this to False (CFG-06).
    enable_refine: bool = False


class JobSummary(BaseModel):
    job_id: str
    status: str
    variant: str
    title: str | None = None
    artist: str | None = None
    error: str | None = None
    result: EngravedOutput | None = None


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

    # Phase 1 (CFG-04): fail fast when enable_refine=true but no API key
    # is configured. The check runs BEFORE PipelineConfig construction so
    # the user sees a clear 400 about the missing env var rather than a
    # downstream validation error.
    effective_enable_refine = body.enable_refine
    if effective_enable_refine and settings.anthropic_api_key is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "enable_refine=true requires OHSHEET_ANTHROPIC_API_KEY to be set. "
                "Set the env var (restart the service) or submit with enable_refine=false."
            ),
        )

    # Phase 1 (CFG-06): kill switch silently coerces enable_refine to False.
    # The log.warning is emitted after manager.submit so we can include the
    # job_id for operator-diffable structured logs. Per Claude's Discretion
    # in CONTEXT.md: no HTTP error, no stage_completed event — if refine is
    # never in the plan, there's no stage to emit for. PipelineConfig is
    # never told about the kill switch — it stays pure (Pitfall 3).
    refine_coerced_by_kill_switch = False
    if effective_enable_refine and settings.refine_kill_switch:
        effective_enable_refine = False
        refine_coerced_by_kill_switch = True

    # Build InputMetadata once — prefer_clean_source is threaded through
    # every variant so uploads can theoretically opt in too (the ingest
    # stage just ignores it when audio is already present).
    metadata_kwargs = {
        "title": body.title,
        "artist": body.artist,
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
        score_pipeline=settings.score_pipeline,
        enable_refine=effective_enable_refine,
    )
    record = await manager.submit(bundle, config)

    # Phase 1 (CFG-06): emit the kill-switch coercion signal ONCE per job,
    # with job_id so operators can correlate with the job's event stream.
    # Message format is load-bearing — the test V13 asserts on the exact
    # substring "refine kill switch active; stripping refine from plan".
    if refine_coerced_by_kill_switch:
        log.warning(
            "refine kill switch active; stripping refine from plan for job_id=%s",
            record.job_id,
        )

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
