"""Per-stage worker endpoints — OrchestratorCommand / WorkerResponse envelope.

These follow contracts §1: an external orchestrator (Temporal, Step Functions,
etc.) calls each stage as a stateless worker. Input/output payloads live in
blob storage; only URIs cross the wire.

The local /v1/jobs flow does **not** go through these endpoints — it calls
the services directly via the in-process PipelineRunner. These endpoints exist
so the same service code can be invoked by an out-of-process orchestrator.
"""
from __future__ import annotations

from typing import Annotated, Any, Awaitable, Callable

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.api.deps import get_blob_store, get_runner
from backend.contracts import (
    SCHEMA_VERSION,
    EngravedOutput,
    HumanizedPerformance,
    InputBundle,
    OrchestratorCommand,
    PianoScore,
    TranscriptionResult,
    WorkerResponse,
)
from backend.jobs.runner import PipelineRunner
from backend.storage.local import LocalBlobStore

router = APIRouter()


def _check_envelope(cmd: OrchestratorCommand) -> None:
    if cmd.schema_version != SCHEMA_VERSION:
        raise HTTPException(
            status_code=409,
            detail=(
                f"schema_version mismatch: worker={SCHEMA_VERSION}, "
                f"command={cmd.schema_version}"
            ),
        )


async def _run_stage(
    cmd: OrchestratorCommand,
    blob: LocalBlobStore,
    *,
    input_model: type[BaseModel],
    coro: Callable[[Any], Awaitable[BaseModel]],
    output_key: str,
) -> WorkerResponse:
    _check_envelope(cmd)
    try:
        payload_dict = blob.get_json(cmd.payload_uri)
        payload = input_model.model_validate(payload_dict)
        result = await coro(payload)
        out_uri = blob.put_json(
            f"jobs/{cmd.job_id}/{cmd.step_id}/{output_key}",
            result.model_dump(mode="json"),
        )
        return WorkerResponse(
            schema_version=SCHEMA_VERSION,
            job_id=cmd.job_id,
            status="success",
            output_uri=out_uri,
        )
    except Exception as exc:  # noqa: BLE001 — boundary
        return WorkerResponse(
            schema_version=SCHEMA_VERSION,
            job_id=cmd.job_id,
            status="fatal_error",
            output_uri=None,
            logs=repr(exc),
        )


@router.post("/stages/ingest", response_model=WorkerResponse)
async def stage_ingest(
    cmd: OrchestratorCommand,
    blob: Annotated[LocalBlobStore, Depends(get_blob_store)],
    runner: Annotated[PipelineRunner, Depends(get_runner)],
) -> WorkerResponse:
    return await _run_stage(
        cmd, blob,
        input_model=InputBundle,
        coro=runner.ingest.run,
        output_key="input_bundle.json",
    )


@router.post("/stages/transcribe", response_model=WorkerResponse)
async def stage_transcribe(
    cmd: OrchestratorCommand,
    blob: Annotated[LocalBlobStore, Depends(get_blob_store)],
    runner: Annotated[PipelineRunner, Depends(get_runner)],
) -> WorkerResponse:
    return await _run_stage(
        cmd, blob,
        input_model=InputBundle,
        coro=runner.transcribe.run,
        output_key="transcription.json",
    )


@router.post("/stages/arrange", response_model=WorkerResponse)
async def stage_arrange(
    cmd: OrchestratorCommand,
    blob: Annotated[LocalBlobStore, Depends(get_blob_store)],
    runner: Annotated[PipelineRunner, Depends(get_runner)],
) -> WorkerResponse:
    return await _run_stage(
        cmd, blob,
        input_model=TranscriptionResult,
        coro=runner.arrange.run,
        output_key="score.json",
    )


@router.post("/stages/humanize", response_model=WorkerResponse)
async def stage_humanize(
    cmd: OrchestratorCommand,
    blob: Annotated[LocalBlobStore, Depends(get_blob_store)],
    runner: Annotated[PipelineRunner, Depends(get_runner)],
) -> WorkerResponse:
    return await _run_stage(
        cmd, blob,
        input_model=PianoScore,
        coro=runner.humanize.run,
        output_key="performance.json",
    )


@router.post("/stages/engrave", response_model=WorkerResponse)
async def stage_engrave(
    cmd: OrchestratorCommand,
    blob: Annotated[LocalBlobStore, Depends(get_blob_store)],
    runner: Annotated[PipelineRunner, Depends(get_runner)],
) -> WorkerResponse:
    # Engrave needs the job_id (for blob keying) and title/composer; we
    # forward what's available from the envelope here.
    async def coro(payload: HumanizedPerformance) -> EngravedOutput:
        return await runner.engrave.run(
            payload, job_id=cmd.job_id, title="Untitled", composer="Unknown",
        )

    return await _run_stage(
        cmd, blob,
        input_model=HumanizedPerformance,
        coro=coro,
        output_key="engraved.json",
    )
