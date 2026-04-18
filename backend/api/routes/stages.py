"""Per-stage worker endpoints — OrchestratorCommand / WorkerResponse envelope.

These follow contracts §1: an external orchestrator (Temporal, Step Functions,
etc.) calls each stage as a stateless worker. Input/output payloads live in
blob storage; only URIs cross the wire.

The local /v1/jobs flow does **not** go through these endpoints — it uses
``PipelineRunner`` (Celery-backed). These routes call the same stage services
in-process for external orchestrators.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.api.deps import get_blob_store
from backend.contracts import (
    SCHEMA_VERSION,
    HumanizedPerformance,
    InputBundle,
    OrchestratorCommand,
    PianoScore,
    TranscriptionResult,
    WorkerResponse,
)
from backend.services.arrange import ArrangeService
from backend.services.condense import CondenseService
from backend.services.humanize import HumanizeService
from backend.services.ingest import IngestService
from backend.services.transcribe import TranscribeService
from backend.services.transform import TransformService
from backend.storage.local import LocalBlobStore

log = logging.getLogger(__name__)

router = APIRouter()


def _check_envelope(cmd: OrchestratorCommand) -> None:
    if cmd.schema_version != SCHEMA_VERSION:
        raise HTTPException(
            status_code=409,
            detail=(f"schema_version mismatch: worker={SCHEMA_VERSION}, command={cmd.schema_version}"),
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
    log.info(
        "worker stage %s job_id=%s step_id=%s payload_uri=%s",
        output_key,
        cmd.job_id,
        cmd.step_id,
        cmd.payload_uri,
    )
    try:
        payload_dict = blob.get_json(cmd.payload_uri)
        payload = input_model.model_validate(payload_dict)
        result = await coro(payload)
        out_uri = blob.put_json(
            f"jobs/{cmd.job_id}/{cmd.step_id}/{output_key}",
            result.model_dump(mode="json"),
        )
        log.info(
            "worker stage %s job_id=%s step_id=%s output_uri=%s",
            output_key,
            cmd.job_id,
            cmd.step_id,
            out_uri,
        )
        return WorkerResponse(
            schema_version=SCHEMA_VERSION,
            job_id=cmd.job_id,
            status="success",
            output_uri=out_uri,
        )
    except Exception as exc:  # noqa: BLE001 — boundary
        log.exception(
            "worker stage %s job_id=%s step_id=%s failed",
            output_key,
            cmd.job_id,
            cmd.step_id,
        )
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
) -> WorkerResponse:
    ingest = IngestService(blob_store=blob)

    async def coro(payload: InputBundle) -> InputBundle:
        return await ingest.run(payload)

    return await _run_stage(
        cmd,
        blob,
        input_model=InputBundle,
        coro=coro,
        output_key="input_bundle.json",
    )


@router.post("/stages/transcribe", response_model=WorkerResponse)
async def stage_transcribe(
    cmd: OrchestratorCommand,
    blob: Annotated[LocalBlobStore, Depends(get_blob_store)],
) -> WorkerResponse:
    transcribe = TranscribeService(blob_store=blob)

    async def coro(payload: InputBundle) -> TranscriptionResult:
        return await transcribe.run(payload, job_id=cmd.job_id)

    return await _run_stage(
        cmd,
        blob,
        input_model=InputBundle,
        coro=coro,
        output_key="transcription.json",
    )


@router.post("/stages/arrange", response_model=WorkerResponse)
async def stage_arrange(
    cmd: OrchestratorCommand,
    blob: Annotated[LocalBlobStore, Depends(get_blob_store)],
) -> WorkerResponse:
    arrange = ArrangeService()

    async def coro(payload: TranscriptionResult) -> PianoScore:
        return await arrange.run(payload, blob_store=blob)

    return await _run_stage(
        cmd,
        blob,
        input_model=TranscriptionResult,
        coro=coro,
        output_key="score.json",
    )


@router.post("/stages/condense", response_model=WorkerResponse)
async def stage_condense(
    cmd: OrchestratorCommand,
    blob: Annotated[LocalBlobStore, Depends(get_blob_store)],
) -> WorkerResponse:
    condense = CondenseService()

    async def coro(payload: TranscriptionResult) -> PianoScore:
        return await condense.run(payload)

    return await _run_stage(
        cmd,
        blob,
        input_model=TranscriptionResult,
        coro=coro,
        output_key="score_condensed.json",
    )


@router.post("/stages/transform", response_model=WorkerResponse)
async def stage_transform(
    cmd: OrchestratorCommand,
    blob: Annotated[LocalBlobStore, Depends(get_blob_store)],
) -> WorkerResponse:
    transform = TransformService()

    async def coro(payload: PianoScore) -> PianoScore:
        return await transform.run(payload)

    return await _run_stage(
        cmd,
        blob,
        input_model=PianoScore,
        coro=coro,
        output_key="score.json",
    )


@router.post("/stages/humanize", response_model=WorkerResponse)
async def stage_humanize(
    cmd: OrchestratorCommand,
    blob: Annotated[LocalBlobStore, Depends(get_blob_store)],
) -> WorkerResponse:
    humanize = HumanizeService()

    async def coro(payload: PianoScore) -> HumanizedPerformance:
        return await humanize.run(payload)

    return await _run_stage(
        cmd,
        blob,
        input_model=PianoScore,
        coro=coro,
        output_key="performance.json",
    )


