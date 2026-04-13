"""Celery task for the refine pipeline stage.

Thin claim-check wrapper around RefineService (Plan 03). The ONE place in
the codebase that constructs the Anthropic async client and reads the API
key (CFG-03 single-site rule). Exceptions propagate to PipelineRunner
(Plan 05), which translates them to `stage_completed` events with
`message="refine_skipped: <reason>"` per INT-03.

Input envelope (from PipelineRunner):
    {
      "payload_type": "HumanizedPerformance" | "PianoScore",
      "payload": <Pydantic.model_dump(mode="json") dict>,
      "job_id": str, "title": str, "composer": str,
    }

Outputs (both written to blob store):
    jobs/{job_id}/refine/output.json     - {"payload_type": "RefinedPerformance", "payload": ...}
    jobs/{job_id}/refine/llm_trace.json  - RefineTrace (flat dict)

Returns the output.json URI; the runner rewraps it for engrave's envelope
(engrave already handles payload_type="RefinedPerformance" per Phase-1 D-07
and gap-closure 01-09).
"""
from __future__ import annotations

import asyncio
import logging

import anthropic
from shared.contracts import HumanizedPerformance, PianoScore
from shared.storage.local import LocalBlobStore

from backend.config import settings
from backend.services.refine import RefineService
from backend.services.refine_validate import RefineValidator
from backend.workers.celery_app import celery_app

log = logging.getLogger(__name__)


@celery_app.task(name="refine.run")
def run(job_id: str, payload_uri: str) -> str:
    """Read refine input envelope, run RefineService, write output + trace.

    Defense-in-depth: raises RuntimeError if anthropic_api_key is None even
    though the CFG-04 400-gate in create_job should have prevented the
    submission from reaching this point. Belt-and-braces for operator
    misconfiguration (e.g., key cleared post-submission).
    """
    if settings.anthropic_api_key is None:
        raise RuntimeError(
            "refine worker invoked without OHSHEET_ANTHROPIC_API_KEY. "
            "The API layer should have rejected this submission with HTTP 400 "
            "(CFG-04); reaching the worker indicates a configuration drift."
        )

    blob = LocalBlobStore(settings.blob_root)
    envelope = blob.get_json(payload_uri)

    payload_type: str = envelope["payload_type"]
    payload_data = envelope["payload"]
    title: str = envelope.get("title", "Untitled")
    composer: str = envelope.get("composer", "Unknown")

    if payload_type == "HumanizedPerformance":
        performance: HumanizedPerformance | PianoScore = HumanizedPerformance.model_validate(payload_data)
    elif payload_type == "PianoScore":
        performance = PianoScore.model_validate(payload_data)
    else:
        raise ValueError(
            f"Unknown refine payload_type: {payload_type!r}. "
            f"Expected 'HumanizedPerformance' or 'PianoScore'."
        )

    # CFG-03 / STG-03: exactly one .get_secret_value() call in the entire codebase.
    # Any new caller of .get_secret_value() should be rejected in code review.
    client = anthropic.AsyncAnthropic(
        api_key=settings.anthropic_api_key.get_secret_value(),
    )
    validator = RefineValidator(settings)
    service = RefineService(client=client, validator=validator, settings=settings)

    metadata = {"title": title, "composer": composer}

    # asyncio.run() is safe with Celery's default prefork pool; breaks with gevent/eventlet.
    refined, trace = asyncio.run(service.run(performance, metadata))

    # Stage-boundary envelope for downstream engrave (Plan 05 runner rewraps this).
    output_uri = blob.put_json(
        f"jobs/{job_id}/refine/output.json",
        {
            "payload_type": "RefinedPerformance",
            "payload": refined.model_dump(mode="json"),
        },
    )

    # STG-10: llm_trace.json artifact (retrievable via GET /v1/artifacts/{id}/refine-trace).
    blob.put_json(
        f"jobs/{job_id}/refine/llm_trace.json",
        trace.model_dump(mode="json"),
    )

    log.info(
        "refine worker: job_id=%s applied=%d rejected=%d tokens_in=%d tokens_out=%d cost_usd=%.6f",
        job_id,
        len(trace.applied_edits),
        len(trace.rejected_edits),
        trace.usage.input_tokens,
        trace.usage.output_tokens,
        trace.estimated_cost_usd,
    )

    return output_uri
