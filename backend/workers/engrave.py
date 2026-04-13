"""Celery task for the engrave pipeline stage.

Engrave accepts HumanizedPerformance, PianoScore, or (Phase 1+)
RefinedPerformance payloads plus extra args (job_id, title, composer).
The task envelope wraps these as a JSON object with a ``payload_type``
discriminator. For ``payload_type == "RefinedPerformance"`` the
worker unwraps to the nested ``refined_performance`` (HumanizedPerformance)
per D-07 in .planning/phases/01-contracts-and-plumbing/01-CONTEXT.md.
"""
import asyncio

from shared.contracts import HumanizedPerformance, PianoScore, RefinedPerformance
from shared.storage.local import LocalBlobStore

from backend.config import settings
from backend.services.engrave import EngraveService
from backend.workers.celery_app import celery_app


@celery_app.task(name="engrave.run")
def run(job_id: str, payload_uri: str) -> str:
    blob = LocalBlobStore(settings.blob_root)
    raw = blob.get_json(payload_uri)

    payload_type = raw["payload_type"]
    payload_data = raw["payload"]
    title = raw.get("title", "Untitled")
    composer = raw.get("composer", "Unknown")

    if payload_type == "HumanizedPerformance":
        payload = HumanizedPerformance.model_validate(payload_data)
    elif payload_type == "PianoScore":
        payload = PianoScore.model_validate(payload_data)
    elif payload_type == "RefinedPerformance":
        # D-07: unwrap to the inner HumanizedPerformance — engrave renders
        # the POST-edit result exactly as if it had arrived on the
        # HumanizedPerformance path. Edits, citations, model, and digest
        # are preserved on the refined blob but not consumed by engrave;
        # they are for observability (llm_trace.json artifact, Phase 2).
        refined = RefinedPerformance.model_validate(payload_data)
        payload = refined.refined_performance
    else:
        raise ValueError(
            f"Unknown payload_type: {payload_type!r}. "
            f"Expected 'HumanizedPerformance', 'PianoScore', or 'RefinedPerformance'."
        )

    service = EngraveService(blob_store=blob)
    # asyncio.run() is safe with Celery's default prefork pool; breaks with gevent/eventlet.
    result = asyncio.run(service.run(payload, job_id=job_id, title=title, composer=composer))

    output_uri = blob.put_json(
        f"jobs/{job_id}/engrave/output.json",
        result.model_dump(mode="json"),
    )
    return output_uri
