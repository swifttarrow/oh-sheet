"""Celery task for the refine pipeline stage.

Like engrave, refine accepts a discriminated envelope so the worker
can hydrate the right Pydantic model:

    {
        "payload_type": "PianoScore" | "HumanizedPerformance",
        "payload": <model JSON>,
        "title_hint": str | None,
        "artist_hint": str | None,
        "filename_hint": str | None,
    }

The output envelope mirrors the input shape, so the runner can unwrap
it and feed the refined score straight into engrave.
"""
import asyncio

from shared.contracts import HumanizedPerformance, PianoScore
from shared.storage.local import LocalBlobStore

from backend.config import settings
from backend.services.refine import RefineService
from backend.workers.celery_app import celery_app


@celery_app.task(name="refine.run")
def run(job_id: str, payload_uri: str) -> str:
    blob = LocalBlobStore(settings.blob_root)
    raw = blob.get_json(payload_uri)

    payload_type = raw.get("payload_type")
    if payload_type is None:
        raise ValueError("refine envelope missing required field 'payload_type'")
    payload_data = raw.get("payload")
    if payload_data is None:
        raise ValueError("refine envelope missing required field 'payload'")
    title_hint = raw.get("title_hint")
    artist_hint = raw.get("artist_hint")
    filename_hint = raw.get("filename_hint")

    if payload_type == "HumanizedPerformance":
        payload = HumanizedPerformance.model_validate(payload_data)
    elif payload_type == "PianoScore":
        payload = PianoScore.model_validate(payload_data)
    else:
        raise ValueError(
            f"Unknown payload_type: {payload_type!r}. "
            "Expected 'HumanizedPerformance' or 'PianoScore'."
        )

    service = RefineService(blob_store=blob)
    # asyncio.run() is safe with Celery's default prefork pool; breaks with gevent/eventlet.
    result = asyncio.run(
        service.run(
            payload,
            title_hint=title_hint,
            artist_hint=artist_hint,
            filename_hint=filename_hint,
        ),
    )

    out = {
        "payload_type": payload_type,
        "payload": result.model_dump(mode="json"),
    }
    output_uri = blob.put_json(
        f"jobs/{job_id}/refine/output.json",
        out,
    )
    return output_uri
