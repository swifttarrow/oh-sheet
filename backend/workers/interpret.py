"""Celery task for the interpret pipeline stage.

Reads an envelope from the input blob:

    {
        "txr": <TranscriptionResult JSON>,
        "prompt": str,
        "title_hint": str | None,
        "artist_hint": str | None,
    }

Calls InterpretService to enrich the txr with ArrangementHints, then writes:

    {"txr": <enriched TranscriptionResult JSON>}

to ``jobs/{job_id}/interpret/output.json`` and returns the output URI.
"""
import asyncio

from shared.contracts import TranscriptionResult
from shared.storage.local import LocalBlobStore

from backend.config import settings
from backend.services.interpret import InterpretService
from backend.workers.celery_app import celery_app


@celery_app.task(name="interpret.run")
def run(job_id: str, payload_uri: str) -> str:
    blob = LocalBlobStore(settings.blob_root)
    envelope = blob.get_json(payload_uri)

    txr = TranscriptionResult.model_validate(envelope["txr"])
    prompt: str = envelope["prompt"]
    title_hint: str | None = envelope.get("title_hint")
    artist_hint: str | None = envelope.get("artist_hint")

    service = InterpretService(blob_store=blob)
    # asyncio.run() is safe with Celery's default prefork pool;
    # breaks with gevent/eventlet.
    result = asyncio.run(
        service.run(
            txr,
            prompt=prompt,
            title_hint=title_hint,
            artist_hint=artist_hint,
        ),
    )

    out = {"txr": result.model_dump(mode="json")}
    output_uri = blob.put_json(
        f"jobs/{job_id}/interpret/output.json",
        out,
    )
    return output_uri
