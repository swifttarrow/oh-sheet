"""Celery task for the arrange pipeline stage."""
import asyncio

from shared.contracts import TranscriptionResult
from shared.storage.local import LocalBlobStore

from backend.config import settings
from backend.services.arrange import ArrangeService
from backend.workers.celery_app import celery_app


@celery_app.task(name="arrange.run")
def run(job_id: str, payload_uri: str) -> str:
    blob = LocalBlobStore(settings.blob_root)
    raw = blob.get_json(payload_uri)
    txr = TranscriptionResult.model_validate(raw)

    service = ArrangeService()
    # asyncio.run() is safe with Celery's default prefork pool; breaks with gevent/eventlet.
    result = asyncio.run(service.run(txr))

    output_uri = blob.put_json(
        f"jobs/{job_id}/arrange/output.json",
        result.model_dump(mode="json"),
    )
    return output_uri
