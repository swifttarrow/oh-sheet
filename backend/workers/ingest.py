"""Celery task for the ingest pipeline stage."""
import asyncio

from shared.contracts import InputBundle
from shared.storage.local import LocalBlobStore

from backend.config import settings
from backend.services.ingest import IngestService
from backend.workers.celery_app import celery_app


@celery_app.task(name="ingest.run")
def run(job_id: str, payload_uri: str) -> str:
    blob = LocalBlobStore(settings.blob_root)
    raw = blob.get_json(payload_uri)
    bundle = InputBundle.model_validate(raw)

    service = IngestService(blob_store=blob)
    # asyncio.run() is safe with Celery's default prefork pool; breaks with gevent/eventlet.
    result = asyncio.run(service.run(bundle))

    output_uri = blob.put_json(
        f"jobs/{job_id}/ingest/output.json",
        result.model_dump(mode="json"),
    )
    return output_uri
