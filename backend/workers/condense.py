"""Celery task for the condense pipeline stage."""
import asyncio

from shared.contracts import TranscriptionResult
from shared.storage.local import LocalBlobStore

from backend.config import settings
from backend.services.condense import CondenseService
from backend.workers.celery_app import celery_app


@celery_app.task(name="condense.run")
def run(job_id: str, payload_uri: str) -> str:
    blob = LocalBlobStore(settings.blob_root)
    raw = blob.get_json(payload_uri)
    txr = TranscriptionResult.model_validate(raw)

    service = CondenseService()
    result = asyncio.run(service.run(txr))

    output_uri = blob.put_json(
        f"jobs/{job_id}/condense/output.json",
        result.model_dump(mode="json"),
    )
    return output_uri
