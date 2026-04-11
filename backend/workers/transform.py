"""Celery task for the transform pipeline stage."""
import asyncio

from shared.contracts import PianoScore
from shared.storage.local import LocalBlobStore

from backend.config import settings
from backend.services.transform import TransformService
from backend.workers.celery_app import celery_app


@celery_app.task(name="transform.run")
def run(job_id: str, payload_uri: str) -> str:
    blob = LocalBlobStore(settings.blob_root)
    raw = blob.get_json(payload_uri)
    score = PianoScore.model_validate(raw)

    service = TransformService(blob_store=blob)
    result = asyncio.run(service.run(score, job_id=job_id))

    output_uri = blob.put_json(
        f"jobs/{job_id}/transform/output.json",
        result.model_dump(mode="json"),
    )
    return output_uri
