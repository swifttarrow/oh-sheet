"""Celery task for the humanize pipeline stage."""
import asyncio

from shared.contracts import PianoScore
from shared.storage.local import LocalBlobStore

from backend.config import settings
from backend.services.humanize import HumanizeService
from backend.workers.celery_app import celery_app


@celery_app.task(name="humanize.run")
def run(job_id: str, payload_uri: str) -> str:
    blob = LocalBlobStore(settings.blob_root)
    raw = blob.get_json(payload_uri)
    score = PianoScore.model_validate(raw)

    service = HumanizeService()
    # asyncio.run() is safe with Celery's default prefork pool; breaks with gevent/eventlet.
    result = asyncio.run(service.run(score))

    output_uri = blob.put_json(
        f"jobs/{job_id}/humanize/output.json",
        result.model_dump(mode="json"),
    )
    return output_uri
