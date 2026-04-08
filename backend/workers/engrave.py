"""Celery task for the engrave pipeline stage.

Engrave is unique: it accepts either HumanizedPerformance or PianoScore,
plus extra args (job_id, title, composer). The task envelope wraps these
as a JSON object with a `payload_type` discriminator.
"""
import asyncio

from backend.config import settings
from backend.services.engrave import EngraveService
from backend.workers.celery_app import celery_app
from shared.contracts import HumanizedPerformance, PianoScore
from shared.storage.local import LocalBlobStore


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
    else:
        payload = PianoScore.model_validate(payload_data)

    service = EngraveService(blob_store=blob)
    result = asyncio.run(service.run(payload, job_id=job_id, title=title, composer=composer))

    output_uri = blob.put_json(
        f"jobs/{job_id}/engrave/output.json",
        result.model_dump(mode="json"),
    )
    return output_uri
