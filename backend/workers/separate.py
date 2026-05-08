"""Celery task for the source-separation pipeline stage."""
from shared.contracts import InputBundle
from shared.storage.local import LocalBlobStore

from backend.config import settings
from backend.services.separate import SeparateService
from backend.workers.celery_app import celery_app


@celery_app.task(name="separate.run")
def run(job_id: str, payload_uri: str) -> str:
    blob = LocalBlobStore(settings.blob_root)
    raw = blob.get_json(payload_uri)
    bundle = InputBundle.model_validate(raw)

    service = SeparateService(blob_store=blob)
    result = service.run(bundle)

    output_uri = blob.put_json(
        f"jobs/{job_id}/separate/output.json",
        result.model_dump(mode="json"),
    )
    return output_uri
