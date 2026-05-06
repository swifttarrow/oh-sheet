"""Celery task for the transcribe pipeline stage."""
import asyncio

from shared.contracts import InputBundle
from shared.storage.local import LocalBlobStore

from backend.config import settings
from backend.services.transcribe import TranscribeService
from backend.workers.celery_app import celery_app


@celery_app.task(name="transcribe.run")
def run(job_id: str, payload_uri: str) -> str:
    blob = LocalBlobStore(settings.blob_root)
    raw = blob.get_json(payload_uri)
    bundle = InputBundle.model_validate(raw)

    service = TranscribeService(blob_store=blob)
    # asyncio.run() is safe with Celery's default prefork pool; breaks with gevent/eventlet.
    # ``variant_hint`` survives the InputBundle → JSON boundary and lets
    # the dispatcher in TranscribeService pick the AMT-APC cover-mode
    # pipeline when the user chose "Piano cover" upstream.
    result = asyncio.run(
        service.run(
            bundle,
            job_id=job_id,
            variant=bundle.metadata.variant_hint,
        )
    )

    output_uri = blob.put_json(
        f"jobs/{job_id}/transcribe/output.json",
        result.model_dump(mode="json"),
    )
    return output_uri
