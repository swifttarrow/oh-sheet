"""Decomposer Celery task -- stub delegates to TranscribeService."""
import asyncio
import os
from pathlib import Path

from decomposer.celery_app import celery_app
from shared.contracts import InputBundle
from shared.storage.local import LocalBlobStore

_BLOB_ROOT = Path(os.environ.get("OHSHEET_BLOB_ROOT", "./blob"))


def _get_blob_store() -> LocalBlobStore:
    return LocalBlobStore(_BLOB_ROOT)


@celery_app.task(name="decomposer.run")
def run(job_id: str, payload_uri: str) -> str:
    blob = _get_blob_store()
    raw = blob.get_json(payload_uri)
    bundle = InputBundle.model_validate(raw)

    # Stub: delegate to existing TranscribeService
    from backend.services.transcribe import TranscribeService

    service = TranscribeService()
    result = asyncio.run(service.run(bundle))

    output_uri = blob.put_json(
        f"jobs/{job_id}/decomposer/output.json",
        result.model_dump(mode="json"),
    )
    return output_uri
