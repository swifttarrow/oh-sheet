"""Assembler Celery task — stub delegates to ArrangeService."""
import asyncio
import os
from pathlib import Path

from assembler.celery_app import celery_app
from shared.contracts import TranscriptionResult
from shared.storage.local import LocalBlobStore

_BLOB_ROOT = Path(os.environ.get("OHSHEET_BLOB_ROOT", "./blob"))


def _get_blob_store() -> LocalBlobStore:
    return LocalBlobStore(_BLOB_ROOT)


@celery_app.task(name="assembler.run")
def run(job_id: str, payload_uri: str) -> str:
    blob = _get_blob_store()
    raw = blob.get_json(payload_uri)
    txr = TranscriptionResult.model_validate(raw)

    from backend.services.arrange import ArrangeService

    service = ArrangeService()
    result = asyncio.run(service.run(txr))

    output_uri = blob.put_json(
        f"jobs/{job_id}/assembler/output.json",
        result.model_dump(mode="json"),
    )
    return output_uri
