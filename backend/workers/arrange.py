"""Celery task for the arrange pipeline stage."""
import asyncio

from shared.contracts import TranscriptionResult
from shared.storage.local import LocalBlobStore

from backend.config import settings
from backend.services.arrange import ArrangeService
from backend.services.arrange_simplify import simplify_score
from backend.workers.celery_app import celery_app


@celery_app.task(name="arrange.run")
def run(job_id: str, payload_uri: str) -> str:
    blob = LocalBlobStore(settings.blob_root)
    raw = blob.get_json(payload_uri)
    txr = TranscriptionResult.model_validate(raw)

    service = ArrangeService()
    # asyncio.run() is safe with Celery's default prefork pool; breaks with gevent/eventlet.
    result = asyncio.run(service.run(txr, blob_store=blob))

    # Post-process: aggressive simplification for sheet music readability.
    # Swaps dense per-note transcription output for a cleaner notation layer.
    if settings.arrange_simplify_enabled:
        result = simplify_score(
            result,
            min_velocity=settings.arrange_simplify_min_velocity,
            chord_merge_beats=settings.arrange_simplify_chord_merge_beats,
            max_onsets_per_beat=settings.arrange_simplify_max_onsets_per_beat,
            min_duration_beats=settings.arrange_simplify_min_duration_beats,
        )

    output_uri = blob.put_json(
        f"jobs/{job_id}/arrange/output.json",
        result.model_dump(mode="json"),
    )
    return output_uri
