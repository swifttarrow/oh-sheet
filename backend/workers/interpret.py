"""Celery task for the interpret pipeline stage.

Reads an envelope from the input blob:

    {
        "txr": <TranscriptionResult JSON>,
        "prompt": str,
        "title_hint": str | None,
        "artist_hint": str | None,
    }

Calls InterpretService to enrich the txr with ArrangementHints, then writes:

    {"txr": <enriched TranscriptionResult JSON>}

to ``jobs/{job_id}/interpret/output.json`` and returns the output URI.
"""
import asyncio
import logging

from shared.contracts import TranscriptionResult
from shared.storage.local import LocalBlobStore

from backend.config import settings
from backend.services.interpret import InterpretService
from backend.workers.celery_app import celery_app

log = logging.getLogger(__name__)

# ``asyncio.run`` creates and tears down its own event loop on every call.
# That's fine under Celery's default ``prefork`` pool (each task runs in
# a worker subprocess with no ambient loop), but breaks under ``gevent``
# / ``eventlet`` where the pool's monkey-patched loop is already live —
# the second ``asyncio.run`` call would raise or, worse, silently mis-
# schedule. Guard the assumption explicitly so a future ``-P gevent``
# config change fails loudly here instead of at first cover-mode job.
_SUPPORTED_POOLS: frozenset[str] = frozenset({"prefork", "solo", "threads", "processes"})


def _assert_pool_supports_asyncio_run() -> None:
    pool = celery_app.conf.get("worker_pool", "prefork")
    if pool not in _SUPPORTED_POOLS:
        raise RuntimeError(
            f"interpret worker requires a Celery pool that does not pre-install "
            f"an asyncio event loop (got worker_pool={pool!r}); "
            f"asyncio.run() will conflict under gevent/eventlet."
        )


@celery_app.task(name="interpret.run")
def run(job_id: str, payload_uri: str) -> str:
    _assert_pool_supports_asyncio_run()
    blob = LocalBlobStore(settings.blob_root)
    envelope = blob.get_json(payload_uri)

    raw_txr = envelope.get("txr")
    if raw_txr is None:
        raise ValueError("interpret envelope missing required field 'txr'")
    prompt = envelope.get("prompt")
    if prompt is None:
        raise ValueError("interpret envelope missing required field 'prompt'")
    txr = TranscriptionResult.model_validate(raw_txr)
    title_hint: str | None = envelope.get("title_hint")
    artist_hint: str | None = envelope.get("artist_hint")

    service = InterpretService(blob_store=blob)
    result = asyncio.run(
        service.run(
            txr,
            prompt=prompt,
            title_hint=title_hint,
            artist_hint=artist_hint,
        ),
    )

    out = {"txr": result.model_dump(mode="json")}
    output_uri = blob.put_json(
        f"jobs/{job_id}/interpret/output.json",
        out,
    )
    return output_uri
