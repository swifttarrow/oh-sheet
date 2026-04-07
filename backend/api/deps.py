"""Shared dependency providers — wire singletons here.

These are kept as ``lru_cache``-d module-level functions so:

  * FastAPI's ``Depends(...)`` resolves them once per process.
  * Tests can ``cache_clear()`` between cases (see tests/conftest.py).
"""
from __future__ import annotations

from functools import lru_cache

from backend.config import settings
from backend.jobs.manager import JobManager
from backend.jobs.runner import PipelineRunner
from backend.services.arrange import ArrangeService
from backend.services.engrave import EngraveService
from backend.services.humanize import HumanizeService
from backend.services.ingest import IngestService
from backend.services.transcribe import TranscribeService
from backend.storage.local import LocalBlobStore


@lru_cache(maxsize=1)
def get_blob_store() -> LocalBlobStore:
    return LocalBlobStore(settings.blob_root)


@lru_cache(maxsize=1)
def get_runner() -> PipelineRunner:
    blob = get_blob_store()
    return PipelineRunner(
        ingest=IngestService(blob_store=blob),
        transcribe=TranscribeService(),
        arrange=ArrangeService(),
        humanize=HumanizeService(),
        engrave=EngraveService(blob_store=blob),
    )


@lru_cache(maxsize=1)
def get_job_manager() -> JobManager:
    return JobManager(runner=get_runner())
