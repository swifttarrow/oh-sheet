"""Shared dependency providers — wire singletons here.

These are kept as ``lru_cache``-d module-level functions so:

  * FastAPI's ``Depends(...)`` resolves them once per process.
  * Tests can ``cache_clear()`` between cases (see tests/conftest.py).
"""
from __future__ import annotations

from functools import lru_cache

from ohsheet.config import settings
from ohsheet.jobs.manager import JobManager
from ohsheet.jobs.runner import PipelineRunner
from ohsheet.services.arrange import ArrangeService
from ohsheet.services.engrave import EngraveService
from ohsheet.services.humanize import HumanizeService
from ohsheet.services.ingest import IngestService
from ohsheet.services.transcribe import TranscribeService
from ohsheet.storage.local import LocalBlobStore


@lru_cache(maxsize=1)
def get_blob_store() -> LocalBlobStore:
    return LocalBlobStore(settings.blob_root)


@lru_cache(maxsize=1)
def get_runner() -> PipelineRunner:
    blob = get_blob_store()
    return PipelineRunner(
        ingest=IngestService(),
        transcribe=TranscribeService(),
        arrange=ArrangeService(),
        humanize=HumanizeService(),
        engrave=EngraveService(blob_store=blob),
    )


@lru_cache(maxsize=1)
def get_job_manager() -> JobManager:
    return JobManager(runner=get_runner())
