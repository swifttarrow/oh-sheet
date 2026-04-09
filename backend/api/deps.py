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
from backend.storage.local import LocalBlobStore
from backend.workers.celery_app import celery_app


@lru_cache(maxsize=1)
def get_blob_store() -> LocalBlobStore:
    return LocalBlobStore(settings.blob_root)


@lru_cache(maxsize=1)
def get_runner() -> PipelineRunner:
    return PipelineRunner(
        blob_store=get_blob_store(),
        celery_app=celery_app,
    )


@lru_cache(maxsize=1)
def get_job_manager() -> JobManager:
    return JobManager(runner=get_runner())
