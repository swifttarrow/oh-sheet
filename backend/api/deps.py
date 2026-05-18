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
from backend.jobs.youtube_cache import YouTubeJobCache
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
def get_youtube_cache() -> YouTubeJobCache:
    """YouTube job cache wired to the Celery/result-backend Redis.

    The cache is fail-open: if redis-py can't connect, every get/set
    becomes a no-op (see YouTubeJobCache docstring). That means we
    construct the client unconditionally here — operators don't need
    to provision a separate Redis just to disable the cache. To
    actually disable it, set OHSHEET_YOUTUBE_CACHE_ENABLED=false; the
    route checks that flag before consulting the cache.
    """
    import redis  # noqa: PLC0415 — lazy so non-Redis dev paths don't import

    client = redis.Redis.from_url(settings.redis_url)
    return YouTubeJobCache(
        redis_client=client,
        ttl_seconds=settings.youtube_cache_ttl_sec,
    )


@lru_cache(maxsize=1)
def get_job_manager() -> JobManager:
    return JobManager(runner=get_runner(), youtube_cache=get_youtube_cache())
