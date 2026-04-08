"""Celery application instance for the decomposer worker."""
import os

from celery import Celery

_redis_url = os.environ.get("OHSHEET_REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "decomposer",
    broker=_redis_url,
    backend=_redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
)
