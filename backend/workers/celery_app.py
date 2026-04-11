"""Celery application instance shared by all monolith workers."""
from celery import Celery

from backend.config import settings

celery_app = Celery(
    "ohsheet",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_default_queue="default",
    task_routes={
        "ingest.run": {"queue": "ingest"},
        "transcribe.run": {"queue": "transcribe"},
        "arrange.run": {"queue": "arrange"},
        "condense.run": {"queue": "arrange"},
        "transform.run": {"queue": "arrange"},
        "humanize.run": {"queue": "humanize"},
        "engrave.run": {"queue": "engrave"},
    },
)

# Auto-discover tasks in the backend.workers package.
celery_app.autodiscover_tasks(["backend.workers"])
