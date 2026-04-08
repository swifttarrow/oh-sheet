"""Celery workers for pipeline stages that live in the monolith."""
import backend.workers.engrave  # noqa: F401
import backend.workers.humanize  # noqa: F401
import backend.workers.ingest  # noqa: F401
