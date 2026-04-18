"""Celery workers for pipeline stages that live in the monolith."""
import backend.workers.arrange  # noqa: F401
import backend.workers.condense  # noqa: F401
import backend.workers.humanize  # noqa: F401
import backend.workers.ingest  # noqa: F401
import backend.workers.refine  # noqa: F401
import backend.workers.transcribe  # noqa: F401
import backend.workers.transform  # noqa: F401
