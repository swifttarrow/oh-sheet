"""JobManager — in-memory job registry with WebSocket pub/sub.

Single-process v1. Each job has its own list of subscribers (asyncio Queues).
The runner emits JobEvents which the manager fans out to every queue.
Replace with Redis Streams / Postgres LISTEN-NOTIFY for multi-worker deploys.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field

from backend.contracts import EngravedOutput, InputBundle, PipelineConfig
from backend.jobs.events import JobEvent, JobStatus
from backend.jobs.runner import PipelineRunner

log = logging.getLogger(__name__)


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus
    config: PipelineConfig
    bundle: InputBundle
    result: EngravedOutput | None = None
    error: str | None = None
    events: list[JobEvent] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    task: asyncio.Task | None = None


class JobManager:
    def __init__(self, runner: PipelineRunner) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._runner = runner
        self._lock = asyncio.Lock()

    # ---- pub/sub fan-out -------------------------------------------------

    def _emit(self, record: JobRecord, event: JobEvent) -> None:
        record.events.append(event)
        # Iterate over a copy so we can drop dead subscribers in-place.
        for queue in list(record.subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                log.warning("dropping slow subscriber for job %s", record.job_id)
                record.subscribers.remove(queue)

    # ---- submission ------------------------------------------------------

    async def submit(self, bundle: InputBundle, config: PipelineConfig) -> JobRecord:
        job_id = uuid.uuid4().hex[:12]
        record = JobRecord(
            job_id=job_id,
            status="pending",
            config=config,
            bundle=bundle,
        )
        async with self._lock:
            self._jobs[job_id] = record

        log.info(
            "job created job_id=%s variant=%s source=%s",
            job_id,
            config.variant,
            bundle.metadata.source,
        )
        self._emit(record, JobEvent(job_id=job_id, type="job_created"))
        record.task = asyncio.create_task(self._execute(record))
        return record

    async def _execute(self, record: JobRecord) -> None:
        record.status = "running"
        self._emit(record, JobEvent(job_id=record.job_id, type="job_started"))
        log.info("job running job_id=%s", record.job_id)
        t0 = time.perf_counter()
        try:
            result = await self._runner.run(
                job_id=record.job_id,
                bundle=record.bundle,
                config=record.config,
                on_event=lambda ev: self._emit(record, ev),
            )
            record.result = result
            record.status = "succeeded"
            elapsed_ms = (time.perf_counter() - t0) * 1000
            log.info(
                "job succeeded job_id=%s duration_ms=%.0f pdf_uri=%s",
                record.job_id,
                elapsed_ms,
                result.pdf_uri,
            )
            self._emit(
                record,
                JobEvent(
                    job_id=record.job_id,
                    type="job_succeeded",
                    data=result.model_dump(mode="json"),
                ),
            )
        except Exception as exc:  # noqa: BLE001 — top-level supervisor
            elapsed_ms = (time.perf_counter() - t0) * 1000
            log.exception(
                "job failed job_id=%s duration_ms=%.0f",
                record.job_id,
                elapsed_ms,
            )
            record.status = "failed"
            record.error = repr(exc)
            self._emit(
                record,
                JobEvent(job_id=record.job_id, type="job_failed", message=str(exc)),
            )

    # ---- queries ---------------------------------------------------------

    def get(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    def list(self) -> list[JobRecord]:
        return list(self._jobs.values())

    # ---- websocket subscription -----------------------------------------

    async def subscribe(self, job_id: str) -> asyncio.Queue | None:
        record = self.get(job_id)
        if record is None:
            return None
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        # Replay everything that's already happened so late subscribers see
        # the full event history (e.g. job_created/job_started + stages).
        for ev in record.events:
            queue.put_nowait(ev)
        record.subscribers.append(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        record = self.get(job_id)
        if record and queue in record.subscribers:
            record.subscribers.remove(queue)
