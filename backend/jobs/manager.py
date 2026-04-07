"""JobManager — in-memory job registry with WebSocket pub/sub.

Single-process v1. Each job has its own list of subscribers (asyncio Queues).
The runner emits JobEvents which the manager fans out to every queue.
Replace with Redis Streams / Postgres LISTEN-NOTIFY for multi-worker deploys.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

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
    result: Optional[EngravedOutput] = None
    error: Optional[str] = None
    events: list[JobEvent] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    task: Optional[asyncio.Task] = None


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

        self._emit(record, JobEvent(job_id=job_id, type="job_created"))
        record.task = asyncio.create_task(self._execute(record))
        return record

    async def _execute(self, record: JobRecord) -> None:
        record.status = "running"
        self._emit(record, JobEvent(job_id=record.job_id, type="job_started"))
        try:
            result = await self._runner.run(
                job_id=record.job_id,
                bundle=record.bundle,
                config=record.config,
                on_event=lambda ev: self._emit(record, ev),
            )
            record.result = result
            record.status = "succeeded"
            self._emit(
                record,
                JobEvent(
                    job_id=record.job_id,
                    type="job_succeeded",
                    data=result.model_dump(mode="json"),
                ),
            )
        except Exception as exc:  # noqa: BLE001 — top-level supervisor
            log.exception("job %s failed", record.job_id)
            record.status = "failed"
            record.error = repr(exc)
            self._emit(
                record,
                JobEvent(job_id=record.job_id, type="job_failed", message=str(exc)),
            )

    # ---- queries ---------------------------------------------------------

    def get(self, job_id: str) -> Optional[JobRecord]:
        return self._jobs.get(job_id)

    def list(self) -> list[JobRecord]:
        return list(self._jobs.values())

    # ---- websocket subscription -----------------------------------------

    async def subscribe(self, job_id: str) -> Optional[asyncio.Queue]:
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
