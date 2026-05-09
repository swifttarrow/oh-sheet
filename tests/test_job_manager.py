"""Unit tests for the in-memory JobManager.

Exercises behaviour that doesn't need the full FastAPI route layer —
specifically the bounded-completed-jobs eviction policy and the
exception-surface scrubbing applied to ``JobRecord.error``.
"""
from __future__ import annotations

import asyncio

import pytest

from backend.config import settings
from backend.contracts import (
    EngravedOutput,
    EngravedScoreData,
    InputBundle,
    InputMetadata,
    PipelineConfig,
)
from backend.jobs.events import JobEvent
from backend.jobs.manager import JobManager


def _make_engraved_output() -> EngravedOutput:
    return EngravedOutput(
        metadata=EngravedScoreData(
            includes_dynamics=False,
            includes_pedal_marks=False,
            includes_fingering=False,
            includes_chord_symbols=False,
            title="t",
            composer="",
        ),
        pdf_uri="file:///fake.pdf",
        musicxml_uri="file:///fake.xml",
        humanized_midi_uri="file:///fake.mid",
    )


def _make_bundle() -> InputBundle:
    return InputBundle(
        metadata=InputMetadata(
            title="t",
            artist="a",
            source="title_lookup",
        ),
    )


def _make_config() -> PipelineConfig:
    return PipelineConfig(variant="full")


class _SuccessRunner:
    """Minimal fake PipelineRunner that returns a stub EngravedOutput."""

    async def run(self, *, job_id, bundle, config, on_event):  # noqa: ARG002
        return _make_engraved_output()


class _FailingRunner:
    """Fake runner that raises a recognizable exception."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def run(self, *, job_id, bundle, config, on_event):  # noqa: ARG002
        raise self._exc


async def _wait_for(record, *, timeout: float = 2.0) -> None:
    """Wait for a JobRecord's background task to finish."""
    assert record.task is not None
    await asyncio.wait_for(record.task, timeout=timeout)


@pytest.mark.asyncio
async def test_completed_jobs_are_capped(monkeypatch):
    """Submitting more than max_completed_jobs evicts the oldest records."""
    cap = 5
    monkeypatch.setattr(settings, "max_completed_jobs", cap)

    manager = JobManager(_SuccessRunner())  # type: ignore[arg-type]

    # Submit cap + 3 jobs and let each one complete before submitting the
    # next so completion order is deterministic. (asyncio.create_task starts
    # the task but only the awaited completion writes into _completed_jobs.)
    submitted_ids: list[str] = []
    for _ in range(cap + 3):
        record = await manager.submit(_make_bundle(), _make_config())
        await _wait_for(record)
        submitted_ids.append(record.job_id)

    # Only the most recent ``cap`` records should survive.
    assert len(manager._jobs) == cap
    surviving_ids = set(manager._jobs.keys())
    assert surviving_ids == set(submitted_ids[-cap:])
    # The three oldest must be gone.
    for evicted_id in submitted_ids[:3]:
        assert evicted_id not in manager._jobs
        assert manager.get(evicted_id) is None


@pytest.mark.asyncio
async def test_running_jobs_are_not_evicted(monkeypatch):
    """A still-running job past the cap must not be dropped from _jobs."""
    cap = 2
    monkeypatch.setattr(settings, "max_completed_jobs", cap)

    # Block the runner on an event we control so the job stays in
    # ``running`` state until we're ready to release it. ``started`` lets
    # the test wait until the runner has actually entered .run() before
    # swapping in the success runner — otherwise the long-running task
    # might pick up the post-swap runner and complete instantly.
    release = asyncio.Event()
    started = asyncio.Event()

    class _Blocking:
        async def run(self, *, job_id, bundle, config, on_event):  # noqa: ARG002
            started.set()
            await release.wait()
            return _make_engraved_output()

    manager = JobManager(_Blocking())  # type: ignore[arg-type]
    long_running = await manager.submit(_make_bundle(), _make_config())
    # Park here until the blocking runner is actually inside .run().
    await asyncio.wait_for(started.wait(), timeout=2.0)

    # Now drive cap + 1 successful jobs to completion against a separate
    # runner. We swap the runner attribute directly — same JobManager.
    manager._runner = _SuccessRunner()  # type: ignore[assignment]
    completed_ids: list[str] = []
    for _ in range(cap + 1):
        record = await manager.submit(_make_bundle(), _make_config())
        await _wait_for(record)
        completed_ids.append(record.job_id)

    # The long-running job is still tracked even though completed_jobs
    # already holds cap+1 entries — eviction only removes entries that
    # have actually completed.
    assert long_running.job_id in manager._jobs
    assert manager.get(long_running.job_id) is not None
    # The oldest completed record was evicted.
    assert completed_ids[0] not in manager._jobs

    # Cleanup: release the blocking runner so the task finishes.
    release.set()
    await _wait_for(long_running)


@pytest.mark.asyncio
async def test_failed_job_records_only_exception_class_name():
    """JobRecord.error must not leak repr(exc) — only the exception class name."""
    secret = "DB_PASSWORD=hunter2 at /etc/secrets.json"
    runner = _FailingRunner(RuntimeError(secret))
    manager = JobManager(runner)  # type: ignore[arg-type]

    record = await manager.submit(_make_bundle(), _make_config())
    await _wait_for(record)

    assert record.status == "failed"
    assert record.error == "RuntimeError"
    # The leaky bits must not appear anywhere in the surfaced error field.
    assert secret not in (record.error or "")
    assert "hunter2" not in (record.error or "")


@pytest.mark.asyncio
async def test_failed_jobs_count_against_eviction_cap(monkeypatch):
    """Failed jobs are terminal too — they participate in eviction."""
    cap = 3
    monkeypatch.setattr(settings, "max_completed_jobs", cap)

    manager = JobManager(_FailingRunner(RuntimeError("boom")))  # type: ignore[arg-type]

    submitted_ids: list[str] = []
    for _ in range(cap + 2):
        record = await manager.submit(_make_bundle(), _make_config())
        await _wait_for(record)
        submitted_ids.append(record.job_id)

    assert len(manager._jobs) == cap
    assert set(manager._jobs.keys()) == set(submitted_ids[-cap:])


@pytest.mark.asyncio
async def test_subscriber_queue_drains_after_eviction(monkeypatch):
    """A websocket subscriber holding a queue ref still drains after eviction.

    The eviction logic deliberately does not touch subscribers' queues —
    they're independent asyncio.Queue objects, so dropping the JobRecord
    only removes the lookup entry. Already-enqueued events stay readable.
    """
    cap = 1
    monkeypatch.setattr(settings, "max_completed_jobs", cap)

    manager = JobManager(_SuccessRunner())  # type: ignore[arg-type]

    # Submit one job, subscribe before completion, then wait for it to
    # finish. The subscriber's queue should hold the replay events.
    record = await manager.submit(_make_bundle(), _make_config())
    queue = await manager.subscribe(record.job_id)
    assert queue is not None
    await _wait_for(record)

    # Submit a second job to push the first past the cap.
    record2 = await manager.submit(_make_bundle(), _make_config())
    await _wait_for(record2)

    # The first JobRecord is evicted from the lookup map…
    assert manager.get(record.job_id) is None
    # …but the original subscriber's queue is still drainable.
    drained: list[JobEvent] = []
    while not queue.empty():
        drained.append(queue.get_nowait())
    types = {ev.type for ev in drained}
    assert "job_created" in types
    assert "job_succeeded" in types
