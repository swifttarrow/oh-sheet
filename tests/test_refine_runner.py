"""Integration test: PipelineRunner dispatches refine and passes refined envelope to engrave."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from shared.contracts import (
    InputBundle,
    InputMetadata,
    PipelineConfig,
)
from shared.storage.local import LocalBlobStore

from backend.config import settings
from backend.jobs.runner import PipelineRunner
from backend.services import ml_engraver_client
from backend.workers.celery_app import celery_app

_FAKE_MUSICXML = (
    b'<?xml version="1.0" encoding="UTF-8" standalone="no"?>'
    b'<score-partwise version="3.1"><part id="P1"/></score-partwise>'
)


@pytest.fixture(autouse=True)
def mock_ml_engraver(monkeypatch):
    async def fake_engrave(midi_bytes: bytes) -> bytes:
        return _FAKE_MUSICXML

    monkeypatch.setattr(ml_engraver_client, "engrave_midi_via_ml_service", fake_engrave)


@pytest.mark.asyncio
async def test_runner_invokes_refine_before_engrave():
    """With enable_refine=True, the runner dispatches refine before engrave.

    Engrave no longer flows through Celery — it's an inline ML HTTP call —
    so we assert refine.run was dispatched and that the pipeline produced
    a MusicXML artifact after refine ran.
    """
    stages_dispatched: list[str] = []

    original_dispatch = PipelineRunner._dispatch_task

    async def _spy_dispatch(self, task_name, job_id, payload_uri, timeout):
        stages_dispatched.append(task_name)
        return await original_dispatch(self, task_name, job_id, payload_uri, timeout)

    with patch.object(PipelineRunner, "_dispatch_task", _spy_dispatch):
        blob = LocalBlobStore(settings.blob_root)
        runner = PipelineRunner(blob_store=blob, celery_app=celery_app)
        bundle = InputBundle(
            metadata=InputMetadata(title="test", source="audio_upload"),
        )
        config = PipelineConfig(variant="audio_upload", enable_refine=True)
        result = await runner.run(job_id="t-refine", bundle=bundle, config=config)

    assert "refine.run" in stages_dispatched
    # engrave is no longer dispatched as a Celery task — it runs inline.
    assert "engrave.run" not in stages_dispatched
    assert result.musicxml_uri


@pytest.mark.asyncio
async def test_runner_skips_refine_when_disabled():
    """enable_refine=False omits refine from dispatched tasks."""
    stages_dispatched: list[str] = []

    original_dispatch = PipelineRunner._dispatch_task

    async def _spy_dispatch(self, task_name, job_id, payload_uri, timeout):
        stages_dispatched.append(task_name)
        return await original_dispatch(self, task_name, job_id, payload_uri, timeout)

    with patch.object(PipelineRunner, "_dispatch_task", _spy_dispatch):
        blob = LocalBlobStore(settings.blob_root)
        runner = PipelineRunner(blob_store=blob, celery_app=celery_app)
        bundle = InputBundle(
            metadata=InputMetadata(title="test", source="audio_upload"),
        )
        config = PipelineConfig(variant="audio_upload", enable_refine=False)
        await runner.run(job_id="t-no-refine", bundle=bundle, config=config)

    assert "refine.run" not in stages_dispatched
