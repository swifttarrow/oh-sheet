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
from backend.workers.celery_app import celery_app


@pytest.mark.asyncio
async def test_runner_invokes_refine_before_engrave():
    """With enable_refine=True, the runner dispatches refine before engrave
    and the refined envelope is what engrave receives."""
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
    # Ordering: refine dispatched strictly before engrave.
    assert stages_dispatched.index("refine.run") < stages_dispatched.index("engrave.run")
    assert result.pdf_uri  # engrave still produced an output


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
