"""Test the refactored PipelineRunner with Celery dispatch.

Uses Celery's eager mode (task_always_eager=True) so tasks execute
in-process synchronously — no Redis needed for unit tests.
"""

import pytest
from shared.contracts import (
    InputBundle,
    InputMetadata,
    PipelineConfig,
    RemoteAudioFile,
)
from shared.storage.local import LocalBlobStore

from backend.config import settings
from backend.jobs.events import JobEvent
from backend.jobs.runner import PipelineRunner
from backend.workers.celery_app import celery_app


@pytest.fixture
def blob(tmp_path):
    root = tmp_path / "blob"
    root.mkdir(exist_ok=True)
    settings.blob_root = root
    return LocalBlobStore(root)


@pytest.fixture
def runner(blob):
    return PipelineRunner(blob_store=blob, celery_app=celery_app)


@pytest.mark.asyncio
async def test_full_pipeline_via_celery(runner):
    """An audio_upload job should traverse all 5 stages via Celery tasks."""
    events: list[JobEvent] = []

    bundle = InputBundle(
        audio=RemoteAudioFile(
            uri="file:///fake/audio.wav",
            format="wav",
            sample_rate=44100,
            duration_sec=10.0,
            channels=1,
        ),
        metadata=InputMetadata(title="Test", artist="Tester", source="audio_upload"),
    )
    config = PipelineConfig(variant="audio_upload")

    result = await runner.run(
        job_id="test-celery-001",
        bundle=bundle,
        config=config,
        on_event=events.append,
    )

    assert result.pdf_uri
    assert result.musicxml_uri
    assert result.humanized_midi_uri

    stage_names = [e.stage for e in events if e.type == "stage_completed"]
    assert stage_names == ["ingest", "transcribe", "arrange", "humanize", "engrave"]
