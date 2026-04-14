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
    RemoteMidiFile,
)
from shared.storage.local import LocalBlobStore

from backend.config import settings
from backend.jobs.events import JobEvent
from backend.jobs.runner import PipelineRunner
from backend.workers.celery_app import celery_app


@pytest.fixture
def blob():
    """Return a LocalBlobStore rooted at the isolated_blob_root path."""
    return LocalBlobStore(settings.blob_root)


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
    config = PipelineConfig(variant="audio_upload", skip_humanizer=False)

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


@pytest.mark.asyncio
async def test_transcription_midi_uri_survives_pipeline(runner):
    """transcription_midi_uri set by the decomposer must appear on the final EngravedOutput."""
    bundle = InputBundle(
        audio=RemoteAudioFile(
            uri="file:///fake/audio.wav",
            format="wav",
            sample_rate=44100,
            duration_sec=10.0,
            channels=1,
        ),
        metadata=InputMetadata(title="MIDI URI Test", artist="Tester", source="audio_upload"),
    )
    config = PipelineConfig(variant="audio_upload")

    result = await runner.run(
        job_id="test-midi-uri-001",
        bundle=bundle,
        config=config,
    )

    assert result.transcription_midi_uri is not None
    assert "basic-pitch.mid" in result.transcription_midi_uri


@pytest.mark.asyncio
async def test_midi_upload_pipeline_via_celery(runner):
    """A midi_upload job should skip transcribe and use _bundle_to_transcription fallback."""
    events: list[JobEvent] = []

    bundle = InputBundle(
        midi=RemoteMidiFile(
            uri="file:///fake/input.mid",
            ticks_per_beat=480,
        ),
        metadata=InputMetadata(title="MIDI Test", artist="Tester", source="midi_upload"),
    )
    config = PipelineConfig(variant="midi_upload", skip_humanizer=False)

    result = await runner.run(
        job_id="test-celery-midi-001",
        bundle=bundle,
        config=config,
        on_event=events.append,
    )

    assert result.pdf_uri
    assert result.musicxml_uri
    assert result.humanized_midi_uri

    stage_names = [e.stage for e in events if e.type == "stage_completed"]
    assert stage_names == ["ingest", "arrange", "humanize", "engrave"]


@pytest.mark.asyncio
async def test_sheet_only_pipeline_via_celery(runner):
    """A sheet_only job should skip humanize; engrave receives PianoScore directly."""
    events: list[JobEvent] = []

    bundle = InputBundle(
        audio=RemoteAudioFile(
            uri="file:///fake/audio.wav",
            format="wav",
            sample_rate=44100,
            duration_sec=10.0,
            channels=1,
        ),
        metadata=InputMetadata(title="Sheet Test", artist="Tester", source="audio_upload"),
    )
    config = PipelineConfig(variant="sheet_only")

    result = await runner.run(
        job_id="test-celery-sheet-001",
        bundle=bundle,
        config=config,
        on_event=events.append,
    )

    assert result.pdf_uri
    assert result.musicxml_uri

    stage_names = [e.stage for e in events if e.type == "stage_completed"]
    assert stage_names == ["ingest", "transcribe", "arrange", "engrave"]
