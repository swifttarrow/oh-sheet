"""Regression guard: MLEngraverError must propagate as a job failure.

The ml-pipeline engrave HTTP call is the only engraving path — there is
no local fallback. An outage of the ML service MUST surface as a failed
job rather than being swallowed; otherwise users would see a "succeeded"
job with no usable artifact and operators would miss the outage signal.

This invariant used to be covered by a test in
``test_engraver_inference_toggle.py``. When the toggle was removed
along with the fallback path, the invariant became *more* important —
this test documents it as a standalone contract.
"""
from __future__ import annotations

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
from backend.jobs.runner import PipelineRunner
from backend.services import ml_engraver_client
from backend.workers.celery_app import celery_app


@pytest.fixture
def runner():
    return PipelineRunner(
        blob_store=LocalBlobStore(settings.blob_root),
        celery_app=celery_app,
    )


@pytest.mark.asyncio
async def test_ml_engraver_error_fails_audio_upload_job(runner, monkeypatch):
    """An MLEngraverError during audio_upload engrave must propagate
    rather than degrade to a local fallback (there is no fallback)."""
    async def raising(midi_bytes: bytes) -> bytes:
        raise ml_engraver_client.MLEngraverError("simulated outage")

    monkeypatch.setattr(ml_engraver_client, "engrave_midi_via_ml_service", raising)

    bundle = InputBundle(
        audio=RemoteAudioFile(
            uri="file:///fake/audio.wav",
            format="wav",
            sample_rate=44100,
            duration_sec=10.0,
            channels=1,
        ),
        metadata=InputMetadata(title="Err Test", artist="Tester", source="audio_upload"),
    )
    config = PipelineConfig(variant="audio_upload", enable_refine=False)

    with pytest.raises(ml_engraver_client.MLEngraverError, match="simulated outage"):
        await runner.run(
            job_id="ml-err-audio-001",
            bundle=bundle,
            config=config,
        )


@pytest.mark.asyncio
async def test_ml_engraver_error_fails_midi_upload_job(runner, monkeypatch):
    """Same contract for midi_upload — no fallback, error propagates."""
    async def raising(midi_bytes: bytes) -> bytes:
        raise ml_engraver_client.MLEngraverError("simulated 503")

    monkeypatch.setattr(ml_engraver_client, "engrave_midi_via_ml_service", raising)

    bundle = InputBundle(
        midi=RemoteMidiFile(uri="file:///fake/input.mid", ticks_per_beat=480),
        metadata=InputMetadata(title="Err Test MIDI", artist="Tester", source="midi_upload"),
    )
    config = PipelineConfig(variant="midi_upload", enable_refine=False)

    with pytest.raises(ml_engraver_client.MLEngraverError, match="simulated 503"):
        await runner.run(
            job_id="ml-err-midi-001",
            bundle=bundle,
            config=config,
        )
