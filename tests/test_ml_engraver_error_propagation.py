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
    when the remote HTTP backend is the only configured engrave route."""
    monkeypatch.setattr(settings, "engrave_backend", "remote_http")

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
    """Same contract for midi_upload under the remote_http backend."""
    monkeypatch.setattr(settings, "engrave_backend", "remote_http")

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


@pytest.mark.asyncio
async def test_title_lookup_falls_through_to_local_engrave_when_tunechat_disabled(
    runner, monkeypatch,
):
    """When TuneChat is disabled, title_lookup jobs fall through to the
    local Oh Sheet pipeline (transcribe → arrange → humanize → engrave)
    and produce a result rather than crashing. A lower-quality local
    result is better than a user-facing error.
    """
    monkeypatch.setattr(settings, "tunechat_enabled", False)

    bundle = InputBundle(
        audio=RemoteAudioFile(
            uri="file:///fake/audio.wav",
            format="wav",
            sample_rate=44100,
            duration_sec=10.0,
            channels=1,
        ),
        metadata=InputMetadata(
            title="Title Lookup Song",
            artist="Tester",
            source="title_lookup",
        ),
    )
    config = PipelineConfig(variant="audio_upload", enable_refine=False)

    result = await runner.run(
        job_id="title-lookup-tunechat-disabled",
        bundle=bundle,
        config=config,
    )
    assert result is not None
    assert result.musicxml_uri or result.tunechat_job_id


@pytest.mark.asyncio
async def test_title_lookup_falls_through_to_local_engrave_when_tunechat_returns_none(
    runner, monkeypatch,
):
    """When TuneChat is enabled but returns None (service down, quota
    exceeded, etc.), title_lookup jobs fall through to the local Oh Sheet
    pipeline and produce a result instead of crashing.
    """
    monkeypatch.setattr(settings, "tunechat_enabled", True)

    from backend.services import tunechat_client  # noqa: PLC0415

    async def _tunechat_returns_none(*args, **kwargs):
        return None

    monkeypatch.setattr(
        tunechat_client, "transcribe_via_tunechat", _tunechat_returns_none,
    )

    bundle = InputBundle(
        audio=RemoteAudioFile(
            uri="file:///fake/audio.wav",
            format="wav",
            sample_rate=44100,
            duration_sec=10.0,
            channels=1,
        ),
        metadata=InputMetadata(
            title="Title Lookup Song",
            artist="Tester",
            source="title_lookup",
        ),
    )
    config = PipelineConfig(variant="audio_upload", enable_refine=False)

    result = await runner.run(
        job_id="title-lookup-tunechat-none",
        bundle=bundle,
        config=config,
    )
    assert result is not None
    assert result.musicxml_uri or result.tunechat_job_id


@pytest.mark.asyncio
async def test_title_lookup_tunechat_fallback_emits_no_phantom_events(
    runner, monkeypatch,
):
    """Regression: when TuneChat returns None, the runner used to fire a
    synthetic ``transcribe stage_started`` (and a duplicate ``ingest
    stage_completed``) before the fallback, leaving subscribers with
    a transcribe-started event that never had a matching completion.
    On the fallback path, only the regular per-step loop events should
    fire; synthetic events stay confined to the success branch.
    """
    monkeypatch.setattr(settings, "tunechat_enabled", True)

    from backend.services import tunechat_client  # noqa: PLC0415

    async def _tunechat_returns_none(*args, **kwargs):
        return None

    monkeypatch.setattr(
        tunechat_client, "transcribe_via_tunechat", _tunechat_returns_none,
    )

    events: list[tuple[str, str]] = []

    def on_event(ev) -> None:
        events.append((ev.stage or "", ev.type))

    bundle = InputBundle(
        audio=RemoteAudioFile(
            uri="file:///fake/audio.wav",
            format="wav",
            sample_rate=44100,
            duration_sec=10.0,
            channels=1,
        ),
        metadata=InputMetadata(
            title="Title Lookup Song",
            artist="Tester",
            source="title_lookup",
        ),
    )
    config = PipelineConfig(variant="audio_upload", enable_refine=False)

    await runner.run(
        job_id="title-lookup-tunechat-fallback-events",
        bundle=bundle,
        config=config,
        on_event=on_event,
    )

    started = [s for s, t in events if t == "stage_started"]
    completed = [s for s, t in events if t == "stage_completed"]

    # Every stage that started must also complete on the fallback path.
    assert sorted(started) == sorted(completed), (
        f"unbalanced stage events on fallback: started={started}, "
        f"completed={completed}"
    )
    # No duplicate stage_completed for ingest — used to fire twice
    # (once inside the TuneChat branch at progress=0.25 and again from
    # the outer loop after the branch returned).
    ingest_completed = [s for s, t in events if t == "stage_completed" and s == "ingest"]
    assert len(ingest_completed) == 1, (
        f"ingest stage_completed fired {len(ingest_completed)} times: {events}"
    )
