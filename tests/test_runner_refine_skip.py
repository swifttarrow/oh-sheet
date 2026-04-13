"""End-to-end PipelineRunner tests for the refine stage (INT-01, INT-03, INT-04, INT-05).

These tests exercise the runner with a STUBBED refine.run Celery task
(registered via celery_eager_mode fixture). They do NOT exercise the real
backend.workers.refine.run — that is Plan 04's concern. Here we verify
that the RUNNER wires refine into the execution plan correctly, handles
exceptions per INT-03, emits events in the correct order per INT-05,
and leaves the job succeeded per INT-04.
"""
from __future__ import annotations

import logging
from typing import Any

import pytest
from shared.contracts import (
    ExpressionMap,
    ExpressiveNote,
    HumanizedPerformance,
    PianoScore,
    QualitySignal,
    RefinedPerformance,
    ScoreMetadata,
    ScoreNote,
    TempoMapEntry,
)
from shared.storage.local import LocalBlobStore

from backend.config import settings
from backend.contracts import (
    SCHEMA_VERSION,
    InputBundle,
    InputMetadata,
    PipelineConfig,
    RemoteAudioFile,
)
from backend.jobs.events import JobEvent
from backend.jobs.runner import PipelineRunner
from backend.workers.celery_app import celery_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _audio_bundle(blob: LocalBlobStore) -> InputBundle:
    """Build a bundle for the audio_upload variant using a stub audio URI."""
    audio_uri = blob.put_bytes("jobs/test/uploads/audio/test.mp3", b"\x00" * 64)
    return InputBundle(
        schema_version=SCHEMA_VERSION,
        audio=RemoteAudioFile(
            uri=audio_uri, format="mp3", sample_rate=44100,
            duration_sec=1.0, channels=2,
        ),
        midi=None,
        metadata=InputMetadata(title="Test Song", artist="Tester", source="audio_upload"),
    )


def _midi_bundle(blob: LocalBlobStore) -> InputBundle:
    """Build a bundle for the sheet_only variant.

    sheet_only is constructed directly on PipelineConfig and does not route
    through the JobCreateRequest.variant inference; we still need an audio
    bundle in the InputBundle because the runner's ingest stage validates
    on either audio or midi being set. The skip_real_transcription conftest
    fixture stubs the transcribe path so fake audio bytes are fine.
    """
    audio_uri = blob.put_bytes("jobs/test/uploads/audio/test.mp3", b"\x00" * 64)
    return InputBundle(
        schema_version=SCHEMA_VERSION,
        audio=RemoteAudioFile(
            uri=audio_uri, format="mp3", sample_rate=44100,
            duration_sec=1.0, channels=2,
        ),
        midi=None,
        metadata=InputMetadata(title="Test Sheet", artist="Tester", source="audio_upload"),
    )


def _canned_refined_output_envelope(
    *,
    inner_type: str = "HumanizedPerformance",
) -> dict:
    """A minimal but valid RefinedPerformance envelope the stub writes."""
    if inner_type == "HumanizedPerformance":
        score = PianoScore(
            right_hand=[ScoreNote(id="rh-0000", pitch=60, onset_beat=0.0,
                                   duration_beat=0.5, velocity=80, voice=1)],
            left_hand=[ScoreNote(id="lh-0000", pitch=48, onset_beat=0.0,
                                  duration_beat=0.5, velocity=80, voice=1)],
            metadata=ScoreMetadata(
                key="C:major", time_signature=(4, 4),
                tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
                difficulty="intermediate",
            ),
        )
        inner: HumanizedPerformance | PianoScore = HumanizedPerformance(
            expressive_notes=[
                ExpressiveNote(
                    score_note_id="rh-0000", pitch=60, onset_beat=0.0,
                    duration_beat=0.5, velocity=80, hand="rh", voice=1,
                    timing_offset_ms=0.0, velocity_offset=0,
                ),
            ],
            expression=ExpressionMap(), score=score,
            quality=QualitySignal(overall_confidence=0.9, warnings=[]),
        )
    else:  # PianoScore
        inner = PianoScore(
            right_hand=[ScoreNote(id="rh-0000", pitch=60, onset_beat=0.0,
                                   duration_beat=0.5, velocity=80, voice=1)],
            left_hand=[],
            metadata=ScoreMetadata(
                key="C:major", time_signature=(4, 4),
                tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
                difficulty="intermediate",
            ),
        )
    refined = RefinedPerformance(
        refined_performance=inner, edits=[], citations=[],
        model="claude-sonnet-4-6", source_performance_digest="0" * 64,
    )
    return {
        "payload_type": "RefinedPerformance",
        "payload": refined.model_dump(mode="json"),
    }


def _install_refine_stub(
    blob: LocalBlobStore,
    monkeypatch: pytest.MonkeyPatch,
    *,
    raises: Exception | None = None,
    inner_type: str = "HumanizedPerformance",
) -> dict[str, Any]:
    """Register a stub refine.run task + wrap blob.put_json to spy on engrave input.

    Returns a probe dict with refine-invocation counters and the engrave input
    envelope (captured via a put_json spy — NOT by shadowing engrave.run, which
    would race with Celery's task-registration dispatch).

    Strategy (W5 fix — replaces the former engrave.run-shadowing approach):
      * Stub refine.run via celery_app.task registration.
      * DO NOT shadow engrave.run — capturing the existing Celery Task object
        and re-registering a wrapper creates order-dependent routing behavior
        and potential dispatch loops.
      * Spy on blob.put_json. The runner's engrave branch calls
        `blob_store.put_json(f"jobs/{job_id}/engrave/input.json", envelope)`
        BEFORE dispatching engrave.run — capturing this write asserts the
        payload_type the runner hands downstream, while leaving the real
        engrave.run dispatch intact so the pipeline completes naturally.
    """
    probe: dict[str, Any] = {
        "refine_calls": 0,
        "last_refine_input": None,
        "last_engrave_input": None,
    }

    def _refine_stub(job_id: str, payload_uri: str) -> str:
        probe["refine_calls"] += 1
        probe["last_refine_input"] = blob.get_json(payload_uri)
        if raises is not None:
            raise raises
        return blob.put_json(
            f"jobs/{job_id}/refine/output.json",
            _canned_refined_output_envelope(inner_type=inner_type),
        )

    # Stub refine.run. W5+fix: Celery's @task(name=X) returns the ALREADY-
    # registered task when a task by that name already exists — it does NOT
    # replace. This would make a stub registered in test N stick around for
    # tests N+1, N+2, ... closing over a stale blob reference. Workaround:
    # pop the current registration FIRST, then let @task register fresh.
    # Every test that touches refine.run goes through _install_refine_stub,
    # so the pop-then-register pattern yields a fresh closure per test.
    celery_app.tasks.pop("refine.run", None)
    celery_app.task(name="refine.run", bind=False)(_refine_stub)

    # Spy on blob.put_json to capture the engrave input envelope. Key is
    # canonical: "jobs/{job_id}/engrave/input.json" (see backend/jobs/runner.py
    # PipelineRunner._serialize_stage_input convention). Any other stage also
    # calls put_json, so we filter on the key suffix.
    original_put_json = blob.put_json

    def _put_json_spy(key: str, payload: dict) -> str:
        if "engrave/input" in key or key.endswith("engrave/input.json"):
            # Copy so later mutation doesn't leak into the probe snapshot.
            probe["last_engrave_input"] = dict(payload)
        return original_put_json(key, payload)

    monkeypatch.setattr(blob, "put_json", _put_json_spy)
    return probe


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def blob() -> LocalBlobStore:
    return LocalBlobStore(settings.blob_root)


@pytest.fixture
def runner(blob: LocalBlobStore) -> PipelineRunner:
    return PipelineRunner(blob_store=blob, celery_app=celery_app)


# ---------------------------------------------------------------------------
# INT-01: Happy path wires refine between humanize and engrave
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_happy_path_with_refine_succeeds(
    blob: LocalBlobStore, runner: PipelineRunner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _install_refine_stub(blob, monkeypatch, inner_type="HumanizedPerformance")
    events: list[JobEvent] = []
    bundle = _audio_bundle(blob)

    config = PipelineConfig(variant="audio_upload", enable_refine=True)
    result = await runner.run(
        job_id="test-refine-happy", bundle=bundle, config=config,
        on_event=events.append,
    )

    # Result is an EngravedOutput
    assert result.pdf_uri is not None

    # INT-01: refine was dispatched
    assert probe["refine_calls"] == 1

    # INT-02: engrave received RefinedPerformance envelope
    assert probe["last_engrave_input"] is not None
    assert probe["last_engrave_input"]["payload_type"] == "RefinedPerformance"

    # No skip message in any event
    for e in events:
        if e.message:
            assert not e.message.startswith("refine_skipped:"), (
                f"unexpected skip: {e.message}"
            )

    # INT-05: event order humanize → refine → engrave (stage_started markers)
    stage_starts = [e.stage for e in events if e.type == "stage_started"]
    assert stage_starts.index("humanize") < stage_starts.index("refine") < stage_starts.index("engrave")


# ---------------------------------------------------------------------------
# INT-03 / INT-04: Skip-on-failure semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_refine_exception_emits_refine_skipped(
    blob: LocalBlobStore, runner: PipelineRunner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INT-03: RuntimeError from refine → stage_completed with skip message, NOT stage_failed."""
    probe = _install_refine_stub(blob, monkeypatch, raises=RuntimeError("boom"))
    events: list[JobEvent] = []
    bundle = _audio_bundle(blob)

    config = PipelineConfig(variant="audio_upload", enable_refine=True)
    result = await runner.run(
        job_id="test-refine-skip-runtime", bundle=bundle, config=config,
        on_event=events.append,
    )

    # INT-04: job still produces an EngravedOutput
    assert result.pdf_uri is not None

    # INT-03: stage_completed with skip message present
    refine_completions = [
        e for e in events
        if e.stage == "refine" and e.type == "stage_completed"
    ]
    assert len(refine_completions) == 1
    msg = refine_completions[0].message or ""
    assert msg.startswith("refine_skipped:"), f"expected skip message, got {msg!r}"
    assert "runtimeerror" in msg.lower()

    # INT-03 explicit: NO stage_failed for refine
    refine_failures = [
        e for e in events if e.stage == "refine" and e.type == "stage_failed"
    ]
    assert refine_failures == [], f"refine must NEVER emit stage_failed: {refine_failures}"

    # INT-02: engrave ran against unrefined payload (HumanizedPerformance, not RefinedPerformance)
    assert probe["last_engrave_input"]["payload_type"] == "HumanizedPerformance"


@pytest.mark.asyncio
async def test_runner_refine_skip_logs_counter(
    blob: LocalBlobStore, runner: PipelineRunner, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """INT-03: refine_skip_total counter log with reason."""
    _install_refine_stub(blob, monkeypatch, raises=ValueError("invalid thing"))
    # Ensure backend.* logs propagate to caplog (existing helper pattern — see
    # _enable_backend_log_capture in tests/test_jobs_refine.py).
    backend_logger = logging.getLogger("backend")
    monkeypatch.setattr(backend_logger, "propagate", True)

    bundle = _audio_bundle(blob)
    config = PipelineConfig(variant="audio_upload", enable_refine=True)
    with caplog.at_level(logging.INFO, logger="backend.jobs.runner"):
        await runner.run(
            job_id="test-refine-counter", bundle=bundle, config=config,
            on_event=lambda e: None,
        )

    counter_records = [
        r for r in caplog.records
        if "refine_skip_total" in r.getMessage()
    ]
    assert len(counter_records) == 1, f"expected exactly 1 counter log, got {len(counter_records)}"
    msg = counter_records[0].getMessage()
    assert "reason=valueerror" in msg
    assert "job_id=test-refine-counter" in msg


@pytest.mark.asyncio
async def test_runner_refine_skip_preserves_stage_order(
    blob: LocalBlobStore, runner: PipelineRunner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INT-05: stage ordering humanize → refine → engrave under skip."""
    _install_refine_stub(blob, monkeypatch, raises=RuntimeError("boom"))
    events: list[JobEvent] = []
    bundle = _audio_bundle(blob)

    config = PipelineConfig(variant="audio_upload", enable_refine=True)
    await runner.run(
        job_id="test-refine-order", bundle=bundle, config=config,
        on_event=events.append,
    )

    # Canonical order of stage markers (started + completed)
    markers = [(e.stage, e.type) for e in events if e.type in ("stage_started", "stage_completed")]
    # Filter to stages relevant here
    relevant = [m for m in markers if m[0] in ("humanize", "refine", "engrave")]
    # Expected markers in strict order: humanize_started, humanize_completed,
    # refine_started, refine_completed, engrave_started, engrave_completed.
    expected = [
        ("humanize", "stage_started"), ("humanize", "stage_completed"),
        ("refine", "stage_started"), ("refine", "stage_completed"),
        ("engrave", "stage_started"), ("engrave", "stage_completed"),
    ]
    assert relevant == expected, f"stage order violation: {relevant}"


# ---------------------------------------------------------------------------
# WR-02 / INT-02 invariant: PianoScore passthrough on sheet_only skip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_refine_happy_path_sheet_only_variant(
    blob: LocalBlobStore, runner: PipelineRunner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """sheet_only + enable_refine=True → plan is [ingest, transcribe, arrange, refine, engrave]."""
    probe = _install_refine_stub(blob, monkeypatch, inner_type="PianoScore")
    bundle = _midi_bundle(blob)

    config = PipelineConfig(variant="sheet_only", enable_refine=True)
    result = await runner.run(
        job_id="test-sheet-only-refine", bundle=bundle, config=config,
        on_event=lambda e: None,
    )
    assert result.pdf_uri is not None
    assert probe["refine_calls"] == 1
    # Refine input was a PianoScore envelope (no humanize upstream)
    assert probe["last_refine_input"]["payload_type"] == "PianoScore"
    # Engrave input is RefinedPerformance envelope
    assert probe["last_engrave_input"]["payload_type"] == "RefinedPerformance"


@pytest.mark.asyncio
async def test_runner_refine_skipped_passes_unrefined_score_to_engrave_sheet_only(
    blob: LocalBlobStore, runner: PipelineRunner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip on sheet_only → engrave runs on PianoScore, not RefinedPerformance."""
    probe = _install_refine_stub(blob, monkeypatch, raises=RuntimeError("sheet-only-refine-down"))
    bundle = _midi_bundle(blob)

    config = PipelineConfig(variant="sheet_only", enable_refine=True)
    result = await runner.run(
        job_id="test-sheet-only-skip", bundle=bundle, config=config,
        on_event=lambda e: None,
    )
    assert result.pdf_uri is not None
    # Unrefined PianoScore flows to engrave
    assert probe["last_engrave_input"]["payload_type"] == "PianoScore"
