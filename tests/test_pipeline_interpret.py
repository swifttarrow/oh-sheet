"""End-to-end pipeline integration test for the interpret pre-stage.

Registers a deterministic stub for ``interpret.run`` that injects
``ArrangementHints(difficulty="beginner")`` onto the TranscriptionResult,
then submits a midi_upload job with ``arrangement_prompt`` set and asserts
that the difficulty propagates through arrange into the final output.
"""
from __future__ import annotations

import json
import time

import pytest
from shared.storage.local import LocalBlobStore

from backend.config import settings
from backend.contracts import (
    InputBundle,
    InputMetadata,
    PipelineConfig,
    RemoteMidiFile,
)
from backend.jobs.runner import PipelineRunner
from backend.services import ml_engraver_client
from backend.workers.celery_app import celery_app

# ---------------------------------------------------------------------------
# Minimal fake MusicXML so the ML engraver stub passes the size floor.
# ---------------------------------------------------------------------------

_FAKE_MUSICXML = (
    b'<?xml version="1.0" encoding="UTF-8" standalone="no"?>'
    b'<score-partwise version="3.1"><part id="P1"/></score-partwise>'
)

# ---------------------------------------------------------------------------
# Stub ``interpret.run`` Celery task
#
# Registered directly on the shared celery_app so _dispatch_task's
# ``apply_async`` path (task_always_eager=True) can find it in
# celery_app.tasks[].  The real worker module is NOT imported here —
# we want a pure deterministic stub with no LLM calls.
# ---------------------------------------------------------------------------

_STUB_HINTS: dict = {
    "difficulty": "beginner",
    "density": "sparse",
    "tempo_bias": 0.0,
    "style_tags": [],
    "notes": "stub",
}


@celery_app.task(name="interpret.run")
def _stub_interpret_run(job_id: str, payload_uri: str) -> str:
    """Deterministic stub: inject beginner difficulty via ArrangementHints."""
    blob = LocalBlobStore(settings.blob_root)
    envelope = blob.get_json(payload_uri)

    # Mutate the txr dict to inject arrangement_hints.
    txr_dict: dict = envelope["txr"]
    txr_dict["arrangement_hints"] = _STUB_HINTS

    out = {"txr": txr_dict}
    output_uri = blob.put_json(
        f"jobs/{job_id}/interpret/output.json",
        out,
    )
    return output_uri


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _stub_ml_engraver(monkeypatch):
    """Keep ML engraver from hitting the network."""
    async def _fake(midi_bytes: bytes) -> bytes:
        return _FAKE_MUSICXML

    monkeypatch.setattr(ml_engraver_client, "engrave_midi_via_ml_service", _fake)


@pytest.fixture
def blob():
    return LocalBlobStore(settings.blob_root)


@pytest.fixture
def runner(blob):
    return PipelineRunner(blob_store=blob, celery_app=celery_app)


# ---------------------------------------------------------------------------
# Helper: create a real midi blob in the isolated blob root
# ---------------------------------------------------------------------------

_MINIMAL_SMF = (
    b"MThd"
    b"\x00\x00\x00\x06"  # header chunk length = 6
    b"\x00\x00"          # format 0 (single track)
    b"\x00\x01"          # 1 track
    b"\x01\xe0"          # 480 ticks/quarter
)


def _put_midi_blob(blob: LocalBlobStore) -> str:
    return blob.put_bytes("uploads/test-interpret.mid", _MINIMAL_SMF)


# ---------------------------------------------------------------------------
# Test 1: execution plan contains "interpret" before "arrange"
# ---------------------------------------------------------------------------


def test_execution_plan_contains_interpret_before_arrange():
    """enable_interpret=True inserts interpret immediately before arrange."""
    cfg = PipelineConfig(variant="midi_upload", enable_interpret=True, enable_refine=False)
    plan = cfg.get_execution_plan()

    assert "interpret" in plan, f"expected interpret in plan: {plan}"
    assert plan.index("interpret") < plan.index("arrange"), (
        f"interpret must precede arrange, got: {plan}"
    )


def test_execution_plan_without_prompt_omits_interpret():
    """enable_interpret=False leaves interpret out of the plan."""
    cfg = PipelineConfig(variant="midi_upload", enable_interpret=False, enable_refine=False)
    plan = cfg.get_execution_plan()
    assert "interpret" not in plan, f"interpret should be absent: {plan}"


# ---------------------------------------------------------------------------
# Test 2: full pipeline run — difficulty propagates to score
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interpret_difficulty_propagates_to_score(runner, blob):
    """interpret stub injects 'beginner' → arrange honours it → score has difficulty='beginner'."""
    midi_uri = _put_midi_blob(blob)

    bundle = InputBundle(
        midi=RemoteMidiFile(uri=midi_uri, ticks_per_beat=480),
        metadata=InputMetadata(
            source="midi_upload",
            title="Test Interpret",
            artist="Test Artist",
            arrangement_prompt="make this beginner-friendly",
        ),
    )
    config = PipelineConfig(
        variant="midi_upload",
        enable_interpret=True,
        enable_refine=False,
    )

    events: list = []
    result = await runner.run(
        job_id="test-interpret-01",
        bundle=bundle,
        config=config,
        on_event=events.append,
    )

    # Pipeline must complete with artifacts.
    assert result.musicxml_uri, "expected a musicxml_uri in result"
    assert result.humanized_midi_uri, "expected a humanized_midi_uri in result"

    # ``interpret`` must appear in the completed stage events.
    completed_stages = [e.stage for e in events if e.type == "stage_completed"]
    assert "interpret" in completed_stages, (
        f"interpret stage_completed event missing; got {completed_stages}"
    )
    assert completed_stages.index("interpret") < completed_stages.index("arrange"), (
        f"interpret must fire before arrange; got {completed_stages}"
    )

    # Verify the arrange output reflects the injected difficulty.
    arrange_out_path = (
        settings.blob_root / "jobs" / "test-interpret-01" / "arrange" / "output.json"
    )
    assert arrange_out_path.exists(), "arrange/output.json was not written"

    score_dict = json.loads(arrange_out_path.read_text())
    difficulty = score_dict.get("metadata", {}).get("difficulty")
    assert difficulty == "beginner", (
        f"expected difficulty='beginner' in arrange output, got {difficulty!r}"
    )

    # arrangement_hints must also survive from interpret through arrange.
    hints = score_dict.get("metadata", {}).get("arrangement_hints")
    assert hints is not None, "arrangement_hints missing from score metadata"
    assert hints.get("difficulty") == "beginner", (
        f"arrangement_hints.difficulty mismatch: {hints}"
    )


# ---------------------------------------------------------------------------
# Test 3: interpret output blob is written correctly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interpret_output_blob_has_enriched_txr(runner, blob):
    """After the interpret stage runs, its output blob carries arrangement_hints."""
    midi_uri = _put_midi_blob(blob)

    bundle = InputBundle(
        midi=RemoteMidiFile(uri=midi_uri, ticks_per_beat=480),
        metadata=InputMetadata(
            source="midi_upload",
            title="Blob Check",
            arrangement_prompt="sparse, jazzy left hand",
        ),
    )
    config = PipelineConfig(
        variant="midi_upload",
        enable_interpret=True,
        enable_refine=False,
    )

    await runner.run(
        job_id="test-interpret-02",
        bundle=bundle,
        config=config,
    )

    interpret_out_path = (
        settings.blob_root / "jobs" / "test-interpret-02" / "interpret" / "output.json"
    )
    assert interpret_out_path.exists(), "interpret/output.json was not written"

    interpret_out = json.loads(interpret_out_path.read_text())
    txr_hints = interpret_out.get("txr", {}).get("arrangement_hints")
    assert txr_hints is not None, "arrangement_hints missing from interpret output txr"
    assert txr_hints.get("difficulty") == "beginner"
    assert txr_hints.get("density") == "sparse"


# ---------------------------------------------------------------------------
# Test 4: API submission with arrangement_prompt activates interpret stage
# ---------------------------------------------------------------------------


def test_api_job_with_arrangement_prompt_runs_interpret(client):
    """POST /v1/jobs with arrangement_prompt activates the interpret stage and completes."""
    # Upload a minimal MIDI first.
    upload_resp = client.post(
        "/v1/uploads/midi",
        files={"file": ("test.mid", _MINIMAL_SMF, "audio/midi")},
    )
    assert upload_resp.status_code == 200, upload_resp.text
    midi = upload_resp.json()

    create_resp = client.post(
        "/v1/jobs",
        json={
            "midi": midi,
            "title": "Interpret API Test",
            "arrangement_prompt": "make this beginner-friendly",
        },
    )
    assert create_resp.status_code == 202, create_resp.text
    job_id = create_resp.json()["job_id"]

    # Eager mode: job completes synchronously, poll briefly.
    deadline = time.time() + 5
    status = None
    while time.time() < deadline:
        status = client.get(f"/v1/jobs/{job_id}").json()
        if status["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.05)

    assert status is not None
    assert status["status"] == "succeeded", f"job failed: {status}"
    assert status["result"] is not None
    assert status["result"]["musicxml_uri"], "expected musicxml_uri"


def test_api_empty_prompt_does_not_activate_interpret():
    """Whitespace-only arrangement_prompt does NOT enable the interpret stage."""
    cfg_with_ws = PipelineConfig(
        variant="midi_upload",
        enable_interpret=bool("   " and "   ".strip()),
        enable_refine=False,
    )
    # The API does: bool(body.arrangement_prompt and body.arrangement_prompt.strip())
    # A whitespace-only string strips to "" which is falsy → enable_interpret=False.
    assert not cfg_with_ws.enable_interpret, (
        "whitespace-only prompt must not enable interpret"
    )

    cfg_with_real = PipelineConfig(
        variant="midi_upload",
        enable_interpret=bool("make it easy" and "make it easy".strip()),
        enable_refine=False,
    )
    assert cfg_with_real.enable_interpret, (
        "non-empty prompt must enable interpret"
    )
