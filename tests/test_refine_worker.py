"""Unit tests for backend/workers/refine.py (STG-09, STG-10)."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr
from shared.contracts import (
    ExpressionMap,
    ExpressiveNote,
    HumanizedPerformance,
    PianoScore,
    QualitySignal,
    RefinedPerformance,
    RefineEditOp,
    ScoreMetadata,
    ScoreNote,
    TempoMapEntry,
)
from shared.storage.local import LocalBlobStore

from backend.config import settings
from backend.services.refine import RefinedEditOpList
from tests.fakes.refine_client import FakeParseResponse, FakeRefineClient, FakeUsage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def blob() -> LocalBlobStore:
    return LocalBlobStore(settings.blob_root)


def _key_to_uri(blob: LocalBlobStore, key: str) -> str:
    """Build a file:// URI for a blob key — LocalBlobStore.get_json requires a URI."""
    return (blob.root / key.lstrip("/")).as_uri()


def _piano_score() -> PianoScore:
    return PianoScore(
        right_hand=[
            ScoreNote(id="rh-0000", pitch=60, onset_beat=0.0, duration_beat=0.5, velocity=80, voice=1),
            ScoreNote(id="rh-0001", pitch=64, onset_beat=1.0, duration_beat=0.5, velocity=80, voice=1),
        ],
        left_hand=[
            ScoreNote(id="lh-0000", pitch=48, onset_beat=0.0, duration_beat=0.5, velocity=80, voice=1),
        ],
        metadata=ScoreMetadata(
            key="C:major", time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
            difficulty="intermediate",
        ),
    )


def _humanized() -> HumanizedPerformance:
    score = _piano_score()
    exp_notes = [
        ExpressiveNote(
            score_note_id=n.id, pitch=n.pitch, onset_beat=n.onset_beat,
            duration_beat=n.duration_beat, velocity=n.velocity, hand="rh",
            voice=n.voice, timing_offset_ms=0.0, velocity_offset=0,
        ) for n in score.right_hand
    ] + [
        ExpressiveNote(
            score_note_id=n.id, pitch=n.pitch, onset_beat=n.onset_beat,
            duration_beat=n.duration_beat, velocity=n.velocity, hand="lh",
            voice=n.voice, timing_offset_ms=0.0, velocity_offset=0,
        ) for n in score.left_hand
    ]
    return HumanizedPerformance(
        expressive_notes=exp_notes, expression=ExpressionMap(),
        score=score, quality=QualitySignal(overall_confidence=0.9, warnings=[]),
    )


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    responses: list[FakeParseResponse] | None = None,
    raises: Exception | None = None,
    raise_each_call: bool = False,
) -> dict[str, Any]:
    """Monkeypatch anthropic.AsyncAnthropic inside the worker module.

    Returns a container dict that tests can inspect to see the constructed
    fake instance (the factory stores the most recent instance under 'instance').
    """
    container: dict[str, Any] = {"instance": None, "constructed_with": None}
    default_responses = responses if responses is not None else [
        FakeParseResponse(
            parsed=RefinedEditOpList(
                edits=[RefineEditOp(op="delete", target_note_id="r-0000",
                                    rationale="ghost_note_removal")],
                citations=[],
            ),
            stop_reason="end_turn",
            model="claude-sonnet-4-6",
            usage=FakeUsage(input_tokens=100, output_tokens=50),
        ),
    ]

    def _factory(**kwargs: Any) -> FakeRefineClient:
        container["constructed_with"] = kwargs
        instance = FakeRefineClient(
            responses=list(default_responses),
            raises=raises,
            raise_each_call=raise_each_call,
        )
        container["instance"] = instance
        return instance

    monkeypatch.setattr("backend.workers.refine.anthropic.AsyncAnthropic", _factory)
    return container


def _write_envelope(
    blob: LocalBlobStore,
    job_id: str,
    *,
    payload_type: str,
    payload_dict: dict,
    title: str = "Test Song",
    composer: str = "Test Composer",
) -> str:
    return blob.put_json(
        f"jobs/{job_id}/refine/input.json",
        {
            "payload_type": payload_type,
            "payload": payload_dict,
            "job_id": job_id,
            "title": title,
            "composer": composer,
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_refine_worker_happy_path_humanized(
    blob: LocalBlobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HumanizedPerformance envelope -> RefinedPerformance output + trace."""
    monkeypatch.setattr(settings, "anthropic_api_key", SecretStr("sk-ant-api03-dummy-key"))
    container = _install_fake_client(monkeypatch)

    from backend.workers.refine import run as refine_run

    job_id = "test-happy-humanized"
    perf = _humanized()
    input_uri = _write_envelope(
        blob, job_id,
        payload_type="HumanizedPerformance",
        payload_dict=perf.model_dump(mode="json"),
    )

    output_uri = refine_run(job_id, input_uri)

    # Output URI shape
    assert f"jobs/{job_id}/refine/output.json" in output_uri
    out = blob.get_json(output_uri)
    assert out["payload_type"] == "RefinedPerformance"
    refined = RefinedPerformance.model_validate(out["payload"])
    assert refined.schema_version == "3.1.0"
    assert len(refined.edits) == 1
    # Trace artifact
    trace = blob.get_json(_key_to_uri(blob, f"jobs/{job_id}/refine/llm_trace.json"))
    assert trace["prompt_version"] == "2026.04-v1"  # D-10 lock
    # Client was constructed with api_key argument
    assert "api_key" in container["constructed_with"]
    assert container["constructed_with"]["api_key"] == "sk-ant-api03-dummy-key"


def test_refine_worker_happy_path_piano_score(
    blob: LocalBlobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PianoScore envelope (sheet_only variant) -> RefinedPerformance with PianoScore inner."""
    monkeypatch.setattr(settings, "anthropic_api_key", SecretStr("sk-ant-api03-dummy-key"))
    _install_fake_client(monkeypatch)

    from backend.workers.refine import run as refine_run

    job_id = "test-happy-piano"
    ps = _piano_score()
    input_uri = _write_envelope(
        blob, job_id,
        payload_type="PianoScore",
        payload_dict=ps.model_dump(mode="json"),
    )

    output_uri = refine_run(job_id, input_uri)
    out = blob.get_json(output_uri)
    refined = RefinedPerformance.model_validate(out["payload"])
    # Inner is a PianoScore (WR-02)
    assert isinstance(refined.refined_performance, PianoScore)


def test_refine_worker_unknown_payload_type_raises(
    blob: LocalBlobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "anthropic_api_key", SecretStr("sk-ant-api03-dummy-key"))
    _install_fake_client(monkeypatch)

    from backend.workers.refine import run as refine_run

    job_id = "test-bad-type"
    input_uri = _write_envelope(
        blob, job_id,
        payload_type="Gobbledygook",
        payload_dict={},
    )
    with pytest.raises(ValueError, match="Gobbledygook"):
        refine_run(job_id, input_uri)


def test_refine_worker_missing_api_key_raises(
    blob: LocalBlobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defense-in-depth: worker fails cleanly if CFG-04 400-gate was bypassed."""
    monkeypatch.setattr(settings, "anthropic_api_key", None)
    _install_fake_client(monkeypatch)

    from backend.workers.refine import run as refine_run

    job_id = "test-no-key"
    perf = _humanized()
    input_uri = _write_envelope(
        blob, job_id,
        payload_type="HumanizedPerformance",
        payload_dict=perf.model_dump(mode="json"),
    )

    with pytest.raises(RuntimeError, match="OHSHEET_ANTHROPIC_API_KEY"):
        refine_run(job_id, input_uri)


def test_refine_worker_calls_get_secret_value_exactly_once(
    blob: LocalBlobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CFG-03 / STG-03: single-site secret-access lock."""
    real_secret = SecretStr("sk-ant-api03-dummy-key")
    # Wrap the SecretStr with a MagicMock that wraps .get_secret_value
    call_counter = {"count": 0}
    original_get = real_secret.get_secret_value

    def _counted_get() -> str:
        call_counter["count"] += 1
        return original_get()

    wrapped = MagicMock(spec=SecretStr)
    wrapped.get_secret_value.side_effect = _counted_get

    monkeypatch.setattr(settings, "anthropic_api_key", wrapped)
    _install_fake_client(monkeypatch)

    from backend.workers.refine import run as refine_run

    job_id = "test-secret-once"
    perf = _humanized()
    input_uri = _write_envelope(
        blob, job_id,
        payload_type="HumanizedPerformance",
        payload_dict=perf.model_dump(mode="json"),
    )
    refine_run(job_id, input_uri)
    assert call_counter["count"] == 1, (
        f"CFG-03 single-site rule: .get_secret_value() must be called exactly once "
        f"per worker invocation; got {call_counter['count']}"
    )


def test_refine_worker_writes_both_blob_artifacts(
    blob: LocalBlobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "anthropic_api_key", SecretStr("sk-ant-api03-dummy-key"))
    _install_fake_client(monkeypatch)

    from backend.workers.refine import run as refine_run

    job_id = "test-dual-write"
    perf = _humanized()
    input_uri = _write_envelope(
        blob, job_id,
        payload_type="HumanizedPerformance",
        payload_dict=perf.model_dump(mode="json"),
    )
    refine_run(job_id, input_uri)

    # Both artifacts exist
    output_uri = _key_to_uri(blob, f"jobs/{job_id}/refine/output.json")
    trace_uri = _key_to_uri(blob, f"jobs/{job_id}/refine/llm_trace.json")
    assert blob.exists(output_uri)
    assert blob.exists(trace_uri)
    trace = blob.get_json(trace_uri)
    assert trace is not None
    # Trace has all required STG-10 fields
    assert trace["prompt_version"] == "2026.04-v1"
    assert "prompt_system" in trace
    assert "prompt_user" in trace
    assert "model" in trace
    assert "stop_reason" in trace
    assert "applied_edits" in trace
    assert "rejected_edits" in trace
    assert "citations" in trace
    assert "usage" in trace
    assert "estimated_cost_usd" in trace


def test_refine_worker_exception_propagates(
    blob: LocalBlobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Worker does NOT catch service exceptions — Plan 05 runner handles INT-03."""
    import anthropic
    monkeypatch.setattr(settings, "anthropic_api_key", SecretStr("sk-ant-api03-dummy-key"))
    exc = anthropic.RateLimitError(
        message="rate limited forever",
        response=_http_response(429),
        body={"type": "error"},
    )
    _install_fake_client(monkeypatch, raises=exc, raise_each_call=True)

    from backend.workers.refine import run as refine_run

    job_id = "test-exc-propagate"
    perf = _humanized()
    input_uri = _write_envelope(
        blob, job_id,
        payload_type="HumanizedPerformance",
        payload_dict=perf.model_dump(mode="json"),
    )
    with pytest.raises(anthropic.RateLimitError):
        refine_run(job_id, input_uri)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _http_response(status_code: int) -> Any:
    class _R:
        def __init__(self, s: int) -> None:
            self.status_code = s
            self.headers = {}
            self.request = None
    return _R(status_code)
