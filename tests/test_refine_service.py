"""Unit tests for backend/services/refine.py (STG-01 through STG-08; D-08, D-10)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import anthropic
import pytest
from shared.contracts import (
    ExpressionMap,
    ExpressiveNote,
    HumanizedPerformance,
    PianoScore,
    QualitySignal,
    RefineCitation,
    RefinedPerformance,
    RefineEditOp,
    ScoreMetadata,
    ScoreNote,
    TempoMapEntry,
)

from backend.services.refine import (
    _PRICE_PER_MTOK_USD,
    RefinedEditOpList,
    RefineLLMError,
    RefineService,
    RefineTraceRejectedEdit,
)
from backend.services.refine_prompt import REFINE_PROMPT_VERSION
from backend.services.refine_validate import RefineValidator
from tests.fakes.refine_client import FakeParseResponse, FakeRefineClient, FakeUsage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class _FakeSettings:
    refine_model: str = "claude-sonnet-4-6"
    refine_max_tokens: int = 4096
    refine_web_search_max_uses: int = 5
    refine_max_retries: int = 3
    refine_ghost_velocity_max: int = 40


def _piano_score(rh: list[tuple[int, float, int]] | None = None,
                 lh: list[tuple[int, float, int]] | None = None) -> PianoScore:
    rh = rh or [(60, 0.0, 80), (64, 1.0, 80)]
    lh = lh or [(48, 0.0, 80), (52, 1.0, 80)]
    return PianoScore(
        right_hand=[
            ScoreNote(id=f"rh-{i:04d}", pitch=p, onset_beat=o, duration_beat=0.5, velocity=v, voice=1)
            for i, (p, o, v) in enumerate(rh)
        ],
        left_hand=[
            ScoreNote(id=f"lh-{i:04d}", pitch=p, onset_beat=o, duration_beat=0.5, velocity=v, voice=1)
            for i, (p, o, v) in enumerate(lh)
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
            duration_beat=n.duration_beat, velocity=n.velocity,
            hand="rh" if n in score.right_hand else "lh",
            voice=n.voice, timing_offset_ms=0.0, velocity_offset=0,
        )
        for n in (*score.right_hand, *score.left_hand)
    ]
    return HumanizedPerformance(
        expressive_notes=exp_notes, expression=ExpressionMap(),
        score=score, quality=QualitySignal(overall_confidence=0.9, warnings=[]),
    )


def _make_service(
    *, responses: list[FakeParseResponse] | None = None,
    raises: Exception | None = None,
    raise_each_call: bool = False,
    settings: _FakeSettings | None = None,
) -> tuple[RefineService, FakeRefineClient]:
    settings = settings or _FakeSettings()
    fake = FakeRefineClient(responses=responses, raises=raises, raise_each_call=raise_each_call)
    svc = RefineService(client=fake, validator=RefineValidator(settings), settings=settings)
    return svc, fake


def _parsed_ok(
    edits: list[RefineEditOp] | None = None,
    citations: list[RefineCitation] | None = None,
) -> RefinedEditOpList:
    return RefinedEditOpList(edits=edits or [], citations=citations or [])


def _resp(
    edits: list[RefineEditOp] | None = None,
    citations: list[RefineCitation] | None = None,
    *, stop_reason: str = "end_turn",
    model: str = "claude-sonnet-4-6",
    usage: FakeUsage | None = None,
) -> FakeParseResponse:
    return FakeParseResponse(
        parsed=_parsed_ok(edits, citations),
        stop_reason=stop_reason,
        model=model,
        usage=usage or FakeUsage(input_tokens=100, output_tokens=50),
    )


# ---------------------------------------------------------------------------
# Happy path (STG-01, STG-02, STG-03, STG-06)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_full_pipeline() -> None:
    """Two valid edits -> RefinedPerformance with both applied; trace stamped."""
    perf = _humanized()
    edits = [
        RefineEditOp(op="delete", target_note_id="r-0000", rationale="ghost_note_removal"),
        RefineEditOp(op="modify", target_note_id="l-0000", rationale="velocity_cleanup", velocity=60),
    ]
    svc, fake = _make_service(responses=[_resp(edits=edits)])

    refined, trace = await svc.run(perf, metadata={"title": "x", "composer": "y"})

    assert isinstance(refined, RefinedPerformance)
    assert refined.schema_version == "3.1.0"
    assert len(refined.edits) == 2
    assert refined.model == "claude-sonnet-4-6"
    assert len(refined.source_performance_digest) == 64  # sha256 hex
    assert trace.prompt_version == REFINE_PROMPT_VERSION
    assert trace.prompt_version == "2026.04-v1"  # D-10 literal lock
    assert trace.stop_reason == "end_turn"
    assert len(trace.applied_edits) == 2
    assert trace.rejected_edits == []
    assert fake.call_count == 1


@pytest.mark.asyncio
async def test_piano_score_source_returns_piano_score_inner() -> None:
    """WR-02 invariant: PianoScore in -> PianoScore inner out (type preserved)."""
    ps = _piano_score()
    svc, _ = _make_service(responses=[_resp()])
    refined, _ = await svc.run(ps, metadata={})
    assert isinstance(refined.refined_performance, PianoScore)
    assert not isinstance(refined.refined_performance, HumanizedPerformance)


# ---------------------------------------------------------------------------
# STG-06: input never mutated
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_input_performance_is_not_mutated_by_apply_edits() -> None:
    """STG-06: deep-copy guarantee. Source.model_dump unchanged after run."""
    perf = _humanized()
    before = perf.model_dump()
    edits = [
        RefineEditOp(op="delete", target_note_id="r-0000", rationale="ghost_note_removal"),
        RefineEditOp(op="modify", target_note_id="l-0000", rationale="velocity_cleanup", velocity=10),
    ]
    svc, _ = _make_service(responses=[_resp(edits=edits)])
    await svc.run(perf, metadata={})
    after = perf.model_dump()
    assert before == after, "RefineService.run must not mutate its input (STG-06)"


@pytest.mark.asyncio
async def test_apply_edits_delete_removes_note() -> None:
    perf = _humanized()
    original_count = len(perf.expressive_notes)
    edits = [RefineEditOp(op="delete", target_note_id="r-0000", rationale="ghost_note_removal")]
    svc, _ = _make_service(responses=[_resp(edits=edits)])
    refined, _ = await svc.run(perf, metadata={})
    assert len(refined.refined_performance.expressive_notes) == original_count - 1


@pytest.mark.asyncio
async def test_apply_edits_modify_changes_pitch() -> None:
    perf = _humanized()
    edits = [RefineEditOp(op="modify", target_note_id="r-0000", rationale="octave_correction", pitch=72)]
    svc, _ = _make_service(responses=[_resp(edits=edits)])
    refined, _ = await svc.run(perf, metadata={})
    # The edit targets the lowest-onset, lowest-pitch right-hand note.
    # After sort by (onset, pitch): r-0000 is pitch 60 originally -> edited to 72.
    mutated = [n for n in refined.refined_performance.expressive_notes if n.pitch == 72]
    assert len(mutated) >= 1


@pytest.mark.asyncio
async def test_apply_edits_delete_then_modify_targets_original_notes() -> None:
    """WR-01 regression: when [delete r-0000, modify r-0001 pitch=72] arrive
    together, the modify must target the note the LLM saw as r-0001 (pitch 64,
    onset 1.0), NOT whichever note re-indexes to r-0001 after the delete.

    Before the WR-01 fix, ``_apply_edits`` rebuilt ``id_map`` after each delete,
    which shifted subsequent IDs. A modify that followed a delete would then
    silently misdirect to a different physical note than validation approved.
    """
    perf = _humanized()
    # Baseline RH notes (sorted by onset, pitch): r-0000 -> pitch 60 onset 0.0,
    # r-0001 -> pitch 64 onset 1.0. The modify targets r-0001 (pitch 64).
    edits = [
        RefineEditOp(op="delete", target_note_id="r-0000", rationale="ghost_note_removal"),
        RefineEditOp(op="modify", target_note_id="r-0001", rationale="octave_correction", pitch=76),
    ]
    svc, _ = _make_service(responses=[_resp(edits=edits)])
    refined, _ = await svc.run(perf, metadata={})

    rh_notes = [n for n in refined.refined_performance.expressive_notes if n.hand == "rh"]
    # After delete: one RH note remains. That note MUST be the one the LLM
    # knew as r-0001 (originally pitch 64 at onset 1.0), now modified to pitch 76.
    assert len(rh_notes) == 1
    assert rh_notes[0].onset_beat == 1.0
    assert rh_notes[0].pitch == 76


@pytest.mark.asyncio
async def test_apply_edits_timing_offset_translates_to_onset_beat_for_score_note() -> None:
    """WR-02 regression: when a refine edit carries timing_offset_ms and the
    source is a PianoScore (ScoreNote, which has no timing_offset_ms field),
    the service translates ms -> beat at 120 BPM nominal and adjusts
    onset_beat. Before the fix this was silently dropped.

    120 BPM = 2 beats/sec, so +50 ms = +0.1 beat. r-0001 has original
    onset_beat=1.0 -> expected 1.1 after the modify.
    """
    ps = _piano_score()
    edits = [
        RefineEditOp(
            op="modify", target_note_id="r-0001",
            rationale="timing_cleanup", timing_offset_ms=50.0,
        ),
    ]
    svc, _ = _make_service(responses=[_resp(edits=edits)])
    refined, _ = await svc.run(ps, metadata={})

    assert isinstance(refined.refined_performance, PianoScore)
    rh = refined.refined_performance.right_hand
    # r-0001 corresponds to the pitch-64/onset-1.0 RH note after sort.
    target = next(n for n in rh if n.pitch == 64)
    assert target.onset_beat == pytest.approx(1.1, rel=1e-6)


@pytest.mark.asyncio
async def test_apply_edits_timing_offset_floors_score_note_onset_at_zero() -> None:
    """WR-02: negative timing_offset_ms on a ScoreNote near beat 0 must not
    drive onset_beat below 0.0 (contract invariant)."""
    ps = _piano_score()  # r-0000 has onset 0.0 (pitch 60)
    edits = [
        RefineEditOp(
            op="modify", target_note_id="r-0000",
            rationale="timing_cleanup", timing_offset_ms=-50.0,
        ),
    ]
    svc, _ = _make_service(responses=[_resp(edits=edits)])
    refined, _ = await svc.run(ps, metadata={})

    assert isinstance(refined.refined_performance, PianoScore)
    rh = refined.refined_performance.right_hand
    target = next(n for n in rh if n.pitch == 60)
    assert target.onset_beat == 0.0  # floored, not -0.1


# ---------------------------------------------------------------------------
# STG-08: stop_reason != end_turn -> RefineLLMError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_stop", ["pause_turn", "max_tokens", "tool_use", "stop_sequence"])
async def test_non_end_turn_stop_reason_raises(bad_stop: str) -> None:
    """STG-08: any non-end_turn stop_reason is a failure; pause_turn never auto-resumed."""
    perf = _humanized()
    svc, _ = _make_service(responses=[_resp(stop_reason=bad_stop)])
    with pytest.raises(RefineLLMError) as exc_info:
        await svc.run(perf, metadata={})
    assert bad_stop in str(exc_info.value)


# ---------------------------------------------------------------------------
# STG-07: retry scope and policy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_succeeds_after_transient_rate_limit() -> None:
    """FakeRefineClient raises once, then second call returns valid response."""
    perf = _humanized()
    # FakeRefineClient with raises= fires BEFORE checking responses — but then
    # clears so the second call returns the queued response.
    responses = [_resp()]
    exc = anthropic.RateLimitError(
        message="rate limited", response=_http_response(429), body={"type": "error"},
    )
    svc, fake = _make_service(responses=responses, raises=exc, raise_each_call=False)
    refined, trace = await svc.run(perf, metadata={})
    assert refined is not None
    assert fake.call_count == 2  # retry happened


@pytest.mark.asyncio
async def test_retry_exhausts_reraises_rate_limit() -> None:
    """max_retries exhausted -> original exception propagates (tenacity reraise=True)."""
    perf = _humanized()
    exc = anthropic.RateLimitError(
        message="rate limited forever", response=_http_response(429), body={"type": "error"},
    )
    svc, fake = _make_service(
        raises=exc, raise_each_call=True,
        settings=_FakeSettings(refine_max_retries=2),
    )
    with pytest.raises(anthropic.RateLimitError):
        await svc.run(perf, metadata={})
    assert fake.call_count == 2  # exactly refine_max_retries attempts


@pytest.mark.asyncio
async def test_bad_request_is_not_retried() -> None:
    """anthropic.BadRequestError is NOT in retry set -> fails on first attempt."""
    perf = _humanized()
    exc = anthropic.BadRequestError(
        message="bad request", response=_http_response(400), body={"type": "error"},
    )
    svc, fake = _make_service(raises=exc, raise_each_call=True)
    with pytest.raises(anthropic.BadRequestError):
        await svc.run(perf, metadata={})
    assert fake.call_count == 1


@pytest.mark.asyncio
async def test_authentication_error_is_not_retried() -> None:
    perf = _humanized()
    exc = anthropic.AuthenticationError(
        message="auth failed", response=_http_response(401), body={"type": "error"},
    )
    svc, fake = _make_service(raises=exc, raise_each_call=True)
    with pytest.raises(anthropic.AuthenticationError):
        await svc.run(perf, metadata={})
    assert fake.call_count == 1


# ---------------------------------------------------------------------------
# Validator integration: rejections populate trace, do NOT raise
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validator_rejections_populate_trace_but_dont_raise() -> None:
    """run() does not raise on per-edit rejections; trace captures them."""
    perf = _humanized()
    edits = [
        RefineEditOp(op="delete", target_note_id="r-0000", rationale="ghost_note_removal"),   # OK
        RefineEditOp(op="delete", target_note_id="r-9999", rationale="ghost_note_removal"),   # unknown id
        # out of range (pitch=9 < PIANO_LOW_MIDI=21)
        RefineEditOp(op="modify", target_note_id="l-0000", rationale="octave_correction", pitch=9),
    ]
    svc, _ = _make_service(responses=[_resp(edits=edits)])
    refined, trace = await svc.run(perf, metadata={})
    assert len(refined.edits) == 1  # only the valid delete
    assert len(trace.rejected_edits) == 2
    reasons = {r.reason for r in trace.rejected_edits}
    assert "unknown_target_note_id" in reasons
    assert "pitch_out_of_range" in reasons


@pytest.mark.asyncio
async def test_trace_includes_rejected_edits_with_reasons() -> None:
    perf = _humanized()
    edits = [RefineEditOp(op="delete", target_note_id="r-9999", rationale="other")]
    svc, _ = _make_service(responses=[_resp(edits=edits)])
    refined, trace = await svc.run(perf, metadata={})
    assert len(trace.rejected_edits) == 1
    assert isinstance(trace.rejected_edits[0], RefineTraceRejectedEdit)
    assert trace.rejected_edits[0].reason == "unknown_target_note_id"
    assert trace.rejected_edits[0].edit.target_note_id == "r-9999"


# ---------------------------------------------------------------------------
# Digest determinism
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_digest_is_deterministic_for_identical_input() -> None:
    perf_a = _humanized()
    perf_b = _humanized()  # structurally identical construction
    svc_a, _ = _make_service(responses=[_resp()])
    svc_b, _ = _make_service(responses=[_resp()])
    refined_a, _ = await svc_a.run(perf_a, metadata={})
    refined_b, _ = await svc_b.run(perf_b, metadata={})
    assert refined_a.source_performance_digest == refined_b.source_performance_digest


@pytest.mark.asyncio
async def test_digest_differs_for_different_inputs() -> None:
    perf_a = _humanized()
    perf_b = HumanizedPerformance(
        expressive_notes=perf_a.expressive_notes[:-1],  # remove one note
        expression=perf_a.expression, score=perf_a.score, quality=perf_a.quality,
    )
    svc_a, _ = _make_service(responses=[_resp()])
    svc_b, _ = _make_service(responses=[_resp()])
    refined_a, _ = await svc_a.run(perf_a, metadata={})
    refined_b, _ = await svc_b.run(perf_b, metadata={})
    assert refined_a.source_performance_digest != refined_b.source_performance_digest


# ---------------------------------------------------------------------------
# SDK call shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_bundle_passed_to_sdk() -> None:
    perf = _humanized()
    svc, fake = _make_service(responses=[_resp()])
    await svc.run(perf, metadata={"title": "Debut", "composer": "Sera"})
    call = fake.calls[0]
    assert call["model"] == "claude-sonnet-4-6"
    assert call["max_tokens"] == 4096
    assert "MUST NOT add" in call["system"]  # SYSTEM_PROMPT content sanity
    assert "Debut" in call["messages"][0]["content"]
    assert "Sera" in call["messages"][0]["content"]
    assert isinstance(call["tools"], list) and len(call["tools"]) == 1
    assert call["tools"][0]["type"].startswith("web_search_")


# ---------------------------------------------------------------------------
# Cost estimate (STG-10)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cost_estimate_uses_hardcoded_price_table() -> None:
    perf = _humanized()
    usage = FakeUsage(input_tokens=1000, output_tokens=500)
    svc, _ = _make_service(responses=[_resp(usage=usage, model="claude-sonnet-4-6")])
    _, trace = await svc.run(perf, metadata={})
    # 1000/1e6 * $3 + 500/1e6 * $15 = 0.003 + 0.0075 = 0.0105
    assert trace.estimated_cost_usd == pytest.approx(0.0105, rel=1e-6)
    assert trace.usage.input_tokens == 1000
    assert trace.usage.output_tokens == 500


@pytest.mark.asyncio
async def test_unknown_model_yields_zero_cost_and_warns(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # backend.main._configure_app_logging() sets backend.propagate=False during
    # lifespan startup; other tests that run the FastAPI app leak that flag
    # into subsequent tests, which prevents caplog (root-attached) from seeing
    # backend.services.refine records. Restore propagate for this test — the
    # monkeypatch teardown rolls it back so production config is preserved.
    monkeypatch.setattr(logging.getLogger("backend"), "propagate", True)

    perf = _humanized()
    svc, _ = _make_service(responses=[_resp(model="claude-mystery-5")])
    with caplog.at_level(logging.WARNING, logger="backend.services.refine"):
        _, trace = await svc.run(perf, metadata={})
    assert trace.estimated_cost_usd == 0.0
    assert any("no pricing" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Price-table sanity (protects against accidental bump)
# ---------------------------------------------------------------------------


def test_price_table_has_sonnet_and_opus() -> None:
    assert "claude-sonnet-4-6" in _PRICE_PER_MTOK_USD
    assert "claude-opus-4-6" in _PRICE_PER_MTOK_USD


# ---------------------------------------------------------------------------
# Helpers (construct anthropic exceptions without making HTTP calls)
# ---------------------------------------------------------------------------


def _http_response(status_code: int) -> Any:
    """Minimal stand-in for httpx.Response — anthropic errors accept a response= kw."""
    class _R:
        def __init__(self, status: int) -> None:
            self.status_code = status
            self.headers = {}
            self.request = None
    return _R(status_code)
