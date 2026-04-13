"""V6 — Engrave worker payload_type dispatch (CTR-05, D-07).

Directly invokes `backend.workers.engrave.run` with handcrafted envelopes
(HumanizedPerformance, PianoScore, RefinedPerformance, and an invalid tag)
to prove the dispatch chain handles all three valid payload_types and
rejects unknowns with a helpful error message.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from backend.contracts import (
    ExpressionMap,
    ExpressiveNote,
    HumanizedPerformance,
    PianoScore,
    QualitySignal,
    RefineCitation,
    RefineEditOp,
    RefinedPerformance,
    ScoreMetadata,
    ScoreNote,
    TempoMapEntry,
)
from backend.workers import engrave as engrave_mod
from shared.storage.local import LocalBlobStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_humanized_performance() -> HumanizedPerformance:
    """Smallest-valid HumanizedPerformance — one RH note."""
    score = PianoScore(
        right_hand=[
            ScoreNote(
                id="rh-0001",
                pitch=60,
                onset_beat=0.0,
                duration_beat=1.0,
                velocity=80,
                voice=1,
            )
        ],
        left_hand=[],
        metadata=ScoreMetadata(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
            difficulty="intermediate",
        ),
    )
    return HumanizedPerformance(
        expressive_notes=[
            ExpressiveNote(
                score_note_id="rh-0001",
                pitch=60,
                onset_beat=0.0,
                duration_beat=1.0,
                velocity=80,
                hand="rh",
                voice=1,
                timing_offset_ms=0.0,
                velocity_offset=0,
            )
        ],
        expression=ExpressionMap(),
        score=score,
        quality=QualitySignal(overall_confidence=0.9, warnings=[]),
    )


def _compute_digest(perf: HumanizedPerformance) -> str:
    canonical = json.dumps(
        perf.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _minimal_piano_score() -> PianoScore:
    """Smallest-valid PianoScore for sheet_only+enable_refine engrave tests.

    Mirrors the inner score used by _minimal_humanized_performance, but
    returns the standalone PianoScore — matches the RefinedPerformance
    widening in 01-09 (WR-02) where sheet_only's refine stage wraps a
    PianoScore rather than a HumanizedPerformance.
    """
    return PianoScore(
        right_hand=[
            ScoreNote(
                id="rh-0001",
                pitch=60,
                onset_beat=0.0,
                duration_beat=1.0,
                velocity=80,
                voice=1,
            )
        ],
        left_hand=[],
        metadata=ScoreMetadata(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
            difficulty="intermediate",
        ),
    )


def _write_envelope(blob: LocalBlobStore, job_id: str, payload_type: str, payload_data: dict) -> str:
    """Serialize an engrave envelope to blob storage and return its URI."""
    envelope = {
        "payload": payload_data,
        "payload_type": payload_type,
        "job_id": job_id,
        "title": "Test Song",
        "composer": "Tester",
    }
    return blob.put_json(f"jobs/{job_id}/engrave/input.json", envelope)


# ---------------------------------------------------------------------------
# V6 — RefinedPerformance unwrap
# ---------------------------------------------------------------------------


def test_engrave_unwraps_refined_performance_to_humanized(tmp_path: Path) -> None:
    """D-07 / V6: engrave accepts payload_type='RefinedPerformance', unwraps inner HumanizedPerformance."""
    # The autouse `isolated_blob_root` fixture has already pointed
    # settings.blob_root at a tmp dir; reuse it so engrave.run's
    # internally-constructed LocalBlobStore reads from the same root we wrote to.
    from backend.config import settings
    blob = LocalBlobStore(settings.blob_root)
    job_id = "test-refined-001"
    perf = _minimal_humanized_performance()
    refined = RefinedPerformance(
        refined_performance=perf,
        edits=[
            RefineEditOp(
                op="delete",
                target_note_id="rh-0001",
                rationale="duplicate_removal",
            )
        ],
        citations=[
            RefineCitation(url="https://example.com", snippet="x", confidence=0.9)
        ],
        model="claude-sonnet-4-6",
        source_performance_digest=_compute_digest(perf),
    )
    envelope_data = refined.model_dump(mode="json")

    payload_uri = _write_envelope(blob, job_id, "RefinedPerformance", envelope_data)

    output_uri = engrave_mod.run(job_id, payload_uri)

    # The output is EngravedOutput JSON — just confirm the URI is reachable
    # and the output has a schema_version (proves the unwrap-and-render path
    # completed, regardless of LilyPond stub vs real).
    output = blob.get_json(output_uri)
    assert isinstance(output, dict)
    assert output.get("schema_version"), output


# ---------------------------------------------------------------------------
# V6 — unknown payload_type error message
# ---------------------------------------------------------------------------


def test_engrave_rejects_unknown_payload_type_with_new_tag_list(tmp_path: Path) -> None:
    """CTR-05 / V6: unknown payload_type error names all three valid tags."""
    from backend.config import settings
    blob = LocalBlobStore(settings.blob_root)
    job_id = "test-bad-001"
    payload_uri = _write_envelope(blob, job_id, "FooBarBaz", {"arbitrary": "data"})

    with pytest.raises(ValueError) as exc_info:
        engrave_mod.run(job_id, payload_uri)

    msg = str(exc_info.value)
    assert "FooBarBaz" in msg, msg
    assert "HumanizedPerformance" in msg, msg
    assert "PianoScore" in msg, msg
    assert "RefinedPerformance" in msg, msg


# ---------------------------------------------------------------------------
# Regression guard — HumanizedPerformance still works
# ---------------------------------------------------------------------------


def test_engrave_still_accepts_humanized_performance(tmp_path: Path) -> None:
    """Regression: HumanizedPerformance branch untouched by the RefinedPerformance addition."""
    from backend.config import settings
    blob = LocalBlobStore(settings.blob_root)
    job_id = "test-humanized-001"
    perf = _minimal_humanized_performance()
    payload_uri = _write_envelope(blob, job_id, "HumanizedPerformance", perf.model_dump(mode="json"))

    output_uri = engrave_mod.run(job_id, payload_uri)

    output = blob.get_json(output_uri)
    assert isinstance(output, dict)
    assert output.get("schema_version"), output


# ---------------------------------------------------------------------------
# Gap closure 01-09 (WR-02) — engrave unwraps RefinedPerformance(PianoScore)
# ---------------------------------------------------------------------------
#
# Closes VERIFICATION gap 2 on the engrave-worker side. After widening
# RefinedPerformance.refined_performance to HumanizedPerformance | PianoScore,
# the engrave worker at backend/workers/engrave.py:34-41 needs no code
# change (EngraveService.run already handles both types via isinstance
# dispatch). This test proves that claim end-to-end: a RefinedPerformance
# envelope wrapping a PianoScore, handed to engrave.run, successfully
# unwraps to a PianoScore and renders an EngravedOutput.


def test_engrave_unwraps_refined_performance_to_piano_score(tmp_path: Path) -> None:
    """WR-02 gap closure: engrave unwraps RefinedPerformance(PianoScore) via the sheet_only path.

    The sheet_only+enable_refine=True execution plan emits
    ['ingest','transcribe','arrange','refine','engrave'] — refine receives
    a PianoScore upstream. This test hands engrave a RefinedPerformance
    envelope with a PianoScore inside and asserts the engrave worker
    successfully renders it. The underlying EngraveService.run accepts
    HumanizedPerformance | PianoScore, so no branching is needed in the
    worker beyond the existing `payload = refined.refined_performance` line.
    """
    # Re-use the same isolated_blob_root fixture as the existing tests.
    from backend.config import settings
    blob = LocalBlobStore(settings.blob_root)
    job_id = "test-refined-piano-001"

    score = _minimal_piano_score()
    # Digest computed over the PianoScore itself (the pre-refine payload on
    # the sheet_only path is the PianoScore from arrange).
    canonical = json.dumps(
        score.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()

    refined = RefinedPerformance(
        refined_performance=score,  # PianoScore — sheet_only refine upstream
        edits=[
            RefineEditOp(
                op="delete",
                target_note_id="rh-0001",
                rationale="duplicate_removal",
            )
        ],
        citations=[
            RefineCitation(url="https://example.com", snippet="x", confidence=0.9)
        ],
        model="claude-sonnet-4-6",
        source_performance_digest=digest,
    )
    envelope_data = refined.model_dump(mode="json")

    payload_uri = _write_envelope(blob, job_id, "RefinedPerformance", envelope_data)

    # Run the worker end-to-end — this exercises:
    #   1. RefinedPerformance.model_validate(payload_data) with the
    #      widened union (PianoScore branch selected structurally)
    #   2. payload = refined.refined_performance — returns PianoScore
    #   3. service.run(payload, ...) — existing isinstance dispatch in
    #      EngraveService handles the PianoScore branch
    output_uri = engrave_mod.run(job_id, payload_uri)

    # Output is EngravedOutput JSON — confirm the URI is reachable and
    # has a schema_version (proves the unwrap + render path completed).
    output = blob.get_json(output_uri)
    assert isinstance(output, dict)
    assert output.get("schema_version"), output


def test_engrave_refined_wrap_regression_humanized_path_unchanged(tmp_path: Path) -> None:
    """Regression: RefinedPerformance(HumanizedPerformance) still unwraps correctly after the widen.

    Paired with test_engrave_unwraps_refined_performance_to_humanized (which
    covers the same shape via a different test invocation). This variant
    exists specifically to guard against a Pydantic union-dispatch bug where
    a HumanizedPerformance payload could be mis-classified as a PianoScore
    after the widen. If this test fails, the union is ambiguous and the
    widen needs a discriminator tag.
    """
    from backend.config import settings
    blob = LocalBlobStore(settings.blob_root)
    job_id = "test-refined-humanized-regression-001"

    perf = _minimal_humanized_performance()
    refined = RefinedPerformance(
        refined_performance=perf,  # HumanizedPerformance — full/audio/midi variants
        edits=[],
        citations=[],
        model="claude-sonnet-4-6",
        source_performance_digest=_compute_digest(perf),
    )

    # Before handing to engrave, confirm the inner type stayed
    # HumanizedPerformance after model construction (Pydantic did not
    # silently promote it to PianoScore).
    assert isinstance(refined.refined_performance, HumanizedPerformance), (
        type(refined.refined_performance).__name__
    )

    payload_uri = _write_envelope(blob, job_id, "RefinedPerformance", refined.model_dump(mode="json"))
    output_uri = engrave_mod.run(job_id, payload_uri)

    output = blob.get_json(output_uri)
    assert isinstance(output, dict)
    assert output.get("schema_version"), output
