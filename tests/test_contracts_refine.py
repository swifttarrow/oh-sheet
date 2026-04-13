"""V1-V4 contract unit tests for CTR-01..CTR-05.

Scope: import, round-trip, closed-enum rejection, modify-field validator,
deterministic digest, citation bounds. No file I/O, no HTTP.
"""
from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from backend.contracts import (
    SCHEMA_VERSION,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_humanized_performance() -> HumanizedPerformance:
    """Smallest-valid HumanizedPerformance for round-trip tests."""
    score = PianoScore(
        schema_version=SCHEMA_VERSION,
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
        schema_version=SCHEMA_VERSION,
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
    """Claude's Discretion digest algorithm: sha256 of canonical sorted-key JSON."""
    canonical = json.dumps(
        perf.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


# ---------------------------------------------------------------------------
# V1 — round-trip carries 3.1.0
# ---------------------------------------------------------------------------


def test_refined_performance_roundtrip_carries_3_1_0() -> None:
    """CTR-05: RefinedPerformance round-trips through model_dump/model_validate carrying schema_version='3.1.0'."""
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
            RefineCitation(url="https://example.com", snippet="x", confidence=0.8)
        ],
        model="claude-sonnet-4-6",
        source_performance_digest=_compute_digest(perf),
    )
    assert refined.schema_version == "3.1.0", refined.schema_version

    dumped = refined.model_dump(mode="json")
    assert dumped["schema_version"] == "3.1.0", dumped["schema_version"]

    reloaded = RefinedPerformance.model_validate(dumped)
    assert reloaded.schema_version == "3.1.0"
    assert reloaded.model == "claude-sonnet-4-6"
    assert len(reloaded.edits) == 1
    assert reloaded.edits[0].op == "delete"


# ---------------------------------------------------------------------------
# V2 — closed rationale enum
# ---------------------------------------------------------------------------


def test_rationale_rejects_unknown_value() -> None:
    """CTR-01: RefineEditOp.rationale is a closed Literal; unknown values raise."""
    with pytest.raises(ValidationError):
        RefineEditOp(
            op="delete",
            target_note_id="rh-0001",
            rationale="totally_made_up_rationale",  # not in the 9-value set
        )


# ---------------------------------------------------------------------------
# V3 — modify requires at least one field
# ---------------------------------------------------------------------------


def test_modify_requires_at_least_one_field() -> None:
    """CTR-01: op='modify' with no modify-payload fields raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        RefineEditOp(
            op="modify",
            target_note_id="rh-0001",
            rationale="harmony_correction",
        )
    # ValueError message surfaces via ValidationError.errors()[0]['msg']
    msg = str(exc_info.value)
    assert "at least one of" in msg, msg


def test_modify_accepts_single_field() -> None:
    """op='modify' with just pitch=60 is valid (no "all fields required" mistake)."""
    op = RefineEditOp(
        op="modify",
        target_note_id="rh-0001",
        rationale="octave_correction",
        pitch=72,
    )
    assert op.pitch == 72
    assert op.velocity is None


def test_refine_edit_op_delete_accepts_only_target_note_id() -> None:
    """op='delete' needs no modify fields — target_note_id + rationale is enough."""
    op = RefineEditOp(
        op="delete",
        target_note_id="rh-0001",
        rationale="duplicate_removal",
    )
    assert op.op == "delete"
    assert op.pitch is None


# ---------------------------------------------------------------------------
# V4 — digest determinism
# ---------------------------------------------------------------------------


def test_source_performance_digest_is_deterministic_across_roundtrip() -> None:
    """D-06: SHA-256 of canonical JSON must be stable across model round-trips."""
    perf = _minimal_humanized_performance()
    digest_1 = _compute_digest(perf)

    # Round-trip the model and recompute
    perf_reloaded = HumanizedPerformance.model_validate(perf.model_dump(mode="json"))
    digest_2 = _compute_digest(perf_reloaded)

    assert digest_1 == digest_2, (digest_1, digest_2)
    assert len(digest_1) == 64, len(digest_1)
    assert all(c in "0123456789abcdef" for c in digest_1), digest_1


# ---------------------------------------------------------------------------
# V4 (shape) — citation confidence bounds
# ---------------------------------------------------------------------------


def test_refine_citation_confidence_bounds() -> None:
    """RefineCitation.confidence is [0.0, 1.0] — mirrors QualitySignal convention."""
    ok = RefineCitation(url="x", snippet="y", confidence=0.5)
    assert ok.confidence == 0.5

    with pytest.raises(ValidationError):
        RefineCitation(url="x", snippet="y", confidence=1.5)

    with pytest.raises(ValidationError):
        RefineCitation(url="x", snippet="y", confidence=-0.1)
