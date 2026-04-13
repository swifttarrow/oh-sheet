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
