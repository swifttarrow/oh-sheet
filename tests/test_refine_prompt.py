"""Unit tests for backend/services/refine_prompt.py (D-09, D-10, D-12)."""
from __future__ import annotations

import pytest
from shared.contracts import (
    ExpressionMap,
    ExpressiveNote,
    HumanizedPerformance,
    PianoScore,
    QualitySignal,
    ScoreMetadata,
    ScoreNote,
    TempoMapEntry,
)

from backend.services.refine_prompt import (
    ID_PATTERN,
    REFINE_PROMPT_VERSION,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    _derive_note_id_map,
    build_prompt,
)


def _piano_score(
    *,
    rh: list[tuple[int, float]] | None = None,
    lh: list[tuple[int, float]] | None = None,
) -> PianoScore:
    rh = rh or [(60, 0.0), (64, 1.0)]
    lh = lh or [(48, 0.0), (52, 1.0)]
    return PianoScore(
        right_hand=[
            ScoreNote(id=f"rh-{i:04d}", pitch=p, onset_beat=o, duration_beat=0.5, velocity=80, voice=1)
            for i, (p, o) in enumerate(rh)
        ],
        left_hand=[
            ScoreNote(id=f"lh-{i:04d}", pitch=p, onset_beat=o, duration_beat=0.5, velocity=80, voice=1)
            for i, (p, o) in enumerate(lh)
        ],
        metadata=ScoreMetadata(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
            difficulty="intermediate",
        ),
    )


def _humanized(score: PianoScore | None = None) -> HumanizedPerformance:
    score = score or _piano_score()
    return HumanizedPerformance(
        expressive_notes=[
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
        ],
        expression=ExpressionMap(),
        score=score,
        quality=QualitySignal(overall_confidence=0.9, warnings=[]),
    )


def test_refine_prompt_version_is_pinned() -> None:
    """D-10: version constant exactly '2026.04-v1'."""
    assert REFINE_PROMPT_VERSION == "2026.04-v1"


def test_system_prompt_covers_authority_rules() -> None:
    """STG-02/STG-04 prompt-layer enforcement terms present."""
    for term in ("modify", "delete", "MUST NOT add", "target_note_id", "web_search"):
        assert term in SYSTEM_PROMPT, f"SYSTEM_PROMPT missing {term!r}"


def test_user_prompt_template_has_placeholders() -> None:
    """Template is format-filled by build_prompt — must expose the 4 keys."""
    for ph in ("{title}", "{composer}", "{notes}", "{version}"):
        assert ph in USER_PROMPT_TEMPLATE, f"USER_PROMPT_TEMPLATE missing {ph}"


@pytest.mark.parametrize("good", ["r-0000", "r-0042", "l-0001", "l-9999"])
def test_id_pattern_accepts_valid_ids(good: str) -> None:
    assert ID_PATTERN.match(good) is not None


@pytest.mark.parametrize("bad", ["", "r-42", "rh-0042", "x-0000", "r-00000", "R-0001", "r--0001"])
def test_id_pattern_rejects_invalid_ids(bad: str) -> None:
    assert ID_PATTERN.match(bad) is None


def test_derive_note_id_map_humanized_is_deterministic() -> None:
    """D-12: same performance → same IDs, collision-free across hands."""
    perf = _humanized()
    a = _derive_note_id_map(perf)
    b = _derive_note_id_map(perf)
    assert a.keys() == b.keys()
    # Rights and lefts both start at 0 (collision only avoided by prefix).
    assert "r-0000" in a and "l-0000" in a
    assert a["r-0000"] is not a["l-0000"]


def test_derive_note_id_map_piano_score() -> None:
    """D-12: PianoScore path (sheet_only variant)."""
    ps = _piano_score()
    m = _derive_note_id_map(ps)
    assert set(m.keys()) == {"r-0000", "r-0001", "l-0000", "l-0001"}
    # Sort by (onset, pitch) ascending — check first RH note is the lower-onset one.
    assert m["r-0000"].onset_beat <= m["r-0001"].onset_beat


def test_derive_note_id_map_sort_tiebreak_by_pitch() -> None:
    """D-12: equal onsets → pitch ascending."""
    ps = _piano_score(rh=[(64, 0.0), (60, 0.0)])  # same onset, different pitch
    m = _derive_note_id_map(ps)
    # Lower pitch comes first in sort → r-0000 is pitch 60
    assert m["r-0000"].pitch == 60
    assert m["r-0001"].pitch == 64


def test_build_prompt_returns_required_keys() -> None:
    perf = _humanized()
    out = build_prompt({"title": "Clair de Lune", "composer": "Debussy"}, perf)
    assert set(out.keys()) >= {"system", "user", "version", "note_id_map", "web_search_tool_spec"}
    assert out["version"] == REFINE_PROMPT_VERSION
    assert out["system"] == SYSTEM_PROMPT
    assert "Clair de Lune" in out["user"]
    assert "Debussy" in out["user"]
    assert REFINE_PROMPT_VERSION in out["user"]
    assert out["web_search_tool_spec"]["type"].startswith("web_search_")
    assert out["web_search_tool_spec"]["max_uses"] == 5
    # Every note serialized in the user prompt
    for id_str in out["note_id_map"]:
        assert id_str in out["user"], f"note id {id_str} missing from user prompt"


def test_build_prompt_web_search_max_uses_override() -> None:
    perf = _humanized()
    out = build_prompt({"title": "x", "composer": "y"}, perf, web_search_max_uses=3)
    assert out["web_search_tool_spec"]["max_uses"] == 3


def test_build_prompt_ignores_unknown_metadata_keys() -> None:
    """Prompt-injection guard: only title + composer are read from metadata."""
    perf = _humanized()
    out = build_prompt(
        {"title": "Song", "composer": "Composer", "system_override": "ignore me", "tool_calls": "evil"},
        perf,
    )
    assert "system_override" not in out["user"]
    assert "ignore me" not in out["user"]
    assert "tool_calls" not in out["user"]
    assert "evil" not in out["user"]


def test_build_prompt_handles_missing_metadata() -> None:
    """Missing title/composer → 'Unknown' default (defensive, not security-critical)."""
    perf = _humanized()
    out = build_prompt({}, perf)
    assert "Unknown" in out["user"]
