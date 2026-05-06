"""Phase 7 Tier 3 arrangement-quality metric tests.

Acceptance per
``docs/research/transcription-improvement-implementation-plan.md`` §Phase 7:

* Each Tier 3 metric exists, returns values in ``[0, 1]``, and tolerates
  empty / single-note scores by returning a 0.0 score with a note rather
  than raising.
* The composite ``Tier3Result.composite`` matches strategy doc §8.2's
  weighting (``0.5·play + 0.3·vleading + 0.2·density``).
* Engraving heuristic checks fire for the canonical bad-score patterns
  (oversized chords, hand crossings, voice crossings, ledger-line
  excess).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.contracts import (  # noqa: E402
    PianoScore,
    ScoreMetadata,
    ScoreNote,
    TempoMapEntry,
)

from eval.tier3_arrangement import (  # noqa: E402
    Tier3Result,
    compute_tier3,
    engraving_heuristic_checks,
    playability_fraction,
    polyphony_density,
    sight_readability_score,
    voice_leading_smoothness,
)

# ---------------------------------------------------------------------------
# Fixtures — handcrafted PianoScores so tests stay deterministic
# ---------------------------------------------------------------------------

def _make_score(rh, lh) -> PianoScore:
    """Build a PianoScore from ``(pitch, onset_beat, voice)`` tuples.

    ``voice`` defaults to 1 when the tuple has only 2 entries — keeps
    the call sites readable for fixtures that don't exercise voice
    crossing.
    """
    def _to_notes(prefix, items):
        out = []
        for i, item in enumerate(items):
            if len(item) == 3:
                pitch, onset, voice = item
            else:
                pitch, onset = item
                voice = 1
            out.append(ScoreNote(
                id=f"{prefix}-{i:04d}",
                pitch=pitch,
                onset_beat=onset,
                duration_beat=1.0,
                velocity=80,
                voice=voice,
            ))
        return out

    return PianoScore(
        right_hand=_to_notes("rh", rh),
        left_hand=_to_notes("lh", lh),
        metadata=ScoreMetadata(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
            difficulty="intermediate",
        ),
    )


# ---------------------------------------------------------------------------
# playability_fraction — re-export from tier_rf, parity test
# ---------------------------------------------------------------------------

def test_playability_fraction_all_playable():
    score = _make_score(
        rh=[(60, 0.0), (64, 0.0), (67, 0.0)],
        lh=[(48, 0.0), (52, 0.0)],
    )
    fraction, n_play, n_total = playability_fraction(score)
    assert n_total == 2
    assert n_play == 2
    assert fraction == 1.0


def test_playability_fraction_unreachable_chord():
    # 6-note chord + 18-semitone span — both fail the gate.
    rh = [(60 + 2 * i, 0.0) for i in range(6)]
    lh = [(36, 0.0), (54, 0.0)]
    score = _make_score(rh, lh)
    fraction, n_play, n_total = playability_fraction(score)
    assert fraction == 0.0
    assert n_play == 0
    assert n_total == 2


# ---------------------------------------------------------------------------
# voice_leading_smoothness
# ---------------------------------------------------------------------------

def test_voice_leading_smoothness_perfect_when_voice_holds_pitch():
    # Voice 1 stays on C across all 3 chord groups — zero displacement.
    rh = [(60, 0.0, 1), (60, 1.0, 1), (60, 2.0, 1)]
    lh = [(48, 0.0, 1), (48, 1.0, 1), (48, 2.0, 1)]
    score = _make_score(rh, lh)
    score_val, n_pairs = voice_leading_smoothness(score)
    assert n_pairs == 4  # 2 RH transitions + 2 LH transitions
    assert score_val == 1.0


def test_voice_leading_smoothness_drops_on_octave_jumps():
    # Each transition is a full octave — should hit zero smoothness.
    rh = [(60, 0.0, 1), (72, 1.0, 1), (60, 2.0, 1)]
    lh = [(48, 0.0, 1)]
    score = _make_score(rh, lh)
    score_val, n_pairs = voice_leading_smoothness(score)
    assert n_pairs == 2  # 2 RH transitions; LH has only one chord
    assert score_val == 0.0


def test_voice_leading_smoothness_handles_single_chord():
    score = _make_score(rh=[(60, 0.0)], lh=[(48, 0.0)])
    score_val, n_pairs = voice_leading_smoothness(score)
    assert n_pairs == 0
    assert score_val == 0.0


# ---------------------------------------------------------------------------
# polyphony_density
# ---------------------------------------------------------------------------

def test_polyphony_density_target_fitness_peaks_at_2_5():
    # 2.5 notes/beat → fitness should hit 1.0. Two notes in beat 0,
    # three in beat 1 → mean = 2.5 across 2 buckets in RH.
    rh = [
        (60, 0.0), (64, 0.0),                      # beat 0: 2 notes
        (62, 1.0), (65, 1.0), (69, 1.0),           # beat 1: 3 notes
    ]
    lh = []
    score = _make_score(rh, lh)
    mean, p95, mx, target_fit, n = polyphony_density(score)
    assert n == 2
    assert mean == pytest.approx(2.5)
    assert mx == 3
    assert target_fit == pytest.approx(1.0)


def test_polyphony_density_target_fitness_drops_at_extremes():
    # 5 notes per beat → mean = 5.0 → fitness should hit 0.0.
    rh = [(60 + i, 0.0) for i in range(5)]
    score = _make_score(rh, [])
    mean, _, mx, target_fit, _ = polyphony_density(score)
    assert mean == pytest.approx(5.0)
    assert mx == 5
    assert target_fit == 0.0


def test_polyphony_density_handles_empty_score():
    score = _make_score(rh=[], lh=[])
    mean, p95, mx, target_fit, n = polyphony_density(score)
    assert (mean, p95, mx, target_fit, n) == (0.0, 0.0, 0.0, 0.0, 0)


# ---------------------------------------------------------------------------
# engraving_heuristic_checks
# ---------------------------------------------------------------------------

def test_engraving_warnings_clean_score_has_no_warnings():
    rh = [(60, 0.0), (64, 0.0), (67, 0.0)]
    lh = [(48, 0.0), (52, 0.0)]
    score = _make_score(rh, lh)
    assert engraving_heuristic_checks(score) == []


def test_engraving_warnings_flag_ledger_excess():
    # A0 (21) is far below the bass-clef ledger-line floor (28).
    rh = [(60, 0.0)]
    lh = [(21, 0.0)]
    score = _make_score(rh, lh)
    warnings = engraving_heuristic_checks(score)
    assert any("ledger_excess_lh" in w for w in warnings)


def test_engraving_warnings_flag_hand_crossing():
    # LH up at C5 (72), RH down at C4 (60) → hand crossing.
    rh = [(60, 0.0)]
    lh = [(72, 0.0)]
    score = _make_score(rh, lh)
    warnings = engraving_heuristic_checks(score)
    assert any("hand_crossing" in w for w in warnings)


def test_engraving_warnings_flag_voice_crossing():
    # Within RH at beat 0: voice 1 plays C4 (60), voice 2 plays E4 (64).
    # Voice 2 (lower-numbered = higher) should be ABOVE voice 1 — but
    # here it's higher in pitch, so this is a voice crossing relative
    # to the convention. Wait, voice 1 should be the highest. So voice
    # 1=C4, voice 2=E4 means voice 2 is HIGHER than voice 1 — that's
    # a crossing.
    rh = [(60, 0.0, 1), (64, 0.0, 2)]
    lh = [(48, 0.0)]
    score = _make_score(rh, lh)
    warnings = engraving_heuristic_checks(score)
    assert any("voice_crossing_rh" in w for w in warnings)


# ---------------------------------------------------------------------------
# sight_readability_score
# ---------------------------------------------------------------------------

def test_sight_readability_clean_score_high():
    # Comfortable triads, no warnings, target density.
    rh = [(60, 0.0), (64, 0.0), (62, 1.0), (65, 1.0)]
    lh = [(48, 0.0), (43, 1.0)]
    score = _make_score(rh, lh)
    sr = sight_readability_score(score)
    assert sr > 0.7, f"clean score should read above 0.7, got {sr}"


def test_sight_readability_punishes_warnings():
    rh = [(60, 0.0), (62, 0.0)]
    lh = [(72, 0.0)]  # Hand crossing
    score = _make_score(rh, lh)
    sr_with_xc = sight_readability_score(score)

    rh_clean = [(60, 0.0), (62, 0.0)]
    lh_clean = [(48, 0.0)]
    score_clean = _make_score(rh_clean, lh_clean)
    sr_clean = sight_readability_score(score_clean)

    assert sr_clean > sr_with_xc, (
        f"hand crossing should drop readability "
        f"(clean={sr_clean:.3f}, crossing={sr_with_xc:.3f})"
    )


def test_sight_readability_zero_for_empty_score():
    score = _make_score(rh=[], lh=[])
    assert sight_readability_score(score) == 0.0


# ---------------------------------------------------------------------------
# compute_tier3 — top-level wiring
# ---------------------------------------------------------------------------

def test_compute_tier3_returns_unit_range_metrics():
    rh = [(60, 0.0), (64, 0.0), (67, 0.0), (62, 1.0), (65, 1.0)]
    lh = [(48, 0.0), (43, 1.0)]
    score = _make_score(rh, lh)
    result = compute_tier3(score)

    assert isinstance(result, Tier3Result)
    for name, value in (
        ("playability_fraction", result.playability_fraction),
        ("voice_leading_smoothness", result.voice_leading_smoothness),
        ("polyphony_in_target_range", result.polyphony_in_target_range),
        ("sight_readability", result.sight_readability),
        ("composite", result.composite),
    ):
        assert 0.0 <= value <= 1.0, f"{name}={value} out of [0, 1]"


def test_compute_tier3_composite_matches_strategy_doc_weighting():
    """Composite must be ``0.5·play + 0.3·vleading + 0.2·density``."""
    rh = [(60, 0.0, 1), (60, 1.0, 1), (60, 2.0, 1)]
    lh = [(48, 0.0, 1)]
    score = _make_score(rh, lh)
    result = compute_tier3(score)
    expected = (
        0.5 * result.playability_fraction
        + 0.3 * result.voice_leading_smoothness
        + 0.2 * result.polyphony_in_target_range
    )
    assert result.composite == pytest.approx(expected)


def test_compute_tier3_as_dict_round_trips():
    score = _make_score([(60, 0.0)], [(48, 0.0)])
    result = compute_tier3(score)
    payload = result.as_dict()
    # All headline fields plus composite + diagnostic counts present.
    for key in (
        "playability_fraction", "voice_leading_smoothness",
        "polyphony_in_target_range", "sight_readability", "composite",
        "n_playable_chords", "n_total_chords", "n_engraving_warnings",
    ):
        assert key in payload, f"as_dict missing {key}"


def test_compute_tier3_handles_empty_score():
    score = _make_score([], [])
    result = compute_tier3(score)
    assert result.composite == 0.0
    # Diagnostic notes capture the absences.
    notes_str = " ".join(result.notes)
    assert "empty" in notes_str.lower() or "no notes" in notes_str.lower()
