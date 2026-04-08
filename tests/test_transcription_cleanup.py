"""Unit tests for Phase 1 transcription cleanup heuristics.

The cleanup module is pure Python with no runtime dependencies, so the
whole file runs in CI regardless of whether the ``basic-pitch`` extra is
installed. Each test constructs a synthetic ``note_events`` list (the
tuple format Basic Pitch returns) and asserts on the cleaned output.
"""
from __future__ import annotations

from backend.services.transcription_cleanup import (
    DEFAULT_GHOST_AMP_MEDIAN_SCALE,
    DEFAULT_GHOST_MAX_DURATION_SEC,
    DEFAULT_MERGE_GAP_SEC,
    DEFAULT_OCTAVE_AMP_RATIO,
    DEFAULT_OCTAVE_ONSET_TOL_SEC,
    cleanup_note_events,
)


# Shorthand: (start, end, pitch, amplitude, pitch_bends)
def _n(start: float, end: float, pitch: int, amp: float, bends=None):
    return (start, end, pitch, amp, bends)


# ---------------------------------------------------------------------------
# Merge fragmented sustains
# ---------------------------------------------------------------------------

def test_merge_joins_same_pitch_notes_within_gap():
    events = [
        _n(0.00, 0.50, 60, 0.9),
        _n(0.52, 1.00, 60, 0.8),  # gap = 20 ms ≤ 30 ms → merge
        _n(1.05, 1.40, 60, 0.7),  # gap = 50 ms > 30 ms → stays separate
    ]
    cleaned, stats = cleanup_note_events(events)
    assert stats.merged == 1
    # Expect two notes at pitch 60: [0.00, 1.00] and [1.05, 1.40]
    p60 = [e for e in cleaned if e[2] == 60]
    assert len(p60) == 2
    assert p60[0][0] == 0.00 and p60[0][1] == 1.00
    assert p60[0][3] == 0.9  # max amplitude kept
    assert p60[1][0] == 1.05


def test_merge_does_not_cross_pitches():
    events = [
        _n(0.00, 0.50, 60, 0.9),
        _n(0.50, 1.00, 62, 0.9),  # different pitch, even zero-gap → no merge
    ]
    cleaned, stats = cleanup_note_events(events)
    assert stats.merged == 0
    assert len(cleaned) == 2


def test_merge_preserves_max_amplitude():
    events = [
        _n(0.00, 0.50, 60, 0.3),
        _n(0.51, 1.00, 60, 0.9),  # quieter→louder fragment transition
    ]
    cleaned, _ = cleanup_note_events(events)
    assert len(cleaned) == 1
    assert cleaned[0][3] == 0.9


def test_merge_chains_multiple_fragments():
    # Four fragments all within merge_gap — should collapse to one.
    events = [
        _n(0.00, 0.20, 60, 0.8),
        _n(0.22, 0.40, 60, 0.85),
        _n(0.42, 0.60, 60, 0.7),
        _n(0.61, 0.80, 60, 0.9),
    ]
    cleaned, stats = cleanup_note_events(events)
    assert stats.merged == 3
    assert len(cleaned) == 1
    assert cleaned[0][0] == 0.0
    assert cleaned[0][1] == 0.80
    assert cleaned[0][3] == 0.9


# ---------------------------------------------------------------------------
# Octave-ghost pruning
# ---------------------------------------------------------------------------

def test_octave_ghost_dropped_when_upper_is_quiet():
    # Loud C4 with a quiet C5 ghost at the same onset → ghost removed.
    events = [
        _n(0.00, 0.50, 60, 0.9),
        _n(0.01, 0.30, 72, 0.3),  # 0.3 < 0.6 * 0.9 = 0.54 → ghost
    ]
    cleaned, stats = cleanup_note_events(events)
    assert stats.octave_ghosts_dropped == 1
    pitches = sorted(e[2] for e in cleaned)
    assert pitches == [60]


def test_real_octave_doubling_preserved():
    # Both notes equally loud → legitimate doubling, keep both.
    events = [
        _n(0.00, 0.50, 60, 0.9),
        _n(0.00, 0.50, 72, 0.85),
    ]
    cleaned, stats = cleanup_note_events(events)
    assert stats.octave_ghosts_dropped == 0
    pitches = sorted(e[2] for e in cleaned)
    assert pitches == [60, 72]


def test_octave_ghost_outside_onset_tolerance_kept():
    # Upper note onsets 200 ms after the lower → not a ghost, keep it.
    events = [
        _n(0.00, 0.50, 60, 0.9),
        _n(0.20, 0.50, 72, 0.3),
    ]
    cleaned, stats = cleanup_note_events(events)
    assert stats.octave_ghosts_dropped == 0
    assert len(cleaned) == 2


def test_octave_ghost_at_lowest_pitch_is_ignored():
    # A note at pitch 5 would look "down" to pitch -7 — must not crash.
    events = [_n(0.00, 0.50, 5, 0.9)]
    cleaned, stats = cleanup_note_events(events)
    assert stats.octave_ghosts_dropped == 0
    assert len(cleaned) == 1


# ---------------------------------------------------------------------------
# Ghost-tail pruning
# ---------------------------------------------------------------------------

def test_short_quiet_note_is_dropped():
    # Median amp across the set is 0.8 → threshold = 0.5 * 0.8 = 0.4.
    # The 40 ms note at amp 0.2 is below threshold and shorter than 60 ms.
    events = [
        _n(0.00, 0.50, 60, 0.8),
        _n(0.50, 1.00, 62, 0.8),
        _n(1.00, 1.50, 64, 0.8),
        _n(1.50, 1.54, 70, 0.2),  # ghost tail
    ]
    cleaned, stats = cleanup_note_events(events)
    assert stats.ghost_tails_dropped == 1
    assert all(e[2] != 70 for e in cleaned)


def test_short_loud_staccato_preserved():
    # Short (40 ms) but loud staccato should survive.
    events = [
        _n(0.00, 0.50, 60, 0.5),
        _n(0.50, 1.00, 62, 0.5),
        _n(1.00, 1.04, 64, 0.9),  # short but above amp threshold
    ]
    cleaned, stats = cleanup_note_events(events)
    assert stats.ghost_tails_dropped == 0
    assert len(cleaned) == 3


def test_long_quiet_sustained_pad_preserved():
    # Long (800 ms) but quiet pad should survive — only short+quiet gets cut.
    events = [
        _n(0.00, 0.50, 60, 0.9),
        _n(0.50, 1.00, 62, 0.9),
        _n(1.00, 1.80, 64, 0.1),  # quiet, below threshold, but long
    ]
    cleaned, stats = cleanup_note_events(events)
    assert stats.ghost_tails_dropped == 0
    assert len(cleaned) == 3


# ---------------------------------------------------------------------------
# End-to-end behaviour
# ---------------------------------------------------------------------------

def test_empty_input_yields_empty_output():
    cleaned, stats = cleanup_note_events([])
    assert cleaned == []
    assert stats.input_count == 0
    assert stats.output_count == 0
    assert stats.merged == 0
    assert stats.octave_ghosts_dropped == 0
    assert stats.ghost_tails_dropped == 0


def test_idempotent_on_clean_input():
    # A set with no artifacts should pass through untouched.
    events = [
        _n(0.00, 0.50, 60, 0.8),
        _n(0.50, 1.00, 62, 0.75),
        _n(1.00, 1.50, 64, 0.85),
        _n(1.50, 2.00, 65, 0.8),
    ]
    cleaned_once, _ = cleanup_note_events(events)
    cleaned_twice, stats = cleanup_note_events(cleaned_once)
    assert cleaned_once == cleaned_twice
    assert stats.merged == 0
    assert stats.octave_ghosts_dropped == 0
    assert stats.ghost_tails_dropped == 0


def test_stats_warnings_only_report_nonzero_passes():
    events = [
        _n(0.00, 0.50, 60, 0.9),
        _n(0.51, 1.00, 60, 0.9),  # merged
        _n(0.00, 0.50, 72, 0.2),  # octave ghost
    ]
    _, stats = cleanup_note_events(events)
    warnings = stats.as_warnings()
    assert any("merged" in w for w in warnings)
    assert any("octave" in w for w in warnings)
    # No ghost-tail drops expected — confirm that line is absent.
    assert not any("ghost-tail" in w for w in warnings)


def test_pass_ordering_merge_then_octave_then_ghost():
    # Construct a case where the order matters: two fragments of a note
    # at pitch 60 (each < ghost duration on its own) plus a ghost at 72.
    # If merge ran *after* ghost pruning, both fragments would look like
    # ghost-tails and get dropped — we want them merged first.
    events = [
        _n(0.00, 0.03, 60, 0.9),
        _n(0.031, 0.06, 60, 0.9),  # merged with the first into a 60 ms note
        _n(0.00, 0.06, 72, 0.3),  # octave ghost dropped after merge
    ]
    cleaned, stats = cleanup_note_events(events)
    assert stats.merged == 1
    assert stats.octave_ghosts_dropped == 1
    # The merged note should survive ghost-tail pruning because its
    # duration equals the threshold (not strictly less than).
    pitches = sorted(e[2] for e in cleaned)
    assert pitches == [60]


def test_defaults_match_config_defaults():
    # The config knobs default to the module-level DEFAULT_* constants —
    # make sure both sides agree so callers get the documented behaviour
    # when they don't override.
    from backend.config import Settings

    s = Settings()
    assert s.cleanup_merge_gap_sec == DEFAULT_MERGE_GAP_SEC
    assert s.cleanup_octave_amp_ratio == DEFAULT_OCTAVE_AMP_RATIO
    assert s.cleanup_octave_onset_tol_sec == DEFAULT_OCTAVE_ONSET_TOL_SEC
    assert s.cleanup_ghost_max_duration_sec == DEFAULT_GHOST_MAX_DURATION_SEC
    assert s.cleanup_ghost_amp_median_scale == DEFAULT_GHOST_AMP_MEDIAN_SCALE
