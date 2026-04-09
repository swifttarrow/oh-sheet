"""Unit tests for CREPE-based vocal melody extraction.

The end-to-end path (load audio → torchcrepe.predict → segment)
needs torch + torchcrepe, which are optional deps, so those tests
``importorskip`` and only run when the basic-pitch + crepe extras
are installed. The pure-Python segmenter ``_f0_to_notes`` is
dependency-free, so its tests always run and cover the note-
boundary logic that's most likely to regress.
"""
from __future__ import annotations

import pytest

from backend.services.crepe_melody import _f0_to_notes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hz(midi: int) -> float:
    """Rounded MIDI → Hz (A440 tuning)."""
    return 440.0 * (2 ** ((midi - 69) / 12.0))


def _stream(pitches: list[int | None], periodicity: float = 0.9) -> tuple[list[float], list[float]]:
    """Build a (pitch_hz, periodicity) stream from integer MIDI pitches.

    ``None`` entries are unvoiced (zero Hz). Used to hand-build
    per-frame streams that exercise the segmenter's merge/split
    boundary logic without touching torchcrepe.
    """
    hz = [0.0 if p is None else _hz(p) for p in pitches]
    pers = [0.0 if p is None else periodicity for p in pitches]
    return hz, pers


# 100 Hz frame rate (10 ms/frame) — matches the DEFAULT_HOP_LENGTH_SAMPLES
# config (160 samples at 16 kHz), and keeps the math simple: a 10-frame
# run is exactly 100 ms.
FRAME_RATE = 100.0


# ---------------------------------------------------------------------------
# Basic segmentation
# ---------------------------------------------------------------------------

def test_contiguous_same_pitch_becomes_one_note():
    # 30 frames (300 ms) of sustained MIDI 69 (A4).
    hz, pers = _stream([69] * 30)
    notes = _f0_to_notes(
        hz, pers, FRAME_RATE,
        min_note_duration_sec=0.06,
        merge_gap_sec=0.06,
        amp_min=0.1,
        amp_max=1.0,
    )
    assert len(notes) == 1
    start, end, pitch, amp, _ = notes[0]
    assert pitch == 69
    assert start == pytest.approx(0.0)
    assert end == pytest.approx(0.30)
    assert 0.85 <= amp <= 0.95  # periodicity was 0.9


def test_adjacent_different_pitches_become_separate_notes():
    # 10 frames A4 then 10 frames B4 — one pitch change, one boundary.
    hz, pers = _stream([69] * 10 + [71] * 10)
    notes = _f0_to_notes(
        hz, pers, FRAME_RATE,
        min_note_duration_sec=0.06,
        merge_gap_sec=0.06,
        amp_min=0.1,
        amp_max=1.0,
    )
    assert len(notes) == 2
    assert notes[0][2] == 69 and notes[1][2] == 71
    assert notes[0][1] == pytest.approx(0.10)
    assert notes[1][0] == pytest.approx(0.10)


def test_unvoiced_frames_break_notes():
    # Voiced / unvoiced / voiced — both islands of voicing get a note.
    hz, pers = _stream([69] * 10 + [None] * 20 + [69] * 10)
    notes = _f0_to_notes(
        hz, pers, FRAME_RATE,
        min_note_duration_sec=0.06,
        merge_gap_sec=0.05,       # shorter than the 200 ms gap
        amp_min=0.1,
        amp_max=1.0,
    )
    assert len(notes) == 2
    # Gap is 10→30 (0.10→0.30 sec) — second note starts at 0.30.
    assert notes[1][0] == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# Merge + min-duration thresholds
# ---------------------------------------------------------------------------

def test_short_gap_same_pitch_merges_into_one_note():
    # Short unvoiced bridge (50 ms = 5 frames) between two 100 ms
    # runs of the same pitch. With merge_gap_sec=0.06 they should
    # collapse into one continuous note.
    hz, pers = _stream([69] * 10 + [None] * 5 + [69] * 10)
    notes = _f0_to_notes(
        hz, pers, FRAME_RATE,
        min_note_duration_sec=0.06,
        merge_gap_sec=0.06,
        amp_min=0.1,
        amp_max=1.0,
    )
    assert len(notes) == 1
    assert notes[0][2] == 69
    assert notes[0][0] == pytest.approx(0.0)
    # Merged end is the second run's end, not a naive sum of the two
    # individual-run durations.
    assert notes[0][1] == pytest.approx(0.25)


def test_short_notes_dropped_below_min_duration():
    # 4 frames = 40 ms — below the 60 ms minimum.
    hz, pers = _stream([69] * 4)
    notes = _f0_to_notes(
        hz, pers, FRAME_RATE,
        min_note_duration_sec=0.06,
        merge_gap_sec=0.06,
        amp_min=0.1,
        amp_max=1.0,
    )
    assert notes == []


# ---------------------------------------------------------------------------
# Defensive handling
# ---------------------------------------------------------------------------

def test_nan_and_inf_pitch_treated_as_unvoiced():
    """torchcrepe uses ``UNVOICED == nan`` — the segmenter must not crash.

    The caller in ``extract_vocal_melody_crepe`` scrubs NaN with
    ``numpy.nan_to_num`` before calling us, but the segmenter is
    the line of last defense: a pure-Python function that's easy to
    reuse, and a NaN→``int()`` crash in production is a nightmare
    to diagnose after the fact.
    """
    hz = [_hz(69)] * 5 + [float("nan"), float("inf"), float("-inf")] + [_hz(69)] * 5
    pers = [0.9] * len(hz)
    notes = _f0_to_notes(
        hz, pers, FRAME_RATE,
        min_note_duration_sec=0.04,   # allow the 50 ms runs through
        merge_gap_sec=0.04,           # but don't merge across the gap
        amp_min=0.1,
        amp_max=1.0,
    )
    # Two 50 ms voiced runs with a 30 ms NaN bridge — below merge_gap,
    # so they should merge. Single output note.
    assert len(notes) == 1
    assert notes[0][2] == 69


def test_empty_input_returns_empty_list():
    assert _f0_to_notes([], [], FRAME_RATE,
                        min_note_duration_sec=0.06,
                        merge_gap_sec=0.06,
                        amp_min=0.1,
                        amp_max=1.0) == []


# ---------------------------------------------------------------------------
# Config defaults sync
# ---------------------------------------------------------------------------

def test_config_defaults_match_crepe_module_defaults():
    """``Settings.crepe_model`` must stay in lockstep with the module's
    ``DEFAULT_MODEL`` constant — otherwise a future rename on one side
    would silently drift from the other and the stems pipeline would
    run a different weights bag than the config docstring claims.
    """
    from backend.config import Settings
    from backend.services.crepe_melody import DEFAULT_MODEL

    s = Settings()
    assert s.crepe_model == DEFAULT_MODEL
