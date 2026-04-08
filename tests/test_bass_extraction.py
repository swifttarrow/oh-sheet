"""Unit tests for Phase 3 bass extraction.

Mirrors the synthetic-contour approach used by
``tests/test_melody_extraction.py`` — we build a hand-crafted ``(T, 264)``
salience matrix and drive the shared Viterbi tracer directly, skipping
the full Basic Pitch pipeline. That keeps the tests hermetic and fast.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from backend.services.bass_extraction import (  # noqa: E402
    DEFAULT_BASS_HIGH_MIDI,
    DEFAULT_BASS_LOW_MIDI,
    DEFAULT_BASS_MATCH_FRACTION,
    DEFAULT_BASS_MAX_TRANSITION_BINS,
    DEFAULT_BASS_TRANSITION_WEIGHT,
    DEFAULT_BASS_VOICING_FLOOR,
    extract_bass,
)
from backend.services.melody_extraction import (  # noqa: E402
    FRAME_RATE_HZ,
    N_CONTOUR_BINS,
    midi_to_bin,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_contour(n_frames: int, baseline: float = 0.02):
    return np.full((n_frames, N_CONTOUR_BINS), baseline, dtype=np.float32)


def _paint(contour, start_frame: int, end_frame: int, midi: int, salience: float = 0.9):
    contour[start_frame:end_frame, midi_to_bin(midi)] = salience


def _ne(start: float, end: float, pitch: int, amp: float = 0.8):
    return (start, end, pitch, amp, None)


# ---------------------------------------------------------------------------
# Tracer behavior in the low band
# ---------------------------------------------------------------------------

def test_extract_bass_tags_low_pedal_as_bass():
    # Stable E2 (MIDI 40) pedal for one second.
    frames = int(round(1.1 * FRAME_RATE_HZ))
    c = _blank_contour(frames)
    _paint(c, 0, int(round(1.0 * FRAME_RATE_HZ)), 40, salience=0.9)

    events = [_ne(0.0, 1.0, 40)]
    bass, remaining, stats = extract_bass(c, events)
    assert not stats.skipped
    assert bass == events
    assert remaining == []
    assert stats.bass_note_count == 1
    assert stats.remaining_note_count == 0
    assert stats.voiced_frame_fraction > 0.8


def test_extract_bass_ignores_peak_above_high_midi():
    # Peak at MIDI 67 (G4) — well above the bass band — should not attract
    # the low-register Viterbi path. The event is in the melody range, so
    # it also falls out of the bass band mask and stays in remaining.
    frames = int(round(1.1 * FRAME_RATE_HZ))
    c = _blank_contour(frames)
    _paint(c, 0, int(round(1.0 * FRAME_RATE_HZ)), 67, salience=0.9)

    events = [_ne(0.0, 1.0, 67)]
    bass, remaining, stats = extract_bass(c, events)
    assert bass == []
    assert remaining == events


def test_extract_bass_tracks_a_walking_line():
    # Walking bass: E2 → G2 → A2 → B2, quarter-second each.
    frames_per_sec = FRAME_RATE_HZ
    c = _blank_contour(int(round(1.1 * frames_per_sec)))
    _paint(c, 0, int(round(0.25 * frames_per_sec)), 40)
    _paint(c, int(round(0.25 * frames_per_sec)), int(round(0.5 * frames_per_sec)), 43)
    _paint(c, int(round(0.5 * frames_per_sec)), int(round(0.75 * frames_per_sec)), 45)
    _paint(c, int(round(0.75 * frames_per_sec)), int(round(1.0 * frames_per_sec)), 47)

    events = [
        _ne(0.0, 0.25, 40),
        _ne(0.25, 0.5, 43),
        _ne(0.5, 0.75, 45),
        _ne(0.75, 1.0, 47),
    ]
    bass, remaining, stats = extract_bass(c, events)
    assert sorted(e[2] for e in bass) == [40, 43, 45, 47]
    assert remaining == []


def test_extract_bass_rejects_disagreeing_note():
    # Path traces E2 throughout; an event at B1 (35) disagrees.
    frames = int(round(1.1 * FRAME_RATE_HZ))
    c = _blank_contour(frames)
    _paint(c, 0, int(round(1.0 * FRAME_RATE_HZ)), 40)

    events = [_ne(0.0, 1.0, 35)]  # 5 semitones away → outside tol
    bass, remaining, stats = extract_bass(c, events)
    assert bass == []
    assert remaining == events


def test_extract_bass_skips_when_contour_is_none():
    events = [_ne(0.0, 1.0, 40)]
    bass, remaining, stats = extract_bass(None, events)
    assert stats.skipped
    assert bass == []
    assert remaining == events
    assert any("skipped" in w for w in stats.as_warnings())


def test_extract_bass_skips_on_malformed_contour_shape():
    bad = np.zeros((100, 128), dtype=np.float32)  # wrong width
    events = [_ne(0.0, 1.0, 40)]
    bass, remaining, stats = extract_bass(bad, events)
    assert stats.skipped
    assert remaining == events


def test_extract_bass_empty_events_short_circuits():
    c = _blank_contour(50)
    _paint(c, 0, 50, 40)
    bass, remaining, stats = extract_bass(c, [])
    assert bass == [] and remaining == []
    assert not stats.skipped


def test_extract_bass_tiny_contour_is_skipped():
    c = np.zeros((1, N_CONTOUR_BINS), dtype=np.float32)
    events = [_ne(0.0, 1.0, 40)]
    bass, remaining, stats = extract_bass(c, events)
    assert stats.skipped
    assert remaining == events


# ---------------------------------------------------------------------------
# Config defaults sanity check
# ---------------------------------------------------------------------------

def test_config_defaults_match_bass_module_defaults():
    from backend.config import Settings

    s = Settings()
    assert s.bass_low_midi == DEFAULT_BASS_LOW_MIDI
    assert s.bass_high_midi == DEFAULT_BASS_HIGH_MIDI
    assert s.bass_voicing_floor == DEFAULT_BASS_VOICING_FLOOR
    assert s.bass_transition_weight == DEFAULT_BASS_TRANSITION_WEIGHT
    assert s.bass_max_transition_bins == DEFAULT_BASS_MAX_TRANSITION_BINS
    assert s.bass_match_fraction == DEFAULT_BASS_MATCH_FRACTION
    assert s.bass_extraction_enabled is True
