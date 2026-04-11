"""Unit tests for offset/duration accuracy improvements.

Tests the Pass 5 energy gating cleanup, the per-role cleanup threshold
selection, and edge cases around short/quiet notes that must not be
trimmed. All tests use synthetic data — no ML deps required.
"""
from __future__ import annotations

from types import SimpleNamespace

from backend.services.transcription_cleanup import (
    DEFAULT_ENERGY_GATE_FLOOR_RATIO,
    DEFAULT_ENERGY_GATE_MAX_SUSTAIN_SEC,
    DEFAULT_ENERGY_GATE_TAIL_SEC,
    AmplitudeEnvelope,
    CleanupStats,
    _gate_offsets_by_energy,
    cleanup_for_role,
    cleanup_note_events,
)


# Shorthand: (start, end, pitch, amplitude, pitch_bends)
def _n(start: float, end: float, pitch: int, amp: float, bends=None):
    return (start, end, pitch, amp, bends)


# ---------------------------------------------------------------------------
# Pass 5 — energy gating with amplitude envelope
# ---------------------------------------------------------------------------

def test_energy_gate_trims_long_note_at_decay_point():
    """A 5-second note whose envelope decays at t=1.5 should be trimmed."""
    events = [_n(0.0, 5.0, 60, 0.8)]
    # Envelope: loud from 0-1s, decays sharply at 1.5s
    envelope: AmplitudeEnvelope = [
        (0.0, 0.8), (0.5, 0.9), (1.0, 0.7),
        (1.5, 0.05),  # below 0.1 * 0.9 = 0.09 floor
        (2.0, 0.03), (3.0, 0.01), (4.0, 0.01),
    ]
    result, count = _gate_offsets_by_energy(
        events,
        max_sustain_sec=2.0,
        floor_ratio=0.1,
        tail_sec=0.05,
        amplitude_envelope=envelope,
    )
    assert count == 1
    assert len(result) == 1
    # Trimmed to decay point (1.5) + tail (0.05) = 1.55
    assert abs(result[0][1] - 1.55) < 0.001


def test_energy_gate_preserves_short_notes():
    """Notes shorter than max_sustain_sec should not be touched."""
    events = [_n(0.0, 1.5, 60, 0.8)]
    envelope: AmplitudeEnvelope = [
        (0.0, 0.8), (0.5, 0.05), (1.0, 0.01),
    ]
    result, count = _gate_offsets_by_energy(
        events,
        max_sustain_sec=2.0,
        floor_ratio=0.1,
        tail_sec=0.05,
        amplitude_envelope=envelope,
    )
    assert count == 0
    assert result[0][1] == 1.5  # unchanged


def test_energy_gate_heuristic_trims_long_quiet_note():
    """Without envelope, long + quiet notes should be trimmed to max_sustain."""
    events = [
        _n(0.0, 1.0, 60, 0.9),    # loud — above median
        _n(1.0, 2.0, 62, 0.8),    # loud — above median
        _n(2.0, 6.0, 64, 0.2),    # quiet + long → trim
    ]
    result, count = _gate_offsets_by_energy(
        events,
        max_sustain_sec=2.0,
        floor_ratio=0.1,
        tail_sec=0.05,
        amplitude_envelope=None,  # heuristic path
    )
    assert count == 1
    assert len(result) == 3
    # The quiet long note should be trimmed to start + max_sustain
    assert abs(result[2][1] - 4.0) < 0.001  # 2.0 + 2.0


def test_energy_gate_heuristic_preserves_long_loud_note():
    """Long but loud notes should not be trimmed by the heuristic."""
    events = [
        _n(0.0, 1.0, 60, 0.5),
        _n(1.0, 2.0, 62, 0.5),
        _n(2.0, 6.0, 64, 0.9),    # loud — above median → keep
    ]
    result, count = _gate_offsets_by_energy(
        events,
        max_sustain_sec=2.0,
        floor_ratio=0.1,
        tail_sec=0.05,
        amplitude_envelope=None,
    )
    assert count == 0
    assert result[2][1] == 6.0  # unchanged


def test_energy_gate_preserves_very_short_notes():
    """Very short notes (< 100ms) must never be affected by energy gating."""
    events = [
        _n(0.0, 0.05, 60, 0.3),   # 50ms staccato
        _n(0.1, 0.15, 62, 0.2),   # 50ms staccato
    ]
    result, count = _gate_offsets_by_energy(
        events,
        max_sustain_sec=2.0,
        floor_ratio=0.1,
        tail_sec=0.05,
        amplitude_envelope=None,
    )
    assert count == 0
    assert len(result) == 2
    assert result[0] == events[0]
    assert result[1] == events[1]


def test_energy_gate_empty_input():
    """Empty event list should produce empty output."""
    result, count = _gate_offsets_by_energy(
        [],
        max_sustain_sec=2.0,
        floor_ratio=0.1,
        tail_sec=0.05,
        amplitude_envelope=None,
    )
    assert result == []
    assert count == 0


def test_energy_gate_envelope_no_decay():
    """If RMS never drops below floor, note should not be trimmed."""
    events = [_n(0.0, 5.0, 60, 0.8)]
    # Envelope stays loud the entire time
    envelope: AmplitudeEnvelope = [
        (0.0, 0.8), (1.0, 0.7), (2.0, 0.6),
        (3.0, 0.5), (4.0, 0.4), (5.0, 0.3),
    ]
    result, count = _gate_offsets_by_energy(
        events,
        max_sustain_sec=2.0,
        floor_ratio=0.1,
        tail_sec=0.05,
        amplitude_envelope=envelope,
    )
    assert count == 0
    assert result[0][1] == 5.0  # unchanged


def test_energy_gate_multiple_notes_mixed():
    """Mix of notes: only the long ones with envelope decay get trimmed."""
    events = [
        _n(0.0, 1.0, 60, 0.8),    # short — untouched
        _n(1.0, 5.0, 62, 0.7),    # long, envelope decays at 2.5
        _n(5.0, 5.5, 64, 0.6),    # short — untouched
    ]
    envelope: AmplitudeEnvelope = [
        (0.0, 0.9), (0.5, 0.8), (1.0, 0.85), (1.5, 0.7),
        (2.0, 0.3), (2.5, 0.05),  # decay below floor
        (3.0, 0.02), (4.0, 0.01), (5.0, 0.6), (5.5, 0.5),
    ]
    result, count = _gate_offsets_by_energy(
        events,
        max_sustain_sec=2.0,
        floor_ratio=0.1,
        tail_sec=0.05,
        amplitude_envelope=envelope,
    )
    assert count == 1
    assert result[0][1] == 1.0     # first note unchanged
    assert result[1][1] < 5.0      # second note trimmed
    assert result[2][1] == 5.5     # third note unchanged


# ---------------------------------------------------------------------------
# Integration: energy gating via cleanup_note_events
# ---------------------------------------------------------------------------

def test_cleanup_note_events_runs_energy_gating():
    """Verify Pass 5 fires when energy_gate_enabled=True."""
    events = [
        _n(0.0, 1.0, 60, 0.9),
        _n(1.0, 2.0, 62, 0.8),
        _n(2.0, 6.0, 64, 0.2),    # long + quiet → heuristic trim
    ]
    cleaned, stats = cleanup_note_events(
        events,
        energy_gate_enabled=True,
        energy_gate_max_sustain_sec=2.0,
    )
    assert stats.energy_gated >= 1
    # The long quiet note should have been trimmed
    p64 = [e for e in cleaned if e[2] == 64]
    assert p64[0][1] < 6.0


def test_cleanup_note_events_skips_energy_gating_when_disabled():
    """Verify Pass 5 does not fire when energy_gate_enabled=False."""
    events = [
        _n(0.0, 1.0, 60, 0.9),
        _n(1.0, 2.0, 62, 0.8),
        _n(2.0, 6.0, 64, 0.2),
    ]
    cleaned, stats = cleanup_note_events(
        events,
        energy_gate_enabled=False,
    )
    assert stats.energy_gated == 0
    p64 = [e for e in cleaned if e[2] == 64]
    assert p64[0][1] == 6.0  # unchanged


def test_energy_gated_stat_in_warnings():
    """CleanupStats.as_warnings() should report energy-gated count."""
    stats = CleanupStats(energy_gated=5)
    warnings = stats.as_warnings()
    assert any("energy-gated" in w for w in warnings)


def test_energy_gated_stat_not_in_warnings_when_zero():
    """CleanupStats.as_warnings() should not mention energy gating when zero."""
    stats = CleanupStats(energy_gated=0)
    warnings = stats.as_warnings()
    assert not any("energy-gated" in w for w in warnings)


# ---------------------------------------------------------------------------
# Per-role cleanup threshold selection
# ---------------------------------------------------------------------------

def _make_settings(**overrides):
    """Build a SimpleNamespace mimicking backend.config.Settings for testing."""
    defaults = {
        # Global cleanup defaults
        "cleanup_merge_gap_sec": 0.03,
        "cleanup_octave_amp_ratio": 0.6,
        "cleanup_octave_onset_tol_sec": 0.05,
        "cleanup_ghost_max_duration_sec": 0.05,
        "cleanup_ghost_amp_median_scale": 0.5,
        # Per-role overrides
        "cleanup_melody_merge_gap_sec": 0.02,
        "cleanup_melody_ghost_max_duration_sec": 0.04,
        "cleanup_bass_merge_gap_sec": 0.04,
        "cleanup_bass_ghost_max_duration_sec": 0.06,
        "cleanup_chords_merge_gap_sec": 0.04,
        "cleanup_chords_octave_amp_ratio": 0.5,
        # Energy gating
        "cleanup_energy_gate_enabled": True,
        "cleanup_energy_gate_max_sustain_sec": 2.0,
        "cleanup_energy_gate_floor_ratio": 0.1,
        "cleanup_energy_gate_tail_sec": 0.05,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_cleanup_for_role_melody_uses_tighter_merge():
    """Melody role should use 0.02 merge gap instead of global 0.03."""
    # Two fragments with 25ms gap: below global 0.03 but above melody's 0.02
    events = [
        _n(0.0, 0.50, 60, 0.9),
        _n(0.525, 1.00, 60, 0.8),  # 25ms gap
    ]
    s = _make_settings()

    # With melody role (0.02 gap) — should NOT merge (25ms > 20ms)
    cleaned_melody, stats_melody = cleanup_for_role(events, "melody", s)
    assert stats_melody.merged == 0
    assert len([e for e in cleaned_melody if e[2] == 60]) == 2

    # With global defaults (0.03 gap) — should merge (25ms < 30ms)
    cleaned_global, stats_global = cleanup_note_events(events)
    assert stats_global.merged == 1
    assert len([e for e in cleaned_global if e[2] == 60]) == 1


def test_cleanup_for_role_bass_uses_looser_merge():
    """Bass role should use 0.04 merge gap instead of global 0.03."""
    # Two fragments with 35ms gap: above global 0.03 but below bass's 0.04
    events = [
        _n(0.0, 0.50, 36, 0.9),
        _n(0.535, 1.00, 36, 0.8),  # 35ms gap
    ]
    s = _make_settings()

    # With bass role (0.04 gap) — should merge (35ms < 40ms)
    cleaned_bass, stats_bass = cleanup_for_role(events, "bass", s)
    assert stats_bass.merged == 1

    # With global defaults (0.03 gap) — should NOT merge (35ms > 30ms)
    cleaned_global, stats_global = cleanup_note_events(events)
    assert stats_global.merged == 0


def test_cleanup_for_role_chords_uses_stricter_octave_ratio():
    """Chords role should use 0.5 octave ratio instead of global 0.6."""
    # Upper note at amp 0.35 with lower at 0.7:
    #   0.35 < 0.5 * 0.7 = 0.35 → borderline, dropped by chords (0.5 ratio)
    #   0.35 < 0.6 * 0.7 = 0.42 → would also be dropped by global (0.6 ratio)
    # Use amp 0.40 instead:
    #   0.40 < 0.5 * 0.7 = 0.35 → NOT dropped by chords
    #   0.40 < 0.6 * 0.7 = 0.42 → dropped by global (0.6 ratio)
    events = [
        _n(0.0, 0.50, 60, 0.7),
        _n(0.0, 0.50, 72, 0.40),  # upper octave
    ]
    s = _make_settings()

    # With chords role (0.5 ratio): 0.40 < 0.5 * 0.7 = 0.35? No, 0.40 > 0.35 → keep
    cleaned_chords, stats_chords = cleanup_for_role(events, "chords", s)
    assert stats_chords.octave_ghosts_dropped == 0

    # With global (0.6 ratio): 0.40 < 0.6 * 0.7 = 0.42? Yes → drop
    cleaned_global, stats_global = cleanup_note_events(events)
    assert stats_global.octave_ghosts_dropped == 1


def test_cleanup_for_role_unknown_role_uses_globals():
    """An unrecognized role should fall back to global defaults."""
    events = [
        _n(0.0, 0.50, 60, 0.9),
        _n(0.525, 1.00, 60, 0.8),  # 25ms gap
    ]
    s = _make_settings()

    # Unknown role uses global 0.03 merge gap — 25ms < 30ms → merge
    cleaned, stats = cleanup_for_role(events, "unknown", s)
    assert stats.merged == 1


def test_cleanup_for_role_with_envelope():
    """Per-role cleanup should pass envelope through to energy gating."""
    events = [_n(0.0, 5.0, 60, 0.8)]
    envelope: AmplitudeEnvelope = [
        (0.0, 0.8), (0.5, 0.9), (1.0, 0.7),
        (1.5, 0.05), (2.0, 0.03), (2.5, 0.02),
    ]
    s = _make_settings()

    cleaned, stats = cleanup_for_role(
        events, "melody", s, amplitude_envelope=envelope,
    )
    assert stats.energy_gated == 1
    assert cleaned[0][1] < 5.0


def test_cleanup_for_role_energy_gate_disabled():
    """Per-role cleanup should skip energy gating when disabled in settings."""
    events = [
        _n(0.0, 1.0, 60, 0.9),
        _n(1.0, 6.0, 62, 0.2),  # long + quiet
    ]
    s = _make_settings(cleanup_energy_gate_enabled=False)

    cleaned, stats = cleanup_for_role(events, "melody", s)
    assert stats.energy_gated == 0


# ---------------------------------------------------------------------------
# Config defaults agreement
# ---------------------------------------------------------------------------

def test_energy_gate_defaults_match_config():
    """Ensure the module DEFAULT_* constants match the config defaults."""
    from backend.config import Settings

    s = Settings()
    assert s.cleanup_energy_gate_max_sustain_sec == DEFAULT_ENERGY_GATE_MAX_SUSTAIN_SEC
    assert s.cleanup_energy_gate_floor_ratio == DEFAULT_ENERGY_GATE_FLOOR_RATIO
    assert s.cleanup_energy_gate_tail_sec == DEFAULT_ENERGY_GATE_TAIL_SEC
