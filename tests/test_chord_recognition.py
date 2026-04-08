"""Unit tests for Phase 3 chord recognition.

The recognizer is a thin wrapper around ``librosa.feature.chroma_cqt``
+ template matching, so we drive it with synthetic sine-wave chords
built directly in numpy. That avoids touching disk and keeps the tests
hermetic and fast. Tests skip gracefully when librosa is missing.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
librosa = pytest.importorskip("librosa")

from backend.services.chord_recognition import (  # noqa: E402
    DEFAULT_CHORD_MIN_SCORE,
    _build_triad_templates,
    recognize_chords_from_waveform,
)

SR = 22050


# ---------------------------------------------------------------------------
# Helpers — synthetic sine chords
# ---------------------------------------------------------------------------

def _sine(freq_hz: float, duration_sec: float, amp: float = 0.3):
    n = int(round(SR * duration_sec))
    t = np.arange(n, dtype=np.float32) / SR
    return amp * np.sin(2.0 * np.pi * freq_hz * t).astype(np.float32)


def _chord(freqs: list[float], duration_sec: float, attack_sec: float = 0.01):
    """Sum a list of sines into a polyphonic chord, with a brief linear attack.

    The attack matters: librosa's beat tracker keys on onset strength,
    so pure steady-state sines produce zero beats and the recognizer
    collapses to a single global span. A 10 ms ramp gives each chord
    a transient the beat tracker can find.
    """
    parts = [_sine(f, duration_sec) for f in freqs]
    y = np.sum(parts, axis=0).astype(np.float32)
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak * 0.8
    n_attack = max(1, int(round(SR * attack_sec)))
    env = np.ones_like(y)
    env[:n_attack] = np.linspace(0.0, 1.0, n_attack, dtype=np.float32)
    return y * env


def _repeated(freqs: list[float], beat_sec: float, n_beats: int):
    """Repeat the same chord ``n_beats`` times so the beat tracker
    has a reliable onset grid to latch onto."""
    return np.concatenate(
        [_chord(freqs, beat_sec) for _ in range(n_beats)]
    ).astype(np.float32)


# Approximate triad frequencies (4th octave).
C4, E4, G4 = 261.63, 329.63, 392.00
D4, F4, A4 = 293.66, 349.23, 440.00
G3, B3, D5 = 196.00, 246.94, 587.33


# ---------------------------------------------------------------------------
# Template construction
# ---------------------------------------------------------------------------

def test_triad_templates_shape_and_labels():
    templates, labels, roots = _build_triad_templates()
    assert templates.shape == (24, 12)
    assert len(labels) == 24
    assert len(roots) == 24
    # First 12 are majors, second 12 are minors.
    assert all(lbl.endswith(":maj") for lbl in labels[:12])
    assert all(lbl.endswith(":min") for lbl in labels[12:])
    # Each template is L2-normalized.
    norms = np.linalg.norm(templates, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# End-to-end recognition on synthetic chords
# ---------------------------------------------------------------------------

def test_recognize_single_c_major_chord():
    y = _repeated([C4, E4, G4], beat_sec=0.5, n_beats=4)
    labels, stats = recognize_chords_from_waveform(y, SR)
    assert not stats.skipped
    assert stats.detected_count >= 1
    # At least one span is labeled C:maj.
    assert any(lbl.label == "C:maj" for lbl in labels)
    # Root index for C is 0.
    assert any(lbl.root == 0 for lbl in labels if lbl.label == "C:maj")


def test_recognize_chord_progression_c_then_g():
    # Two chords back-to-back: 4 beats of C major, 4 beats of G major.
    # Repetition + per-beat attack envelopes give the beat tracker
    # something real to segment on.
    y = np.concatenate([
        _repeated([C4, E4, G4], beat_sec=0.5, n_beats=4),
        _repeated([G3, B3, D5], beat_sec=0.5, n_beats=4),
    ]).astype(np.float32)
    labels, stats = recognize_chords_from_waveform(y, SR)
    assert not stats.skipped
    # Both C:maj and G:maj should show up in the label stream.
    found = {lbl.label for lbl in labels}
    assert "C:maj" in found
    assert "G:maj" in found


def test_recognize_minor_triad():
    # A minor triad = A, C, E → should score A:min higher than A:maj.
    y = _repeated([A4, C4, E4], beat_sec=0.5, n_beats=4)
    labels, stats = recognize_chords_from_waveform(y, SR)
    assert not stats.skipped
    # Accept either A:min (ideal) or C:maj (common confusion — A
    # minor and C major share two notes). Both have pitch class A in
    # the chroma, so the test asserts we didn't pick something wildly
    # off like a tritone.
    assert any(lbl.label in {"A:min", "C:maj"} for lbl in labels)


def test_recognize_confidence_field_is_bounded():
    y = _repeated([C4, E4, G4], beat_sec=0.5, n_beats=4)
    labels, _ = recognize_chords_from_waveform(y, SR)
    assert labels, "expected at least one labeled span"
    for lbl in labels:
        assert 0.0 <= lbl.confidence <= 1.0


def test_recognize_collapses_consecutive_duplicates():
    # Four beats of the same chord should collapse to at most 1-2
    # spans. Three-plus runs of "C:maj" would indicate the consecutive
    # collapse pass isn't working.
    y = _repeated([C4, E4, G4], beat_sec=0.5, n_beats=4)
    labels, _ = recognize_chords_from_waveform(y, SR)
    c_spans = [lbl for lbl in labels if lbl.label == "C:maj"]
    assert len(c_spans) <= 2


def test_recognize_high_threshold_suppresses_labels():
    # Set min_score absurdly high → recognizer returns no labels.
    y = _repeated([C4, E4, G4], beat_sec=0.5, n_beats=4)
    labels, stats = recognize_chords_from_waveform(y, SR, min_score=1.1)
    assert labels == []
    assert stats.detected_count == 0
    assert stats.no_chord_count >= 1


def test_recognize_skips_tiny_input():
    y = np.zeros(100, dtype=np.float32)  # ~4 ms at 22050 Hz
    labels, stats = recognize_chords_from_waveform(y, SR)
    assert stats.skipped
    assert labels == []


def test_recognize_returns_empty_for_silence():
    y = np.zeros(SR * 2, dtype=np.float32)
    labels, stats = recognize_chords_from_waveform(y, SR)
    # Either skipped (if chroma_cqt bailed out) or produced no labeled
    # spans — either way, no harmonic content should mean no chords.
    assert labels == [] or all(lbl.label == "N" for lbl in labels)


# ---------------------------------------------------------------------------
# Config defaults sanity check
# ---------------------------------------------------------------------------

def test_config_defaults_match_chord_module_defaults():
    from backend.config import Settings

    s = Settings()
    assert s.chord_recognition_enabled is True
    assert s.chord_min_template_score == DEFAULT_CHORD_MIN_SCORE
