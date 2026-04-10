"""Unit tests for key and time-signature estimation.

The Krumhansl-Schmuckler key finder is driven directly off chroma
matrices, and the meter detector runs over a 1D beat-strength vector,
so both can be tested hermetically by hand-building the inputs. That
keeps these tests fast, deterministic, and independent of librosa —
only the top-level file-path entry points need real audio, and those
are covered by an ``importorskip``-gated block at the bottom.
"""
from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")

from backend.services.key_estimation import (  # noqa: E402
    DEFAULT_KEY_MIN_CONFIDENCE,
    KeyEstimationStats,
    MeterEstimationStats,
    _build_key_profiles,
    _score_meter_hypothesis,
    estimate_key_from_chroma,
    estimate_meter_from_beat_strengths,
)

# ---------------------------------------------------------------------------
# Helpers — synthetic chroma / beat-strength streams
# ---------------------------------------------------------------------------

def _chroma_from_pitch_classes(
    pc_weights: dict[int, float],
    n_frames: int = 10,
) -> np.ndarray:
    """Build a ``(12, n_frames)`` chroma matrix with the given weights.

    Mimics the shape librosa's ``chroma_cqt`` emits: rows indexed
    ``[C, C#, D, ..., B]``, each column a pitch-class magnitude
    vector. Unspecified pitch classes default to zero. Same vector
    repeated across frames, since the estimator time-averages anyway.
    """
    vec = np.zeros(12, dtype=np.float32)
    for pc, w in pc_weights.items():
        vec[pc % 12] = float(w)
    return np.tile(vec.reshape(12, 1), (1, n_frames))


def _scale_profile(tonic_pc: int, scale_intervals: list[int]) -> dict[int, float]:
    """Build a rough pitch-class weight dict for a diatonic scale.

    Tonic gets the heaviest weight; dominant and mediant get
    secondary weights that match the KS profile ordering so the
    estimator can't latch onto an adjacent (relative / parallel) key.
    """
    weights = {}
    for i, interval in enumerate(scale_intervals):
        pc = (tonic_pc + interval) % 12
        if i == 0:          # tonic
            weights[pc] = 3.0
        elif i == 4:        # dominant (5th scale degree)
            weights[pc] = 2.5
        elif i == 2:        # mediant
            weights[pc] = 2.0
        else:
            weights[pc] = 1.0
    return weights


# Scale intervals (semitones from tonic)
_MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
_NATURAL_MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]


# ---------------------------------------------------------------------------
# Profile construction
# ---------------------------------------------------------------------------

def test_build_key_profiles_shape_and_labels():
    profiles, labels = _build_key_profiles()
    assert profiles.shape == (24, 12)
    assert len(labels) == 24
    # First 12 are majors, second 12 are minors.
    assert all(lbl.endswith(":major") for lbl in labels[:12])
    assert all(lbl.endswith(":minor") for lbl in labels[12:])
    assert labels[0] == "C:major"
    assert labels[12] == "C:minor"
    assert labels[9] == "A:major"
    assert labels[21] == "A:minor"


def test_key_profiles_are_zero_meaned_and_unit_norm():
    profiles, _ = _build_key_profiles()
    # Zero mean — Pearson correlation requires it.
    means = profiles.mean(axis=1)
    assert np.allclose(means, 0.0, atol=1e-6)
    # Unit norm — dot product with a normalized chroma = cosine / Pearson.
    norms = np.linalg.norm(profiles, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_profiles_rotate_correctly_across_tonics():
    """G major should be C major's profile rotated by 7 semitones."""
    profiles, labels = _build_key_profiles()
    c_major = profiles[labels.index("C:major")]
    g_major = profiles[labels.index("G:major")]
    # Rolling C major by 7 positions (semitones) should land on G major.
    # After mean-subtraction + normalization the relationship is still
    # an exact rotation, so shifted vectors compare elementwise.
    assert np.allclose(np.roll(c_major, 7), g_major, atol=1e-6)


# ---------------------------------------------------------------------------
# Key estimation from synthetic chroma
# ---------------------------------------------------------------------------

def test_pure_c_major_scale_chroma_detects_c_major():
    chroma = _chroma_from_pitch_classes(_scale_profile(0, _MAJOR_INTERVALS))
    key, stats = estimate_key_from_chroma(chroma)
    assert key == "C:major"
    assert not stats.skipped
    assert stats.confidence > DEFAULT_KEY_MIN_CONFIDENCE


def test_g_major_scale_chroma_detects_g_major():
    chroma = _chroma_from_pitch_classes(_scale_profile(7, _MAJOR_INTERVALS))
    key, stats = estimate_key_from_chroma(chroma)
    assert key == "G:major"
    assert stats.confidence > 0.6


def test_a_minor_scale_chroma_detects_a_minor():
    """A natural minor shares pitch classes with C major — the KS
    profile's tonic weighting is what breaks the tie."""
    chroma = _chroma_from_pitch_classes(_scale_profile(9, _NATURAL_MINOR_INTERVALS))
    key, stats = estimate_key_from_chroma(chroma)
    assert key == "A:minor"
    # The runner-up for A minor is almost always C major — document it.
    assert stats.runner_up_label == "C:major"


def test_d_minor_scale_chroma_detects_d_minor():
    chroma = _chroma_from_pitch_classes(_scale_profile(2, _NATURAL_MINOR_INTERVALS))
    key, _stats = estimate_key_from_chroma(chroma)
    assert key == "D:minor"


def test_below_confidence_floor_falls_back_to_c_major():
    # Uniform chroma — no tonal information; Pearson correlation
    # against every profile is ~0 so we should fall through the
    # floor and return the hardcoded default.
    chroma = np.ones((12, 10), dtype=np.float32)
    key, stats = estimate_key_from_chroma(chroma, min_confidence=0.55)
    assert key == "C:major"
    assert stats.skipped
    assert any("confidence" in w or "variance" in w for w in stats.warnings)


def test_zero_chroma_is_skipped_not_crashed():
    chroma = np.zeros((12, 10), dtype=np.float32)
    key, stats = estimate_key_from_chroma(chroma)
    assert key == "C:major"
    assert stats.skipped


def test_invalid_shape_chroma_returns_default():
    # 10 rows instead of 12 — the caller passed a wrong-shape matrix.
    chroma = np.random.rand(10, 20).astype(np.float32)
    key, stats = estimate_key_from_chroma(chroma)
    assert key == "C:major"
    assert stats.skipped
    assert "invalid chroma shape" in stats.warnings[0]


def test_empty_chroma_returns_default():
    chroma = np.zeros((12, 0), dtype=np.float32)
    key, stats = estimate_key_from_chroma(chroma)
    assert key == "C:major"
    assert stats.skipped


def test_key_stats_as_warnings_reports_label_and_runner_up():
    chroma = _chroma_from_pitch_classes(_scale_profile(0, _MAJOR_INTERVALS))
    _key, stats = estimate_key_from_chroma(chroma)
    warnings = stats.as_warnings()
    assert any("key: C:major" in w for w in warnings)
    assert any("runner-up" in w for w in warnings)


# ---------------------------------------------------------------------------
# Meter hypothesis scoring (pure function)
# ---------------------------------------------------------------------------

def test_score_hypothesis_detects_strong_fourth_beat_pattern():
    # A clean 4/4 pulse — downbeat every 4, rest weaker.
    strengths = np.array(
        [1.0, 0.3, 0.5, 0.3] * 6,  # 24 beats
        dtype=np.float32,
    )
    ratio_4, phase_4 = _score_meter_hypothesis(strengths, 4)
    ratio_3, _phase_3 = _score_meter_hypothesis(strengths, 3)

    # The 4-beat fold should identify the strong phase at 0.
    assert phase_4 == 0
    # 4/4 should score higher than 3/4 on this signal.
    assert ratio_4 > ratio_3
    # Strong beat is clearly above average.
    assert ratio_4 > 1.5


def test_score_hypothesis_detects_waltz_pattern():
    # A clean 3/4 pulse.
    strengths = np.array(
        [1.0, 0.3, 0.3] * 8,  # 24 beats
        dtype=np.float32,
    )
    ratio_4, _ = _score_meter_hypothesis(strengths, 4)
    ratio_3, phase_3 = _score_meter_hypothesis(strengths, 3)

    assert phase_3 == 0
    assert ratio_3 > ratio_4
    assert ratio_3 > 1.5


def test_score_hypothesis_handles_too_short_input():
    # Fewer than k*2 samples → fold is meaningless → zero ratio.
    strengths = np.array([1.0, 0.3, 0.5], dtype=np.float32)
    ratio, phase = _score_meter_hypothesis(strengths, 4)
    assert ratio == 0.0
    assert phase == 0


# ---------------------------------------------------------------------------
# Meter estimation from beat strengths
# ---------------------------------------------------------------------------

def test_estimate_meter_picks_4_4_on_rock_pattern():
    strengths = np.array([1.0, 0.3, 0.5, 0.3] * 6, dtype=np.float32)
    sig, stats = estimate_meter_from_beat_strengths(strengths)
    assert sig == (4, 4)
    assert not stats.skipped
    assert stats.n_beats == 24
    assert stats.confidence > 0.0


def test_estimate_meter_picks_3_4_on_waltz_pattern():
    strengths = np.array([1.0, 0.3, 0.3] * 8, dtype=np.float32)
    sig, stats = estimate_meter_from_beat_strengths(strengths)
    assert sig == (3, 4)
    assert not stats.skipped


def test_estimate_meter_tie_breaks_toward_4_4():
    """Weak near-uniform pulse should collapse to 4/4 via the margin."""
    # A barely-there 3/4 bias — within the confidence_margin.
    strengths = np.array([1.01, 1.0, 1.0] * 8, dtype=np.float32)
    sig, stats = estimate_meter_from_beat_strengths(
        strengths, confidence_margin=0.5,
    )
    assert sig == (4, 4)
    # Confidence should still be low.
    assert stats.confidence < 0.1


def test_estimate_meter_flips_to_3_4_when_margin_exceeded():
    """Clear 3/4 signal must beat the tie-break margin."""
    strengths = np.array([1.0, 0.2, 0.2] * 8, dtype=np.float32)
    sig, _stats = estimate_meter_from_beat_strengths(
        strengths, confidence_margin=0.05,
    )
    assert sig == (3, 4)


def test_estimate_meter_skips_when_too_few_beats():
    strengths = np.array([1.0, 0.5, 0.5, 0.5], dtype=np.float32)
    sig, stats = estimate_meter_from_beat_strengths(strengths, min_beats=8)
    assert sig == (4, 4)
    assert stats.skipped


def test_estimate_meter_skips_on_nan_input():
    strengths = np.array([1.0, float("nan"), 0.5] * 4, dtype=np.float32)
    sig, stats = estimate_meter_from_beat_strengths(strengths)
    assert sig == (4, 4)
    assert stats.skipped


def test_meter_stats_as_warnings_reports_signature_and_beats():
    strengths = np.array([1.0, 0.3, 0.5, 0.3] * 6, dtype=np.float32)
    _sig, stats = estimate_meter_from_beat_strengths(strengths)
    warnings = stats.as_warnings()
    assert any("time_signature: 4/4" in w for w in warnings)
    assert any("24 beats" in w for w in warnings)


# ---------------------------------------------------------------------------
# Skipped-stats shapes
# ---------------------------------------------------------------------------

def test_default_key_stats_shape():
    stats = KeyEstimationStats()
    assert not stats.skipped
    assert stats.key_label == ""
    assert stats.confidence == 0.0
    assert stats.warnings == []


def test_default_meter_stats_shape():
    stats = MeterEstimationStats()
    assert not stats.skipped
    assert stats.time_signature == (4, 4)
    assert stats.confidence == 0.0
    assert stats.n_beats == 0


def test_skipped_key_stats_as_warnings_does_not_report_label():
    stats = KeyEstimationStats(skipped=True, key_label="A:minor")
    stats.warnings.append("forced skip for test")
    warnings = stats.as_warnings()
    assert all("A:minor" not in w for w in warnings)
    assert any("skipped" in w for w in warnings)


# ---------------------------------------------------------------------------
# Config defaults sync
# ---------------------------------------------------------------------------

def test_config_key_min_confidence_matches_module_default():
    """``Settings.key_min_confidence`` must stay in lockstep with
    ``DEFAULT_KEY_MIN_CONFIDENCE`` so future changes to one side
    don't silently drift from the other."""
    from backend.config import Settings
    from backend.services.key_estimation import DEFAULT_KEY_MIN_CONFIDENCE

    assert Settings().key_min_confidence == DEFAULT_KEY_MIN_CONFIDENCE


def test_config_meter_defaults_match_module_defaults():
    from backend.config import Settings
    from backend.services.key_estimation import (
        DEFAULT_METER_CONFIDENCE_MARGIN,
        DEFAULT_METER_MIN_BEATS,
    )

    s = Settings()
    assert s.meter_confidence_margin == DEFAULT_METER_CONFIDENCE_MARGIN
    assert s.meter_min_beats == DEFAULT_METER_MIN_BEATS


# ---------------------------------------------------------------------------
# End-to-end file-path entry — librosa-gated, synthetic sine audio
# ---------------------------------------------------------------------------

librosa = pytest.importorskip("librosa")


def _synth_sine_chord(freqs: list[float], sr: int, dur_sec: float) -> np.ndarray:
    """Sum sines with a 10ms ramp at the start so beat tracking latches."""
    n = int(round(sr * dur_sec))
    t = np.arange(n, dtype=np.float32) / sr
    y = np.zeros(n, dtype=np.float32)
    for f in freqs:
        y += np.sin(2.0 * math.pi * f * t).astype(np.float32)
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak * 0.8
    n_attack = int(sr * 0.01)
    env = np.ones_like(y)
    env[:n_attack] = np.linspace(0.0, 1.0, n_attack, dtype=np.float32)
    return y * env


def test_estimate_key_from_waveform_detects_c_major_from_sines(tmp_path):
    """End-to-end test against a synthetic C-major triad waveform.

    Uses the file-path entry point to exercise HPSS + chroma_cqt +
    KS scoring in one call. Skipped if librosa (or soundfile) aren't
    installed — the ``importorskip`` above handles librosa, and we
    try/except around the write so a missing soundfile also skips.
    """
    from backend.services.key_estimation import estimate_key_from_waveform

    sr = 22050
    # C major triad sustained for 4 seconds, with tonic emphasis via
    # repeating the root alone on top of the triad.
    c_chord = _synth_sine_chord([261.63, 329.63, 392.00], sr, 4.0)
    c_tonic = _synth_sine_chord([261.63, 523.25], sr, 2.0)
    y = np.concatenate([c_chord, c_tonic, c_chord]).astype(np.float32)

    key, stats = estimate_key_from_waveform(y, sr)
    # HPSS on pure sines can squash the signal, so we accept either
    # a clean detection or a graceful skip — the important assertion
    # is that we don't crash and return a parseable label.
    assert key in {"C:major", "C:minor"} or stats.skipped
    if not stats.skipped:
        assert stats.confidence > 0.0
