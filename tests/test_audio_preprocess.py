"""Unit tests for the audio pre-processing stage.

Drives :mod:`backend.services.audio_preprocess` with synthetic
sine + percussive impulse waveforms so we can verify HPSS actually
strips transients and RMS normalization hits its target. Tests skip
gracefully when librosa is missing — mirroring the chord / melody /
bass test modules.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
librosa = pytest.importorskip("librosa")
soundfile = pytest.importorskip("soundfile")

from backend.services.audio_preprocess import (  # noqa: E402
    DEFAULT_PEAK_CEILING_DBFS,
    DEFAULT_TARGET_RMS_DBFS,
    PreprocessStats,
    _peak_dbfs,
    _rms_dbfs,
    preprocess_audio_file,
    preprocess_waveform,
)

SR = 22050


# ---------------------------------------------------------------------------
# Synthetic waveform helpers
# ---------------------------------------------------------------------------

def _sine(freq_hz: float, duration_sec: float, amp: float = 0.3) -> np.ndarray:
    n = int(round(SR * duration_sec))
    t = np.arange(n, dtype=np.float32) / SR
    return (amp * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def _impulse_train(duration_sec: float, period_sec: float, amp: float = 0.8) -> np.ndarray:
    """A sparse impulse train — pure percussive content for HPSS to strip."""
    n = int(round(SR * duration_sec))
    y = np.zeros(n, dtype=np.float32)
    step = max(1, int(round(SR * period_sec)))
    y[::step] = amp
    return y


def _mixed_signal(duration_sec: float = 2.0) -> np.ndarray:
    """Sine (harmonic) + impulse train (percussive) — the canonical HPSS fixture."""
    harmonic = _sine(440.0, duration_sec, amp=0.2)
    percussive = _impulse_train(duration_sec, period_sec=0.25, amp=0.6)
    return (harmonic + percussive).astype(np.float32)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def test_rms_dbfs_silence_is_negative_infinity():
    y = np.zeros(SR, dtype=np.float32)
    assert _rms_dbfs(y) == float("-inf")


def test_rms_dbfs_full_scale_is_zero():
    y = np.ones(SR, dtype=np.float32)  # all-ones → RMS = 1.0 = 0 dBFS
    assert _rms_dbfs(y) == pytest.approx(0.0, abs=1e-6)


def test_peak_dbfs_matches_max_abs():
    y = np.array([0.0, 0.5, -0.25, 0.1], dtype=np.float32)
    assert _peak_dbfs(y) == pytest.approx(20.0 * math.log10(0.5), abs=1e-6)


# ---------------------------------------------------------------------------
# preprocess_waveform — core logic on in-memory arrays
# ---------------------------------------------------------------------------

def test_preprocess_waveform_skipped_when_both_passes_disabled():
    y = _mixed_signal()
    out, stats = preprocess_waveform(
        y, SR, hpss_enabled=False, normalize_enabled=False,
    )
    assert stats.skipped
    assert not stats.hpss_applied
    assert not stats.normalize_applied
    # Signal passes through unchanged.
    np.testing.assert_array_equal(out, y)


def test_preprocess_waveform_skips_tiny_input():
    y = np.zeros(100, dtype=np.float32)  # ~4.5 ms at 22050 Hz
    _, stats = preprocess_waveform(y, SR)
    assert stats.skipped
    assert any("too short" in w for w in stats.warnings)


def test_preprocess_waveform_empty_input():
    y = np.array([], dtype=np.float32)
    _, stats = preprocess_waveform(y, SR)
    assert stats.skipped


def test_preprocess_waveform_hpss_reduces_percussive_energy():
    """HPSS should drop the impulse-train component by a large margin.

    We sum sine + impulse-train and verify the peak of the processed
    signal is dramatically lower than the peak of the raw signal. A
    sine-only waveform has peak ≈ amplitude (0.2); the raw mixed signal
    has peak ≈ 0.6 (the impulse). After HPSS the impulses should be
    gone and the peak should be back in sine-only territory.
    """
    y = _mixed_signal(duration_sec=2.0)
    raw_peak = float(np.max(np.abs(y)))
    assert raw_peak > 0.5, "fixture should have clear percussive spikes"

    # Run HPSS only (no normalize) so we can compare absolute peaks.
    out, stats = preprocess_waveform(
        y, SR,
        hpss_enabled=True,
        normalize_enabled=False,
        hpss_margin=3.0,  # aggressive enough to matter on a synthetic fixture
    )
    assert stats.hpss_applied
    out_peak = float(np.max(np.abs(out)))
    # Expect a meaningful reduction — impulses should be largely gone.
    assert out_peak < raw_peak * 0.7, (
        f"HPSS did not strip transients: raw peak {raw_peak:.3f}, out peak {out_peak:.3f}"
    )


def test_preprocess_waveform_normalize_hits_target_rms():
    # Quiet sine → normalize should amplify to ~target RMS.
    y = _sine(440.0, 2.0, amp=0.01)  # ~-43 dBFS RMS
    _, stats = preprocess_waveform(
        y, SR,
        hpss_enabled=False,
        normalize_enabled=True,
        target_rms_dbfs=-20.0,
        peak_ceiling_dbfs=-1.0,
    )
    assert stats.normalize_applied
    assert stats.output_rms_dbfs is not None
    # Target is -20 dBFS; allow a small margin for float precision.
    assert stats.output_rms_dbfs == pytest.approx(-20.0, abs=0.5)
    # Input was quieter than output — gain was positive.
    assert stats.input_rms_dbfs is not None
    assert stats.output_rms_dbfs > stats.input_rms_dbfs


def test_preprocess_waveform_normalize_honors_peak_ceiling():
    """A loud sine should be capped by the peak ceiling, not lifted to target RMS."""
    # Already-hot input: peak at 0.95 (~ -0.45 dBFS), RMS ~ -3 dBFS.
    # Asking for -3 dBFS RMS but with a -6 dBFS peak ceiling should
    # *reduce* the signal — the ceiling wins.
    y = _sine(440.0, 2.0, amp=0.95)
    _, stats = preprocess_waveform(
        y, SR,
        hpss_enabled=False,
        normalize_enabled=True,
        target_rms_dbfs=-3.0,
        peak_ceiling_dbfs=-6.0,
    )
    assert stats.normalize_applied
    assert stats.output_peak_dbfs is not None
    # Peak is at or below the ceiling (with a small tolerance).
    assert stats.output_peak_dbfs <= -6.0 + 0.5


def test_preprocess_waveform_normalize_skips_silence():
    y = np.zeros(int(SR * 1.0), dtype=np.float32)
    _, stats = preprocess_waveform(
        y, SR,
        hpss_enabled=False,
        normalize_enabled=True,
    )
    assert not stats.normalize_applied
    assert any("silent" in w for w in stats.warnings)


def test_preprocess_waveform_records_input_output_levels():
    y = _sine(440.0, 1.5, amp=0.1)
    _, stats = preprocess_waveform(y, SR)
    assert stats.input_rms_dbfs is not None
    assert stats.output_rms_dbfs is not None
    assert stats.input_peak_dbfs is not None
    assert stats.output_peak_dbfs is not None
    assert math.isfinite(stats.input_rms_dbfs)
    assert math.isfinite(stats.output_rms_dbfs)


# ---------------------------------------------------------------------------
# PreprocessStats.as_warnings formatting
# ---------------------------------------------------------------------------

def test_as_warnings_skipped_includes_reason():
    stats = PreprocessStats(skipped=True, warnings=["load failed: bad bytes"])
    msgs = stats.as_warnings()
    assert any("audio preprocess skipped" in m for m in msgs)
    assert any("bad bytes" in m for m in msgs)


def test_as_warnings_skipped_no_reason():
    stats = PreprocessStats(skipped=True)
    msgs = stats.as_warnings()
    assert msgs == ["audio preprocess skipped"]


def test_as_warnings_active_includes_rms_delta():
    stats = PreprocessStats(
        hpss_applied=True,
        normalize_applied=True,
        input_rms_dbfs=-35.0,
        output_rms_dbfs=-20.0,
        input_peak_dbfs=-20.0,
        output_peak_dbfs=-5.0,
    )
    msgs = stats.as_warnings()
    joined = " ".join(msgs)
    assert "hpss" in joined
    assert "rms" in joined
    assert "-35.0" in joined
    assert "-20.0" in joined


# ---------------------------------------------------------------------------
# preprocess_audio_file — file-path entry point
# ---------------------------------------------------------------------------

def test_preprocess_audio_file_roundtrips_through_disk(tmp_path: Path):
    # Write a fixture WAV to disk, run the file-path entry point, then
    # load the output and check RMS.
    y = _sine(440.0, 1.5, amp=0.02)
    src = tmp_path / "quiet.wav"
    soundfile.write(str(src), y, SR, subtype="FLOAT")

    out_path, stats = preprocess_audio_file(
        src,
        hpss_enabled=False,   # isolate the normalization path for this test
        normalize_enabled=True,
        target_rms_dbfs=-20.0,
    )
    assert not stats.skipped
    assert stats.normalize_applied
    # A new file was written (not the original path).
    assert out_path != src
    assert out_path.exists()

    # Load back and verify the output hit target RMS within tolerance.
    out_y, out_sr = soundfile.read(str(out_path), dtype="float32")
    assert out_sr == SR
    rms = float(np.sqrt(np.mean(np.square(out_y, dtype=np.float64))))
    measured_dbfs = 20.0 * math.log10(rms)
    assert measured_dbfs == pytest.approx(-20.0, abs=0.5)

    # Caller owns cleanup.
    out_path.unlink()


def test_preprocess_audio_file_returns_original_on_skipped(tmp_path: Path):
    y = _sine(440.0, 1.5, amp=0.1)
    src = tmp_path / "in.wav"
    soundfile.write(str(src), y, SR, subtype="FLOAT")

    out_path, stats = preprocess_audio_file(
        src, hpss_enabled=False, normalize_enabled=False,
    )
    assert stats.skipped
    # No temp file written — returns the input path unchanged so the
    # caller's "unlink if different" check is correct.
    assert out_path == src


def test_preprocess_audio_file_skips_unreadable_file(tmp_path: Path):
    bogus = tmp_path / "not-audio.wav"
    bogus.write_bytes(b"\x00\x01\x02not a wav file at all")

    out_path, stats = preprocess_audio_file(bogus)
    assert stats.skipped
    assert out_path == bogus
    assert any("load failed" in w for w in stats.warnings)


def test_preprocess_audio_file_missing_file(tmp_path: Path):
    ghost = tmp_path / "does-not-exist.wav"
    out_path, stats = preprocess_audio_file(ghost)
    assert stats.skipped
    assert out_path == ghost


# ---------------------------------------------------------------------------
# Config defaults sanity check
# ---------------------------------------------------------------------------

def test_config_defaults_match_module_defaults():
    from backend.config import Settings

    s = Settings()
    # Off by default — the 25-file clean_midi eval showed preprocessing
    # hurts percussion-heavy tracks (Hound Dog -0.088, Beat It -0.125)
    # even though the confidence headline ticks up slightly. HPSS is
    # stripping onset transients Basic Pitch relies on.
    assert s.audio_preprocess_enabled is False
    assert s.audio_preprocess_target_rms_dbfs == DEFAULT_TARGET_RMS_DBFS
    assert s.audio_preprocess_peak_ceiling_dbfs == DEFAULT_PEAK_CEILING_DBFS
