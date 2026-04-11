"""Unit tests for onset refinement via spectral onset-strength peaks.

Tests mock specific librosa / scipy functions rather than the import
machinery, since both packages are available in the test environment.
The mocks isolate the onset_refine logic from real audio I/O and signal
processing.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from backend.services.onset_refine import OnsetRefineStats, refine_onsets

try:
    import librosa  # noqa: F401
    _has_librosa = True
except ImportError:
    _has_librosa = False

needs_librosa = pytest.mark.skipif(not _has_librosa, reason="librosa not installed")


# Shorthand: (start, end, pitch, amplitude, pitch_bends)
def _n(start: float, end: float, pitch: int, amp: float, bends=None):
    return (start, end, pitch, amp, bends)


# ---------------------------------------------------------------------------
# Shared mock setup
# ---------------------------------------------------------------------------

def _mock_deps(odf_values, odf_times, peak_indices):
    """Return a dict of patches for librosa + scipy functions.

    ``odf_values``: array-like onset strength values.
    ``odf_times``:  array-like time axis for the ODF.
    ``peak_indices``: list of int indices into odf_values where peaks are.
    """
    odf_arr = np.array(odf_values, dtype=float)
    times_arr = np.array(odf_times, dtype=float)
    peaks_arr = np.array(peak_indices, dtype=int)

    def _mock_load(path, sr=None, mono=True):
        return np.zeros(22050), 22050

    def _mock_onset_strength(y=None, sr=None, hop_length=None):
        return odf_arr

    def _mock_frames_to_time(frames, sr=None, hop_length=None):
        return times_arr

    def _mock_find_peaks(x, height=None):
        return peaks_arr, {}

    return {
        "librosa.load": _mock_load,
        "librosa.onset.onset_strength": _mock_onset_strength,
        "librosa.frames_to_time": _mock_frames_to_time,
        "scipy.signal.find_peaks": _mock_find_peaks,
    }


class _PatchContext:
    """Context manager that patches librosa/scipy inside onset_refine."""

    def __init__(self, odf_values, odf_times, peak_indices):
        self.mocks = _mock_deps(odf_values, odf_times, peak_indices)
        self._patches = []

    def __enter__(self):
        # Patch the functions as they're called inside onset_refine.
        # The module does `import librosa` then calls `librosa.load(...)` etc.
        # We patch at the librosa/scipy module level.
        import librosa
        import scipy.signal

        self._patches = [
            patch.object(librosa, "load", side_effect=self.mocks["librosa.load"]),
            patch.object(librosa.onset, "onset_strength", side_effect=self.mocks["librosa.onset.onset_strength"]),
            patch.object(librosa, "frames_to_time", side_effect=self.mocks["librosa.frames_to_time"]),
            patch.object(scipy.signal, "find_peaks", side_effect=self.mocks["scipy.signal.find_peaks"]),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()


# ---------------------------------------------------------------------------
# Tests: onsets near a spectral peak get shifted
# ---------------------------------------------------------------------------

@needs_librosa
def test_onset_near_peak_gets_shifted(tmp_path):
    """A note whose onset is close to an ODF peak should snap to the peak."""
    odf_times = [i * 0.1 for i in range(20)]
    odf_values = [0.1] * 20
    odf_values[10] = 0.9  # big peak at t=1.0

    audio = tmp_path / "test.wav"
    audio.write_bytes(b"fake")

    events = [_n(1.02, 2.0, 60, 0.8)]  # onset at 1.02, peak at 1.00

    with _PatchContext(odf_values, odf_times, peak_indices=[10]):
        refined, stats = refine_onsets(events, audio, max_shift_sec=0.05)

    assert stats.refined_count == 1
    assert refined[0][0] == pytest.approx(1.0, abs=1e-6)  # snapped to peak
    assert refined[0][1] == 2.0  # end unchanged
    assert stats.mean_shift_sec == pytest.approx(0.02, abs=1e-3)


@needs_librosa
def test_onset_far_from_peak_stays_unchanged(tmp_path):
    """A note whose onset is far from any peak should keep its original onset."""
    odf_times = [i * 0.1 for i in range(20)]
    odf_values = [0.1] * 20
    odf_values[10] = 0.9  # peak at t=1.0

    audio = tmp_path / "test.wav"
    audio.write_bytes(b"fake")

    events = [_n(0.5, 1.5, 60, 0.8)]  # onset at 0.5, peak at 1.0 — distance=0.5 >> max_shift

    with _PatchContext(odf_values, odf_times, peak_indices=[10]):
        refined, stats = refine_onsets(events, audio, max_shift_sec=0.05)

    assert stats.refined_count == 0
    assert refined[0][0] == 0.5  # unchanged


@needs_librosa
def test_max_shift_sec_respected(tmp_path):
    """Even if a peak is the nearest, it must be within max_shift_sec."""
    odf_times = [i * 0.1 for i in range(20)]
    odf_values = [0.1] * 20
    odf_values[10] = 0.9  # peak at t=1.0

    audio = tmp_path / "test.wav"
    audio.write_bytes(b"fake")

    # onset at 0.93, peak at 1.0 — distance = 0.07, max_shift = 0.05 -> too far
    events = [_n(0.93, 2.0, 60, 0.8)]

    with _PatchContext(odf_values, odf_times, peak_indices=[10]):
        refined, stats = refine_onsets(events, audio, max_shift_sec=0.05)

    assert stats.refined_count == 0
    assert refined[0][0] == 0.93  # unchanged

    # Same event but with max_shift=0.10 -> peak is reachable
    with _PatchContext(odf_values, odf_times, peak_indices=[10]):
        refined2, stats2 = refine_onsets(events, audio, max_shift_sec=0.10)

    assert stats2.refined_count == 1
    assert refined2[0][0] == pytest.approx(1.0, abs=1e-6)


@needs_librosa
def test_onset_never_shifts_past_note_end(tmp_path):
    """The refined onset must never be >= end - 0.01."""
    odf_times = [i * 0.1 for i in range(20)]
    odf_values = [0.1] * 20
    odf_values[10] = 0.9  # peak at t=1.0

    audio = tmp_path / "test.wav"
    audio.write_bytes(b"fake")

    # onset at 0.98, end at 1.005. Peak at 1.0 is within max_shift, but
    # clamping to end - 0.01 = 0.995 should prevent it from reaching 1.0.
    events = [_n(0.98, 1.005, 60, 0.8)]

    with _PatchContext(odf_values, odf_times, peak_indices=[10]):
        refined, stats = refine_onsets(events, audio, max_shift_sec=0.05)

    # The onset should be clamped to end - 0.01 = 0.995, not 1.0
    assert refined[0][0] <= refined[0][1] - 0.01


def test_empty_events_returns_empty(tmp_path):
    """An empty event list should return empty with zero stats."""
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"fake")

    refined, stats = refine_onsets([], audio)
    assert refined == []
    assert stats.total_notes == 0
    assert stats.refined_count == 0


def test_missing_audio_returns_events_unchanged(tmp_path):
    """When the audio file doesn't exist, events should pass through unchanged."""
    events = [_n(1.0, 2.0, 60, 0.8)]
    missing = tmp_path / "nonexistent.wav"

    refined, stats = refine_onsets(events, missing)

    assert stats.skipped is True
    assert len(refined) == 1
    assert refined[0][0] == 1.0  # unchanged
    assert "not found" in stats.warnings[0]


def test_missing_librosa_returns_events_unchanged(tmp_path):
    """When librosa import fails, events should pass through unchanged."""
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"fake")

    events = [_n(1.0, 2.0, 60, 0.8)]

    # Temporarily hide librosa from sys.modules to force ImportError
    import sys
    saved = sys.modules.get("librosa")
    sys.modules["librosa"] = None  # type: ignore[assignment]
    try:
        # Also need to make sure the import fails
        with patch.dict("sys.modules", {"librosa": None}):
            refined, stats = refine_onsets(events, audio)
    finally:
        if saved is not None:
            sys.modules["librosa"] = saved

    assert stats.skipped is True
    assert len(refined) == 1
    assert refined[0][0] == 1.0


@needs_librosa
def test_multiple_notes_refined_independently(tmp_path):
    """Each note finds its own nearest peak independently."""
    # Two peaks: at t=0.5 (index 5) and t=1.5 (index 15)
    odf_times = [i * 0.1 for i in range(20)]
    odf_values = [0.1] * 20
    odf_values[5] = 0.9   # peak at t=0.5
    odf_values[15] = 0.9  # peak at t=1.5

    audio = tmp_path / "test.wav"
    audio.write_bytes(b"fake")

    events = [
        _n(0.52, 1.0, 60, 0.8),  # near peak at 0.5
        _n(1.48, 2.5, 64, 0.7),  # near peak at 1.5
        _n(0.90, 1.2, 67, 0.6),  # far from both peaks -> unchanged
    ]

    with _PatchContext(odf_values, odf_times, peak_indices=[5, 15]):
        refined, stats = refine_onsets(events, audio, max_shift_sec=0.05)

    assert stats.refined_count == 2
    assert refined[0][0] == pytest.approx(0.5, abs=1e-6)   # snapped to 0.5
    assert refined[1][0] == pytest.approx(1.5, abs=1e-6)   # snapped to 1.5
    assert refined[2][0] == 0.90                            # unchanged


def test_stats_dataclass_as_warnings():
    """OnsetRefineStats.as_warnings() formats human-readable summary lines."""
    stats = OnsetRefineStats(
        total_notes=100,
        refined_count=25,
        mean_shift_sec=0.012,
        max_shift_sec=0.045,
    )
    warnings = stats.as_warnings()
    assert len(warnings) == 1
    assert "25/100" in warnings[0]
    assert "12.0ms" in warnings[0]
    assert "45.0ms" in warnings[0]


def test_stats_skipped_as_warnings():
    """When skipped, as_warnings() returns the skip reason."""
    stats = OnsetRefineStats(skipped=True)
    stats.warnings.append("onset-refine: skipped — missing dep")
    warnings = stats.as_warnings()
    assert len(warnings) == 1
    assert "skipped" in warnings[0]


@needs_librosa
def test_no_peaks_found_returns_unchanged(tmp_path):
    """When the ODF has no peaks above median, events pass through."""
    odf_times = [i * 0.1 for i in range(20)]
    odf_values = [0.1] * 20  # flat — no peaks

    audio = tmp_path / "test.wav"
    audio.write_bytes(b"fake")

    events = [_n(1.0, 2.0, 60, 0.8)]

    # find_peaks returns empty with flat data
    with _PatchContext(odf_values, odf_times, peak_indices=[]):
        refined, stats = refine_onsets(events, audio, max_shift_sec=0.05)

    assert stats.skipped is True
    assert refined[0][0] == 1.0  # unchanged
    assert "no ODF peaks" in stats.warnings[0]
