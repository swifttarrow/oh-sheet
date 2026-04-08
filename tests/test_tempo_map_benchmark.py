"""Benchmark-style tests: beat-derived tempo map vs wrong constant BPM.

These are **synthetic** checks (known ground-truth tempo). They do not prove
quality on real songs without labeled data; they show the *measurable* gap
when the pipeline assumes the wrong global BPM instead of beat-aligned maps.

Run the printable report: ``python scripts/benchmark_tempo_map.py`` (repo root).
"""
from __future__ import annotations

import wave

import pytest

from backend.contracts import TempoMapEntry, sec_to_beat
from backend.services.audio_timing import build_tempo_map_from_beat_times, tempo_map_from_audio_path
from tests.tempo_map_benchmark_metrics import (
    linspace,
    max_abs_beat_error,
    mean_abs_beat_error,
)


def _sample_grid(t0: float, t1: float, n: int = 200) -> list[float]:
    return linspace(t0, t1, n)


def test_wrong_constant_bpm_drifts_from_truth():
    """Old-style single map with incorrect BPM accumulates beat error over time."""
    true_bpm = 100.0
    wrong_map = [TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)]
    times = _sample_grid(0.05, 9.95, 200)
    mae = mean_abs_beat_error(times, wrong_map, true_bpm=true_bpm)
    mx = max_abs_beat_error(times, wrong_map, true_bpm=true_bpm)
    # By ~10 s, error should be multiple beats (roughly (2 - 5/3)*10 ≈ 3.33 beats max direction)
    assert mae > 1.0, f"expected large MAE in beats, got {mae}"
    assert mx > 2.5, f"expected large max error in beats, got {mx}"


def test_oracle_beat_times_negligible_error():
    """Piecewise map from exact beat instants matches constant-tempo truth."""
    true_bpm = 100.0
    period = 60.0 / true_bpm
    beats = [i * period for i in range(40)]  # 0 .. ~23.4 s
    oracle_map = build_tempo_map_from_beat_times(
        beats, duration_sec=beats[-1] + period, fallback_bpm=true_bpm
    )
    times = _sample_grid(0.05, 20.0, 300)
    mae = mean_abs_beat_error(times, oracle_map, true_bpm=true_bpm)
    assert mae < 0.02, f"oracle beat map MAE should be tiny, got {mae}"


def test_variable_tempo_beat_map_beats_single_wrong_constant():
    """Two tempo regions: piecewise map vs one wrong global BPM.

    Ground truth is the **discrete** beat grid (integer beat indices on known
    beat times), not a continuous ``t * bpm`` curve — that matches how the
    pipeline indexes bars after a tempo change.
    """
    bpm_a, bpm_b = 80.0, 120.0
    pa, pb = 60.0 / bpm_a, 60.0 / bpm_b
    beat_times: list[float] = []
    t = 0.0
    while t < 4.0 - 1e-9:
        beat_times.append(t)
        t += pa
    t = 4.0
    while t <= 12.0 + 1e-9:
        beat_times.append(t)
        t += pb

    piecewise = build_tempo_map_from_beat_times(
        beat_times, duration_sec=12.0, fallback_bpm=bpm_b
    )

    def true_beat_at(time_sec: float) -> float:
        if time_sec <= beat_times[0]:
            return 0.0
        for i in range(len(beat_times) - 1):
            t0, t1 = beat_times[i], beat_times[i + 1]
            if t0 <= time_sec <= t1:
                return float(i) + (time_sec - t0) / (t1 - t0)
        # past last anchor: extrapolate with tail segment tempo
        dt = max(beat_times[-1] - beat_times[-2], 1e-3)
        bpm_tail = 60.0 / dt
        last_i = float(len(beat_times) - 1)
        return last_i + (time_sec - beat_times[-1]) * (bpm_tail / 60.0)

    times = _sample_grid(0.05, 11.95, 250)
    err_piece = sum(abs(sec_to_beat(t, piecewise) - true_beat_at(t)) for t in times) / len(times)

    wrong_global = [TempoMapEntry(time_sec=0.0, beat=0.0, bpm=100.0)]
    err_wrong = sum(abs(sec_to_beat(t, wrong_global) - true_beat_at(t)) for t in times) / len(times)

    assert err_piece < 0.05, f"piecewise oracle MAE {err_piece}"
    assert err_wrong > err_piece * 2.0, "wrong global map should be worse than oracle piecewise"


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("librosa") is None,
    reason="librosa not installed (optional; install ohsheet[mt3] or librosa)",
)
def test_librosa_recovers_synthetic_click_tempo(tmp_path):
    """Synthetic click train: librosa beat map closer to truth than wrong constant."""
    import numpy as np

    true_bpm = 108.0
    sr = 22_050
    duration = 8.0
    period = int(sr * 60.0 / true_bpm)
    n_samples = int(sr * duration)
    y = np.zeros(n_samples, dtype=np.float32)
    for i in range(0, n_samples, period):
        end = min(i + 2000, n_samples)
        y[i:end] += 0.8 * np.hanning(end - i).astype(np.float32)

    wav = tmp_path / "clicks.wav"
    audio_i16 = (y * 32767).astype(np.int16)
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_i16.tobytes())

    recovered = tempo_map_from_audio_path(wav, sr=sr)
    assert recovered is not None, "librosa should return a map for synthetic clicks"

    times = _sample_grid(0.25, 7.5, 120)
    mae_rec = mean_abs_beat_error(times, recovered, true_bpm=true_bpm)
    wrong = [TempoMapEntry(time_sec=0.0, beat=0.0, bpm=72.0)]
    mae_wrong = mean_abs_beat_error(times, wrong, true_bpm=true_bpm)

    assert mae_rec < mae_wrong, (
        f"recovered map should beat wrong constant: mae_rec={mae_rec}, mae_wrong={mae_wrong}"
    )
    assert mae_rec < 0.35, f"librosa map should be reasonably tight, mae={mae_rec} beats"
