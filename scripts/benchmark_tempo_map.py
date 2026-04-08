#!/usr/bin/env python3
"""Print synthetic tempo-map benchmark numbers (repo root: python scripts/benchmark_tempo_map.py).

Interprets **mean absolute error in beats** between ``sec_to_beat(t, map)`` and
ground truth ``t * true_bpm / 60`` over a time grid. This is **not** a listening
test; it quantifies grid mismatch when the global BPM is wrong vs beat-aligned.
"""
from __future__ import annotations

import os
import sys

# Repo root on sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backend.contracts import TempoMapEntry  # noqa: E402
from backend.services.audio_timing import (  # noqa: E402
    build_tempo_map_from_beat_times,
    tempo_map_from_audio_path,
)
from tests.tempo_map_benchmark_metrics import (  # noqa: E402
    linspace,
    max_abs_beat_error,
    mean_abs_beat_error,
)


def main() -> None:
    true_bpm = 100.0
    times = linspace(0.05, 9.95, 200)

    wrong = [TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)]
    mae_wrong = mean_abs_beat_error(times, wrong, true_bpm=true_bpm)
    max_wrong = max_abs_beat_error(times, wrong, true_bpm=true_bpm)

    period = 60.0 / true_bpm
    oracle_beats = [i * period for i in range(40)]
    oracle_map = build_tempo_map_from_beat_times(
        oracle_beats, duration_sec=oracle_beats[-1] + period, fallback_bpm=true_bpm
    )
    mae_oracle = mean_abs_beat_error(times, oracle_map, true_bpm=true_bpm)
    max_oracle = max_abs_beat_error(times, oracle_map, true_bpm=true_bpm)

    print("Synthetic benchmark: constant true tempo = 100 BPM")
    print(f"  Samples: {len(times)} times from {times[0]:.2f}s to {times[-1]:.2f}s")
    print()
    print(f"  Wrong single map (120 BPM):  MAE = {mae_wrong:.3f} beats, max = {max_wrong:.3f} beats")
    print(f"  Oracle beat-derived map:     MAE = {mae_oracle:.3f} beats, max = {max_oracle:.3f} beats")
    if mae_oracle < 1e-6:
        print("  MAE ratio:                   oracle ~0 (beat grid matches truth exactly on this grid)")
    elif mae_wrong > 0:
        print(f"  MAE ratio (wrong / oracle):  {mae_wrong / mae_oracle:.1f}x")
    print()

    # Optional librosa synthetic file
    try:
        import tempfile
        import wave
        from pathlib import Path

        import numpy as np
    except ImportError:
        print("(NumPy not installed — skip librosa synthetic line.)")
        return

    try:
        import importlib.util

        if importlib.util.find_spec("librosa") is None:
            print("(librosa not installed — skip librosa synthetic line.)")
            return
    except ImportError:
        return

    sr = 22_050
    bpm = 108.0
    duration = 6.0
    period_samp = int(sr * 60.0 / bpm)
    n_samples = int(sr * duration)
    y = np.zeros(n_samples, dtype=np.float32)
    for i in range(0, n_samples, period_samp):
        end = min(i + 2000, n_samples)
        y[i:end] += 0.8 * np.hanning(end - i).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = Path(tf.name)
    try:
        audio_i16 = (y * 32767).astype(np.int16)
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio_i16.tobytes())

        recovered = tempo_map_from_audio_path(wav_path, sr=sr)
        grid = linspace(0.2, 5.5, 100)
        if recovered:
            mae_lib = mean_abs_beat_error(grid, recovered, true_bpm=bpm)
            mae_bad = mean_abs_beat_error(
                grid, [TempoMapEntry(0.0, 0.0, 72.0)], true_bpm=bpm
            )
            print(f"Synthetic click WAV @ ~{bpm} BPM (librosa path):")
            print(f"  Wrong map (72 BPM):   MAE = {mae_bad:.3f} beats")
            print(f"  Librosa-derived map:  MAE = {mae_lib:.3f} beats")
        else:
            print("Librosa did not return a tempo map for synthetic WAV.")
    finally:
        wav_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
