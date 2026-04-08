"""Shared metrics for tempo-map benchmarks (imported by tests and scripts).

Compares ``sec_to_beat`` under a candidate ``tempo_map`` to a **ground-truth**
constant BPM grid: ``beat_truth(t) = t * true_bpm / 60``.

This models the old pipeline shape (one wrong global BPM) vs a map built from
detected beat times (piecewise correct local tempo).
"""
from __future__ import annotations

from collections.abc import Sequence

from backend.contracts import TempoMapEntry, sec_to_beat


def beat_truth_constant_tempo(time_sec: float, true_bpm: float) -> float:
    return time_sec * (true_bpm / 60.0)


def mean_abs_beat_error(
    sample_times_sec: Sequence[float],
    tempo_map: list[TempoMapEntry],
    *,
    true_bpm: float,
) -> float:
    """Mean |sec_to_beat(t, map) - t * true_bpm/60| over samples."""
    errs: list[float] = []
    for t in sample_times_sec:
        pred = sec_to_beat(float(t), tempo_map)
        truth = beat_truth_constant_tempo(float(t), true_bpm)
        errs.append(abs(pred - truth))
    return sum(errs) / max(len(errs), 1)


def max_abs_beat_error(
    sample_times_sec: Sequence[float],
    tempo_map: list[TempoMapEntry],
    *,
    true_bpm: float,
) -> float:
    return max(
        abs(sec_to_beat(float(t), tempo_map) - beat_truth_constant_tempo(float(t), true_bpm))
        for t in sample_times_sec
    )


def linspace(a: float, b: float, n: int) -> list[float]:
    if n < 2:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]
