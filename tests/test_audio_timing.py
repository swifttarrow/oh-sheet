"""Unit tests for waveform-derived tempo maps.

These tests exercise the pure Python math — ``librosa`` is not required,
so the whole file runs in CI regardless of whether the ``basic-pitch``
extra is installed.
"""
from __future__ import annotations

from backend.contracts import sec_to_beat
from backend.services.audio_timing import build_tempo_map_from_beat_times


def test_uniform_120_bpm_grid_is_monotonic_and_continuous():
    # Beats every 0.5 s → 120 BPM
    beats = [0.0, 0.5, 1.0, 1.5, 2.0]
    m = build_tempo_map_from_beat_times(beats, fallback_bpm=99.0)

    # One anchor per beat: n-1 segments + 1 tail anchor = n entries.
    assert len(m) == len(beats)
    for entry in m:
        assert abs(entry.bpm - 120.0) < 1e-6

    # sec_to_beat is continuous and monotonic across segment boundaries.
    prev = sec_to_beat(0.0, m)
    for s in [0.0, 0.1, 0.25, 0.49, 0.5, 0.75, 1.0, 1.25, 1.9, 2.0]:
        b = sec_to_beat(s, m)
        assert b >= prev - 1e-9, f"non-monotonic at {s}: {b} < {prev}"
        prev = b

    # Each detected beat lands on integer beat index (within fp tolerance).
    for i, t in enumerate(beats):
        assert abs(sec_to_beat(t, m) - float(i)) < 1e-6


def test_variable_tempo_segments_use_local_bpm():
    # Slow half (0.6s between beats = 100 BPM), then fast (0.4s = 150 BPM)
    beats = [0.0, 0.6, 1.2, 1.6, 2.0]
    m = build_tempo_map_from_beat_times(beats)

    # First two segments should be ~100 BPM, last two ~150 BPM.
    assert abs(m[0].bpm - 100.0) < 1e-3
    assert abs(m[1].bpm - 100.0) < 1e-3
    assert abs(m[2].bpm - 150.0) < 1e-3
    assert abs(m[3].bpm - 150.0) < 1e-3

    # Mid first segment (slow): 0.3s should be exactly 0.5 beats.
    assert abs(sec_to_beat(0.3, m) - 0.5) < 1e-6
    # Mid fast segment: t=1.4 is halfway between beats 2 and 3 → 2.5.
    assert abs(sec_to_beat(1.4, m) - 2.5) < 1e-6


def test_empty_or_single_beat_falls_back_to_single_anchor():
    m = build_tempo_map_from_beat_times([], fallback_bpm=90.0)
    assert len(m) == 1
    assert m[0].time_sec == 0.0 and m[0].beat == 0.0
    assert m[0].bpm == 90.0

    # Sanity: 1 second at 90 BPM == 1.5 beats
    assert abs(sec_to_beat(1.0, m) - 1.5) < 1e-6

    m = build_tempo_map_from_beat_times([0.4], fallback_bpm=120.0)
    assert len(m) == 1
    assert m[0].bpm == 120.0


def test_bpm_clamped_to_sane_band():
    # Two beats 1 ms apart would imply 60000 BPM — must be clamped.
    beats = [0.0, 0.001]
    m = build_tempo_map_from_beat_times(beats)
    for entry in m:
        assert 30.0 <= entry.bpm <= 300.0

    # Two beats 5 seconds apart would imply 12 BPM — must be clamped up.
    beats = [0.0, 5.0]
    m = build_tempo_map_from_beat_times(beats)
    for entry in m:
        assert 30.0 <= entry.bpm <= 300.0


def test_fallback_bpm_is_also_clamped():
    m = build_tempo_map_from_beat_times([], fallback_bpm=5.0)
    assert m[0].bpm == 30.0
    m = build_tempo_map_from_beat_times([], fallback_bpm=10_000.0)
    assert m[0].bpm == 300.0


def test_unsorted_input_is_sorted():
    beats = [1.0, 0.0, 0.5]
    m = build_tempo_map_from_beat_times(beats)
    # Anchors must be non-decreasing in time_sec.
    times = [e.time_sec for e in m]
    assert times == sorted(times)
