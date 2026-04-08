"""Unit tests for audio-derived tempo maps (beat → ``TempoMapEntry`` list)."""
from __future__ import annotations

from backend.contracts import sec_to_beat
from backend.services.audio_timing import build_tempo_map_from_beat_times


def test_build_tempo_map_uniform_120_bpm():
    # Beats every 0.5 s → 120 BPM
    beats = [0.0, 0.5, 1.0, 1.5]
    m = build_tempo_map_from_beat_times(beats, duration_sec=2.0, fallback_bpm=99.0)
    assert len(m) == len(beats)  # n-1 segments + tail anchor
    assert m[0].time_sec == 0.0 and m[0].beat == 0.0 and m[0].bpm == 120.0
    # Mid first segment
    assert abs(sec_to_beat(0.25, m) - 0.5) < 1e-6
    # On second beat
    assert abs(sec_to_beat(0.5, m) - 1.0) < 1e-6
    # Between second and third
    assert abs(sec_to_beat(0.75, m) - 1.5) < 1e-6


def test_build_tempo_map_few_beats_falls_back():
    m = build_tempo_map_from_beat_times([0.0], duration_sec=1.0, fallback_bpm=90.0)
    assert len(m) == 1
    assert m[0].bpm == 90.0
    assert abs(sec_to_beat(1.0, m) - 1.5) < 1e-6  # 1 sec at 90 bpm = 1.5 beats
