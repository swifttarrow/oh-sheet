"""Tests for the Phase 9B voice/staff GNN hand assigner."""
from __future__ import annotations

import pytest

from backend.config import settings
from backend.contracts import (
    HarmonicAnalysis,
    InstrumentRole,
    MidiTrack,
    Note,
    QualitySignal,
    TempoMapEntry,
    TranscriptionResult,
)
from backend.services.arrange import _arrange_sync
from backend.services.voice_gnn import (
    DEFAULT_SPLIT_FALLBACK,
    VoiceGNNConfig,
    VoiceGNNStats,
    _choose_split,
    _cluster_streams,
    _stream_centroid,
    _stream_cost,
    assign_hands_gnn,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestStreamCost:
    cfg = VoiceGNNConfig()

    def test_same_pitch_zero_gap(self):
        last = (60, 0.0, 0.5, 80)
        nxt = (60, 0.5, 0.5, 80)
        # Pitch 0, time gap 0, vel 0 → cost 0
        assert _stream_cost(last, nxt, cfg=self.cfg) == 0.0

    def test_pitch_jump_costs(self):
        last = (60, 0.0, 0.5, 80)
        nxt = (72, 0.5, 0.5, 80)
        cost = _stream_cost(last, nxt, cfg=self.cfg)
        # 12 semitones * pitch_weight=1.0
        assert cost == pytest.approx(12.0)

    def test_time_gap_costs(self):
        last = (60, 0.0, 0.5, 80)
        nxt = (60, 1.5, 0.5, 80)  # gap = 1.5 - 0.5 = 1.0 beat
        cost = _stream_cost(last, nxt, cfg=self.cfg)
        # 1.0 beat * time_weight=4.0
        assert cost == pytest.approx(4.0)

    def test_overlapping_notes_no_negative_cost(self):
        last = (60, 0.0, 1.0, 80)
        nxt = (60, 0.25, 0.5, 80)  # onset before last's offset
        cost = _stream_cost(last, nxt, cfg=self.cfg)
        # time_gap should be clamped at 0, so cost = 0
        assert cost == 0.0


class TestClusterStreams:
    cfg = VoiceGNNConfig()

    def test_empty_returns_empty(self):
        assert _cluster_streams([], cfg=self.cfg) == []

    def test_single_note_one_stream(self):
        notes = [(60, 0.0, 1.0, 80)]
        streams = _cluster_streams(notes, cfg=self.cfg)
        assert streams == [[(60, 0.0, 1.0, 80)]]

    def test_smooth_line_one_stream(self):
        # Stepwise scale at adjacent times — should all join one stream.
        notes = [
            (60, 0.0, 0.5, 80),
            (62, 0.5, 0.5, 80),
            (64, 1.0, 0.5, 80),
            (65, 1.5, 0.5, 80),
        ]
        streams = _cluster_streams(notes, cfg=self.cfg)
        assert len(streams) == 1
        assert len(streams[0]) == 4

    def test_disjoint_register_two_streams(self):
        # Two simultaneous voices: high melody + bass line.
        notes = [
            (72, 0.0, 0.5, 80),
            (48, 0.0, 0.5, 80),
            (74, 0.5, 0.5, 80),
            (50, 0.5, 0.5, 80),
        ]
        cfg = VoiceGNNConfig(join_threshold=8.0)
        streams = _cluster_streams(notes, cfg=cfg)
        assert len(streams) == 2
        # One stream is the upper voice, the other is the lower.
        means = sorted(sum(p for p, _, _, _ in s) / len(s) for s in streams)
        assert means[0] < 60 < means[1]

    def test_long_gap_starts_new_stream(self):
        # Same pitch but a wide gap — beyond join_threshold of 8 with default
        # time_weight=4.0, gap of 3.0 → cost 12.0 (above threshold).
        notes = [
            (60, 0.0, 0.5, 80),
            (60, 5.0, 0.5, 80),
        ]
        streams = _cluster_streams(notes, cfg=self.cfg)
        assert len(streams) == 2


class TestStreamCentroid:
    def test_single_note(self):
        assert _stream_centroid([(60, 0.0, 1.0, 80)]) == 60.0

    def test_duration_weighted(self):
        # A long C4 plus a brief C5 → centroid should pull toward C4.
        stream = [(60, 0.0, 4.0, 80), (72, 4.0, 0.25, 80)]
        c = _stream_centroid(stream)
        # Weighted: (60*4.0 + 72*0.25) / 4.25 = 240.0 + 18.0 / 4.25 ≈ 60.7
        assert 60.0 <= c < 62.0

    def test_empty_stream_returns_default(self):
        assert _stream_centroid([]) == DEFAULT_SPLIT_FALLBACK


class TestChooseSplit:
    cfg = VoiceGNNConfig()

    def test_empty_falls_back_to_default(self):
        assert _choose_split([], self.cfg) == DEFAULT_SPLIT_FALLBACK

    def test_single_centroid_falls_back_to_default(self):
        # One stream → no split decision possible — fall back.
        assert _choose_split([72.0], self.cfg) == DEFAULT_SPLIT_FALLBACK

    def test_clamped_to_window(self):
        # Median of [40, 42] = 41 — but clamp to min_split_hint=55.
        assert _choose_split([40.0, 42.0], self.cfg) == self.cfg.min_split_hint
        # Median of [80, 90] = 85 — but clamp to max_split_hint=65.
        assert _choose_split([80.0, 90.0], self.cfg) == self.cfg.max_split_hint

    def test_typical_two_voice_split(self):
        # Bass at C3, treble at C5 → median 60 → middle C.
        assert _choose_split([48.0, 72.0], self.cfg) == 60


class TestAssignHandsGnn:
    def test_empty_returns_empty(self):
        rh, lh, stats = assign_hands_gnn([])
        assert rh == [] and lh == []
        assert stats.skipped is True

    def test_separates_typical_pop_voicing(self):
        # Treble melody + LH bass — should split clean.
        notes = [
            (72, 0.0, 0.5, 80),
            (74, 0.5, 0.5, 80),
            (76, 1.0, 0.5, 80),
            (40, 0.0, 1.0, 70),
            (43, 1.0, 1.0, 70),
        ]
        rh, lh, stats = assign_hands_gnn(notes)
        assert len(rh) == 3
        assert len(lh) == 2
        assert all(p >= 60 for p, _, _, _ in rh)
        assert all(p < 60 for p, _, _, _ in lh)
        assert stats.n_notes == 5
        assert stats.n_streams >= 2

    def test_output_sorted_by_onset(self):
        notes = [
            (72, 1.0, 0.5, 80),
            (74, 0.0, 0.5, 80),
            (76, 0.5, 0.5, 80),
        ]
        rh, _, _ = assign_hands_gnn(notes)
        onsets = [n[1] for n in rh]
        assert onsets == sorted(onsets)

    def test_does_not_mutate_input(self):
        notes = [(72, 0.0, 0.5, 80), (40, 0.0, 1.0, 70)]
        original = list(notes)
        assign_hands_gnn(notes)
        assert notes == original

    def test_monophonic_high_line_all_rh(self):
        # Single voice well above middle C — all RH.
        notes = [(67 + i, i * 0.5, 0.5, 80) for i in range(8)]
        rh, lh, stats = assign_hands_gnn(notes)
        assert len(rh) == 8
        assert len(lh) == 0
        # One stream, default split fires.
        assert stats.n_streams == 1
        assert stats.split_pitch == DEFAULT_SPLIT_FALLBACK


# ---------------------------------------------------------------------------
# Integration with arrange — the wired-in flag swap.
# ---------------------------------------------------------------------------


def _payload(tracks: list[MidiTrack]) -> TranscriptionResult:
    return TranscriptionResult(
        midi_tracks=tracks,
        analysis=HarmonicAnalysis(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
        ),
        quality=QualitySignal(overall_confidence=0.9),
    )


class TestArrangeIntegration:
    def test_default_disabled_uses_split_pitch(self, monkeypatch):
        monkeypatch.setattr(settings, "voice_gnn_enabled", False)
        # All notes in OTHER track — naive split puts >=60 RH, <60 LH.
        notes = [
            Note(pitch=72, onset_sec=0.0, offset_sec=0.5, velocity=80),
            Note(pitch=48, onset_sec=0.0, offset_sec=0.5, velocity=80),
        ]
        payload = _payload([
            MidiTrack(notes=notes, instrument=InstrumentRole.OTHER, confidence=0.9),
        ])
        score = _arrange_sync(payload, "intermediate")
        assert {n.pitch for n in score.right_hand} == {72}
        assert {n.pitch for n in score.left_hand} == {48}

    def test_enabled_routes_through_gnn(self, monkeypatch):
        monkeypatch.setattr(settings, "voice_gnn_enabled", True)
        # Same input — GNN should still split high/low cleanly.
        notes = [
            Note(pitch=72, onset_sec=0.0, offset_sec=0.5, velocity=80),
            Note(pitch=48, onset_sec=0.0, offset_sec=0.5, velocity=80),
        ]
        payload = _payload([
            MidiTrack(notes=notes, instrument=InstrumentRole.OTHER, confidence=0.9),
        ])
        score = _arrange_sync(payload, "intermediate")
        # Streams cluster cleanly by register; high pitch lands RH.
        assert any(n.pitch == 72 for n in score.right_hand)
        assert any(n.pitch == 48 for n in score.left_hand)

    def test_melody_track_still_routes_to_rh_when_enabled(self, monkeypatch):
        # MELODY hint must win over GNN re-routing.
        monkeypatch.setattr(settings, "voice_gnn_enabled", True)
        # A low-pitch note tagged MELODY — naive split would send it LH;
        # explicit MELODY routing must keep it in RH.
        notes = [Note(pitch=48, onset_sec=0.0, offset_sec=0.5, velocity=80)]
        payload = _payload([
            MidiTrack(notes=notes, instrument=InstrumentRole.MELODY, confidence=0.9),
        ])
        score = _arrange_sync(payload, "intermediate")
        assert len(score.right_hand) == 1
        assert score.right_hand[0].pitch == 48
        assert len(score.left_hand) == 0


class TestStatsAsWarnings:
    def test_empty_no_warnings(self):
        s = VoiceGNNStats(n_notes=0)
        assert s.as_warnings() == []

    def test_skipped_returns_warnings(self):
        s = VoiceGNNStats(skipped=True, warnings=["x"])
        assert s.as_warnings() == ["x"]

    def test_assigned_returns_summary(self):
        s = VoiceGNNStats(
            n_notes=5, n_streams=2, split_pitch=60, n_rh=3, n_lh=2,
        )
        out = s.as_warnings()
        assert len(out) == 1
        assert "5 notes" in out[0]
        assert "2 streams" in out[0]
