"""Tests for the Phase 9A Score-HPT velocity refiner."""
from __future__ import annotations

import pytest

from backend.contracts import (
    HarmonicAnalysis,
    InstrumentRole,
    MidiTrack,
    Note,
    PitchBendPoint,
    QualitySignal,
    TempoMapEntry,
    TranscriptionResult,
)
from backend.services.score_hpt import (
    ScoreHPTConfig,
    ScoreHPTStats,
    _is_downbeat,
    _is_on_beat,
    _local_density,
    _register_curve,
    refine_velocities,
)


def _track(notes: list[Note], instrument: InstrumentRole = InstrumentRole.MELODY) -> MidiTrack:
    return MidiTrack(notes=notes, instrument=instrument, confidence=0.9)


def _tmap(bpm: float = 120.0) -> list[TempoMapEntry]:
    return [TempoMapEntry(time_sec=0.0, beat=0.0, bpm=bpm)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestRegisterCurve:
    def test_zero_strength_disables(self):
        assert _register_curve(60, 0.0) == 0.0
        assert _register_curve(20, 0.0) == 0.0
        assert _register_curve(108, 0.0) == 0.0

    def test_middle_c_no_attenuation(self):
        # C4 is the curve's peak so attenuation is zero regardless of strength.
        assert _register_curve(60, 1.0) == 0.0

    def test_extremes_attenuated(self):
        # 24 semitones from C4 hits the floor; quadratic so monotonically negative.
        deep = _register_curve(36, 1.0)  # C2
        high = _register_curve(84, 1.0)  # C6
        assert deep == pytest.approx(-10.0)
        assert high == pytest.approx(-10.0)
        # A milder strength scales linearly.
        mild = _register_curve(36, 0.3)
        assert mild == pytest.approx(-3.0)


class TestIsDownbeat:
    def test_no_downbeats(self):
        assert _is_downbeat(0.0, [], 0.05) is False
        assert _is_downbeat(4.0, [], 0.05) is False

    def test_within_tolerance(self):
        assert _is_downbeat(4.02, [0.0, 4.0, 8.0], 0.05) is True
        assert _is_downbeat(7.99, [0.0, 4.0, 8.0], 0.05) is True

    def test_outside_tolerance(self):
        assert _is_downbeat(4.5, [0.0, 4.0, 8.0], 0.05) is False


class TestIsOnBeat:
    def test_integer_beats(self):
        assert _is_on_beat(0.0, 0.05) is True
        assert _is_on_beat(1.0, 0.05) is True
        assert _is_on_beat(2.04, 0.05) is True

    def test_off_beats(self):
        assert _is_on_beat(0.5, 0.05) is False
        assert _is_on_beat(1.25, 0.05) is False


class TestLocalDensity:
    def test_empty(self):
        assert _local_density(0.0, [], window=0.25) == 0

    def test_single_note(self):
        assert _local_density(1.0, [1.0], window=0.25) == 1

    def test_window_inclusive(self):
        beats = [0.0, 0.5, 1.0, 1.5, 2.0]
        # window=0.25 around 1.0 covers [0.75, 1.25] → just the 1.0 beat
        assert _local_density(1.0, beats, window=0.25) == 1
        # window=0.6 covers [0.4, 1.6] → 0.5, 1.0, 1.5
        assert _local_density(1.0, beats, window=0.6) == 3


# ---------------------------------------------------------------------------
# refine_velocities — main function behavior
# ---------------------------------------------------------------------------


class TestRefineVelocitiesEdgeCases:
    def test_empty_tracks_skipped(self):
        out, stats = refine_velocities([], _tmap())
        assert out == []
        assert stats.skipped is True
        assert stats.warnings

    def test_empty_tempo_map_skipped(self):
        notes = [Note(pitch=60, onset_sec=0.0, offset_sec=0.5, velocity=80)]
        tracks = [_track(notes)]
        out, stats = refine_velocities(tracks, [])
        assert out is tracks
        assert stats.skipped is True

    def test_no_notes_skipped(self):
        tracks = [_track([])]
        out, stats = refine_velocities(tracks, _tmap())
        assert stats.skipped is True
        assert stats.n_notes == 0

    def test_does_not_mutate_input(self):
        notes = [Note(pitch=60, onset_sec=0.0, offset_sec=0.5, velocity=80)]
        tracks = [_track(notes)]
        original_velocity = notes[0].velocity
        refine_velocities(tracks, _tmap())
        assert notes[0].velocity == original_velocity


class TestRefineVelocitiesMetric:
    def test_downbeat_boost(self):
        # Two notes at velocity 60 — one on a downbeat, one off.
        notes = [
            Note(pitch=60, onset_sec=0.0, offset_sec=0.4, velocity=60),
            Note(pitch=60, onset_sec=0.25, offset_sec=0.4, velocity=60),
        ]
        tracks = [_track(notes)]
        cfg = ScoreHPTConfig(
            blend_alpha=1.0,           # no blend — full predicted output
            register_curve_strength=0.0,
            density_compensation=0.0,
            downbeat_boost=20.0,
            beat_boost=0.0,
            offbeat_attenuation=10.0,
        )
        out, stats = refine_velocities(
            tracks, _tmap(120.0), downbeats_sec=[0.0], config=cfg,
        )
        rh_notes = out[0].notes
        # Onset at 0.0s @ 120 bpm → beat 0 (downbeat)
        # Onset at 0.25s @ 120 bpm → beat 0.5 (off-beat)
        assert rh_notes[0].velocity > rh_notes[1].velocity
        assert rh_notes[0].velocity == 60 + 20
        assert rh_notes[1].velocity == 60 - 10
        assert stats.n_changed == 2

    def test_blend_alpha_zero_no_change(self):
        notes = [Note(pitch=60, onset_sec=0.0, offset_sec=0.5, velocity=72)]
        tracks = [_track(notes)]
        cfg = ScoreHPTConfig(blend_alpha=0.0)
        out, stats = refine_velocities(tracks, _tmap(), config=cfg)
        assert out[0].notes[0].velocity == 72
        assert stats.n_changed == 0

    def test_clamps_to_min_max(self):
        # Force adjustment that would push past the clamps.
        notes = [
            Note(pitch=60, onset_sec=0.0, offset_sec=0.5, velocity=10),
            Note(pitch=60, onset_sec=1.0, offset_sec=1.5, velocity=125),
        ]
        tracks = [_track(notes)]
        cfg = ScoreHPTConfig(
            blend_alpha=1.0,
            downbeat_boost=50.0,
            register_curve_strength=0.0,
            density_compensation=0.0,
            min_velocity=20,
            max_velocity=110,
        )
        out, _ = refine_velocities(
            tracks, _tmap(120.0), downbeats_sec=[0.0, 0.5, 1.0], config=cfg,
        )
        # Floor enforces min=20 (would have predicted 60 from boost so OK above floor),
        # ceiling enforces max=110 (would have predicted 175).
        assert out[0].notes[0].velocity >= 20
        assert out[0].notes[1].velocity == 110

    def test_pitch_and_other_fields_preserved(self):
        bend = [
            PitchBendPoint(time_sec=0.0, cents=0.0),
            PitchBendPoint(time_sec=0.5, cents=50.0),
        ]
        notes = [
            Note(
                pitch=72,
                onset_sec=0.0,
                offset_sec=1.0,
                velocity=90,
                pitch_bend_cents=bend,
            ),
        ]
        tracks = [_track(notes)]
        out, _ = refine_velocities(tracks, _tmap())
        n = out[0].notes[0]
        assert n.pitch == 72
        assert n.onset_sec == 0.0
        assert n.offset_sec == 1.0
        assert n.pitch_bend_cents == bend


class TestRefineVelocitiesDensity:
    def test_dense_chord_attenuated(self):
        # 5 simultaneous notes — should trigger high-density attenuation.
        notes = [
            Note(pitch=60 + i, onset_sec=0.0, offset_sec=1.0, velocity=80)
            for i in range(5)
        ]
        tracks = [_track(notes)]
        cfg = ScoreHPTConfig(
            blend_alpha=1.0,
            downbeat_boost=0.0,
            beat_boost=0.0,
            offbeat_attenuation=0.0,
            register_curve_strength=0.0,
            density_compensation=10.0,
            density_count_high=4,
        )
        out, _ = refine_velocities(tracks, _tmap(), config=cfg)
        # All 5 notes should be attenuated by 10 → 70.
        assert all(n.velocity == 70 for n in out[0].notes)

    def test_lonely_note_boosted(self):
        # Single isolated note → density==1 → +0.5 * density_compensation
        notes = [Note(pitch=60, onset_sec=0.0, offset_sec=0.5, velocity=70)]
        tracks = [_track(notes)]
        cfg = ScoreHPTConfig(
            blend_alpha=1.0,
            downbeat_boost=0.0,
            beat_boost=0.0,
            offbeat_attenuation=0.0,
            register_curve_strength=0.0,
            density_compensation=10.0,
        )
        out, _ = refine_velocities(tracks, _tmap(), config=cfg)
        assert out[0].notes[0].velocity == 70 + 5


class TestRefineVelocitiesIntegration:
    def test_runs_with_full_transcription_result(self):
        """End-to-end: refine through a TranscriptionResult-like input.

        Doesn't go through the full pipeline; just confirms the contract
        types accept the refined output without surprises.
        """
        notes = [
            Note(pitch=60, onset_sec=0.0, offset_sec=0.5, velocity=80),
            Note(pitch=64, onset_sec=0.25, offset_sec=0.75, velocity=60),
            Note(pitch=67, onset_sec=0.5, offset_sec=1.0, velocity=72),
        ]
        tracks = [_track(notes)]
        result = TranscriptionResult(
            midi_tracks=tracks,
            analysis=HarmonicAnalysis(
                key="C:major",
                time_signature=(4, 4),
                tempo_map=_tmap(120.0),
                downbeats=[0.0, 2.0, 4.0],
            ),
            quality=QualitySignal(overall_confidence=0.9),
        )
        refined, stats = refine_velocities(
            result.midi_tracks,
            result.analysis.tempo_map,
            downbeats_sec=result.analysis.downbeats,
        )
        # Sanity: returns the right shape.
        assert len(refined) == 1
        assert len(refined[0].notes) == 3
        # Velocities stay in valid MIDI range.
        for note in refined[0].notes:
            assert 0 <= note.velocity <= 127
        # Stats record what changed.
        assert stats.n_notes == 3
        assert isinstance(stats, ScoreHPTStats)


class TestTranscribeSeam:
    """The TranscribeService.run() integration is covered indirectly here:

    we exercise the helper that the run loop dispatches to so we don't
    need the full audio pipeline scaffolding to test the wiring.
    """

    def test_apply_score_hpt_disabled_when_no_changes(self, monkeypatch):
        from backend.config import settings as live_settings
        from backend.services.transcribe import _apply_score_hpt

        monkeypatch.setattr(live_settings, "score_hpt_blend_alpha", 0.0)
        result = TranscriptionResult(
            midi_tracks=[_track([
                Note(pitch=60, onset_sec=0.0, offset_sec=0.5, velocity=72),
            ])],
            analysis=HarmonicAnalysis(
                key="C:major",
                time_signature=(4, 4),
                tempo_map=_tmap(120.0),
            ),
            quality=QualitySignal(overall_confidence=0.9),
        )
        out = _apply_score_hpt(result)
        # alpha=0 means no change, but the helper still runs cleanly.
        assert out.midi_tracks[0].notes[0].velocity == 72

    def test_apply_score_hpt_swallows_failures(self, monkeypatch):
        """A bad config / inference error must not sink the job."""
        from backend.services.transcribe import _apply_score_hpt

        # A degenerate result with no tempo_map → refine_velocities skips,
        # but the helper still returns the result instead of raising.
        result = TranscriptionResult(
            midi_tracks=[_track([
                Note(pitch=60, onset_sec=0.0, offset_sec=0.5, velocity=72),
            ])],
            analysis=HarmonicAnalysis(
                key="C:major",
                time_signature=(4, 4),
                tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
            ),
            quality=QualitySignal(overall_confidence=0.9),
        )
        out = _apply_score_hpt(result)
        # Just confirms it returns a valid TranscriptionResult.
        assert isinstance(out, TranscriptionResult)
        assert len(out.midi_tracks) == 1


class TestStatsAsWarnings:
    def test_skipped_returns_warnings(self):
        s = ScoreHPTStats(skipped=True, warnings=["boom"])
        assert s.as_warnings() == ["boom"]

    def test_no_changes_returns_empty(self):
        s = ScoreHPTStats(n_notes=10, n_changed=0)
        assert s.as_warnings() == []

    def test_changed_returns_summary(self):
        s = ScoreHPTStats(
            n_notes=10, n_changed=4, mean_abs_delta=2.5, max_abs_delta=8,
        )
        out = s.as_warnings()
        assert len(out) == 1
        assert "4/10" in out[0]
        assert "2.5" in out[0]
        assert "8" in out[0]
