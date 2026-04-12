"""Unit tests for the assemble service."""
from __future__ import annotations

import pytest
from shared.contracts import (
    HarmonicAnalysis,
    InstrumentRole,
    MidiTrack,
    Note,
    QualitySignal,
    TempoMapEntry,
    TranscriptionResult,
)

from backend.services.assemble import AssembleService


def _make_txr(
    melody_notes: list[Note],
    accomp_notes: list[Note],
    key: str = "C:major",
    bpm: float = 120.0,
) -> TranscriptionResult:
    """Build a TranscriptionResult with melody + other tracks."""
    return TranscriptionResult(
        midi_tracks=[
            MidiTrack(notes=melody_notes, instrument=InstrumentRole.MELODY, program=0, confidence=0.9),
            MidiTrack(notes=accomp_notes, instrument=InstrumentRole.OTHER, program=0, confidence=0.9),
        ],
        analysis=HarmonicAnalysis(
            key=key,
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
            chords=[],
            sections=[],
        ),
        quality=QualitySignal(overall_confidence=0.9),
    )


def test_melody_goes_to_right_hand():
    """All melody notes should appear in right_hand."""
    melody = [Note(pitch=72, onset_sec=0.0, offset_sec=0.5, velocity=80)]
    accomp = [Note(pitch=48, onset_sec=0.0, offset_sec=0.5, velocity=60)]
    txr = _make_txr(melody, accomp)

    score = AssembleService().run(txr, difficulty="beginner")
    assert len(score.right_hand) == 1
    assert score.right_hand[0].pitch == 72


def test_bass_goes_to_left_hand():
    """Lowest accompaniment note at each beat should appear in left_hand."""
    melody = [Note(pitch=72, onset_sec=0.0, offset_sec=0.5, velocity=80)]
    accomp = [Note(pitch=48, onset_sec=0.0, offset_sec=0.5, velocity=60)]
    txr = _make_txr(melody, accomp)

    score = AssembleService().run(txr, difficulty="beginner")
    assert len(score.left_hand) == 1
    assert score.left_hand[0].pitch == 48


def test_right_hand_is_monophonic():
    """RH should have max 1 note per quantized beat."""
    melody = [
        Note(pitch=72, onset_sec=0.0, offset_sec=0.5, velocity=80),
        Note(pitch=74, onset_sec=0.0, offset_sec=0.5, velocity=70),
    ]
    txr = _make_txr(melody, [])

    score = AssembleService().run(txr, difficulty="beginner")
    onsets: dict[float, int] = {}
    for n in score.right_hand:
        onsets[n.onset_beat] = onsets.get(n.onset_beat, 0) + 1
    assert all(count == 1 for count in onsets.values())


def test_left_hand_is_monophonic():
    """LH should have max 1 note per quantized beat."""
    accomp = [
        Note(pitch=48, onset_sec=0.0, offset_sec=0.5, velocity=60),
        Note(pitch=52, onset_sec=0.0, offset_sec=0.5, velocity=60),
        Note(pitch=55, onset_sec=0.0, offset_sec=0.5, velocity=60),
    ]
    txr = _make_txr([], accomp)

    score = AssembleService().run(txr, difficulty="beginner")
    onsets: dict[float, int] = {}
    for n in score.left_hand:
        onsets[n.onset_beat] = onsets.get(n.onset_beat, 0) + 1
    assert all(count == 1 for count in onsets.values())


def test_eighth_note_quantization():
    """All onsets and durations should snap to 0.5-beat grid."""
    # onset_sec=0.1 at 120 BPM = 0.2 beats -> should snap to 0.0
    melody = [Note(pitch=72, onset_sec=0.1, offset_sec=0.4, velocity=80)]
    txr = _make_txr(melody, [])

    score = AssembleService().run(txr, difficulty="beginner")
    for n in score.right_hand:
        assert n.onset_beat % 0.5 == 0.0
        assert n.duration_beat % 0.5 == 0.0 or n.duration_beat >= 0.5


def test_range_clamping_rh():
    """RH notes outside C4-C6 (60-84) should be octave-shifted inward."""
    melody = [Note(pitch=90, onset_sec=0.0, offset_sec=0.5, velocity=80)]
    txr = _make_txr(melody, [])

    score = AssembleService().run(txr, difficulty="beginner")
    assert all(60 <= n.pitch <= 84 for n in score.right_hand)


def test_range_clamping_lh():
    """LH notes outside C2-B3 (36-59) should be octave-shifted inward."""
    accomp = [Note(pitch=30, onset_sec=0.0, offset_sec=0.5, velocity=60)]
    txr = _make_txr([], accomp)

    score = AssembleService().run(txr, difficulty="beginner")
    assert all(36 <= n.pitch <= 59 for n in score.left_hand)


def test_accompaniment_above_middle_c_discarded_from_rh():
    """Accompaniment notes above middle C should NOT go to RH."""
    melody = [Note(pitch=72, onset_sec=0.0, offset_sec=0.5, velocity=80)]
    accomp = [Note(pitch=65, onset_sec=0.5, offset_sec=1.0, velocity=60)]
    txr = _make_txr(melody, accomp)

    score = AssembleService().run(txr, difficulty="beginner")
    assert len(score.right_hand) == 1
    assert score.right_hand[0].pitch == 72


def test_difficulty_not_implemented_raises():
    """Non-beginner difficulty should raise NotImplementedError."""
    txr = _make_txr([], [])
    with pytest.raises(NotImplementedError):
        AssembleService().run(txr, difficulty="advanced")


def test_score_metadata():
    """Output metadata should have correct difficulty and key."""
    melody = [Note(pitch=72, onset_sec=0.0, offset_sec=0.5, velocity=80)]
    txr = _make_txr(melody, [], key="C:major")

    score = AssembleService().run(txr, difficulty="beginner")
    assert score.metadata.difficulty == "beginner"
    assert score.metadata.key == "C:major"


def test_note_ids_are_unique():
    """Each ScoreNote should have a unique id."""
    melody = [
        Note(pitch=72, onset_sec=0.0, offset_sec=0.5, velocity=80),
        Note(pitch=74, onset_sec=0.5, offset_sec=1.0, velocity=80),
    ]
    accomp = [
        Note(pitch=48, onset_sec=0.0, offset_sec=0.5, velocity=60),
        Note(pitch=50, onset_sec=0.5, offset_sec=1.0, velocity=60),
    ]
    txr = _make_txr(melody, accomp)

    score = AssembleService().run(txr, difficulty="beginner")
    all_ids = [n.id for n in score.right_hand + score.left_hand]
    assert len(all_ids) == len(set(all_ids))


def test_empty_input_returns_empty_score():
    """No notes in -> empty score out."""
    txr = _make_txr([], [])
    score = AssembleService().run(txr, difficulty="beginner")

    assert len(score.right_hand) == 0
    assert len(score.left_hand) == 0
