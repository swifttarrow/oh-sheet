from __future__ import annotations

from backend.contracts import (
    HarmonicAnalysis,
    InstrumentRole,
    MidiTrack,
    Note,
    QualitySignal,
    RealtimeChordEvent,
    Section,
    SectionLabel,
    TempoMapEntry,
    TranscriptionResult,
)
from backend.services.arrange import _arrange_sync


def _payload(
    tracks: list[MidiTrack],
    *,
    chords: list[RealtimeChordEvent] | None = None,
    sections: list[Section] | None = None,
    tempo_map: list[TempoMapEntry] | None = None,
) -> TranscriptionResult:
    return TranscriptionResult(
        midi_tracks=tracks,
        analysis=HarmonicAnalysis(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=tempo_map or [TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
            chords=chords or [],
            sections=sections or [],
        ),
        quality=QualitySignal(overall_confidence=0.9, warnings=[]),
    )


def test_arrange_builds_two_hands_and_converts_metadata() -> None:
    payload = _payload(
        tracks=[
            MidiTrack(
                instrument=InstrumentRole.MELODY,
                program=0,
                confidence=0.9,
                notes=[Note(pitch=64, onset_sec=0.13, offset_sec=0.38, velocity=90)],
            ),
            MidiTrack(
                instrument=InstrumentRole.BASS,
                program=32,
                confidence=0.9,
                notes=[Note(pitch=48, onset_sec=0.62, offset_sec=1.00, velocity=80)],
            ),
        ],
        chords=[RealtimeChordEvent(time_sec=1.0, duration_sec=0.5, label="C:maj", root=0, confidence=0.8)],
        sections=[Section(start_sec=0.0, end_sec=2.0, label=SectionLabel.VERSE)],
    )

    score = _arrange_sync(payload, "intermediate")

    assert len(score.right_hand) == 1
    assert score.right_hand[0].pitch == 64
    assert score.right_hand[0].onset_beat == 0.25
    assert score.right_hand[0].duration_beat == 0.5
    assert score.right_hand[0].voice == 1

    assert len(score.left_hand) == 1
    assert score.left_hand[0].pitch == 48
    assert score.left_hand[0].onset_beat == 1.25
    assert score.left_hand[0].duration_beat == 0.75
    assert score.left_hand[0].voice == 1

    assert len(score.metadata.chord_symbols) == 1
    assert score.metadata.chord_symbols[0].beat == 2.0
    assert score.metadata.chord_symbols[0].duration_beat == 1.0
    assert score.metadata.chord_symbols[0].label == "C:maj"

    assert len(score.metadata.sections) == 1
    assert score.metadata.sections[0].start_beat == 0.0
    assert score.metadata.sections[0].end_beat == 4.0
    assert score.metadata.sections[0].label == SectionLabel.VERSE


def test_arrange_dedups_same_pitch_across_tracks_keep_loudest() -> None:
    payload = _payload(
        tracks=[
            MidiTrack(
                instrument=InstrumentRole.CHORDS,
                program=0,
                confidence=0.9,
                notes=[Note(pitch=72, onset_sec=0.50, offset_sec=0.80, velocity=40)],
            ),
            MidiTrack(
                instrument=InstrumentRole.CHORDS,
                program=1,
                confidence=0.9,
                notes=[Note(pitch=72, onset_sec=0.66, offset_sec=0.96, velocity=100)],
            ),
        ]
    )

    score = _arrange_sync(payload, "intermediate")

    assert len(score.right_hand) == 1
    # 0.66s at 120 BPM => 1.32 beats, quantized to 1.25 (the louder note).
    assert score.right_hand[0].onset_beat == 1.25
    assert score.right_hand[0].pitch == 72
    assert len(score.left_hand) == 0


def test_arrange_beginner_reduces_max_polyphony() -> None:
    payload = _payload(
        tracks=[
            MidiTrack(
                instrument=InstrumentRole.MELODY,
                program=0,
                confidence=0.95,
                notes=[
                    Note(pitch=72, onset_sec=0.0, offset_sec=1.0, velocity=80),
                    Note(pitch=76, onset_sec=0.0, offset_sec=1.0, velocity=81),
                    Note(pitch=79, onset_sec=0.0, offset_sec=1.0, velocity=82),
                ],
            )
        ]
    )

    beginner = _arrange_sync(payload, "beginner")
    intermediate = _arrange_sync(payload, "intermediate")

    assert len(beginner.right_hand) == 2
    assert len(intermediate.right_hand) == 3
    assert max(n.voice for n in beginner.right_hand) <= 2


def test_arrange_falls_back_to_all_tracks_when_all_are_other() -> None:
    payload = _payload(
        tracks=[
            MidiTrack(
                instrument=InstrumentRole.OTHER,
                program=10,
                confidence=0.95,
                notes=[Note(pitch=67, onset_sec=0.2, offset_sec=0.5, velocity=88)],
            )
        ]
    )

    score = _arrange_sync(payload, "intermediate")

    assert len(score.right_hand) == 1
    assert score.right_hand[0].pitch == 67
