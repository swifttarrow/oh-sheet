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
from backend.services.condense import _condense_sync


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


def test_condense_merges_tracks_in_chronological_order() -> None:
    """Like ``merge_tracks``, later events from any track follow earlier ones by time."""
    payload = _payload(
        tracks=[
            MidiTrack(
                instrument=InstrumentRole.MELODY,
                program=0,
                confidence=0.9,
                notes=[Note(pitch=64, onset_sec=1.0, offset_sec=1.2, velocity=80)],
            ),
            MidiTrack(
                instrument=InstrumentRole.BASS,
                program=32,
                confidence=0.9,
                notes=[Note(pitch=65, onset_sec=0.0, offset_sec=0.5, velocity=80)],
            ),
        ],
    )
    score = _condense_sync(payload, "intermediate")
    rh = sorted(score.right_hand, key=lambda n: n.onset_beat)
    assert len(rh) == 2
    assert rh[0].pitch == 65
    assert rh[0].onset_beat == 0.0
    assert rh[1].pitch == 64
    assert rh[1].onset_beat == 2.0


def test_condense_splits_hands_at_middle_c() -> None:
    payload = _payload(
        tracks=[
            MidiTrack(
                instrument=InstrumentRole.PIANO,
                program=0,
                confidence=0.9,
                notes=[
                    Note(pitch=48, onset_sec=0.0, offset_sec=0.5, velocity=70),
                    Note(pitch=72, onset_sec=0.0, offset_sec=0.5, velocity=70),
                ],
            ),
        ],
    )
    score = _condense_sync(payload, "intermediate")
    assert len(score.left_hand) == 1
    assert len(score.right_hand) == 1
    assert score.left_hand[0].pitch == 48
    assert score.right_hand[0].pitch == 72


def test_condense_keeps_near_duplicate_pitches_unlike_arrange_dedup() -> None:
    """Merge script keeps separate note events; we do not cross-track dedup."""
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
                notes=[Note(pitch=72, onset_sec=0.55, offset_sec=0.85, velocity=100)],
            ),
        ],
    )
    score = _condense_sync(payload, "intermediate")
    assert len(score.right_hand) == 2


def test_condense_converts_chords_and_sections_to_beats() -> None:
    payload = _payload(
        tracks=[
            MidiTrack(
                instrument=InstrumentRole.PIANO,
                program=0,
                confidence=0.9,
                notes=[Note(pitch=60, onset_sec=0.0, offset_sec=0.5, velocity=80)],
            ),
        ],
        chords=[RealtimeChordEvent(time_sec=1.0, duration_sec=0.5, label="C:maj", root=0, confidence=0.8)],
        sections=[Section(start_sec=0.0, end_sec=2.0, label=SectionLabel.VERSE)],
    )
    score = _condense_sync(payload, "intermediate")
    assert len(score.metadata.chord_symbols) == 1
    assert score.metadata.chord_symbols[0].beat == 2.0
    assert len(score.metadata.sections) == 1
    assert score.metadata.sections[0].label == SectionLabel.VERSE


def test_condense_empty_tracks_yields_empty_hands_with_metadata() -> None:
    payload = _payload(tracks=[])
    score = _condense_sync(payload, "intermediate")
    assert score.right_hand == []
    assert score.left_hand == []
    assert score.metadata.key == "C:major"


def test_condense_caps_polyphony_per_hand() -> None:
    """Greedy voice assignment drops excess simultaneous notes (cap = 16)."""
    notes = [
        Note(pitch=60 + i, onset_sec=0.0, offset_sec=0.5, velocity=80)
        for i in range(20)
    ]
    payload = _payload(
        tracks=[
            MidiTrack(
                instrument=InstrumentRole.PIANO,
                program=0,
                confidence=0.9,
                notes=notes,
            ),
        ],
    )
    score = _condense_sync(payload, "intermediate")
    assert len(score.right_hand) == 16
