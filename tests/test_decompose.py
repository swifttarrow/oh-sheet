"""Unit tests for the decompose service."""
from __future__ import annotations

from shared.contracts import (
    HarmonicAnalysis,
    InstrumentRole,
    MidiTrack,
    Note,
    QualitySignal,
    TempoMapEntry,
    TranscriptionResult,
)

from backend.services.decompose import DecomposeService


def _make_txr(notes_per_track: list[list[Note]], roles: list[InstrumentRole] | None = None) -> TranscriptionResult:
    """Helper to build a TranscriptionResult with given note lists."""
    if roles is None:
        roles = [InstrumentRole.PIANO] * len(notes_per_track)
    tracks = [
        MidiTrack(notes=notes, instrument=role, program=0, confidence=0.9)
        for notes, role in zip(notes_per_track, roles)
    ]
    return TranscriptionResult(
        midi_tracks=tracks,
        analysis=HarmonicAnalysis(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
            chords=[],
            sections=[],
        ),
        quality=QualitySignal(overall_confidence=0.9),
    )


def test_produces_exactly_two_tracks():
    """Decomposer must always output exactly two tracks: melody and other."""
    notes = [
        Note(pitch=72, onset_sec=0.0, offset_sec=0.5, velocity=80),
        Note(pitch=60, onset_sec=0.0, offset_sec=0.5, velocity=60),
        Note(pitch=48, onset_sec=0.0, offset_sec=0.5, velocity=60),
    ]
    txr = _make_txr([notes])
    result = DecomposeService().run(txr)

    assert len(result.midi_tracks) == 2
    roles = {t.instrument for t in result.midi_tracks}
    assert roles == {InstrumentRole.MELODY, InstrumentRole.OTHER}


def test_merges_all_input_tracks():
    """Notes from multiple input tracks should all appear in the output."""
    track1 = [Note(pitch=72, onset_sec=0.0, offset_sec=0.5, velocity=80)]
    track2 = [Note(pitch=48, onset_sec=0.0, offset_sec=0.5, velocity=60)]
    txr = _make_txr(
        [track1, track2],
        roles=[InstrumentRole.MELODY, InstrumentRole.BASS],
    )
    result = DecomposeService().run(txr)

    total_out = sum(len(t.notes) for t in result.midi_tracks)
    total_in = len(track1) + len(track2)
    assert total_out == total_in


def test_melody_track_is_monophonic():
    """Melody track should have at most one note sounding at any onset."""
    # Three simultaneous notes — only the highest should be melody
    notes = [
        Note(pitch=72, onset_sec=0.0, offset_sec=1.0, velocity=80),
        Note(pitch=60, onset_sec=0.0, offset_sec=1.0, velocity=70),
        Note(pitch=48, onset_sec=0.0, offset_sec=1.0, velocity=60),
    ]
    txr = _make_txr([notes])
    result = DecomposeService().run(txr)

    melody = next(t for t in result.midi_tracks if t.instrument == InstrumentRole.MELODY)
    # Group by onset — each onset should have exactly 1 note
    onsets: dict[float, int] = {}
    for n in melody.notes:
        onsets[n.onset_sec] = onsets.get(n.onset_sec, 0) + 1
    assert all(count == 1 for count in onsets.values())


def test_preserves_analysis_and_quality():
    """Analysis and quality signal must pass through unchanged."""
    notes = [Note(pitch=60, onset_sec=0.0, offset_sec=0.5, velocity=80)]
    txr = _make_txr([notes])
    result = DecomposeService().run(txr)

    assert result.analysis.key == txr.analysis.key
    assert result.analysis.tempo_map == txr.analysis.tempo_map
    assert result.schema_version == txr.schema_version


def test_empty_input_returns_empty_tracks():
    """An input with no notes should produce two empty tracks."""
    txr = _make_txr([[]])
    result = DecomposeService().run(txr)

    assert len(result.midi_tracks) == 2
    assert all(len(t.notes) == 0 for t in result.midi_tracks)
