"""Tests for adaptive quantization grid in the arrange stage."""
from __future__ import annotations

import pytest

from backend.config import settings
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
from backend.services.arrange import (
    QUANT_GRID,
    ArrangeService,
    _arrange_sync,
    _estimate_best_grid,
)


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
                # Within NEAR_OVERLAP_TOL after quantization so dedup keeps the louder hit.
                notes=[Note(pitch=72, onset_sec=0.54, offset_sec=0.84, velocity=100)],
            ),
        ]
    )

    score = _arrange_sync(payload, "intermediate")

    assert len(score.right_hand) == 1
    # Both land on the 1.0 beat grid; louder velocity survives.
    assert score.right_hand[0].onset_beat == 1.0
    assert score.right_hand[0].pitch == 72
    assert len(score.left_hand) == 0


def test_arrange_beginner_reduces_max_polyphony() -> None:
    """Voice caps: intermediate=2 RH voices, beginner=1. PR-10 / plan 3.1.

    Three overlapping notes at beat 0 force the voice allocator up to
    three concurrent voices. Intermediate keeps voices 1 and 2 (the
    piano cap) and drops the third; beginner keeps only voice 1 and
    drops both extras.
    """
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

    assert len(beginner.right_hand) == 1
    assert len(intermediate.right_hand) == 2
    assert max(n.voice for n in intermediate.right_hand) <= 2
    assert max(n.voice for n in beginner.right_hand) == 1


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


# ---------------------------------------------------------------------------
# _estimate_best_grid unit tests
# ---------------------------------------------------------------------------

DEFAULT_CANDIDATES = [0.167, 0.25, 0.333, 0.5]


class TestEstimateBestGrid:
    """Tests for the grid-estimation helper."""

    def test_picks_quarter_sixteenth_grid(self):
        onsets = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        grid = _estimate_best_grid(onsets, candidates=DEFAULT_CANDIDATES, min_notes=4)
        assert grid == pytest.approx(0.25)

    def test_picks_triplet_grid(self):
        onsets = [0.0, 0.333, 0.667, 1.0, 1.333]
        grid = _estimate_best_grid(onsets, candidates=DEFAULT_CANDIDATES, min_notes=4)
        assert grid == pytest.approx(0.333)

    def test_picks_eighth_note_grid(self):
        onsets = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        grid = _estimate_best_grid(onsets, candidates=DEFAULT_CANDIDATES, min_notes=4)
        assert grid == pytest.approx(0.5)

    def test_picks_sixteenth_triplet_grid(self):
        onsets = [0.0, 0.167, 0.334, 0.501, 0.668]
        grid = _estimate_best_grid(onsets, candidates=DEFAULT_CANDIDATES, min_notes=4)
        assert grid == pytest.approx(0.167)

    def test_fallback_when_too_few_notes(self):
        onsets = [0.0, 0.5, 1.0]
        grid = _estimate_best_grid(onsets, candidates=DEFAULT_CANDIDATES, min_notes=4)
        assert grid == QUANT_GRID

    def test_empty_onsets_returns_default(self):
        grid = _estimate_best_grid([], candidates=DEFAULT_CANDIDATES, min_notes=4)
        assert grid == QUANT_GRID


# ---------------------------------------------------------------------------
# Integration: config switch preserves old behaviour
# ---------------------------------------------------------------------------


def _make_transcription(onset_offsets: list[tuple[float, float]], bpm: float = 120.0) -> TranscriptionResult:
    notes = [
        Note(pitch=64, onset_sec=o, offset_sec=off, velocity=80)
        for o, off in onset_offsets
    ]
    return TranscriptionResult(
        midi_tracks=[MidiTrack(notes=notes, instrument=InstrumentRole.MELODY, confidence=0.9)],
        analysis=HarmonicAnalysis(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=bpm)],
        ),
        quality=QualitySignal(overall_confidence=0.9),
    )


@pytest.mark.asyncio
async def test_disabled_adaptive_grid_uses_default(monkeypatch):
    monkeypatch.setattr(settings, "arrange_adaptive_grid_enabled", False)
    monkeypatch.setattr(settings, "arrange_beat_snap_enabled", False)

    onset_offsets = [
        (0.0, 0.15),
        (0.1665, 0.32),
        (0.3335, 0.49),
        (0.5, 0.65),
        (0.6665, 0.82),
    ]
    payload = _make_transcription(onset_offsets, bpm=120.0)
    svc = ArrangeService()
    score = await svc.run(payload)

    for note in score.right_hand:
        remainder = note.onset_beat % 0.25
        assert remainder == pytest.approx(0.0, abs=1e-9), (
            f"onset {note.onset_beat} is not on the 0.25 grid"
        )


@pytest.mark.asyncio
async def test_enabled_adaptive_grid_picks_triplet(monkeypatch):
    monkeypatch.setattr(settings, "arrange_adaptive_grid_enabled", True)
    monkeypatch.setattr(settings, "arrange_beat_snap_enabled", False)

    onset_offsets = [
        (0.0, 0.15),
        (0.1665, 0.32),
        (0.3335, 0.49),
        (0.5, 0.65),
        (0.6665, 0.82),
    ]
    payload = _make_transcription(onset_offsets, bpm=120.0)
    svc = ArrangeService()
    score = await svc.run(payload)

    for note in score.right_hand:
        remainder = note.onset_beat % 0.333
        assert remainder == pytest.approx(0.0, abs=0.01) or \
               remainder == pytest.approx(0.333, abs=0.01), (
            f"onset {note.onset_beat} is not on the 0.333 grid"
        )
