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
    TempoMapEntry,
    TranscriptionResult,
)
from backend.services.arrange import (
    QUANT_GRID,
    ArrangeService,
    _estimate_best_grid,
)

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
