"""Tests for backend.services.midi_render.

The critical contract is symmetric to ``_looks_like_stub`` on the
response side: producing a stub MIDI on the request side would
round-trip as a blank MusicXML (and get caught downstream anyway, but
the error message would point at the wrong layer). Fail loudly here.
"""
from __future__ import annotations

import sys

import pytest
from shared.contracts import (
    ExpressionMap,
    ExpressiveNote,
    HumanizedPerformance,
    PianoScore,
    QualitySignal,
    ScoreMetadata,
    ScoreNote,
    TempoMapEntry,
)

from backend.services.midi_render import MidiRenderError, render_midi_bytes


def _perf(expressive: list[ExpressiveNote], score_notes: list[ScoreNote] | None = None) -> HumanizedPerformance:
    return HumanizedPerformance(
        expressive_notes=expressive,
        expression=ExpressionMap(),
        score=PianoScore(
            right_hand=score_notes or [],
            left_hand=[],
            metadata=ScoreMetadata(
                key="C:major",
                time_signature=(4, 4),
                tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
                difficulty="intermediate",
            ),
        ),
        quality=QualitySignal(overall_confidence=0.9, warnings=[]),
    )


def test_render_midi_raises_when_pretty_midi_missing(monkeypatch):
    """pretty_midi is a top-level dep; a missing import means the
    image is broken. Fail loudly rather than emit a stub MIDI that
    would travel to the engraver and round-trip as blank MusicXML.
    """
    monkeypatch.setitem(sys.modules, "pretty_midi", None)
    with pytest.raises(MidiRenderError, match="pretty_midi is not installed"):
        render_midi_bytes(_perf([]))


def test_render_midi_raises_on_empty_performance():
    """Zero renderable notes → upstream regression or silent audio.
    A blank engrave is not a useful artifact; surface the problem.
    """
    with pytest.raises(MidiRenderError, match="no renderable notes"):
        render_midi_bytes(_perf([]))


def test_render_midi_raises_when_all_notes_below_min_duration():
    """The renderer filters notes shorter than MIN_NOTE_DUR (30ms).
    A performance where every note is sub-threshold resolves to zero
    notes in pretty_midi's instrument and must trigger the same guard.
    """
    tiny_notes = [
        ExpressiveNote(
            score_note_id="n-1",
            pitch=60,
            onset_beat=0.0,
            duration_beat=0.001,  # ~1 ms at 120 bpm — below MIN_NOTE_DUR
            velocity=80,
            hand="rh",
            voice=1,
            timing_offset_ms=0.0,
            velocity_offset=0,
        ),
    ]
    with pytest.raises(MidiRenderError, match="no renderable notes"):
        render_midi_bytes(_perf(tiny_notes))


def test_render_midi_happy_path_returns_bytes():
    notes = [
        ExpressiveNote(
            score_note_id=f"n-{i}",
            pitch=60 + i,
            onset_beat=float(i),
            duration_beat=1.0,
            velocity=80,
            hand="rh",
            voice=1,
            timing_offset_ms=0.0,
            velocity_offset=0,
        )
        for i in range(4)
    ]
    out = render_midi_bytes(_perf(notes))
    assert out.startswith(b"MThd")
    assert len(out) > 50  # real MIDI has a non-trivial body
