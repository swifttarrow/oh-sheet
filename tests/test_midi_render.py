"""Tests for backend.services.midi_render.

The critical contract is symmetric to ``_looks_like_stub`` on the
response side: producing a stub MIDI on the request side would
round-trip as a blank MusicXML (and get caught downstream anyway, but
the error message would point at the wrong layer). Fail loudly here.
"""
from __future__ import annotations

import io
import sys

import pytest
from shared.contracts import (
    DynamicMarking,
    ExpressionMap,
    ExpressiveNote,
    HumanizedPerformance,
    PedalEvent,
    PianoScore,
    QualitySignal,
    ScoreChordEvent,
    ScoreMetadata,
    ScoreNote,
    TempoMapEntry,
)

from backend.services.midi_render import (
    MidiRenderError,
    _key_string_to_key_number,
    render_midi,
    render_midi_bytes,
)


def _perf(
    expressive: list[ExpressiveNote],
    score_notes: list[ScoreNote] | None = None,
    *,
    key: str = "C:major",
    tempo_map: list[TempoMapEntry] | None = None,
    chord_symbols: list[ScoreChordEvent] | None = None,
    downbeats: list[float] | None = None,
    pedal_events: list[PedalEvent] | None = None,
    dynamics: list[DynamicMarking] | None = None,
) -> HumanizedPerformance:
    return HumanizedPerformance(
        expressive_notes=expressive,
        expression=ExpressionMap(
            pedal_events=pedal_events or [],
            dynamics=dynamics or [],
        ),
        score=PianoScore(
            right_hand=score_notes or [],
            left_hand=[],
            metadata=ScoreMetadata(
                key=key,
                time_signature=(4, 4),
                tempo_map=tempo_map or [TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
                difficulty="intermediate",
                chord_symbols=chord_symbols or [],
                downbeats=downbeats or [],
            ),
        ),
        quality=QualitySignal(overall_confidence=0.9, warnings=[]),
    )


def _basic_notes(n: int = 4) -> list[ExpressiveNote]:
    return [
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
        for i in range(n)
    ]


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
    out = render_midi_bytes(_perf(_basic_notes()))
    assert out.startswith(b"MThd")
    assert len(out) > 50  # real MIDI has a non-trivial body


# ── Stream 2A: feature emission ──────────────────────────────────────


@pytest.mark.parametrize(
    "key,expected",
    [
        ("C:major", 0),
        ("C:minor", 12),
        ("F#:major", 6),
        ("F#:minor", 18),
        ("Bb:major", 10),
        ("A:minor", 21),
        ("Cb:major", 11),  # Cb == B
        # Modes collapse onto major/minor by their parallel.
        ("D:dorian", 14),  # dorian → minor
        ("G:mixolydian", 7),  # mixolydian → major
        ("C", 0),  # bare letter, treated as major
        ("not a key", None),
        ("", None),
    ],
)
def test_key_string_to_key_number_parses_common_forms(key, expected):
    assert _key_string_to_key_number(key) == expected


def test_render_emits_key_signature_from_metadata_key():
    """A10: pretty_midi.KeySignature ships in the rendered bytes."""
    import mido  # noqa: PLC0415

    rendered = render_midi(_perf(_basic_notes(), key="A:minor"))
    assert rendered.features.key_signature is True

    mid = mido.MidiFile(file=io.BytesIO(rendered.midi_bytes))
    key_sigs = [m for tr in mid.tracks for m in tr if m.type == "key_signature"]
    assert key_sigs, "expected a key_signature meta event in the MIDI"
    # mido encodes A minor as 'Am' (or similar minor convention).
    assert "m" in key_sigs[0].key.lower() or key_sigs[0].key == "Am"


def test_unparseable_key_falls_through_silently():
    """Wrong key is worse than no key — emit nothing rather than guess."""
    rendered = render_midi(_perf(_basic_notes(), key="garbage"))
    assert rendered.features.key_signature is False
    import mido  # noqa: PLC0415

    mid = mido.MidiFile(file=io.BytesIO(rendered.midi_bytes))
    key_sigs = [m for tr in mid.tracks for m in tr if m.type == "key_signature"]
    assert not key_sigs


def test_render_emits_all_tempo_map_entries():
    """A10: every tempo_map anchor (after the first) becomes a set_tempo event."""
    import mido  # noqa: PLC0415

    tempo_map = [
        TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0),
        TempoMapEntry(time_sec=2.0, beat=4.0, bpm=90.0),
        TempoMapEntry(time_sec=4.0, beat=7.0, bpm=140.0),
    ]
    rendered = render_midi(_perf(_basic_notes(), tempo_map=tempo_map))

    assert rendered.features.tempo_changes is True
    assert rendered.features.tempo_change_count == 2

    mid = mido.MidiFile(file=io.BytesIO(rendered.midi_bytes))
    tempo_events = [m for tr in mid.tracks for m in tr if m.type == "set_tempo"]
    # pretty_midi writes the initial tempo plus our 2 injected changes.
    assert len(tempo_events) >= 3
    bpms = sorted(60_000_000 // m.tempo for m in tempo_events)
    # Expect 90, 120, 140 BPM (rounded — set_tempo is microseconds/qn).
    assert 89 <= bpms[0] <= 91
    assert 119 <= bpms[1] <= 121
    assert 139 <= bpms[2] <= 141


def test_identical_tempo_repeats_collapse_to_no_change():
    """Two anchors at the same BPM shouldn't generate a tempo event."""
    tempo_map = [
        TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0),
        TempoMapEntry(time_sec=2.0, beat=4.0, bpm=120.0),
    ]
    rendered = render_midi(_perf(_basic_notes(), tempo_map=tempo_map))
    assert rendered.features.tempo_changes is False
    assert rendered.features.tempo_change_count == 0


def test_render_emits_chord_symbols_as_marker_events():
    """Chord symbols ride on MIDI Marker meta events (FF 06)."""
    import mido  # noqa: PLC0415

    chords = [
        ScoreChordEvent(beat=0.0, duration_beat=2.0, label="C:maj7", root=0),
        ScoreChordEvent(beat=2.0, duration_beat=2.0, label="A:min", root=9),
        ScoreChordEvent(beat=4.0, duration_beat=2.0, label="F:maj", root=5),
    ]
    rendered = render_midi(_perf(_basic_notes(), chord_symbols=chords))

    assert rendered.features.chord_symbols is True
    assert rendered.features.chord_marker_count == 3

    mid = mido.MidiFile(file=io.BytesIO(rendered.midi_bytes))
    markers = [m for tr in mid.tracks for m in tr if m.type == "marker"]
    assert len(markers) == 3
    assert {m.text for m in markers} == {"C:maj7", "A:min", "F:maj"}


def test_empty_chord_label_is_skipped():
    """Whitespace / empty labels shouldn't pollute the marker stream."""
    chords = [
        ScoreChordEvent(beat=0.0, duration_beat=2.0, label="   ", root=0),
        ScoreChordEvent(beat=2.0, duration_beat=2.0, label="C", root=0),
    ]
    rendered = render_midi(_perf(_basic_notes(), chord_symbols=chords))
    assert rendered.features.chord_marker_count == 1


def test_render_emits_downbeats_as_cue_points():
    """Beat This! downbeats become MIDI Cue Point meta events (FF 07)."""
    import mido  # noqa: PLC0415

    rendered = render_midi(_perf(_basic_notes(), downbeats=[0.0, 2.0, 4.0]))

    assert rendered.features.downbeats is True
    assert rendered.features.downbeat_cue_count == 3

    mid = mido.MidiFile(file=io.BytesIO(rendered.midi_bytes))
    cues = [m for tr in mid.tracks for m in tr if m.type == "cue_marker"]
    assert len(cues) == 3
    assert [c.text for c in cues] == ["bar1", "bar2", "bar3"]


def test_render_skips_downbeat_cues_when_field_empty():
    """No downbeats from the tracker → no cue_marker events, flag stays False."""
    import mido  # noqa: PLC0415

    rendered = render_midi(_perf(_basic_notes(), downbeats=[]))
    assert rendered.features.downbeats is False
    assert rendered.features.downbeat_cue_count == 0
    mid = mido.MidiFile(file=io.BytesIO(rendered.midi_bytes))
    cues = [m for tr in mid.tracks for m in tr if m.type == "cue_marker"]
    assert not cues


def test_pedal_events_set_includes_pedal_marks_flag():
    """Sustain pedal events emit CC64 and surface on the features flag."""
    pedal = [PedalEvent(onset_beat=0.0, offset_beat=2.0, type="sustain")]
    rendered = render_midi(_perf(_basic_notes(), pedal_events=pedal))
    assert rendered.features.pedal_marks is True
    assert rendered.features.pedal_event_count == 1


def test_dynamics_marking_sets_dynamics_flag():
    """Dynamics live on the score, not the MIDI bytes — flag still flips."""
    dynamics = [DynamicMarking(beat=0.0, type="mf")]
    rendered = render_midi(_perf(_basic_notes(), dynamics=dynamics))
    assert rendered.features.dynamics is True


def test_render_midi_bytes_legacy_wrapper_returns_just_bytes():
    """Callers that don't care about features still get raw bytes back."""
    out = render_midi_bytes(_perf(_basic_notes()))
    assert isinstance(out, bytes)
    assert out.startswith(b"MThd")
