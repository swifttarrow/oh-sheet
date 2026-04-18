"""Render a HumanizedPerformance to raw MIDI bytes via pretty_midi.

Used by PipelineRunner to produce the MIDI payload shipped to the
ML engraver service.

Failure modes raise ``MidiRenderError`` rather than emitting a stub
MIDI file. A stub would travel to the engraver and come back as a
blank MusicXML — the same silent-failure shape this PR set out to
kill on the response side. The symmetric fix is to fail loudly on the
request side.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from backend.contracts import HumanizedPerformance, beat_to_sec

log = logging.getLogger(__name__)


class MidiRenderError(RuntimeError):
    """Raised when a HumanizedPerformance cannot be rendered to MIDI.

    Two cases:
      * ``pretty_midi`` isn't importable — deploy configuration error
        (pretty_midi is a top-level dependency in pyproject.toml).
      * The performance contains no renderable notes after overlap
        resolution and min-duration filtering — likely a transcription
        regression or a silent audio file. Either way, a blank engrave
        isn't a useful artifact, so fail loudly.
    """


def render_midi_bytes(perf: HumanizedPerformance) -> bytes:
    """Render the humanized performance to MIDI bytes via pretty_midi.

    Raises ``MidiRenderError`` when pretty_midi is unavailable or the
    performance contains no renderable notes.
    """
    try:
        import pretty_midi  # noqa: PLC0415 — optional dep
    except ImportError as exc:
        raise MidiRenderError(
            "pretty_midi is not installed; cannot render MIDI. "
            "This is a deploy configuration error — pretty_midi is a "
            "top-level dependency in pyproject.toml. Refusing to send a "
            "stub MIDI to the engraver (would round-trip as blank MusicXML)."
        ) from exc

    tempo_map = perf.score.metadata.tempo_map
    initial_bpm = tempo_map[0].bpm if tempo_map else 120.0
    midi_time_offset = tempo_map[0].time_sec if tempo_map else 0.0

    midi = pretty_midi.PrettyMIDI(initial_tempo=initial_bpm)
    ts = perf.score.metadata.time_signature
    midi.time_signature_changes = [
        pretty_midi.TimeSignature(numerator=ts[0], denominator=ts[1], time=0.0)
    ]

    piano = pretty_midi.Instrument(program=0, name="Piano")

    raw_notes: list[tuple[int, float, float, int]] = []
    for en in perf.expressive_notes:
        try:
            onset = beat_to_sec(en.onset_beat, tempo_map) - midi_time_offset
            offset = beat_to_sec(en.onset_beat + en.duration_beat, tempo_map) - midi_time_offset
        except ValueError:
            continue
        # timing_offset_ms is an *onset-only* nudge — the release boundary
        # stays on the metronomic grid, so a late-struck note plays shorter
        # and an early-struck note plays longer. Mirrors humanize._humanize_timing,
        # which models downbeat anticipation / backbeat push as attack-time
        # gestures with no release component.
        onset += en.timing_offset_ms / 1000.0
        onset = max(0.0, onset)
        offset = max(onset + 0.01, offset)
        raw_notes.append((en.pitch, onset, offset, max(1, min(127, en.velocity))))

    # Resolve same-pitch overlaps — MIDI can't have two notes-on for the same key
    by_pitch: dict[int, list[int]] = {}
    for i, (p, _s, _e, _v) in enumerate(raw_notes):
        by_pitch.setdefault(p, []).append(i)
    for indices in by_pitch.values():
        indices.sort(key=lambda i: raw_notes[i][1])
        for j in range(1, len(indices)):
            pp, ps, pe, pv = raw_notes[indices[j - 1]]
            _cp, cs, _ce, _cv = raw_notes[indices[j]]
            if pe > cs:
                raw_notes[indices[j - 1]] = (pp, ps, cs, pv)

    MIN_NOTE_DUR = 0.03
    for pitch, start, end, vel in raw_notes:
        if end - start < MIN_NOTE_DUR:
            continue
        piano.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end))

    # Pedal events → GM control changes. Sustain = CC64, sostenuto = CC66,
    # una corda = CC67. All three use the standard on=127 / off=0 encoding
    # (below 64 disengages on most synths, but 0 is the unambiguous convention).
    _PEDAL_CC = {"sustain": 64, "sostenuto": 66, "una_corda": 67}
    for pedal in perf.expression.pedal_events:
        cc_num = _PEDAL_CC.get(pedal.type)
        if cc_num is None:
            continue
        try:
            on_sec = beat_to_sec(pedal.onset_beat, tempo_map) - midi_time_offset
            off_sec = beat_to_sec(pedal.offset_beat, tempo_map) - midi_time_offset
        except ValueError:
            continue
        piano.control_changes.append(
            pretty_midi.ControlChange(number=cc_num, value=127, time=max(0.0, on_sec))
        )
        piano.control_changes.append(
            pretty_midi.ControlChange(number=cc_num, value=0, time=max(0.0, off_sec))
        )

    midi.instruments.append(piano)

    if not piano.notes:
        raise MidiRenderError(
            f"humanized performance contained no renderable notes after "
            f"overlap resolution and min-duration filtering "
            f"(input expressive_notes={len(perf.expressive_notes)}). "
            f"Likely a transcription regression or silent audio; refusing "
            f"to send a stub MIDI to the engraver."
        )

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        midi.write(str(tmp_path))
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)
