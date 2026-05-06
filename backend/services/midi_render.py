"""Render a HumanizedPerformance to raw MIDI bytes via pretty_midi + mido.

Used by PipelineRunner to produce the MIDI payload shipped to the
ML engraver service.

Failure modes raise ``MidiRenderError`` rather than emitting a stub
MIDI file. A stub would travel to the engraver and come back as a
blank MusicXML — the same silent-failure shape this PR set out to
kill on the response side. The symmetric fix is to fail loudly on the
request side.

Phase 2 (Stream 2A) extends what we hand the engraver:

  * **KeySignature** events from ``metadata.key`` so the rendered score
    no longer guesses (or blanks) the accidentals.
  * **All ``tempo_map`` entries** (not just the first) so variable-
    tempo recordings keep their pulse all the way through.
  * **Marker text events** for each chord symbol, so the remote
    engraver has chord data to read until Phase 4 swaps in MusicXML.
  * **Cue Point text events** for downbeats from Beat This!, giving
    the engraver waveform-locked bar-line anchors.

Returns a :class:`RenderedMidi` carrying the bytes plus an
:class:`EmittedFeatures` summary so the runner can compute
``EngravedScoreData.includes_*`` from real content instead of hard-
coded ``False`` flags.
"""
from __future__ import annotations

import logging
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.contracts import HumanizedPerformance, beat_to_sec

log = logging.getLogger(__name__)

# Microseconds per quarter note for a given BPM — MIDI ``set_tempo`` payload.
def _bpm_to_us_per_qn(bpm: float) -> int:
    return max(1, int(round(60_000_000.0 / max(bpm, 1.0))))


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


@dataclass
class EmittedFeatures:
    """What the renderer actually wrote into the MIDI bytes.

    Computed from the inputs at emit time, not declared upfront. The
    runner reads this to populate ``EngravedScoreData.includes_*``
    flags so they reflect reality instead of being hard-coded ``False``.
    """
    key_signature: bool = False
    chord_symbols: bool = False
    pedal_marks: bool = False
    dynamics: bool = False
    tempo_changes: bool = False
    downbeats: bool = False
    chord_marker_count: int = 0
    downbeat_cue_count: int = 0
    tempo_change_count: int = 0
    pedal_event_count: int = 0


@dataclass
class RenderedMidi:
    """Bytes + a summary of what was actually written into them."""
    midi_bytes: bytes
    features: EmittedFeatures = field(default_factory=EmittedFeatures)


# ── Key-string parsing ─────────────────────────────────────────────────

# Pitch-class lookup for a leading note letter + accidental(s).
_PITCH_CLASS = {
    "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11,
}

# Mode keywords mapped to ``minor`` (True) / ``major`` (False). pretty_midi
# encodes key_number = pitch_class for major, +12 for minor. We collapse
# the church modes onto major/minor by their parallel — the rendered score
# loses the modal flavour, but keeps the correct accidentals.
_MINOR_MODES = {"minor", "min", "m", "aeolian", "dorian", "phrygian", "locrian"}
_MAJOR_MODES = {"major", "maj", "ionian", "lydian", "mixolydian", ""}


def _key_string_to_key_number(key: str) -> int | None:
    """Parse ``"C:major"`` / ``"F#:minor"`` / ``"Bb:dorian"`` into pretty_midi's key_number.

    pretty_midi.KeySignature uses ``key_number ∈ [0, 23]``: 0..11 for the
    twelve major keys (C..B), 12..23 for the minor keys (c..b). Returns
    ``None`` when the input doesn't parse — the caller falls back to
    not emitting a KeySignature, which is still better than emitting a
    wrong one.
    """
    if not key:
        return None
    # Accept "C:major", "C major", "Cmajor", "C", with optional accidentals.
    m = re.match(r"\s*([A-Ga-g])([#b]*)\s*[: ]?\s*([A-Za-z]*)\s*$", key)
    if not m:
        return None
    letter, accidentals, mode = m.group(1).upper(), m.group(2), m.group(3).lower()
    base = _PITCH_CLASS.get(letter)
    if base is None:
        return None
    pc = base
    for ch in accidentals:
        pc = (pc + 1) % 12 if ch == "#" else (pc - 1) % 12
    if mode in _MINOR_MODES:
        return pc + 12
    if mode in _MAJOR_MODES:
        return pc
    # Unknown mode token (e.g. user-supplied "C:weird" or "garbage") —
    # refuse rather than guess. A wrong key signature renders worse
    # than no key signature.
    return None


# ── Public renderer ────────────────────────────────────────────────────


def render_midi_bytes(perf: HumanizedPerformance) -> bytes:
    """Render and return raw MIDI bytes.

    Thin wrapper over :func:`render_midi` for callers that don't care
    which features were emitted. The runner uses :func:`render_midi`
    directly so it can populate ``EngravedScoreData.includes_*`` from
    the returned :class:`EmittedFeatures`.
    """
    return render_midi(perf).midi_bytes


def render_midi(perf: HumanizedPerformance) -> RenderedMidi:
    """Render the humanized performance to MIDI bytes + a feature summary.

    Raises ``MidiRenderError`` when pretty_midi/mido is unavailable or the
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

    try:
        import mido  # noqa: PLC0415 — bundled with pretty_midi
    except ImportError as exc:
        raise MidiRenderError(
            "mido is not installed; cannot inject tempo / marker / cue point "
            "meta events. mido ships as a pretty_midi dependency, so a missing "
            "import means the runtime is misconfigured."
        ) from exc

    metadata = perf.score.metadata
    tempo_map = metadata.tempo_map
    initial_bpm = tempo_map[0].bpm if tempo_map else 120.0
    midi_time_offset = tempo_map[0].time_sec if tempo_map else 0.0

    midi = pretty_midi.PrettyMIDI(initial_tempo=initial_bpm)
    ts = metadata.time_signature
    midi.time_signature_changes = [
        pretty_midi.TimeSignature(numerator=ts[0], denominator=ts[1], time=0.0)
    ]

    features = EmittedFeatures()

    # Key signature: parse "C:major" / "A:minor" → key_number; skip
    # silently when the string doesn't parse (preferable to a wrong key).
    key_number = _key_string_to_key_number(metadata.key)
    if key_number is not None:
        midi.key_signature_changes = [
            pretty_midi.KeySignature(key_number=key_number, time=0.0)
        ]
        features.key_signature = True

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
    pedal_count = 0
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
        pedal_count += 1

    if pedal_count:
        features.pedal_marks = True
        features.pedal_event_count = pedal_count

    midi.instruments.append(piano)

    if not piano.notes:
        raise MidiRenderError(
            f"humanized performance contained no renderable notes after "
            f"overlap resolution and min-duration filtering "
            f"(input expressive_notes={len(perf.expressive_notes)}). "
            f"Likely a transcription regression or silent audio; refusing "
            f"to send a stub MIDI to the engraver."
        )

    # Dynamics: pretty_midi has no native API for ``mf`` / ``cresc.``
    # markings; the engraver reads them from the upstream score. We only
    # report the flag here so the runner can stamp it on EngravedScoreData.
    if perf.expression.dynamics:
        features.dynamics = True

    # ── Write base MIDI, then post-process with mido ──────────────
    # Tempo changes, markers, and cue points all need MIDI meta events
    # in track 0. pretty_midi only exposes ``initial_tempo`` and has no
    # public marker/cue API, so we round-trip through mido to inject them.
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        midi.write(str(tmp_path))
        mid = mido.MidiFile(str(tmp_path))

        meta_events = _build_meta_events(
            mido,
            tempo_map=tempo_map,
            chord_symbols=metadata.chord_symbols,
            downbeats=metadata.downbeats,
            initial_time_offset=midi_time_offset,
            ticks_per_beat=mid.ticks_per_beat,
            initial_bpm=initial_bpm,
            features=features,
        )

        if meta_events:
            _inject_meta_events(mid, meta_events)
            mid.save(str(tmp_path))

        return RenderedMidi(midi_bytes=tmp_path.read_bytes(), features=features)
    finally:
        tmp_path.unlink(missing_ok=True)


# ── Internal helpers ───────────────────────────────────────────────────


def _build_meta_events(
    mido_module,
    *,
    tempo_map,
    chord_symbols,
    downbeats,
    initial_time_offset: float,
    ticks_per_beat: int,
    initial_bpm: float,
    features: EmittedFeatures,
) -> list:
    """Build a sorted list of (absolute_tick, MetaMessage) for injection.

    Returns absolute-tick MetaMessages so :func:`_inject_meta_events` can
    splice them into track 0 in correct delta-time order. The pretty_midi
    file already carries the *first* tempo (via ``initial_tempo``) and
    the time signature; we only add the *additional* tempo changes plus
    chord markers and downbeat cues.
    """
    events: list = []

    # Helper: convert an absolute time (seconds, in render-coordinates,
    # i.e. already relative to ``midi_time_offset``) into MIDI ticks
    # using a constant initial tempo. This is an approximation when the
    # tempo map varies, but it's the same shortcut pretty_midi takes
    # internally for ``set_tempo`` events written via mido — the engraver
    # interprets the meta events alongside the tempo changes, so the
    # result lines up to the user's ear.
    us_per_beat_initial = _bpm_to_us_per_qn(initial_bpm)
    seconds_per_tick = us_per_beat_initial / 1_000_000.0 / ticks_per_beat

    def _sec_to_tick(sec: float) -> int:
        if seconds_per_tick <= 0:
            return 0
        return max(0, int(round(sec / seconds_per_tick)))

    # Tempo changes — emit every entry except the first (already in the
    # ``initial_tempo``). Reset the tempo whenever the BPM differs from
    # the previous anchor; identical-BPM repeats are skipped to keep the
    # meta-event count down.
    tempo_change_count = 0
    if tempo_map and len(tempo_map) > 1:
        prev_bpm = tempo_map[0].bpm
        for entry in tempo_map[1:]:
            if abs(entry.bpm - prev_bpm) < 1e-3:
                continue
            tick = _sec_to_tick(entry.time_sec - initial_time_offset)
            events.append((
                tick,
                mido_module.MetaMessage(
                    "set_tempo",
                    tempo=_bpm_to_us_per_qn(entry.bpm),
                ),
            ))
            tempo_change_count += 1
            prev_bpm = entry.bpm
    if tempo_change_count:
        features.tempo_changes = True
        features.tempo_change_count = tempo_change_count

    # Chord symbols → Marker text events (one per chord change). The
    # remote engraver's MIDI parser reads markers as text and surfaces
    # them as comments / chord-symbol annotations. Phase 4 will replace
    # this with proper MusicXML chord rendering, but for now this keeps
    # the data alive across the engrave boundary.
    chord_count = 0
    for chord in chord_symbols or []:
        try:
            time_sec = beat_to_sec(chord.beat, tempo_map) - initial_time_offset
        except ValueError:
            continue
        tick = _sec_to_tick(max(0.0, time_sec))
        # Marker payload is just the Harte-style label ("C:maj7", "Am",
        # …). Truncate to 127 chars — MIDI text events are length-byte
        # encoded and some engravers reject longer payloads.
        text = (chord.label or "").strip()[:127]
        if not text:
            continue
        events.append((tick, mido_module.MetaMessage("marker", text=text)))
        chord_count += 1
    if chord_count:
        features.chord_symbols = True
        features.chord_marker_count = chord_count

    # Downbeats → Cue Point text events. ``cue_marker`` is mido's name
    # for MIDI meta event FF 07. Bar numbers are 1-indexed for human
    # readability when an engraver dumps them as text; engravers that
    # actually use the cue points read the *time* and ignore the label.
    downbeat_count = 0
    for i, db_sec in enumerate(downbeats or []):
        sec_in_render = float(db_sec) - initial_time_offset
        if sec_in_render < 0:
            continue
        tick = _sec_to_tick(sec_in_render)
        events.append((
            tick,
            mido_module.MetaMessage("cue_marker", text=f"bar{i + 1}"),
        ))
        downbeat_count += 1
    if downbeat_count:
        features.downbeats = True
        features.downbeat_cue_count = downbeat_count

    events.sort(key=lambda e: e[0])
    return events


def _inject_meta_events(mid, abs_events: list) -> None:
    """Splice absolute-tick meta events into track 0 with correct deltas.

    pretty_midi already wrote track 0 with ``set_tempo`` (initial),
    ``time_signature``, ``key_signature``, and the trailing
    ``end_of_track``. We rebuild the track in absolute-tick form, merge
    the new events in, then re-encode as deltas.
    """
    if not mid.tracks:
        return
    import mido  # noqa: PLC0415

    track0 = mid.tracks[0]
    abs_track: list[tuple[int, Any]] = []
    end_of_track_msg = None
    cursor = 0
    for msg in track0:
        cursor += msg.time
        if msg.type == "end_of_track":
            end_of_track_msg = msg
            continue
        abs_track.append((cursor, msg))
    end_of_track_tick = cursor

    abs_track.extend(abs_events)
    end_of_track_tick = max(end_of_track_tick, max((t for t, _ in abs_events), default=0))

    # Stable sort: existing pretty_midi events keep their relative order
    # against newly injected ones at the same tick (so e.g. an injected
    # ``set_tempo`` lands AFTER pretty_midi's existing time_signature at
    # tick 0 — engravers don't care, but the output is deterministic).
    abs_track.sort(key=lambda e: e[0])

    rebuilt = mido.MidiTrack()
    prev_tick = 0
    for tick, msg in abs_track:
        delta = max(0, tick - prev_tick)
        # ``msg.copy(time=delta)`` works for both Message and MetaMessage.
        rebuilt.append(msg.copy(time=delta))
        prev_tick = tick
    eot = end_of_track_msg or mido.MetaMessage("end_of_track")
    rebuilt.append(eot.copy(time=max(0, end_of_track_tick - prev_tick)))
    mid.tracks[0] = rebuilt
