"""MIDI artifact construction and serialization for blob storage.

Helpers that build ``pretty_midi.PrettyMIDI`` objects from internal
``NoteEvent`` lists and serialize them to raw ``.mid`` bytes. The blob
MIDI is a debugging artifact (not authoritative — the contract carries
the real tracks). Extracted from ``transcribe.py``.
"""
from __future__ import annotations

import io
import logging
from typing import Any

from backend.contracts import InstrumentRole
from backend.services.transcription_cleanup import NoteEvent

log = logging.getLogger(__name__)


def _rebuild_blob_midi(
    events: list[NoteEvent],
    *,
    initial_bpm: float,
) -> Any:
    """Build a fresh pretty_midi from a flat event list for blob storage.

    ``initial_bpm`` becomes the MIDI ``set_tempo`` meta-event on
    track 0. basic-pitch's own ``note_events_to_midi`` hard-codes
    120 BPM, which mismatches every piece of music that isn't at
    120, and notation importers (MuseScore's MIDI wizard in
    particular) treat the declared tempo as a hint and re-infer
    metric structure from note density when it looks wrong —
    occasionally landing on the wrong time signature in the
    process. Wiring the librosa-derived tempo through here makes
    the blob MIDI agree with its notes, so importers have no
    reason to re-infer. Callers pass ``audio_tempo_map[0].bpm``
    when available; the 120 default matches basic-pitch for the
    (rare) case where beat tracking failed.

    The time-signature meta event is also set explicitly — pretty_midi
    would emit a default 4/4 anyway, but making it explicit documents
    intent and keeps any future pretty_midi behavior change from
    silently dropping the event.

    We build the pretty_midi directly (no basic-pitch dependency) —
    ``PrettyMIDI(initial_tempo=...)`` wires the seconds-per-tick map
    through the MIDI ``set_tempo`` meta event, and the notes are a
    straight translation of the ``NoteEvent`` tuple. Per-note pitch
    bends are intentionally dropped: the blob MIDI is a debugging
    artifact (not authoritative — the contract carries the tracks),
    and basic-pitch's own encoding bloats the file with one
    instrument per pitch-bend group. Keeping everything in a single
    piano instrument makes the blob easier to inspect in MuseScore /
    Logic without hiding the real transcription data.

    Avoiding ``basic_pitch.note_creation.note_events_to_midi`` here
    also means this code path works on the CI dev install (which
    only pulls in ``.[dev]``, not ``.[basic-pitch]``) — the stems
    tests monkeypatch ``_basic_pitch_single_pass`` but still reach
    the blob rebuild, so a missing basic_pitch import would silently
    fall back to a blank PrettyMIDI and drop the audio-derived tempo
    on the floor.

    Returns ``None`` on any failure (missing pretty_midi, empty
    events) so callers can fall back to a pre-existing pretty_midi
    without sinking blob persistence.
    """
    if not events:
        return None
    try:
        import pretty_midi  # noqa: PLC0415 — optional dep
    except ImportError:
        return None
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(initial_bpm))
    instrument = pretty_midi.Instrument(program=0)
    for start, end, pitch, amplitude, _pitch_bend in events:
        velocity = int(round(127 * float(amplitude)))
        velocity = max(1, min(127, velocity))
        instrument.notes.append(
            pretty_midi.Note(
                velocity=velocity,
                pitch=int(pitch),
                start=float(start),
                end=float(end),
            )
        )
    pm.instruments.append(instrument)
    pm.time_signature_changes = [
        pretty_midi.TimeSignature(numerator=4, denominator=4, time=0.0)
    ]
    return pm


def _combined_midi_from_events(
    events_by_role: dict[InstrumentRole, list[NoteEvent]],
    fallback: Any,
    *,
    initial_bpm: float = 120.0,
) -> Any:
    """Build a single pretty_midi from the concatenated per-role events.

    Used on the Demucs path where we have three independent Basic
    Pitch passes (vocals / bass / other) and need *one* ``.mid``
    artifact for blob storage. We flatten the per-role note lists
    into time-sorted order and delegate to :func:`_rebuild_blob_midi`,
    which handles the pretty_midi construction + tempo / TS meta
    event wiring. The blob MIDI is a debugging artifact, not
    authoritative, so collapsing the role split there is fine (the
    contract ``TranscriptionResult`` keeps the split intact).

    On any failure we fall back to ``fallback`` (typically the last
    pretty_midi we produced in the loop) so blob persistence stays
    best-effort.
    """
    combined: list[NoteEvent] = []
    for events in events_by_role.values():
        combined.extend(events)
    if not combined:
        return fallback
    combined.sort(key=lambda e: (e[0], e[2]))
    pm = _rebuild_blob_midi(combined, initial_bpm=initial_bpm)
    return pm if pm is not None else fallback


def _serialize_pretty_midi(pm: Any) -> bytes | None:
    """Serialize a pretty_midi.PrettyMIDI to raw .mid bytes.

    pretty_midi >= 0.2.10 accepts a file-like object in ``write()`` and
    forwards it to mido as ``file=...``, so we can avoid a temp file.
    Returns None on any failure — blob persistence is best-effort and must
    never break transcription.
    """
    try:
        buf = io.BytesIO()
        pm.write(buf)
        return buf.getvalue()
    except Exception as exc:  # noqa: BLE001 — best-effort serialization
        log.warning("Failed to serialize pretty_midi for blob storage: %s", exc)
        return None
