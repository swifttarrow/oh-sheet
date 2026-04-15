"""Materialize raw MIDI bytes from a ``TranscriptionResult`` (claim-check or rebuild)."""
from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

from backend.contracts import TranscriptionResult

if TYPE_CHECKING:
    from backend.storage.base import BlobStore

log = logging.getLogger(__name__)


def serialize_transcription_to_midi_bytes(txr: TranscriptionResult) -> bytes | None:
    """Rebuild a single-piano ``.mid`` from contract tracks + analysis (best-effort).

    Uses the first tempo-map anchor BPM as ``initial_tempo`` and the analysis
    time signature. Variable-tempo maps are not fully encoded — sufficient for
    HF stubs and arrange fallbacks.
    """
    if not txr.midi_tracks:
        return None
    try:
        import pretty_midi  # noqa: PLC0415
    except ImportError:
        log.warning("pretty_midi not installed — cannot serialize TranscriptionResult to MIDI")
        return None

    tempo_map = txr.analysis.tempo_map or []
    initial_bpm = float(tempo_map[0].bpm) if tempo_map else 120.0
    pm = pretty_midi.PrettyMIDI(initial_tempo=initial_bpm)
    instrument = pretty_midi.Instrument(program=0)
    for track in txr.midi_tracks:
        for n in track.notes:
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=int(max(1, min(127, n.velocity))),
                    pitch=int(n.pitch),
                    start=float(n.onset_sec),
                    end=float(max(n.offset_sec, n.onset_sec + 0.01)),
                ),
            )
    if not instrument.notes:
        return None
    pm.instruments.append(instrument)
    num, den = txr.analysis.time_signature
    pm.time_signature_changes = [
        pretty_midi.TimeSignature(numerator=int(num), denominator=int(den), time=0.0),
    ]
    buf = io.BytesIO()
    try:
        pm.write(buf)
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to serialize rebuilt MIDI: %s", exc)
        return None
    return buf.getvalue()


def materialize_transcription_midi_bytes(
    txr: TranscriptionResult,
    blob_store: BlobStore | None,
) -> bytes:
    """Return canonical input MIDI bytes for HF arrange.

    Prefer ``transcription_midi_uri`` via ``blob_store``; otherwise rebuild from
    ``midi_tracks`` (requires ``pretty_midi``).
    """
    if txr.transcription_midi_uri:
        if blob_store is None:
            raise ValueError(
                "transcription_midi_uri is set but blob_store is None — cannot load MIDI bytes",
            )
        return blob_store.get_bytes(txr.transcription_midi_uri)
    rebuilt = serialize_transcription_to_midi_bytes(txr)
    if rebuilt is None:
        raise ValueError("Could not materialize MIDI bytes (no URI and serialization failed)")
    return rebuilt
