"""Parse model-output MIDI into a ``TranscriptionResult`` (notes + template analysis)."""
from __future__ import annotations

import io

from backend.contracts import (
    SCHEMA_VERSION,
    QualitySignal,
    TranscriptionResult,
)
from backend.services.pretty_midi_tracks import midi_tracks_from_pretty_midi


def transcription_from_midi_bytes(
    data: bytes,
    template: TranscriptionResult,
    *,
    extra_warnings: list[str] | None = None,
) -> TranscriptionResult:
    """Parse ``data`` as MIDI; keep ``template`` harmonic analysis and quality shape."""
    import pretty_midi  # noqa: PLC0415

    try:
        pm = pretty_midi.PrettyMIDI(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"pretty_midi failed to parse HF output MIDI: {exc}") from exc

    midi_tracks = midi_tracks_from_pretty_midi(pm)
    if not midi_tracks:
        raise ValueError("HF output MIDI contained no note tracks")

    warns = list(template.quality.warnings)
    if extra_warnings:
        warns.extend(extra_warnings)

    return TranscriptionResult(
        schema_version=SCHEMA_VERSION,
        midi_tracks=midi_tracks,
        analysis=template.analysis,
        quality=QualitySignal(
            overall_confidence=template.quality.overall_confidence,
            warnings=warns,
        ),
        transcription_midi_uri=template.transcription_midi_uri,
    )
