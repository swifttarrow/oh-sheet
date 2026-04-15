"""Shared PrettyMIDI → contract ``MidiTrack`` conversion (runner + HF arrange)."""
from __future__ import annotations

from typing import Any

from backend.contracts import (
    HarmonicAnalysis,
    InstrumentRole,
    MidiTrack,
    Note,
    TempoMapEntry,
)


def gm_program_to_role(program: int, is_drum: bool) -> InstrumentRole:
    if is_drum:
        return InstrumentRole.OTHER
    if program < 8:
        return InstrumentRole.PIANO
    if 32 <= program <= 39:
        return InstrumentRole.BASS
    if 72 <= program <= 79:
        return InstrumentRole.MELODY
    return InstrumentRole.CHORDS


def midi_tracks_from_pretty_midi(pm: Any) -> list[MidiTrack]:
    """Fold each non-empty ``pretty_midi.Instrument`` into a contract ``MidiTrack``."""
    midi_tracks: list[MidiTrack] = []
    for instrument in pm.instruments:
        notes = [
            Note(
                pitch=int(n.pitch),
                onset_sec=float(n.start),
                offset_sec=float(max(n.end, n.start + 0.01)),
                velocity=int(max(1, min(127, n.velocity))),
            )
            for n in instrument.notes
        ]
        if not notes:
            continue
        midi_tracks.append(
            MidiTrack(
                notes=notes,
                instrument=gm_program_to_role(int(instrument.program), bool(instrument.is_drum)),
                program=None if instrument.is_drum else int(instrument.program),
                confidence=0.9,
            ),
        )
    return midi_tracks


def harmonic_analysis_from_pretty_midi(pm: Any) -> HarmonicAnalysis:
    """Recover tempo map, time signature, and key from ``pretty_midi.PrettyMIDI``."""
    import pretty_midi  # noqa: PLC0415

    tempo_times, tempo_bpms = pm.get_tempo_changes()
    tempo_map: list[TempoMapEntry] = []
    if len(tempo_times) > 0:
        beat_cursor = 0.0
        prev_time = 0.0
        prev_bpm = float(tempo_bpms[0])
        for t, bpm in zip(tempo_times, tempo_bpms):
            t = float(t)
            bpm = float(bpm)
            beat_cursor += (t - prev_time) * (prev_bpm / 60.0)
            tempo_map.append(TempoMapEntry(time_sec=t, beat=beat_cursor, bpm=bpm))
            prev_time = t
            prev_bpm = bpm
    if not tempo_map:
        tempo_map = [TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)]

    time_signature: tuple[int, int] = (4, 4)
    if pm.time_signature_changes:
        first = pm.time_signature_changes[0]
        time_signature = (int(first.numerator), int(first.denominator))

    key = "C:major"
    if pm.key_signature_changes:
        try:
            key_number = int(pm.key_signature_changes[0].key_number)
            key = pretty_midi.key_number_to_key_name(key_number).replace(" ", ":")
        except Exception:  # noqa: BLE001
            pass

    return HarmonicAnalysis(
        key=key,
        time_signature=time_signature,
        tempo_map=tempo_map,
        chords=[],
        sections=[],
    )
