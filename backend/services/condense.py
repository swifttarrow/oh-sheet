"""Condense stage — merge all transcription tracks into one piano-oriented score.

Conceptually matches ``merge_midi_to_piano.py``:

* Ignore per-track GM ``program`` / role when building the merged stream (like
  dropping ``program_change`` and folding onto one channel).
* Order every note by absolute time (like ``mido.merge_tracks``).
* Emit a single piano-oriented ``PianoScore`` (GM Acoustic Grand is implicit
  in the sheet contract).

Inputs are ``TranscriptionResult`` notes in seconds plus ``analysis.tempo_map``;
timing is converted with ``sec_to_beat`` and **not** quantized so beat
positions stay faithful to the source map (parallel to preserving MIDI ticks).
Hands are split at middle C so the result fits the two-staff ``PianoScore``
shape.
"""
from __future__ import annotations

import asyncio
import logging

from backend.contracts import (
    SCHEMA_VERSION,
    Difficulty,
    MidiTrack,
    Note,
    PianoScore,
    RealtimeChordEvent,
    ScoreChordEvent,
    ScoreMetadata,
    ScoreNote,
    ScoreSection,
    Section,
    TempoMapEntry,
    TranscriptionResult,
    sec_to_beat,
)

log = logging.getLogger(__name__)

# Middle C — same default split as ``arrange`` for non-melody/bass material.
SPLIT_PITCH = 60
MIN_DURATION_BEAT = 1e-4
# Merged MIDI can be very polyphonic; keep a generous cap before dropping voices.
MAX_VOICES_PER_HAND = 16


def _chord_to_score_chord(
    chord: RealtimeChordEvent,
    tempo_map: list[TempoMapEntry],
) -> ScoreChordEvent:
    onset = sec_to_beat(chord.time_sec, tempo_map)
    end = sec_to_beat(chord.time_sec + chord.duration_sec, tempo_map)
    return ScoreChordEvent(
        beat=onset,
        duration_beat=max(end - onset, MIN_DURATION_BEAT),
        label=chord.label,
        root=chord.root,
        confidence=chord.confidence,
    )


def _section_to_score_section(
    section: Section,
    tempo_map: list[TempoMapEntry],
) -> ScoreSection:
    return ScoreSection(
        start_beat=sec_to_beat(section.start_sec, tempo_map),
        end_beat=sec_to_beat(section.end_sec, tempo_map),
        label=section.label,
    )


def _note_to_beat_tuple(
    n: Note,
    tempo_map: list[TempoMapEntry],
) -> tuple[int, float, float, int]:
    onset_b = sec_to_beat(n.onset_sec, tempo_map)
    offset_b = sec_to_beat(n.offset_sec, tempo_map)
    dur = max(offset_b - onset_b, MIN_DURATION_BEAT)
    return (n.pitch, onset_b, dur, n.velocity)


def _merge_tracks_chronologically(
    tracks: list[MidiTrack],
    tempo_map: list[TempoMapEntry],
) -> list[tuple[int, float, float, int]]:
    merged: list[tuple[int, float, float, int]] = []
    for track in tracks:
        for n in track.notes:
            merged.append(_note_to_beat_tuple(n, tempo_map))
    merged.sort(key=lambda t: (t[1], t[0]))
    return merged


def _split_hands(
    merged: list[tuple[int, float, float, int]],
) -> tuple[list[tuple[int, float, float, int]], list[tuple[int, float, float, int]]]:
    rh: list[tuple[int, float, float, int]] = []
    lh: list[tuple[int, float, float, int]] = []
    for pitch, onset, dur, vel in merged:
        (rh if pitch >= SPLIT_PITCH else lh).append((pitch, onset, dur, vel))
    return rh, lh


def _assign_voices(
    notes: list[tuple[int, float, float, int]],
    max_voices: int,
) -> list[tuple[int, float, float, int, int]]:
    """Greedy voice assignment in onset order (no quantization)."""
    ordered = sorted(notes, key=lambda n: (n[1], n[0]))
    voice_ends: list[float] = []
    out: list[tuple[int, float, float, int, int]] = []
    for pitch, onset, dur, vel in ordered:
        assigned: int | None = None
        for vi, end in enumerate(voice_ends):
            if onset + 1e-9 >= end:
                assigned = vi
                voice_ends[vi] = onset + dur
                break
        if assigned is None:
            if len(voice_ends) < max_voices:
                assigned = len(voice_ends)
                voice_ends.append(onset + dur)
            else:
                continue
        out.append((pitch, onset, dur, vel, assigned + 1))
    return out


def _condense_sync(payload: TranscriptionResult, difficulty: Difficulty) -> PianoScore:
    analysis = payload.analysis
    tempo_map = analysis.tempo_map or [TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)]

    tracks = list(payload.midi_tracks)
    if not tracks:
        log.info("Condense: no tracks — empty score")
        return PianoScore(
            schema_version=SCHEMA_VERSION,
            right_hand=[],
            left_hand=[],
            metadata=ScoreMetadata(
                key=analysis.key,
                time_signature=analysis.time_signature,
                tempo_map=tempo_map,
                difficulty=difficulty,
                sections=[_section_to_score_section(s, tempo_map) for s in analysis.sections],
                chord_symbols=[_chord_to_score_chord(c, tempo_map) for c in analysis.chords],
            ),
        )

    merged = _merge_tracks_chronologically(tracks, tempo_map)
    rh_raw, lh_raw = _split_hands(merged)
    rh_voiced = _assign_voices(rh_raw, MAX_VOICES_PER_HAND)
    lh_voiced = _assign_voices(lh_raw, MAX_VOICES_PER_HAND)

    right_hand = [
        ScoreNote(
            id=f"rh-{i:04d}",
            pitch=pitch,
            onset_beat=onset,
            duration_beat=dur,
            velocity=max(1, min(127, vel)),
            voice=voice,
        )
        for i, (pitch, onset, dur, vel, voice) in enumerate(rh_voiced)
    ]
    left_hand = [
        ScoreNote(
            id=f"lh-{i:04d}",
            pitch=pitch,
            onset_beat=onset,
            duration_beat=dur,
            velocity=max(1, min(127, vel)),
            voice=voice,
        )
        for i, (pitch, onset, dur, vel, voice) in enumerate(lh_voiced)
    ]

    log.info("Condensed: RH=%d notes, LH=%d notes", len(right_hand), len(left_hand))

    return PianoScore(
        schema_version=SCHEMA_VERSION,
        right_hand=right_hand,
        left_hand=left_hand,
        metadata=ScoreMetadata(
            key=analysis.key,
            time_signature=analysis.time_signature,
            tempo_map=tempo_map,
            difficulty=difficulty,
            sections=[_section_to_score_section(s, tempo_map) for s in analysis.sections],
            chord_symbols=[_chord_to_score_chord(c, tempo_map) for c in analysis.chords],
        ),
    )


class CondenseService:
    name = "condense"

    async def run(
        self,
        payload: TranscriptionResult,
        *,
        difficulty: Difficulty = "intermediate",
    ) -> PianoScore:
        return await asyncio.to_thread(_condense_sync, payload, difficulty)
