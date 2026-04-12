"""Assemble stage -- strict rule-based piano arrangement.

Takes a TranscriptionResult (expected from the decomposer: melody + other tracks)
and produces a PianoScore using rigid difficulty-specific rules.

Currently only "beginner" difficulty is implemented:
  - 8th-note quantization (0.5-beat grid)
  - Melody -> RH, max 1 note per beat (highest pitch wins)
  - Lowest accompaniment note per beat -> LH, max 1 note per beat
  - Accompaniment notes above middle C discarded
  - RH range: C4-C6 (MIDI 60-84), LH range: C2-B3 (MIDI 36-59)
  - Notes outside range octave-shifted inward
"""
from __future__ import annotations

import logging

from shared.contracts import (
    SCHEMA_VERSION,
    InstrumentRole,
    Note,
    PianoScore,
    ScoreChordEvent,
    ScoreMetadata,
    ScoreNote,
    ScoreSection,
    TempoMapEntry,
    TranscriptionResult,
    sec_to_beat,
)

log = logging.getLogger(__name__)

# Beginner constants
_QUANT_GRID = 0.5  # 8th note
_RH_MIN = 60       # C4
_RH_MAX = 84       # C6
_LH_MIN = 36       # C2
_LH_MAX = 59       # B3
_SPLIT_PITCH = 60  # Middle C


def _quantize(value: float, grid: float) -> float:
    """Snap a value to the nearest grid point."""
    return round(value / grid) * grid


def _clamp_pitch(pitch: int, lo: int, hi: int) -> int:
    """Octave-shift a pitch until it falls within [lo, hi]."""
    while pitch > hi:
        pitch -= 12
    while pitch < lo:
        pitch += 12
    return pitch


def _sec_to_beat_notes(
    notes: list[Note],
    tempo_map: list[TempoMapEntry],
) -> list[tuple[float, float, Note]]:
    """Convert notes from seconds to beats, return (onset_beat, duration_beat, original)."""
    result = []
    for n in notes:
        onset = sec_to_beat(n.onset_sec, tempo_map)
        offset = sec_to_beat(n.offset_sec, tempo_map)
        duration = max(offset - onset, 0.01)
        result.append((onset, duration, n))
    return result


class AssembleService:
    def run(self, txr: TranscriptionResult, *, difficulty: str = "beginner") -> PianoScore:
        if difficulty != "beginner":
            raise NotImplementedError(
                f"difficulty={difficulty!r} is not implemented; only 'beginner' is supported"
            )

        tempo_map = txr.analysis.tempo_map
        melody_notes: list[Note] = []
        accomp_notes: list[Note] = []

        for track in txr.midi_tracks:
            if track.instrument == InstrumentRole.MELODY:
                melody_notes.extend(track.notes)
            else:
                accomp_notes.extend(track.notes)

        rh = self._build_right_hand(melody_notes, tempo_map)
        lh = self._build_left_hand(accomp_notes, tempo_map)

        # Convert chords and sections to beat domain
        chord_symbols = [
            ScoreChordEvent(
                beat=sec_to_beat(c.time_sec, tempo_map),
                duration_beat=max(
                    sec_to_beat(c.time_sec + c.duration_sec, tempo_map)
                    - sec_to_beat(c.time_sec, tempo_map),
                    0.01,
                ),
                label=c.label,
                root=c.root,
                confidence=c.confidence,
            )
            for c in txr.analysis.chords
        ]
        sections = [
            ScoreSection(
                start_beat=sec_to_beat(s.start_sec, tempo_map),
                end_beat=sec_to_beat(s.end_sec, tempo_map),
                label=s.label,
            )
            for s in txr.analysis.sections
        ]

        log.info(
            "assemble beginner: %d melody -> %d RH, %d accomp -> %d LH",
            len(melody_notes),
            len(rh),
            len(accomp_notes),
            len(lh),
        )

        return PianoScore(
            schema_version=SCHEMA_VERSION,
            right_hand=rh,
            left_hand=lh,
            metadata=ScoreMetadata(
                key=txr.analysis.key,
                time_signature=txr.analysis.time_signature,
                tempo_map=tempo_map,
                difficulty="beginner",
                sections=sections,
                chord_symbols=chord_symbols,
            ),
        )

    def _build_right_hand(
        self,
        melody_notes: list[Note],
        tempo_map: list[TempoMapEntry],
    ) -> list[ScoreNote]:
        """Melody -> RH: quantize, enforce monophony (highest pitch wins), clamp range."""
        beat_notes = _sec_to_beat_notes(melody_notes, tempo_map)

        # Quantize onsets and durations to 8th-note grid
        quantized: list[tuple[float, float, Note]] = []
        for onset, dur, note in beat_notes:
            q_onset = _quantize(onset, _QUANT_GRID)
            q_dur = max(_quantize(dur, _QUANT_GRID), _QUANT_GRID)
            quantized.append((q_onset, q_dur, note))

        # Enforce monophony: group by quantized onset, keep highest pitch
        groups: dict[float, list[tuple[float, float, Note]]] = {}
        for onset, dur, note in quantized:
            groups.setdefault(onset, []).append((onset, dur, note))

        rh: list[ScoreNote] = []
        for idx, onset in enumerate(sorted(groups)):
            group = groups[onset]
            group.sort(key=lambda t: -t[2].pitch)
            _, dur, note = group[0]
            pitch = _clamp_pitch(note.pitch, _RH_MIN, _RH_MAX)
            rh.append(ScoreNote(
                id=f"rh-{idx:04d}",
                pitch=pitch,
                onset_beat=onset,
                duration_beat=dur,
                velocity=min(max(note.velocity, 1), 127),
                voice=1,
            ))

        return rh

    def _build_left_hand(
        self,
        accomp_notes: list[Note],
        tempo_map: list[TempoMapEntry],
    ) -> list[ScoreNote]:
        """Accompaniment -> LH: discard above middle C, quantize, monophony (lowest wins), clamp range."""
        below_split = [n for n in accomp_notes if n.pitch < _SPLIT_PITCH]

        beat_notes = _sec_to_beat_notes(below_split, tempo_map)

        quantized: list[tuple[float, float, Note]] = []
        for onset, dur, note in beat_notes:
            q_onset = _quantize(onset, _QUANT_GRID)
            q_dur = max(_quantize(dur, _QUANT_GRID), _QUANT_GRID)
            quantized.append((q_onset, q_dur, note))

        groups: dict[float, list[tuple[float, float, Note]]] = {}
        for onset, dur, note in quantized:
            groups.setdefault(onset, []).append((onset, dur, note))

        lh: list[ScoreNote] = []
        for idx, onset in enumerate(sorted(groups)):
            group = groups[onset]
            group.sort(key=lambda t: t[2].pitch)
            _, dur, note = group[0]
            pitch = _clamp_pitch(note.pitch, _LH_MIN, _LH_MAX)
            lh.append(ScoreNote(
                id=f"lh-{idx:04d}",
                pitch=pitch,
                onset_beat=onset,
                duration_beat=dur,
                velocity=min(max(note.velocity, 1), 127),
                voice=1,
            ))

        return lh
