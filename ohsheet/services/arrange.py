"""Arrangement stage — STUB.

Real implementation: temp1/arrange.py. Converts a seconds-domain
TranscriptionResult into a quantized two-handed PianoScore in beat-domain.
This stub does a trivial 1:1 mapping with naive RH/LH split by pitch.
"""
from __future__ import annotations

import asyncio

from ohsheet.contracts import (
    SCHEMA_VERSION,
    Difficulty,
    PianoScore,
    ScoreMetadata,
    ScoreNote,
    TranscriptionResult,
)


class ArrangeService:
    name = "arrange"

    async def run(
        self,
        payload: TranscriptionResult,
        *,
        difficulty: Difficulty = "intermediate",
    ) -> PianoScore:
        await asyncio.sleep(0.1)

        # Use the first tempo map entry to convert seconds → beats. The real
        # arranger walks the full tempo map per the contract guarantee.
        bpm = payload.analysis.tempo_map[0].bpm if payload.analysis.tempo_map else 120.0
        beats_per_sec = bpm / 60.0

        rh: list[ScoreNote] = []
        lh: list[ScoreNote] = []
        for track_idx, track in enumerate(payload.midi_tracks):
            for note_idx, n in enumerate(track.notes):
                onset_beat = round(n.onset_sec * beats_per_sec * 4) / 4
                duration_beat = max(0.25, round((n.offset_sec - n.onset_sec) * beats_per_sec * 4) / 4)
                node = ScoreNote(
                    id=f"t{track_idx}-n{note_idx}",
                    pitch=n.pitch,
                    onset_beat=onset_beat,
                    duration_beat=duration_beat,
                    velocity=n.velocity,
                    voice=1,
                )
                (rh if n.pitch >= 60 else lh).append(node)

        return PianoScore(
            schema_version=SCHEMA_VERSION,
            right_hand=rh,
            left_hand=lh,
            metadata=ScoreMetadata(
                key=payload.analysis.key,
                time_signature=payload.analysis.time_signature,
                tempo_map=payload.analysis.tempo_map,
                difficulty=difficulty,
                sections=[],
                chord_symbols=[],
            ),
        )
