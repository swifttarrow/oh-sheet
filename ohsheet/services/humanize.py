"""Humanization stage — STUB.

Real implementation: temp1/humanize.py — applies micro-timing deviations,
velocity shaping, and expressive markings. This stub is a pass-through that
emits one ExpressiveNote per ScoreNote with zero deviation, satisfying the
1:1 invariant from contracts §4.
"""
from __future__ import annotations

import asyncio

from ohsheet.contracts import (
    SCHEMA_VERSION,
    ExpressionMap,
    ExpressiveNote,
    HumanizedPerformance,
    PianoScore,
    QualitySignal,
)


class HumanizeService:
    name = "humanize"

    async def run(self, payload: PianoScore) -> HumanizedPerformance:
        await asyncio.sleep(0.05)

        expressive_notes: list[ExpressiveNote] = []
        for hand_name, notes in (("rh", payload.right_hand), ("lh", payload.left_hand)):
            for n in notes:
                expressive_notes.append(
                    ExpressiveNote(
                        score_note_id=n.id,
                        pitch=n.pitch,
                        onset_beat=n.onset_beat,
                        duration_beat=n.duration_beat,
                        velocity=n.velocity,
                        hand=hand_name,
                        voice=n.voice,
                        timing_offset_ms=0.0,
                        velocity_offset=0,
                    )
                )

        return HumanizedPerformance(
            schema_version=SCHEMA_VERSION,
            expressive_notes=expressive_notes,
            expression=ExpressionMap(),
            score=payload,
            quality=QualitySignal(
                overall_confidence=0.9,
                warnings=["stub humanizer — pass-through with zero deviation"],
            ),
        )
