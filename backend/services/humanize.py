"""Humanization stage — rule-based micro-timing & velocity shaping.

Mirrors temp1/humanize.py at a smaller scale: timing offsets keyed off
beat phase, velocity offsets shaped by section arcs, dynamic markings
inferred from per-section average velocity, sustain pedal events keyed
off chord changes (or per-bar fallback), and basic articulation tagging.

A future revision will swap the rule-based core for a trained model.
"""
from __future__ import annotations

import asyncio
import logging
import math
import random

from backend.contracts import (
    SCHEMA_VERSION,
    Articulation,
    DynamicMarking,
    ExpressionMap,
    ExpressiveNote,
    HumanizedPerformance,
    PedalEvent,
    PianoScore,
    QualitySignal,
    ScoreChordEvent,
    ScoreNote,
    ScoreSection,
)

log = logging.getLogger(__name__)

MAX_TIMING_OFFSET_MS = 30.0
MAX_VELOCITY_OFFSET = 15
MIN_PEDAL_GAP = 0.5
DYN_LEVELS = ["pp", "p", "mp", "mf", "f", "ff"]


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def _humanize_timing(notes: list[ScoreNote], hand: str, seed: int) -> dict[str, float]:
    rng = random.Random(seed if hand == "rh" else seed + 1)
    offsets: dict[str, float] = {}
    for n in notes:
        beat_phase = n.onset_beat % 1.0
        if beat_phase < 0.1:
            base = -5.0           # downbeat anticipation
        elif 0.4 < beat_phase < 0.6:
            base = 3.0            # backbeat push
        else:
            base = 0.0
        noise = rng.gauss(0, MAX_TIMING_OFFSET_MS * 0.3)
        offsets[n.id] = round(
            max(-MAX_TIMING_OFFSET_MS, min(MAX_TIMING_OFFSET_MS, base + noise)), 2,
        )
    return offsets


# ---------------------------------------------------------------------------
# Velocity
# ---------------------------------------------------------------------------

def _humanize_velocity(
    notes: list[ScoreNote],
    sections: list[ScoreSection],
    seed: int,
) -> dict[str, int]:
    rng = random.Random(seed)
    offsets: dict[str, int] = {}
    for n in notes:
        beat_phase = n.onset_beat % 1.0
        if beat_phase < 0.1:
            accent = 5
        elif abs(beat_phase - 0.5) < 0.1:
            accent = 2
        else:
            accent = -2
        phrase_shape = 0
        for sec in sections:
            if sec.start_beat <= n.onset_beat < sec.end_beat:
                progress = (n.onset_beat - sec.start_beat) / max(
                    sec.end_beat - sec.start_beat, 1.0,
                )
                phrase_shape = int(8 * math.sin(progress * math.pi))
                break
        noise = rng.randint(-3, 3)
        offsets[n.id] = max(
            -MAX_VELOCITY_OFFSET,
            min(MAX_VELOCITY_OFFSET, accent + phrase_shape + noise),
        )
    return offsets


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

def _infer_dynamics(
    notes: list[ScoreNote],
    velocity_offsets: dict[str, int],
    sections: list[ScoreSection],
) -> list[DynamicMarking]:
    markings: list[DynamicMarking] = []
    for sec in sections:
        sec_notes = [n for n in notes if sec.start_beat <= n.onset_beat < sec.end_beat]
        if not sec_notes:
            continue
        avg = sum(n.velocity + velocity_offsets.get(n.id, 0) for n in sec_notes) / len(sec_notes)
        idx = min(5, max(0, int(avg / 127 * 6)))
        markings.append(DynamicMarking(beat=sec.start_beat, type=DYN_LEVELS[idx]))  # type: ignore[arg-type]
    return markings


# ---------------------------------------------------------------------------
# Pedal
# ---------------------------------------------------------------------------

def _generate_pedal(
    chord_symbols: list[ScoreChordEvent],
    sections: list[ScoreSection],
    time_sig: tuple[int, int],
    fallback_end_beat: float,
) -> list[PedalEvent]:
    events: list[PedalEvent] = []
    if chord_symbols:
        for i, chord in enumerate(chord_symbols):
            onset = chord.beat
            if i + 1 < len(chord_symbols):
                offset = chord_symbols[i + 1].beat - 0.125
            else:
                offset = chord.beat + chord.duration_beat
            if offset - onset >= MIN_PEDAL_GAP:
                events.append(PedalEvent(
                    onset_beat=onset,
                    offset_beat=round(offset, 4),
                    type="sustain",
                ))
        return events

    end_beat = max((s.end_beat for s in sections), default=fallback_end_beat)
    if end_beat <= 0:
        return events
    bar_length = max(time_sig[0], 1)
    beat = 0.0
    while beat < end_beat:
        events.append(PedalEvent(
            onset_beat=beat,
            offset_beat=beat + bar_length - 0.125,
            type="sustain",
        ))
        beat += bar_length
    return events


# ---------------------------------------------------------------------------
# Articulations
# ---------------------------------------------------------------------------

def _detect_articulations(notes: list[ScoreNote], hand: str) -> list[Articulation]:
    arts: list[Articulation] = []
    sorted_notes = sorted(notes, key=lambda n: n.onset_beat)
    for i, n in enumerate(sorted_notes):
        if i + 1 < len(sorted_notes):
            gap = sorted_notes[i + 1].onset_beat - n.onset_beat
            if gap > 0 and n.duration_beat < gap * 0.4:
                arts.append(Articulation(
                    beat=n.onset_beat,
                    hand=hand,  # type: ignore[arg-type]
                    score_note_id=n.id,
                    type="staccato",
                ))
            elif gap > 0 and n.duration_beat >= gap * 0.95:
                arts.append(Articulation(
                    beat=n.onset_beat,
                    hand=hand,  # type: ignore[arg-type]
                    score_note_id=n.id,
                    type="legato",
                ))
        if (n.onset_beat % 1.0) > 0.2 and n.velocity > 100:
            arts.append(Articulation(
                beat=n.onset_beat,
                hand=hand,  # type: ignore[arg-type]
                score_note_id=n.id,
                type="accent",
            ))
    return arts


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

def _humanize_sync(score: PianoScore, seed: int) -> HumanizedPerformance:
    meta = score.metadata
    sections = meta.sections

    rh_timing = _humanize_timing(score.right_hand, "rh", seed)
    lh_timing = _humanize_timing(score.left_hand, "lh", seed)
    rh_velocity = _humanize_velocity(score.right_hand, sections, seed)
    lh_velocity = _humanize_velocity(score.left_hand, sections, seed + 100)

    expressive_notes: list[ExpressiveNote] = []
    for n in score.right_hand:
        vo = rh_velocity.get(n.id, 0)
        expressive_notes.append(ExpressiveNote(
            score_note_id=n.id,
            pitch=n.pitch,
            onset_beat=n.onset_beat,
            duration_beat=n.duration_beat,
            velocity=max(1, min(127, n.velocity + vo)),
            hand="rh",
            voice=n.voice,
            timing_offset_ms=rh_timing.get(n.id, 0.0),
            velocity_offset=vo,
        ))
    for n in score.left_hand:
        vo = lh_velocity.get(n.id, 0)
        expressive_notes.append(ExpressiveNote(
            score_note_id=n.id,
            pitch=n.pitch,
            onset_beat=n.onset_beat,
            duration_beat=n.duration_beat,
            velocity=max(1, min(127, n.velocity + vo)),
            hand="lh",
            voice=n.voice,
            timing_offset_ms=lh_timing.get(n.id, 0.0),
            velocity_offset=vo,
        ))

    all_notes = score.right_hand + score.left_hand
    all_vel_offsets = {**rh_velocity, **lh_velocity}
    dynamics = _infer_dynamics(all_notes, all_vel_offsets, sections)

    fallback_end = max(
        (n.onset_beat + n.duration_beat for n in all_notes),
        default=0.0,
    )
    pedal = _generate_pedal(
        meta.chord_symbols, sections, meta.time_signature, fallback_end,
    )

    articulations = (
        _detect_articulations(score.right_hand, "rh")
        + _detect_articulations(score.left_hand, "lh")
    )

    expression = ExpressionMap(
        dynamics=dynamics,
        articulations=articulations,
        pedal_events=pedal,
        tempo_changes=[],
    )

    log.info(
        "Humanized: %d notes, %d dynamics, %d pedal events, %d articulations",
        len(expressive_notes), len(dynamics), len(pedal), len(articulations),
    )

    return HumanizedPerformance(
        schema_version=SCHEMA_VERSION,
        expressive_notes=expressive_notes,
        expression=expression,
        score=score,
        quality=QualitySignal(
            overall_confidence=0.7,
            warnings=["Rule-based humanization (no trained model yet)"],
        ),
    )


class HumanizeService:
    name = "humanize"

    def __init__(self, *, seed: int = 42) -> None:
        self.seed = seed

    async def run(self, payload: PianoScore) -> HumanizedPerformance:
        log.info(
            "humanize: start rh=%d lh=%d",
            len(payload.right_hand),
            len(payload.left_hand),
        )
        perf = await asyncio.to_thread(_humanize_sync, payload, self.seed)
        log.info("humanize: done expressive_notes=%d", len(perf.expressive_notes))
        return perf
