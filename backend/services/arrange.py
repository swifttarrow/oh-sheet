"""Arrangement stage — basic two-hand piano reduction.

Takes a seconds-domain TranscriptionResult and emits a beat-domain
PianoScore. The pipeline mirrors temp1/arrange.py at a smaller scale:

  1. Cross-track dedup of same-pitch notes (MT3 emits duplicates per program)
  2. Hand assignment — melody → RH, bass → LH, anything else by middle C
  3. Quantize onsets/durations to a 1/16th-note grid
  4. Voice assignment within each hand
  5. Velocity normalization across both hands
  6. Build PianoScore with chords / sections converted to beat-domain
"""
from __future__ import annotations

import asyncio
import logging

from backend.contracts import (
    SCHEMA_VERSION,
    Difficulty,
    InstrumentRole,
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

QUANT_GRID = 0.25            # 1/16th note
SPLIT_PITCH = 60             # middle C — pitches >= split go right hand
MAX_VOICES_RH = 4
MAX_VOICES_LH = 3
MIN_TRACK_CONFIDENCE = 0.35
NEAR_OVERLAP_TOL = 0.15      # beats
DEDUP_TIME_TOL = 0.20        # seconds

_VEL_TARGET_MIN = 35
_VEL_TARGET_MAX = 120
_VEL_TARGET_MEAN = 75


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

def _quantize(onset: float, grid: float = QUANT_GRID) -> float:
    return round(onset / grid) * grid


def _quantize_duration(dur: float, grid: float = QUANT_GRID) -> float:
    return max(round(dur / grid) * grid, grid)


# ---------------------------------------------------------------------------
# Cross-track deduplication
# ---------------------------------------------------------------------------

def _dedup_across_tracks(tracks: list[MidiTrack]) -> list[MidiTrack]:
    """Merge same-pitch notes across tracks within DEDUP_TIME_TOL seconds.

    MT3 frequently emits the same audio event under several GM programs;
    keeping the loudest copy avoids them landing on different grid spots.
    """
    indexed: list[tuple[int, float, int, int, int]] = []  # (pitch, onset, vel, ti, ni)
    for ti, track in enumerate(tracks):
        for ni, n in enumerate(track.notes):
            indexed.append((n.pitch, n.onset_sec, n.velocity, ti, ni))
    if not indexed:
        return tracks

    indexed.sort(key=lambda x: (x[0], x[1]))

    keep: set[tuple[int, int]] = set()
    used = [False] * len(indexed)
    for i in range(len(indexed)):
        if used[i]:
            continue
        best = i
        j = i + 1
        while (
            j < len(indexed)
            and indexed[j][0] == indexed[i][0]
            and indexed[j][1] - indexed[i][1] <= DEDUP_TIME_TOL
        ):
            if indexed[j][2] > indexed[best][2]:
                best = j
            j += 1
        for k in range(i, j):
            used[k] = True
        keep.add((indexed[best][3], indexed[best][4]))

    result: list[MidiTrack] = []
    for ti, track in enumerate(tracks):
        filtered = [n for ni, n in enumerate(track.notes) if (ti, ni) in keep]
        if filtered:
            result.append(MidiTrack(
                notes=filtered,
                instrument=track.instrument,
                program=track.program,
                confidence=track.confidence,
            ))
    return result


# ---------------------------------------------------------------------------
# Hand assignment
# ---------------------------------------------------------------------------

def _notes_to_beats(
    notes: list[Note],
    tempo_map: list[TempoMapEntry],
) -> list[tuple[int, float, float, int]]:
    out: list[tuple[int, float, float, int]] = []
    for n in notes:
        onset = sec_to_beat(n.onset_sec, tempo_map)
        offset = sec_to_beat(n.offset_sec, tempo_map)
        out.append((n.pitch, onset, max(offset - onset, QUANT_GRID), n.velocity))
    return out


def _assign_hands(
    tracks: list[MidiTrack],
    tempo_map: list[TempoMapEntry],
) -> tuple[list[tuple[int, float, float, int]], list[tuple[int, float, float, int]]]:
    rh: list[tuple[int, float, float, int]] = []
    lh: list[tuple[int, float, float, int]] = []
    for track in tracks:
        if track.confidence < MIN_TRACK_CONFIDENCE:
            log.info(
                "Skipping low-confidence track (program=%s, conf=%.2f)",
                track.program, track.confidence,
            )
            continue
        beat_notes = _notes_to_beats(track.notes, tempo_map)
        if track.instrument == InstrumentRole.MELODY:
            rh.extend(beat_notes)
        elif track.instrument == InstrumentRole.BASS:
            lh.extend(beat_notes)
        else:
            for n in beat_notes:
                (rh if n[0] >= SPLIT_PITCH else lh).append(n)
    return rh, lh


# ---------------------------------------------------------------------------
# Overlap resolution + voice assignment
# ---------------------------------------------------------------------------

def _resolve_overlaps(
    notes: list[tuple[int, float, float, int]],
    max_voices: int,
) -> list[tuple[int, float, float, int, int]]:
    """Quantize, drop near-duplicates, and greedily assign voice numbers."""
    quantized = [
        (pitch, _quantize(onset), _quantize_duration(dur), vel)
        for pitch, onset, dur, vel in notes
    ]

    # Same-pitch dedup within tolerance — keep the loudest
    quantized.sort(key=lambda n: (n[0], n[1]))
    deduped: list[tuple[int, float, float, int]] = []
    i = 0
    while i < len(quantized):
        best = quantized[i]
        j = i + 1
        while (
            j < len(quantized)
            and quantized[j][0] == best[0]
            and abs(quantized[j][1] - best[1]) <= NEAR_OVERLAP_TOL
        ):
            if quantized[j][3] > best[3]:
                best = quantized[j]
            j += 1
        deduped.append(best)
        i = j

    # Trim same-pitch overlaps so a new note's onset chops the previous one.
    deduped.sort(key=lambda n: (n[1], -n[0]))
    last_by_pitch: dict[int, int] = {}
    drop: set[int] = set()
    for idx, (pitch, onset, dur, vel) in enumerate(deduped):
        if pitch in last_by_pitch:
            prev_idx = last_by_pitch[pitch]
            pp, po, pd, pv = deduped[prev_idx]
            gap = onset - po
            if po + pd > onset:
                if gap < QUANT_GRID * 0.5:
                    drop.add(idx if pv >= vel else prev_idx)
                else:
                    deduped[prev_idx] = (pp, po, gap, pv)
        last_by_pitch[pitch] = idx
    if drop:
        deduped = [n for k, n in enumerate(deduped) if k not in drop]

    # Greedy voice assignment
    voice_ends: list[float] = []
    result: list[tuple[int, float, float, int, int]] = []
    for pitch, onset, dur, vel in deduped:
        assigned: int | None = None
        for vi, end in enumerate(voice_ends):
            if onset >= end:
                assigned = vi
                voice_ends[vi] = onset + dur
                break
        if assigned is None:
            if len(voice_ends) < max_voices:
                assigned = len(voice_ends)
                voice_ends.append(onset + dur)
            else:
                continue  # exceeds polyphony — drop
        result.append((pitch, onset, dur, vel, assigned + 1))
    return result


# ---------------------------------------------------------------------------
# Velocity normalization
# ---------------------------------------------------------------------------

def _normalize_velocity(
    rh: list[tuple[int, float, float, int, int]],
    lh: list[tuple[int, float, float, int, int]],
) -> tuple[
    list[tuple[int, float, float, int, int]],
    list[tuple[int, float, float, int, int]],
]:
    all_notes = rh + lh
    if not all_notes:
        return rh, lh
    vels = [n[3] for n in all_notes]
    v_min, v_max = min(vels), max(vels)
    v_mean = sum(vels) / len(vels)
    if v_max - v_min < 10:
        shift = _VEL_TARGET_MEAN - v_mean
        def remap(v: int) -> int:
            return max(1, min(127, int(v + shift)))
    else:
        scale = (_VEL_TARGET_MAX - _VEL_TARGET_MIN) / (v_max - v_min)
        offset = _VEL_TARGET_MIN - v_min * scale
        def remap(v: int) -> int:
            return max(1, min(127, int(v * scale + offset)))
    apply = lambda notes: [
        (p, o, d, remap(v), voice) for p, o, d, v, voice in notes
    ]
    return apply(rh), apply(lh)


# ---------------------------------------------------------------------------
# Domain conversions for chords / sections
# ---------------------------------------------------------------------------

def _chord_to_score_chord(
    chord: RealtimeChordEvent,
    tempo_map: list[TempoMapEntry],
) -> ScoreChordEvent:
    onset = sec_to_beat(chord.time_sec, tempo_map)
    end = sec_to_beat(chord.time_sec + chord.duration_sec, tempo_map)
    return ScoreChordEvent(
        beat=onset,
        duration_beat=max(end - onset, QUANT_GRID),
        label=chord.label,
        root=chord.root,
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


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

def _arrange_sync(
    payload: TranscriptionResult,
    difficulty: Difficulty,
) -> PianoScore:
    analysis = payload.analysis
    tempo_map = analysis.tempo_map or [TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)]

    pitched_tracks = [t for t in payload.midi_tracks if t.instrument != InstrumentRole.OTHER]
    if not pitched_tracks:
        # Fall back to all tracks rather than emit an empty score
        pitched_tracks = list(payload.midi_tracks)

    deduped = _dedup_across_tracks(pitched_tracks)
    rh_raw, lh_raw = _assign_hands(deduped, tempo_map)

    max_rh = MAX_VOICES_RH if difficulty != "beginner" else 2
    max_lh = MAX_VOICES_LH if difficulty != "beginner" else 1
    rh_voiced = _resolve_overlaps(rh_raw, max_rh)
    lh_voiced = _resolve_overlaps(lh_raw, max_lh)
    rh_voiced, lh_voiced = _normalize_velocity(rh_voiced, lh_voiced)

    right_hand = [
        ScoreNote(
            id=f"rh-{i:04d}",
            pitch=pitch,
            onset_beat=onset,
            duration_beat=dur,
            velocity=vel,
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
            velocity=vel,
            voice=voice,
        )
        for i, (pitch, onset, dur, vel, voice) in enumerate(lh_voiced)
    ]

    chord_symbols = [_chord_to_score_chord(c, tempo_map) for c in analysis.chords]
    sections = [_section_to_score_section(s, tempo_map) for s in analysis.sections]

    log.info(
        "Arranged: RH=%d notes, LH=%d notes (difficulty=%s)",
        len(right_hand), len(left_hand), difficulty,
    )

    return PianoScore(
        schema_version=SCHEMA_VERSION,
        right_hand=right_hand,
        left_hand=left_hand,
        metadata=ScoreMetadata(
            key=analysis.key,
            time_signature=analysis.time_signature,
            tempo_map=tempo_map,
            difficulty=difficulty,
            sections=sections,
            chord_symbols=chord_symbols,
        ),
    )


class ArrangeService:
    name = "arrange"

    async def run(
        self,
        payload: TranscriptionResult,
        *,
        difficulty: Difficulty = "intermediate",
    ) -> PianoScore:
        return await asyncio.to_thread(_arrange_sync, payload, difficulty)
