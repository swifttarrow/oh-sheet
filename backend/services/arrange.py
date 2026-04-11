"""Arrangement stage — basic two-hand piano reduction.

Takes a seconds-domain TranscriptionResult and emits a beat-domain
PianoScore. The pipeline mirrors temp1/arrange.py at a smaller scale:

  1. Hand assignment — melody → RH, bass → LH, anything else by middle C
  2. Quantize onsets/durations to a 1/16th-note grid
  3. Voice assignment within each hand
  4. Velocity normalization across both hands
  5. Build PianoScore with chords / sections converted to beat-domain
"""
from __future__ import annotations

import asyncio
import logging

from backend.config import settings
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

QUANT_GRID = 0.25            # 1/16th note (default / fallback)
SPLIT_PITCH = 60             # middle C — pitches >= split go right hand
# Piano notation is defined by at most two voices per staff (stems up =
# melody, stems down = accompaniment). Capping here means engrave can
# trust the voice assignment and emit ``<voice>1</voice>`` / ``<voice>2</voice>``
# directly instead of collapsing everything back down. Going wider
# (4/3 previously) produced voice-3 notes that OSMD's VexFlow backend
# crashes on and that no pianist can actually read as four parallel
# lines on one staff.
MAX_VOICES_RH = 2
MAX_VOICES_LH = 2
MIN_TRACK_CONFIDENCE = 0.35
NEAR_OVERLAP_TOL = 0.15      # beats

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


def _parse_grid_candidates(raw: str) -> list[float]:
    """Parse a comma-separated string of floats into a list."""
    return [float(tok.strip()) for tok in raw.split(",") if tok.strip()]


def _estimate_best_grid(
    beat_onsets: list[float],
    *,
    candidates: list[float] | None = None,
    min_notes: int | None = None,
) -> float:
    """Pick the candidate grid that minimises mean quantization residual.

    Falls back to QUANT_GRID when there are too few notes.
    """
    if min_notes is None:
        min_notes = settings.arrange_min_notes_for_grid_estimation

    if len(beat_onsets) < min_notes:
        return QUANT_GRID

    if candidates is None:
        candidates = _parse_grid_candidates(settings.arrange_grid_candidates)

    _EPS = 1e-9
    best_grid = QUANT_GRID
    best_residual = float("inf")
    for grid in candidates:
        residual = sum(abs(o - round(o / grid) * grid) for o in beat_onsets) / len(beat_onsets)
        # On tied residual prefer the coarser (larger) grid for simpler notation.
        if (residual + _EPS < best_residual) or (
            abs(residual - best_residual) <= _EPS and grid > best_grid
        ):
            best_residual = residual
            best_grid = grid
    return best_grid


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
    *,
    grid: float = QUANT_GRID,
    overlap_tol: float = NEAR_OVERLAP_TOL,
) -> list[tuple[int, float, float, int, int]]:
    """Quantize, drop near-duplicates, and greedily assign voice numbers."""
    quantized = [
        (pitch, _quantize(onset, grid), _quantize_duration(dur, grid), vel)
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
            and abs(quantized[j][1] - best[1]) <= overlap_tol
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
                if gap < grid * 0.5:
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
# Beat-synchronous note snapping (post-quantization correction)
# ---------------------------------------------------------------------------

def _get_beat_positions(max_beat: float, subdivision: float = 0.5) -> list[float]:
    """Generate beat positions at the given subdivision granularity.

    For subdivision=0.5: beats at 0, 0.5, 1.0, 1.5, 2.0, ...
    For subdivision=1.0: beats at 0, 1, 2, 3, ...
    """
    positions: list[float] = []
    b = 0.0
    while b <= max_beat:
        positions.append(b)
        b += subdivision
    return positions


def _beat_alignment(onset: float, beat_positions: list[float]) -> float:
    """How close is this onset to the nearest beat position?

    Returns a score in (0, 1] — higher means better alignment.
    """
    if not beat_positions:
        return 0.0
    min_dist = min(abs(onset - b) for b in beat_positions)
    return 1.0 / (1.0 + min_dist)


def _beat_snap(
    notes: list[tuple[int, float, float, int, int]],
    tempo_map: list[TempoMapEntry],
    grid: float,
    snap_weight: float = 0.3,
    subdivision: float = 0.5,
) -> list[tuple[int, float, float, int, int]]:
    """Shift note onsets by ±1 grid step when it better aligns with beats.

    Pure beat-space pass — no I/O, no audio loading. Runs after quantization
    and voice assignment, respecting voice collision constraints.

    Parameters
    ----------
    notes:
        (pitch, onset_beat, duration_beat, velocity, voice) tuples,
        sorted by (onset_beat, -pitch) from ``_resolve_overlaps``.
    tempo_map:
        The piece's tempo map (used only for max-beat calculation).
    grid:
        Quantization grid in beats (e.g. 0.25 for 1/16th note).
    snap_weight:
        Minimum improvement in alignment score required to trigger a shift.
    subdivision:
        Beat subdivision granularity for generating beat positions.

    Returns
    -------
    New list of note tuples with adjusted onsets/durations.
    """
    if not notes:
        return notes

    max_beat = max(n[1] + n[2] for n in notes)
    beat_positions = _get_beat_positions(max_beat + grid, subdivision)

    # Work on a mutable copy; group by voice for collision checks
    result: list[tuple[int, float, float, int, int]] = list(notes)

    # Pre-compute per-voice ordered indices
    by_voice: dict[int, list[int]] = {}
    for idx, (_, onset, dur, _, voice) in enumerate(result):
        by_voice.setdefault(voice, []).append(idx)

    shifted_count = 0

    for voice, indices in by_voice.items():
        # indices are in insertion order from _resolve_overlaps (onset-sorted)
        for pos, idx in enumerate(indices):
            pitch, onset, dur, vel, v = result[idx]
            current_score = _beat_alignment(onset, beat_positions)

            best_onset = onset
            best_score = current_score

            for candidate_onset in (onset - grid, onset + grid):
                if candidate_onset < 0:
                    continue
                cand_score = _beat_alignment(candidate_onset, beat_positions)
                if cand_score > best_score:
                    best_onset = candidate_onset
                    best_score = cand_score

            # Only shift if improvement exceeds threshold
            if best_onset == onset or (best_score - current_score) <= snap_weight:
                continue

            # Check voice collision: would the new onset overlap the previous
            # note's end in the same voice?
            if pos > 0:
                prev_idx = indices[pos - 1]
                _, prev_onset, prev_dur, _, _ = result[prev_idx]
                if best_onset < prev_onset + prev_dur:
                    continue  # would collide with previous note

            # Check voice collision: would the new onset + duration overlap
            # the next note's onset in the same voice?
            # (We adjust duration below, but the note's end stays fixed,
            # so only onset-before-previous-end matters. However, if we
            # shift forward, the note's onset could equal or exceed the
            # next note's onset — block that.)
            if pos + 1 < len(indices):
                next_idx = indices[pos + 1]
                _, next_onset, _, _, _ = result[next_idx]
                if best_onset >= next_onset:
                    continue  # would collide with next note

            # Adjust duration to maintain the note's end position
            new_dur = dur + (onset - best_onset)
            new_dur = max(new_dur, grid)  # clamp to minimum grid step

            result[idx] = (pitch, best_onset, new_dur, vel, v)
            shifted_count += 1

    if shifted_count:
        log.info("beat_snap: shifted %d notes", shifted_count)

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
            return max(1, min(127, round(v + shift)))
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

    rh_raw, lh_raw = _assign_hands(pitched_tracks, tempo_map)

    # Adaptive grid estimation — pick the best quantization grid for this piece
    if settings.arrange_adaptive_grid_enabled:
        all_onsets = [n[1] for n in rh_raw] + [n[1] for n in lh_raw]
        grid = _estimate_best_grid(all_onsets)
        log.info("arrange: estimated grid=%.3f beats", grid)
    else:
        grid = QUANT_GRID
    overlap_tol = 0.6 * grid

    max_rh = MAX_VOICES_RH if difficulty != "beginner" else 1
    max_lh = MAX_VOICES_LH if difficulty != "beginner" else 1
    rh_voiced = _resolve_overlaps(rh_raw, max_rh, grid=grid, overlap_tol=overlap_tol)
    lh_voiced = _resolve_overlaps(lh_raw, max_lh, grid=grid, overlap_tol=overlap_tol)

    # Post-quantization beat-synchronous snapping
    if settings.arrange_beat_snap_enabled:
        rh_voiced = _beat_snap(
            rh_voiced,
            tempo_map,
            grid=grid,
            snap_weight=settings.arrange_beat_snap_weight,
            subdivision=settings.arrange_beat_snap_subdivision,
        )
        lh_voiced = _beat_snap(
            lh_voiced,
            tempo_map,
            grid=grid,
            snap_weight=settings.arrange_beat_snap_weight,
            subdivision=settings.arrange_beat_snap_subdivision,
        )

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
        log.info(
            "arrange: start tracks_in=%d difficulty=%s",
            len(payload.midi_tracks),
            difficulty,
        )
        score = await asyncio.to_thread(_arrange_sync, payload, difficulty)
        log.info(
            "arrange: done rh_notes=%d lh_notes=%d",
            len(score.right_hand),
            len(score.left_hand),
        )
        return score
