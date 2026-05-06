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
    PedalEvent,
    PianoScore,
    RealtimeChordEvent,
    RealtimePedalEvent,
    ScoreChordEvent,
    ScoreMetadata,
    ScoreNote,
    ScoreSection,
    Section,
    TempoMapEntry,
    TranscriptionResult,
    sec_to_beat,
)
from backend.services.hf_arrange.inference import run_hf_midi_inference
from backend.services.hf_arrange.midi_bridge import transcription_from_midi_bytes
from backend.services.transcription_midi_materialize import materialize_transcription_midi_bytes
from backend.storage.base import BlobStore

log = logging.getLogger(__name__)

QUANT_GRID = 0.25            # 1/16th note (default / fallback)
SPLIT_PITCH = 60             # middle C — pitches >= split go right hand
# Default voice caps. Standard piano notation supports up to four voices
# per staff (two stems-up, two stems-down). Settings overrides come from
# ``settings.arrange_max_voices_{rh,lh}`` so operators can drop back to
# 2 if the engraver chokes on voice 3/4.
MAX_VOICES_RH = 4
MAX_VOICES_LH = 4
# Floor applied to track confidence after the previous hard-drop gate
# was relaxed: keep low-confidence tracks but mark them as borderline so
# downstream consumers can de-emphasize them without losing the notes.
MIN_TRACK_CONFIDENCE = 0.05
NEAR_OVERLAP_TOL = 0.15      # beats

_VEL_TARGET_MIN = 35
_VEL_TARGET_MAX = 120
_VEL_TARGET_MEAN = 75
# Percentile clamp endpoints for the velocity remap. Using percentiles
# instead of min/max stops a single outlier from compressing the rest
# of the dynamic range into a tiny window.
_VEL_PERCENTILE_LOW = 5.0
_VEL_PERCENTILE_HIGH = 95.0


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
        # Clamp to non-negative — notes before the first detected beat
        # (e.g. Pop2Piano emitting from 0.0s when the beat tracker anchors
        # beat 0 later in the audio) would otherwise produce negative beat
        # positions that music21 cannot place in any measure.
        if onset < 0.0:
            onset = 0.0
        if offset < 0.0:
            continue  # entirely before the first beat — drop it
        out.append((n.pitch, onset, max(offset - onset, QUANT_GRID), n.velocity))
    return out


def _assign_hands(
    tracks: list[MidiTrack],
    tempo_map: list[TempoMapEntry],
) -> tuple[list[tuple[int, float, float, int]], list[tuple[int, float, float, int]]]:
    """Route notes to the right or left hand.

    Tracks tagged ``MELODY`` always go RH; tracks tagged ``BASS`` always go LH;
    everything else falls into the ``OTHER`` bucket which gets a hand
    assignment via either the legacy middle-C pitch split (default) or
    the Phase 9B Cluster-and-Separate GNN-style assigner when
    ``settings.voice_gnn_enabled`` is True.
    """
    rh: list[tuple[int, float, float, int]] = []
    lh: list[tuple[int, float, float, int]] = []
    other_notes: list[tuple[int, float, float, int]] = []
    for track in tracks:
        if track.confidence < MIN_TRACK_CONFIDENCE:
            log.warning(
                "Low-confidence track included (program=%s, conf=%.2f, floor=%.2f)",
                track.program, track.confidence, MIN_TRACK_CONFIDENCE,
            )
        beat_notes = _notes_to_beats(track.notes, tempo_map)
        if track.instrument == InstrumentRole.MELODY:
            rh.extend(beat_notes)
        elif track.instrument == InstrumentRole.BASS:
            lh.extend(beat_notes)
        else:
            other_notes.extend(beat_notes)

    if other_notes:
        if settings.voice_gnn_enabled:
            from backend.services.voice_gnn import (  # noqa: PLC0415
                VoiceGNNConfig,
                assign_hands_gnn,
            )
            gnn_cfg = VoiceGNNConfig(
                pitch_weight=settings.voice_gnn_pitch_weight,
                time_weight=settings.voice_gnn_time_weight,
                velocity_weight=settings.voice_gnn_velocity_weight,
                join_threshold=settings.voice_gnn_join_threshold,
                min_split_hint=settings.voice_gnn_min_split_hint,
                max_split_hint=settings.voice_gnn_max_split_hint,
            )
            try:
                gnn_rh, gnn_lh, gnn_stats = assign_hands_gnn(
                    other_notes, config=gnn_cfg,
                )
            except Exception as exc:  # noqa: BLE001 — never sink the job
                log.warning(
                    "voice_gnn: assignment failed (%s); falling back to %d-pitch split",
                    exc, SPLIT_PITCH,
                )
                for n in other_notes:
                    (rh if n[0] >= SPLIT_PITCH else lh).append(n)
            else:
                log.info(
                    "voice_gnn: streams=%d split_pitch=%d rh=%d lh=%d",
                    gnn_stats.n_streams, gnn_stats.split_pitch,
                    gnn_stats.n_rh, gnn_stats.n_lh,
                )
                rh.extend(gnn_rh)
                lh.extend(gnn_lh)
        else:
            for n in other_notes:
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

def _percentile(values: list[int], pct: float) -> float:
    """Linear-interpolation percentile without numpy."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    rank = (pct / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _normalize_velocity(
    rh: list[tuple[int, float, float, int, int]],
    lh: list[tuple[int, float, float, int, int]],
) -> tuple[
    list[tuple[int, float, float, int, int]],
    list[tuple[int, float, float, int, int]],
]:
    """Remap raw note velocities into a musical [VEL_TARGET_MIN, MAX] band.

    Uses 5th/95th percentile clamps instead of true min/max so a single
    outlier (a stray clipped onset, or one extra-loud accent) cannot
    compress the rest of the dynamic range into a sliver. Notes outside
    the percentile window are clipped to the band endpoints. Falls back
    to a mean-shift when the percentile span is too narrow to be
    meaningful (sub-10 units of velocity).
    """
    all_notes = rh + lh
    if not all_notes:
        return rh, lh
    vels = [n[3] for n in all_notes]
    v_lo = _percentile(vels, _VEL_PERCENTILE_LOW)
    v_hi = _percentile(vels, _VEL_PERCENTILE_HIGH)
    v_mean = sum(vels) / len(vels)
    if v_hi - v_lo < 10:
        shift = _VEL_TARGET_MEAN - v_mean
        def remap(v: int) -> int:
            return max(1, min(127, round(v + shift)))
    else:
        scale = (_VEL_TARGET_MAX - _VEL_TARGET_MIN) / (v_hi - v_lo)
        offset = _VEL_TARGET_MIN - v_lo * scale
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


# ``MIDI Control Change`` numbers for the three pedals music21 / MusicXML
# can render. Anything else from the transcriber gets dropped at the
# arrange boundary — the engraver has no representation for, e.g., CC1
# modulation as a pedal mark.
_CC_TO_PEDAL_TYPE = {
    64: "sustain",
    66: "sostenuto",
    67: "una_corda",
}


def _pedal_to_score_pedal(
    pedal: RealtimePedalEvent,
    tempo_map: list[TempoMapEntry],
) -> PedalEvent | None:
    """Convert a seconds-domain Kong pedal into a beat-domain PedalEvent.

    Returns ``None`` when the CC number doesn't map to a pedal type the
    engraver can render. Onset/offset are clamped to non-negative beats
    so a pedal that started before the first detected beat doesn't
    surface as a negative-beat anchor (which would crash music21's
    measure layout).
    """
    pedal_type = _CC_TO_PEDAL_TYPE.get(pedal.cc)
    if pedal_type is None:
        return None
    on = sec_to_beat(pedal.onset_sec, tempo_map)
    off = sec_to_beat(pedal.offset_sec, tempo_map)
    if on < 0.0:
        on = 0.0
    if off <= on:
        return None
    return PedalEvent(
        onset_beat=round(on, 4),
        offset_beat=round(off, 4),
        type=pedal_type,  # type: ignore[arg-type]
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

    cfg_max_rh = settings.arrange_max_voices_rh
    cfg_max_lh = settings.arrange_max_voices_lh
    max_rh = cfg_max_rh if difficulty != "beginner" else 1
    max_lh = cfg_max_lh if difficulty != "beginner" else 1
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

    # Phase 6: convert Kong's seconds-domain pedal events into the
    # beat-domain PedalEvent the engraver / humanize understands. Empty
    # list means the active transcriber didn't model pedal — humanize
    # will fall back to its chord-symbol heuristic.
    pedal_events: list[PedalEvent] = []
    for raw in payload.pedal_events:
        converted = _pedal_to_score_pedal(raw, tempo_map)
        if converted is not None:
            pedal_events.append(converted)

    log.info(
        "Arranged: RH=%d notes, LH=%d notes (difficulty=%s) pedal_events=%d",
        len(right_hand), len(left_hand), difficulty, len(pedal_events),
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
            downbeats=list(analysis.downbeats),
            pedal_events=pedal_events,
            arrangement_hints=payload.arrangement_hints,
        ),
    )


def _arrange_hf_sync(
    payload: TranscriptionResult,
    difficulty: Difficulty,
    blob_store: BlobStore,
) -> PianoScore:
    """Materialize MIDI, run HF inference, parse back, then classic arrange."""
    mid_in = materialize_transcription_midi_bytes(payload, blob_store)
    mid_out = run_hf_midi_inference(mid_in, settings.arrange_hf_inference_mode)
    rebuilt = transcription_from_midi_bytes(
        mid_out,
        payload,
        extra_warnings=[f"hf_arrange: inference_mode={settings.arrange_hf_inference_mode}"],
    )
    return _arrange_sync(rebuilt, difficulty)


class ArrangeService:
    name = "arrange"

    async def run(
        self,
        payload: TranscriptionResult,
        *,
        difficulty: Difficulty = "intermediate",
        blob_store: BlobStore | None = None,
    ) -> PianoScore:
        # If the interpret stage supplied hints, let them win over the worker default.
        # Future iterations will also consume density / style_tags / dynamic_emphasis / hand_balance.
        hints = payload.arrangement_hints
        if hints is not None and hints.difficulty is not None:
            difficulty = hints.difficulty

        log.info(
            "arrange: start backend=%s tracks_in=%d difficulty=%s",
            settings.arrange_backend,
            len(payload.midi_tracks),
            difficulty,
        )
        use_hf = (
            settings.arrange_backend == "hf_midi_identity"
            and blob_store is not None
        )
        if settings.arrange_backend not in ("rules", "hf_midi_identity"):
            raise ValueError(f"Unknown arrange_backend: {settings.arrange_backend!r}")
        if settings.arrange_backend == "hf_midi_identity" and blob_store is None:
            if not settings.arrange_hf_fallback_to_rules:
                raise ValueError("hf_midi_identity requires blob_store")
            log.warning("hf_midi_identity requires blob_store — falling back to rules")

        if use_hf:
            try:
                score = await asyncio.to_thread(
                    _arrange_hf_sync, payload, difficulty, blob_store,
                )
            except Exception:
                if not settings.arrange_hf_fallback_to_rules:
                    raise
                log.exception("HF arrange failed; falling back to rules")
                score = await asyncio.to_thread(_arrange_sync, payload, difficulty)
        else:
            score = await asyncio.to_thread(_arrange_sync, payload, difficulty)
        log.info(
            "arrange: done rh_notes=%d lh_notes=%d",
            len(score.right_hand),
            len(score.left_hand),
        )
        return score
