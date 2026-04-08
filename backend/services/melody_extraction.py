"""Phase 2 post-processing — waveform-guided melody / harmony split.

Basic Pitch's polyphonic pitch tracker emits one flat stream of notes
with no instrument or role tagging. The downstream arrange stage has a
MELODY → right-hand routing (see ``arrange._assign_hands``) that goes
unused because the transcribe stage only ever emits a single ``PIANO``
track. This module fills that gap by exploiting a piece of Basic Pitch's
own output we weren't using: ``model_output["contour"]``.

``contour`` is a ``(frames, 264)`` salience matrix at 86.13 Hz, with 3
bins per semitone (``CONTOURS_BINS_PER_SEMITONE = 3``) covering MIDI 21
through 108. That's a pre-computed front end for a monophonic F0
tracker — we just weren't using the fine-grained salience, only the
coarser ``note`` grid. Running a Viterbi path over the masked contour
yields a smooth, temporally-coherent lead-voice F0 contour at roughly
33-cent resolution, which we then use to tag each Basic Pitch note as
``MELODY`` (agreement with the path) or ``CHORDS`` (disagreement).

No extra model weights, no extra dependencies — everything here runs
on ``numpy`` which is already pulled in by the ``basic-pitch`` extra.

The module is deliberately decoupled from the rest of the transcribe
pipeline: the entry point ``extract_melody`` takes a numpy contour
matrix + a list of Basic Pitch ``NoteEvent`` tuples and returns a
``(melody_events, chord_events, stats)`` triple. That makes it easy to
unit-test against hand-built synthetic contour matrices.

Bin ↔ MIDI mapping (derived from ``basic_pitch.note_creation``):

    bin = 12 * CONTOURS_BINS_PER_SEMITONE * log2(freq / BASE_FREQ)
        = 3 * (midi - 21)

so integer MIDI pitch ``m`` lives at bin ``3 * (m - 21)``, and each
semitone spans 3 consecutive bins (``3k``, ``3k+1``, ``3k+2``).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from backend.services.transcription_cleanup import NoteEvent

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants mirrored from basic_pitch.constants so the module can be
# imported / unit-tested without pulling basic_pitch in.
# ---------------------------------------------------------------------------

BASE_MIDI = 21                       # A0 — bin 0 of the contour matrix
BINS_PER_SEMITONE = 3                # CONTOURS_BINS_PER_SEMITONE
N_CONTOUR_BINS = 264                 # 88 semitones × 3 bins = 264
FRAME_RATE_HZ = 22050.0 / 256.0      # AUDIO_SAMPLE_RATE / FFT_HOP ≈ 86.13 Hz


def midi_to_bin(midi: float) -> int:
    """Nearest contour bin for a (possibly fractional) MIDI pitch."""
    return int(round(BINS_PER_SEMITONE * (midi - BASE_MIDI)))


def bin_to_midi(bin_idx: int) -> int:
    """Round a contour bin back to the nearest integer MIDI pitch."""
    return BASE_MIDI + int(round(bin_idx / BINS_PER_SEMITONE))


# ---------------------------------------------------------------------------
# Defaults — conservative, tuned against the synthetic test fixtures. Real
# audio will want iteration; see the docstring on ``extract_melody``.
# ---------------------------------------------------------------------------

DEFAULT_MELODY_LOW_MIDI = 55         # G3
DEFAULT_MELODY_HIGH_MIDI = 90        # F#6
DEFAULT_VOICING_FLOOR = 0.15
DEFAULT_TRANSITION_WEIGHT = 0.25     # cost per bin of pitch change
DEFAULT_MAX_TRANSITION_BINS = 12     # ≈ 4 semitones per 11.6 ms frame
DEFAULT_VOICED_ENTER_COST = 1.0      # entering voiced from unvoiced
DEFAULT_UNVOICED_ENTER_COST = 1.0    # leaving voiced to unvoiced
DEFAULT_MIN_NOTE_FRAMES = 4          # ≈ 46 ms
DEFAULT_MATCH_TOL_SEMITONES = 1.0
DEFAULT_MATCH_FRACTION = 0.6

# ---------------------------------------------------------------------------
# Back-fill defaults — recover stable Viterbi runs that the upstream Basic
# Pitch note tracker never emitted (soft sustained melody notes drowned out
# by louder accompaniment). Deliberately conservative: only long, clean
# runs are back-filled, and the synthesized amplitude is clipped low so
# recovery notes don't outshine real ones in the downstream velocity map.
# ---------------------------------------------------------------------------

DEFAULT_BACKFILL_ENABLED = True
DEFAULT_BACKFILL_MIN_DURATION_SEC = 0.12
DEFAULT_BACKFILL_OVERLAP_FRACTION = 0.5
DEFAULT_BACKFILL_MIN_AMP = 0.15
DEFAULT_BACKFILL_MAX_AMP = 0.60


# ---------------------------------------------------------------------------
# Stats surfaced back to the caller for telemetry / QualitySignal
# ---------------------------------------------------------------------------

@dataclass
class MelodyExtractionStats:
    """Per-run summary of what the melody extractor decided."""
    input_note_count: int = 0
    melody_note_count: int = 0
    chord_note_count: int = 0
    voiced_frame_fraction: float = 0.0  # share of frames on the Viterbi path
    # Count of melody notes synthesized from the Viterbi path because no
    # matching upstream note event existed (back-fill of soft sustained
    # melody lines the Basic Pitch note tracker missed).
    backfilled_note_count: int = 0
    warnings: list[str] = field(default_factory=list)
    # True when numpy (or contour input) was unavailable and we skipped
    # extraction entirely. Callers should fall back to a single-track
    # TranscriptionResult in that case.
    skipped: bool = False

    def as_warnings(self) -> list[str]:
        if self.skipped:
            return ["melody extraction skipped (numpy or contour unavailable)"]
        out: list[str] = []
        if self.melody_note_count or self.chord_note_count:
            out.append(
                f"melody split: {self.melody_note_count} melody / "
                f"{self.chord_note_count} chord notes "
                f"({self.voiced_frame_fraction * 100:.0f}% voiced)"
            )
        if self.backfilled_note_count:
            out.append(
                f"melody: back-filled {self.backfilled_note_count} "
                f"missed notes from Viterbi path"
            )
        out.extend(self.warnings)
        return out


# ---------------------------------------------------------------------------
# Core Viterbi tracer
# ---------------------------------------------------------------------------

def _trace_f0_contour(
    contour: Any,  # np.ndarray (frames, N_CONTOUR_BINS)
    *,
    low_bin: int,
    high_bin: int,
    voicing_floor: float,
    transition_weight: float,
    max_transition_bins: int,
    voiced_enter_cost: float,
    unvoiced_enter_cost: float,
) -> Any:  # np.ndarray (frames,), int — -1 for unvoiced
    """Run a Viterbi path over the masked contour matrix.

    The state space is ``N_CONTOUR_BINS + 1`` — one state per contour
    bin plus a single "unvoiced" state. Voiced emission cost is
    ``-log(contour[t, b] + eps)`` within the melody band, ``+inf``
    outside it. The unvoiced state has a constant emission cost
    ``-log(voicing_floor + eps)`` — a frame prefers unvoiced iff no bin
    inside the band is at least as salient as ``voicing_floor``.

    Transition costs:
      * voiced → voiced (Δb bins): ``transition_weight * |Δb|``, capped
        at ``max_transition_bins``
      * voiced → unvoiced: ``unvoiced_enter_cost``
      * unvoiced → voiced: ``voiced_enter_cost``
      * self-loops: free

    Implementation notes:
      * Vectorized as ``2 * max_transition_bins + 1`` shift-and-min ops
        per frame via ``np.roll`` with boundary masking — O(T * W * N).
      * Back-pointers are packed into ``int16`` — the delta is in
        ``[-W, W]`` and the "came from unvoiced" marker is stored as the
        sentinel ``-W - 1``.
    """
    import numpy as np  # noqa: PLC0415

    n_frames, n_bins = contour.shape
    if n_bins != N_CONTOUR_BINS:
        raise ValueError(
            f"contour has {n_bins} bins, expected {N_CONTOUR_BINS}"
        )

    eps = 1e-6
    inf = np.float32(1e18)

    # Band mask — bins outside [low_bin, high_bin] get +inf emission.
    band_mask = np.zeros(n_bins, dtype=bool)
    lo = max(0, low_bin)
    hi = min(n_bins, high_bin + 1)
    band_mask[lo:hi] = True

    # Precompute emission for voiced states: shape (T, N)
    voiced_emission = -np.log(np.clip(contour, eps, None).astype(np.float32))
    voiced_emission[:, ~band_mask] = inf
    # Unvoiced emission — constant scalar, but vectorized per-frame.
    unvoiced_emission = float(-math.log(voicing_floor + eps))

    W = int(max_transition_bins)
    delta_range = np.arange(-W, W + 1, dtype=np.int32)
    transition = (transition_weight * np.abs(delta_range)).astype(np.float32)

    # dp_voiced[b] = best cost to end at voiced state b at time t
    dp_voiced = np.full(n_bins, inf, dtype=np.float32)

    # Initial frame: start from any state uniformly. dp_unvoiced stays a
    # plain Python float throughout — it's a scalar and mixing it with
    # numpy scalars just confuses type checking for no runtime benefit.
    dp_voiced[:] = voiced_emission[0]
    dp_unvoiced: float = float(unvoiced_emission)

    # Back-pointers: for each frame t > 0 and each state, what delta
    # (from voiced) or "came from unvoiced" produced the min.
    # Voiced back-ptr sentinels: delta ∈ [-W, W] plus the "from unvoiced"
    # marker -W - 1. Unvoiced back-ptr is 0 for "self" and 1 for "from
    # voiced (the argmin bin of dp_prev)".
    bp_voiced = np.zeros((n_frames, n_bins), dtype=np.int16)
    bp_unvoiced = np.zeros(n_frames, dtype=np.int16)
    bp_unvoiced_arg = np.zeros(n_frames, dtype=np.int32)

    for t in range(1, n_frames):
        # ---------- voiced ← voiced (shift-and-min over ±W bins) -----
        best_voiced_cost = np.full(n_bins, inf, dtype=np.float32)
        best_delta = np.zeros(n_bins, dtype=np.int16)

        for di, d_arr in enumerate(delta_range):
            # dp_prev[b - d] = cost of being at bin b-d at t-1,
            # then moving by +d to bin b at t.
            d = int(d_arr)
            shifted = np.full(n_bins, inf, dtype=np.float32)
            if d >= 0:
                shifted[d:] = dp_voiced[: n_bins - d]
            else:
                shifted[: n_bins + d] = dp_voiced[-d:]
            candidate = shifted + transition[di]
            improved = candidate < best_voiced_cost
            best_voiced_cost = np.where(improved, candidate, best_voiced_cost)
            best_delta = np.where(improved, np.int16(d), best_delta)

        # ---------- voiced ← unvoiced ---------------------------------
        from_unv = dp_unvoiced + voiced_enter_cost
        take_unv = from_unv < best_voiced_cost
        best_voiced_cost = np.where(take_unv, from_unv, best_voiced_cost)
        best_delta = np.where(take_unv, np.int16(-W - 1), best_delta)

        new_voiced = voiced_emission[t] + best_voiced_cost

        # ---------- unvoiced ← {self, voiced} -------------------------
        voiced_argmin = int(np.argmin(dp_voiced))
        from_voiced = float(dp_voiced[voiced_argmin]) + unvoiced_enter_cost
        if dp_unvoiced <= from_voiced:
            new_unvoiced_scalar: float = unvoiced_emission + dp_unvoiced
            bp_unvoiced[t] = 0  # stayed unvoiced
        else:
            new_unvoiced_scalar = unvoiced_emission + from_voiced
            bp_unvoiced[t] = 1  # came from voiced
            bp_unvoiced_arg[t] = voiced_argmin

        bp_voiced[t] = best_delta
        dp_voiced = new_voiced
        dp_unvoiced = new_unvoiced_scalar

    # ---------- Backtrack -------------------------------------------------
    path = np.full(n_frames, -1, dtype=np.int32)
    final_voiced_argmin = int(np.argmin(dp_voiced))
    final_voiced_cost = float(dp_voiced[final_voiced_argmin])
    if final_voiced_cost <= float(dp_unvoiced):
        state: int = final_voiced_argmin
        voiced = True
    else:
        state = -1
        voiced = False

    for t in range(n_frames - 1, -1, -1):
        if voiced:
            path[t] = state
            if t == 0:
                break
            bt_delta = int(bp_voiced[t, state])
            if bt_delta == -W - 1:
                voiced = False
            else:
                state = state - bt_delta
        else:
            path[t] = -1
            if t == 0:
                break
            if bp_unvoiced[t] == 1:
                voiced = True
                state = int(bp_unvoiced_arg[t])
            # else stay unvoiced
    return path


# ---------------------------------------------------------------------------
# Path → note-range segmentation
# ---------------------------------------------------------------------------

def _path_to_midi_runs(path: Any) -> list[tuple[int, int, int]]:
    """Group consecutive voiced frames at the same integer MIDI pitch.

    Returns a list of ``(start_frame, end_frame_exclusive, midi_pitch)``
    tuples. Unvoiced frames (path value < 0) break runs. We round each
    voiced bin to its nearest integer MIDI pitch via :func:`bin_to_midi`
    — the sub-semitone precision is thrown away here because the
    downstream tagging only needs semitone resolution.
    """
    runs: list[tuple[int, int, int]] = []
    n = len(path)
    i = 0
    while i < n:
        if path[i] < 0:
            i += 1
            continue
        midi = bin_to_midi(int(path[i]))
        j = i + 1
        while j < n and path[j] >= 0 and bin_to_midi(int(path[j])) == midi:
            j += 1
        runs.append((i, j, midi))
        i = j
    return runs


# ---------------------------------------------------------------------------
# Note tagging against a Viterbi path
# ---------------------------------------------------------------------------

def _split_notes_by_path_agreement(
    events: list[NoteEvent],
    path: Any,
    *,
    low_midi: int,
    high_midi: int,
    frame_rate_hz: float,
    match_tolerance_semitones: float,
    match_fraction: float,
) -> tuple[list[NoteEvent], list[NoteEvent]]:
    """Tag each note tuple as in-voice or out-of-voice w.r.t. a Viterbi path.

    A note becomes *in-voice* (first return list) when:
      1. Its pitch lies inside ``[low_midi, high_midi]``.
      2. At least ``match_fraction`` of its frames overlap a voiced
         Viterbi path frame within ``match_tolerance_semitones``.

    Everything else flows to the *out-of-voice* (second return list)
    bucket. The function is voice-agnostic — Phase 2 (melody) and Phase
    3 (bass) call it with different bands and paths.
    """
    if not events:
        return [], []

    n_frames = len(path)
    in_voice: list[NoteEvent] = []
    out_of_voice: list[NoteEvent] = []
    tol_bins = match_tolerance_semitones * BINS_PER_SEMITONE

    for ev in events:
        start_sec, end_sec, pitch, amp, bends = ev
        if pitch < low_midi or pitch > high_midi:
            out_of_voice.append(ev)
            continue

        f0 = max(0, int(round(start_sec * frame_rate_hz)))
        f1 = min(n_frames, int(round(end_sec * frame_rate_hz)))
        if f1 <= f0:
            out_of_voice.append(ev)
            continue

        target_bin = midi_to_bin(pitch)
        match = 0
        total = f1 - f0
        for f in range(f0, f1):
            pb = int(path[f])
            if pb < 0:
                continue
            if abs(pb - target_bin) <= tol_bins:
                match += 1

        if total > 0 and (match / total) >= match_fraction:
            in_voice.append(ev)
        else:
            out_of_voice.append(ev)

    return in_voice, out_of_voice


# ---------------------------------------------------------------------------
# Back-fill — synthesize notes for stable Viterbi runs with no matching event
# ---------------------------------------------------------------------------

def _backfill_missed_melody_notes(
    melody_events: list[NoteEvent],
    contour: Any,  # np.ndarray (frames, N_CONTOUR_BINS)
    path: Any,     # np.ndarray (frames,), int; -1 = unvoiced
    *,
    frame_rate_hz: float,
    min_duration_sec: float,
    overlap_fraction: float,
    min_amp: float,
    max_amp: float,
    match_tolerance_semitones: float,
) -> tuple[list[NoteEvent], int]:
    """Synthesize melody note events for stable Viterbi runs with no match.

    The Viterbi F0 tracer sees the contour salience matrix directly, so
    it finds stable melody fundamentals that the upstream Basic Pitch
    note tracker sometimes misses — typically soft sustained lines under
    louder accompaniment. When we find a run of ≥ ``min_duration_sec``
    frames at a single pitch that does not overlap any existing melody
    event (within ``match_tolerance_semitones`` and by at least
    ``overlap_fraction`` of the run's duration), we build a synthetic
    ``NoteEvent`` from it.

    The synthesized amplitude is the mean contour salience inside the
    run at the target bin, **clipped** to ``[min_amp, max_amp]``. The
    upper clip matters: these are recovery notes, not hero notes, and
    uncapped they'd arrive at the downstream velocity formula with
    values that rival real detections.

    **Do not re-run** :func:`cleanup_note_events` over a back-filled
    list — the low clipped amplitude makes synthesized notes look like
    ghost tails and they get dropped, defeating the purpose.

    Returns ``(combined_events, added_count)``.
    """
    import numpy as np  # noqa: PLC0415

    runs = _path_to_midi_runs(path)
    if not runs:
        return list(melody_events), 0

    min_frames = max(1, int(round(min_duration_sec * frame_rate_hz)))
    added: list[NoteEvent] = []

    for start_f, end_f, midi in runs:
        if end_f - start_f < min_frames:
            continue

        start_sec = start_f / frame_rate_hz
        end_sec = end_f / frame_rate_hz
        run_duration = end_sec - start_sec
        if run_duration <= 0:
            continue

        # Skip the run if any existing melody event already covers it.
        has_match = False
        for ev in melody_events:
            ev_start, ev_end, ev_pitch, _, _ = ev
            if abs(ev_pitch - midi) > match_tolerance_semitones:
                continue
            overlap_start = max(start_sec, ev_start)
            overlap_end = min(end_sec, ev_end)
            overlap = overlap_end - overlap_start
            if overlap >= overlap_fraction * run_duration:
                has_match = True
                break
        if has_match:
            continue

        # Synthesize. Amplitude = clipped mean salience at the target
        # bin across the run. Casting to float keeps the tuple layout
        # consistent with the Basic Pitch-sourced entries.
        target_bin = midi_to_bin(midi)
        if 0 <= target_bin < contour.shape[1]:
            bin_slice = contour[start_f:end_f, target_bin]
            mean_salience = float(np.mean(bin_slice)) if bin_slice.size else 0.0
        else:
            mean_salience = 0.0
        amp = max(min_amp, min(max_amp, mean_salience))

        added.append((start_sec, end_sec, int(midi), amp, None))

    if not added:
        return list(melody_events), 0

    combined = list(melody_events) + added
    combined.sort(key=lambda e: (e[0], e[2]))
    return combined, len(added)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_melody(
    contour: Any,  # np.ndarray (frames, 264) or None
    note_events: list[NoteEvent],
    *,
    melody_low_midi: int = DEFAULT_MELODY_LOW_MIDI,
    melody_high_midi: int = DEFAULT_MELODY_HIGH_MIDI,
    voicing_floor: float = DEFAULT_VOICING_FLOOR,
    transition_weight: float = DEFAULT_TRANSITION_WEIGHT,
    max_transition_bins: int = DEFAULT_MAX_TRANSITION_BINS,
    voiced_enter_cost: float = DEFAULT_VOICED_ENTER_COST,
    unvoiced_enter_cost: float = DEFAULT_UNVOICED_ENTER_COST,
    match_tolerance_semitones: float = DEFAULT_MATCH_TOL_SEMITONES,
    match_fraction: float = DEFAULT_MATCH_FRACTION,
    frame_rate_hz: float = FRAME_RATE_HZ,
    backfill_enabled: bool = DEFAULT_BACKFILL_ENABLED,
    backfill_min_duration_sec: float = DEFAULT_BACKFILL_MIN_DURATION_SEC,
    backfill_overlap_fraction: float = DEFAULT_BACKFILL_OVERLAP_FRACTION,
    backfill_min_amp: float = DEFAULT_BACKFILL_MIN_AMP,
    backfill_max_amp: float = DEFAULT_BACKFILL_MAX_AMP,
) -> tuple[list[NoteEvent], list[NoteEvent], MelodyExtractionStats]:
    """Run Phase 2 melody extraction over Basic Pitch output.

    Returns ``(melody_events, chord_events, stats)``. When numpy is
    unavailable or the contour matrix is missing / malformed, returns
    ``([], note_events, stats)`` with ``stats.skipped = True`` so the
    caller can fall back to a single-track result without losing notes.

    When ``backfill_enabled``, stable Viterbi runs with no matching
    Basic Pitch note are synthesized into the melody list before
    return. See :func:`_backfill_missed_melody_notes` for the semantics
    — in short, this recovers soft sustained lead notes the polyphonic
    tracker lost under accompaniment.

    Tuning: the defaults are conservative. For songs with a strong
    vocal lead, raising ``match_fraction`` to ~0.75 and narrowing
    ``melody_high_midi`` cuts false positives; for instrumental tracks
    with a quieter lead, lowering ``voicing_floor`` to ~0.08 picks up
    softer melody notes at the cost of more spurious voicing.
    """
    stats = MelodyExtractionStats(input_note_count=len(note_events))

    if contour is None:
        stats.skipped = True
        return [], list(note_events), stats

    try:
        import numpy as np  # noqa: PLC0415
    except ImportError:
        stats.skipped = True
        return [], list(note_events), stats

    contour_arr = np.asarray(contour)
    if contour_arr.ndim != 2 or contour_arr.shape[1] != N_CONTOUR_BINS:
        stats.warnings.append(
            f"contour has unexpected shape {contour_arr.shape}; skipping"
        )
        stats.skipped = True
        return [], list(note_events), stats
    if contour_arr.shape[0] < 2:
        stats.skipped = True
        return [], list(note_events), stats

    low_bin = midi_to_bin(melody_low_midi)
    high_bin = midi_to_bin(melody_high_midi)
    if low_bin >= high_bin:
        stats.warnings.append(
            f"invalid melody band [{melody_low_midi}, {melody_high_midi}]"
        )
        stats.skipped = True
        return [], list(note_events), stats

    try:
        path = _trace_f0_contour(
            contour_arr,
            low_bin=low_bin,
            high_bin=high_bin,
            voicing_floor=voicing_floor,
            transition_weight=transition_weight,
            max_transition_bins=max_transition_bins,
            voiced_enter_cost=voiced_enter_cost,
            unvoiced_enter_cost=unvoiced_enter_cost,
        )
    except Exception as exc:  # noqa: BLE001 — Viterbi must never sink transcribe
        log.warning("Viterbi F0 tracer failed: %s", exc)
        stats.warnings.append(f"viterbi failed: {exc}")
        stats.skipped = True
        return [], list(note_events), stats

    voiced_frac = float((path >= 0).mean()) if len(path) else 0.0
    stats.voiced_frame_fraction = voiced_frac

    melody, chords = _split_notes_by_path_agreement(
        note_events,
        path,
        low_midi=melody_low_midi,
        high_midi=melody_high_midi,
        frame_rate_hz=frame_rate_hz,
        match_tolerance_semitones=match_tolerance_semitones,
        match_fraction=match_fraction,
    )

    # Back-fill stable Viterbi runs that no Basic Pitch event covers.
    # Runs whose rounded-integer MIDI falls outside the melody band are
    # filtered here (via ``low/high_midi``) so a low-band peak the tracer
    # catches doesn't get back-filled as a melody note.
    if backfill_enabled:
        try:
            melody, backfill_count = _backfill_missed_melody_notes(
                melody,
                contour_arr,
                path,
                frame_rate_hz=frame_rate_hz,
                min_duration_sec=backfill_min_duration_sec,
                overlap_fraction=backfill_overlap_fraction,
                min_amp=backfill_min_amp,
                max_amp=backfill_max_amp,
                match_tolerance_semitones=match_tolerance_semitones,
            )
            # Restrict added notes to the melody band — the Viterbi path
            # is already band-masked, but be defensive in case a future
            # caller widens it.
            if backfill_count:
                melody = [
                    ev for ev in melody
                    if melody_low_midi <= ev[2] <= melody_high_midi
                ]
            stats.backfilled_note_count = backfill_count
        except Exception as exc:  # noqa: BLE001 — never let back-fill sink transcribe
            log.warning("melody back-fill failed: %s", exc)
            stats.warnings.append(f"back-fill failed: {exc}")

    stats.melody_note_count = len(melody)
    stats.chord_note_count = len(chords)
    log.debug(
        "melody extraction: %d notes → %d melody / %d chords "
        "(+%d back-filled, %.0f%% voiced)",
        len(note_events),
        len(melody),
        len(chords),
        stats.backfilled_note_count,
        voiced_frac * 100.0,
    )
    return melody, chords, stats
