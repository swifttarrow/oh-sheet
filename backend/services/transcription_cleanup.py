"""Phase 1 post-processing heuristics for Basic Pitch transcriptions.

Basic Pitch is a polyphonic pitch tracker, not a musical transcriber, and
its raw output has three classes of artifact that hurt the downstream
arrangement:

1. **Fragmented sustains.** A single held note often gets split into two
   or three short notes with sub-frame gaps, because the frame-level
   activation dips below ``frame_threshold`` for a moment mid-sustain.

2. **Octave ghosts.** A note at pitch ``p`` induces a much quieter
   detection one octave up at ``p+12`` from the overtone series. These
   harmonic artifacts are distinguishable from real octave doublings
   because the upper note has a much lower activation amplitude.

3. **Ghost-tail notes.** Brief, quiet notes appearing just after a real
   note decays (reverb tails, model uncertainty at offset boundaries).

This module runs a pure-Python cleanup pass over Basic Pitch's
``note_events`` tuple format so we can rebuild ``pretty_midi`` from a
cleaner set of notes before handing off to the contract.

Everything here is deterministic and dependency-free — no numpy, no
librosa, no basic_pitch import — so it runs in CI regardless of whether
the ``basic-pitch`` extra is installed, and is easy to unit-test with
synthetic fixtures.

Future phases will layer waveform-guided melody/bass tracing on top of
this using Basic Pitch's ``model_output["contour"]`` salience matrix.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# Basic Pitch's note_events format:
#   (start_sec, end_sec, pitch_midi, amplitude, pitch_bends_or_none)
# We keep the last slot (pitch_bends) opaque and pass it through unchanged
# so downstream ``note_events_to_midi`` still sees the bend track.
NoteEvent = tuple[float, float, int, float, Any]


# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------
# These defaults err on the conservative side — they catch clear artifacts
# without pruning legitimate musical notes. Tunable via config in the caller.

DEFAULT_MERGE_GAP_SEC = 0.03
DEFAULT_OCTAVE_AMP_RATIO = 0.6
DEFAULT_OCTAVE_ONSET_TOL_SEC = 0.05
DEFAULT_GHOST_MAX_DURATION_SEC = 0.05
DEFAULT_GHOST_AMP_MEDIAN_SCALE = 0.5

# Pass 5 — energy gating
DEFAULT_ENERGY_GATE_MAX_SUSTAIN_SEC = 2.0
DEFAULT_ENERGY_GATE_FLOOR_RATIO = 0.1
DEFAULT_ENERGY_GATE_TAIL_SEC = 0.05


@dataclass
class CleanupStats:
    """How many notes each pass touched — used for telemetry + warnings."""
    input_count: int = 0
    output_count: int = 0
    merged: int = 0                  # pairs merged (count reduction)
    octave_ghosts_dropped: int = 0
    ghost_tails_dropped: int = 0
    energy_gated: int = 0            # notes trimmed by energy gating (Pass 5)
    warnings: list[str] = field(default_factory=list)

    def as_warnings(self) -> list[str]:
        """One-line human summary entries for the QualitySignal."""
        out: list[str] = []
        if self.merged:
            out.append(f"cleanup: merged {self.merged} fragmented sustains")
        if self.octave_ghosts_dropped:
            out.append(f"cleanup: dropped {self.octave_ghosts_dropped} octave ghosts")
        if self.ghost_tails_dropped:
            out.append(f"cleanup: dropped {self.ghost_tails_dropped} ghost-tail notes")
        if self.energy_gated:
            out.append(f"cleanup: energy-gated {self.energy_gated} note offsets")
        out.extend(self.warnings)
        return out


# ---------------------------------------------------------------------------
# Pass 1 — merge fragmented sustains
# ---------------------------------------------------------------------------

def _merge_fragmented_sustains(
    events: list[NoteEvent],
    max_gap_sec: float,
) -> tuple[list[NoteEvent], int]:
    """Join consecutive same-pitch notes whose gap is below ``max_gap_sec``.

    Merged note spans ``[min(starts), max(ends)]`` and keeps the loudest
    amplitude of the two (the upstream velocity formula ``round(127 * amp)``
    is monotonic, so this preserves the louder of the two velocities). We
    do not attempt to merge pitch-bend tracks — whichever event comes
    first keeps its bends, which is the same convention basic_pitch's
    ``drop_overlapping_pitch_bends`` uses.
    """
    if not events:
        return [], 0

    by_pitch: dict[int, list[NoteEvent]] = defaultdict(list)
    for ev in events:
        by_pitch[ev[2]].append(ev)

    merged_out: list[NoteEvent] = []
    merged_count = 0
    for pitch, group in by_pitch.items():
        group.sort(key=lambda e: e[0])
        run: NoteEvent = group[0]
        for ev in group[1:]:
            gap = ev[0] - run[1]
            if gap <= max_gap_sec:
                run = (
                    run[0],
                    max(run[1], ev[1]),
                    pitch,
                    max(run[3], ev[3]),
                    run[4],
                )
                merged_count += 1
            else:
                merged_out.append(run)
                run = ev
        merged_out.append(run)

    merged_out.sort(key=lambda e: (e[0], e[2]))
    return merged_out, merged_count


# ---------------------------------------------------------------------------
# Pass 2 — octave-ghost pruning
# ---------------------------------------------------------------------------

def _prune_octave_ghosts(
    events: list[NoteEvent],
    amp_ratio: float,
    onset_tol_sec: float,
) -> tuple[list[NoteEvent], int]:
    """Drop a note at pitch ``p+12`` if a near-simultaneous note at ``p``
    is loud enough that the upper is plausibly a harmonic artifact.

    An upper event is flagged when there exists a lower event whose onset
    falls within ``onset_tol_sec`` of the upper's onset *and* whose
    amplitude satisfies ``upper.amp < amp_ratio * lower.amp``. Real
    musical octave doublings (both loud) pass through untouched because
    their amplitudes are comparable.
    """
    if not events:
        return [], 0

    # Index by pitch so each ghost lookup is O(candidates_at_lower_pitch).
    by_pitch: dict[int, list[tuple[float, float, int]]] = defaultdict(list)
    for idx, ev in enumerate(events):
        by_pitch[ev[2]].append((ev[0], ev[3], idx))
    for lst in by_pitch.values():
        lst.sort()

    drop: set[int] = set()
    for idx, ev in enumerate(events):
        lower_pitch = ev[2] - 12
        if lower_pitch < 0:
            continue
        candidates = by_pitch.get(lower_pitch)
        if not candidates:
            continue
        for onset, amp_lower, _j in candidates:
            if onset > ev[0] + onset_tol_sec:
                break
            if abs(onset - ev[0]) <= onset_tol_sec and ev[3] < amp_ratio * amp_lower:
                drop.add(idx)
                break

    if not drop:
        return list(events), 0
    return [e for i, e in enumerate(events) if i not in drop], len(drop)


# ---------------------------------------------------------------------------
# Pass 3 — ghost-tail pruning
# ---------------------------------------------------------------------------

def _prune_ghost_tails(
    events: list[NoteEvent],
    max_duration_sec: float,
    amp_median_scale: float,
) -> tuple[list[NoteEvent], int]:
    """Drop notes that are both shorter than ``max_duration_sec`` **and**
    quieter than ``amp_median_scale × median(amplitudes)``.

    The two-predicate rule is deliberate: a short but loud staccato is
    probably real, and a long but quiet sustained pad is probably real.
    Only the intersection — short *and* quiet — matches the ghost-tail
    profile we want to kill.
    """
    if not events:
        return [], 0

    amps = sorted(e[3] for e in events)
    mid = len(amps) // 2
    median_amp = amps[mid] if len(amps) % 2 else 0.5 * (amps[mid - 1] + amps[mid])
    threshold = amp_median_scale * median_amp

    kept: list[NoteEvent] = []
    dropped = 0
    for ev in events:
        duration = ev[1] - ev[0]
        if duration < max_duration_sec and ev[3] < threshold:
            dropped += 1
            continue
        kept.append(ev)
    return kept, dropped


# ---------------------------------------------------------------------------
# Pass 5 — amplitude envelope gating
# ---------------------------------------------------------------------------

# Type alias for the envelope: a sequence of (time_sec, rms_value) pairs
# produced by _compute_amplitude_envelope() in transcribe.py.
AmplitudeEnvelope = list[tuple[float, float]]


def _gate_offsets_by_energy(
    events: list[NoteEvent],
    *,
    max_sustain_sec: float,
    floor_ratio: float,
    tail_sec: float,
    amplitude_envelope: AmplitudeEnvelope | None = None,
) -> tuple[list[NoteEvent], int]:
    """Tighten note offsets based on amplitude decay.

    Basic Pitch's frame activation lingers past the real note end due to
    reverb, sustain pedal, and hysteresis in the onset/offset model. This
    pass trims suspiciously long notes to the point where energy drops
    below a floor.

    When ``amplitude_envelope`` is provided (list of ``(time, rms)``
    tuples from the source waveform):

      * For each note longer than ``max_sustain_sec``, look at the RMS
        values during the note's time span.
      * Find the peak RMS within the note and locate the first point
        where RMS drops below ``floor_ratio * peak_rms``.
      * Trim the offset to that decay point plus ``tail_sec`` allowance.

    When ``amplitude_envelope`` is ``None`` (no waveform data), a simpler
    heuristic fires:

      * If a note is longer than ``max_sustain_sec`` **and** its
        amplitude (from Basic Pitch) is below the median of all note
        amplitudes, trim it to ``max_sustain_sec``.
      * This catches the worst reverb-bleed offenders without waveform
        data.

    Returns the (possibly trimmed) event list and the count of notes
    whose offsets were adjusted.
    """
    if not events:
        return [], 0

    gated = 0

    if amplitude_envelope is not None and amplitude_envelope:
        # Build a time-sorted copy for binary-search lookups.
        env = sorted(amplitude_envelope, key=lambda p: p[0])
        env_times = [p[0] for p in env]
        env_rms = [p[1] for p in env]

        result: list[NoteEvent] = []
        for ev in events:
            start, end, pitch, amp, bends = ev
            duration = end - start
            if duration <= max_sustain_sec:
                result.append(ev)
                continue

            # Find envelope samples within the note's span.
            lo = _bisect_left(env_times, start)
            hi = _bisect_right(env_times, end)
            if lo >= hi:
                # No envelope data in this range — keep unchanged.
                result.append(ev)
                continue

            peak_rms = max(env_rms[lo:hi])
            if peak_rms <= 0.0:
                result.append(ev)
                continue

            threshold = floor_ratio * peak_rms
            # Scan for a sustained drop below the floor (3+ consecutive
            # frames) to avoid false triggers from momentary dips like
            # reverb flutter or transient noise.
            decay_time: float | None = None
            consecutive = 0
            hysteresis = 3
            for i in range(lo, hi):
                if env_rms[i] < threshold and env_times[i] > start:
                    consecutive += 1
                    if consecutive >= hysteresis:
                        # Mark decay at the first frame of the run.
                        decay_time = env_times[i - hysteresis + 1]
                        break
                else:
                    consecutive = 0

            if decay_time is not None:
                new_end = min(end, decay_time + tail_sec)
                # Never shorten a note to less than its onset.
                new_end = max(new_end, start + 0.01)
                if new_end < end:
                    result.append((start, new_end, pitch, amp, bends))
                    gated += 1
                else:
                    result.append(ev)
            else:
                result.append(ev)
        return result, gated

    # Heuristic path — no envelope available.
    amps = sorted(e[3] for e in events)
    mid = len(amps) // 2
    median_amp = amps[mid] if len(amps) % 2 else 0.5 * (amps[mid - 1] + amps[mid])

    result_h: list[NoteEvent] = []
    for ev in events:
        start, end, pitch, amp, bends = ev
        duration = end - start
        if duration > max_sustain_sec and amp < median_amp:
            new_end = start + max_sustain_sec
            result_h.append((start, new_end, pitch, amp, bends))
            gated += 1
        else:
            result_h.append(ev)
    return result_h, gated


def _bisect_left(a: list[float], x: float) -> int:
    """Pure-Python bisect_left — no bisect import needed."""
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _bisect_right(a: list[float], x: float) -> int:
    """Pure-Python bisect_right — no bisect import needed."""
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------------------
# Per-role cleanup helper
# ---------------------------------------------------------------------------

def cleanup_for_role(
    events: list[NoteEvent],
    role: str,
    settings: Any,
    *,
    amplitude_envelope: AmplitudeEnvelope | None = None,
) -> tuple[list[NoteEvent], CleanupStats]:
    """Run cleanup with role-specific thresholds.

    ``role`` is one of ``"melody"``, ``"bass"``, ``"chords"``, or any
    other string (falls back to global defaults). ``settings`` is the
    application ``Settings`` instance.

    Role-specific overrides are pulled from config attributes named
    ``cleanup_{role}_{param}``; any that don't exist fall through to
    the global ``cleanup_{param}`` default.
    """
    # Build kwargs from per-role config, falling back to globals.
    def _get(role_key: str, global_key: str) -> float:
        val = getattr(settings, role_key, None)
        if val is not None:
            return float(val)
        return float(getattr(settings, global_key))

    merge_gap = _get(f"cleanup_{role}_merge_gap_sec", "cleanup_merge_gap_sec")
    octave_amp = _get(f"cleanup_{role}_octave_amp_ratio", "cleanup_octave_amp_ratio")
    ghost_dur = _get(f"cleanup_{role}_ghost_max_duration_sec", "cleanup_ghost_max_duration_sec")

    energy_enabled = getattr(settings, "cleanup_energy_gate_enabled", True)
    max_sustain = float(getattr(settings, "cleanup_energy_gate_max_sustain_sec",
                                DEFAULT_ENERGY_GATE_MAX_SUSTAIN_SEC))
    floor_ratio = float(getattr(settings, "cleanup_energy_gate_floor_ratio",
                                DEFAULT_ENERGY_GATE_FLOOR_RATIO))
    tail_sec = float(getattr(settings, "cleanup_energy_gate_tail_sec",
                             DEFAULT_ENERGY_GATE_TAIL_SEC))

    cleaned, stats = cleanup_note_events(
        events,
        merge_gap_sec=merge_gap,
        octave_amp_ratio=octave_amp,
        octave_onset_tol_sec=float(getattr(settings, "cleanup_octave_onset_tol_sec",
                                           DEFAULT_OCTAVE_ONSET_TOL_SEC)),
        ghost_max_duration_sec=ghost_dur,
        ghost_amp_median_scale=float(getattr(settings, "cleanup_ghost_amp_median_scale",
                                             DEFAULT_GHOST_AMP_MEDIAN_SCALE)),
        amplitude_envelope=amplitude_envelope if energy_enabled else None,
        energy_gate_max_sustain_sec=max_sustain,
        energy_gate_floor_ratio=floor_ratio,
        energy_gate_tail_sec=tail_sec,
        energy_gate_enabled=energy_enabled,
    )
    return cleaned, stats


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def cleanup_note_events(
    note_events: list[NoteEvent],
    *,
    merge_gap_sec: float = DEFAULT_MERGE_GAP_SEC,
    octave_amp_ratio: float = DEFAULT_OCTAVE_AMP_RATIO,
    octave_onset_tol_sec: float = DEFAULT_OCTAVE_ONSET_TOL_SEC,
    ghost_max_duration_sec: float = DEFAULT_GHOST_MAX_DURATION_SEC,
    ghost_amp_median_scale: float = DEFAULT_GHOST_AMP_MEDIAN_SCALE,
    amplitude_envelope: AmplitudeEnvelope | None = None,
    energy_gate_max_sustain_sec: float = DEFAULT_ENERGY_GATE_MAX_SUSTAIN_SEC,
    energy_gate_floor_ratio: float = DEFAULT_ENERGY_GATE_FLOOR_RATIO,
    energy_gate_tail_sec: float = DEFAULT_ENERGY_GATE_TAIL_SEC,
    energy_gate_enabled: bool = True,
) -> tuple[list[NoteEvent], CleanupStats]:
    """Run the Phase 1 cleanup pipeline over Basic Pitch note events.

    Order of operations matters:

    1. **Merge** first so a fragmented sustain is seen as one note by
       the later passes (otherwise the first fragment may be flagged as
       a ghost tail because its duration is tiny).
    2. **Octave prune** next so downstream amplitude statistics reflect
       the real set of notes, not harmonic duplicates.
    3. **Ghost-tail prune** last, operating on an already-merged,
       already-dedup'd list.
    4. **Trim overlapping offsets** — same-pitch overlap resolution
       (placeholder for future implementation).
    5. **Energy gating** — tighten offsets of suspiciously long notes
       using amplitude envelope data (when available) or a
       duration/amplitude heuristic (when not).

    Returns the cleaned events and a :class:`CleanupStats` summary for
    the caller to surface in ``QualitySignal.warnings``.
    """
    stats = CleanupStats(input_count=len(note_events))

    merged, merged_count = _merge_fragmented_sustains(
        list(note_events), max_gap_sec=merge_gap_sec,
    )
    stats.merged = merged_count

    deghosted, octave_dropped = _prune_octave_ghosts(
        merged,
        amp_ratio=octave_amp_ratio,
        onset_tol_sec=octave_onset_tol_sec,
    )
    stats.octave_ghosts_dropped = octave_dropped

    cleaned, tails_dropped = _prune_ghost_tails(
        deghosted,
        max_duration_sec=ghost_max_duration_sec,
        amp_median_scale=ghost_amp_median_scale,
    )
    stats.ghost_tails_dropped = tails_dropped

    # Pass 5 — energy gating. Trim suspiciously long note offsets
    # based on amplitude envelope decay (when available) or a simpler
    # duration/amplitude heuristic (when not).
    if energy_gate_enabled:
        cleaned, energy_count = _gate_offsets_by_energy(
            cleaned,
            max_sustain_sec=energy_gate_max_sustain_sec,
            floor_ratio=energy_gate_floor_ratio,
            tail_sec=energy_gate_tail_sec,
            amplitude_envelope=amplitude_envelope,
        )
        stats.energy_gated = energy_count

    stats.output_count = len(cleaned)
    log.debug(
        "transcription cleanup: %d → %d "
        "(merged=%d, octaves=%d, ghosts=%d, energy_gated=%d)",
        stats.input_count,
        stats.output_count,
        stats.merged,
        stats.octave_ghosts_dropped,
        stats.ghost_tails_dropped,
        stats.energy_gated,
    )
    return cleaned, stats
