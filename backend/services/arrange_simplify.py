"""Post-arrangement simplification pass.

Reduces note density in a PianoScore so the engraved notation is
readable as sheet music rather than a raw audio transcription. This is
the gap between "all the notes Basic Pitch detected" and "what a human
would write on a staff to play the song."

Operations applied per hand, in order:

1. **Velocity threshold** — drop notes below a minimum velocity. Kills
   background overtones, quiet artifacts, and bleed from other instruments.
2. **Duration snapping** — round each note's duration to the nearest
   standard musical value (16th, 8th, quarter, half, whole) so rhythms
   read cleanly instead of having weird fractional durations.
3. **Micro-note pruning** — after snapping, drop any note shorter than
   the configured minimum (default: 16th note).
4. **Chord cluster merging** — notes whose onsets fall within a small
   beat window are treated as a single chord and share the earliest
   onset in the cluster. Same-pitch duplicates are collapsed to the
   loudest copy. This turns "arpeggio-like onset smearing from audio
   transcription" into proper block chords.
5. **Density cap** — cap the number of distinct onset times per beat.
   When more than ``max_onsets_per_beat`` compete for the same beat,
   keep only the loudest onsets (ranked by max velocity in their group).

Design notes:

- Works on the ``PianoScore`` contract, not raw notes, so it runs
  **after** ``_arrange_sync()``'s hand-split and voice assignment.
- Does **not** touch ``chord_symbols``, ``sections``, or ``tempo_map`` —
  only the ``right_hand`` and ``left_hand`` note lists.
- All thresholds are parameters on ``simplify_score`` so tests can
  tune them directly; the arrange task wires production defaults from
  ``backend.config.settings``.
"""
from __future__ import annotations

import logging

from shared.contracts import PianoScore, ScoreNote

log = logging.getLogger(__name__)

# Standard note durations in beats (quarter-note = 1.0).
# 16th, 8th, quarter, half, whole.
_DURATION_GRID: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)


def simplify_score(
    score: PianoScore,
    *,
    min_velocity: int = 55,
    chord_merge_beats: float = 0.125,
    max_onsets_per_beat: int = 4,
    min_duration_beats: float = 0.25,
) -> PianoScore:
    """Return a simplified copy of ``score`` with fewer, cleaner notes.

    See module docstring for the operation order. Metadata is passed
    through unchanged.
    """
    right = _simplify_hand(
        score.right_hand,
        min_velocity=min_velocity,
        chord_merge_beats=chord_merge_beats,
        max_onsets_per_beat=max_onsets_per_beat,
        min_duration_beats=min_duration_beats,
    )
    left = _simplify_hand(
        score.left_hand,
        min_velocity=min_velocity,
        chord_merge_beats=chord_merge_beats,
        max_onsets_per_beat=max_onsets_per_beat,
        min_duration_beats=min_duration_beats,
    )

    log.info(
        "simplify: RH %d→%d, LH %d→%d",
        len(score.right_hand), len(right),
        len(score.left_hand), len(left),
    )

    return score.model_copy(update={"right_hand": right, "left_hand": left})


def _simplify_hand(
    notes: list[ScoreNote],
    *,
    min_velocity: int,
    chord_merge_beats: float,
    max_onsets_per_beat: int,
    min_duration_beats: float,
) -> list[ScoreNote]:
    """Apply the 5-step simplification pipeline to one hand's notes."""
    if not notes:
        return []

    # Step 1: velocity filter
    kept = [n for n in notes if n.velocity >= min_velocity]

    # Step 2: duration snap to standard grid
    snapped = [
        n.model_copy(update={"duration_beat": _snap_duration(n.duration_beat)})
        for n in kept
    ]

    # Step 3: drop anything still shorter than min_duration_beats after snapping
    long_enough = [n for n in snapped if n.duration_beat >= min_duration_beats]

    # Step 4: cluster near-simultaneous onsets into chords
    merged = _merge_chord_clusters(long_enough, chord_merge_beats)

    # Step 5: cap onset density per beat, keeping loudest
    capped = _cap_density(merged, max_onsets_per_beat)

    return sorted(capped, key=lambda n: (n.onset_beat, n.pitch))


def _snap_duration(duration_beat: float) -> float:
    """Round duration to the nearest value in ``_DURATION_GRID``."""
    return min(_DURATION_GRID, key=lambda g: abs(g - duration_beat))


def _merge_chord_clusters(
    notes: list[ScoreNote],
    window: float,
) -> list[ScoreNote]:
    """Group notes whose onsets lie within ``window`` beats of each other.

    All notes in a cluster are re-stamped to the earliest onset in the
    cluster so they render as a block chord. When the same pitch appears
    twice in a cluster, only the loudest copy survives.
    """
    if not notes:
        return []

    sorted_notes = sorted(notes, key=lambda n: n.onset_beat)
    result: list[ScoreNote] = []
    cluster: list[ScoreNote] = [sorted_notes[0]]
    cluster_onset = sorted_notes[0].onset_beat

    def flush() -> None:
        # Deduplicate by pitch, keeping the loudest instance.
        by_pitch: dict[int, ScoreNote] = {}
        for c in cluster:
            existing = by_pitch.get(c.pitch)
            if existing is None or c.velocity > existing.velocity:
                by_pitch[c.pitch] = c
        # Re-stamp every surviving note to the cluster's earliest onset.
        for n in by_pitch.values():
            result.append(n.model_copy(update={"onset_beat": cluster_onset}))

    for note in sorted_notes[1:]:
        if note.onset_beat - cluster_onset < window:
            cluster.append(note)
        else:
            flush()
            cluster = [note]
            cluster_onset = note.onset_beat
    flush()

    return result


def _cap_density(
    notes: list[ScoreNote],
    max_per_beat: int,
) -> list[ScoreNote]:
    """Limit the number of distinct onset times per beat.

    Groups notes by ``floor(onset_beat)``. Inside each beat bucket, groups
    further by exact onset time. If more than ``max_per_beat`` distinct
    onsets compete for that beat, ranks onset groups by their max velocity
    and keeps only the top ``max_per_beat`` groups (all notes in a kept
    onset group survive together).
    """
    if not notes or max_per_beat <= 0:
        return list(notes)

    # Bucket notes by integer beat index.
    by_beat: dict[int, list[ScoreNote]] = {}
    for n in notes:
        beat_bucket = int(n.onset_beat)
        by_beat.setdefault(beat_bucket, []).append(n)

    kept: list[ScoreNote] = []
    for beat_notes in by_beat.values():
        # Sub-group by exact onset time within the beat.
        by_onset: dict[float, list[ScoreNote]] = {}
        for n in beat_notes:
            by_onset.setdefault(round(n.onset_beat, 4), []).append(n)

        if len(by_onset) <= max_per_beat:
            kept.extend(beat_notes)
            continue

        # Too many distinct onsets — rank by max velocity per onset group,
        # keep the top N groups.
        ranked = sorted(
            by_onset.items(),
            key=lambda kv: -max(n.velocity for n in kv[1]),
        )
        surviving_onsets = {onset for onset, _ in ranked[:max_per_beat]}
        for onset_time, onset_notes in by_onset.items():
            if onset_time in surviving_onsets:
                kept.extend(onset_notes)

    return kept
