"""Cluster-and-Separate voice / staff assignment (Phase 9B).

Heuristic stand-in for Karystinaios & Widmer 2024 (`arXiv:2407.21030
<https://arxiv.org/html/2407.21030v1>`_) — clusters notes into streams
("voices") via greedy similarity matching, then separates streams into
right-hand / left-hand staves by pitch centroid. No PyTorch / PyG
dependency at writing; the same input/output API can later be served by
the real GNN once a labeled training set lands (strategy doc §C5).

The algorithm has three stages:

1. **Cluster.** Each note is appended to the open stream with the lowest
   transition cost (pitch jump + time gap + velocity drift). Streams
   start fresh when no open stream is close enough.
2. **Centroid.** Each stream's duration-weighted pitch mean defines a
   single number that summarises its register.
3. **Separate.** The split point is the median centroid, clamped to a
   piano-physiologically plausible window around middle C (default
   A3–F4). Streams above the split go RH; below go LH.

The output is shape-compatible with the legacy ``arrange._assign_hands``
return type so it drops in via a feature flag without touching anything
downstream of arrange.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

DEFAULT_PITCH_WEIGHT = 1.0
DEFAULT_TIME_WEIGHT = 4.0
DEFAULT_VELOCITY_WEIGHT = 0.05
DEFAULT_JOIN_THRESHOLD = 8.0
DEFAULT_MIN_SPLIT_HINT = 55           # A3
DEFAULT_MAX_SPLIT_HINT = 65           # F4
DEFAULT_SPLIT_FALLBACK = 60           # middle C


# Note tuple shape matches arrange.py's beat-domain representation:
# (pitch, onset_beat, duration_beat, velocity).
Note4 = tuple[int, float, float, int]


@dataclass(frozen=True)
class VoiceGNNConfig:
    pitch_weight: float = DEFAULT_PITCH_WEIGHT
    time_weight: float = DEFAULT_TIME_WEIGHT
    velocity_weight: float = DEFAULT_VELOCITY_WEIGHT
    join_threshold: float = DEFAULT_JOIN_THRESHOLD
    min_split_hint: int = DEFAULT_MIN_SPLIT_HINT
    max_split_hint: int = DEFAULT_MAX_SPLIT_HINT
    default_split: int = DEFAULT_SPLIT_FALLBACK


@dataclass
class VoiceGNNStats:
    """Telemetry from one hand-assignment pass."""
    n_notes: int = 0
    n_streams: int = 0
    split_pitch: int = DEFAULT_SPLIT_FALLBACK
    n_rh: int = 0
    n_lh: int = 0
    skipped: bool = False
    warnings: list[str] = field(default_factory=list)

    def as_warnings(self) -> list[str]:
        out: list[str] = []
        if self.skipped:
            out.extend(self.warnings)
            return out
        if self.n_notes:
            out.append(
                f"voice-gnn: {self.n_notes} notes → {self.n_streams} streams "
                f"(split={self.split_pitch}, rh={self.n_rh}, lh={self.n_lh})"
            )
        return out


def _stream_cost(
    last: Note4,
    candidate: Note4,
    *,
    cfg: VoiceGNNConfig,
) -> float:
    """Cost of appending ``candidate`` to a stream whose last note is ``last``.

    Lower is better. Captures three musical-stream cues:

    * **Pitch proximity.** Big leaps are unlikely within a single voice.
    * **Temporal continuity.** Long gaps mark a new stream.
    * **Velocity coherence.** Sudden dynamic jumps suggest a different line.

    ``time_gap`` is clamped at zero so overlapping notes (which arise
    when the transcriber emits a chord on a single track) don't get a
    spurious negative-cost bonus.
    """
    last_pitch, last_onset, last_dur, last_vel = last
    new_pitch, new_onset, _, new_vel = candidate
    pitch_d = abs(new_pitch - last_pitch)
    last_offset = last_onset + last_dur
    time_gap = max(new_onset - last_offset, 0.0)
    vel_d = abs(new_vel - last_vel)
    return (
        cfg.pitch_weight * pitch_d
        + cfg.time_weight * time_gap
        + cfg.velocity_weight * vel_d
    )


def _cluster_streams(
    notes: list[Note4],
    *,
    cfg: VoiceGNNConfig,
) -> list[list[Note4]]:
    """Greedy cluster notes into streams via the cost function.

    Notes are scanned in (onset, -pitch) order so simultaneously-onsetting
    notes go to streams in top-down pitch order — biases the higher voice
    of a chord toward the same stream as the previous high note.
    """
    if not notes:
        return []

    sorted_notes = sorted(notes, key=lambda n: (n[1], -n[0]))
    streams: list[list[Note4]] = []
    for note in sorted_notes:
        best_idx: int | None = None
        best_cost = cfg.join_threshold
        for idx, stream in enumerate(streams):
            cost = _stream_cost(stream[-1], note, cfg=cfg)
            if cost < best_cost:
                best_cost = cost
                best_idx = idx
        if best_idx is None:
            streams.append([note])
        else:
            streams[best_idx].append(note)
    return streams


def _stream_centroid(stream: list[Note4]) -> float:
    """Duration-weighted mean pitch of a stream.

    Weighting by duration so a long sustained low note pulls the
    centroid down even if the stream contains brief upper notes.
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for pitch, _, dur, _ in stream:
        w = max(dur, 1e-3)
        weighted_sum += pitch * w
        total_weight += w
    return weighted_sum / total_weight if total_weight else float(DEFAULT_SPLIT_FALLBACK)


def _choose_split(centroids: list[float], cfg: VoiceGNNConfig) -> int:
    """Pick a split pitch from the median of stream centroids.

    Falls back to ``cfg.default_split`` when there's only one stream
    (every note in the same hand on a monophonic line). Clamped to
    ``[min_split_hint, max_split_hint]`` so an outlier-heavy piece can't
    push the split into a register that would never be played by either
    hand alone.
    """
    if not centroids or len(centroids) == 1:
        return cfg.default_split
    sorted_c = sorted(centroids)
    n = len(sorted_c)
    median = (
        (sorted_c[n // 2 - 1] + sorted_c[n // 2]) / 2.0
        if n % 2 == 0
        else sorted_c[n // 2]
    )
    return int(round(max(cfg.min_split_hint, min(cfg.max_split_hint, median))))


def assign_hands_gnn(
    notes: list[Note4],
    *,
    config: VoiceGNNConfig | None = None,
) -> tuple[list[Note4], list[Note4], VoiceGNNStats]:
    """Cluster-and-Separate hand assignment over a flat note list.

    Drop-in replacement for the inner ``pitch >= SPLIT_PITCH`` decision
    in :func:`backend.services.arrange._assign_hands`. The caller is
    responsible for preserving any explicit ``InstrumentRole`` routing
    (MELODY → RH, BASS → LH); this function makes no assumption about
    where the notes came from.

    Returns
    -------
    A tuple ``(rh_notes, lh_notes, stats)``. Each hand list is sorted by
    ``(onset_beat, -pitch)`` so downstream voice assignment in
    :func:`backend.services.arrange._resolve_overlaps` sees a stable
    order.
    """
    cfg = config or VoiceGNNConfig()
    stats = VoiceGNNStats(n_notes=len(notes))

    if not notes:
        stats.skipped = True
        return [], [], stats

    streams = _cluster_streams(notes, cfg=cfg)
    stats.n_streams = len(streams)

    centroids = [_stream_centroid(s) for s in streams]
    split_pitch = _choose_split(centroids, cfg)
    stats.split_pitch = split_pitch

    rh: list[Note4] = []
    lh: list[Note4] = []
    for stream, centroid in zip(streams, centroids):
        if centroid >= split_pitch:
            rh.extend(stream)
        else:
            lh.extend(stream)

    rh.sort(key=lambda n: (n[1], -n[0]))
    lh.sort(key=lambda n: (n[1], -n[0]))

    stats.n_rh = len(rh)
    stats.n_lh = len(lh)
    return rh, lh, stats
