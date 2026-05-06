"""Score-HPT-style per-note velocity refinement (Phase 9A).

Heuristic stand-in for Foscarin et al.'s Score-HPT (Hierarchical
Performance Transformer; paper-only at writing). The real model is a
~1M-param BiLSTM/Transformer head that re-estimates note velocities
from score / structural features. Until that lands, this module
implements a deterministic, dependency-free refiner that captures the
same intent: combine the transcriber's raw velocity with metric-position,
register, and onset-density features so downstream arrangement /
humanization see musical numbers instead of Basic Pitch's flat output.

Operates on seconds-domain :class:`backend.contracts.Note` objects so
the arrange stage's existing :func:`arrange._normalize_velocity`
percentile-band remap continues to run unchanged on top. Pure function
— no I/O, no global state — so it can be unit-tested without fixtures.

The seam (transcribe → arrange) is wired in
:class:`backend.services.transcribe.TranscribeService.run` and gated by
``settings.score_hpt_enabled``. Default off until validated against the
mini-eval / Phase 3 pop eval set.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from backend.contracts import (
    MidiTrack,
    Note,
    TempoMapEntry,
    sec_to_beat,
)

log = logging.getLogger(__name__)

DEFAULT_BLEND_ALPHA = 0.5
DEFAULT_DOWNBEAT_BOOST = 8.0
DEFAULT_BEAT_BOOST = 4.0
DEFAULT_OFFBEAT_ATTENUATION = 4.0
DEFAULT_REGISTER_CURVE_STRENGTH = 0.3
DEFAULT_DENSITY_COMPENSATION = 6.0
DEFAULT_MIN_VELOCITY = 20
DEFAULT_MAX_VELOCITY = 120
DEFAULT_BEAT_TOLERANCE = 0.08
DEFAULT_DENSITY_WINDOW = 0.25
DEFAULT_DENSITY_COUNT_HIGH = 4


@dataclass(frozen=True)
class ScoreHPTConfig:
    """Tunable knobs for the velocity refiner.

    All fields default to the module-level constants so the bare
    ``ScoreHPTConfig()`` matches what the wired-up service uses.
    """
    blend_alpha: float = DEFAULT_BLEND_ALPHA
    downbeat_boost: float = DEFAULT_DOWNBEAT_BOOST
    beat_boost: float = DEFAULT_BEAT_BOOST
    offbeat_attenuation: float = DEFAULT_OFFBEAT_ATTENUATION
    register_curve_strength: float = DEFAULT_REGISTER_CURVE_STRENGTH
    density_compensation: float = DEFAULT_DENSITY_COMPENSATION
    min_velocity: int = DEFAULT_MIN_VELOCITY
    max_velocity: int = DEFAULT_MAX_VELOCITY
    beat_tolerance: float = DEFAULT_BEAT_TOLERANCE
    density_window: float = DEFAULT_DENSITY_WINDOW
    density_count_high: int = DEFAULT_DENSITY_COUNT_HIGH


@dataclass
class ScoreHPTStats:
    """Telemetry for one velocity-refinement pass."""
    n_notes: int = 0
    n_changed: int = 0
    mean_abs_delta: float = 0.0
    max_abs_delta: int = 0
    skipped: bool = False
    warnings: list[str] = field(default_factory=list)

    def as_warnings(self) -> list[str]:
        out: list[str] = []
        if self.skipped:
            out.extend(self.warnings)
            return out
        if self.n_changed:
            out.append(
                f"score-hpt: refined {self.n_changed}/{self.n_notes} velocities "
                f"(mean_abs_delta={self.mean_abs_delta:.1f}, "
                f"max_abs_delta={self.max_abs_delta})"
            )
        return out


def _register_curve(pitch: int, strength: float) -> float:
    """Bell-shaped attenuation peaking at C4 (MIDI 60).

    Real piano performance tends to use less force at the extreme bass
    (sub-contra register) and very high treble. ``strength=0`` disables;
    ``strength=1`` reaches roughly -10 velocity points at the keyboard
    extremes (24 semitones from middle C).
    """
    if strength <= 0.0:
        return 0.0
    dist = abs(pitch - 60)
    falloff = min(dist / 24.0, 1.0)
    return -strength * 10.0 * falloff * falloff


def _is_downbeat(beat: float, downbeats: list[float], tolerance: float) -> bool:
    if not downbeats:
        return False
    nearest = min(downbeats, key=lambda d: abs(d - beat))
    return abs(nearest - beat) <= tolerance


def _is_on_beat(beat: float, tolerance: float) -> bool:
    """True when the onset lies within ``tolerance`` of an integer beat."""
    return abs(beat - round(beat)) <= tolerance


def _local_density(
    onset: float,
    onset_beats_sorted: list[float],
    *,
    window: float,
) -> int:
    """Count notes (including self) onsetting within ``window`` beats.

    Uses bisect for an O(log n + k) scan against the sorted onset list
    rather than O(n) per-note. ``window`` is symmetric around ``onset``.
    """
    if not onset_beats_sorted:
        return 0
    import bisect  # noqa: PLC0415 — stdlib lazy import keeps cold start light
    lo = bisect.bisect_left(onset_beats_sorted, onset - window)
    hi = bisect.bisect_right(onset_beats_sorted, onset + window)
    return hi - lo


def refine_velocities(
    tracks: list[MidiTrack],
    tempo_map: list[TempoMapEntry],
    *,
    downbeats_sec: list[float] | None = None,
    config: ScoreHPTConfig | None = None,
) -> tuple[list[MidiTrack], ScoreHPTStats]:
    """Re-estimate per-note velocities from metric / register / density.

    Parameters
    ----------
    tracks:
        Seconds-domain ``MidiTrack`` list from ``TranscriptionResult``.
    tempo_map:
        Used to convert seconds onsets into beats for the metric features.
    downbeats_sec:
        Optional downbeat times in seconds (from
        :attr:`HarmonicAnalysis.downbeats`). When absent, only the
        beat / off-beat distinction is used.
    config:
        Tunable thresholds; see :class:`ScoreHPTConfig`.

    Returns
    -------
    A tuple ``(refined_tracks, stats)`` where ``refined_tracks`` is a
    fresh list of ``MidiTrack`` with updated ``Note.velocity`` values
    (the input is not mutated). Falls through with the original tracks
    plus a populated ``stats.skipped`` flag when there's nothing to
    refine.

    The output schema matches the input exactly — every other field on
    ``Note`` (``pitch``, ``onset_sec``, ``offset_sec``, ``pitch_bend_cents``)
    survives intact, so the swap is safe to drop in front of any
    velocity-aware downstream stage.
    """
    cfg = config or ScoreHPTConfig()
    stats = ScoreHPTStats()

    if not tracks:
        stats.skipped = True
        stats.warnings.append("score-hpt: no tracks; skipped")
        return tracks, stats

    if not tempo_map:
        stats.skipped = True
        stats.warnings.append("score-hpt: empty tempo_map; skipped")
        return tracks, stats

    n_total = sum(len(t.notes) for t in tracks)
    if n_total == 0:
        stats.skipped = True
        stats.warnings.append("score-hpt: no notes; skipped")
        return tracks, stats

    all_onset_beats: list[float] = []
    for track in tracks:
        for note in track.notes:
            all_onset_beats.append(sec_to_beat(note.onset_sec, tempo_map))
    all_onset_beats.sort()

    downbeats_in_beats = [
        sec_to_beat(t, tempo_map) for t in (downbeats_sec or [])
    ]

    deltas: list[int] = []
    refined_tracks: list[MidiTrack] = []
    for track in tracks:
        new_notes: list[Note] = []
        for note in track.notes:
            beat = sec_to_beat(note.onset_sec, tempo_map)
            metric_adjust = 0.0

            if _is_downbeat(beat, downbeats_in_beats, cfg.beat_tolerance):
                metric_adjust += cfg.downbeat_boost
            elif _is_on_beat(beat, cfg.beat_tolerance):
                metric_adjust += cfg.beat_boost
            else:
                metric_adjust -= cfg.offbeat_attenuation

            register_adjust = _register_curve(
                note.pitch, cfg.register_curve_strength,
            )

            density = _local_density(
                beat, all_onset_beats, window=cfg.density_window,
            )
            if density >= cfg.density_count_high:
                density_adjust = -cfg.density_compensation
            elif density == 1:
                density_adjust = +cfg.density_compensation * 0.5
            else:
                density_adjust = 0.0

            predicted = (
                note.velocity
                + metric_adjust
                + register_adjust
                + density_adjust
            )
            blended = (
                (1.0 - cfg.blend_alpha) * note.velocity
                + cfg.blend_alpha * predicted
            )
            new_velocity = max(
                cfg.min_velocity,
                min(cfg.max_velocity, int(round(blended))),
            )

            if new_velocity != note.velocity:
                deltas.append(new_velocity - note.velocity)

            new_notes.append(note.model_copy(update={"velocity": new_velocity}))

        refined_tracks.append(track.model_copy(update={"notes": new_notes}))

    stats.n_notes = n_total
    stats.n_changed = len(deltas)
    if deltas:
        abs_deltas = [abs(d) for d in deltas]
        stats.mean_abs_delta = sum(abs_deltas) / len(abs_deltas)
        stats.max_abs_delta = max(abs_deltas)

    return refined_tracks, stats
