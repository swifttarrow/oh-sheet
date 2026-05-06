"""Tier 3 arrangement-quality metrics for the Phase 7 eval ladder.

Tier 3 asks whether the **sheet music itself** is a competent piano
arrangement. Every metric here operates directly on a ``PianoScore``
(the post-arrange artifact) — no audio, no reference. This is the
metric class that catches arrange-stage regressions where transcription
quality is fine but the arranger produced an unplayable or ugly score
(e.g. middle-C split forcing impossible bass spans, voice cap dropping
legitimate inner voices, percentile remap collapsing dynamics).

See ``docs/research/transcription-improvement-strategy.md`` Part III
§2.3 for the metric definitions and citations and §8.2 for the
composite weighting used by the production-Q score.

The five metrics:

* :func:`playability_fraction` — fraction of per-hand chord groupings
  with span ≤14 semitones AND ≤5 simultaneous notes. Re-exported from
  :mod:`eval.tier_rf` so Tier 3's public surface is self-contained.
* :func:`voice_leading_smoothness` — ``1 - mean_displacement / 12``.
  Mean displacement is the per-voice semitone change averaged across
  consecutive chord groupings within each hand. Lower displacement →
  smoother voicing → score nearer 1.0.
* :func:`polyphony_density` — notes-per-beat-per-hand (mean / p95 /
  max), beat-bucketed via floor on ``onset_beat``. Returns a
  fitness-against-target score using the strategy doc's 2.5 notes/beat
  target.
* :func:`engraving_heuristic_checks` — list of pure-data warnings
  (ledger-line excess, voice crossings within a hand, hand crossings
  between hands). Cheap to compute and a hard signal that the score
  would be confusing on the page even before MV2H / RubricNet.
* :func:`sight_readability_score` — composite ``[0, 1]`` heuristic
  combining the four above plus the engraving warning density. A
  RubricNet stand-in until the strategy doc's C3 model lands.

The :func:`compute_tier3` entry point runs all five for one score and
returns a :class:`Tier3Result`. The composite ``Tier3Result.composite``
matches strategy doc §8.2's ``tier3 = 0.5·play + 0.3·vleading + 0.2·density``
shape — that's what the production Q score consumes.

Heavy deps (``numpy``) are imported lazily inside the metric functions
so the module is import-cheap and easy to unit-test in isolation. Each
metric tolerates an empty or single-note score by returning a 0.0
score plus an entry in ``Tier3Result.notes`` rather than raising.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from eval.tier_rf import (
    PLAYABILITY_MAX_NOTES_PER_HAND,
    PLAYABILITY_MAX_SPAN_SEMITONES,
    playability_rf_score,
)

if TYPE_CHECKING:
    from shared.contracts import PianoScore, ScoreNote

log = logging.getLogger(__name__)

# Two notes share a chord grouping when their ``onset_beat`` values are
# within this tolerance. Mirrors :data:`eval.tier_rf._CHORD_GROUP_BEAT_EPS`
# so the playability + voice-leading groupings stay aligned.
_CHORD_GROUP_BEAT_EPS = 1e-6

# Voice-leading rescale factor: a 12-semitone (octave) per-voice jump
# rescales to 0.0 smoothness. Strategy doc §2.3 cites Lerdahl Tonal
# Pitch Space; an octave is the conventional "rough" upper bound for
# a single voice's per-chord move.
_VOICE_LEADING_NORMALIZER_SEMITONES = 12.0

# Polyphony fitness target: strategy doc §8.2 ``s_density =
# density_in_target_range(score, target=2.5)``. The density score is
# 1.0 at the target and decays to 0.0 at +/- ``_DENSITY_HALFWIDTH``.
_DENSITY_TARGET_NOTES_PER_BEAT = 2.5
_DENSITY_HALFWIDTH = 2.5  # → 0.0 at 0.0 notes/beat and at 5.0 notes/beat

# Engraving heuristic windows. Strategy doc §2.3 cites "ledger lines >3"
# as a warning class. With middle-C at MIDI 60 and 5 ledger lines under
# that = MIDI 41 (F2), 5 above = MIDI 81 (A5). Allow ±2 semitones of
# slack so a normal pop score doesn't trip every bar.
_LEDGER_LINE_RH_MIN = 53   # below this = excessive ledger lines under treble clef
_LEDGER_LINE_LH_MAX = 67   # above this = excessive ledger lines over bass clef
_LEDGER_LINE_RH_MAX = 96   # C7 — above this = excessive ledger lines over treble clef
_LEDGER_LINE_LH_MIN = 28   # E1 — below this = excessive ledger lines under bass clef


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class Tier3Result:
    """Per-score Tier 3 metrics produced by :func:`compute_tier3`.

    Each headline field is in ``[0, 1]``. Diagnostic counts (n_voices /
    n_warnings / etc.) are surfaced so the harness can sparkline them
    against history without re-deriving from the score.

    The composite :attr:`composite` matches strategy doc §8.2's
    ``tier3 = 0.5·play + 0.3·vleading + 0.2·density`` which is what the
    production-Q score consumes.
    """

    playability_fraction: float
    voice_leading_smoothness: float
    polyphony_mean: float
    polyphony_p95: float
    polyphony_max: float
    polyphony_in_target_range: float
    sight_readability: float
    n_playable_chords: int
    n_total_chords: int
    n_voice_pairs: int
    engraving_warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def composite(self) -> float:
        """Strategy doc §8.2 weighting: 0.5·play + 0.3·vleading + 0.2·density."""
        return (
            0.5 * self.playability_fraction
            + 0.3 * self.voice_leading_smoothness
            + 0.2 * self.polyphony_in_target_range
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "playability_fraction": round(self.playability_fraction, 4),
            "voice_leading_smoothness": round(self.voice_leading_smoothness, 4),
            "polyphony_mean": round(self.polyphony_mean, 4),
            "polyphony_p95": round(self.polyphony_p95, 4),
            "polyphony_max": round(self.polyphony_max, 4),
            "polyphony_in_target_range": round(self.polyphony_in_target_range, 4),
            "sight_readability": round(self.sight_readability, 4),
            "composite": round(self.composite, 4),
            "n_playable_chords": self.n_playable_chords,
            "n_total_chords": self.n_total_chords,
            "n_voice_pairs": self.n_voice_pairs,
            "n_engraving_warnings": len(self.engraving_warnings),
            "engraving_warnings": list(self.engraving_warnings),
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Playability — re-export from tier_rf
# ---------------------------------------------------------------------------

def playability_fraction(
    score: PianoScore,
    *,
    max_span_semitones: int = PLAYABILITY_MAX_SPAN_SEMITONES,
    max_notes_per_hand: int = PLAYABILITY_MAX_NOTES_PER_HAND,
) -> tuple[float, int, int]:
    """Identical to :func:`eval.tier_rf.playability_rf_score`.

    Re-exported here so the Tier 3 surface is self-contained — the
    harness reaches for ``tier3_arrangement.playability_fraction``
    without a cross-tier import. Returns ``(fraction, n_playable, n_total)``.
    """
    return playability_rf_score(
        score,
        max_span_semitones=max_span_semitones,
        max_notes_per_hand=max_notes_per_hand,
    )


# ---------------------------------------------------------------------------
# Voice-leading smoothness
# ---------------------------------------------------------------------------

def voice_leading_smoothness(
    score: PianoScore,
    *,
    normalizer_semitones: float = _VOICE_LEADING_NORMALIZER_SEMITONES,
) -> tuple[float, int]:
    """``1 - mean_voice_displacement / 12``, clamped to ``[0, 1]``.

    For each hand, group notes by ``onset_beat``. For each pair of
    consecutive groupings, match voices by ``ScoreNote.voice`` ID
    (when both groupings carry the same voice ID) and accumulate the
    absolute pitch delta. The mean across all matched voice-pairs
    is the displacement; the score is ``1 - mean_disp / normalizer``.

    A score of 1.0 means every voice held its pitch across consecutive
    chords; 0.0 means the average voice jumped by an octave or more.
    Strategy doc §2.3 cites a ≤2.5-semitone median displacement as
    "smooth pop voicing" — that maps to ``smoothness ≥ 0.79`` here.

    Returns ``(smoothness, n_voice_pairs)``. When there are fewer than
    two chord groupings in either hand, returns ``(0.0, 0)`` with a
    note in the caller-passed ``notes`` list — single-chord scores have
    no voice-leading to evaluate.
    """
    rh_disp, rh_pairs = _hand_voice_displacement(score.right_hand)
    lh_disp, lh_pairs = _hand_voice_displacement(score.left_hand)

    total_pairs = rh_pairs + lh_pairs
    if total_pairs == 0:
        return 0.0, 0

    mean_disp = (rh_disp + lh_disp) / total_pairs
    smoothness = 1.0 - (mean_disp / normalizer_semitones)
    return max(0.0, min(1.0, smoothness)), total_pairs


def _hand_voice_displacement(
    notes: list[ScoreNote],
) -> tuple[float, int]:
    """Sum of absolute pitch deltas + count of matched voice pairs.

    Helper for :func:`voice_leading_smoothness`. Returns
    ``(total_abs_displacement_semitones, n_matched_voice_pairs)``.
    """
    if len(notes) < 2:
        return 0.0, 0
    groups = _chord_groups(notes)
    if len(groups) < 2:
        return 0.0, 0

    total_disp = 0.0
    n_pairs = 0
    for prev, curr in zip(groups, groups[1:]):
        prev_by_voice = {n.voice: n.pitch for n in prev}
        curr_by_voice = {n.voice: n.pitch for n in curr}
        shared = prev_by_voice.keys() & curr_by_voice.keys()
        for v in shared:
            total_disp += abs(curr_by_voice[v] - prev_by_voice[v])
            n_pairs += 1
    return total_disp, n_pairs


# ---------------------------------------------------------------------------
# Polyphony density
# ---------------------------------------------------------------------------

def polyphony_density(
    score: PianoScore,
    *,
    target: float = _DENSITY_TARGET_NOTES_PER_BEAT,
    halfwidth: float = _DENSITY_HALFWIDTH,
) -> tuple[float, float, float, float, int]:
    """Beat-bucketed notes-per-beat statistics + a target-fitness score.

    Buckets every note's ``onset_beat`` by ``floor`` and counts notes
    per bucket per hand. Returns ``(mean, p95, max, in_target_range, n_buckets)``.

    The ``in_target_range`` score is a triangular fitness:

    * 1.0 when the mean equals ``target`` (default 2.5 notes/beat).
    * 0.0 when the mean is at ``target ± halfwidth`` (default 0.0
      notes/beat or 5.0 notes/beat).

    A linear decay rather than Gaussian because the strategy doc's
    weighting is itself a placeholder — keep it transparent and easy
    to recalibrate when Tier 5 lands. Strategy doc §8.2:
    ``s_density = density_in_target_range(score, target=2.5)``.

    A score with no notes returns ``(0.0, 0.0, 0.0, 0.0, 0)`` with a
    diagnostic note from the caller.
    """
    import numpy as np  # noqa: PLC0415

    rh_buckets = _per_beat_counts(score.right_hand)
    lh_buckets = _per_beat_counts(score.left_hand)
    if not rh_buckets and not lh_buckets:
        return 0.0, 0.0, 0.0, 0.0, 0

    # Concatenate per-hand counts: each hand contributes its own
    # sequence of beat-bucketed counts, so a sparse-LH / busy-RH song
    # gets the same diagnostic signal as a busy-LH / sparse-RH one.
    counts = np.array([*rh_buckets, *lh_buckets], dtype=float)
    mean = float(counts.mean())
    p95 = float(np.percentile(counts, 95))
    mx = float(counts.max())
    in_target = max(0.0, 1.0 - abs(mean - target) / halfwidth)
    return mean, p95, mx, in_target, int(counts.size)


def _per_beat_counts(notes: list[ScoreNote]) -> list[int]:
    """Notes per integer beat bucket — empty buckets dropped.

    Beat 0 holds onsets in ``[0.0, 1.0)``; beat 1 holds ``[1.0, 2.0)``;
    etc. Ignores notes with negative onset (defensive — shouldn't
    happen post-arrange but keeps the metric crash-free).
    """
    if not notes:
        return []
    counts: dict[int, int] = {}
    for n in notes:
        if n.onset_beat < 0:
            continue
        bucket = int(math.floor(n.onset_beat))
        counts[bucket] = counts.get(bucket, 0) + 1
    return list(counts.values())


# ---------------------------------------------------------------------------
# Engraving heuristic checks
# ---------------------------------------------------------------------------

def engraving_heuristic_checks(score: PianoScore) -> list[str]:
    """Cheap RF warnings for "this score will be hard to read".

    Three classes of warning, all derived from :class:`PianoScore`
    without needing MusicXML or a rendered score:

    * **Ledger-line excess** — RH notes below
      :data:`_LEDGER_LINE_RH_MIN` or above :data:`_LEDGER_LINE_RH_MAX`,
      LH notes below :data:`_LEDGER_LINE_LH_MIN` or above
      :data:`_LEDGER_LINE_LH_MAX`. Reports counts.
    * **Voice crossing within a hand** — voice 2 above voice 1 (or
      generally lower-numbered voice should be the higher-pitched one)
      at the same onset.
    * **Hand crossing** — LH max pitch above RH min pitch at the same
      onset, indicating the hands have crossed on the staff.

    Returns a list of human-readable warning strings (empty when the
    score is clean). Each warning includes the count + an example
    location to make spot-checking fast.
    """
    warnings: list[str] = []

    rh_excess = sum(
        1 for n in score.right_hand
        if n.pitch < _LEDGER_LINE_RH_MIN or n.pitch > _LEDGER_LINE_RH_MAX
    )
    lh_excess = sum(
        1 for n in score.left_hand
        if n.pitch < _LEDGER_LINE_LH_MIN or n.pitch > _LEDGER_LINE_LH_MAX
    )
    if rh_excess:
        warnings.append(
            f"ledger_excess_rh: {rh_excess} note(s) outside "
            f"[{_LEDGER_LINE_RH_MIN}, {_LEDGER_LINE_RH_MAX}]"
        )
    if lh_excess:
        warnings.append(
            f"ledger_excess_lh: {lh_excess} note(s) outside "
            f"[{_LEDGER_LINE_LH_MIN}, {_LEDGER_LINE_LH_MAX}]"
        )

    voice_cross = _voice_crossings_in_hand(score.right_hand)
    if voice_cross:
        warnings.append(f"voice_crossing_rh: {voice_cross} occurrence(s)")
    voice_cross_lh = _voice_crossings_in_hand(score.left_hand)
    if voice_cross_lh:
        warnings.append(f"voice_crossing_lh: {voice_cross_lh} occurrence(s)")

    hand_cross = _hand_crossings(score)
    if hand_cross:
        warnings.append(f"hand_crossing: {hand_cross} onset(s)")

    return warnings


def _voice_crossings_in_hand(notes: list[ScoreNote]) -> int:
    """Count onsets where a higher-numbered voice has a higher pitch.

    Convention: voice 1 = top voice (highest), voice 2 = next down,
    etc. Within one hand at a single onset, voice 2's pitch should
    be ``≤`` voice 1's pitch. A crossing means engraver will need
    voice swap markings to disambiguate.
    """
    if len(notes) < 2:
        return 0
    by_onset = _chord_groups(notes)
    crossings = 0
    for group in by_onset:
        # Group by voice, then check pairwise that lower voice number
        # has higher (or equal) pitch.
        by_voice: dict[int, list[int]] = {}
        for n in group:
            by_voice.setdefault(n.voice, []).append(n.pitch)
        voices_sorted = sorted(by_voice.keys())
        for i, v_high in enumerate(voices_sorted):
            for v_low in voices_sorted[i + 1:]:
                if max(by_voice[v_low]) > min(by_voice[v_high]):
                    crossings += 1
    return crossings


def _hand_crossings(score: PianoScore) -> int:
    """Onsets where LH max pitch > RH min pitch (hands have crossed)."""
    if not score.right_hand or not score.left_hand:
        return 0
    rh_groups = {round(g[0].onset_beat, 6): g for g in _chord_groups(score.right_hand)}
    lh_groups = {round(g[0].onset_beat, 6): g for g in _chord_groups(score.left_hand)}
    crossings = 0
    for beat, rh_group in rh_groups.items():
        lh_group = lh_groups.get(beat)
        if lh_group is None:
            continue
        rh_min = min(n.pitch for n in rh_group)
        lh_max = max(n.pitch for n in lh_group)
        if lh_max > rh_min:
            crossings += 1
    return crossings


# ---------------------------------------------------------------------------
# Sight-readability heuristic
# ---------------------------------------------------------------------------

def sight_readability_score(
    score: PianoScore,
    *,
    playability: float | None = None,
    voice_leading: float | None = None,
    density_fitness: float | None = None,
    engraving_warning_count: int | None = None,
) -> float:
    """Composite ``[0, 1]`` heuristic standing in for RubricNet.

    Weights playability (40%), voice-leading (25%), density-fitness
    (15%), and warning-density (20%). Warning density is computed as
    ``min(1, n_warnings / max(1, n_notes / 20))`` — one warning per
    20 notes saturates the penalty term — so engraver-frightening
    scores can still hit a non-zero readability if their note math
    is otherwise clean.

    Sub-scores are accepted as parameters so the caller (typically
    :func:`compute_tier3`) can avoid recomputing them. Defaults
    populate by calling each sub-metric on demand.

    Strategy doc §2.3 acknowledges this is a "RubricNet-style"
    placeholder until the C3 model lands — keep the formula
    transparent so it's easy to recalibrate.
    """
    if playability is None:
        playability, _, _ = playability_fraction(score)
    if voice_leading is None:
        voice_leading, _ = voice_leading_smoothness(score)
    if density_fitness is None:
        _, _, _, density_fitness, _ = polyphony_density(score)
    if engraving_warning_count is None:
        engraving_warning_count = len(engraving_heuristic_checks(score))

    n_notes = len(score.right_hand) + len(score.left_hand)
    if n_notes == 0:
        return 0.0
    warning_density = min(1.0, engraving_warning_count / max(1.0, n_notes / 20.0))
    warning_term = 1.0 - warning_density
    composite = (
        0.40 * playability
        + 0.25 * voice_leading
        + 0.15 * density_fitness
        + 0.20 * warning_term
    )
    return max(0.0, min(1.0, composite))


# ---------------------------------------------------------------------------
# Top-level entrypoint
# ---------------------------------------------------------------------------

def compute_tier3(score: PianoScore) -> Tier3Result:
    """Run all five Tier 3 metrics for one ``PianoScore``.

    Single-pass over the score — sub-metrics share grouping work
    where possible. Empty scores produce a zeroed result with a
    note explaining the absence of signal.
    """
    notes: list[str] = []

    play, n_play, n_total = playability_fraction(score)
    if n_total == 0:
        notes.append("playability_fraction: empty score")

    vleading, n_voice_pairs = voice_leading_smoothness(score)
    if n_voice_pairs == 0:
        notes.append("voice_leading_smoothness: <2 chord groupings in any hand")

    poly_mean, poly_p95, poly_max, poly_target, n_buckets = polyphony_density(score)
    if n_buckets == 0:
        notes.append("polyphony_density: no notes")

    warnings = engraving_heuristic_checks(score)

    sr = sight_readability_score(
        score,
        playability=play,
        voice_leading=vleading,
        density_fitness=poly_target,
        engraving_warning_count=len(warnings),
    )

    return Tier3Result(
        playability_fraction=play,
        voice_leading_smoothness=vleading,
        polyphony_mean=poly_mean,
        polyphony_p95=poly_p95,
        polyphony_max=poly_max,
        polyphony_in_target_range=poly_target,
        sight_readability=sr,
        n_playable_chords=n_play,
        n_total_chords=n_total,
        n_voice_pairs=n_voice_pairs,
        engraving_warnings=warnings,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Internal — chord-group builder shared with tier_rf.playability_rf_score
# ---------------------------------------------------------------------------

def _chord_groups(notes: list[ScoreNote]) -> list[list[ScoreNote]]:
    """Bucket notes by ``onset_beat`` (within :data:`_CHORD_GROUP_BEAT_EPS`).

    Local copy of :func:`eval.tier_rf._chord_groups`; duplicated so
    Tier 3 doesn't reach into Tier RF's private surface. The two
    must stay byte-identical — a divergence would silently produce
    different chord groupings between the playability metric (uses
    tier_rf's helper) and the voice-leading metric (uses this one).
    """
    if not notes:
        return []
    sorted_notes = sorted(notes, key=lambda n: n.onset_beat)
    groups: list[list[ScoreNote]] = [[sorted_notes[0]]]
    for n in sorted_notes[1:]:
        if abs(n.onset_beat - groups[-1][0].onset_beat) <= _CHORD_GROUP_BEAT_EPS:
            groups[-1].append(n)
        else:
            groups.append([n])
    return groups
