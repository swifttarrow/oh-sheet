"""Unified eval orchestration — shared by ``scripts/eval.py`` Click CLI.

The Phase 0 ``scripts/eval_mini.py`` already runs the in-process
pipeline (transcribe → arrange → humanize → midi_render) and computes
Tier RF metrics. Phase 7 widens the metric surface to Tier 2 / 3 / 4
and adds CI-gate checks. Rather than duplicate the pipeline-driver
logic, this module reuses ``eval_mini`` for the pipeline drive and
layers all five tiers + composite-Q on top.

Public surface
--------------

* :class:`TierSelection` — declarative bag of "which tiers to run".
* :func:`run_one_song` — score one song against the requested tiers.
* :func:`run_eval_set` — orchestrate :func:`run_one_song` over a
  manifest, aggregate, write JSON, return the payload.
* :func:`apply_ci_gates` — compare a payload against a baseline JSON
  and produce a ``GateReport`` with pass/fail per gate.

The Click CLI in ``scripts/eval.py`` wraps these with subcommand-
specific defaults: ``ci`` runs Tier 2 + Tier 3 on a 5-song subset
and applies gates; ``nightly`` runs everything + composite-Q; the
diagnostic subcommands (``arrange``, ``engrave``, ``round-trip``)
narrow the tier set.

Heavy ML deps (CLAP / MERT) and audio deps (librosa / fluidsynth) are
imported lazily inside the metric modules so this orchestrator is
import-cheap.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from shared.contracts import PianoScore

REPO_ROOT = Path(__file__).resolve().parent.parent

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier selection + result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TierSelection:
    """Which tiers to compute on each song.

    Mirrors the strategy doc §4.2 subcommand split: ``ci`` runs a
    cheap subset, ``nightly`` flips on the heavy perceptual metrics.
    Bool flags rather than a ``set[str]`` so type checkers + IDEs
    show the supported tiers without a runtime lookup.
    """

    tier_rf: bool = True       # Phase 0 metrics — chord_rf / playability_rf / chroma_rf
    tier2: bool = True         # structural fidelity — key / tempo / beat / chord / section
    tier3: bool = True         # arrangement quality — playability + vleading + density + readability
    tier4: bool = False        # perceptual — chroma + round-trip + (optionally) CLAP / MERT
    tier4_round_trip: bool = False  # explicit toggle for the expensive round-trip transcribe pass
    tier4_clap: bool = False
    tier4_mert: bool = False
    composite_q: bool = False  # §8.2 composite quality score

    @classmethod
    def ci(cls) -> TierSelection:
        """Cheap PR-gate selection: Tier RF + Tier 2 + Tier 3."""
        return cls(tier_rf=True, tier2=True, tier3=True, tier4=False, composite_q=True)

    @classmethod
    def nightly(cls) -> TierSelection:
        """Full corpus selection: all tiers + composite Q. CLAP/MERT off by default — flip when deps installed."""
        return cls(
            tier_rf=True, tier2=True, tier3=True, tier4=True,
            tier4_round_trip=True, tier4_clap=False, tier4_mert=False,
            composite_q=True,
        )

    @classmethod
    def end_to_end(cls) -> TierSelection:
        """All tiers, callable from local dev runs. Same as nightly minus heavy deps."""
        return cls(
            tier_rf=True, tier2=True, tier3=True, tier4=True,
            tier4_round_trip=True, composite_q=True,
        )

    @classmethod
    def round_trip_only(cls) -> TierSelection:
        """Just the round-trip self-consistency probe."""
        return cls(
            tier_rf=False, tier2=False, tier3=False, tier4=True,
            tier4_round_trip=True, composite_q=False,
        )

    @classmethod
    def arrange_only(cls) -> TierSelection:
        """Tier 3 only — diagnose arrange-stage regressions in isolation."""
        return cls(
            tier_rf=False, tier2=False, tier3=True, tier4=False, composite_q=False,
        )


@dataclass
class SongScore:
    """One song's per-tier scores plus diagnostics."""

    slug: str
    title: str | None = None
    artist: str | None = None
    genre: str | None = None
    audio_path: str | None = None
    key_label: str | None = None
    wall_sec: float = 0.0
    error: str | None = None

    tier_rf: dict[str, Any] | None = None
    tier2: dict[str, Any] | None = None
    tier3: dict[str, Any] | None = None
    tier4: dict[str, Any] | None = None
    composite_q: float | None = None

    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "slug": self.slug,
            "title": self.title,
            "artist": self.artist,
            "genre": self.genre,
            "audio_path": self.audio_path,
            "key_label": self.key_label,
            "wall_sec": round(self.wall_sec, 2),
            "error": self.error,
            "notes": list(self.notes),
        }
        if self.tier_rf is not None:
            out["tier_rf"] = self.tier_rf
        if self.tier2 is not None:
            out["tier2"] = self.tier2
        if self.tier3 is not None:
            out["tier3"] = self.tier3
        if self.tier4 is not None:
            out["tier4"] = self.tier4
        if self.composite_q is not None:
            out["composite_q"] = round(self.composite_q, 4)
        return out


# ---------------------------------------------------------------------------
# Per-song orchestration
# ---------------------------------------------------------------------------

def run_one_song(
    *,
    slug: str,
    audio_path: Path,
    target_duration_sec: float,
    chord_recognition_key: str,
    tiers: TierSelection,
    transcribe_callable: Any | None = None,
    title: str | None = None,
    artist: str | None = None,
    genre: str | None = None,
) -> SongScore:
    """Score one song against the requested tier selection.

    Drives ``scripts.eval_mini._run_pipeline`` for the actual transcribe →
    arrange → humanize → midi-render run, then layers Tier 2/3/4 on
    top of the score + engraved MIDI it produces. Returns a
    :class:`SongScore`; per-tier failures populate ``notes`` rather
    than raising — Phase 0's contract.
    """
    row = SongScore(slug=slug, title=title, artist=artist, genre=genre)
    row.audio_path = (
        str(audio_path.relative_to(REPO_ROOT))
        if audio_path.is_relative_to(REPO_ROOT)
        else str(audio_path)
    )

    t0 = time.perf_counter()
    try:
        from scripts.eval_mini import _run_pipeline  # noqa: PLC0415
        artifacts = _run_pipeline(audio_path)
        row.key_label = artifacts.key_label
        score = artifacts.score
        midi_bytes = artifacts.midi_bytes
        key_for_recog = (
            artifacts.key_label
            if chord_recognition_key == "auto"
            else chord_recognition_key
        )

        if tiers.tier_rf:
            row.tier_rf = _compute_tier_rf(audio_path, score, midi_bytes, key_for_recog)

        if tiers.tier2:
            row.tier2 = _compute_tier2(audio_path, midi_bytes, key_for_recog)

        if tiers.tier3:
            row.tier3 = _compute_tier3(score)

        if tiers.tier4:
            row.tier4 = _compute_tier4(
                audio_path, midi_bytes,
                transcribe_callable=transcribe_callable if tiers.tier4_round_trip else None,
                enable_clap=tiers.tier4_clap,
                enable_mert=tiers.tier4_mert,
            )

        if tiers.composite_q:
            row.composite_q = _compute_composite_q(row)

    except Exception as exc:  # noqa: BLE001 — one bad song must not sink the run
        log.exception("per-song eval failed slug=%s", slug)
        row.error = f"{type(exc).__name__}: {exc}"

    row.wall_sec = time.perf_counter() - t0
    return row


def _compute_tier_rf(
    audio_path: Path, score: PianoScore, midi_bytes: bytes, key_label: str,
) -> dict[str, Any]:
    from eval.tier_rf import compute_tier_rf  # noqa: PLC0415
    result = compute_tier_rf(audio_path, score, midi_bytes, key_label=key_label)
    return result.as_dict()


def _compute_tier2(
    audio_path: Path, midi_bytes: bytes, key_label: str,
) -> dict[str, Any]:
    from eval.tier2_structural import compute_tier2  # noqa: PLC0415
    result = compute_tier2(audio_path, midi_bytes, key_label=key_label)
    return result.as_dict()


def _compute_tier3(score: PianoScore) -> dict[str, Any]:
    from eval.tier3_arrangement import compute_tier3  # noqa: PLC0415
    result = compute_tier3(score)
    return result.as_dict()


def _compute_tier4(
    audio_path: Path,
    midi_bytes: bytes,
    *,
    transcribe_callable: Any | None,
    enable_clap: bool,
    enable_mert: bool,
) -> dict[str, Any]:
    from eval.tier4_perceptual import compute_tier4  # noqa: PLC0415
    result = compute_tier4(
        audio_path,
        midi_bytes,
        transcribe_callable=transcribe_callable,
        enable_clap=enable_clap,
        enable_mert=enable_mert,
    )
    return result.as_dict()


def _compute_composite_q(row: SongScore) -> float | None:
    """Composite Q per strategy doc §8.2 — drops missing tiers and re-averages.

    Strategy doc §8.2 weighting: ``Q = 0.30·tier2 + 0.30·tier3 + 0.40·tier4``.
    When a tier is missing (e.g. CI run with ``tier4=False``), drop its
    weight from the denominator and re-normalize over present terms.
    Returns ``None`` when no tier ran (degenerate case caught in
    aggregation).
    """
    parts: list[tuple[float, float]] = []  # (weight, value)
    if row.tier2 is not None:
        mean = row.tier2.get("mean_score")
        if isinstance(mean, (int, float)):
            parts.append((0.30, float(mean)))
    if row.tier3 is not None:
        composite = row.tier3.get("composite")
        if isinstance(composite, (int, float)):
            parts.append((0.30, float(composite)))
    if row.tier4 is not None:
        composite = row.tier4.get("composite")
        if isinstance(composite, (int, float)):
            parts.append((0.40, float(composite)))
    if not parts:
        return None
    total_weight = sum(w for w, _ in parts)
    if total_weight <= 0:
        return None
    return sum(w * v for w, v in parts) / total_weight


# ---------------------------------------------------------------------------
# Eval-set orchestration
# ---------------------------------------------------------------------------

def run_eval_set(
    *,
    eval_set_path: Path,
    output_dir: Path,
    tiers: TierSelection,
    only_slug: str | None = None,
    only_first_n: int | None = None,
    transcribe_callable: Any | None = None,
    is_ci: bool = False,
    is_nightly: bool = False,
    label: str | None = None,
) -> dict[str, Any]:
    """Orchestrate :func:`run_one_song` over every song in a manifest.

    Writes ``aggregate.json`` to ``output_dir`` and returns the same
    payload in-memory. ``only_slug`` and ``only_first_n`` are mutually
    compatible: slug-filter first, then take head.
    """
    from scripts.eval_mini import (  # noqa: PLC0415
        _load_manifest,
        _print_table,
        _resolve_input_audio,
    )

    eval_set_path = Path(eval_set_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(eval_set_path)
    songs = manifest["songs"]
    n_total = len(songs)
    if only_slug is not None:
        songs = [s for s in songs if s.get("slug") == only_slug]
        if not songs:
            raise ValueError(f"no song with slug={only_slug!r} in manifest")
    if only_first_n is not None:
        songs = songs[: only_first_n]

    target_duration = float(manifest.get("target_duration_sec", 30.0))
    chord_key = str(manifest.get("chord_recognition_key", "auto"))

    print("=== eval ===")
    print(f"  manifest:   {eval_set_path / 'manifest.yaml'}")
    print(f"  songs:      {len(songs)} of {n_total}")
    print(f"  duration:   {target_duration:.1f}s per song")
    print(f"  tiers:      {_describe_tiers(tiers)}")
    if label is not None:
        print(f"  label:      {label}")
    print()

    rows: list[SongScore] = []
    for song in songs:
        slug = song.get("slug", "<no-slug>")
        print(f"  [{slug}] running…", flush=True)
        try:
            audio_path = _resolve_input_audio(song, eval_set_path, target_duration)
        except Exception as exc:  # noqa: BLE001
            row = SongScore(slug=slug)
            row.error = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            print(f"    ! resolve error: {row.error}")
            continue

        row = run_one_song(
            slug=slug,
            audio_path=audio_path,
            target_duration_sec=target_duration,
            chord_recognition_key=chord_key,
            tiers=tiers,
            transcribe_callable=transcribe_callable,
            title=song.get("title"),
            artist=song.get("artist"),
            genre=song.get("genre"),
        )
        rows.append(row)
        if row.error:
            print(f"    ! error: {row.error}")
        else:
            print(_format_song_summary(row, tiers))

    agg = aggregate_rows(rows, tiers)

    payload = {
        "schema_version": 2,
        "eval_set": manifest.get("eval_set", "unknown"),
        "manifest_relpath": _safe_relpath(eval_set_path / "manifest.yaml"),
        "run_id": dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ"),
        "git_sha": _git_sha(),
        "label": label,
        "is_ci": is_ci,
        "is_nightly": is_nightly,
        "tiers": _tiers_as_dict(tiers),
        "config": {
            "target_duration_sec": target_duration,
            "chord_recognition_key": chord_key,
            "n_songs_filtered": len(songs),
            "n_songs_total_in_manifest": n_total,
        },
        "songs": [r.as_dict() for r in rows],
        "aggregate": agg,
    }

    out_path = output_dir / "aggregate.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    # Reuse the Phase 0 6-line table for the legacy chord_rf / playability_rf /
    # chroma_rf surface; the Phase 7 broader surface comes through in JSON.
    if tiers.tier_rf:
        legacy_rows = _to_legacy_song_rows(rows)
        legacy_agg = _to_legacy_aggregate(rows)
        _print_table(legacy_rows, legacy_agg)
    print(f"\nWrote {out_path}")
    return payload


def aggregate_rows(rows: list[SongScore], tiers: TierSelection) -> dict[str, Any]:
    """Per-tier means + medians across all successfully scored songs.

    Failures (``error is not None``) are excluded from the aggregate
    means so a single FluidSynth error doesn't drag the mean to 0. The
    payload still records ``n_songs_errored`` so trend dashboards can
    track failure rates.
    """
    ok = [r for r in rows if r.error is None]
    agg: dict[str, Any] = {
        "n_songs_total": len(rows),
        "n_songs_scored": len(ok),
        "n_songs_errored": len(rows) - len(ok),
    }
    if not ok:
        return agg

    if tiers.tier_rf:
        agg.update(_aggregate_dict_keys(
            ok, "tier_rf",
            ("chord_rf", "playability_rf", "chroma_rf"),
        ))
    if tiers.tier2:
        agg.update(_aggregate_dict_keys(
            ok, "tier2",
            ("key_score", "tempo_score", "beat_score", "chord_score", "mean_score"),
        ))
    if tiers.tier3:
        agg.update(_aggregate_dict_keys(
            ok, "tier3",
            (
                "playability_fraction", "voice_leading_smoothness",
                "polyphony_in_target_range", "sight_readability", "composite",
            ),
        ))
    if tiers.tier4:
        agg.update(_aggregate_dict_keys(
            ok, "tier4",
            ("chroma_cosine", "round_trip_f1_no_offset", "clap_cosine", "composite"),
        ))
    if tiers.composite_q:
        qs = [r.composite_q for r in ok if isinstance(r.composite_q, (int, float))]
        if qs:
            agg["mean_composite_q"] = round(statistics.fmean(qs), 4)
            agg["median_composite_q"] = round(statistics.median(qs), 4)

    agg["mean_wall_sec"] = round(statistics.fmean(r.wall_sec for r in ok), 2)
    return agg


def _aggregate_dict_keys(
    rows: list[SongScore], tier_attr: str, keys: tuple[str, ...],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in keys:
        values: list[float] = []
        for r in rows:
            tier_dict = getattr(r, tier_attr, None)
            if not isinstance(tier_dict, dict):
                continue
            v = tier_dict.get(k)
            if isinstance(v, (int, float)):
                values.append(float(v))
        if values:
            out[f"mean_{tier_attr}_{k}"] = round(statistics.fmean(values), 4)
            out[f"median_{tier_attr}_{k}"] = round(statistics.median(values), 4)
    return out


# ---------------------------------------------------------------------------
# CI gate evaluation
# ---------------------------------------------------------------------------

@dataclass
class GateOutcome:
    name: str
    passed: bool
    head_value: float | None
    baseline_value: float | None
    delta: float | None
    threshold: float
    direction: str  # "regression_if_drop_gt" or "regression_if_rise_gt"
    message: str


@dataclass
class GateReport:
    outcomes: list[GateOutcome]
    all_passed: bool
    head_run_id: str
    baseline_run_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "all_passed": self.all_passed,
            "head_run_id": self.head_run_id,
            "baseline_run_id": self.baseline_run_id,
            "outcomes": [
                {
                    "name": o.name,
                    "passed": o.passed,
                    "head_value": o.head_value,
                    "baseline_value": o.baseline_value,
                    "delta": o.delta,
                    "threshold": o.threshold,
                    "direction": o.direction,
                    "message": o.message,
                }
                for o in self.outcomes
            ],
        }


# Gate definitions per strategy doc §5.1. Each entry: (name,
# aggregate-key, threshold, direction). ``regression_if_drop_gt``
# means the gate fails when ``head < baseline - threshold``;
# ``regression_if_rise_gt`` flips the sign for metrics where higher
# is worse (none in §5.1 but kept for future extension).
DEFAULT_CI_GATES: tuple[tuple[str, str, float, str], ...] = (
    ("chord_mirex_regression", "mean_tier2_chord_score", 0.03, "regression_if_drop_gt"),
    ("playability_regression", "mean_tier3_playability_fraction", 0.05, "regression_if_drop_gt"),
    ("round_trip_regression", "mean_tier4_round_trip_f1_no_offset", 0.05, "regression_if_drop_gt"),
    ("clap_regression", "mean_tier4_clap_cosine", 0.05, "regression_if_drop_gt"),
)


def apply_ci_gates(
    head_payload: dict[str, Any],
    baseline_payload: dict[str, Any],
    *,
    gates: tuple[tuple[str, str, float, str], ...] = DEFAULT_CI_GATES,
    selected_tiers: TierSelection | None = None,
) -> GateReport:
    """Compare head against baseline payload and report per-gate outcomes.

    Each gate examines one aggregate metric. A gate **passes** when:

    * The CI run did not request the tier the metric belongs to (e.g.
      the cheap CI run leaves Tier 4 off, so Tier 4 gates skip cleanly).
    * The head metric is at most ``threshold`` worse than baseline.

    A gate **fails** when:

    * The head metric is missing while the run *did* request that
      tier — points at a harness/code bug.
    * The baseline metric is missing while head has it — the committed
      baseline is stale and needs to be regenerated against the current
      schema; otherwise a regression could merge silently.
    * The head metric drops by more than ``threshold`` versus baseline.

    Returns a :class:`GateReport` even when nothing failed — the report
    is a useful artifact on green PRs (per-tier delta sparkline).
    """
    head_agg = head_payload.get("aggregate", {})
    base_agg = baseline_payload.get("aggregate", {})
    outcomes: list[GateOutcome] = []
    for name, key, threshold, direction in gates:
        head_v = head_agg.get(key)
        base_v = base_agg.get(key)
        head_present = isinstance(head_v, (int, float))
        base_present = isinstance(base_v, (int, float))

        if not head_present and not base_present:
            # Tier wasn't enabled on either side — gate is genuinely
            # not applicable. Treat as pass to keep the report green
            # for cheap CI runs that leave Tier 4 off by design.
            tier_inactive = (
                selected_tiers is not None
                and not _tier_active_for_key(selected_tiers, key)
            )
            outcomes.append(GateOutcome(
                name=name,
                passed=tier_inactive or selected_tiers is None,
                head_value=None,
                baseline_value=None,
                delta=None,
                threshold=threshold,
                direction=direction,
                message=(
                    f"skipped: tier inactive for this run ('{key}')"
                    if tier_inactive
                    else f"skipped: missing both head and baseline['{key}']"
                ),
            ))
            continue

        if not head_present:
            # CI ran the tier but head failed to produce the metric —
            # signals a harness regression. Block merge so the bug is
            # surfaced before the eval set drifts further.
            outcomes.append(GateOutcome(
                name=name,
                passed=False,
                head_value=None,
                baseline_value=float(base_v) if base_present else None,
                delta=None,
                threshold=threshold,
                direction=direction,
                message=(
                    f"FAIL: head missing aggregate['{key}'] (regression in harness "
                    "or dependency); cannot evaluate gate"
                ),
            ))
            continue

        if not base_present:
            # Head has the metric but the committed baseline doesn't —
            # the baseline predates the current metric schema. Fail
            # loud so a real regression can't merge under the cover of
            # a "skipped" gate. Fix by regenerating the baseline JSON
            # against the current head schema.
            outcomes.append(GateOutcome(
                name=name,
                passed=False,
                head_value=float(head_v),
                baseline_value=None,
                delta=None,
                threshold=threshold,
                direction=direction,
                message=(
                    f"FAIL: baseline missing aggregate['{key}'] — committed baseline "
                    "is stale; regenerate against current schema"
                ),
            ))
            continue

        delta = head_v - base_v
        if direction == "regression_if_drop_gt":
            passed = delta >= -threshold
        else:
            passed = delta <= threshold
        outcomes.append(GateOutcome(
            name=name,
            passed=passed,
            head_value=float(head_v),
            baseline_value=float(base_v),
            delta=float(delta),
            threshold=threshold,
            direction=direction,
            message=(
                f"head={head_v:.4f} baseline={base_v:.4f} delta={delta:+.4f} "
                f"threshold={threshold:.4f} {'PASS' if passed else 'FAIL'}"
            ),
        ))
    all_passed = all(o.passed for o in outcomes)
    return GateReport(
        outcomes=outcomes,
        all_passed=all_passed,
        head_run_id=str(head_payload.get("run_id", "unknown")),
        baseline_run_id=str(baseline_payload.get("run_id", "unknown")),
    )


# Map gate aggregate-key prefix → ``TierSelection`` attribute. Used by
# ``apply_ci_gates`` to tell "tier wasn't requested" (legitimate skip)
# from "tier ran but the metric is missing" (real failure).
_TIER_KEY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("mean_tier_rf_", "tier_rf"),
    ("median_tier_rf_", "tier_rf"),
    ("mean_tier2_", "tier2"),
    ("median_tier2_", "tier2"),
    ("mean_tier3_", "tier3"),
    ("median_tier3_", "tier3"),
    ("mean_tier4_", "tier4"),
    ("median_tier4_", "tier4"),
    ("mean_composite_q", "composite_q"),
    ("median_composite_q", "composite_q"),
)


def _tier_active_for_key(tiers: TierSelection, agg_key: str) -> bool:
    """Return True iff the tier owning ``agg_key`` is enabled in ``tiers``."""
    for prefix, attr in _TIER_KEY_PREFIXES:
        if agg_key.startswith(prefix):
            return bool(getattr(tiers, attr, False))
    return True


def render_gate_summary(report: GateReport) -> str:
    """Pretty-print a :class:`GateReport` for stdout / Slack / CI logs."""
    lines = [
        f"=== CI Eval Gates ({'PASS' if report.all_passed else 'FAIL'}) ===",
        f"  head:     {report.head_run_id}",
        f"  baseline: {report.baseline_run_id}",
    ]
    for o in report.outcomes:
        marker = "✓" if o.passed else "✗"
        lines.append(f"  {marker} {o.name:<28} {o.message}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _describe_tiers(tiers: TierSelection) -> str:
    parts = []
    if tiers.tier_rf:
        parts.append("rf")
    if tiers.tier2:
        parts.append("t2")
    if tiers.tier3:
        parts.append("t3")
    if tiers.tier4:
        sub = []
        if tiers.tier4_round_trip:
            sub.append("rt")
        if tiers.tier4_clap:
            sub.append("clap")
        if tiers.tier4_mert:
            sub.append("mert")
        sub_str = "+".join(sub) if sub else "chroma"
        parts.append(f"t4({sub_str})")
    if tiers.composite_q:
        parts.append("Q")
    return "+".join(parts) or "(none)"


def _tiers_as_dict(tiers: TierSelection) -> dict[str, bool]:
    return {
        "tier_rf": tiers.tier_rf,
        "tier2": tiers.tier2,
        "tier3": tiers.tier3,
        "tier4": tiers.tier4,
        "tier4_round_trip": tiers.tier4_round_trip,
        "tier4_clap": tiers.tier4_clap,
        "tier4_mert": tiers.tier4_mert,
        "composite_q": tiers.composite_q,
    }


def _format_song_summary(row: SongScore, tiers: TierSelection) -> str:
    parts: list[str] = []
    if tiers.tier_rf and row.tier_rf is not None:
        parts.append(
            f"chord_rf={row.tier_rf.get('chord_rf', 0):.3f} "
            f"playability_rf={row.tier_rf.get('playability_rf', 0):.3f} "
            f"chroma_rf={row.tier_rf.get('chroma_rf', 0):.3f}"
        )
    if tiers.tier3 and row.tier3 is not None:
        parts.append(
            f"tier3_composite={row.tier3.get('composite', 0):.3f}"
        )
    if tiers.tier4 and row.tier4 is not None:
        rt = row.tier4.get("round_trip_f1_no_offset")
        if isinstance(rt, (int, float)):
            parts.append(f"round_trip_f1={rt:.3f}")
    if tiers.composite_q and row.composite_q is not None:
        parts.append(f"Q={row.composite_q:.3f}")
    parts.append(f"({row.wall_sec:.1f}s)")
    return "    " + "  ".join(parts)


def _safe_relpath(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------------------
# Legacy tier_rf bridge — keeps the eval_mini stdout table working
# ---------------------------------------------------------------------------

def _to_legacy_song_rows(rows: list[SongScore]) -> list[Any]:
    """Translate :class:`SongScore` to ``eval_mini.SongRow`` for the stdout table.

    The Phase 0 ``_print_table`` helper expects a flat row with
    ``chord_rf`` / ``playability_rf`` / ``chroma_rf`` attributes;
    our :class:`SongScore` carries these inside ``row.tier_rf`` for
    Phase 7. This helper bridges the two so the harness reuses the
    existing pretty-printer instead of duplicating it.
    """
    from scripts.eval_mini import SongRow  # noqa: PLC0415
    legacy: list[Any] = []
    for r in rows:
        legacy_row = SongRow(slug=r.slug, title=r.title, artist=r.artist, genre=r.genre)
        legacy_row.audio_path = r.audio_path
        legacy_row.key_label = r.key_label
        legacy_row.wall_sec = r.wall_sec
        legacy_row.error = r.error
        legacy_row.notes = list(r.notes)
        if r.tier_rf is not None:
            legacy_row.chord_rf = float(r.tier_rf.get("chord_rf", 0.0))
            legacy_row.playability_rf = float(r.tier_rf.get("playability_rf", 0.0))
            legacy_row.chroma_rf = float(r.tier_rf.get("chroma_rf", 0.0))
            legacy_row.n_chord_segments_input = int(r.tier_rf.get("n_chord_segments_input", 0))
            legacy_row.n_chord_segments_resynth = int(r.tier_rf.get("n_chord_segments_resynth", 0))
            legacy_row.n_playable_chords = int(r.tier_rf.get("n_playable_chords", 0))
            legacy_row.n_total_chords = int(r.tier_rf.get("n_total_chords", 0))
            legacy_row.n_beats = int(r.tier_rf.get("n_beats", 0))
        legacy.append(legacy_row)
    return legacy


def _to_legacy_aggregate(rows: list[SongScore]) -> dict[str, Any]:
    ok = [r for r in rows if r.error is None and r.tier_rf is not None]
    if not ok:
        return {}
    return {
        "mean_chord_rf": round(
            statistics.fmean(r.tier_rf["chord_rf"] for r in ok), 4,  # type: ignore[index]
        ),
        "mean_playability_rf": round(
            statistics.fmean(r.tier_rf["playability_rf"] for r in ok), 4,  # type: ignore[index]
        ),
        "mean_chroma_rf": round(
            statistics.fmean(r.tier_rf["chroma_rf"] for r in ok), 4,  # type: ignore[index]
        ),
    }


# ---------------------------------------------------------------------------
# Compat — ensure REPO_ROOT is on sys.path for in-process imports
# ---------------------------------------------------------------------------

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# YAML manifest helper — kept lightweight; full validation lives in eval/loader.py
# ---------------------------------------------------------------------------

def load_baseline(baseline_path: Path) -> dict[str, Any]:
    """Read a baseline JSON file. Raises FileNotFoundError if absent.

    Distinct from :func:`yaml.safe_load` because baselines are JSON
    (the harness writes them as JSON for byte-stability across
    platforms — YAML would re-order keys on round-trip).
    """
    raw = json.loads(Path(baseline_path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"baseline at {baseline_path} did not parse as a mapping")
    return raw


def load_yaml_manifest(manifest_path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(manifest_path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"manifest at {manifest_path} did not parse as a mapping")
    return raw
