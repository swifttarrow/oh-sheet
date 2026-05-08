"""Phase 7 unified eval CLI — Click app with per-stage subcommands.

Subcommand surface (per strategy doc §4.2):

* ``eval transcribe``   — Tier 1 audio→MIDI eval (delegates to
                          ``scripts/eval_transcription.py``).
* ``eval arrange``      — Tier 3 metrics on the post-arrange ``PianoScore``.
* ``eval engrave``      — Tier 3 + chord/dynamics surface checks on the
                          engraved MIDI (and, when present, MusicXML).
* ``eval round-trip``   — Self-consistency probe (transcribe → engrave →
                          resynth → re-transcribe + ``mir_eval`` F1).
* ``eval end-to-end``   — All tiers on the manifest. ``nightly``-shaped run
                          you can drive locally.
* ``eval ci``           — Cheap PR gate: 5-song subset, Tier RF + 2 + 3.
                          Compares vs. a baseline JSON and exits non-zero
                          if any §5.1 gate trips.
* ``eval nightly``      — Full corpus, all tiers, optional CLAP/MERT.
* ``eval compare``      — Diff two run JSONs (delegates to
                          ``scripts/compare_eval_runs.py``).

Each subcommand calls into :mod:`eval.harness` so the orchestration
logic is shared.

Note on import shadowing: when this script runs as ``__main__``,
Python adds ``scripts/`` to ``sys.path[0]`` BEFORE we get to insert
REPO_ROOT. A bare ``from eval.harness import …`` then resolves
``eval`` to *this script* (a single-file module) and fails with
"eval is not a package". The fix is to insert REPO_ROOT at the top
and remove ``scripts/`` from ``sys.path`` before any ``eval.*``
import — done in the small bootstrap block below.
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap — must run before any ``eval.*`` import.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent

# Drop the scripts directory from sys.path so ``import eval`` resolves
# to the ``eval/`` package at REPO_ROOT, not this script file. Python
# auto-adds the script dir when running as ``__main__``; without this
# scrub, ``eval`` shadows itself.
sys.path[:] = [p for p in sys.path if Path(p).resolve() != SCRIPTS_DIR]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Now safe to import the eval package.
# ---------------------------------------------------------------------------

import json  # noqa: E402
import logging  # noqa: E402

import click  # noqa: E402

from eval.harness import (  # noqa: E402
    TierSelection,
    apply_ci_gates,
    load_baseline,
    render_gate_summary,
    run_eval_set,
)

DEFAULT_EVAL_SET = REPO_ROOT / "eval" / "pop_mini_v0"


def _setup_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


# ---------------------------------------------------------------------------
# Top-level Click app
# ---------------------------------------------------------------------------

@click.group(
    help="Phase 7 eval ladder — Tier 1/2/3/4 metric runner + CI gate."
)
@click.option(
    "-v", "--verbose", is_flag=True,
    help="Enable INFO-level logging.",
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


# ---------------------------------------------------------------------------
# eval ci — strategy doc §5.1
# ---------------------------------------------------------------------------

@cli.command(name="ci", help="Run the per-PR gate: 5-song subset, Tier RF + 2 + 3, compare vs. baseline JSON.")
@click.argument(
    "eval_set_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DEFAULT_EVAL_SET,
    required=False,
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Where aggregate.json and gates.json get written.",
)
@click.option(
    "--baseline",
    type=click.Path(dir_okay=False, path_type=Path),
    required=False,
    help="Baseline JSON for gate comparison. Skipped when omitted (reports without gating).",
)
@click.option(
    "--limit", "-n",
    type=int,
    default=5,
    show_default=True,
    help="Cap the number of songs scored (per strategy doc §5.1's 5-song CI subset).",
)
@click.option(
    "--label",
    type=str,
    default="ci",
    show_default=True,
    help="Free-form label written into the run JSON for dashboard grouping.",
)
@click.option(
    "--allow-missing-baseline",
    is_flag=True,
    help="When the baseline path is missing, skip gating (don't fail).",
)
@click.option(
    "--song-timeout-sec",
    type=int,
    default=180,
    show_default=True,
    help=(
        "Per-song wall-clock budget. A hung song (network fetch, ONNX "
        "deadlock) trips the alarm and the harness moves on to the next "
        "song instead of blocking the workflow timeout. 0 disables."
    ),
)
def cmd_ci(
    eval_set_path: Path,
    output_dir: Path,
    baseline: Path | None,
    limit: int,
    label: str,
    allow_missing_baseline: bool,
    song_timeout_sec: int,
) -> None:
    """Run the cheap CI gate.

    Exits 0 on green, 1 on any §5.1 gate failure, 2 on harness-internal
    errors (so a flaky test infrastructure issue can be distinguished
    from a real regression in CI logs).
    """
    tiers = TierSelection.ci()
    payload = run_eval_set(
        eval_set_path=eval_set_path,
        output_dir=output_dir,
        tiers=tiers,
        only_first_n=limit,
        is_ci=True,
        label=label,
        song_timeout_sec=song_timeout_sec or None,
    )

    # A run that scored zero songs cannot meaningfully gate on any
    # metric — every aggregate key drops out and gates "skip" silently
    # (pre-fix). Treat that as a hard failure so manifest/audio/dep
    # breakage in CI surfaces instead of merging green.
    agg = payload.get("aggregate", {})
    n_scored = int(agg.get("n_songs_scored", 0))
    n_total = int(agg.get("n_songs_total", 0))
    if n_scored == 0:
        click.echo(
            f"[ci] FAIL: 0 of {n_total} songs scored — cannot evaluate gates. "
            "Inspect aggregate.json for per-song errors.",
            err=True,
        )
        sys.exit(1)

    if baseline is None:
        click.echo("[ci] no --baseline provided; skipping gate evaluation.")
        sys.exit(0)
    if not baseline.is_file():
        if allow_missing_baseline:
            click.echo(
                f"[ci] baseline not found at {baseline}; skipping gate evaluation "
                "(--allow-missing-baseline)."
            )
            sys.exit(0)
        click.echo(f"[ci] baseline not found at {baseline}", err=True)
        sys.exit(2)

    baseline_payload = load_baseline(baseline)
    report = apply_ci_gates(payload, baseline_payload, selected_tiers=tiers)
    gates_path = output_dir / "gates.json"
    gates_path.write_text(json.dumps(report.as_dict(), indent=2) + "\n")
    click.echo(render_gate_summary(report))
    click.echo(f"\nWrote {gates_path}")
    sys.exit(0 if report.all_passed else 1)


# ---------------------------------------------------------------------------
# eval bootstrap-baseline — regenerate a baseline JSON against current head
# ---------------------------------------------------------------------------

@cli.command(
    name="bootstrap-baseline",
    help=(
        "Run the CI tier selection on an eval-set and write the result as "
        "a baseline JSON. Use when the metric schema changes (e.g. adding "
        "tier2/tier3 keys) and the committed baseline goes stale."
    ),
)
@click.argument(
    "eval_set_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DEFAULT_EVAL_SET,
    required=False,
)
@click.option(
    "--baseline-out",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Path to write the baseline JSON. Existing file is overwritten.",
)
@click.option(
    "--limit", "-n",
    type=int,
    default=5,
    show_default=True,
    help="Cap the number of songs scored — match the CI subset size.",
)
@click.option(
    "--label",
    type=str,
    default="baseline",
    show_default=True,
)
def cmd_bootstrap_baseline(
    eval_set_path: Path,
    baseline_out: Path,
    limit: int,
    label: str,
) -> None:
    """Regenerate a baseline against the current head's metric schema.

    The CI gate compares aggregate keys produced by the current
    ``eval.harness`` against the committed baseline JSON. When the
    metric surface evolves (adding tier2/tier3 keys, etc.) the
    pre-existing baseline becomes schema-stale: gates can't compare
    keys the baseline doesn't carry. This subcommand runs the same
    tier selection as ``eval ci`` and writes the result as a fresh
    baseline so the next CI run has something real to gate on.

    Run from the repo root, e.g.::

        python scripts/eval.py bootstrap-baseline eval/pop_mini_v0/ \\
            --baseline-out eval/baselines/pop_mini_v0__main_<sha>.json
    """
    output_dir = baseline_out.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    tiers = TierSelection.ci()
    payload = run_eval_set(
        eval_set_path=eval_set_path,
        output_dir=output_dir,
        tiers=tiers,
        only_first_n=limit,
        is_ci=False,
        label=label,
    )
    # ``run_eval_set`` writes ``aggregate.json``; rename/copy to the
    # caller-specified baseline path so a side-by-side diff against
    # the previous baseline stays trivial.
    baseline_out.write_text(json.dumps(payload, indent=2) + "\n")
    click.echo(f"Wrote baseline to {baseline_out}")


# ---------------------------------------------------------------------------
# eval nightly — strategy doc §5.2
# ---------------------------------------------------------------------------

@cli.command(name="nightly", help="Full corpus, all tiers + composite Q (CLAP/MERT off by default).")
@click.argument(
    "eval_set_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DEFAULT_EVAL_SET,
    required=False,
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
)
@click.option("--enable-clap", is_flag=True, help="Run Tier 4 CLAP-music cosine (requires laion_clap).")
@click.option("--enable-mert", is_flag=True, help="Run Tier 4 MERT cosine (requires transformers + weights).")
@click.option("--enable-round-trip/--disable-round-trip", default=True, show_default=True)
@click.option("--label", type=str, default="nightly", show_default=True)
def cmd_nightly(
    eval_set_path: Path,
    output_dir: Path,
    enable_clap: bool,
    enable_mert: bool,
    enable_round_trip: bool,
    label: str,
) -> None:
    """Run the nightly corpus."""
    base = TierSelection.nightly()
    tiers = TierSelection(
        tier_rf=base.tier_rf,
        tier2=base.tier2,
        tier3=base.tier3,
        tier4=base.tier4,
        tier4_round_trip=enable_round_trip,
        tier4_clap=enable_clap,
        tier4_mert=enable_mert,
        composite_q=True,
    )
    transcribe_callable = _build_transcribe_callable() if enable_round_trip else None
    payload = run_eval_set(
        eval_set_path=eval_set_path,
        output_dir=output_dir,
        tiers=tiers,
        transcribe_callable=transcribe_callable,
        is_nightly=True,
        label=label,
    )
    click.echo(f"\n[nightly] aggregate keys: {list(payload['aggregate'])[:8]}…")


# ---------------------------------------------------------------------------
# eval end-to-end — full ladder, locally drivable
# ---------------------------------------------------------------------------

@cli.command(name="end-to-end", help="Run all tiers on the manifest. ``nightly``-shaped output you can drive locally.")
@click.argument(
    "eval_set_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DEFAULT_EVAL_SET,
    required=False,
)
@click.option("--output-dir", "-o", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--limit", "-n", type=int, default=None, help="Cap songs (default: all).")
@click.option("--slug", type=str, default=None, help="Run only the song with this slug.")
@click.option("--enable-round-trip/--disable-round-trip", default=False, show_default=True)
@click.option("--label", type=str, default="end-to-end")
def cmd_end_to_end(
    eval_set_path: Path,
    output_dir: Path,
    limit: int | None,
    slug: str | None,
    enable_round_trip: bool,
    label: str,
) -> None:
    base = TierSelection.end_to_end()
    tiers = TierSelection(
        tier_rf=base.tier_rf,
        tier2=base.tier2,
        tier3=base.tier3,
        tier4=base.tier4,
        tier4_round_trip=enable_round_trip,
        composite_q=True,
    )
    transcribe_callable = _build_transcribe_callable() if enable_round_trip else None
    run_eval_set(
        eval_set_path=eval_set_path,
        output_dir=output_dir,
        tiers=tiers,
        only_slug=slug,
        only_first_n=limit,
        transcribe_callable=transcribe_callable,
        label=label,
    )


# ---------------------------------------------------------------------------
# eval arrange — Tier 3 only on a single song
# ---------------------------------------------------------------------------

@cli.command(name="arrange", help="Tier 3 only — diagnose arrange-stage regressions in isolation.")
@click.argument(
    "eval_set_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DEFAULT_EVAL_SET,
    required=False,
)
@click.option("--output-dir", "-o", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--slug", type=str, default=None)
@click.option("--limit", "-n", type=int, default=None)
def cmd_arrange(
    eval_set_path: Path,
    output_dir: Path,
    slug: str | None,
    limit: int | None,
) -> None:
    run_eval_set(
        eval_set_path=eval_set_path,
        output_dir=output_dir,
        tiers=TierSelection.arrange_only(),
        only_slug=slug,
        only_first_n=limit,
        label="arrange-only",
    )


# ---------------------------------------------------------------------------
# eval engrave — chord/dynamics/pedal surface checks
# ---------------------------------------------------------------------------

@cli.command(name="engrave", help="Engraver-output sanity surface (chord-symbol / pedal / dynamics presence).")
@click.argument(
    "eval_set_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DEFAULT_EVAL_SET,
    required=False,
)
@click.option("--output-dir", "-o", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--slug", type=str, default=None)
@click.option("--limit", "-n", type=int, default=None)
def cmd_engrave(
    eval_set_path: Path,
    output_dir: Path,
    slug: str | None,
    limit: int | None,
) -> None:
    """Engrave-stage diagnostic. Same shape as ``arrange`` but layers the
    Tier 2 chord-presence check on top so a chord-symbol drop in
    ``midi_render`` is visible without running the full ladder.
    """
    tiers = TierSelection(
        tier_rf=False, tier2=True, tier3=True, tier4=False, composite_q=False,
    )
    run_eval_set(
        eval_set_path=eval_set_path,
        output_dir=output_dir,
        tiers=tiers,
        only_slug=slug,
        only_first_n=limit,
        label="engrave-only",
    )


# ---------------------------------------------------------------------------
# eval round-trip — Tier 4 round-trip self-consistency only
# ---------------------------------------------------------------------------

@cli.command(name="round-trip", help="Self-consistency probe — Tier 4 round-trip F1 only.")
@click.argument(
    "eval_set_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DEFAULT_EVAL_SET,
    required=False,
)
@click.option("--output-dir", "-o", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--slug", type=str, default=None)
@click.option("--limit", "-n", type=int, default=None)
def cmd_round_trip(
    eval_set_path: Path,
    output_dir: Path,
    slug: str | None,
    limit: int | None,
) -> None:
    transcribe_callable = _build_transcribe_callable()
    run_eval_set(
        eval_set_path=eval_set_path,
        output_dir=output_dir,
        tiers=TierSelection.round_trip_only(),
        only_slug=slug,
        only_first_n=limit,
        transcribe_callable=transcribe_callable,
        label="round-trip",
    )


# ---------------------------------------------------------------------------
# eval transcribe — delegate to scripts/eval_transcription.py
# ---------------------------------------------------------------------------

@cli.command(
    name="transcribe",
    help="Tier 1 audio→MIDI eval — delegates to scripts/eval_transcription.py.",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.pass_context
def cmd_transcribe(ctx: click.Context) -> None:
    """Pass-through wrapper for the existing Tier 1 harness.

    Strategy doc §4.2 lists ``transcribe`` as a Click subcommand;
    ``scripts/eval_transcription.py`` is already the canonical Tier 1
    runner with 1000+ lines of well-tested behavior. Rather than
    re-host that here, we exec it as a module so the existing CLI
    flags (``--device``, ``--limit``, ``--cache-dir``…) keep working.
    """
    import runpy  # noqa: PLC0415
    args = ctx.args
    sys.argv = [str(REPO_ROOT / "scripts" / "eval_transcription.py"), *args]
    runpy.run_path(
        str(REPO_ROOT / "scripts" / "eval_transcription.py"),
        run_name="__main__",
    )


# ---------------------------------------------------------------------------
# eval compare — delegate to scripts/compare_eval_runs.py
# ---------------------------------------------------------------------------

@cli.command(
    name="compare",
    help="Diff two run JSONs — delegates to scripts/compare_eval_runs.py.",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.pass_context
def cmd_compare(ctx: click.Context) -> None:
    """Pass-through wrapper for the existing diff tool."""
    import runpy  # noqa: PLC0415
    args = ctx.args
    sys.argv = [str(REPO_ROOT / "scripts" / "compare_eval_runs.py"), *args]
    runpy.run_path(
        str(REPO_ROOT / "scripts" / "compare_eval_runs.py"),
        run_name="__main__",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_transcribe_callable():
    """Build a ``Path -> midi_bytes`` callable backed by Basic Pitch.

    Returns a callable: when Basic Pitch's import fails (CI without
    the ``[basic-pitch]`` extra), the callable raises a clear
    RuntimeError that the harness catches and surfaces in the per-song
    notes. We prefer that explicit failure over silently turning
    round-trip off.
    """
    def _transcribe(path: Path) -> bytes:
        from backend.services.transcribe import _run_basic_pitch_sync  # noqa: PLC0415

        # Phase 8: ``_run_basic_pitch_sync`` returns a 3-tuple
        # ``(TranscriptionResult, midi_bytes, realtime_pedal_events)``
        # — the third element is unused for the round-trip probe.
        _txr, midi_bytes, _pedals = _run_basic_pitch_sync(path)
        if not midi_bytes:
            raise RuntimeError("Basic Pitch returned empty MIDI bytes")
        return midi_bytes

    return _transcribe


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli(obj={})
