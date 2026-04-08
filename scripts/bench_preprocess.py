"""A/B benchmark: preprocessing off vs preprocessing on.

Runs :func:`backend.services.transcribe._run_basic_pitch_sync` twice on
the same audio file — once with ``audio_preprocess_enabled=False``, once
with it True — and prints a side-by-side diagnostic so we can see how
HPSS + RMS normalization shift the distribution of Basic Pitch's output,
how cleanup reacts, and how the role-split counts land.

There's no ground truth here; the goal is to *measure the shift*, not to
score accuracy. Specifically we want to know:

  * Does HPSS pull Basic Pitch's per-note amplitudes down as a block?
    (If so, ``basic_pitch_onset_threshold`` / ``frame_threshold`` may
    need to come down in the preprocess-on profile.)

  * Does the merge / octave-prune / ghost-tail cleanup drop more or
    fewer notes after preprocessing? (Tells us whether cleanup is
    still doing useful work or has been rendered redundant.)

  * Do the per-role counts (melody / bass / chords) land in roughly
    the same proportions? (If the distribution flips wildly, the
    Viterbi extractors may need their voicing floors re-evaluated.)

Usage::

    python scripts/bench_preprocess.py assets/rising-sun-1.mp3
    python scripts/bench_preprocess.py assets/rising-sun-{1,2,3}.mp3

Any failure in a single file is logged and the script continues —
benchmarking is best-effort, and we'd rather see partial results than
abort on the first unreadable fixture.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile — no numpy dep needed at the top level."""
    if not values:
        return float("nan")
    sorted_vals = sorted(values)
    k = max(0, min(len(sorted_vals) - 1, int(round(pct / 100.0 * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def _summarize_amps(amps: list[float]) -> dict[str, float]:
    if not amps:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "p10": float("nan"), "p90": float("nan"),
                "min": float("nan"), "max": float("nan")}
    return {
        "n": len(amps),
        "mean": statistics.fmean(amps),
        "median": statistics.median(amps),
        "p10": _percentile(amps, 10),
        "p90": _percentile(amps, 90),
        "min": min(amps),
        "max": max(amps),
    }


def _amps_from_result(result: Any) -> list[float]:
    """Pull per-note 'amplitude' proxies from a TranscriptionResult.

    Contract ``Note`` stores ``velocity`` (int 1–127) not the raw Basic
    Pitch amplitude, but the two are monotonic (``velocity = round(127 *
    amp)``) so the shape of the distribution is preserved. We return
    amplitudes in the 0–1 range for comparability with Basic Pitch's
    raw output.
    """
    out: list[float] = []
    for track in result.midi_tracks:
        for note in track.notes:
            out.append(note.velocity / 127.0)
    return out


def _print_row(label: str, a: Any, b: Any, fmt: str = "{:>10}") -> None:
    print(f"  {label:<28} " + fmt.format(str(a)) + "   " + fmt.format(str(b)))


def _print_header() -> None:
    print(f"  {'metric':<28} {'preprocess OFF':>10}   {'preprocess ON':>10}")
    print(f"  {'-'*28} {'-'*10}   {'-'*10}")


def _extract_warning_summary(result: Any) -> dict[str, str]:
    """Pick out the interesting per-stage summary lines from warnings."""
    out: dict[str, str] = {}
    for w in result.quality.warnings:
        if w.startswith("cleanup:"):
            out.setdefault("cleanup", []).append(w[len("cleanup:"):].strip())  # type: ignore[attr-defined]
        elif w.startswith("melody"):
            out["melody"] = w
        elif w.startswith("bass"):
            out["bass"] = w
        elif w.startswith("chords:"):
            out["chords"] = w
        elif w.startswith("audio preprocess"):
            out["preprocess"] = w
    # Collapse cleanup list back to a string.
    if "cleanup" in out and isinstance(out["cleanup"], list):
        out["cleanup"] = "; ".join(out["cleanup"])  # type: ignore[index]
    return out  # type: ignore[return-value]


def _run_once(audio_path: Path, preprocess_enabled: bool) -> tuple[Any, float]:
    """Run _run_basic_pitch_sync once with preprocess flag flipped."""
    import backend.config as cfg
    import backend.services.transcribe as T
    from backend.config import Settings

    # Build a fresh Settings instance with the flag overridden so env
    # vars don't leak between runs.
    old_settings = cfg.settings
    override = Settings(audio_preprocess_enabled=preprocess_enabled)
    cfg.settings = override
    T.settings = override

    try:
        t0 = time.perf_counter()
        result, _midi_bytes = T._run_basic_pitch_sync(audio_path)
        elapsed = time.perf_counter() - t0
        return result, elapsed
    finally:
        cfg.settings = old_settings
        T.settings = old_settings


def _role_counts(result: Any) -> dict[str, int]:
    """Return {role_name: note_count} for each non-empty midi track."""
    out: dict[str, int] = {}
    for track in result.midi_tracks:
        out[track.instrument.value] = len(track.notes)
    return out


def _bench_file(audio_path: Path) -> None:
    print(f"\n=== {audio_path.name} ===")
    if not audio_path.is_file():
        print("  ! not a file, skipping")
        return

    try:
        off_result, off_wall = _run_once(audio_path, preprocess_enabled=False)
    except Exception as exc:
        print(f"  ! preprocess OFF run failed: {exc}")
        return

    try:
        on_result, on_wall = _run_once(audio_path, preprocess_enabled=True)
    except Exception as exc:
        print(f"  ! preprocess ON run failed: {exc}")
        return

    off_amps = _amps_from_result(off_result)
    on_amps = _amps_from_result(on_result)
    off_stats = _summarize_amps(off_amps)
    on_stats = _summarize_amps(on_amps)

    _print_header()
    _print_row("wall time (s)", f"{off_wall:.2f}", f"{on_wall:.2f}")
    _print_row("overall confidence",
               f"{off_result.quality.overall_confidence:.2f}",
               f"{on_result.quality.overall_confidence:.2f}")
    _print_row("total notes", off_stats["n"], on_stats["n"])
    _print_row("amp mean",
               f"{off_stats['mean']:.3f}" if off_stats["n"] else "-",
               f"{on_stats['mean']:.3f}" if on_stats["n"] else "-")
    _print_row("amp median",
               f"{off_stats['median']:.3f}" if off_stats["n"] else "-",
               f"{on_stats['median']:.3f}" if on_stats["n"] else "-")
    _print_row("amp p10",
               f"{off_stats['p10']:.3f}" if off_stats["n"] else "-",
               f"{on_stats['p10']:.3f}" if on_stats["n"] else "-")
    _print_row("amp p90",
               f"{off_stats['p90']:.3f}" if off_stats["n"] else "-",
               f"{on_stats['p90']:.3f}" if on_stats["n"] else "-")

    # Per-role counts.
    off_roles = _role_counts(off_result)
    on_roles = _role_counts(on_result)
    all_roles = sorted(set(off_roles) | set(on_roles))
    for role in all_roles:
        _print_row(f"role: {role}",
                   off_roles.get(role, 0),
                   on_roles.get(role, 0))

    # Stage-summary warnings (cleanup / melody / bass / chords / preprocess).
    off_w = _extract_warning_summary(off_result)
    on_w = _extract_warning_summary(on_result)
    print()
    for key in ("cleanup", "melody", "bass", "chords", "preprocess"):
        if key in off_w or key in on_w:
            print(f"  {key:<12}")
            print(f"    OFF: {off_w.get(key, '-')}")
            print(f"    ON:  {on_w.get(key, '-')}")

    # Derived deltas for the three thresholds that matter.
    print()
    if off_stats["n"] and on_stats["n"]:
        amp_shift = on_stats["mean"] - off_stats["mean"]
        med_shift = on_stats["median"] - off_stats["median"]
        count_shift = on_stats["n"] - off_stats["n"]
        count_pct = 100.0 * count_shift / off_stats["n"]
        print(f"  Δ amp mean:    {amp_shift:+.3f}")
        print(f"  Δ amp median:  {med_shift:+.3f}")
        print(f"  Δ note count:  {count_shift:+d} ({count_pct:+.1f}%)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audio_paths",
        nargs="+",
        type=Path,
        help="Audio file(s) to benchmark (WAV/MP3/M4A/etc.).",
    )
    args = parser.parse_args()

    print("Benchmarking preprocess OFF vs ON")
    print("Note: same model, same thresholds; only audio_preprocess_enabled differs.")

    for path in args.audio_paths:
        _bench_file(path)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
