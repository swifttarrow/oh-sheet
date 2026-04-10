#!/usr/bin/env python3
"""Compare two or more eval-results JSON files side-by-side.

Usage::

    python scripts/compare_eval_runs.py eval-results-baseline.json eval-results-crepe-sweep.json

    # With short labels
    python scripts/compare_eval_runs.py \
        --label baseline eval-results-baseline.json \
        --label "CREPE 0.45/0.15" eval-results-crepe-045-015.json

Prints:
  1. Overall headline metrics (F1, P, R) for each run
  2. Per-role F1 deltas
  3. Per-file F1 table with deltas and winners
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _short_label(path: Path) -> str:
    """Derive a short label from the filename."""
    stem = path.stem
    if stem.startswith("eval-results-"):
        stem = stem[len("eval-results-"):]
    return stem or path.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path,
        help="eval-results JSON files to compare")
    parser.add_argument("--label", "-l", action="append", default=[],
        help="label for the next positional file (repeat for each)")
    args = parser.parse_args()

    runs: list[tuple[str, dict]] = []
    for i, f in enumerate(args.files):
        label = args.label[i] if i < len(args.label) else _short_label(f)
        runs.append((label, _load(f)))

    if len(runs) < 2:
        print("Need at least 2 files to compare.", file=sys.stderr)
        return 1

    # --- Overall headline ---
    max_label = max(len(r[0]) for r in runs)
    print("=" * 72)
    print("Overall F1 (no-offset) — the headline metric")
    print("=" * 72)
    header = f"  {'Run':<{max_label}}  {'mean P':>7}  {'mean R':>7}  {'mean F':>7}  {'med F':>7}  {'mean onset':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, data in runs:
        agg = data["aggregate"]
        print(
            f"  {label:<{max_label}}  "
            f"{agg['mean_p_no_offset']:>7.3f}  "
            f"{agg['mean_r_no_offset']:>7.3f}  "
            f"{agg['mean_f1_no_offset']:>7.3f}  "
            f"{agg.get('median_f1_no_offset', 0):>7.3f}  "
            f"{agg.get('mean_f1_with_offset', 0):>10.3f}"
        )

    # Delta row (last vs first)
    if len(runs) >= 2:
        first_agg = runs[0][1]["aggregate"]
        last_agg = runs[-1][1]["aggregate"]
        delta_f = last_agg["mean_f1_no_offset"] - first_agg["mean_f1_no_offset"]
        delta_p = last_agg["mean_p_no_offset"] - first_agg["mean_p_no_offset"]
        delta_r = last_agg["mean_r_no_offset"] - first_agg["mean_r_no_offset"]
        delta_med = (last_agg.get("median_f1_no_offset", 0) -
                     first_agg.get("median_f1_no_offset", 0))
        delta_onset = (last_agg.get("mean_f1_with_offset", 0) -
                       first_agg.get("mean_f1_with_offset", 0))
        print(
            f"  {'Δ (last−first)':<{max_label}}  "
            f"{delta_p:>+7.3f}  "
            f"{delta_r:>+7.3f}  "
            f"{delta_f:>+7.3f}  "
            f"{delta_med:>+7.3f}  "
            f"{delta_onset:>+10.3f}"
        )

    # --- Per-role F1 ---
    print()
    print("=" * 72)
    print("Per-role F1 (last − first)")
    print("=" * 72)
    first_roles = runs[0][1].get("per_role", {})
    last_roles = runs[-1][1].get("per_role", {})
    all_roles = sorted(set(list(first_roles) + list(last_roles)))
    print(f"  {'Role':<10}", end="")
    for label, _ in runs:
        print(f"  {label:>12}", end="")
    print(f"  {'Delta':>8}")
    print("  " + "-" * (10 + 14 * len(runs) + 10))
    for role in all_roles:
        print(f"  {role:<10}", end="")
        vals = []
        for _, data in runs:
            v = data.get("per_role", {}).get(role, {}).get("mean_f1_no_offset", 0)
            vals.append(v)
            print(f"  {v:>12.3f}", end="")
        delta = vals[-1] - vals[0] if len(vals) >= 2 else 0
        sign = "+" if delta >= 0 else ""
        print(f"  {sign}{delta:>7.3f}")

    # --- Per-file F1 table ---
    print()
    print("=" * 72)
    print("Per-file F1 (no-offset)")
    print("=" * 72)
    # Index files by path
    file_indices: list[dict[str, dict]] = []
    all_paths: set[str] = set()
    for _, data in runs:
        idx = {f["path"]: f for f in data.get("files", [])}
        file_indices.append(idx)
        all_paths.update(idx.keys())

    sorted_paths = sorted(all_paths)
    path_w = min(max(len(p) for p in sorted_paths), 45)
    header_parts = [f"  {'File':<{path_w}}"]
    for label, _ in runs:
        header_parts.append(f"  {label[:10]:>10}")
    header_parts.append(f"  {'Delta':>7}")
    print("".join(header_parts))
    print("  " + "-" * (path_w + 12 * len(runs) + 10))

    improvements = 0
    regressions = 0
    for path in sorted_paths:
        short = path[:path_w]
        vals = []
        for idx in file_indices:
            f = idx.get(path)
            if f and f.get("error") is None:
                vals.append(f["f1_no_offset"])
            else:
                vals.append(None)

        print(f"  {short:<{path_w}}", end="")
        for v in vals:
            if v is None:
                print(f"  {'ERR':>10}", end="")
            else:
                print(f"  {v:>10.3f}", end="")

        if len(vals) >= 2 and vals[0] is not None and vals[-1] is not None:
            d = vals[-1] - vals[0]
            sign = "+" if d >= 0 else ""
            marker = " ✓" if d > 0.01 else (" ✗" if d < -0.01 else "")
            print(f"  {sign}{d:>6.3f}{marker}", end="")
            if d > 0.01:
                improvements += 1
            elif d < -0.01:
                regressions += 1
        print()

    print()
    print(f"  Improvements (>+0.01): {improvements}")
    print(f"  Regressions  (<-0.01): {regressions}")
    print(f"  Neutral:               {len(sorted_paths) - improvements - regressions}")

    # --- Key takeaways ---
    if len(runs) >= 2:
        first_agg = runs[0][1]["aggregate"]
        last_agg = runs[-1][1]["aggregate"]
        delta_f1 = last_agg["mean_f1_no_offset"] - first_agg["mean_f1_no_offset"]
        print()
        print("=" * 72)
        verdict = "IMPROVEMENT" if delta_f1 > 0.005 else (
            "REGRESSION" if delta_f1 < -0.005 else "NEUTRAL"
        )
        print(f"Verdict: {verdict} (ΔF1 = {delta_f1:+.3f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
