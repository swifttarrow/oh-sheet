# pop_mini_v0 — Phase 0 reference-free mini-eval set

A 5-song reference-free eval set that runs in under 2 minutes and produces a
JSON before/after diff usable as the comparand for every quick-win PR in
Phases 1–7. See
[`docs/research/transcription-improvement-implementation-plan.md`](../../docs/research/transcription-improvement-implementation-plan.md)
§Phase 0 for the goal and acceptance criteria, and
[`docs/research/transcription-improvement-strategy.md`](../../docs/research/transcription-improvement-strategy.md)
Part III §2.2–2.4 for the metric definitions this implementation targets.

## What's in the box

| File | Purpose |
|---|---|
| `manifest.yaml` | Names the 5 songs, their licenses, content hashes, and source provenance. |
| `../tier_rf.py` | Three reference-free metrics (chord-RF, playability-RF, chroma-RF). |
| `../../scripts/eval_mini.py` | CLI that runs the pipeline on every song in `manifest.yaml` and writes `aggregate.json`. |
| `../baselines/pop_mini_v0__main_<sha>.json` | Snapshotted baseline runs — one per shipped phase. |

## Running

```bash
# One-shot run; writes aggregate.json into a fresh directory.
python scripts/eval_mini.py eval/pop_mini_v0/ eval/runs/$(date -u +%Y%m%dT%H%M%SZ)/

# Skip the per-song pipeline and only compute deterministic metrics
# (handy when iterating on tier_rf.py without re-running Basic Pitch).
python scripts/eval_mini.py eval/pop_mini_v0/ eval/runs/$(date -u +%Y%m%dT%H%M%SZ)/ --reuse-cache
```

The CLI prints a 6-line stdout table (5 songs + aggregate) and writes
`aggregate.json` with per-song breakdowns plus the run's git SHA.

## The three reference-free metrics

| Metric | Range | What it measures |
|---|---|---|
| `chord_rf` | [0, 1] | `mir_eval.chord` MIREX score between chord recognition on the input audio and chord recognition on the FluidSynth re-synth of the engraved MIDI. **Anchor:** strategy doc §2.2 / §4.7. |
| `playability_rf` | [0, 1] | Fraction of per-hand chord groupings (notes that share an `onset_beat`) with span ≤14 semitones AND ≤5 notes. Computed from `PianoScore.right_hand` / `left_hand`. **Anchor:** strategy doc §2.3.1. |
| `chroma_rf` | [0, 1] | Mean per-beat cosine similarity between `chroma_cqt` of the input audio and `chroma_cqt` of the FluidSynth re-synth of the engraved MIDI. Beats are tracked once on the input and shared between both chromas so timestamps line up. **Anchor:** strategy doc §2.4.3. |

All three are *reference-free*: they need no paired ground-truth MIDI/score.
That's the lever that lets us bootstrap before Phase 3's paid 30-song eval
set ships.

## What's in each slot (curated)

| Slug | Track | License | Why this slot |
|---|---|---|---|
| `mini_pop_001` | Jahzzar — Two days | CC BY-SA 3.0 | FMA Pop, mainstream singer-songwriter feel |
| `mini_pop_002` | Tours — Enthusiast | CC BY 3.0 | FMA Electronic, ~335k listens (most-listened CC track on FMA) — vocoder-pop |
| `mini_pop_003` | Alaclair Ensemble — Intergalactique | CC BY-SA 3.0 | FMA Soul-RnB, Quebec hip-hop / R&B-pop |
| `mini_kpop_001` | TWICE — Feel Special (instrumental cover by EVEYLIA / YOUA K Pop) | uploader-asserted CC BY 3.0 | YouTube CC-BY K-pop slot. Cover, not original master — see caveat below |
| `mini_ballad_001` | Peter Rudenko — Snowing | CC BY 3.0 | FMA Classical, solo piano, sparse-feel ballad (~178k listens) |

The 3 FMA Pop / Electronic / Soul-RnB tracks were picked via
`scripts/fma_catalog_filter.py --genre <Pop|Electronic|Soul-RnB>` sorted
by listens. The K-pop slot was found via a YouTube search filtered by
license metadata; the ballad slot is the most-listened CC-BY classical
track on FMA. Per-slot rationale and FMA track ids are in the
`source.notes` blocks of `manifest.yaml`.

### License caveat — `mini_kpop_001`

The K-pop slot is an instrumental cover of TWICE's "Feel Special"
uploaded under CC BY 3.0 by `EVEYLIA / YOUA K Pop` on YouTube
(`R9KNBIFDohw`). License declarations on YouTube are uploader-asserted,
not platform-verified. The original composition is © JYP Entertainment;
this manifest entry references the CC-BY-uploaded *cover* only. Verify
before any commercial redistribution.

### Replacing a slot

Run the curate script with a new URL or local file:

```bash
python scripts/curate_pop_mini_v0.py mini_kpop_001 \
    https://www.youtube.com/watch?v=<id> \
    --license cc-by --force
```

`--force` overwrites the previously-curated audio. The manifest gets
rewritten with the new content hash, license, and source URL; the
`intended_source` block is preserved verbatim.

## Baselines

`eval/baselines/pop_mini_v0__main_<short-sha>.json` is the canonical
baseline for a given commit on `main`. Multiple baselines ship
side-by-side so you can see how the metric surface evolved:

| File | Schema | Source | Notes |
|---|---|---|---|
| `pop_mini_v0__main_bde4268.json` | v2 | Curated audio (current manifest) | **Active CI baseline.** Phase 7 metric surface (tier_rf + tier2 + tier3 + composite Q). 4 delivered slots, 1 undelivered (mini_kpop_001). |
| `pop_mini_v0__main_5532577_real.json` | v1 (legacy) | Curated audio (pre-Phase-7 manifest) | Reference-free metrics only. Kept for historical comparison. |
| `pop_mini_v0__main_5532577.json` | v1 (legacy) | Bootstrap (synthetic_from_midi via FluidSynth) | Pre-curation snapshot. Faster to re-run (~35 s) but less faithful to real-pop conditions. |

### Regenerating the active baseline

The `eval-ci.yml` workflow gates on aggregate keys (e.g.
`mean_tier2_chord_score`, `mean_tier3_playability`) by comparing the
PR's run against the committed baseline. Two situations need a regen:

1. **Metric surface evolves** — a new tier or key lands in `eval/harness.py`
   and the committed baseline doesn't carry it. The gate now fails loud
   ("baseline missing key X") rather than silently skipping, so a stale
   baseline shows up as a red CI run on the next PR.
2. **Intentional baseline shift** — a change is *meant* to move metrics
   (e.g. enabling a new transcriber path). Regenerate so future PRs
   compare against the new normal.

Run from the repo root on a clean working copy of `main`:

```bash
python scripts/eval.py bootstrap-baseline eval/pop_mini_v0/ \
    --baseline-out eval/baselines/pop_mini_v0__main_<short-sha>.json
```

Then update the `--baseline` arg in
[`.github/workflows/eval-ci.yml`](../../.github/workflows/eval-ci.yml)
to point at the new file and commit both in the same change. The
workflow itself carries the same recipe in a top-of-file comment for
discoverability when triaging a red gate.

Subsequent PRs compare their `aggregate.json` against the active baseline
using the `scripts/eval.py ci` subcommand (cheap PR gate) or
`scripts/compare_eval_runs.py` for ad-hoc diffing.

A baseline is committed only when its tier metrics are stable across
two consecutive runs (caching makes re-runs deterministic except for
the Basic Pitch backend's tiny floating-point variance).
