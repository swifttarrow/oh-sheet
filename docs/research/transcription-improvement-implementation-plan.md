# Oh Sheet — Transcription Improvement Implementation Plan

**Companion to:** [`transcription-improvement-strategy.md`](./transcription-improvement-strategy.md)
**Date:** 2026-04-25
**Audience:** engineering — the executable counterpart to the research deliverable
**Optimization targets (in priority order):**
1. **Session-scoped phases** — every phase fits inside a single focused engineering session (≈3–6 hours / 1 day).
2. **Workflow parallelization** — phases that can ship concurrently are explicitly grouped; per-phase parallel work streams are mapped to agent assignments.
3. **Quick turnaround + demonstrable improvement** — every phase produces a user-visible artifact (not just CI green) and has a documented demo checkpoint.

---

## How this plan differs from the strategy document's roadmap

The strategy doc proposes an 8-milestone sequence (Sprint 0 → M8) optimized for engineer-weeks of throughput. This plan reshapes that sequence around three constraints the strategy doc doesn't optimize for:

| Strategy doc | This plan | Why the change |
|---|---|---|
| Sprint 0: build a 30-song paid eval set ($900–1.5k, ~1 week) before any engineering | **Phase 0:** stand up a **5-song internal mini-eval** with reference-free metrics in **one session**; let the 30-song paid set build in parallel with Phases 1–3 | Quick wins must be measurable on day 1. The 30-song set is on a transcriber's clock (4–8 weeks); we cannot block engineering on it. RF metrics work without paired references. |
| M2 (Demucs) → M3 (Kong) → M4 (engraver) | **Engraver first (Phase 4)** before Demucs (Phase 5) and Kong (Phase 6) | Strategy doc §428 itself argues "Fix Themes E + C + F1 first, then validate, then decide whether model swap is worth complexity." Engrave channel preservation multiplies every downstream model improvement. Doing Demucs first means measuring it through a lossy engrave. |
| M1 = "13 tasks, one sprint, one engineer" | **Phase 1 = 3 parallel agents, one session** | Quick wins A1–A13 are 12 mostly-independent edits. Three agents working in parallel collapse a 2-week sprint into a half-day plus review. |
| M3 bundles Kong + Beat This! | Beat This! ships with **Phase 2** (engrave plumbing window); Kong gets its own phase | Beat This! is a 1-day madmom replacement with no dependency on Kong. Bundling lengthens the critical path. |
| Pop2Piano stays as default through M5 | **Phase 1 demotes Pop2Piano** behind a feature flag; AMT-APC takes the default in Phase 8 | Pop2Piano's "no LICENSE file" risk doesn't justify keeping it as default for an extra 5 milestones. Demote first, swap when AMT-APC ships. |
| Single sequential milestone chain | Explicit DAG with parallel tracks (Phases 5+6 run side-by-side; Phase 7 telemetry runs in parallel with Phases 4–6 by a separate engineer) | Two engineers can finish the M1–M7 work in ≈6 weeks instead of ≈12 by parallelizing eval/telemetry against model integration. |

The acceptance criteria in this plan are inherited from the strategy doc with one tightening: every phase requires a **demo checkpoint** (a short Loom or screenshot of the user-visible improvement), not just a CI green.

---

## Phase summary table

Numbers in **Anchor** column reference the strategy doc's A/B/C IDs (Section II) plus weakness IDs (Part I §1–7).

| # | Phase | Anchor | Sessions | Parallel? | Demo checkpoint | Pre-requisite |
|---|---|---|---|---|---|---|
| 0 | **Mini-eval foundation** | F1, F2 (start) | 1 | — | RF metrics print for 5 fixture songs; before/after JSON | none |
| 1 | **Config quick wins** (3 parallel streams) | A1, A2, A3, A4, A6, A7, A8, A9, A11, A12 | 1 (3 parallel agents) | ✅ within phase | Bigger LH, varied dynamics, no stub arpeggios on failure; Phase 0 metrics show lift | Phase 0 |
| 2 | **Engrave channel-preservation in MIDI** + Beat This! | A5 (partial), A10, A13, B3 | 1 | ✅ A10/A13 vs. B3 | Engraved score now carries key sig + pedal + downbeats | Phase 1 |
| 3 | **Pop eval set v1.0 (30 songs)** | F1 finish | 1 (engineering) + 4 weeks (transcribers, parallel) | ✅ runs alongside 4–6 | Holdout-versioned manifest; first real-pop F1 number | Phase 0 |
| 4 | **Local engraver: music21 → MusicXML → Verovio + LilyPond** | A5 (finish), B7, E1, E2, E3 | 2–3 | ✅ vs. Phase 3 build, Phase 7 telemetry | Real `pdf_uri` non-null with title/composer/dynamics | Phase 2 |
| 5 | **Source separation: HTDemucs** | B1, A1 (finish), A2 | 1–2 | ✅ vs. Phase 4 | A/B page: vocal-stem vs. full-mix waveforms; Phase 0 lift | Phase 1 |
| 6 | **Piano-stem AMT: Kong + pedal events** | B2, D6 | 2 | ✅ vs. Phase 4 | Engraved score with sustain pedal markings on a piano-cover input | Phases 2 + 5 |
| 7 | **Eval ladder + telemetry hardening** | F1, F4, F7, "production Q" | 2 | ✅ vs. Phases 4–6 (different engineer) | Grafana dashboard with per-tier sparklines; CI badge | Phase 3 |
| 8 | **AMT-APC cover mode + Pop2Piano retirement** | B6, B2 (finish), G3 | 1–2 | ✅ vs. Phase 7 | UI toggle "Faithful / Cover"; A/B preference vote ≥55% | Phases 5 + 7 |
| 9 | **Velocity refinement (Score-HPT) + voice/staff GNN** | B5, B8 | 1–2 each | ✅ B5 vs. B8 | Dynamics that breathe; correct hand assignment on tenor-range pop | Phase 8 |
| 10 | **Research bets (Q3–Q4)** | C1, C2, C3, C4 | quarter-scale | ✅ all four | Internal corpus; difficulty critic; AMT-APC fine-tune | Phase 9 |

**Critical path (sequential):** Phase 0 → 1 → 2 → 4 → 6 → 8 → 9. ≈6 sessions plus Phase 4's 2-session engraver work plus Phase 6's 2-session Kong work = **~10–12 focused sessions**.

**Maximum parallel throughput:** with two engineers, Phase 5 (Demucs) ships alongside Phase 4 (engraver); Phase 7 (telemetry) ships alongside Phase 6 (Kong). Real-elapsed time from Phase 0 start to Phase 8 demo: **~6 calendar weeks**.

---

## Phase 0 — Mini-eval foundation (1 session)

**Goal:** Stand up a 5-song reference-free eval that runs in <2 minutes and produces a JSON before/after diff usable for every subsequent PR.

### Why this is Phase 0 instead of Sprint 0

The strategy doc's Sprint 0 is a 30-song paid eval ($900–1.5k, 4+ weeks). That cost and timeline are correct for *release-gate* evaluation, but they block *engineering iteration*. Phase 0 ships in a single session and unblocks Phases 1–7. The 30-song paid set runs in parallel as Phase 3.

### Tasks

1. **Pick 5 fixture songs from a free source** — 3 from FMA `commercial=true` pop, 1 K-pop YouTube clip (CC-BY only), 1 sparse ballad. Store under `/Users/ross/oh-sheet/eval/pop_mini_v0/songs/<slug>/source.audio.{wav,mp3}` plus `manifest.yaml` with content-hash + license.
2. **Add `eval/tier_rf.py`** — three reference-free metrics, all per-song, no paired ground truth required:
   - **Tier 2 RF**: chord-progression accuracy via `mir_eval.chord` MIREX mode, where the "reference" is `chord_recognition` run on the input audio and the "estimate" is `chord_recognition` run on the FluidSynth re-synthesis of the engraved MIDI. (Anchor against audio-side chord estimator on the same input.)
   - **Tier 3 RF**: playability fraction = % of chords with span ≤14 semitones AND ≤5 notes per hand. Compute from `PianoScore` after arrange.
   - **Tier 4 RF**: chroma cosine between input audio and FluidSynth re-synth of the engraved MIDI, beat-bucketed via the existing `audio_timing` beat tracker.
3. **Wire CLI** — `scripts/eval_mini.py` that takes `(eval_set_path, output_dir)` and writes `eval/runs/<run-id>/aggregate.json` with the three metrics + per-song breakdowns. Subcommand-compatible with the eventual `scripts/eval.py` from Phase 7.
4. **Snapshot baseline** — run on `main` head; commit `eval/baselines/pop_mini_v0__main_5532577.json` so subsequent PRs have a comparand.

### Files to touch

- New: `/Users/ross/oh-sheet/eval/pop_mini_v0/manifest.yaml`
- New: `/Users/ross/oh-sheet/eval/pop_mini_v0/songs/<slug>/...`
- New: `/Users/ross/oh-sheet/eval/tier_rf.py`
- New: `/Users/ross/oh-sheet/scripts/eval_mini.py`
- New: `/Users/ross/oh-sheet/eval/baselines/pop_mini_v0__main_<sha>.json`
- Reuse: `scripts/eval_transcription.py:115-131` (FluidSynth + bundled SF2 — already working)
- Reuse: `backend/services/chord_recognition.py`
- Reuse: `backend/services/audio_timing.py:99-148` (beat tracker)

### Demo checkpoint

`python scripts/eval_mini.py eval/pop_mini_v0/ eval/runs/$(date -Iseconds)/` prints a 6-line table to stdout:

```
song              chord-rf  playability-rf  chroma-rf
fma_pop_001        0.42      0.71            0.48
fma_pop_002        0.31      0.65            0.39
yt_kpop_001        0.55      0.74            0.51
fma_ballad_001     0.62      0.81            0.58
fma_indie_001      0.38      0.69            0.44
mean               0.46      0.72            0.48
```

Save as the **Phase 0 baseline** — every subsequent phase reports a delta against this.

### Acceptance

- `pytest scripts/eval_mini.py` exists and runs in <60 s.
- Aggregate JSON file produced and committed as a baseline.
- Three metrics all in [0, 1] for all 5 songs (no NaNs, no exceptions).

### Risks

- Chord-recognition both sides may be too tightly correlated to detect arrange/engrave regressions. **Mitigation:** run a smoke-A/B by deliberately injecting a 5-semitone transpose on the engraved MIDI; the chroma-RF metric must drop ≥0.10. If it doesn't, switch to `mir_eval.chord` with `tetrads` instead of `mirex`.

### Effort

**1 session, single engineer.**

---

## Phase 1 — Config quick wins (1 session, 3 parallel agents)

**Goal:** Land 10 of the 13 quick wins (A1–A13 from strategy §A) in a single session by splitting them into three independent work streams. Phase 0's mini-eval reports the lift.

### Why parallelize

A1–A13 from the strategy doc are sequentially scheduled inside M1 because the doc assumes one engineer. But ten of them are 1-line config edits with no shared file footprint. Three Claude Code agents can work concurrently on three orthogonal file sets and converge in one session.

### Parallel work streams

#### Stream 1A — Default pipeline / arrangement / velocity (Agent A)

**Touches:** `backend/config.py`, `backend/services/arrange.py`, `backend/services/arrange_simplify.py`, `shared/shared/contracts.py`

| ID | Change | Diff hint |
|---|---|---|
| A1 | `score_pipeline = "condense_only"` → `"arrange"` | `backend/config.py:562` |
| A2 | `MAX_VOICES_RH = MAX_VOICES_LH = 2` → `4`; expose `arrange_max_voices_{rh,lh}` settings | `backend/services/arrange.py:51-52, 226-227` |
| A3 | `arrange_simplify_min_velocity = 55` → `25`; replace linear `[35,120]` remap with percentile-based | `backend/config.py:219`, `arrange.py:365-390` |
| A9 | Track-confidence `<0.35 → drop` → `add_warning(...) + clamp(0.05)` | `backend/services/arrange.py:53, 142-146` |
| Demote Pop2Piano | `pop2piano_enabled = True` → `False` (default), keep behind feature flag | `backend/config.py:244` |

**Demo:** before/after on Phase 0 mini-eval — expect LH note-count ≥1.5×, dynamics range visibly wider in resynth waveform.

#### Stream 1B — Stub-fallback removal + test infrastructure (Agent B)

**Touches:** `backend/services/transcribe_result.py`, `tests/conftest.py`, `tests/test_real_transcribe_smoke.py` (new), `README.md`, `docker-compose.yml`

| ID | Change | Diff hint |
|---|---|---|
| A6 | Replace `_stub_result` C-E-G-C arpeggio with explicit `TranscriptionFailure` | `backend/services/transcribe_result.py:249-278` |
| A11 | Remove `autouse=True` on `skip_real_transcription`; add opt-in fixture | `tests/conftest.py:46-68` |
| A11.b | New `tests/test_real_transcribe_smoke.py` — runs real Basic Pitch on 5-second synth-piano clip, asserts F1 ≥ 0.4 | new file |
| A12 | README/code drift: remove `engrave.py`, music21, LilyPond mentions until they exist (Phase 4); document HTTP engrave dependency | `README.md` lines 127, 248, 303 |
| A13 | Add `transcription_midi_uri` field to `EngravedOutput`; emit raw transcription as separate artifact | `backend/jobs/runner.py:501-512`; `shared/shared/contracts.py:348-377` |

**Demo:** intentionally trigger a Basic Pitch failure (e.g., 0-second audio) → user sees error, not a fake C-major arpeggio. CI smoke test runs real BP.

#### Stream 1C — Inference parameters + audio plumbing (Agent C)

**Touches:** `backend/services/transcribe_inference.py`, `backend/services/transcribe_pipeline_pop2piano.py`, `backend/services/transcribe_pipeline_single.py`, `backend/services/transcribe_midi.py`, `shared/shared/contracts.py`

| ID | Change | Diff hint |
|---|---|---|
| A4 | Pass `minimum_frequency = settings.minimum_frequency_hz` (default 30 Hz, configurable) to BP `predict()` | `backend/services/transcribe_inference.py:123-129` |
| A7 | Pass `multiple_pitch_bends=True`; add `pitch_bend_cents: list[tuple[float,float]]` to `Note`; preserve through `transcribe_midi.py` | `transcribe_midi.py:50-54`; `shared/shared/contracts.py:157-161` |
| A8 | Stop forcing mono 22.05 kHz on the Pop2Piano sub-path; only mono-mix when the path's model actually requires it (BP only) | `transcribe_pipeline_pop2piano.py:65`, `transcribe_pipeline_single.py:74` |

**Demo:** sub-bass kick artifact disappears on a hip-hop fixture; pitch bends survive on a vocal/guitar fixture.

### Convergence + integration (end of session)

After all three agents land their PRs:

1. **Re-baseline `eval-baseline.json`** on the existing 25-MIDI synth set — A1 (default arranger) will move numbers; capture the new baseline so M3+ regressions don't blame Phase 1 unfairly.
2. **Run Phase 0 mini-eval**, commit `eval/baselines/pop_mini_v0__phase1_<sha>.json`.
3. **Verify no test breakage** — A1 and A2 will fail snapshot tests in `tests/test_arrange.py`; re-baseline those snapshots in the same PR.

### Acceptance (composite, end of phase)

- All 10 changes merged to `main`.
- Phase 0 mini-eval shows: chord-RF ≥+0.05 vs. baseline, playability-RF ≥−0.05 (allowing for legitimate density growth from A2), chroma-RF ≥+0.05.
- New CI smoke test green.
- README accurate.
- Stub arpeggio gone (test asserts).

### Risks

- **A1 + A2 together** could produce dense, less-readable scores. **Mitigation:** run Phase 0 mini-eval on the *engraved MIDI*, not just the arranged score, so the engraver's collapse smooths over excess density temporarily. Plan to revisit voice cap in Phase 9 (B8 GNN).
- **A3** percentile remap may surface genuine low-velocity noise from Basic Pitch on noisy stems. **Mitigation:** combine with A2 (more voices accepted) — quiet inner-voice notes belong in voice 3/4, not dropped.

### Effort

**1 session, 3 agents in parallel** (≈3 hours wall + 1 hour reviewer integration).

---

## Phase 2 — Engrave channel preservation in MIDI + Beat This! (1 session, 2 parallel streams)

**Goal:** Stop dropping key signature, downbeats, pedal events, and chord symbols at the MIDI render boundary. This is "free quality" — every datum already exists upstream and is silently discarded at `backend/services/midi_render.py:36-137`.

### Why this is Phase 2 (not bundled into Phase 1)

The MIDI render rewrite touches a different file set than the Phase 1 streams and would have introduced merge conflicts. Splitting it into its own phase keeps Phase 1's three agents conflict-free and lets a single Phase 2 engineer focus on the contract-preservation work.

### Parallel work streams

#### Stream 2A — MIDI text-event channel preservation (Agent A)

**Touches:** `backend/services/midi_render.py`, `backend/jobs/runner.py`, `shared/shared/contracts.py`

| ID | Change | Diff hint |
|---|---|---|
| A10 | Emit `pretty_midi.KeySignature` from `metadata.key`; emit *all* `metadata.tempo_map` entries (not just first); emit `expression.tempo_changes` as MIDI tempo change events | `backend/services/midi_render.py:36-137` |
| New | Emit chord symbols as `Marker` MIDI text events (one per chord change) — gives the remote engraver something to read until full MusicXML lands in Phase 4 | `backend/services/midi_render.py` (new function `_emit_chord_markers`) |
| New | Emit downbeats from Beat This! (Stream 2B) as `Cue Point` events for engraver bar-line alignment | `backend/services/midi_render.py` (after Stream 2B lands) |
| E5 fix | `EngravedScoreData.includes_*` flags computed from emitted content, not hard-coded `False` | `backend/jobs/runner.py:516-520` |

**Demo:** open the engraved MIDI in MuseScore Studio — bar lines align to actual downbeats; key signature is correct (no spurious accidentals); chord symbols appear as text annotations.

#### Stream 2B — Replace madmom with Beat This! (Agent B)

**Touches:** `backend/services/audio_timing.py`, `pyproject.toml`, `shared/shared/contracts.py`

| ID | Change | Diff hint |
|---|---|---|
| B3 | Drop madmom `RNNBeatProcessor + DBNBeatTrackingProcessor`; install `beat_this`; emit `beats: list[float]` AND `downbeats: list[float]` | `backend/services/audio_timing.py:99-148` |
| D6 | Add `downbeats: list[float]` to `HarmonicAnalysis` contract | `shared/shared/contracts.py:185-190` |
| Cleanup | Remove madmom from `pyproject.toml`; unpins old numpy/scipy | `pyproject.toml` |

**Demo:** Phase 0 chord-RF and chroma-RF both lift on the K-pop fixture (which has 16th-note hi-hats — current madmom DBN smears them). Confirm madmom uninstall trims dependency tree.

### Convergence

After both streams ship:

1. Stream 2A's `_emit_chord_markers` reads from `metadata.chord_symbols`, which already exists.
2. Stream 2A's downbeat-as-cue-point emission reads from Stream 2B's new `downbeats` field — Stream 2B must land first OR Stream 2A guards on `if downbeats:`.
3. Re-run Phase 0 mini-eval; commit new baseline.

### Acceptance

- `EngravedScoreData.includes_chord_symbols` is `True` on every job (verifiable by JSON inspection).
- Engraver-rendered output (whatever the current external engraver does with our new MIDI) shows visible improvement on at least 3 of 5 mini-eval songs.
- madmom removed from `pyproject.toml`; CI dependency install ≥30s faster.
- Beat F1 on synth baseline within ±5 ppt of madmom (regression guard).

### Risks

- **Beat This! license** is "MIT-ish per README" per the strategy doc — verify before shipping. If license is non-permissive, fall back to keeping madmom and shipping Stream 2A standalone.
- **The remote `oh-sheet-ml-pipeline` engraver may ignore Marker / Cue Point events.** Consider this Phase 2 a "preserve the channel" change — actually rendering chord symbols on the score requires Phase 4 (local engraver). Stream 2A is the *contract* fix; Phase 4 is the *rendering* fix.

### Effort

**1 session, 2 agents in parallel** (≈4 hours wall).

---

## Phase 3 — Pop eval set v1.0 (parallel external work)

**Goal:** Build the 30-song hand-curated pop eval set. Mostly external (transcribers + licensing) — the engineering work is one session of harness scaffolding to ingest the deliverable.

### Why this runs in parallel with Phases 4–6

The transcriber contract takes 4–8 weeks to deliver. Engineering cannot block on it. Phase 0 mini-eval covers the iteration loop until Phase 3 lands; Phase 3 expands the eval surface to support release gates and the Q1 Tier 5 calibration study (see strategy doc Part III §3, §9).

### Tasks

1. **Source 30 songs:** 10 FMA `commercial=true` (free, redistributable) + 20 commercial pop tracks via $50 sync licenses for internal-only use. Cover ≥3 genres (mainstream pop, hip-hop, K-pop, ballad, indie/electronic).
2. **Contract transcribers:** Upwork or Fiverr music transcribers, $30/song × 30 = $900. Specify deliverable format: paired (audio_hash, MIDI, MusicXML, structural.yaml). Use the artifact bundle layout from strategy doc Part III §3.2.
3. **Engineering session:**
   - Build `eval/pop_eval_v1/` directory layout per strategy doc §3.2 / §4.1.
   - Add `eval/loader.py` that reads the manifest and yields `(audio_uri, ref_midi, ref_xml, structural)` tuples.
   - Add `eval/holdout.py` with a 50/50 tune/holdout split and an encrypted `holdout_manifest.yaml.enc` (engineer-readable + one release manager key).
4. **Snapshot first real-pop F1 baseline:** run the existing `scripts/eval_transcription.py` plus the new `scripts/eval_mini.py` on the tune set; commit `eval/baselines/pop_eval_v1__baseline_<phase2_sha>.json`. **This is the first time we have a real number for "what does this pipeline do on a pop song?"** — a critical milestone.

### Files

- New: `/Users/ross/oh-sheet/eval/pop_eval_v1/manifest.yaml`
- New: `/Users/ross/oh-sheet/eval/pop_eval_v1/songs/<slug>/...` (×30)
- New: `/Users/ross/oh-sheet/eval/pop_eval_v1/holdout_manifest.yaml.enc`
- New: `/Users/ross/oh-sheet/eval/loader.py`
- New: `/Users/ross/oh-sheet/eval/holdout.py`
- New: `/Users/ross/oh-sheet/eval/baselines/pop_eval_v1__baseline_<sha>.json`

### Acceptance

- 30 songs × (audio + MIDI + structural.yaml) + tune/holdout split locked, frozen, tagged `pop_eval_v1.0.0`.
- First real-pop transcription F1 number measured and committed (the strategy doc's §9 estimate is 0.05–0.15 — the actual number will set Phase 4–6 acceptance bars).
- Holdout encrypted; only release manager has key.
- `eval/loader.py` `pytest`-tested on a fixture song.

### Risks

- **Licensing risk** for the 20 commercial tracks: confirm sync license terms permit internal research use *and* derivative MIDI redistribution to engineers. If unclear, drop to 30 FMA-only songs (slimmer genre coverage but cleaner legal posture).
- **Transcriber quality varies.** Spot-check 5% manually; reject and re-contract any song with structural-key disagreement vs. a chord-recognition pass.

### Effort

**1 engineering session for harness; 4–8 weeks elapsed for the corpus delivery.**

---

## Phase 4 — Local engraver: music21 → MusicXML → Verovio + LilyPond (2–3 sessions)

**Goal:** Replace the external HTTP black-box engrave with a local stack that consumes structured `(PianoScore, ExpressionMap)` directly. This is the **single largest quality multiplier** in the entire roadmap (strategy doc §10 rank 2 + §E1).

### Why now (not later)

The strategy doc's §428 explicit recommendation: "Fix Themes E + C + F1 first, then validate on real pop, then decide whether the model swap (B) is worth the complexity." Engrave channel preservation amplifies every downstream model improvement. Building Demucs (Phase 5) before fixing engrave means measuring Demucs's quality through a lossy engraver — wasted signal.

### Sessions

#### Session 4.1 — `engrave_local.py` skeleton + MusicXML writer (1 session)

**Touches:** `backend/services/engrave_local.py` (new), `pyproject.toml` (add `music21>=9.7`)

- Build `score_to_musicxml(score: PianoScore, expression: ExpressionMap) -> bytes` using `music21`.
- Emit: key signature, time signature, chord symbols (`metadata.chord_symbols` → `music21.harmony.ChordSymbol`), tempo marking, voice numbers (per `ScoreNote.voice`), dynamics (per `ExpressionMap.dynamics`), pedal marks (sustain CC64).
- Validate output against MusicXML 4.0 XSD; assert no UNRESOLVED warnings.

**Demo checkpoint 4.1:** `python -c "from backend.services.engrave_local import score_to_musicxml; print(score_to_musicxml(...))"` produces a MusicXML that opens cleanly in MuseScore Studio with title, dynamics, and chord symbols visible.

#### Session 4.2 — Verovio (SVG) + LilyPond (PDF) renderers (1 session)

**Touches:** `backend/services/engrave_local.py` (extend), `pyproject.toml` (add `verovio>=4.x`), `Dockerfile` (add `lilypond` apt package)

- `musicxml_to_svg(xml: bytes) -> bytes` via `verovio` (LGPL — link only, never statically embed).
- `musicxml_to_pdf(xml: bytes) -> bytes` via `lilypond` subprocess (GPL, isolated to subprocess only).
- Add timeout guards (LilyPond is single-threaded, can hang on malformed XML); 60-second cap with explicit failure path.

**Demo checkpoint 4.2:** `pdf_uri` is non-null on all 5 Phase 0 mini-eval songs. Open the PDF — title, composer, key signature, tempo marking all present.

#### Session 4.3 — Pipeline integration + external engraver becomes optional (1 session)

**Touches:** `backend/jobs/runner.py`, `backend/services/ml_engraver_client.py`, `backend/config.py`

- New `PipelineConfig.engrave_backend: Literal["local", "remote_http"] = "local"`.
- `runner.py:431-536` engrave block dispatches to `engrave_local` by default; falls through to remote HTTP only when `engrave_backend = "remote_http"` OR local engrave fails.
- Update `EngravedScoreData` flag computation to read from actual MusicXML content (continues Phase 2 Stream 2A's fix).
- `tests/test_engrave_local.py` (new) — MusicXML validates, PDF non-empty, includes_dynamics asserted from real content.

**Demo checkpoint 4.3:** End-to-end run on a Phase 0 mini-eval song produces a PDF with dynamics text, chord symbols, pedal marks. Side-by-side compare to today's output (no PDF, no chord symbols, no dynamics) — the lift is unambiguous.

### Acceptance (end of phase)

- `pdf_uri` non-null on ≥95% of jobs (regression guard for malformed scores).
- 5 pianists rate engraved output ≥3.5/5 on "looks like real sheet music" axis (defer this Tier 5 sub-study to Phase 7 if external engraver was already producing PDFs; otherwise run a small in-network spot-check now).
- Phase 0 mini-eval: chord-RF ≥+0.10 vs. Phase 2 baseline (chord symbols now actually rendered).
- Round-trip metric (audio→engrave→re-synth→re-transcribe) F1 ≥0.85 (strategy doc Tier 4 §4.7).
- External `oh-sheet-ml-pipeline` is now optional; default deployment doesn't require it.

### Risks

- **LilyPond GPL subprocess isolation must be airtight.** Use `subprocess.run` with no Python data sharing; pass MusicXML by file path; never link in-process. Document in code comment + `LICENSING.md`.
- **Verovio LGPL link-only.** Use the `verovio` PyPI wheel which is dynamically linked; never bundle the C++ source statically.
- **music21 startup is ~3 s.** Cache-load it in long-running workers; for one-shot jobs, accept the cost.
- **Container size grows ~150 MB** (LilyPond binary + music21 + verovio). Acceptable per strategy doc §E2.

### Effort

**2–3 sessions, single engineer.** Can run in parallel with Phase 5 (different engineer).

---

## Phase 5 — Source separation: HTDemucs (1–2 sessions)

**Goal:** Insert HTDemucs as a first-class stage between ingest and transcribe, producing per-stem WAVs and an `instrumental.wav` (= bass + other) that downstream piano transcription consumes. Vocals/drums suppressed = false-positive onsets dropped.

### Sessions

#### Session 5.1 — `separate` worker + contract extension (1 session)

**Touches:** `backend/workers/separate.py` (new), `shared/shared/contracts.py`, `backend/jobs/runner.py`, `Dockerfile`, `pyproject.toml`

- `pip install demucs` (MIT, archive 2025-01 — pin a fork or vendor `htdemucs` weights).
- `backend/workers/separate.py` exposes a Celery task that takes 44.1 kHz stereo WAV in, writes 4 stem WAVs out, returns `audio_stems: dict[str, BlobURI]`.
- Extend `TranscriptionResult` with `audio_stems: dict[str, BlobURI] = {}`.
- Pre-cache HTDemucs weights in Docker build to avoid cold-start downloads.
- New `PipelineConfig.separator: Literal["htdemucs", "off"] = "htdemucs"` (default ON for new pipelines, OFF for legacy backwards-compat).

#### Session 5.2 — Stem-routing in transcribe (1 session)

**Touches:** `backend/services/transcribe.py`, `backend/services/transcribe_pipeline_stems.py`

- When `audio_stems` populated, route Basic Pitch (or Pop2Piano legacy) over `bass + other` summed stem instead of full mix.
- Per-stem confidence thresholding (strategy doc §3.3 already partially exists at `config.py:70-83`).
- Cache by `sha256(audio_bytes) + model_id` per strategy doc §5.4.

### Demo checkpoint

A/B page in `eval/runs/<phase5>/per_song/<slug>/`:
- `original.wav` / `vocals.wav` / `drums.wav` / `instrumental.wav` audio players (4 waveforms).
- Mini-eval chord-RF and chroma-RF: expected lift ≥+0.05 vs. Phase 4.
- Engraved score side-by-side: drum-onset ghost notes gone in mini-eval hip-hop fixture.

### Acceptance

- HTDemucs default; 4 stems written per job.
- Phase 0 mini-eval: chord-RF ≥+0.05 vs. Phase 4; chroma-RF ≥+0.05.
- Real-pop melody F1 lift ≥+0.05 vs. Phase 4 (strategy doc M2 acceptance).
- Separation latency ≤60 s on 2-vCPU Cloud Run for HTDemucs on 3-min songs.
- Pop eval v1.0 (Phase 3) re-baselined.

### Risks

- **SDR ↑ does NOT automatically mean transcription F1 ↑** (strategy doc cites Whisper-ALT 2025: separation can trigger AMT hallucinations). **Mitigation:** A/B every PR using mini-eval; if F1 drops on any subset, gate the separator behind a confidence-based routing rule.
- **HTDemucs repo archived 2025-01.** Pin a fork before the first PR; document the pin in `pyproject.toml`.
- **Memory:** HTDemucs 7 GB peak RAM. Pin Cloud Run to 8 GB; document.

### Effort

**1–2 sessions, single engineer.** Runs in parallel with Phase 4 if a second engineer is available.

---

## Phase 6 — Piano-stem AMT: Kong + pedal events (2 sessions)

**Goal:** Replace Basic Pitch on the piano-stem path with ByteDance Kong's piano transcription model. Kong is the *only* mature, pip-installable, commercially-licensed transcriber that emits sustain pedal events — pedal is the difference between "blizzard of staccato eighths" and a readable score.

### Pre-requisite

- Phase 5 (Demucs) provides clean piano-ish stems (`bass + other`).
- Phase 4 (local engraver) renders pedal marks from `pedal_events` — without it, Kong's pedal output is invisible to the user.

### Sessions

#### Session 6.1 — Kong wrapper + `pedal_events` contract (1 session)

**Touches:** `backend/services/transcribe_kong.py` (new), `backend/services/transcribe.py`, `shared/shared/contracts.py`, `pyproject.toml`

- `pip install piano-transcription-inference` (MIT).
- Wrap Kong inference in `transcribe_kong.py`; produces notes + sustain pedal events.
- Extend `TranscriptionResult` with `pedal_events: list[PedalEvent] = []` (per strategy doc §D target schema).
- New `PedalEvent(cc, onset_sec, offset_sec, confidence)` Pydantic model.

#### Session 6.2 — Routing + engrave plumbing (1 session)

**Touches:** `backend/services/transcribe.py:52-119`, `backend/services/arrange.py` (preserve `pedal_events` through), `backend/services/engrave_local.py` (render pedal marks)

- Routing logic: `if vocal_energy < threshold or user_hint == "piano": use Kong; else: use Basic Pitch (or AMT-APC after Phase 8)`.
- `pedal_events` flows: transcribe → arrange (passthrough) → engrave (Verovio/LilyPond render `Ped.` and `*` marks).
- `EngravedScoreData.includes_pedal_marks` is now `True` whenever `pedal_events` non-empty (continues the Phase 2 / Phase 4 flag work).

### Demo checkpoint

Engraved score for a piano-cover input shows sustain pedal markings (`Ped. ___ *`). Compare to Phase 5 output of the same input — Phase 5 was a blizzard of staccato eighths; Phase 6 has held bass notes, dampened resonance, idiomatic piano writing.

### Acceptance

- Pop eval v1.0 (Phase 3) Note F1 ≥0.45 (strategy doc M3 acceptance).
- Pedal events visible in ≥80% of pop songs containing pedal (manual spot-check on 10 piano-cover fixtures).
- MAESTRO test split Note F1 ≥0.95 (regression guard — Kong should be near SOTA on MAESTRO).
- Phase 0 mini-eval: chord-RF ≥+0.05 vs. Phase 5; **playability-RF ≥+0.10 vs. Phase 5** (pedal handles density that staccato cannot).

### Risks

- **Kong is MAESTRO-overfit.** Edwards et al. 2024: −19.2 F1 on pitch-shift, −10.1 on reverb. **Mitigation:** Phase 5 Demucs provides clean piano-ish stem to Kong; never run Kong on raw YouTube audio. Gate Kong invocation on Demucs success.
- **Kong latency:** ~CPU-minutes per minute-of-audio. Acceptable for current Cloud Run; add GPU-spot-pool routing in Phase 10 if user-perceived latency becomes an issue.
- **Repo archived Dec 2025; PyPI package actively maintained** — pin the PyPI version, don't pull from GitHub.

### Effort

**2 sessions, single engineer.** Runs in parallel with Phase 7 (different engineer).

---

## Phase 7 — Eval ladder + telemetry hardening (2 sessions)

**Goal:** Promote Phase 0's mini-eval to the full 5-tier metric ladder from strategy doc Part III. Add CI gates, nightly runs, and production telemetry. This is what makes Phase 8+ improvements **measurable and gateable**.

### Why this runs in parallel with Phases 4–6

Phase 7 is a separate file footprint (`eval/`, `.github/workflows/`, `grafana/`) from the model integration work. A second engineer can ship it while the first works through Phase 4–6.

### Sessions

#### Session 7.1 — Tier 2/3/4 modules + Click CLI (1 session)

**Touches:** `scripts/eval.py` (new), `eval/tier2_structural.py` (new), `eval/tier3_arrangement.py` (new), `eval/tier4_perceptual.py` (new)

- `scripts/eval.py` Click app with subcommands per strategy doc §4.2 (`transcribe`, `arrange`, `engrave`, `end-to-end`, `round-trip`, `ci`, `nightly`, `compare`).
- `eval/tier2_structural.py`: key/tempo/beat/chord/section RF metrics per strategy doc §2.2.
- `eval/tier3_arrangement.py`: playability fraction, voice-leading smoothness, polyphony density, sight-readability per strategy doc §2.3.
- `eval/tier4_perceptual.py`: CLAP-music cosine, MERT cosine, chroma cosine, round-trip self-consistency F1 per strategy doc §2.4.

#### Session 7.2 — CI gates + nightly + telemetry (1 session)

**Touches:** `.github/workflows/eval-ci.yml` (new), `.github/workflows/eval-nightly.yml` (new), `backend/eval/telemetry.py` (new), `backend/jobs/runner.py`

- `eval-ci.yml`: runs `eval ci` on PRs; gates per strategy doc §5.1 (chord-mirex regress >3 ppt = block; playability drop >5 ppt = block; round-trip drop >5 ppt = block; CLAP drop >0.05 = block).
- `eval-nightly.yml`: runs `eval nightly` on cron 2 AM UTC; posts summary to `#oh-sheet-eval` Slack.
- `backend/eval/telemetry.py`: emits per-job composite Q score (strategy doc §8.2) into Postgres `eval_production_quality_scores` table.
- Postgres schema migration (strategy doc §6.1).
- Grafana dashboard JSON (`grafana/dashboards/oh-sheet-eval.json`) with per-tier sparklines.

### Demo checkpoint

- A test PR that intentionally regresses Phase 1's `arrange_simplify_min_velocity` from 25 back to 55 → CI fails on Tier 3 playability gate. Screenshot the PR check.
- Grafana dashboard URL with one week of mini-eval + nightly data.
- Production job `EngravedOutput.evaluation_report` field populated (new field).

### Acceptance

- CI eval pipeline runs end-to-end in <2 min on PRs (strategy doc §5.1).
- Nightly runs in <15 min (strategy doc §5.2).
- Postgres telemetry inserts Q score for every production job.
- At least one PR has been merged that triggered (and respected) a CI gate.
- Grafana dashboard accessible to all engineers.

### Risks

- **CLAP-music cosine variance** is high song-to-song; gate threshold of 0.05 absolute may flap. **Mitigation:** A/B-test the gate on 10 historical PRs before turning it on as blocking.
- **Postgres schema changes** require migration — coordinate with deploy.

### Effort

**2 sessions, single engineer (different from Phase 4–6 engineer).**

---

## Phase 8 — AMT-APC cover mode + Pop2Piano retirement (1–2 sessions)

**Goal:** Add a parallel "Cover Mode" pipeline using AMT-APC (MIT-licensed Pop2Piano-style hFT-Transformer descendant). Surface as UI toggle. Retire Pop2Piano (no LICENSE file = legal risk per strategy doc §G3).

### Pre-requisite

- Phase 5 (Demucs) provides `instrumental.wav` for AMT-APC.
- Phase 7 (eval ladder) gates the A/B comparison.

### Sessions

#### Session 8.1 — AMT-APC integration (1 session)

**Touches:** `backend/services/transcribe_amt_apc.py` (new), `backend/services/transcribe.py`, `shared/shared/contracts.py`, `frontend/lib/upload.dart`

- Clone or pip-install AMT-APC weights (MIT-licensed; pre-cache in Docker build).
- New `PipelineConfig.variant: Literal["audio_upload", "midi_upload", "sheet_only", "pop_cover"]` — adds `pop_cover`.
- `pop_cover` variant: ingest → separate (Phase 5) → AMT-APC on instrumental → **skip arrange entirely** → engrave (Phase 4 local engraver).
- Frontend UI toggle: "Faithful transcription" (default, uses Phase 6 Kong path) vs. "Piano cover" (uses AMT-APC).

#### Session 8.2 — Pop2Piano retirement + A/B preference study (1 session, optional)

**Touches:** `backend/config.py`, `backend/services/transcribe.py`, `backend/services/transcribe_pop2piano.py`, `tests/`

- Delete `transcribe_pop2piano.py` (or move to `legacy/` with a `WARNING.md`).
- Remove `pop2piano_enabled` config flag.
- Run informal A/B preference study: 30 listeners on Phase 3 eval set, AMT-APC vs. Phase 6 Kong-faithful path. Strategy doc M6 acceptance: ≥55% preference for AMT-APC on "Musicality" axis.

### Demo checkpoint

Frontend shows "Faithful / Cover" toggle. User uploads a pop song → Faithful produces sheet-music-correct rendering with pedal; Cover produces a pianistic cover with idiomatic accompaniment patterns. Side-by-side audio comparison.

### Acceptance

- AMT-APC default for `pop_cover` variant.
- Pop2Piano removed or quarantined.
- A/B preference (≥30 listeners) ≥55% for AMT-APC on Musicality axis.
- Phase 7 CI gates green for both variants.
- Container image growth bounded ≤200 MB (AMT-APC + hFT base ≈100 MB).

### Risks

- **AMT-APC trained mostly on J-pop YouTube covers.** Western-pop generalization unaudited. **Mitigation:** track per-genre Phase 0 metrics; if Western-pop chord-RF underperforms, Phase 10 C1 (self-collected corpus + fine-tune) becomes the priority.
- **Output may exceed playability constraints** (cover models often write hand-impossible chords). **Mitigation:** Phase 9's voice/staff GNN + hand-span clamp post-process.

### Effort

**1–2 sessions, single engineer.**

---

## Phase 9 — Velocity refinement (Score-HPT) + voice/staff GNN (1–2 sessions each, parallel)

**Goal:** Two independent quality lifts that share no file footprint. Run as parallel agents in a single session window.

### Stream 9A — Score-HPT velocity refinement

**Anchor:** Strategy doc B5; weakness B7.

**Touches:** `backend/services/score_hpt.py` (new — paper-only at writing, requires reimplementation), `backend/services/transcribe.py` (insert before arrange)

- ~1M-param BiLSTM/Transformer head between transcribe and arrange that re-estimates velocities from onset positions.
- Plumbs back into `Note.velocity` before arrange's `_normalize_velocity` runs.
- **Acceptance:** MAESTRO Note+Off+Vel F1 ≥0.85 (strategy doc M5 acceptance).

**Effort:** 1 session, single engineer.

### Stream 9B — Cluster-and-Separate GNN voice/staff assignment

**Anchor:** Strategy doc B8; weakness C5.

**Touches:** `backend/services/voice_gnn.py` (new), `backend/services/arrange.py:135-156` (replace SPLIT_PITCH=60)

- Replace naive `pitch >= 60` middle-C hand split with GNN-based voice/staff assigner per Karystinaios & Widmer 2024 ([arXiv:2407.21030](https://arxiv.org/html/2407.21030v1)).
- Input: notes with onset, duration, pitch. Output: `voice` and `hand` per note.
- **Acceptance:** Voice/staff F1 ≥0.80 on a labeled 20-song hand-split eval (strategy doc M8 acceptance).

**Effort:** 1–2 sessions, single engineer.

### Combined demo checkpoint

Tenor-range pop hook (e.g., "Despacito" verse) — Phase 8 output had it shredded across hands at middle C; Phase 9B output assigns it cleanly to the right hand. Phase 9A: dynamics that breathe (sustained quarter notes have detectable velocity envelope, not just `int(round(127*amp))`).

### Risks

- **Score-HPT is paper-only** — budget reimplementation time. If author code lands before Phase 9, use it.
- **GNN training data** may be classical-piano-biased. Augment with POP909 (MIT) symbolic data.

---

## Phase 10 — Long-term research bets (Q3–Q4)

The strategy doc's Section C lists six research bets (C1–C6). For this implementation plan, only the highest-leverage four are scoped; the other two stay as Q4+ stretch goals.

| Bet | Description | Effort | Pre-req |
|---|---|---|---|
| **C1** Self-collected pop+piano-cover corpus + AMT-APC fine-tune | Scrape 5,000 pairs (Billboard Hot-100 × top piano-cover videos), audfprint align, pseudo-label via Aria-AMT, fine-tune AMT-APC | 1 quarter, $30k all-in | Phase 8 |
| **C3** Sight-readability critic | Train ~5M-param classifier scoring `PianoScore` for hand-span, voice leading, Henle difficulty grade | 1 quarter | Phase 9B |
| **C4** Anticipatory Music Transformer infill | Symbolic-only LLM for inner-voice infill given melody+chord skeleton | 0.5 quarter (research probe) | Phase 9B |
| **C2** MR-MT3-equivalent multi-track decomposer (license-clean) | Re-train MT3 on Slakh + POP909 + C1 corpus on permissive data | 2 quarters | C1 |

Each runs as its own multi-week effort outside the session-scoped framing of Phases 0–9. Track via `RESEARCH-BETS.md` with quarterly checkpoints.

---

## Parallelization map (visual)

```
Time →    | Week 1     | Week 2     | Week 3-4   | Week 5     | Week 6     | Week 7-12+
Engineer 1| P0 → P1 → P2 → P4 (engraver, 2-3 sessions) → P6 (Kong, 2 sessions) → P8 → P9
Engineer 2|              P3 (eval set, 1 session + 4-8wk wait) → P5 (Demucs, 1-2 sessions) → P7 (eval ladder, 2 sessions) → P9
External  |                                  P3 transcriber contract running ──────────────────►
                                                                                                  Q3-Q4: P10 research bets
```

**Critical-path session count (single engineer):** ≈10–12 focused sessions.
**Two-engineer wall-clock:** ≈6 calendar weeks Phase 0 → Phase 8 demo.

---

## Demo checkpoint inventory (the user-visible artifacts you accumulate)

| Phase | Artifact |
|---|---|
| 0 | 5-song mini-eval JSON; baseline number for every subsequent phase |
| 1 | LH bigger; dynamics audibly varied; no stub C-major arpeggio on failure |
| 2 | Engraved MIDI carries key signature, downbeats; chord symbols as MIDI markers |
| 3 | First **real-pop F1 number** (vs. paid human reference) — likely 0.05–0.15 per strategy doc §9 |
| 4 | Real PDF with title, dynamics, chord symbols, pedal marks |
| 5 | Per-stem WAV downloads; A/B vocal vs. instrumental |
| 6 | Engraved score with `Ped. ___ *` markings on a piano-cover input |
| 7 | Grafana dashboard with per-tier sparklines + CI gate enforcement |
| 8 | Frontend "Faithful / Cover" toggle; A/B preference vote ≥55% |
| 9 | Tenor-range pop hook assigned cleanly to one hand; dynamics breathe |
| 10 | Internal corpus, difficulty critic, fine-tuned AMT-APC |

---

## Risks not bound to any single phase

| Risk | Mitigation |
|---|---|
| Schema bumps break frontend | Add v3→v4 downgrade shim in `shared/shared/contracts.py`; phase the frontend update behind a feature flag. |
| Cloud Run cold starts grow with image size | Bake all model weights into image; warm-start a long-running container per worker type. |
| Per-job latency target slips (today ~30–60s, target 2–4 min on CPU) | Ship a GPU-spot-pool routing rule by Phase 8; degrade gracefully on CPU-only. |
| Tier 5 (human eval) doesn't validate composite Q | Strategy doc §10.8 fallback: don't gate releases on Q until calibration confirms ρ≥0.7. Phase 7 ships Q as advisory only; Q1 release-gate study upgrades to blocking only if calibration succeeds. |
| Holdout leakage during incident response | Holdout encrypted; only release manager has key; document in `RELEASE-GATES.md`. |
| Licensing surprise on a model swap | Run a license audit gate in CI: every new model dep must be checked against the strategy doc §E3 table. |

---

## How to start (today)

1. **Read this plan and the strategy doc Executive Summary** (≈30 min).
2. **Open Phase 0.** Pick 5 fixture songs from FMA. Stand up `eval/pop_mini_v0/` and `scripts/eval_mini.py` per the Phase 0 task list. Commit baseline JSON.
3. **Spawn three Phase 1 agents in parallel** (Stream 1A / 1B / 1C). Re-baseline mini-eval. Open three PRs.
4. **Open Phase 2.** Spawn two parallel agents (Stream 2A / 2B). Re-baseline.
5. **Kick off Phase 3** (transcriber contract) — this runs in the background for 4–8 weeks.
6. **Phase 4 starts on a single engineer**; Phase 5/7 start on a second engineer.
7. By Week 6: Phase 8 demo. The user-visible artifact is the "Faithful / Cover" toggle producing a real PDF with chord symbols, dynamics, and pedal marks on a real pop song.

---

## Reference index

- Strategy doc: [`./transcription-improvement-strategy.md`](./transcription-improvement-strategy.md)
  - Executive Summary: §38
  - Top 10 weaknesses: Part I §8
  - Quick-win diff hints: Part II §A
  - Architectural changes: Part II §B
  - Target architecture diagram: Part II §D
  - Risk/cost table: Part II §E
  - 5-tier metric ladder: Part III §1.3, §2
  - Eval set design: Part III §3
  - Test harness: Part III §4
  - CI/release gates: Part III §5
  - Composite Q score: Part III §8
- This file: implementation phasing + parallelization + session boundaries.
- Phase 1 weakness anchors: A1 (`config.py:562`), A2 (`arrange.py:51-52`), A3 (`config.py:219`), A4 (`transcribe_inference.py:123-129`), A6 (`transcribe_result.py:249-278`), A7 (`transcribe_midi.py:50-54`), A8 (`transcribe_pipeline_pop2piano.py:65`), A9 (`arrange.py:53, 142-146`), A10 (`midi_render.py:36-137`), A11 (`tests/conftest.py:46-68`), A12 (`README.md:127, 248, 303`), A13 (`runner.py:501-512`).
- Phase 2 anchors: A5 partial (`midi_render.py:36-137`), B3 (`audio_timing.py:99-148`).
- Phase 4 anchors: B7 (new `engrave_local.py`), E1/E2 (`runner.py:516-520`, `midi_render.py:36-137`), E3 (`runner.py:524`).
- Phase 5 anchors: B1 (new `workers/separate.py`).
- Phase 6 anchors: B2 (new `transcribe_kong.py`), D6 (`audio_timing.py`).
- Phase 7 anchors: F1/F4 (`scripts/eval.py` new); telemetry (`backend/eval/telemetry.py` new).
- Phase 8 anchors: B6 (new `transcribe_amt_apc.py`); G3 (delete Pop2Piano).
- Phase 9 anchors: B5 (new `score_hpt.py`); B8 (new `voice_gnn.py`, replace `arrange.py:135-156`).
- Phase 10 anchors: C1, C2, C3, C4 (research-bet effort outside session-scoping).
