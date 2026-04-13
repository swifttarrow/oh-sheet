# Roadmap: GAU-105 LLM Engraving Refinement

## Overview

This milestone adds a sixth pipeline stage — `refine` — that slots between `humanize` and `engrave`, using the Anthropic Claude API (with web search grounding) to modify or delete notes in the `HumanizedPerformance` before LilyPond rendering. The stage is opt-in at the job level, defaults to off, and is gated by a kill switch and API-key presence. The delivery journey moves foundation-first (contracts + config), then builds the service and wires it into the pipeline against mocked Anthropic responses, then flips to live Anthropic integration and exposes the toggle in the Flutter frontend, and finally ships an A/B harness that produces the baseline data that gates further work. Every phase leaves the pipeline in a working state: existing jobs are unaffected until `enable_refine=True`, and when refine fails at runtime the job still succeeds with an unrefined PDF.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Contracts and Plumbing** - Schema v3.1.0 models, config knobs, kill switch, and API key handling — no behavior change
- [ ] **Phase 2: Refine Service and Pipeline Integration** - End-to-end refine stage with mocked Anthropic, runner wiring, engrave unwrap, and graceful skip-on-failure
- [ ] **Phase 3: Live Anthropic and Frontend Toggle** - Real Anthropic API validation plus Flutter upload checkbox, progress UI, and skip badge
- [ ] **Phase 4: A/B Harness and Baseline** - Paired pipeline runs across a ≥10-song manifest producing the shipping-gate baseline report

## Phase Details

### Phase 1: Contracts and Plumbing
**Goal**: Land the Pydantic v3.1.0 contract surface, pipeline config fields, and Anthropic API key handling so downstream phases have a frozen foundation to build against.
**Depends on**: Nothing (first phase)
**Requirements**: CTR-01, CTR-02, CTR-03, CTR-04, CTR-05, CFG-01, CFG-02, CFG-03, CFG-04, CFG-05, CFG-06, CFG-07, CFG-08
**Success Criteria** (what must be TRUE):
  1. A developer can import `RefinedPerformance`, `RefineEditOp`, and `RefineCitation` from `shared/shared/contracts.py`, round-trip them through `model_dump(mode="json")`/`model_validate()`, and see schema version `v3.1.0` on the payload
  2. `POST /v1/jobs` with `enable_refine: true` and no `OHSHEET_ANTHROPIC_API_KEY` set returns HTTP 400 with a clear message; with the key set and `enable_refine: false` the request succeeds and the execution plan contains no refine step
  3. Setting `OHSHEET_REFINE_KILL_SWITCH=true` causes any submission with `enable_refine=true` to behave exactly as if `enable_refine=false` — refine is absent from the execution plan
  4. Only models listed in the `OHSHEET_REFINE_ALLOW_OPUS`-aware allowlist are accepted; a request configured for a non-allowlisted model is rejected at config load time
  5. Running `git status` on a fresh checkout confirms `.env` is gitignored and the pre-commit secret-scan hook blocks a commit containing a mock Anthropic key
**Plans**: TBD

### Phase 2: Refine Service and Pipeline Integration
**Goal**: Ship an end-to-end refine stage — service, validator, worker, runner wiring, engrave unwrap, and skip-on-failure — validated against a mocked Anthropic SDK so the happy path and failure paths are both exercised before any real API spend.
**Depends on**: Phase 1
**Requirements**: STG-01, STG-02, STG-03, STG-04, STG-05, STG-06, STG-07, STG-08, STG-09, STG-10, INT-01, INT-02, INT-03, INT-04, INT-05, INT-06, INT-07
**Success Criteria** (what must be TRUE):
  1. A full pipeline job submitted with `enable_refine=true` against a stubbed Anthropic SDK runs all six stages in order, writes a `RefinedPerformance` blob, and produces a PDF whose engrave input was unwrapped from `payload_type="RefinedPerformance"`
  2. When the mocked Anthropic call raises a transient error, the runner emits `stage_completed` with `message="refine_skipped: <reason>"` (never `stage_failed`), the `refine_skip_total` counter increments, engrave runs against the unrefined `HumanizedPerformance`, and the final job status is `succeeded`
  3. The validator rejects an edit whose `target_note_id` is not in the source performance's note ID set, rejects a `harmony_correction` edit on a velocity-floor ghost note, and rejects any edit producing an out-of-range MIDI pitch — each rejection is recorded in `llm_trace.json` with its reason
  4. `GET /v1/artifacts/{job_id}/refine-trace` returns the `llm_trace.json` for a refined job, and `GET /v1/artifacts/{job_id}/lilypond` returns the LilyPond source — both endpoints 404 cleanly for jobs that did not run refine
  5. A WebSocket client subscribed to a refined job observes `stage_started` and `stage_completed` events for the `refine` stage in the correct order relative to humanize and engrave
**Plans**: TBD

### Phase 3: Live Anthropic and Frontend Toggle
**Goal**: Flip the service from mocks to live Anthropic (with web search and retry) and expose the opt-in in the Flutter frontend so an end user can submit a real refined job from the upload screen.
**Depends on**: Phase 2
**Requirements**: UX-01, UX-02, UX-03, UX-04, UX-05
**Success Criteria** (what must be TRUE):
  1. A user opens the upload screen, sees an unchecked "Use AI refinement (experimental)" checkbox with a cost/opt-in tooltip, ticks it, submits a song, and receives a PDF whose engrave input was a `RefinedPerformance` produced by a real Anthropic API call that used the configured web-search tool
  2. The progress screen renders the refine stage in the stage list with the same indicator pattern as existing stages, and on `refine_skipped` shows a distinct "Refinement unavailable" badge (not a red error)
  3. A widget test asserts the upload checkbox is `false` by default and that `OhSheetApi.createJob()` forwards `enableRefine` as `enable_refine` in the POST body
  4. A refined job with `stop_reason="pause_turn"` from the real API surfaces as a skip (not an auto-resume) and the user still receives a working unrefined PDF
  5. Tenacity retries with exponential jitter up to 3 attempts only on transient SDK exceptions; validator failures never retry, and every terminal outcome (success or skip) is captured in `llm_trace.json`
**Plans**: TBD
**UI hint**: yes

### Phase 4: A/B Harness and Baseline
**Goal**: Produce the paired-pipeline A/B harness and the baseline summary that acts as the shipping gate — proving refine does not regress the unrefined path before any default-on rollout can be considered.
**Depends on**: Phase 3
**Requirements**: VAL-01, VAL-02, VAL-03, VAL-04, VAL-05
**Success Criteria** (what must be TRUE):
  1. `scripts/ab_refine.py --dry-run` against `eval/fixtures/refine/manifest.json` (≥10 songs across ≥5 categories) prints a total estimated token cost and exits without making any LLM call
  2. `scripts/ab_refine.py` run for real submits every manifest song through both the refined and unrefined pipelines and writes per-song result files containing both LilyPond sources and the refined diff
  3. The harness produces a diff report that flags: (a) any song where refine's LilyPond fails to compile, (b) any song where refine dropped more than N notes, and (c) any regression against a known-good measurement from the manifest
  4. A baseline summary JSON is committed to `.planning/research/` and a markdown report captures latency, token cost, and regression rate bucketed by manifest category
  5. The manifest exercises all required category coverage: common-key major, hard-key minor, modal, jazz/extended harmony, funk/groove (ghost-note risk), classical polyphony, cover versions, and direct CONCERNS.md artifact-trigger songs
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Contracts and Plumbing | 0/TBD | Not started | - |
| 2. Refine Service and Pipeline Integration | 0/TBD | Not started | - |
| 3. Live Anthropic and Frontend Toggle | 0/TBD | Not started | - |
| 4. A/B Harness and Baseline | 0/TBD | Not started | - |
