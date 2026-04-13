# Oh Sheet

## What This Is

Oh Sheet is an automated pipeline that converts songs — submitted as MP3, MIDI, or YouTube link — into playable piano sheet music. A 5-stage backend (Ingest → Transcribe → Arrange → Humanize → Engrave) renders the score; a Flutter frontend handles upload, live progress, and download. This milestone (GAU-105) adds a new LLM-powered `refine` stage that grounds generated transcriptions against what the song actually is, fixing rhythm and notation artifacts before they hit the engraver.

## Core Value

Sheet music that a pianist can actually read and play — accuracy of the rendered notation matters more than speed, fidelity to the audio waveform, or feature breadth. If the transcription is unplayable, nothing else matters.

## Requirements

### Validated

<!-- Inferred from existing code on main as of 2026-04-13. -->

- ✓ User submits a YouTube URL and receives a piano sheet PDF — existing
- ✓ User uploads an MP3 audio file and receives a piano sheet PDF — existing
- ✓ User uploads a MIDI file and receives a piano sheet PDF — existing
- ✓ User streams live job progress events via WebSocket — existing
- ✓ User downloads generated artifacts (PDF, MusicXML, humanized MIDI, transcription MIDI) — existing
- ✓ Pipeline auto-selects the variant (full / audio_upload / midi_upload / sheet_only) based on input — existing
- ✓ Per-stage Celery dispatch with claim-check blob storage between stages — existing
- ✓ Cross-platform Flutter frontend (Chrome, iOS, Android, macOS web/native) — existing
- ✓ Production deployment via GCP Cloud Run with Caddy reverse proxy and Redis broker — existing
- ✓ Versioned releases with semantic-release and pubspec-driven version footer — existing
- ✓ End-to-end pipeline integration test (GAU-59) — existing

### Active

<!-- GAU-105 — LLM Engraving Refinement milestone scope. -->

- [ ] User can opt into LLM refinement on a per-job basis from the upload screen
- [ ] New `refine` Celery stage runs between `humanize` and `engrave` (and after `arrange` for the `sheet_only` variant)
- [ ] Refine grounds the generated PianoScore against the actual song using Anthropic Claude + web search
- [ ] Refine fixes rhythm/notation correctness — quantization snapping, enharmonic spelling, hand-split, beam grouping
- [ ] Refine operates in modify-and-delete authority only (cannot fabricate notes that weren't transcribed)
- [ ] Refine output is written to blob store as a separate artifact (`jobs/{job_id}/refine/output.json`) for inspection and replay
- [ ] PipelineRunner skips refine and proceeds to engrave on LLM failure — job succeeds with unrefined output and emits a `refine_skipped` event
- [ ] WebSocket JobEvent stream surfaces `stage_started` / `stage_progress` / `stage_completed` for the new stage
- [ ] Frontend exposes a refine toggle on the upload screen and renders refine progress on the progress screen
- [ ] PipelineConfig and JobCreateRequest plumbed end-to-end with `enable_refine: bool` (default false)
- [ ] Anthropic API key configured via `OHSHEET_*` env var; refine fails fast at startup if enabled but key missing
- [ ] A/B test harness runs N reference songs through with/without refine, captures LilyPond output diffs, flags regressions

### Out of Scope

- LLM response caching — same song re-submitted still re-calls the LLM. Revisit when cost or repeat-submit traffic justifies it.
- Streaming partial refine results — refine returns the whole refined score atomically; no in-LLM progress events.
- Self-hosted / local LLM — Anthropic API only for v1; no Ollama / vLLM / OpenAI / multi-provider abstraction.
- Multi-language / non-English songs — English-language songs first; web-search prompts and metadata handling target English. CJK/Spanish/etc. later.
- Eval rubric + golden-set scoring — A/B harness is the v1 validation tool; structured rubric and scored golden set come in a later milestone.
- LLM adding notes — modify+delete authority only; no note fabrication. Prevents the LLM from inventing notes that weren't in the source audio.
- Default-on rollout — refine is opt-in until quality is proven on the A/B harness; no auto-enable for all jobs.
- Cross-cutting fixes from the codebase concerns audit (job persistence, auth, rate limiting, blob path traversal hardening, etc.) — handled in separate milestones; this one stays focused on refine.

## Context

- Brownfield project. Codebase mapped in `.planning/codebase/` on 2026-04-13. Existing 5-stage pipeline ships working transcriptions, but the audit (CONCERNS.md) catalogs known artifacts: voice-cap silently drops notes >2 per hand, hardcoded humanization seed, default-fallback keys/meters, ghost notes from Basic Pitch, weird enharmonic spellings, awkward beaming. These are exactly the class of problems the refine stage is meant to clean up.
- Schema v3.0.0 Pydantic contracts at every pipeline boundary (`shared/shared/contracts.py`). The refine stage may extend `PianoScore` with provenance fields or introduce a `RefinedPianoScore` subtype — to be decided in plan-phase.
- Celery + Redis for stage dispatch; claim-check via the `BlobStore` protocol; per-stage tasks read input by URI, write output by URI. The refine stage slots in cleanly via `PipelineRunner.STEP_TO_TASK` and the per-variant execution plans.
- WebSocket fan-out via `JobEvent` (already wired through `JobManager`). Adding a stage requires emitting events with the `refine` stage name and updating the frontend's stage-name → label map.
- Anthropic Claude (Sonnet 4.6 default; Opus available) is the chosen LLM. Web search is enabled via Anthropic's tool-use API. The team has prior familiarity and the issue specifies Anthropic.
- Linear tracking: GAU-105 — https://linear.app/gauntletai-kevin/issue/GAU-105/llm-engraving-refinement (assigned to jack.jiang, In Progress, branch `feat/GAU-105-llm-engraving-refinement`).
- Branch already exists with no code yet — only the codebase mapping commit. Fresh start for the refine implementation.

## Constraints

- **Tech stack**: Python 3.10+, FastAPI, Celery, Pydantic v2, Anthropic SDK. Refine must be a Celery task wrapping a service, reading input from blob storage by URI and writing output by URI — same pattern as every other stage.
- **External dependency**: Anthropic API. Subject to rate limits, model deprecation, and outages. Mitigated by skip-on-failure semantics.
- **Schema compatibility**: Refine output must be deserializable by the existing `engrave` stage. Either keep `PianoScore` shape and document semantic differences, or introduce a `RefinedPianoScore` discriminated subtype — engrave must accept both.
- **Performance**: LLM round trip + web search adds an estimated 5–30s per job. Acceptable because refine is opt-in; default flow latency unchanged.
- **Cost**: Each refine call = 1+ web-search-augmented LLM call. No caching in v1; per-song cost is the cost. Acceptable for the milestone; revisit if usage spikes.
- **Auth & secrets**: `ANTHROPIC_API_KEY` via `OHSHEET_*` env var (`OHSHEET_ANTHROPIC_API_KEY` to fit the existing convention) loaded by `pydantic-settings`; never logged; same handling as other secrets in `backend/config.py`.
- **Toggle UX**: `enable_refine: bool` field on `JobCreateRequest`, defaulting to `false`. Frontend adds a checkbox on the upload screen. No new variant string; the existing variants stay deterministic.
- **Failure semantics**: If refine fails after configured retries, the runner emits `refine_skipped` and proceeds to engrave on the unrefined `HumanizedPerformance`. The job's final status is `succeeded`, not `failed`.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Implement refine as a separate Celery stage between humanize and engrave (not inlined into engrave) | Decouples LLM latency/failure profile from deterministic engrave; refine output is a separately cacheable/inspectable artifact; matches existing pipeline pattern (PipelineRunner + claim-check) | — Pending |
| Anthropic Claude with web search tool | Strong music reasoning, native tool-use API, team familiarity. No multi-provider abstraction in v1 | — Pending |
| Modify + delete authority (no note addition) | Prevents the LLM from fabricating notes that weren't in the source audio. Catches ghost-note artifacts while keeping refine grounded in transcription evidence | — Pending |
| Per-job opt-in flag (default off) | Keeps default flow latency and behavior unchanged; users who want LLM polish opt in. Avoids surprise cost / surprise behavior changes for existing users | — Pending |
| Skip refine and continue on LLM failure | Refine is enhancement, not blocker. Job always produces a PDF; user sees a `refine_skipped` event if it didn't run. Prevents Anthropic outages from breaking the product | — Pending |
| A/B harness for validation; defer eval rubric + golden set | Ship the smallest validation tool that catches regressions now. Formalize scoring when the refine output stabilizes and we have enough songs to score against | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-13 after initialization*
