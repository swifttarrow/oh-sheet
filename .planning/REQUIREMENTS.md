# Requirements — GAU-105 LLM Engraving Refinement

Scope for the `refine` Celery stage milestone. Derived from `PROJECT.md` Active requirements, anchored to research findings in `.planning/research/SUMMARY.md`, and grounded in the existing pipeline architecture (`.planning/codebase/ARCHITECTURE.md`).

REQ-ID format: `[CATEGORY]-[NUMBER]`. Categories: CFG (config + plumbing), CTR (contracts), STG (refine stage logic), INT (pipeline integration), UX (frontend), VAL (validation harness).

---

## v1 Requirements

### Contracts (CTR)

- [ ] **CTR-01**: `RefineEditOp` Pydantic model with `Literal["modify","delete"]` op field, `target_note_id`, structured `rationale` enum (not free text), and edit payload fields
- [ ] **CTR-02**: `RefineCitation` Pydantic model capturing web-search source URL, snippet, and confidence
- [ ] **CTR-03**: `RefinedPerformance` wrapper Pydantic model containing the full `HumanizedPerformance` plus `edits: list[RefineEditOp]`, `citations: list[RefineCitation]`, `model: str`, `source_performance_digest: str`
- [ ] **CTR-04**: `payload_type` discriminator on engrave input accepts `"RefinedPerformance"` alongside existing `"PianoScore"` and `"HumanizedPerformance"`
- [ ] **CTR-05**: All new contracts round-trip via `model_dump(mode="json")` / `model_validate()` and carry schema version `v3.1.0`

### Config & Plumbing (CFG)

- [ ] **CFG-01**: `PipelineConfig.enable_refine: bool = False` field; `get_execution_plan()` inserts `"refine"` between `"humanize"` and `"engrave"` (or after `"arrange"` for `sheet_only`) when true
- [ ] **CFG-02**: `JobCreateRequest.enable_refine: bool = False` field threaded through to `PipelineConfig` in `backend/api/routes/jobs.py`
- [ ] **CFG-03**: `OHSHEET_ANTHROPIC_API_KEY: SecretStr | None` setting in `backend/config.py`; `.get_secret_value()` called only inside `RefineService.__init__`
- [ ] **CFG-04**: Job submission with `enable_refine=True` fails fast with HTTP 400 if `OHSHEET_ANTHROPIC_API_KEY` is unset
- [ ] **CFG-05**: `OHSHEET_REFINE_MODEL` (default `claude-sonnet-4-6`) + `OHSHEET_REFINE_ALLOW_OPUS: bool = False` model allowlist; reject any model not in the allowlist
- [ ] **CFG-06**: `OHSHEET_REFINE_KILL_SWITCH: bool = False` global disable flag; when true, refine acts as if `enable_refine=False` regardless of per-job setting
- [ ] **CFG-07**: Refine knobs: `OHSHEET_REFINE_MAX_TOKENS`, `OHSHEET_REFINE_WEB_SEARCH_MAX_USES` (default 5), `OHSHEET_REFINE_MAX_RETRIES` (default 3) all loaded via `pydantic-settings`
- [ ] **CFG-08**: `.gitignore` verified to exclude `.env`; pre-commit secret-scan hook (`gitleaks` or `detect-secrets`) added to repo

### Refine Stage (STG)

- [ ] **STG-01**: `backend/services/refine.py` `RefineService.run(performance, metadata) -> RefinedPerformance` async method using Anthropic Python SDK ≥0.94 with `messages.parse(output_format=RefinedEditOpList)` structured output
- [ ] **STG-02**: Refine prompt grounds the LLM against song metadata (title + composer + any ingest-time tags) and explicitly forbids note addition
- [ ] **STG-03**: Web search tool (`web_search_20260209`, fallback to `web_search_20250305`) wired into the Anthropic call; `max_uses` bounded by `OHSHEET_REFINE_WEB_SEARCH_MAX_USES`
- [ ] **STG-04**: `backend/services/refine_validate.py` `RefineValidator` enforces modify+delete authority via three layers: schema (already enforced by `Literal`), prompt (already in STG-02), and post-validation that cross-references every `target_note_id` against the source performance's note ID set
- [ ] **STG-05**: Validator rejects edits with `rationale="harmony_correction"` on velocity-floor notes (groove ghost-note guard) and any edit that would produce an out-of-range MIDI pitch
- [ ] **STG-06**: `RefineService` applies validated edits to a copy of the source performance and returns the resulting `RefinedPerformance` — never mutates the input
- [ ] **STG-07**: Tenacity retry wrapper: `wait_exponential_jitter`, `stop_after_attempt(3)`, retry on transient SDK exceptions only; never retry validator failures
- [ ] **STG-08**: `stop_reason != "end_turn"` is treated as failure (raises `RefineLLMError`); `pause_turn` is never auto-resumed
- [ ] **STG-09**: `backend/workers/refine.py` Celery task wraps the service: deserializes input from blob URI, calls `asyncio.run(service.run(...))`, writes `output.json` and `llm_trace.json` to blob store, returns output URI
- [ ] **STG-10**: `llm_trace.json` artifact captures: prompt, raw LLM response, citations, applied edits, rejected edits with reason, model + version, total tokens, cost estimate

### Pipeline Integration (INT)

- [ ] **INT-01**: `PipelineRunner.STEP_TO_TASK["refine"] = "refine.run"` registered; per-variant execution plans (`full`, `audio_upload`, `midi_upload`, `sheet_only`) updated to include refine when `enable_refine=True`
- [ ] **INT-02**: Engrave worker (`backend/workers/engrave.py`) handles `payload_type="RefinedPerformance"` by unwrapping to the inner performance and proceeding identically
- [ ] **INT-03**: `PipelineRunner` wraps the refine dispatch in try/except; on exception, emits `stage_completed` with `message="refine_skipped: <reason>"` (NOT `stage_failed`), captures `refine_skip_total{reason}` counter, and continues to engrave with the unrefined `HumanizedPerformance`
- [ ] **INT-04**: Job final status is `succeeded` even when refine is skipped; user receives a working PDF
- [ ] **INT-05**: `JobEvent` stream emits `stage_started`, `stage_progress` (optional), and `stage_completed` for the refine stage, fanned out via existing `JobManager` WebSocket pub/sub
- [ ] **INT-06**: `GET /v1/artifacts/{job_id}/refine-trace` endpoint added to `backend/api/routes/artifacts.py` for downloading `llm_trace.json`
- [ ] **INT-07**: `GET /v1/artifacts/{job_id}/lilypond` endpoint added (needed by the A/B harness to diff LilyPond source rather than rendered PDFs)

### Frontend (UX)

- [ ] **UX-01**: Upload screen renders a "Use AI refinement (experimental)" checkbox, default unchecked, with a tooltip explaining cost and opt-in nature
- [ ] **UX-02**: `OhSheetApi.createJob()` accepts `enableRefine: bool` and forwards it as `enable_refine` in the POST body
- [ ] **UX-03**: Progress screen renders the refine stage in the stage list with the same indicator pattern as other stages
- [ ] **UX-04**: When refine is skipped (via `refine_skipped` event), the progress screen shows a visually distinct "Refinement unavailable" badge on the refine stage (not a red error state)
- [ ] **UX-05**: Widget test asserts the upload checkbox defaults to `false` (regression guard against accidental default-on)

### Validation Harness (VAL)

- [ ] **VAL-01**: `scripts/ab_refine.py` standalone CLI submits a list of songs through both pipelines (with and without refine) and writes per-song result files
- [ ] **VAL-02**: A/B harness produces a diff report comparing the LilyPond source from each variant — flags songs where refine produced output that fails to compile, drops more than N notes, or produces a regression in a known-good measurement
- [ ] **VAL-03**: `eval/fixtures/refine/manifest.json` reference manifest covers ≥10 songs spanning ≥5 categories: common-key major, hard-key minor, modal, jazz/extended harmony, funk/groove (ghost-note risk), classical polyphony, cover versions, and direct CONCERNS.md artifact-trigger songs
- [ ] **VAL-04**: Harness emits a baseline summary JSON (committed to `.planning/research/`) and a markdown report (latency, token cost, regression rate per category)
- [ ] **VAL-05**: Dry-run mode estimates total cost from token counts before kicking off any LLM call

---

## v2 Requirements

(Deferred for now — revisit after v1 ships and A/B harness data is in.)

- LLM response caching keyed on `(song fingerprint, prompt version, model)`
- Eval rubric + scored golden set (formal scoring beyond LilyPond diff)
- Default-on rollout once A/B harness shows non-regressing quality
- Streaming partial refine results inside the LLM call
- Multi-provider abstraction (OpenAI, local) — pluggable LLM backend
- Multi-language / non-English song support (web search + prompts)
- LLM-suggested article additions (notation hints, fingering, tempo markings) under a separate authority flag
- Langfuse or similar LLM observability platform integration (current: structured logs only)

## Out of Scope

<!-- Explicit boundaries with reasoning to prevent re-adding. -->

- **LLM response caching** — Same song re-submitted re-calls the LLM. Re-evaluate when cost or repeat-submit traffic justifies a cache layer. Adds complexity (cache invalidation on prompt version) that does not earn its keep until usage pattern data exists.
- **Streaming partial refine results** — Refine returns the whole refined score atomically. Streaming inside an LLM call adds significant complexity (event coalescing, partial-state handling) for marginal UX gain when the typical refine call is 5–30s.
- **Self-hosted / local LLM** — Anthropic API only for v1. Avoids GPU infrastructure, model-hosting overhead, and a multi-provider abstraction. Re-evaluate if cost or latency at scale becomes prohibitive.
- **Multi-language / non-English songs** — Web-search prompts and metadata handling target English first. CJK/Spanish/etc. need prompt translation, potentially different web sources, and validation across languages — meaningful additional scope.
- **Eval rubric + golden-set scoring** — A/B harness (LilyPond diff + regression flagging) is the v1 validation tool. Structured rubric + scored golden set requires a stable refine output and meaningful corpus to evaluate against; defer until both exist.
- **LLM adding notes** — Modify+delete authority only. Note addition allows the LLM to fabricate content not in the source audio, defeating the "ground in real transcription" property. Re-evaluate only with a separate authority flag and explicit user opt-in.
- **Default-on rollout** — Refine is opt-in until the A/B harness demonstrates non-regressing quality. Default-on without that gate would silently increase cost and surface LLM artifacts to users who didn't ask for them.
- **Cross-cutting fixes from CONCERNS.md audit** — Job persistence, auth, rate limiting, blob path traversal hardening, etc., are explicitly out of this milestone's scope. Handled in separate milestones; refine focuses on the LLM stage.
- **OpenAI / multi-provider support** — Anthropic only. Adding a provider abstraction layer up front is premature; no second provider is on the roadmap.
- **A real PDF rendering fallback** when LilyPond is missing — Out of scope. The existing 60-byte stub PDF behavior is unchanged; refine does not change engrave's PDF rendering path.

## Traceability

<!-- Filled by gsd-roadmapper; maps each REQ-ID to its phase. -->

| Requirement | Phase | Status |
|-------------|-------|--------|
| CTR-01 | Phase 1 | Pending |
| CTR-02 | Phase 1 | Pending |
| CTR-03 | Phase 1 | Pending |
| CTR-04 | Phase 1 | Pending |
| CTR-05 | Phase 1 | Pending |
| CFG-01 | Phase 1 | Pending |
| CFG-02 | Phase 1 | Pending |
| CFG-03 | Phase 1 | Pending |
| CFG-04 | Phase 1 | Pending |
| CFG-05 | Phase 1 | Pending |
| CFG-06 | Phase 1 | Pending |
| CFG-07 | Phase 1 | Pending |
| CFG-08 | Phase 1 | Pending |
| STG-01 | Phase 2 | Pending |
| STG-02 | Phase 2 | Pending |
| STG-03 | Phase 2 | Pending |
| STG-04 | Phase 2 | Pending |
| STG-05 | Phase 2 | Pending |
| STG-06 | Phase 2 | Pending |
| STG-07 | Phase 2 | Pending |
| STG-08 | Phase 2 | Pending |
| STG-09 | Phase 2 | Pending |
| STG-10 | Phase 2 | Pending |
| INT-01 | Phase 2 | Pending |
| INT-02 | Phase 2 | Pending |
| INT-03 | Phase 2 | Pending |
| INT-04 | Phase 2 | Pending |
| INT-05 | Phase 2 | Pending |
| INT-06 | Phase 2 | Pending |
| INT-07 | Phase 2 | Pending |
| UX-01 | Phase 3 | Pending |
| UX-02 | Phase 3 | Pending |
| UX-03 | Phase 3 | Pending |
| UX-04 | Phase 3 | Pending |
| UX-05 | Phase 3 | Pending |
| VAL-01 | Phase 4 | Pending |
| VAL-02 | Phase 4 | Pending |
| VAL-03 | Phase 4 | Pending |
| VAL-04 | Phase 4 | Pending |
| VAL-05 | Phase 4 | Pending |

**Coverage:** 40/40 v1 REQ-IDs mapped to exactly one phase. No orphans.

---

*Last updated: 2026-04-13 after roadmap creation*
