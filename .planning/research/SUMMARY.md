# Project Research Summary

**Project:** Oh Sheet — GAU-105 LLM Engraving Refinement
**Domain:** LLM-augmented post-processing stage within a deterministic symbolic music pipeline
**Researched:** 2026-04-13
**Confidence:** HIGH

## Executive Summary

This milestone adds a `refine` stage between `humanize` and `engrave` in Oh Sheet's existing 5-stage Celery pipeline. The stage calls Anthropic Claude (Sonnet 4.6 default) with the web-search tool to ground a generated `HumanizedPerformance` against real-world song metadata — fixing the hardcoded-key/meter defaults, enharmonic misspellings, awkward beaming, and ghost-note artifacts catalogued in the codebase audit. There is no prior art for this exact pattern: end-to-end transcription products (AnthemScore, ScoreCloud, Klangio) treat cleanup as a manual editor task; academic LLM-music work (ChatMusician, NotaGen) addresses generation, not refinement. This milestone is a music-specific application of established LLM application engineering hygiene, not a music ML problem.

The recommended approach borrows from the Cursor two-model "edit trick" pattern: ask the LLM to return a flat list of `RefineEditOp` diffs (modify or delete only), then apply them locally with a validator that enforces modify+delete authority by cross-referencing note IDs against the original. This is safer and cheaper than asking the LLM to rewrite the whole score: shorter output, fewer hallucination surfaces, and a native audit trail. The existing pipeline's claim-check blob storage, Celery prefork dispatch, and `asyncio.run(service.run(...))` worker pattern all accommodate the refine stage with zero architectural changes — it slots in as a new entry in `PipelineRunner.STEP_TO_TASK` and the per-variant execution plans.

The key risks are LLM note fabrication (mitigated by three-layer modify+delete enforcement: schema, prompt, and post-validation ID cross-reference), cost blowout from uncapped retries or accidental Opus selection (mitigated by model allowlist and bounded tenacity budget), and silent quality regression if the A/B harness corpus is too small or unrepresentative (mitigated by a stratified reference manifest covering genre, key complexity, and the known CONCERNS.md artifact categories). Skipping the A/B harness or treating it as optional is the one design choice that would make the entire milestone unshippable in good conscience.

---

## Key Findings

### Recommended Stack

Two net-new dependencies on top of the existing codebase: `anthropic>=0.94.0,<1.0` (official Python SDK, released 2026-04-10, bundles structured outputs, web search, and async client natively) and `tenacity>=8.2,<10` (application-layer retry above the SDK's built-in HTTP retry). No LangChain, no `instructor`, no multi-provider abstraction — all explicitly out of scope per PROJECT.md, and all now superseded for this use case by the SDK's first-party structured outputs GA.

Critical tool-version choice is `web_search_20260209` (supports dynamic filtering on Sonnet 4.6 and Opus 4.6) over `web_search_20250305`. Structured outputs use `client.messages.parse(output_format=RefinedEditOpList)` — not raw tool-use JSON parsing, not free-text JSON prompting. This is the schema-compliance guarantee that makes prompt injection substantially less dangerous and eliminates retry loops for malformed output.

**Core technologies:**
- `anthropic>=0.94.0,<1.0`: Official SDK — structured outputs, web search, async client; no wrapper needed
- `claude-sonnet-4-6`: Default LLM — 1M context, $3/$15 per MTok, full feature support; Opus behind `OHSHEET_REFINE_ALLOW_OPUS` flag only
- `web_search_20260209`: Server-side grounding tool — `max_uses=5` cap, $10/1k searches, dynamic filtering
- `tenacity>=8.2`: Application-layer retry — `wait_exponential_jitter`, `stop_after_attempt(3)`, `retry_if_exception_type`
- `pydantic.SecretStr`: Key protection — `.get_secret_value()` called in exactly one location (`RefineService.__init__`)

### Expected Features

No direct competitor exists doing automated LLM refinement inside a music pipeline. All features come from "how to safely run an LLM inside a deterministic pipeline" literature anchored to CONCERNS.md artifacts this stage is meant to fix.

**Must have (P1 — all map to PROJECT.md Active requirements or known codebase artifacts):**
- Structured output contract (`RefinedEditOpList` Pydantic schema with constrained decoding)
- Modify+delete authority in three layers: schema `Literal["modify","delete"]`, prompt constraint, post-validation ID cross-reference
- Web-search grounding for song identity — fixes hardcoded key/meter defaults
- Skip-on-failure with typed `refine_skipped` event — Anthropic outages must not become product outages
- Opt-in per-job toggle (`enable_refine: bool = False`) plumbed end-to-end Flutter → `PipelineConfig`
- `OHSHEET_ANTHROPIC_API_KEY` via `SecretStr`; fail-fast when refine enabled but key absent
- Stage lifecycle events on WebSocket stream; `llm_trace.json` artifact on every invocation
- Bounded retry (max 3, jittered backoff) + model allowlist + kill switch
- A/B harness diffing LilyPond output across stratified reference corpus

**Should have (P2):**
- `RefinedPerformance` wrapper type carrying `edits`, `citations`, `model`, `quality`, `source_performance_digest`
- Structured `RefineEditOp.rationale` as enum (not free text) to detect over-correction programmatically
- Music21 parse gate before passing to engrave
- Langfuse-ready observability seam (structured logging in v1, pluggable later)

**Defer (v2+):**
- LLM response caching, scored eval rubric, default-on rollout, multi-provider abstraction, self-hosted LLM

### Architecture Approach

Brownfield integration — the refine stage must match the existing pipeline's shape exactly. `RefinedPerformance` is a wrapper type (not a subclass) containing the full `HumanizedPerformance` plus refine-specific provenance. Engrave gets a third `elif payload_type == "RefinedPerformance":` arm that unwraps to the inner performance — 2 lines matching the existing discriminated-dispatch pattern. Skip-on-failure lives in the runner (not the worker) so the runner retains execution-plan control when refine is bypassed.

**Major components:**
1. `RefineService` (`backend/services/refine.py`) — prompt builder, Anthropic call via `asyncio.to_thread`, edit application, output write
2. `RefineValidator` (`backend/services/refine_validate.py`) — three-layer enforcement including ID cross-reference and ghost-note velocity guard
3. `refine` Celery task (`backend/workers/refine.py`) — claim-check wrapper; `asyncio.run(service.run(...))`; writes output + trace URIs
4. `PipelineRunner` modifications (`backend/jobs/runner.py`) — `STEP_TO_TASK["refine"]`; skip-on-failure try/except with `stage_completed` skip event
5. Engrave third-arm update (`backend/workers/engrave.py`) — `payload_type="RefinedPerformance"` unwrap
6. `scripts/ab_refine.py` — standalone CLI script; submits N songs with/without refine; diffs LilyPond output; gates production readiness
7. Frontend modifications — upload checkbox defaulting to false; progress screen with distinct skipped state

### Critical Pitfalls

Research identified 14 pitfalls. Five shape phase boundaries and must be addressed before any real API call:

1. **LLM fabricates notes via phantom note IDs** — Schema `Literal["modify","delete"]` prevents `op: "add"` but not `op: "modify"` targeting a note ID that was never in the source. Only the post-validation ID cross-reference is authoritative. All three layers required in Phase 2; validator must raise before `apply_edits` runs. *Critical.*

2. **Cost blowout: uncapped retries, accidental Opus, `pause_turn` re-submission** — SDK 2x + tenacity 3x = up to 6 billed calls; `pause_turn` re-submitted = unbounded. Fix: model allowlist in config (Phase 1), `stop_reason != "end_turn"` is always a failure (Phase 2), tenacity hard-capped at 3. *Critical.*

3. **LLM "fixes" intentional dissonance, ghost notes, or modal spellings** — Statistical training biases toward common-practice harmony. Mitigations: prompt explicitly names the risk; structured `rationale` enum lets validator reject `"harmony_correction"` category; velocity-floor guard on deletes; genre-stratified A/B harness. *Critical.*

4. **`messages.parse()` schema complexity limit** — `RefinedPerformance → HumanizedPerformance → PianoScore` nesting hits Anthropic's constrained-decoding complexity limit. Primary defense: LLM returns `list[RefineEditOp]` (flat, ~5-50 dicts), not a full score rewrite. Smoke-test schema compilation in Phase 2. *High.*

5. **API key committed to repo or leaked in logs** — `SecretStr` prevents stringification but not committing the raw `.env` value. Phase 1 must include: `.env` in `.gitignore` verified, `gitleaks`/`detect-secrets` pre-commit hook, `SecretStr | None` config type, `.get_secret_value()` in exactly one location. *Critical.*

---

## Implications for Roadmap

### Phase 1: Contracts + Config
**Rationale:** All subsequent phases reference `RefinedPerformance`, `RefineEditOp`, `PipelineConfig.enable_refine`, and `OHSHEET_ANTHROPIC_API_KEY`. Nothing can be written until these types exist. No behavior changes; existing tests stay green.
**Delivers:** `RefineEditOp`, `RefineCitation`, `RefinedPerformance` Pydantic models; `PipelineConfig.enable_refine` with updated `get_execution_plan()` for all four variants; `JobCreateRequest.enable_refine`; `backend/config.py` with `SecretStr` key, model allowlist, kill switch, and all refine knobs; `.gitignore` + pre-commit secret-scan hook.
**Avoids:** API key commit (Pitfall 10), accidental Opus (Pitfall 6).
**Gate:** Contract round-trips; `get_execution_plan()` unit tests for all four variants with `enable_refine=True/False`.

### Phase 2: Service + Worker (Isolated)
**Rationale:** Service must be verified in isolation — including the three-layer validator — before runner wiring. Schema complexity failure or validator logic error must be caught here, not during end-to-end tests.
**Delivers:** `RefineService` with prompt builder, `AsyncAnthropic` + `messages.parse(output_format=RefinedEditOpList)`, edit application; `RefineValidator` with ID cross-reference and velocity-floor guard; `refine` Celery worker; task route registration; schema smoke-test; `llm_trace.json` artifact write.
**Uses:** `anthropic>=0.94.0` (`AsyncAnthropic`, `messages.parse`), `web_search_20260209`, `tenacity`, `stop_reason` guard.
**Avoids:** Note fabrication (Pitfall 1), schema complexity failure (Pitfall 4), LLM over-correction (Pitfall 2), prompt injection (Pitfall 3), observability gap (Pitfall 12).
**Gate:** Unit tests for prompt builder, validator (synthetic "add" rejection, phantom ID rejection), edit applier. Mocked Anthropic integration test returns valid `RefinedPerformance` URI. Schema smoke-test passes.

### Phase 3: Runner Wiring + Engrave Update
**Rationale:** First end-to-end slice. Skip-on-failure semantics must be tested with a real pipeline dispatch. Engrave third-arm and runner wiring are causally linked — land together.
**Delivers:** `runner.py` refine branch with skip-on-failure try/except; `refined_dict` state to engrave; `engrave.py` third arm; `refine_skip_total{reason}` counter; `/v1/artifacts/{id}/refine-trace` endpoint; LilyPond artifact endpoint (needed for Phase 6).
**Avoids:** Silent failure hiding outages (Pitfall 5), Celery pool confusion (Pitfall 14), blob path traversal (Pitfall 14).
**Gate:** Mocked end-to-end: job with `enable_refine=True` produces `EngravedOutput`. Forced-failure test: job status `succeeded`, event stream contains `refine_skipped`, engrave ran on unrefined data.

### Phase 4: Real Anthropic Integration + Live Song Validation
**Rationale:** Mocked tests cannot validate prompt quality, citation extraction, or web-search behavior. Gated behind real `OHSHEET_ANTHROPIC_API_KEY` in dev; CI does not run it.
**Delivers:** `@pytest.mark.integration` tests skipped in CI unless `OHSHEET_RUN_INTEGRATION_TESTS=1`; manual validation on 2-3 real songs; prompt iteration; token metrics to logs; `model_version` captured in trace.
**Avoids:** Rate limit thundering herd (Pitfall 9 — validate concurrency cap), unexpected cost (Pitfall 6 — token logging).
**Gate:** Refine produces plausible output on a real song; citations extracted; latency within 5-30s; no `stop_reason: "max_tokens"` or `"pause_turn"` on normal songs.

### Phase 5: Frontend Toggle
**Rationale:** Backend defaults `enable_refine=false`; frontend provides no user value until backend is fully functional (Phases 1-4).
**Delivers:** Upload screen checkbox (`value: false` enforced by widget test); `enableRefine` threaded through `OhSheetApi.createJob()`; progress screen refine stage label + visually distinct skipped state.
**Avoids:** Accidental default-on (Pitfall 11 — widget test required), invisible skip state (Pitfall 11).
**Gate:** User can toggle refine on upload screen and see events on progress screen; skipped state renders distinctively.

### Phase 6: A/B Harness + Baseline Run
**Rationale:** The shipping gate. Refine is "experimental" until the harness demonstrates net-positive or non-regressing quality on a stratified corpus.
**Delivers:** `scripts/ab_refine.py` standalone CLI; `eval/fixtures/refine/manifest.json` with stratified categories (common-key major, hard-key minor, modal, jazz/extended, funk/groove, classical polyphony, cover versions, CONCERNS.md artifact categories); LilyPond diff report; `refine_regression_rate{category}` reporting; dry-run cost estimator; baseline committed to `.planning/research/`.
**Avoids:** Unrepresentative test set (Pitfall 13).
**Gate:** Reproducible diff report across ≥10 songs covering ≥5 categories; human-reviewed sample per category; baseline committed.

### Phase Ordering Rationale

- Contracts before service: `RefineEditOp` and `RefinedPerformance` are referenced in service signatures.
- Service before runner: `runner.py` dispatches `"refine.run"` — task must exist and be testable before runner wiring.
- Engrave update with runner wiring: `payload_type="RefinedPerformance"` is only exercised once the runner produces one; causally linked.
- Mocked Anthropic (Phases 1-3) before real Anthropic (Phase 4): real API incurs cost and has external availability; all CI tests use a fake/injectable client.
- Frontend last: backend defaults `enable_refine=false`; frontend enables users, does not build the feature.
- A/B harness last: needs a stable end-to-end pipeline to be meaningful.

### Research Flags

**Phases needing deeper research during planning:**
- **Phase 2 (Prompt engineering):** The exact system prompt content for modify+delete authority, CONCERNS.md artifact descriptions, and web-vs-performance authority framing requires empirical iteration. Research defines what the prompt must accomplish and prohibit, not the final wording.
- **Phase 4 (Account configuration):** `web_search_20260209` dynamic filtering requires code-execution tool enabled on the Anthropic Console. Verify account configuration before Phase 4; fallback to `web_search_20250305` if not enabled.
- **Phase 6 (Corpus selection):** Stratified manifest categories are defined by research; specific song selections require domain judgment and coverage of known CONCERNS.md artifact patterns.

**Phases with standard patterns (research not required):**
- **Phase 1:** Pydantic v2 contract extension, `SecretStr` settings, `get_execution_plan()` variant logic are all well-documented with established codebase conventions.
- **Phase 3:** Runner try/except skip semantics and engrave third-arm dispatch follow existing Celery patterns exactly.
- **Phase 5:** Flutter toggle widget is standard; only non-obvious requirement is the `value: false` widget test.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All picks verified against official Anthropic docs as of 2026-04-13. SDK version, model IDs, pricing, tool API, structured outputs GA, Python compatibility confirmed from primary sources. |
| Features | MEDIUM | No direct competitor; feature list synthesized from LLM-pipeline engineering literature and PROJECT.md constraints, not comparable product benchmarks. P1 features well-grounded; P2/P3 involves judgment. |
| Architecture | HIGH | Integration decisions grounded in existing codebase (read directly). Existing worker pattern, discriminated engrave dispatch, and PipelineRunner shape are confirmed working implementations. |
| Pitfalls | HIGH | Critical pitfalls verified against Anthropic docs, SDK changelog, academic music-LLM papers, community postmortems, and codebase CONCERNS.md cross-reference. |

**Overall confidence:** HIGH

### Gaps to Address

- **`RefinedPerformance` vs `RefinedPianoScore` naming:** ARCHITECTURE.md recommends `RefinedPerformance` wrapping `HumanizedPerformance`. Plan phase must finalize this name and propagate it consistently through contracts, workers, and engrave dispatch.
- **`RefinedEditOpList` exact schema:** STACK.md and ARCHITECTURE.md both prescribe emitting `list[RefineEditOp]` rather than a full rewritten score. Exact Pydantic model name and field layout are not finalized; decide in plan phase and validate compilability in Phase 2.
- **`web_search_20260209` account prerequisite:** Dynamic filtering requires code-execution tool enabled in the Anthropic Console. Verify account configuration before Phase 4; do not gate CI on it.
- **LilyPond artifact endpoint:** The existing artifact endpoint serves PDF, MIDI, and MusicXML — not LilyPond. Phase 6 harness requires LilyPond output for diffing. Plan must include this as a Phase 3 addition.
- **A/B harness song selection:** Specific songs for the stratified manifest require domain judgment; plan phase should identify who is responsible for that selection and against which CONCERNS.md artifact categories they must cover.

---

## Sources

### Primary (HIGH confidence)
- https://platform.claude.com/docs/en/about-claude/models/overview — model IDs, pricing, context windows, deprecations
- https://platform.claude.com/docs/en/build-with-claude/tool-use/web-search-tool — `web_search_20260209` parameters, pricing, dynamic filtering
- https://platform.claude.com/docs/en/build-with-claude/structured-outputs — `messages.parse()` API, GA status, Pydantic integration, compatibility matrix
- https://pypi.org/project/anthropic/ + GitHub releases — v0.94.0 confirmed current (2026-04-10)
- https://tenacity.readthedocs.io/ — `wait_exponential_jitter`, async support
- Existing codebase `backend/workers/*.py` — `asyncio.run(service.run(...))` pattern confirmed across 7 workers
- Existing codebase `shared/shared/contracts.py` — `PianoScore`, `HumanizedPerformance`, schema version confirmed

### Secondary (MEDIUM confidence)
- https://arxiv.org/html/2402.16153v1 (ChatMusician) — LLM music hallucination rates; >90% ABC parse success baseline
- https://arxiv.org/html/2502.18008v5 (NotaGen) — CLaMP-DPO refinement pattern; generation-focused but structurally similar
- https://waleedk.medium.com/the-edit-trick-efficient-llm-annotation-of-documents-d078429faf37 — edit-diff output pattern rationale
- https://langfuse.com/docs/observability/overview — Langfuse as observability layer
- https://github.com/EleutherAI/lm-evaluation-harness — A/B harness design precedent
- Anthropic research on prompt injection defenses — 1% attack success rate baseline

### Tertiary (LOW confidence)
- https://arxiv.org/abs/2509.25694v2 (HNote) — hexadecimal encoding for music LLMs; very recent, limited citations
- Community postmortems on `pause_turn` handling — `pydantic-ai` issue #2600; confirmed behavioral pattern, not official docs

---

*Research completed: 2026-04-13*
*Ready for roadmap: yes*
