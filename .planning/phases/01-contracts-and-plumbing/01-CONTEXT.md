# Phase 1: Contracts and Plumbing - Context

**Gathered:** 2026-04-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Land the Pydantic v3.1.0 contract surface (`RefineEditOp`, `RefineCitation`, `RefinedPerformance`, schema bump), pipeline config fields (`enable_refine`, Anthropic key handling, kill switch, model allowlist, refine knobs), `JobCreateRequest` opt-in field, and a secret-scan pre-commit hook — with zero pipeline behavior change until a job actually sets `enable_refine=True`. No Anthropic SDK dep, no refine worker, no prompt code, no frontend work in this phase — those belong to Phases 2–3.

</domain>

<decisions>
## Implementation Decisions

### Schema version strategy
- **D-01:** Hard bump — single module-level `SCHEMA_VERSION = "3.1.0"` in `shared/shared/contracts.py`. The `schema_version` field stays `str` (no `Literal[...]` tightening). Old 3.0.0 payloads still validate because the field is untyped; matches the codebase's current looseness.
- **D-02:** Leave existing test fixtures (`tests/fixtures/scores/*.json`) at `"schema_version": "3.0.0"`. They serve as regression guards that 3.0.0 payloads remain round-trippable after the bump. New `RefinedPerformance` fixtures use `"3.1.0"`.
- **D-03:** Direct `SCHEMA_VERSION` importers (`backend/api/routes/health.py`, `backend/main.py`, `tests/test_stages.py`) auto-pick up the new value — no per-site code changes required.

### payload_type discriminator shape
- **D-04:** Extend the existing if/elif chain in `backend/workers/engrave.py:22-32` with a third branch for `payload_type == "RefinedPerformance"`. Do NOT refactor to a Pydantic discriminated-union envelope. Minimal churn, matches Phase 1's "no behavior change" goal. Producer side (`backend/jobs/runner.py:306-324`) stays a plain dict envelope; just needs a fourth case once the runner also starts emitting `RefinedPerformance` inputs (Phase 2).
- **D-05:** `RefinedPerformance` is **nested composition** over `HumanizedPerformance`. Not subclass (semantically wrong — refined is NOT a humanized-in-place; edits are a separate log). Not parallel-both-copies (doubles wire size for no Phase-1 benefit).
- **D-06:** Contract shape:
  ```python
  class RefinedPerformance(BaseModel):
      schema_version: str = SCHEMA_VERSION  # "3.1.0"
      refined_performance: HumanizedPerformance  # POST-edit result (what engrave renders)
      edits: list[RefineEditOp]
      citations: list[RefineCitation]
      model: str
      source_performance_digest: str  # SHA of pre-edit input, for drift detection
  ```
- **D-07:** Engrave worker unwrap is `payload = refined.refined_performance`; the rest of the engrave path proceeds identically to the HumanizedPerformance branch. Unknown `payload_type` still raises `ValueError` with the expanded list of valid tags.

### Claude's Discretion

Areas the user explicitly left for Claude to pick. Planner may revisit during Phase 1 planning; Phase 2 planner may refine based on empirical prompt behavior.

- **`RefineEditOp.rationale` enum values** — CTR-01 requires a closed enum but doesn't list values. Claude's default closed domain: `harmony_correction`, `ghost_note_removal`, `octave_correction`, `voice_leading`, `duplicate_removal`, `out_of_range`, `timing_cleanup`, `velocity_cleanup`, `other`. `harmony_correction` is already referenced by STG-05's ghost-note guard; `other` kept as a safety valve so prompt evolution doesn't require a schema bump. Phase 2 may tighten after prompt-quality data is in.
- **`RefineEditOp` edit-payload field shape** — CTR-01 says "edit payload fields" without listing them. Claude's default: for `op="modify"`, optional fields `pitch: int | None = None`, `velocity: int | None = None` (absolute — replaces existing), `velocity_offset: int | None = None` (additive — adds to existing, mirrors ExpressiveNote), `timing_offset_ms: float | None = None`, `duration_beat: float | None = None`. Only provided fields are applied. For `op="delete"`, no extra fields (`target_note_id` is sufficient). Validator enforces at-least-one-field on modify.
- **`RefineCitation.confidence` type** — float in `[0.0, 1.0]` matching `QualitySignal.overall_confidence`, `Note.confidence`, `ScoreChordEvent.confidence` convention. Required, not optional.
- **`source_performance_digest` algorithm** — `hashlib.sha256(performance.model_dump_json().encode("utf-8")).hexdigest()` with `pydantic`'s default canonical JSON (`mode="json"` + sorted keys via `sort_keys=True`). Deterministic, hexadecimal, 64-char.
- **Pre-commit secret-scan tool (CFG-08)** — pick `detect-secrets` over `gitleaks`. Reasons: (1) Python-native, so it plugs into the repo's existing uv-managed toolchain via `pre-commit-hooks` without an extra binary install, (2) maintains a committed `.secrets.baseline` file that future legitimate fixtures/example-keys can be whitelisted against, (3) runs fast enough on this repo's working-set size. Install via `pre-commit-hooks` under a new `.pre-commit-config.yaml`; include `detect-secrets-hook` with `--baseline .secrets.baseline`.
- **`OHSHEET_REFINE_KILL_SWITCH` visibility (CFG-06)** — silent strip from the execution plan (matches "behave exactly as if enable_refine=false"), plus a single `log.warning("refine kill switch active; stripping refine from plan for job_id=%s", job_id)` emitted once per job so ops can diff actual-vs-requested behavior from structured logs. No HTTP error, no stage_completed event — if refine is never in the plan, there's no stage to emit for.
- **Model allowlist shape (CFG-05)** — private module-level constant in `backend/config.py`:
  ```python
  _ALLOWED_REFINE_MODELS_SONNET = frozenset({"claude-sonnet-4-6"})
  _ALLOWED_REFINE_MODELS_OPUS = frozenset({"claude-opus-4-6"})
  ```
  Enforced via a `pydantic.model_validator(mode="after")` on `Settings` that checks `refine_model` against `_ALLOWED_REFINE_MODELS_SONNET ∪ (_ALLOWED_REFINE_MODELS_OPUS if refine_allow_opus else frozenset())`. Rejection at config load time per CFG-05 success criterion.

### Folded Todos
(none — no project-level todos were captured for this phase)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements and roadmap
- `.planning/REQUIREMENTS.md` — v1 Requirement IDs `CTR-01..CTR-05`, `CFG-01..CFG-08` (Phase 1 scope). Also "Out of Scope" section lists capabilities explicitly punted (LLM caching, multi-provider, default-on rollout, etc.) — downstream agents must not reintroduce them.
- `.planning/ROADMAP.md` §"Phase 1: Contracts and Plumbing" — phase goal, depends-on chain, 5 success criteria (schema round-trip, 400-on-missing-key, kill-switch equivalence, model allowlist rejection, gitignored .env + pre-commit scan).
- `.planning/STATE.md` — current project position; Phase 1 is active, 0/TBD plans.

### Pydantic contract surface (insertion points)
- `shared/shared/contracts.py` — single contracts module; `SCHEMA_VERSION` constant at line 15, existing contracts §1–§5 + PipelineConfig at the bottom. Append new `RefineEditOp`, `RefineCitation`, `RefinedPerformance` classes here (not a new file).
- `backend/contracts.py` — re-export module; new symbols must be added to the `from shared.contracts import (...)` list.

### Pipeline plumbing (insertion points)
- `backend/config.py` — `Settings(BaseSettings)` with `env_prefix="OHSHEET_"`; append new CFG-* fields here, plus a `model_validator` for the allowlist.
- `backend/api/routes/jobs.py` §`JobCreateRequest` (lines 29-57) and `create_job()` (lines 82-157) — append `enable_refine: bool = False` to the request body, thread into `PipelineConfig`, add the HTTP 400 pre-check for `enable_refine and not settings.anthropic_api_key`.
- `backend/jobs/runner.py` §`STEP_TO_TASK` (lines 45-53) — register `"refine": "refine.run"` now so the runner dispatch chain is wired even though no `refine.run` task exists yet. The runner's if/elif stage dispatcher (lines 253-330) does NOT need a refine branch in Phase 1 (no variant includes refine).
- `backend/workers/engrave.py` (lines 22-32) — three-way if/elif on `payload_type`; add the `"RefinedPerformance"` branch per D-07.
- `shared/shared/contracts.py` §`PipelineConfig.get_execution_plan` (lines 347-365) — extend to insert `"refine"` between `"humanize"` and `"engrave"` (or after `"arrange"` for `sheet_only`) when `enable_refine=True`.

### Test surface
- `tests/fixtures/scores/*.json` — 15 fixture JSONs with hardcoded `"schema_version": "3.0.0"` (DO NOT migrate per D-02).
- `tests/test_stages.py`, `backend/api/routes/health.py`, `backend/main.py` — direct `SCHEMA_VERSION` importers; no code changes needed (D-03).

### Security / secret-scan
- `.gitignore` — already excludes `.env` (one line). Verify present; no change needed.
- `.pre-commit-config.yaml` — file does NOT exist yet; this phase creates it with a `detect-secrets-hook` entry and commits an initial `.secrets.baseline`.

### Missing / out of scope
- `.planning/PROJECT.md` — referenced by `STATE.md` line 5 but does not exist. Creation is out of scope for Phase 1.
- `.planning/research/SUMMARY.md`, `.planning/codebase/ARCHITECTURE.md` — referenced by REQUIREMENTS.md preamble but do not exist. Out of scope; requirements themselves are self-contained enough for Phase 1.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`pydantic.BaseModel` + `Field(..., ge=X, le=Y)` idiom** — the whole contract file uses this; new Refine* contracts follow the same style (no generic `Any` types, no bare `str`).
- **`Literal[...]` for closed enums** — used for `op`, `variant`, `source`, `type`, `hand`, `payload_type`. Use `Literal["modify", "delete"]` for `RefineEditOp.op` per CTR-01.
- **`SecretStr` from pydantic** — not yet used in this codebase for the API key; CFG-03 introduces it. Precedent is setting `redis_url: str` (plain); Settings fields that hold credentials should use `SecretStr | None` and call `.get_secret_value()` only in the service that needs the raw value (per CFG-03).
- **`model_dump(mode="json")` round-trip pattern** — already the default everywhere (`jobs.py`, `runner.py`, `engrave.py`). CTR-05 round-trip tests should use this idiom.
- **`backend/config.py` field_validator patterns** (lines 462-499) — precedent for validators on unit ratios, probabilities, positive floats. Model-allowlist validator follows the same style with `@model_validator(mode="after")`.

### Established Patterns
- **`pydantic-settings>=2.1`** already in `pyproject.toml` — no new dep for CFG fields.
- **`env_prefix="OHSHEET_"`** means `OHSHEET_ANTHROPIC_API_KEY` maps to `anthropic_api_key: SecretStr | None` on `Settings`. All new knobs follow this convention.
- **`extra="ignore"`** in `SettingsConfigDict` — unknown env vars are silently dropped. No regression risk from new fields.
- **Existing `str` field for `schema_version`** — hard bump is a one-line constant change. No downstream parse site tightens validation today.
- **Plain dict envelopes for stage IO** — `runner.py` constructs dicts, workers validate the payload field against a specific Pydantic contract. Extend the pattern, don't replace it (D-04).

### Integration Points
- `PipelineConfig.get_execution_plan()` is the single authoritative execution-plan builder. CFG-01's refine-step insertion and CFG-06's kill switch both live here.
- `create_job()` in `jobs.py` is the single job-submission entry; CFG-02 and CFG-04 both hook here.
- `Settings` in `backend/config.py` is the single settings class; all CFG-* fields live here.
- `backend/workers/engrave.py` `run()` is the single engrave entry point; D-07's unwrap lives here.

### Things Phase 1 Does NOT Touch
- `backend/services/refine.py`, `backend/workers/refine.py`, Anthropic SDK dep — Phase 2.
- `backend/api/routes/artifacts.py` `refine-trace` / `lilypond` endpoints (INT-06, INT-07) — Phase 2.
- `frontend/lib/*`, `OhSheetApi.createJob()` — Phase 3.
- `.planning/PROJECT.md`, `.planning/research/*`, `.planning/codebase/*` — out of scope.

</code_context>

<specifics>
## Specific Ideas

- **"Wrapper, not subclass"** — user explicitly avoided subclass-inheritance for `RefinedPerformance` because "refined IS-A humanized" is semantically wrong: the RefinedPerformance models a refinement EVENT (with edits, citations, model provenance), not a new kind of HumanizedPerformance. Composition preserves that boundary.
- **"Hard bump, fixtures stay at 3.0.0"** — the user's intent is that the version field documents schema intent for producers but is not a validation gate for consumers. Existing fixtures prove backward-compat without Literal enforcement.
- **"Minimal churn"** — repeated theme: Phase 1 is a foundation phase, and every decision gravitated toward the smallest-diff-that-works option (if/elif branch, not union refactor; hard constant bump, not coexistence; leave fixtures alone).

</specifics>

<deferred>
## Deferred Ideas

- **Pydantic discriminated-union engrave envelope** — considered and rejected for Phase 1 (too much churn for no Phase-1 behavior gain). Revisit if engrave later needs to accept a fourth or fifth payload_type, or if the string-based dispatch starts causing real bugs.
- **Strict `Literal["3.0.0", "3.1.0"]` on schema_version fields** — considered and rejected; the field's purpose is documentation, not a validation boundary. Revisit if a future milestone needs to fail-fast on cross-version payloads (e.g., payload replay from a migrated archive).
- **Creating `.planning/PROJECT.md`** — referenced by `STATE.md` but missing. Out of scope here; downstream agents should note this gap and may surface it as a standalone planning task when convenient.

</deferred>

---

*Phase: 01-contracts-and-plumbing*
*Context gathered: 2026-04-13*
