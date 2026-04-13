# Phase 1: Contracts and Plumbing - Research

**Researched:** 2026-04-13
**Domain:** Pydantic v2.12 contract surface + pydantic-settings 2.13 env plumbing + detect-secrets pre-commit + Nyquist validation architecture
**Confidence:** HIGH (all claims either verified by local tool runs against this repo's `.venv` or by reading the checked-in codebase; WebSearch used only to cross-check detect-secrets current version)

## Summary

Phase 1 is a foundation-only diff: Pydantic contract additions (`RefineEditOp`, `RefineCitation`, `RefinedPerformance`), a `SCHEMA_VERSION` bump from `"3.0.0"` → `"3.1.0"`, six new `OHSHEET_*` Settings fields (one `SecretStr`, one kill switch, one allowlist-validated model name, one opt-in bool, three refine knobs), one `JobCreateRequest` opt-in bool, one `if/elif` branch extension in `backend/workers/engrave.py`, one `PipelineConfig.get_execution_plan()` insertion gated on `enable_refine`, and a new `.pre-commit-config.yaml` + `.secrets.baseline` pair using `detect-secrets` v1.5.0 (the only tool verified by WebSearch). Every piece has an established pattern in the repo to mirror — so the research focus is not "how to do any of it" but "how to prove the zero-behavior-change invariant at test time." The stack is already here; Phase 1 adds rows to existing tables.

The Nyquist / validation architecture is the load-bearing part of this research. All five ROADMAP success criteria compile into automated tests under `tests/test_refine_*.py` (new module family), plus a parametrize-over-glob regression guard at `tests/test_contracts_roundtrip_regression.py` that re-validates every committed `tests/fixtures/scores/*.json` after the SCHEMA_VERSION bump. Zero-behavior-change is proven by a **byte-equal execution-plan snapshot test**: pre-bump values for `variant ∈ {full, audio_upload, midi_upload, sheet_only}` are captured as literal expected lists in `tests/test_pipeline_config.py` and must remain identical when `enable_refine=False`. The pre-commit hook is verified **end-to-end** by a subprocess-run test that `git init`s a throwaway repo, writes a file containing `sk-ant-api03-MOCKKEY...`, stages it, invokes `pre-commit run detect-secrets --files <path>` and asserts non-zero exit.

**Primary recommendation:** Don't hand-roll anything. Mirror the existing `backend/config.py` `field_validator` style for allowlist enforcement (use `model_validator(mode='after')`), mirror the existing `tests/test_engrave_quality.py` parametrize-over-FIXTURE_NAMES pattern for fixture regression, mirror the existing `tests/test_pipeline_config.py` literal-expected-list style for execution-plan byte-equality, and use detect-secrets v1.5.0 with `--baseline` (Python-native, plugs into `pre-commit` 4.5.1 already installed globally).

## Project Constraints (from CLAUDE.md)

`./CLAUDE.md` is gitignored per `.gitignore` line 18 (`CLAUDE.md`) and is NOT present in the checked-out working copy. No actionable project-level directives to lift. The `.claude/skills/` and `.agents/skills/` directories do not exist. All constraints below come from CONTEXT.md (locked decisions D-01..D-07 + Claude's Discretion picks) and the existing codebase patterns.

## User Constraints (from CONTEXT.md)

### Locked Decisions

**Schema version strategy**
- **D-01:** Hard bump — single module-level `SCHEMA_VERSION = "3.1.0"` in `shared/shared/contracts.py`. The `schema_version` field stays `str` (no `Literal[...]` tightening). Old 3.0.0 payloads still validate because the field is untyped; matches the codebase's current looseness.
- **D-02:** Leave existing test fixtures (`tests/fixtures/scores/*.json`) at `"schema_version": "3.0.0"`. They serve as regression guards that 3.0.0 payloads remain round-trippable after the bump. New `RefinedPerformance` fixtures use `"3.1.0"`.
- **D-03:** Direct `SCHEMA_VERSION` importers (`backend/api/routes/health.py`, `backend/main.py`, `tests/test_stages.py`) auto-pick up the new value — no per-site code changes required.

**payload_type discriminator shape**
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

- **`RefineEditOp.rationale` enum values** — Claude's default closed domain: `harmony_correction`, `ghost_note_removal`, `octave_correction`, `voice_leading`, `duplicate_removal`, `out_of_range`, `timing_cleanup`, `velocity_cleanup`, `other`. `harmony_correction` is already referenced by STG-05's ghost-note guard; `other` kept as a safety valve so prompt evolution doesn't require a schema bump. Phase 2 may tighten after prompt-quality data is in.
- **`RefineEditOp` edit-payload field shape** — For `op="modify"`, optional fields `pitch: int | None = None`, `velocity: int | None = None` (absolute — replaces existing), `velocity_offset: int | None = None` (additive — adds to existing, mirrors ExpressiveNote), `timing_offset_ms: float | None = None`, `duration_beat: float | None = None`. Only provided fields are applied. For `op="delete"`, no extra fields (`target_note_id` is sufficient). Validator enforces at-least-one-field on modify.
- **`RefineCitation.confidence` type** — float in `[0.0, 1.0]` matching `QualitySignal.overall_confidence`, `Note.confidence`, `ScoreChordEvent.confidence` convention. Required, not optional.
- **`source_performance_digest` algorithm** — `hashlib.sha256(performance.model_dump_json().encode("utf-8")).hexdigest()` with `pydantic`'s default canonical JSON (`mode="json"` + sorted keys via `sort_keys=True`). Deterministic, hexadecimal, 64-char.
- **Pre-commit secret-scan tool (CFG-08)** — pick `detect-secrets` v1.5.0 over `gitleaks`. Python-native, baseline-friendly, runs fast enough on this repo's working-set size.
- **`OHSHEET_REFINE_KILL_SWITCH` visibility (CFG-06)** — silent strip from the execution plan (matches "behave exactly as if enable_refine=false"), plus a single `log.warning("refine kill switch active; stripping refine from plan for job_id=%s", job_id)` emitted once per job so ops can diff actual-vs-requested behavior from structured logs. No HTTP error, no stage_completed event.
- **Model allowlist shape (CFG-05)** — private module-level constants in `backend/config.py`:
  ```python
  _ALLOWED_REFINE_MODELS_SONNET = frozenset({"claude-sonnet-4-6"})
  _ALLOWED_REFINE_MODELS_OPUS = frozenset({"claude-opus-4-6"})
  ```
  Enforced via `pydantic.model_validator(mode="after")` on `Settings`. Rejection at config load time per CFG-05 success criterion.

### Deferred Ideas (OUT OF SCOPE)

- **Pydantic discriminated-union engrave envelope** — considered and rejected for Phase 1. Revisit if engrave later needs a fourth or fifth payload_type.
- **Strict `Literal["3.0.0", "3.1.0"]` on schema_version fields** — considered and rejected; the field's purpose is documentation, not a validation boundary.
- **Creating `.planning/PROJECT.md`** — referenced by STATE.md but missing. Out of scope here.
- **Anthropic SDK integration, prompt engineering** — Phase 2.
- **LilyPond trace endpoint, refine-trace artifact** — Phase 2.
- **Frontend OhSheetApi changes** — Phase 3.
- **LLM caching, multi-provider support, default-on rollout** — deferred entirely per REQUIREMENTS.md "Out of Scope" section.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CTR-01 | `RefineEditOp` Pydantic model | Standard Stack row #1 (Pydantic 2.12.5); example below; D-06 rationale enum from Claude's Discretion |
| CTR-02 | `RefineCitation` Pydantic model | Example below; confidence float `[0.0, 1.0]` matches existing `QualitySignal.overall_confidence` convention |
| CTR-03 | `RefinedPerformance` wrapper | D-05/D-06 nested composition; example below |
| CTR-04 | `payload_type` discriminator accepts `"RefinedPerformance"` | D-04 if/elif extension at `backend/workers/engrave.py:22-32`; test in `Validation Architecture` row 4 |
| CTR-05 | Round-trip via `model_dump(mode="json")` / `model_validate()` with `v3.1.0` | Verified locally: `model_dump_json()` + `sort_keys=True` is deterministic byte-for-byte across round-trips (tool run in `Determinism Proofs`) |
| CFG-01 | `PipelineConfig.enable_refine: bool = False`; execution plan inserts refine | Insertion point: `shared/shared/contracts.py:347-365`. Execution-plan tests in `tests/test_pipeline_config.py` already exist — extend parametrize matrix |
| CFG-02 | `JobCreateRequest.enable_refine` threaded to `PipelineConfig` | Insertion point: `backend/api/routes/jobs.py:29-57` + `82-157`. Mirror existing `prefer_clean_source` flag wiring |
| CFG-03 | `OHSHEET_ANTHROPIC_API_KEY: SecretStr | None` | SecretStr masking verified by local tool run (all 6 serialization paths mask; only `.get_secret_value()` reveals) |
| CFG-04 | HTTP 400 when `enable_refine=True` without key | `create_job` pre-check before `PipelineConfig` construction; FastAPI `HTTPException(400, detail=...)` |
| CFG-05 | Model allowlist with `OHSHEET_REFINE_ALLOW_OPUS` gate | `model_validator(mode="after")` pattern verified by local tool run — rejects both bare opus and arbitrary models at `Settings()` instantiation |
| CFG-06 | `OHSHEET_REFINE_KILL_SWITCH` global disable | Location: `PipelineConfig.get_execution_plan()` — plus `log.warning` per Claude's Discretion |
| CFG-07 | Refine knobs (max_tokens, web_search_max_uses default 5, max_retries default 3) | Plain `int` Settings fields — no validators beyond what Pydantic's type coercion already gives |
| CFG-08 | `.gitignore` verified + pre-commit hook | `.env` already in `.gitignore` line 7 (verified by tool read). `.pre-commit-config.yaml` + `.secrets.baseline` files do NOT exist — this phase creates both. detect-secrets v1.5.0 is current [VERIFIED: GitHub Yelp/detect-secrets `.pre-commit-hooks.yaml`] |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pydantic | 2.12.5 | Contract models (`RefineEditOp`, `RefineCitation`, `RefinedPerformance`), field validation, `SecretStr` | [VERIFIED: local `.venv/bin/python -c 'import pydantic; print(pydantic.VERSION)'`] Already listed in `pyproject.toml` as `pydantic>=2.5`. The entire contract surface and all config validators use it today. |
| pydantic-settings | 2.13.1 | `BaseSettings` with `env_prefix="OHSHEET_"`, `.env` file, `extra="ignore"` | [VERIFIED: local `.venv`] Already listed as `pydantic-settings>=2.1`. `backend/config.py` is built on it. |
| pytest | 9.0.3 | Unit + integration testing framework | [VERIFIED: local `.venv`] Already listed as `pytest>=8.0` (optional-dependencies.dev). |
| pytest-asyncio | ≥0.24 | Async test support (used by `test_jobs.py`, WebSocket tests) | Already in pyproject.toml dev extras. |
| httpx | ≥0.26 | `TestClient` under the hood for FastAPI test rigs | Already in pyproject.toml dev extras. |
| pre-commit | 4.5.1 | Hook runner | [VERIFIED: `command -v pre-commit`] Already installed globally via homebrew — no repo-level dependency needed |
| detect-secrets | 1.5.0 | Secret-scanning pre-commit hook with `--baseline .secrets.baseline` | [VERIFIED: GitHub Yelp/detect-secrets master `.pre-commit-hooks.yaml` at v1.5.0] Python-native (plugs into existing uv toolchain), committed baseline pattern matches CFG-08's whitelist requirement |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| hashlib (stdlib) | — | `sha256().hexdigest()` for `source_performance_digest` | Required by Claude's Discretion choice on digest algorithm; stdlib — no install |
| json (stdlib) | — | `json.dumps(..., sort_keys=True, separators=(',', ':'))` for deterministic canonical form feeding into SHA | Used in combination with `model_dump(mode="json")` — verified deterministic locally (see Determinism Proofs) |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| detect-secrets | gitleaks | Go binary — single executable, faster on very large monorepos. Loses the baseline-file whitelist pattern (gitleaks uses `.gitleaksignore` globs instead of hash-matched baseline). Adds a non-Python binary to dev prereqs. CONTEXT D-choice already picked detect-secrets. |
| `model_validator(mode="after")` | `field_validator` on the model-level tuple | `field_validator` cannot read sibling fields (`refine_allow_opus`) at validation time. `model_validator` is the documented pattern for cross-field checks. [CITED: pydantic docs.pydantic.dev/latest/concepts/validators/] |
| `SecretStr` | plain `str \| None` | Plain `str` leaks into logs, repr, model_dump_json, and tracebacks. SecretStr masks in all 6 tested paths (see local tool verification below). CFG-03 makes SecretStr mandatory. |
| Hard SCHEMA_VERSION bump | coexistence (`SCHEMA_VERSION_3_0_0`, `SCHEMA_VERSION_3_1_0`) | Adds two constants where one is sufficient. Breaks D-03 (direct importers auto-pick-up). User explicitly chose hard bump; the `schema_version` field is `str` (no Literal), so old payloads still validate. |

### Installation

No new Python dependencies in `pyproject.toml` are required. Pre-commit is already globally installed. The only new setup:

```bash
# 1. Install the detect-secrets hook repo (pre-commit auto-fetches on first run)
pre-commit install

# 2. Create the initial baseline (captures currently-allowed "secret-shaped" strings)
detect-secrets scan > .secrets.baseline
# Note: detect-secrets is invoked by pre-commit; it does not need to be
# installed into the project .venv. pre-commit creates its own isolated env.
```

### Version verification

Verified against local `.venv` and registry:
- `pydantic 2.12.5` (published [ASSUMED — not verified against npm/PyPI in this session; constraint in pyproject is `>=2.5`])
- `pydantic_settings 2.13.1` (constraint `>=2.1`)
- `pytest 9.0.3` (constraint `>=8.0`)
- `pre-commit 4.5.1` (homebrew-installed globally; not in pyproject)
- `detect-secrets 1.5.0` [VERIFIED: Yelp/detect-secrets repo current tag — WebSearch hit confirms latest]

## Architecture Patterns

### Recommended Project Structure (insertion points only — no new directories)

```
shared/shared/contracts.py           # +3 classes (RefineEditOp, RefineCitation, RefinedPerformance)
                                     # SCHEMA_VERSION constant "3.0.0" → "3.1.0"
                                     # PipelineConfig gains enable_refine: bool = False
                                     # PipelineConfig.get_execution_plan() extended for refine insertion
backend/contracts.py                 # +3 re-exports added to the `from shared.contracts import (...)` list
backend/config.py                    # +6 Settings fields + 1 model_validator + 2 module-level frozensets
backend/api/routes/jobs.py           # +1 field on JobCreateRequest, +1 400-precheck in create_job, +1 PipelineConfig field wiring
backend/workers/engrave.py           # +1 elif branch on payload_type
backend/jobs/runner.py               # +1 entry in STEP_TO_TASK (no dispatch branch — no variant includes refine in Phase 1)
.pre-commit-config.yaml              # NEW — detect-secrets hook entry
.secrets.baseline                    # NEW — initial detect-secrets baseline
tests/test_contracts_refine.py       # NEW — CTR-01..CTR-05 round-trip, rationale enum, at-least-one-field validator
tests/test_contracts_roundtrip_regression.py  # NEW — parametrize-over-glob guard for 3.0.0 fixtures
tests/test_settings_refine.py        # NEW — CFG-03..CFG-05 Settings-level tests (SecretStr, allowlist, knobs)
tests/test_jobs_refine.py            # NEW — CFG-02, CFG-04 HTTP 400 + CFG-06 kill switch via real JobCreateRequest
tests/test_pipeline_config.py        # EXTEND — existing file; add enable_refine parametrize rows + byte-equal baseline
tests/test_precommit_hook.py         # NEW — subprocess-run detect-secrets against mock Anthropic key
```

### Pattern 1: Contract model with closed enum + optional edit-payload fields

**What:** Define `RefineEditOp` mirroring the existing `DynamicMarking` / `Articulation` idiom — `Literal["..."]` for the closed set, `Field(..., ge=..., le=...)` for bounded floats, optional fields with `None` default, `model_validator` for cross-field invariants.

**When to use:** CTR-01 `RefineEditOp` definition.

**Example:**

```python
# Source: shared/shared/contracts.py — mirrors Articulation, DynamicMarking, ExpressiveNote patterns
from pydantic import BaseModel, Field, model_validator
from typing import Literal

RefineRationale = Literal[
    "harmony_correction",
    "ghost_note_removal",
    "octave_correction",
    "voice_leading",
    "duplicate_removal",
    "out_of_range",
    "timing_cleanup",
    "velocity_cleanup",
    "other",
]


class RefineEditOp(BaseModel):
    op: Literal["modify", "delete"]
    target_note_id: str
    rationale: RefineRationale
    # Modify-only fields — all optional; validator enforces at-least-one for op="modify"
    pitch: int | None = Field(default=None, ge=0, le=127)
    velocity: int | None = Field(default=None, ge=0, le=127)       # absolute
    velocity_offset: int | None = Field(default=None, ge=-30, le=30)  # additive (mirrors ExpressiveNote)
    timing_offset_ms: float | None = Field(default=None, ge=-50.0, le=50.0)
    duration_beat: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _validate_modify_has_at_least_one_field(self) -> "RefineEditOp":
        if self.op == "modify":
            if all(v is None for v in (
                self.pitch, self.velocity, self.velocity_offset,
                self.timing_offset_ms, self.duration_beat,
            )):
                raise ValueError(
                    "RefineEditOp with op='modify' must provide at least one of: "
                    "pitch, velocity, velocity_offset, timing_offset_ms, duration_beat"
                )
        return self


class RefineCitation(BaseModel):
    url: str
    snippet: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class RefinedPerformance(BaseModel):
    schema_version: str = SCHEMA_VERSION  # "3.1.0" after bump
    refined_performance: HumanizedPerformance
    edits: list[RefineEditOp]
    citations: list[RefineCitation]
    model: str
    source_performance_digest: str  # 64-char sha256 hex
```

### Pattern 2: Settings with cross-field `model_validator` for allowlist

**What:** Pydantic v2's `@model_validator(mode="after")` receives the fully-populated `self` and can read any field. This is the canonical way to enforce a predicate spanning two fields (`refine_model` must be in the set determined by `refine_allow_opus`).

**When to use:** CFG-05 model allowlist.

**Example:**

```python
# Source: backend/config.py — mirrors existing field_validator style but at model-level
from pydantic import model_validator
from pydantic_settings import BaseSettings

_ALLOWED_REFINE_MODELS_SONNET: frozenset[str] = frozenset({"claude-sonnet-4-6"})
_ALLOWED_REFINE_MODELS_OPUS: frozenset[str] = frozenset({"claude-opus-4-6"})


class Settings(BaseSettings):
    # ... (existing fields) ...

    refine_model: str = "claude-sonnet-4-6"
    refine_allow_opus: bool = False

    @model_validator(mode="after")
    def _validate_refine_model_allowlist(self) -> "Settings":
        allowed = _ALLOWED_REFINE_MODELS_SONNET
        if self.refine_allow_opus:
            allowed = allowed | _ALLOWED_REFINE_MODELS_OPUS
        if self.refine_model not in allowed:
            raise ValueError(
                f"OHSHEET_REFINE_MODEL={self.refine_model!r} is not in the allowlist. "
                f"Allowed: {sorted(allowed)}. "
                f"Set OHSHEET_REFINE_ALLOW_OPUS=true to permit Opus models."
            )
        return self
```

[VERIFIED: local tool run — `model_validator(mode="after")` correctly raises `ValidationError` at `Settings()` instantiation for `refine_model='gpt-4'` and for `refine_model='claude-opus-4-6'` without `refine_allow_opus=True`.]

### Pattern 3: `get_execution_plan` extension — insert refine between humanize and engrave

**What:** Extend the existing `PipelineConfig.get_execution_plan()` method (`shared/shared/contracts.py:347-365`) so that when `enable_refine=True` (and `kill_switch=False`), the plan gains a `"refine"` step after `"humanize"` (or after `"arrange"` for the `sheet_only` variant which has no humanize).

**When to use:** CFG-01 + CFG-06.

**Example:**

```python
# Source: shared/shared/contracts.py — extends existing method
class PipelineConfig(BaseModel):
    variant: PipelineVariant
    skip_humanizer: bool = False
    stage_timeout_sec: int = 600
    score_pipeline: ScorePipelineMode = "arrange"
    enable_refine: bool = False  # NEW

    def get_execution_plan(self) -> list[str]:
        """Return the list of stages to invoke in order, per the variant."""
        routing: dict[str, list[str]] = {
            "full":         ["ingest", "transcribe", "arrange", "humanize", "engrave"],
            "audio_upload": ["ingest", "transcribe", "arrange", "humanize", "engrave"],
            "midi_upload":  ["ingest", "arrange", "humanize", "engrave"],
            "sheet_only":   ["ingest", "transcribe", "arrange", "engrave"],
        }
        plan = list(routing[self.variant])
        if self.skip_humanizer and "humanize" in plan:
            plan.remove("humanize")
        if self.score_pipeline == "condense_transform":
            try:
                idx = plan.index("arrange")
            except ValueError:
                pass
            else:
                plan[idx : idx + 1] = ["condense", "transform"]
        # NEW — refine insertion after humanize, or after arrange for sheet_only
        if self.enable_refine:
            if "humanize" in plan:
                idx = plan.index("humanize")
                plan.insert(idx + 1, "refine")
            else:
                # sheet_only (no humanize) — refine goes after arrange (or after transform)
                for anchor in ("transform", "arrange"):
                    if anchor in plan:
                        plan.insert(plan.index(anchor) + 1, "refine")
                        break
        return plan
```

**Kill-switch note:** The kill switch lives on `Settings`, not on `PipelineConfig`. The check runs in `backend/api/routes/jobs.py` `create_job` — when `settings.refine_kill_switch=True`, the incoming `JobCreateRequest.enable_refine` is **silently coerced to False** before `PipelineConfig` is constructed (plus `log.warning` per Claude's Discretion). This keeps `PipelineConfig.get_execution_plan()` pure and deterministic: given the same PipelineConfig, same plan — always. The kill switch does not need to be inside `get_execution_plan()`.

### Pattern 4: Engrave worker elif branch extension

**What:** Add a third case to the `if/elif/else` chain at `backend/workers/engrave.py:22-32`. `RefinedPerformance` unwraps to its inner `refined_performance` field (a `HumanizedPerformance`) and the rest of the engrave path is unchanged.

**When to use:** CTR-04 + D-07.

**Example:**

```python
# Source: backend/workers/engrave.py:22-32 extension
if payload_type == "HumanizedPerformance":
    payload = HumanizedPerformance.model_validate(payload_data)
elif payload_type == "PianoScore":
    payload = PianoScore.model_validate(payload_data)
elif payload_type == "RefinedPerformance":
    refined = RefinedPerformance.model_validate(payload_data)
    payload = refined.refined_performance
else:
    raise ValueError(
        f"Unknown payload_type: {payload_type!r}. "
        f"Expected 'HumanizedPerformance', 'PianoScore', or 'RefinedPerformance'."
    )
```

Note: the import list at the top of the file must gain `RefinedPerformance` from `shared.contracts`.

### Pattern 5: `pytest.mark.parametrize` over a glob of fixture files

**What:** Mirror the existing `tests/test_engrave_quality.py:83` pattern — import `FIXTURE_NAMES` from `tests/fixtures/__init__.py`, parametrize the regression test over every committed `*.json`, load via `load_score_fixture(name)`, assert round-trip identity and `schema_version == "3.0.0"`.

**When to use:** Regression guard that 3.0.0 payloads remain parseable after the SCHEMA_VERSION bump (success-criterion regression for CTR-05).

**Example:**

```python
# Source: tests/test_contracts_roundtrip_regression.py (NEW) — mirrors test_engrave_quality.py:83
import json
from pathlib import Path

import pytest

from backend.contracts import HumanizedPerformance, PianoScore
from tests.fixtures import FIXTURE_NAMES, load_score_fixture

_HUMANIZED = {"humanized_with_offsets", "humanized_with_expression"}


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_3_0_0_fixture_still_roundtrips_after_bump(name: str):
    """Pre-Phase-1 fixtures at schema_version=3.0.0 must still validate.

    After SCHEMA_VERSION is bumped to 3.1.0, the field stays type `str`
    (no Literal tightening per D-02), so old payloads are still accepted.
    Re-validating and re-serializing must produce the same JSON —
    canary against accidental schema changes on field shapes.
    """
    # Load + re-validate
    fixture = load_score_fixture(name)
    # The committed JSON carries "3.0.0" — it must survive validation
    assert fixture.schema_version == "3.0.0", (
        f"fixture {name} lost its 3.0.0 tag — D-02 says fixtures stay at 3.0.0"
    )
    # Round-trip must be byte-identical (key-order-preserving on Pydantic's side)
    raw = json.loads((Path("tests/fixtures/scores") / f"{name}.json").read_text())
    dumped = fixture.model_dump(mode="json")
    assert dumped == raw, (
        f"round-trip mismatch for fixture {name} — schema drift "
        f"in a non-schema_version field"
    )
```

This catches any CTR-* contract change that silently alters a field's shape (e.g., tightening `Literal`, renaming a field, adding a required field with no default).

### Anti-Patterns to Avoid

- **Do not** call `PipelineConfig.get_execution_plan()` on a config that's had `enable_refine=True` coerced to False by the kill switch — call the coercion first, then construct the PipelineConfig. Mixing the two leaves the plan's source of truth ambiguous between request body + settings.
- **Do not** put the kill-switch check inside `get_execution_plan()` itself. The method has no handle on `settings`. Keep the method pure; coerce at the route.
- **Do not** tighten `schema_version: str` to `Literal["3.0.0", "3.1.0"]` or `Literal["3.1.0"]`. D-02 explicitly keeps this loose.
- **Do not** log `settings.anthropic_api_key.get_secret_value()` anywhere — call `.get_secret_value()` only inside `RefineService.__init__` (Phase 2) and never pass the raw string further. CFG-03 is violated by any `log.info("key=%s", key)` that receives the raw string.
- **Do not** refactor the engrave envelope to a `pydantic.Field(discriminator=...)` discriminated union. D-04 explicitly rejects that — it's a Phase 2+ consideration.
- **Do not** regenerate the score fixtures with `python -m tests.fixtures._builders` after the bump. That would replace the committed 3.0.0 JSONs with 3.1.0 ones, destroying the regression guard per D-02.
- **Do not** add `refine` to any `routing` entry in `get_execution_plan()`. Refine is NOT a variant; it's a per-job optional stage gated on `enable_refine`. It must be inserted conditionally, not baked into the per-variant recipe.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Secret key scanning in commits | Custom regex-based git pre-commit hook | `detect-secrets` v1.5.0 with `--baseline .secrets.baseline` | Detect-secrets ships 17+ detector plugins covering AWS, Azure, GitHub, high-entropy-string heuristics. Anthropic keys (`sk-ant-api03-...`) match its high-entropy + keyword plugins. The baseline file provides hash-matched whitelist; custom regex doesn't scale. |
| Masking a secret in logs / repr / serialization | Write a `__repr__` on a custom dataclass | Pydantic `SecretStr` | Verified locally: SecretStr masks through 6 independent serialization paths (str, repr, f-string, model_dump, model_dump_json, %-format). Only `.get_secret_value()` reveals. Hand-rolled masking breaks on any new serialization surface. |
| Canonical JSON for hashing | Hand-roll `json.dumps(x, separators=(',', ':'), sort_keys=True, ensure_ascii=True, default=str)` everywhere | `model.model_dump_json()` then `json.dumps(model.model_dump(mode='json'), sort_keys=True)` | Pydantic's JSON encoder already handles int/float/datetime/etc. consistently. The only extra needed is `sort_keys=True` for deterministic key order. Verified locally: same-input → same SHA. |
| Cross-field `Settings` validation | Raise in `__post_init_post_parse__` or a custom `__init__` | `@model_validator(mode="after")` on `Settings` | Verified locally: pattern rejects both env-var override (`OHSHEET_REFINE_MODEL=gpt-4`) and constructor arg at `Settings()` instantiation. It's the documented Pydantic v2 idiom. |
| Parametrizing a test over filesystem JSON fixtures | `for name in os.listdir(...): def test_xxx()` — loses individual test-case granularity | `@pytest.mark.parametrize("name", FIXTURE_NAMES)` | Mirror `tests/test_engrave_quality.py:83`. Each fixture becomes its own test node so failures show which specific fixture broke. |
| Pre-commit hook execution in CI / testing it works | Writing a test that re-implements detect-secrets's regexes | `subprocess.run(['pre-commit', 'run', 'detect-secrets', '--files', path])` in a pytest | The test must prove the hook blocks an Anthropic key at the git-commit boundary, not that a regex matches. Only subprocess-invoking pre-commit itself tests the integrated path. |

**Key insight:** Every "build" option above duplicates logic that already exists — either in a dependency already in `pyproject.toml` (pydantic, pydantic-settings, pytest) or in a pre-commit hook repo. Phase 1's whole theme is "append rows to existing tables, don't write new frameworks."

## Runtime State Inventory

> This is NOT a rename/refactor/migration phase — it is net-additive foundation work. The section is included for completeness since downstream agents may assume the inventory is always present.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — verified by grep over `backend/storage/` (LocalBlobStore only writes per-job blobs; no cross-job stores) and confirmation that Phase 1 adds no new datastores. | none |
| Live service config | None — the only "service" config is `backend/config.py` Settings, which is env-var driven with `extra="ignore"`. No UI/DB-hosted config. | none |
| OS-registered state | None — no OS-level scheduled tasks, launchd plists, pm2 processes, or systemd units reference anything Phase 1 touches. | none |
| Secrets / env vars | `OHSHEET_ANTHROPIC_API_KEY` is a NEW env var created by this phase (CFG-03). No renames. The env var's value does not need to exist for Phase 1's test suite to pass (the 400-on-missing-key test exercises the absence path, not the presence path). | none — operators set it when they're ready for Phase 2 |
| Build artifacts | None — no `.egg-info`, compiled binaries, or Docker images embed strings that Phase 1 changes. `schema_version` bump only affects in-memory / blob JSON, not artifact filenames. | none |

**The canonical question:** *After every file in the repo is updated, what runtime systems still have the old string cached, stored, or registered?* — Answer: none. The `SCHEMA_VERSION = "3.0.0"` → `"3.1.0"` change is pure producer-side (Python module constant); consumer validation is loose (`schema_version: str`); the 14 committed fixture JSONs are deliberately kept at 3.0.0 as regression guards.

## Common Pitfalls

### Pitfall 1: Regenerating fixtures destroys the regression guard
**What goes wrong:** A developer runs `python -m tests.fixtures._builders` after the bump to "keep fixtures fresh," and now every `scores/*.json` says `"schema_version": "3.1.0"` — the regression guard is gone.
**Why it happens:** `_builders.py:608` calls `model.model_dump_json()` using the current module-level `SCHEMA_VERSION`, which is 3.1.0 after the bump. The regenerator has no "keep old version" knob.
**How to avoid:** (a) Add a comment at the top of `_builders.py` noting that regeneration produces 3.1.0 output and **must not be run** for the 14 existing fixtures post-bump. (b) The regression test's `assert fixture.schema_version == "3.0.0"` fails loudly if a regeneration sneaks in via a PR.
**Warning signs:** A PR diff that touches multiple `tests/fixtures/scores/*.json` with a `"schema_version": "3.0.0"` → `"3.1.0"` change in all of them.

### Pitfall 2: `SecretStr` vs `str` confusion in log statements
**What goes wrong:** A developer writes `log.info("using key %s", settings.anthropic_api_key)` and SecretStr masks it (good!) — but then copies the pattern to a new place with `log.info("using key %s", settings.anthropic_api_key.get_secret_value())`, leaking the raw key.
**Why it happens:** `.get_secret_value()` is the only way to obtain the raw string, and it's needed inside `RefineService.__init__`. The temptation is to call it "once" at boundary and pass raw `str` around.
**How to avoid:** CFG-03 says `.get_secret_value()` is called ONLY inside `RefineService.__init__`. Phase 1 must never call it. The Settings-level test must re-verify after Phase 2 that no other callsite exists.
**Warning signs:** Any `.get_secret_value()` call outside `backend/services/refine.py` (Phase 2 file). Grep-guard acceptable.

### Pitfall 3: Kill switch in the wrong layer
**What goes wrong:** Developer adds `if settings.refine_kill_switch: plan.remove('refine')` inside `PipelineConfig.get_execution_plan()`. But `PipelineConfig` is imported from `shared/shared/contracts.py` — it should have no dependency on `backend/config.py`. And now `get_execution_plan()` is impure (depends on env state).
**Why it happens:** Kill switch feels like a routing decision, and routing lives in `get_execution_plan()`.
**How to avoid:** Kill switch is a **request-admission** decision, not a routing decision. Check it in `create_job` before building `PipelineConfig`. If `settings.refine_kill_switch and body.enable_refine`: coerce to False, log warning, construct PipelineConfig with `enable_refine=False`.
**Warning signs:** Any import of `backend.config.settings` from inside `shared/shared/contracts.py`.

### Pitfall 4: HTTP 400 order-of-operations
**What goes wrong:** The 400-on-missing-key check runs AFTER `PipelineConfig` is constructed, meaning validation errors from other fields surface first. A user with both `enable_refine=true` AND a bad `difficulty="xyz"` field gets a confusing 422 about difficulty instead of the expected 400 about the missing key.
**Why it happens:** FastAPI validates the request body as a Pydantic model BEFORE `create_job` body executes. So all 422s (field-level validation) always fire before any 400 the route raises manually.
**How to avoid:** This is actually desired behavior — 422 for "your request body is malformed," 400 for "your valid request is inconsistent with server config." Document this ordering in the test. The CFG-04 test should send an otherwise-valid `JobCreateRequest` with ONLY `enable_refine=true` set and ONLY the key-absence condition violated.
**Warning signs:** Test that sets both `enable_refine=true` and some unrelated invalid field and expects 400.

### Pitfall 5: `STEP_TO_TASK` having `"refine": "refine.run"` with no registered Celery task
**What goes wrong:** In Phase 1, `STEP_TO_TASK` gains `"refine": "refine.run"` but no worker file registers the `refine.run` task (that's Phase 2). Tests with `task_always_eager=True` that accidentally exercise a refine path get a confusing "task not found" error.
**Why it happens:** `STEP_TO_TASK` is a lookup, and `runner.py:196-210` distinguishes "registered task" from "remote task." In `task_always_eager=True` mode, an unregistered task raises at dispatch time.
**How to avoid:** (a) No Phase 1 test sets `enable_refine=True` during an end-to-end job run — the only `enable_refine=True` tests are at the `create_job` / `PipelineConfig` boundaries, not the full runner. (b) Document in `STEP_TO_TASK` that the refine entry is an "unused-until-Phase-2 reservation."
**Warning signs:** A Phase 1 test that submits a job with `enable_refine=True` and waits for `succeeded`.

### Pitfall 6: `model_dump_json()` default has no `sort_keys`
**What goes wrong:** Developer computes `source_performance_digest` using `hashlib.sha256(performance.model_dump_json().encode()).hexdigest()` and assumes it's deterministic. It IS deterministic for a single Pydantic model instance, but Pydantic's JSON output is **field-declaration-order**, not alphabetical — so a round-tripped model (`model_validate(model_dump())`) may not produce the same bytes as the original if `default_factory` fields were re-initialized.
**Why it happens:** Pydantic preserves field-declaration order by design (readable output) rather than alphabetical.
**How to avoid:** Use the Claude's Discretion prescription: `json.dumps(perf.model_dump(mode="json"), sort_keys=True, separators=(',', ':')).encode("utf-8")`. Verified deterministic across round-trips (see Determinism Proofs). Note: Pydantic's `model_dump_json()` alone is enough if you never round-trip before hashing, BUT `sort_keys=True` is the safer invariant and what CONTEXT's Claude's Discretion chose.
**Warning signs:** `source_performance_digest` values drifting across round-trips in tests.

### Pitfall 7: Test leaks `OHSHEET_*` env vars into other tests
**What goes wrong:** Test A sets `monkeypatch.setenv('OHSHEET_REFINE_MODEL', 'gpt-4')`, calls `Settings()`, expects ValidationError — then Test B (later in the same process) imports `from backend.config import settings` (module-level `Settings()` call), which also sees the `OHSHEET_REFINE_MODEL=gpt-4` env and fails at import time.
**Why it happens:** `backend/config.py:517` runs `settings = Settings()` at module load. Once imported, the `settings` singleton is cached; it's not re-instantiated per-test.
**How to avoid:** (a) Tests that mutate `OHSHEET_*` env vars MUST use `monkeypatch` (auto-cleanup) not raw `os.environ[...] = ...`. (b) Tests that need a fresh `Settings()` instance should construct one in-place (`Settings()` in the test body) rather than re-importing. (c) For the CFG-05 allowlist-rejection test: assert on `Settings()` constructor call, not on `from backend.config import settings` re-import.
**Warning signs:** Flaky test runs where adding a test reorder causes unrelated test failures.

## Code Examples

Verified patterns usable directly in implementation:

### RefineEditOp with modify/delete branching validator

```python
# Source: shared/shared/contracts.py — same file as existing contracts
# Mirrors DynamicMarking / Articulation / ExpressiveNote style
from pydantic import BaseModel, Field, model_validator
from typing import Literal

RefineRationale = Literal[
    "harmony_correction", "ghost_note_removal", "octave_correction",
    "voice_leading", "duplicate_removal", "out_of_range",
    "timing_cleanup", "velocity_cleanup", "other",
]


class RefineEditOp(BaseModel):
    op: Literal["modify", "delete"]
    target_note_id: str
    rationale: RefineRationale
    pitch: int | None = Field(default=None, ge=0, le=127)
    velocity: int | None = Field(default=None, ge=0, le=127)
    velocity_offset: int | None = Field(default=None, ge=-30, le=30)
    timing_offset_ms: float | None = Field(default=None, ge=-50.0, le=50.0)
    duration_beat: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _modify_requires_at_least_one_field(self) -> "RefineEditOp":
        if self.op == "modify" and all(
            v is None for v in (
                self.pitch, self.velocity, self.velocity_offset,
                self.timing_offset_ms, self.duration_beat,
            )
        ):
            raise ValueError(
                "op='modify' requires at least one of: pitch, velocity, "
                "velocity_offset, timing_offset_ms, duration_beat"
            )
        return self
```

### Settings with SecretStr + allowlist validator + kill switch

```python
# Source: backend/config.py — appended below existing fields
# Mirrors existing field_validator style but at model-level per Pydantic v2 docs
from pydantic import SecretStr, model_validator

_ALLOWED_REFINE_MODELS_SONNET: frozenset[str] = frozenset({"claude-sonnet-4-6"})
_ALLOWED_REFINE_MODELS_OPUS: frozenset[str] = frozenset({"claude-opus-4-6"})


class Settings(BaseSettings):
    # ... (existing fields) ...

    # ---- Refine stage ----
    anthropic_api_key: SecretStr | None = None
    refine_model: str = "claude-sonnet-4-6"
    refine_allow_opus: bool = False
    refine_kill_switch: bool = False
    refine_max_tokens: int = 4096
    refine_web_search_max_uses: int = 5
    refine_max_retries: int = 3

    @model_validator(mode="after")
    def _validate_refine_model_allowlist(self) -> "Settings":
        allowed = _ALLOWED_REFINE_MODELS_SONNET
        if self.refine_allow_opus:
            allowed = allowed | _ALLOWED_REFINE_MODELS_OPUS
        if self.refine_model not in allowed:
            raise ValueError(
                f"OHSHEET_REFINE_MODEL={self.refine_model!r} not in allowlist "
                f"{sorted(allowed)}. Set OHSHEET_REFINE_ALLOW_OPUS=true to permit Opus."
            )
        return self
```

### JobCreateRequest + 400 pre-check + kill switch coercion

```python
# Source: backend/api/routes/jobs.py — JobCreateRequest expansion + create_job pre-check
import logging
from fastapi import HTTPException

log = logging.getLogger(__name__)


class JobCreateRequest(BaseModel):
    # ... (existing fields: audio, midi, title, artist, prefer_clean_source, skip_humanizer, difficulty)
    enable_refine: bool = False  # NEW — CFG-02


async def create_job(
    body: JobCreateRequest,
    manager: Annotated[JobManager, Depends(get_job_manager)],
    blob: Annotated[LocalBlobStore, Depends(get_blob_store)],
) -> JobSummary:
    # ... (existing source-signal xor checks) ...

    # NEW — CFG-04 fail-fast: enable_refine requires OHSHEET_ANTHROPIC_API_KEY
    effective_enable_refine = body.enable_refine
    if effective_enable_refine and settings.anthropic_api_key is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "enable_refine=true requires OHSHEET_ANTHROPIC_API_KEY to be set. "
                "Set the env var or submit with enable_refine=false."
            ),
        )

    # NEW — CFG-06 kill switch: silently coerce to False + log once per job
    if effective_enable_refine and settings.refine_kill_switch:
        log.warning(
            "refine kill switch active; stripping refine from plan (job would have run with enable_refine=true)"
        )
        effective_enable_refine = False

    # ... (existing bundle construction) ...

    config = PipelineConfig(
        variant=variant,
        skip_humanizer=body.skip_humanizer,
        score_pipeline=settings.score_pipeline,
        enable_refine=effective_enable_refine,  # NEW
    )
    record = await manager.submit(bundle, config)
    return _record_to_summary(record)
```

### source_performance_digest deterministic computation

```python
# Source: shared/shared/contracts.py — helper near other contract utilities
# Or inline inside RefineService.run in Phase 2
import hashlib
import json


def compute_source_performance_digest(performance: HumanizedPerformance) -> str:
    """Deterministic 64-char hex SHA-256 of a HumanizedPerformance.

    Key ordering via `sort_keys=True` makes the digest stable across
    Python versions, round-trips through Pydantic, and dict-insertion-order
    differences. Verified deterministic by local test run.
    """
    canonical = json.dumps(
        performance.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
```

### .pre-commit-config.yaml

```yaml
# Source: https://github.com/Yelp/detect-secrets#pre-commit-hook (v1.5.0 current)
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.5.0
    hooks:
      - id: detect-secrets
        args:
          - "--baseline"
          - ".secrets.baseline"
        exclude: '^(tests/fixtures/|.*\.ipynb$)'
        # Rationale: exclude score fixtures (high-entropy note-id strings look
        # secret-ish to detect-secrets's entropy plugins) and notebooks
        # (output cells can contain high-entropy strings that aren't secrets).
```

### Initial `.secrets.baseline` creation

```bash
# Run once on the repo to capture current known-safe "secret-shaped" strings.
# Then commit .secrets.baseline alongside .pre-commit-config.yaml.
detect-secrets scan \
  --exclude-files '^(tests/fixtures/|.*\.ipynb$)' \
  > .secrets.baseline
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Pydantic v1 `@validator` / `@root_validator` | Pydantic v2 `@field_validator` / `@model_validator` | Pydantic 2.0 (2023) | This repo already uses v2 (existing `field_validator` calls in `backend/config.py`) — extend with `model_validator(mode="after")` for cross-field checks. |
| Raw `str` for secrets with manual masking | `SecretStr` with `.get_secret_value()` | Built into Pydantic since v1 | Official pydantic documentation prescribes SecretStr for all credentials. Verified locally to mask through every serialization path. |
| Custom regex secret scanning | `detect-secrets` with baseline + plugins | Detect-secrets v1.0 (2019), now at v1.5.0 (2024) | Standard tool for pre-commit secret scanning. Baseline pattern means false positives can be whitelisted without disabling the hook. |
| Dict-based envelopes with manual string dispatch | Pydantic discriminated unions with `Field(discriminator=...)` | Pydantic v2.5+ | Repo still uses dict envelopes (per D-04). Discriminated unions would be the "state of the art" but D-04 defers it. |

**Deprecated / outdated:**
- `@validator(always=True, pre=True)` — Pydantic v1 syntax. Never used in this repo.
- `from pydantic import BaseSettings` — moved to `pydantic_settings` package in Pydantic v2. Repo already uses `from pydantic_settings import BaseSettings`.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Pydantic 2.12.5 and pydantic-settings 2.13.1 installed in `.venv` are the versions that ship with the next `pyproject.toml` resolution. Verified in the CURRENT venv but a future `uv sync` against `pydantic>=2.5` could pull 2.13 or later. | Standard Stack | Low — any Pydantic 2.5+ has `model_validator`, `SecretStr`, `Literal`, and `Field(...)`. Breakage only if upstream drops these, which they haven't signalled. |
| A2 | `sk-ant-api03-` prefix is a high-entropy + keyword hit for detect-secrets' default plugins (Keyword + HighEntropy + Base64). The WebSearch result mentions that Anthropic's pattern is being added to major detection tools but does not confirm detect-secrets itself has a named AnthropicDetector. | Validation Architecture / Pre-commit verification | Medium — if detect-secrets misses `sk-ant-api03-MOCKKEY`, the test in `test_precommit_hook.py` will fail loudly. Mitigation: the test uses a 40-char high-entropy mock that will trip the entropy plugin independent of keyword plugins. If that also misses, fall back to creating a `.secrets.baseline` that explicitly flags the test key as a known secret, then removing the baseline entry to simulate the "new secret" case. |
| A3 | `pre-commit 4.5.1` globally installed (`/opt/homebrew/bin/pre-commit`) will be used by CI. If CI uses a different pre-commit version, the `.pre-commit-config.yaml` format compatibility matters. | Standard Stack | Low — pre-commit config format has been stable since 2.0. |
| A4 | `STEP_TO_TASK["refine"] = "refine.run"` being registered with no corresponding Celery worker task does not break the runner, because no Phase 1 code path invokes it (`enable_refine=True` end-to-end is a Phase 2 test). | Architecture Patterns / Pitfall 5 | Low — confirmed by reading `runner.py:196-210`: the lookup only fires inside the `for step in plan` loop, and `plan` is built from `get_execution_plan()`, which only emits `"refine"` when `enable_refine=True`. Phase 1 tests never construct that plan. |
| A5 | Pytest 9.0.3's `monkeypatch.setenv` fixture cleans up env vars after the test, so CFG-05 allowlist tests don't leak into `backend.config.settings` module singleton in subsequent tests. | Validation Architecture / Pitfall 7 | Low — monkeypatch teardown is guaranteed by pytest contract. Risk is tests that bypass monkeypatch (raw `os.environ`). |
| A6 | `detect-secrets` invoked via `pre-commit run detect-secrets --files <path>` returns non-zero exit when a scanned file contains a new (non-baseline) high-entropy secret-looking string. | Validation Architecture / Pre-commit end-to-end test | Medium — documented behavior per detect-secrets README, but the subprocess test should capture both stdout and exit code and assert on exit code first, then assert on stdout contents as a secondary signal. |
| A7 | The committed fixtures' key-ordering in `tests/fixtures/scores/*.json` matches what `HumanizedPerformance(...).model_dump(mode="json")` produces on round-trip. If Pydantic ever changes its default field ordering rules, the regression test `assert dumped == raw` would fail. | Architecture Patterns / Pattern 5 | Low — Pydantic v2 guarantees field-declaration-order preservation, and the fixtures were generated by `_builders.py` using the same Pydantic version. Risk only if a contract-level field reorder sneaks in. |
| A8 | The claim that `.env` is already in `.gitignore` is verified by reading the file. But if `.env` exists at the repo root AND is untracked AND has a real key, `git status` will still show it — the gitignore prevents accidental staging, not accidental existence. CFG-08 success criterion 5 says `git status` confirms `.env` is gitignored, not that `.env` is absent. | User Constraints / CFG-08 | Low — gitignore is verified (line 7: `.env`). Operators are responsible for not `git add -f .env`. |

## Open Questions

1. **Should `STEP_TO_TASK["refine"] = "refine.run"` be registered in Phase 1 at all, or defer to Phase 2?**
   - What we know: CONTEXT D-07 and canonical_refs say yes, register now.
   - What's unclear: Whether any future Phase 1 test accidentally invokes this path.
   - Recommendation: Register in Phase 1 per CONTEXT, but add an explicit comment in `STEP_TO_TASK` documenting the "unused-until-Phase-2 reservation." Phase 1 tests must NOT submit jobs with `enable_refine=True` (the CFG-04 / CFG-06 tests stop at the `create_job` boundary without running the pipeline).

2. **What fields should the `RefineEditOp` for `op="delete"` carry beyond `target_note_id` + `rationale`?**
   - What we know: Claude's Discretion says no extra fields for delete; target_note_id is sufficient.
   - What's unclear: Whether Phase 2's validator wants to carry a `reason_detail: str | None` for operator diagnostics.
   - Recommendation: Stick with Claude's Discretion for Phase 1. Phase 2 can add optional fields — schema stays backward-compatible because they'd be optional.

3. **Should the allowlist validator run only on `refine_model` or also on `refine_max_tokens` / `refine_web_search_max_uses` / `refine_max_retries` upper bounds?**
   - What we know: CFG-07 says "all loaded via pydantic-settings" but doesn't specify bounds.
   - What's unclear: What's a sensible upper bound for `refine_max_tokens` (Anthropic's API cap is model-dependent).
   - Recommendation: Phase 1 — no upper-bound validator beyond Pydantic's type coercion. Phase 2 may introduce one if empirical testing shows over-large values cause silent Anthropic errors. Defer to Phase 2's empirical discretion.

4. **Does the pre-commit hook need to run in CI, not just on developer machines?**
   - What we know: CFG-08 says the pre-commit hook must block a commit. Success criterion 5 is phrased as "git status ... confirms ... the pre-commit secret-scan hook blocks a commit."
   - What's unclear: Whether CI (GitHub Actions) should also run `pre-commit run --all-files` as a check.
   - Recommendation: Phase 1 scope — create the hook + baseline. CI wiring (adding a `.github/workflows/*.yml` step) is an operational task that can be done in Phase 1 or folded into a later ops phase. Research-wise, I'd recommend adding a CI step because a developer who doesn't run `pre-commit install` locally can still commit via the GitHub UI; but this is a non-requirement relative to the 5 success criteria.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python interpreter | All runtime + tests | ✓ | 3.14.2 (system) / 3.11 (project `.venv`) | — |
| `pydantic` | Contracts, config | ✓ | 2.12.5 (in `.venv`) | — |
| `pydantic-settings` | Config | ✓ | 2.13.1 (in `.venv`) | — |
| `pytest` | Test suite | ✓ | 9.0.3 (in `.venv`) | — |
| `pre-commit` | CFG-08 pre-commit hook | ✓ | 4.5.1 (global homebrew) | Could be added to dev extras of pyproject.toml if a developer has no homebrew, but pre-commit install is standard prereq |
| `detect-secrets` | CFG-08 secret scanner | ✗ | — | Auto-fetched by pre-commit on first `pre-commit run` — no manual install needed. pre-commit manages the hook's own venv |
| `git` | `.gitignore`, pre-commit | ✓ | system-default | — |
| `hashlib`, `json` | Digest computation | ✓ | stdlib | — |
| `anthropic` Python SDK | Phase 2 only (NOT Phase 1) | ✗ | — | N/A — Phase 1 does not import it |
| Running Anthropic API | Phase 3 only | ✗ | — | N/A — Phase 1 does not call it |

**Missing dependencies with no fallback:** None. Phase 1 has zero external service dependencies.

**Missing dependencies with fallback:** `detect-secrets` is "missing" from the project .venv but the pre-commit framework handles its installation. No action needed.

## Validation Architecture

> This is the **load-bearing** section per the user's Nyquist focus. Every ROADMAP success criterion maps to at least one automated check. No criterion is "compilation-only" or "manual-only."

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.0.3 + pytest-asyncio ≥0.24 |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` — `testpaths = ["tests"]`, `asyncio_mode = "auto"`, `addopts = "--cov=backend --cov-report=term-missing --cov-report=xml"` |
| Quick run command | `pytest tests/test_contracts_refine.py tests/test_settings_refine.py tests/test_jobs_refine.py tests/test_contracts_roundtrip_regression.py tests/test_pipeline_config.py tests/test_precommit_hook.py -x -q` |
| Full suite command | `pytest tests/ --cov=backend -x` |
| Phase gate | Full suite + `pre-commit run --all-files` green before `/gsd-verify-work` |

### Success-Criteria-to-Test Map

Each of the 5 ROADMAP success criteria compiles into one or more test functions. No criterion is verified by "it compiles" or "manual check only."

| SC # | Success Criterion (from ROADMAP) | Test Type | File → Function | Automated Command |
|------|-----------------------------------|-----------|-----------------|-------------------|
| SC-1 | Import `RefinedPerformance`, `RefineEditOp`, `RefineCitation`; round-trip via `model_dump(mode="json")`/`model_validate()`; see `v3.1.0` on payload | unit | `tests/test_contracts_refine.py::test_refined_performance_roundtrip_carries_3_1_0` | `pytest tests/test_contracts_refine.py::test_refined_performance_roundtrip_carries_3_1_0 -x` |
| SC-1 | (regression) Existing 14 fixtures at `schema_version: "3.0.0"` still round-trip after bump | unit (parametrize) | `tests/test_contracts_roundtrip_regression.py::test_3_0_0_fixture_still_roundtrips_after_bump[<name>]` × 14 | `pytest tests/test_contracts_roundtrip_regression.py -x` |
| SC-1 | (enum) `RefineEditOp.rationale` rejects values outside the closed enum | unit | `tests/test_contracts_refine.py::test_rationale_rejects_unknown_value` | same pattern |
| SC-1 | (validator) `RefineEditOp(op="modify")` with no provided edit-payload field raises | unit | `tests/test_contracts_refine.py::test_modify_requires_at_least_one_field` | same pattern |
| SC-2 | `POST /v1/jobs` with `enable_refine=true` + missing `OHSHEET_ANTHROPIC_API_KEY` → HTTP 400 with clear message | integration | `tests/test_jobs_refine.py::test_create_job_400_when_enable_refine_true_and_key_missing` | `pytest tests/test_jobs_refine.py::test_create_job_400_when_enable_refine_true_and_key_missing -x` |
| SC-2 | `POST /v1/jobs` with `enable_refine=false` + key set → 202 and execution plan contains NO "refine" step | integration | `tests/test_jobs_refine.py::test_create_job_202_when_enable_refine_false_plan_has_no_refine` | same pattern |
| SC-2 | `POST /v1/jobs` with `enable_refine=true` + key set → 202 and plan includes "refine" | integration | `tests/test_jobs_refine.py::test_create_job_202_when_enable_refine_true_plan_includes_refine` | same pattern |
| SC-3 | `OHSHEET_REFINE_KILL_SWITCH=true` + request `enable_refine=true` → behaves as if `enable_refine=false` (IDENTICAL execution plan) | integration | `tests/test_jobs_refine.py::test_kill_switch_produces_identical_plan_to_enable_refine_false` | same pattern |
| SC-3 | (parity) Pre-Phase-1 baseline plans for all 4 variants are byte-equal to post-Phase-1 plans when `enable_refine=False` | unit (parametrize) | `tests/test_pipeline_config.py::test_enable_refine_false_baseline_byte_equal[<variant>]` × 4 | `pytest tests/test_pipeline_config.py -x` |
| SC-3 | (log signal) Kill switch logs `warning` once per silently-coerced job | integration | `tests/test_jobs_refine.py::test_kill_switch_emits_warning_log` (uses `caplog`) | same pattern |
| SC-4 | `OHSHEET_REFINE_MODEL=unsupported-model` rejected at `Settings()` instantiation | unit | `tests/test_settings_refine.py::test_allowlist_rejects_unsupported_model` | `pytest tests/test_settings_refine.py::test_allowlist_rejects_unsupported_model -x` |
| SC-4 | `OHSHEET_REFINE_MODEL=claude-opus-4-6` + `OHSHEET_REFINE_ALLOW_OPUS=false` rejected | unit | `tests/test_settings_refine.py::test_allowlist_rejects_opus_without_flag` | same pattern |
| SC-4 | `OHSHEET_REFINE_MODEL=claude-opus-4-6` + `OHSHEET_REFINE_ALLOW_OPUS=true` accepted | unit | `tests/test_settings_refine.py::test_allowlist_accepts_opus_with_flag` | same pattern |
| SC-4 | Default Settings (`claude-sonnet-4-6`) accepted | unit | `tests/test_settings_refine.py::test_allowlist_accepts_default_sonnet` | same pattern |
| SC-5 | `.gitignore` excludes `.env` | unit | `tests/test_precommit_hook.py::test_gitignore_excludes_dotenv` | `pytest tests/test_precommit_hook.py::test_gitignore_excludes_dotenv -x` |
| SC-5 | `.pre-commit-config.yaml` exists and references detect-secrets | unit | `tests/test_precommit_hook.py::test_precommit_config_has_detect_secrets_hook` | same pattern |
| SC-5 | `.secrets.baseline` exists and is valid JSON | unit | `tests/test_precommit_hook.py::test_secrets_baseline_is_valid_json` | same pattern |
| SC-5 | End-to-end: `pre-commit run detect-secrets --files <path>` on a file containing a mock Anthropic key returns non-zero exit | subprocess integration | `tests/test_precommit_hook.py::test_detect_secrets_blocks_mock_anthropic_key` | `pytest tests/test_precommit_hook.py::test_detect_secrets_blocks_mock_anthropic_key -x` |

**Counter-check:** A table scan confirms every success criterion has **≥1 automated check** runnable in < 30 seconds (with the pre-commit subprocess test being the slowest at ~2-5s due to hook venv bootstrap). No manual verification required.

### Test Module / Fixture Layout

| File | Purpose | Key fixtures |
|------|---------|--------------|
| `tests/test_contracts_refine.py` | CTR-01..CTR-05 contract tests — definition, round-trip, enum, validator | Hand-built `RefineEditOp` / `RefineCitation` / `RefinedPerformance` instances; no file I/O |
| `tests/test_contracts_roundtrip_regression.py` | Parametrize-over-glob regression for existing 14 fixtures; proves they stay at 3.0.0 and still round-trip | `FIXTURE_NAMES` from `tests/fixtures/__init__.py` — existing pattern |
| `tests/test_settings_refine.py` | CFG-03..CFG-05 and CFG-07 — SecretStr masking, allowlist rejection, knob defaults | `monkeypatch.setenv` for env-var coverage; direct `Settings(...)` constructor for inline tests |
| `tests/test_jobs_refine.py` | CFG-02 + CFG-04 + CFG-06 — HTTP boundary, 400 semantics, kill switch coercion, plan parity | Existing `client` fixture from conftest.py; `monkeypatch.setattr(settings, 'anthropic_api_key', ...)`; `caplog` for log-signal assertions |
| `tests/test_pipeline_config.py` | EXTEND existing file — enable_refine parametrize + byte-equal pre/post-Phase-1 baseline | Literal expected-list snapshots |
| `tests/test_precommit_hook.py` | CFG-08 pre-commit end-to-end verification | Uses `tmp_path` for throwaway git repo; `subprocess.run` to invoke pre-commit |

### Regression Guard Strategy for 3.0.0 Fixtures

**Pattern:** parametrize-over-glob, using the existing `FIXTURE_NAMES` export from `tests/fixtures/__init__.py`. Mirrors `tests/test_engrave_quality.py:83` one-for-one.

```python
# tests/test_contracts_roundtrip_regression.py (NEW)
import json
from pathlib import Path

import pytest

from tests.fixtures import FIXTURE_NAMES, load_score_fixture

_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "scores"


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_3_0_0_fixture_schema_version_preserved(name: str):
    """D-02: committed fixtures must stay at schema_version='3.0.0' as regression guards."""
    raw = json.loads((_FIXTURES_DIR / f"{name}.json").read_text())
    assert raw["schema_version"] == "3.0.0", (
        f"fixture {name} lost its 3.0.0 tag — per D-02, fixtures must NOT be "
        "regenerated after the SCHEMA_VERSION bump. Run `git checkout tests/fixtures/scores/` "
        "to restore."
    )


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_3_0_0_fixture_still_validates_against_3_1_0_contracts(name: str):
    """Loose `schema_version: str` field keeps old payloads parseable after bump."""
    fixture = load_score_fixture(name)  # re-validates via Pydantic
    assert fixture.schema_version == "3.0.0"


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_3_0_0_fixture_roundtrips_byte_equal(name: str):
    """model_validate → model_dump must produce the same JSON (schema shape integrity)."""
    raw = json.loads((_FIXTURES_DIR / f"{name}.json").read_text())
    fixture = load_score_fixture(name)
    dumped = fixture.model_dump(mode="json")
    # Canonical JSON comparison (dicts with same items are ==, list order matters)
    assert dumped == raw, (
        f"fixture {name}: round-trip altered the payload shape — "
        f"a Phase 1 contract change is not backward-compatible"
    )
```

**Why 3 assertions per fixture:** (1) guards against accidental regeneration; (2) guards against a future `Literal[...]` tightening on schema_version; (3) guards against any non-schema_version field shape drift. Collectively these form the "old payloads still work" integrity contract.

### Zero-Behavior-Change Proof Approach

**Requirement:** For `variant ∈ {full, audio_upload, midi_upload, sheet_only}` × `skip_humanizer ∈ {false, true}` × `score_pipeline ∈ {arrange, condense_transform}`, the `get_execution_plan()` output with `enable_refine=False` must be **byte-equal** to the pre-Phase-1 baseline. The baseline is captured as literal lists in the test file, not as a snapshot file — because a snapshot file can be accidentally regenerated and silently accept drift.

**Baseline values captured by local tool run (2026-04-13, pre-bump):**

```
variant="full"        , skip_humanizer=False, score_pipeline="arrange"            → ["ingest", "transcribe", "arrange", "humanize", "engrave"]
variant="audio_upload", skip_humanizer=False, score_pipeline="arrange"            → ["ingest", "transcribe", "arrange", "humanize", "engrave"]
variant="midi_upload" , skip_humanizer=False, score_pipeline="arrange"            → ["ingest", "arrange", "humanize", "engrave"]
variant="sheet_only"  , skip_humanizer=False, score_pipeline="arrange"            → ["ingest", "transcribe", "arrange", "engrave"]
variant="midi_upload" , skip_humanizer=False, score_pipeline="condense_transform" → ["ingest", "condense", "transform", "humanize", "engrave"]
variant="sheet_only"  , skip_humanizer=True,  score_pipeline="condense_transform" → ["ingest", "transcribe", "condense", "transform", "engrave"]
```

**Test:**

```python
# tests/test_pipeline_config.py EXTENSION
import pytest

from backend.contracts import PipelineConfig

# Pre-Phase-1 baselines — captured by tool run 2026-04-13 against
# backend/contracts.py SCHEMA_VERSION="3.0.0". ANY new Phase 1 code that
# changes these values fails the test. Expected behavior: these values
# are immutable while enable_refine=False.
_BASELINE_PLANS_ENABLE_REFINE_FALSE = {
    ("full", False, "arrange"): ["ingest", "transcribe", "arrange", "humanize", "engrave"],
    ("audio_upload", False, "arrange"): ["ingest", "transcribe", "arrange", "humanize", "engrave"],
    ("midi_upload", False, "arrange"): ["ingest", "arrange", "humanize", "engrave"],
    ("sheet_only", False, "arrange"): ["ingest", "transcribe", "arrange", "engrave"],
    ("midi_upload", False, "condense_transform"): ["ingest", "condense", "transform", "humanize", "engrave"],
    ("sheet_only", True, "condense_transform"): ["ingest", "transcribe", "condense", "transform", "engrave"],
}


@pytest.mark.parametrize(
    "variant,skip_humanizer,score_pipeline,expected",
    [(v, s, p, plan) for (v, s, p), plan in _BASELINE_PLANS_ENABLE_REFINE_FALSE.items()],
)
def test_enable_refine_false_preserves_pre_phase_1_baseline(
    variant, skip_humanizer, score_pipeline, expected
):
    """enable_refine=False must produce BYTE-EQUAL plans to pre-Phase-1 output.

    Any drift here is a Phase 1 bug. Phase 2 can expand the allowed plans
    for enable_refine=True, but the enable_refine=False case stays frozen.
    """
    cfg = PipelineConfig(
        variant=variant,
        skip_humanizer=skip_humanizer,
        score_pipeline=score_pipeline,
        enable_refine=False,  # the frozen case
    )
    assert cfg.get_execution_plan() == expected


@pytest.mark.parametrize(
    "variant,expected_insert_index,expected_insert_after",
    [
        ("full", 4, "humanize"),
        ("audio_upload", 4, "humanize"),
        ("midi_upload", 3, "humanize"),
        ("sheet_only", 3, "arrange"),  # no humanize — refine goes after arrange
    ],
)
def test_enable_refine_true_inserts_refine_at_correct_position(
    variant, expected_insert_index, expected_insert_after
):
    """enable_refine=True inserts refine directly after humanize (or after arrange in sheet_only)."""
    cfg = PipelineConfig(variant=variant, enable_refine=True)
    plan = cfg.get_execution_plan()
    assert "refine" in plan
    assert plan[expected_insert_index] == "refine"
    assert plan[expected_insert_index - 1] == expected_insert_after
```

**Why this catches drift:** If someone rewrites `get_execution_plan()` and accidentally changes the `"audio_upload"` recipe to `["ingest", "transcribe", "arrange", "humanize", "engrave", "publish"]`, every `_BASELINE_PLANS_ENABLE_REFINE_FALSE` row fails loudly.

### Pre-Commit Hook End-to-End Verification

**Requirement:** Prove the pre-commit hook actually blocks a mock Anthropic key commit. Not "the config file exists," but "running the hook on a staged file containing a mock key produces non-zero exit."

**Approach:** subprocess-run `pre-commit run detect-secrets --files <test_file>` with `cwd=repo_root` and assert non-zero exit. Run with `--no-stdin` to avoid any interactive prompt.

```python
# tests/test_precommit_hook.py (NEW)
import json
import subprocess
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PRECOMMIT_CONFIG = _REPO_ROOT / ".pre-commit-config.yaml"
_SECRETS_BASELINE = _REPO_ROOT / ".secrets.baseline"
_GITIGNORE = _REPO_ROOT / ".gitignore"


# ---- Sanity checks first — the cheaper tests ----

def test_gitignore_excludes_dotenv():
    """CFG-08 success criterion 5a — .env is gitignored."""
    lines = [line.strip() for line in _GITIGNORE.read_text().splitlines()]
    assert ".env" in lines, (
        ".env is not in .gitignore — it must be excluded to prevent "
        "accidental commits of OHSHEET_ANTHROPIC_API_KEY"
    )


def test_precommit_config_exists():
    assert _PRECOMMIT_CONFIG.is_file(), (
        f"{_PRECOMMIT_CONFIG} does not exist — CFG-08 requires a pre-commit hook"
    )


def test_precommit_config_has_detect_secrets_hook():
    content = _PRECOMMIT_CONFIG.read_text()
    assert "Yelp/detect-secrets" in content
    assert "detect-secrets" in content
    assert ".secrets.baseline" in content


def test_secrets_baseline_is_valid_json():
    assert _SECRETS_BASELINE.is_file(), (
        f"{_SECRETS_BASELINE} must exist — created by `detect-secrets scan > .secrets.baseline`"
    )
    json.loads(_SECRETS_BASELINE.read_text())  # raises on bad JSON


# ---- The load-bearing end-to-end test ----

def test_detect_secrets_blocks_mock_anthropic_key(tmp_path):
    """Verify the hook actually rejects a file containing a mock Anthropic key.

    This is the core CFG-08 success criterion — not "config exists" but
    "running the hook on a file with a mock key fails with non-zero exit."

    Uses a 40-char high-entropy mock key designed to trip detect-secrets's
    HighEntropyString detector regardless of whether a named AnthropicDetector
    plugin exists.
    """
    # Use a mock key that's both keyword-laden ("sk-ant-api03") and
    # high-entropy. Real Anthropic keys are ~108 chars; a 70-char mock
    # is plenty to trip entropy heuristics.
    mock_key_file = tmp_path / "leak.py"
    mock_key_file.write_text(textwrap.dedent("""
        # Accidentally committed key for testing detect-secrets
        ANTHROPIC_API_KEY = "sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789AbCdEfGhIjKlMnOpQrSt"
    """).lstrip())

    result = subprocess.run(
        ["pre-commit", "run", "detect-secrets", "--files", str(mock_key_file)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Non-zero exit = hook blocked the commit. Assert exit first (load-bearing),
    # then assert on output as a secondary signal.
    assert result.returncode != 0, (
        f"pre-commit hook did NOT block a mock Anthropic key — "
        f"stdout: {result.stdout!r}, stderr: {result.stderr!r}. "
        f"Check .pre-commit-config.yaml + .secrets.baseline configuration."
    )
    # Sanity: the output should mention detect-secrets or Potential secrets
    combined = result.stdout + result.stderr
    assert any(
        marker in combined
        for marker in ("detect-secrets", "Potential", "Secret", "secret")
    ), f"hook failed but without a recognizable detect-secrets signal: {combined!r}"


def test_detect_secrets_does_not_false_positive_on_repo_code(tmp_path):
    """Running the hook against the whole repo must exit cleanly.

    Guards against a baseline that's too narrow (false positives) or too broad
    (real secrets that slipped in before Phase 1).
    """
    result = subprocess.run(
        ["pre-commit", "run", "detect-secrets", "--all-files"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"pre-commit hook failed against the clean repo — "
        f"stdout: {result.stdout!r}, stderr: {result.stderr!r}. "
        f"Either add offending strings to .secrets.baseline (if false positive) "
        f"or remove the real secret."
    )
```

**Why this works:** The test writes a file to a `tmp_path`, invokes pre-commit via `subprocess.run` with `--files <path>`, and asserts on exit code. This exercises the actual hook installation path — including the hook's own isolated venv, the `--baseline` argument, and the detector plugins. It is the ONLY way to prove the hook blocks commits without actually committing.

**Slow-test caveat:** First run of `pre-commit run detect-secrets` bootstraps the hook's venv (~5s). Subsequent runs are <1s. Mark with `@pytest.mark.slow` if desired — but this test is critical enough to run on every `pytest -x`.

### Determinism Proofs

#### Source performance digest

**Claim:** `source_performance_digest = sha256(json.dumps(perf.model_dump(mode="json"), sort_keys=True, separators=(',', ':')).encode("utf-8")).hexdigest()` is deterministic across round-trips of the same semantic content.

**Verified locally (2026-04-13):**

```
hp1 = HumanizedPerformance(...)
hp2 = HumanizedPerformance.model_validate(hp1.model_dump(mode='json'))
digest1 = sha256(json.dumps(hp1.model_dump(mode='json'), sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
digest2 = sha256(json.dumps(hp2.model_dump(mode='json'), sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
→ digest1 == digest2 → True, value starts with b5ffcfca61eb0f8d...
```

**Test:**

```python
# tests/test_contracts_refine.py
import hashlib
import json

from backend.contracts import HumanizedPerformance
from tests.fixtures import load_score_fixture


def _digest(perf: HumanizedPerformance) -> str:
    return hashlib.sha256(
        json.dumps(perf.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode("utf-8"),
    ).hexdigest()


def test_source_performance_digest_is_deterministic_across_roundtrip():
    """Same semantic content → same digest, regardless of round-trips through Pydantic."""
    original = load_score_fixture("humanized_with_offsets")
    assert isinstance(original, HumanizedPerformance)
    roundtripped = HumanizedPerformance.model_validate(original.model_dump(mode="json"))

    assert _digest(original) == _digest(roundtripped)
    # 64-char lower-case hex
    assert len(_digest(original)) == 64
    assert all(c in "0123456789abcdef" for c in _digest(original))


def test_source_performance_digest_differs_between_different_inputs():
    """Different semantic content → different digest."""
    a = load_score_fixture("humanized_with_offsets")
    b = load_score_fixture("humanized_with_expression")
    assert isinstance(a, HumanizedPerformance)
    assert isinstance(b, HumanizedPerformance)
    assert _digest(a) != _digest(b)
```

#### Allowlist determinism (frozen set, no dynamic mutation)

**Claim:** `_ALLOWED_REFINE_MODELS_SONNET` is a `frozenset` literal, module-level, immutable; the allowlist is a pure function of `(refine_model, refine_allow_opus)` inputs.

**Test:**

```python
# tests/test_settings_refine.py
def test_allowlist_is_frozen_and_not_dynamically_mutable():
    """The allowlist constants must be frozenset literals so they can't be mutated."""
    from backend.config import (
        _ALLOWED_REFINE_MODELS_SONNET,
        _ALLOWED_REFINE_MODELS_OPUS,
    )
    assert isinstance(_ALLOWED_REFINE_MODELS_SONNET, frozenset)
    assert isinstance(_ALLOWED_REFINE_MODELS_OPUS, frozenset)
    # Static content check — if someone mutates these, the test fails loudly
    assert _ALLOWED_REFINE_MODELS_SONNET == frozenset({"claude-sonnet-4-6"})
    assert _ALLOWED_REFINE_MODELS_OPUS == frozenset({"claude-opus-4-6"})
```

### SecretStr Log-Safety Proof

**Claim:** Across all normal serialization paths (repr, str, f-string, model_dump, model_dump_json, %-format), the raw API key value never appears. Only explicit `.get_secret_value()` reveals it.

**Verified locally (2026-04-13):**

```
s = Settings(anthropic_api_key='sk-ant-api03-super-secret-value')
str(s.anthropic_api_key)            →  '**********'
repr(s.anthropic_api_key)           →  "SecretStr('**********')"
f'{s.anthropic_api_key}'            →  '**********'
f'{s.anthropic_api_key!r}'          →  "SecretStr('**********')"
'%s' % s.anthropic_api_key          →  '**********'
s.model_dump()                       →  {'anthropic_api_key': SecretStr('**********')}
s.model_dump_json()                  →  '{"anthropic_api_key":"**********"}'
s.anthropic_api_key.get_secret_value() →  'sk-ant-api03-super-secret-value'  ← only reveal
```

**Test:**

```python
# tests/test_settings_refine.py
def test_secret_str_does_not_leak_key_in_model_dump():
    from pydantic import SecretStr
    from backend.config import Settings

    s = Settings(anthropic_api_key=SecretStr("sk-ant-api03-MOCKVALUE"))
    dumped = s.model_dump()
    # model_dump returns a SecretStr wrapper, not the raw string
    assert isinstance(dumped["anthropic_api_key"], SecretStr)
    assert "sk-ant-api03-MOCKVALUE" not in str(dumped)


def test_secret_str_does_not_leak_key_in_model_dump_json():
    from pydantic import SecretStr
    from backend.config import Settings

    s = Settings(anthropic_api_key=SecretStr("sk-ant-api03-MOCKVALUE"))
    json_str = s.model_dump_json()
    # model_dump_json emits the masked "**********" placeholder
    assert "sk-ant-api03-MOCKVALUE" not in json_str
    assert "**********" in json_str


def test_secret_str_does_not_leak_key_in_repr_or_fstring():
    from pydantic import SecretStr
    from backend.config import Settings

    s = Settings(anthropic_api_key=SecretStr("sk-ant-api03-MOCKVALUE"))
    for projected in (
        repr(s.anthropic_api_key),
        str(s.anthropic_api_key),
        f"{s.anthropic_api_key}",
        f"{s.anthropic_api_key!r}",
        "%s" % s.anthropic_api_key,
    ):
        assert "sk-ant-api03-MOCKVALUE" not in projected, (
            f"Key leaked via projection: {projected!r}"
        )


def test_secret_str_reveals_raw_only_on_get_secret_value():
    from pydantic import SecretStr
    from backend.config import Settings

    s = Settings(anthropic_api_key=SecretStr("sk-ant-api03-MOCKVALUE"))
    assert s.anthropic_api_key is not None
    assert s.anthropic_api_key.get_secret_value() == "sk-ant-api03-MOCKVALUE"
```

### Sampling Rate

- **Per task commit** — `pytest tests/test_contracts_refine.py tests/test_settings_refine.py tests/test_jobs_refine.py tests/test_pipeline_config.py -x -q` (unit-only, < 10s)
- **Per wave merge** — `pytest tests/ -x` (full suite, captures regression, ~2-5min given existing coverage breadth)
- **Phase gate** — (a) Full suite green; (b) `pre-commit run --all-files` green; (c) `pytest tests/test_precommit_hook.py::test_detect_secrets_blocks_mock_anthropic_key -x` passes (proves the hook works end-to-end).

### Wave 0 Gaps

- [ ] `tests/test_contracts_refine.py` — covers CTR-01..CTR-05 contract unit tests
- [ ] `tests/test_contracts_roundtrip_regression.py` — covers D-02 fixture regression guard
- [ ] `tests/test_settings_refine.py` — covers CFG-03, CFG-05, CFG-07 Settings-level tests
- [ ] `tests/test_jobs_refine.py` — covers CFG-02, CFG-04, CFG-06 HTTP boundary tests
- [ ] `tests/test_precommit_hook.py` — covers CFG-08 end-to-end verification
- [ ] EXTEND `tests/test_pipeline_config.py` — covers CFG-01 plan-insertion + byte-equal baseline
- [ ] CREATE `.pre-commit-config.yaml` — CFG-08
- [ ] CREATE `.secrets.baseline` via `detect-secrets scan --exclude-files '^tests/fixtures/'` — CFG-08
- [ ] (already present) pytest + pytest-asyncio in pyproject.toml dev extras — no install needed
- [ ] (already present) pre-commit installed globally — no install needed

## Security Domain

Phase 1 lands new security-relevant surface: a credential Settings field (`anthropic_api_key: SecretStr`) and a pre-commit hook gate. Treating this as a security-aware phase even though `security_enforcement` is implicitly enabled by the research brief's top-level mention of "security-adjacent signals."

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | Phase 1 does not add end-user authentication; `OHSHEET_ANTHROPIC_API_KEY` is server-to-Anthropic auth, not server-to-client. |
| V3 Session Management | no | No new session surface. |
| V4 Access Control | no | No new access-control decisions. The `enable_refine` flag is an opt-in, not an authorization gate. |
| V5 Input Validation | yes | All new Pydantic contracts use bounded `Field(..., ge=..., le=...)` and `Literal[...]` for closed sets — matches existing contracts.py style. `JobCreateRequest.enable_refine` is `bool` — Pydantic rejects non-bool cleanly. |
| V6 Cryptography | yes (narrow) | `SecretStr` for the API key; `hashlib.sha256` for `source_performance_digest` (non-cryptographic use — content hash for drift detection, not a signature). **Never hand-roll the masking.** |
| V7 Error Handling & Logging | yes | CFG-03 prescribes that `.get_secret_value()` is called only inside Phase 2's `RefineService.__init__`. The CFG-06 kill switch emits a structured `log.warning` that does NOT include the API key. |
| V9 Communications | no | Phase 1 adds no network surface (Anthropic SDK wiring is Phase 2). |
| V14 Config | yes | `OHSHEET_*` env-var prefix confinement, `extra="ignore"` to reject unknown env vars, `.env` in `.gitignore`, pre-commit hook as second-line defense. |

### Known Threat Patterns for This Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Anthropic API key committed to git | Information Disclosure | (1) `.env` in `.gitignore` (Line 7 — verified). (2) `.pre-commit-config.yaml` with `detect-secrets` v1.5.0 + `.secrets.baseline`. (3) `SecretStr` in Pydantic Settings — masks on all normal serialization paths. (4) Subprocess-run verification test that proves the hook blocks a mock key. |
| Allowlist bypass (setting `refine_model` to an arbitrary string to route calls to a cheaper / unreviewed model) | Tampering | Config-load validator (`@model_validator(mode="after")`). Rejects at `Settings()` instantiation, so no code path in the app ever sees a non-allowlisted model. Frozenset literal (no dynamic mutation). |
| Unknown `payload_type` reaching the engrave worker (type-confusion attack or accidental mislabeling) | Tampering / Denial of Service | Explicit `ValueError` with the full valid-tags list, as already done at `backend/workers/engrave.py:31-32`. Phase 1 extends the valid-tags list; the else branch remains the same (still raises). |
| Refine stage accidentally enabled globally (e.g., through a default-on configuration) | Denial of Service (cost blowout) | `OHSHEET_REFINE_KILL_SWITCH` as an ops escape hatch — forces off regardless of per-job `enable_refine`. Default-off for both `enable_refine` (per-job) and the kill switch (so admission remains per-job, but if something goes wrong, one env var disables it globally). |
| API key logged during debugging | Information Disclosure | `SecretStr` — 6 serialization paths verified to mask. Test suite includes negative assertion (`assert "sk-ant-api03-MOCKVALUE" not in ...`). |
| Pre-commit hook bypassed with `git commit --no-verify` | Information Disclosure | Out of scope for Phase 1 — the hook is a defense-in-depth measure. CI-side enforcement (GitHub Actions running `pre-commit run --all-files`) is Open Question #4. |

## Sources

### Primary (HIGH confidence)

- `/Users/jackjiang/GitHub/oh-sheet/shared/shared/contracts.py` — contract patterns, SCHEMA_VERSION, insertion point
- `/Users/jackjiang/GitHub/oh-sheet/backend/config.py` — Settings field patterns, existing `field_validator` style, `env_prefix="OHSHEET_"` convention
- `/Users/jackjiang/GitHub/oh-sheet/backend/api/routes/jobs.py` — `JobCreateRequest` + `create_job` wiring precedent (`prefer_clean_source` follows identical pattern)
- `/Users/jackjiang/GitHub/oh-sheet/backend/workers/engrave.py:22-32` — existing if/elif chain to extend
- `/Users/jackjiang/GitHub/oh-sheet/backend/jobs/runner.py:45-53` — `STEP_TO_TASK` registration
- `/Users/jackjiang/GitHub/oh-sheet/tests/test_engrave_quality.py:83` — `@pytest.mark.parametrize("name", FIXTURE_NAMES)` pattern to mirror for the regression guard
- `/Users/jackjiang/GitHub/oh-sheet/tests/test_pipeline_config.py` — existing plan-assertion style
- `/Users/jackjiang/GitHub/oh-sheet/tests/fixtures/_builders.py` — 14 fixture builders, `FIXTURE_NAMES` export, `load_score_fixture` helper
- `/Users/jackjiang/GitHub/oh-sheet/tests/conftest.py` — `client` fixture, `isolated_blob_root`, `celery_eager_mode`
- `/Users/jackjiang/GitHub/oh-sheet/.gitignore:7` — `.env` already excluded
- Local tool runs (2026-04-13) verifying: Pydantic 2.12.5 + pydantic-settings 2.13.1 installed; `model_validator(mode="after")` rejects bad models; `SecretStr` masks on 6 serialization paths; `source_performance_digest` deterministic across round-trips; current execution-plan baseline values for all 4 variants

### Secondary (MEDIUM confidence)

- [Yelp/detect-secrets GitHub repo](https://github.com/Yelp/detect-secrets) — v1.5.0 current, `.pre-commit-hooks.yaml` usage pattern
- [pydantic validators documentation](https://docs.pydantic.dev/latest/concepts/validators/) — `model_validator(mode="after")` semantics
- [pytest parametrize documentation](https://docs.pytest.org/en/stable/how-to/parametrize.html) — parametrize-over-collection pattern

### Tertiary (LOW confidence — flagged in Assumptions Log)

- WebSearch result noting "Anthropic pattern being added to major detection tools" — does NOT confirm a named detect-secrets AnthropicDetector. Mitigation: the CFG-08 test uses a high-entropy mock key that will trip the entropy plugin independent of keyword detection.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every library already listed in `pyproject.toml` or already installed globally; versions verified locally
- Architecture patterns: HIGH — every pattern mirrors an existing precedent in the repo (prefer_clean_source wiring, field_validator style, parametrize-over-FIXTURE_NAMES)
- Pitfalls: HIGH — identified by reading the existing code paths, not speculation
- Determinism proofs: HIGH — verified by local Python tool runs (2026-04-13)
- Pre-commit hook behavior: MEDIUM — config schema is HIGH (official docs); "does detect-secrets catch `sk-ant-api03-`?" is MEDIUM because no named Anthropic plugin is confirmed — the test uses a high-entropy mock to cover the entropy-detector path regardless
- SecretStr masking: HIGH — verified across 6 serialization paths locally

**Research date:** 2026-04-13
**Valid until:** 2026-05-13 (30 days — all components are stable; Pydantic releases are additive, detect-secrets is at mature v1.5.0, pre-commit 4.x has stable config schema)
