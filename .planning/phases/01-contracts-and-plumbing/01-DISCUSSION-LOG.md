# Phase 1: Contracts and Plumbing - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-13
**Phase:** 01-contracts-and-plumbing
**Areas discussed:** Schema version strategy, payload_type discriminator shape
**Areas skipped (Claude's Discretion):** Rationale enum values, Pre-commit secret-scan tool

---

## Gray-area selection

| Option | Description | Selected |
|--------|-------------|----------|
| Schema version strategy | Hard bump vs coexistence Literal on schema_version | ✓ |
| Rationale enum values | Closed domain for RefineEditOp.rationale | |
| payload_type discriminator shape | Extend dict envelope vs Pydantic discriminated union | ✓ |
| Pre-commit secret-scan tool | gitleaks vs detect-secrets | |

**User's choice:** Schema version strategy + payload_type discriminator shape
**Notes:** Rationale enum values and pre-commit hook moved to Claude's Discretion — Claude will pick sensible defaults and the planner can revisit.

---

## Schema version strategy

### Q1 — How should the SCHEMA_VERSION bump from 3.0.0 to 3.1.0 be applied across existing contracts?

| Option | Description | Selected |
|--------|-------------|----------|
| Hard bump (Recommended) | Single SCHEMA_VERSION = "3.1.0" replaces 3.0.0 everywhere; schema_version stays `str` (no Literal); old 3.0.0 blobs still validate | ✓ |
| Strict Literal on new contracts only | Bump constant + tighten only NEW Refine* contracts to schema_version: Literal["3.1.0"]; existing untyped | |
| Coexistence Literal everywhere | Tighten every contract to schema_version: Literal["3.0.0", "3.1.0"]; old blobs round-trip | |

**User's choice:** Hard bump (Recommended)
**Notes:** Minimal churn; matches current codebase style; no production traffic to migrate.

### Q2 — What should happen to the ~15 existing test fixtures (tests/fixtures/scores/*.json) with hardcoded "schema_version": "3.0.0"?

| Option | Description | Selected |
|--------|-------------|----------|
| Leave at 3.0.0 (Recommended) | Keep fixtures at 3.0.0; they still validate (field is `str`), serve as regression guards; new Refine fixtures use 3.1.0 | ✓ |
| Migrate all to 3.1.0 | Bulk-update every fixture; consistent but erases 'old payloads still work' evidence | |
| Mix: migrate humanized/score, leave others | Targeted per-area migration; awkward and inconsistent | |

**User's choice:** Leave at 3.0.0 (Recommended)
**Notes:** Fixtures stay as regression guards.

---

## payload_type discriminator shape

### Q1 — How should the engrave worker accept the new RefinedPerformance payload_type?

| Option | Description | Selected |
|--------|-------------|----------|
| Extend if/elif chain (Recommended) | Add third branch in backend/workers/engrave.py; ~8 LOC; no new abstractions | ✓ |
| Pydantic discriminated-union envelope | Introduce EngraveInput Annotated Union with discriminator; cleaner but bigger refactor across producer + consumer | |
| Tagged-union on payload field only | Keep dict envelope but make payload field a discriminated union; awkward — no natural type tag on existing contracts | |

**User's choice:** Extend if/elif chain (Recommended)
**Notes:** Minimal churn, matches Phase 1 "no behavior change" goal.

### Q2 — How should RefinedPerformance relate to HumanizedPerformance structurally?

| Option | Description | Selected |
|--------|-------------|----------|
| Nested composition (Recommended) | RefinedPerformance has a nested HumanizedPerformance field + edits/citations/model/digest alongside | ✓ |
| Subclass HumanizedPerformance | RefinedPerformance(HumanizedPerformance) adds edit-metadata fields; shorter but semantically wrong | |
| Parallel: source + refined perf, both nested | Both pre/post performances embedded; 2x wire size, enables local diff | |

**User's choice:** Nested composition (Recommended)
**Notes:** Composition preserves the "refine is a wrapping event" boundary.

### Q3 — What should the embedded HumanizedPerformance field be named on RefinedPerformance?

(Reworded after surfacing a bug in Q2's option copy — `source_performance` was misleading because the embedded performance is the POST-edit result, not the source. `source_performance_digest` is a SHA of the pre-edit input.)

| Option | Description | Selected |
|--------|-------------|----------|
| refined_performance (Recommended) | Explicit: embedded field is the post-edit result; digest's "source" naming stays consistent as pre-edit | ✓ |
| performance | Short; context makes clear which performance | |
| humanized_performance | Matches contract class name; redundant but unambiguous about type | |

**User's choice:** refined_performance (Recommended)
**Notes:** Semantics locked — `refined_performance` is POST-edit, `source_performance_digest` is SHA of pre-edit input.

---

## Claude's Discretion

User left these for Claude to pick (captured in CONTEXT.md Claude's Discretion section):

- `RefineEditOp.rationale` enum values — default domain proposed
- `RefineEditOp` edit-payload field shape for `op="modify"` vs `op="delete"`
- `RefineCitation.confidence` type (float ge=0.0 le=1.0 matching existing convention)
- `source_performance_digest` algorithm (SHA-256 of `model_dump_json()`)
- Pre-commit secret-scan tool (detect-secrets preferred for Python-native + baseline file)
- `OHSHEET_REFINE_KILL_SWITCH` visibility (silent strip + single WARN log per job)
- Model allowlist shape (private frozenset constants + model_validator on Settings)

## Deferred Ideas

- Pydantic discriminated-union engrave envelope — deferred; revisit if engrave gets more payload types
- Strict `Literal["3.0.0", "3.1.0"]` on schema_version — deferred; purpose is documentation, not validation
- Creating `.planning/PROJECT.md` — referenced by STATE.md but missing; out of scope
