---
phase: 1
slug: contracts-and-plumbing
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-13
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Sourced from `01-RESEARCH.md` §"Validation Architecture" (line 724).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.0.3 + pytest-asyncio ≥0.24 |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` (already present) |
| **Quick run command** | `pytest tests/test_contracts_refine.py tests/test_settings_refine.py tests/test_jobs_refine.py tests/test_contracts_roundtrip_regression.py tests/test_pipeline_config.py tests/test_precommit_hook.py -x -q` |
| **Full suite command** | `pytest tests/ --cov=backend -x` |
| **Estimated runtime** | ~10s quick / ~2–5min full (hook bootstrap adds 5s cold) |

---

## Sampling Rate

- **After every task commit:** Run quick suite (unit-only, <10s).
- **After every plan wave:** Run full suite (`pytest tests/ -x`).
- **Before `/gsd-verify-work`:**
  1. Full suite green.
  2. `pre-commit run --all-files` green.
  3. `pytest tests/test_precommit_hook.py::test_detect_secrets_blocks_mock_anthropic_key -x` green (proves hook blocks end-to-end).
- **Max feedback latency:** 10 seconds (quick) / 300 seconds (full).

---

## Per-Task Verification Map

> Populated by planner. Each task below is the target anchor for automated verification.
> Task IDs follow the pattern `{phase}-{plan}-{task}` (e.g., `1-01-01`). Planner fills the Task ID, Plan, and Wave columns; all other columns are fixed by this document and carry forward verbatim into `<automated>` blocks on each task.

| # | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists |
|---|-------------|------------|-----------------|-----------|-------------------|-------------|
| V1 | CTR-01, CTR-02, CTR-03 | T-1-01 / — | Refine contracts importable, round-trip, schema_version carries "3.1.0" | unit | `pytest tests/test_contracts_refine.py::test_refined_performance_roundtrip_carries_3_1_0 -x` | ❌ W0 |
| V2 | CTR-01 | — | `RefineEditOp.rationale` rejects values outside closed enum | unit | `pytest tests/test_contracts_refine.py::test_rationale_rejects_unknown_value -x` | ❌ W0 |
| V3 | CTR-01 | — | `RefineEditOp(op="modify")` with no provided edit-payload field raises | unit | `pytest tests/test_contracts_refine.py::test_modify_requires_at_least_one_field -x` | ❌ W0 |
| V4 | CTR-02, D-06 | — | `source_performance_digest` is deterministic across round-trips (sha256, 64-char lowercase hex) | unit | `pytest tests/test_contracts_refine.py::test_source_performance_digest_is_deterministic_across_roundtrip -x` | ❌ W0 |
| V5 | CTR-04 | — | Regression: 14 existing fixtures stay at `schema_version:"3.0.0"` and round-trip byte-equal | unit (parametrize ×14) | `pytest tests/test_contracts_roundtrip_regression.py -x` | ❌ W0 |
| V6 | CTR-05 | — | D-07 engrave-worker branch dispatches `RefinedPerformance` correctly (unwraps → humanized payload) | unit | `pytest tests/test_engrave_payload_dispatch.py -x` (or extension of existing `tests/test_stages.py`) | ❌ W0 |
| V7 | CFG-01 | — | `PipelineConfig.get_execution_plan()` inserts `"refine"` after `"humanize"` (or after `"arrange"` for `sheet_only`) when `enable_refine=True` | unit (parametrize ×4 variants) | `pytest tests/test_pipeline_config.py::test_enable_refine_true_inserts_refine_at_correct_position -x` | ✅ (EXTEND) |
| V8 | CFG-01 (frozen baseline) | — | `enable_refine=False` plans are byte-equal to pre-Phase-1 baseline across 6 variant combinations | unit (parametrize ×6) | `pytest tests/test_pipeline_config.py::test_enable_refine_false_preserves_pre_phase_1_baseline -x` | ✅ (EXTEND) |
| V9 | CFG-02 | T-1-03 | `POST /v1/jobs` with `enable_refine=true` + missing `OHSHEET_ANTHROPIC_API_KEY` → HTTP 400 with clear message | integration | `pytest tests/test_jobs_refine.py::test_create_job_400_when_enable_refine_true_and_key_missing -x` | ❌ W0 |
| V10 | CFG-02 | — | `POST /v1/jobs` with `enable_refine=false` + key set → 202 and plan contains NO "refine" | integration | `pytest tests/test_jobs_refine.py::test_create_job_202_when_enable_refine_false_plan_has_no_refine -x` | ❌ W0 |
| V11 | CFG-04 | — | `POST /v1/jobs` with `enable_refine=true` + key set → 202 and plan includes "refine" | integration | `pytest tests/test_jobs_refine.py::test_create_job_202_when_enable_refine_true_plan_includes_refine -x` | ❌ W0 |
| V12 | CFG-06 | T-1-04 | `OHSHEET_REFINE_KILL_SWITCH=true` + `enable_refine=true` → plan IDENTICAL to `enable_refine=false` | integration | `pytest tests/test_jobs_refine.py::test_kill_switch_produces_identical_plan_to_enable_refine_false -x` | ❌ W0 |
| V13 | CFG-06 | — | Kill switch emits exactly one `log.warning` per coerced job (caplog assertion) | integration | `pytest tests/test_jobs_refine.py::test_kill_switch_emits_warning_log -x` | ❌ W0 |
| V14 | CFG-03 | T-1-05 | `SecretStr` does NOT leak key in `model_dump`, `model_dump_json`, `repr`, `str`, f-string, or `%s`-format | unit (6 assertions) | `pytest tests/test_settings_refine.py::test_secret_str_does_not_leak_key_in_model_dump tests/test_settings_refine.py::test_secret_str_does_not_leak_key_in_model_dump_json tests/test_settings_refine.py::test_secret_str_does_not_leak_key_in_repr_or_fstring -x` | ❌ W0 |
| V15 | CFG-03 | — | `.get_secret_value()` is the ONLY reveal path | unit | `pytest tests/test_settings_refine.py::test_secret_str_reveals_raw_only_on_get_secret_value -x` | ❌ W0 |
| V16 | CFG-05 | T-1-02 | `OHSHEET_REFINE_MODEL=unsupported-model` rejected at `Settings()` instantiation (ValidationError) | unit | `pytest tests/test_settings_refine.py::test_allowlist_rejects_unsupported_model -x` | ❌ W0 |
| V17 | CFG-05 | — | `OHSHEET_REFINE_MODEL=claude-opus-4-6` + `OHSHEET_REFINE_ALLOW_OPUS=false` rejected; `=true` accepted | unit (2 tests) | `pytest tests/test_settings_refine.py::test_allowlist_rejects_opus_without_flag tests/test_settings_refine.py::test_allowlist_accepts_opus_with_flag -x` | ❌ W0 |
| V18 | CFG-05 | — | Default Settings (`claude-sonnet-4-6`) accepted; allowlist constants are `frozenset` literals | unit | `pytest tests/test_settings_refine.py::test_allowlist_accepts_default_sonnet tests/test_settings_refine.py::test_allowlist_is_frozen_and_not_dynamically_mutable -x` | ❌ W0 |
| V19 | CFG-07 | — | All four refine knobs (`refine_temperature`, `refine_max_tokens`, `refine_max_retries`, any budget) load from `OHSHEET_*` env vars with correct types and defaults | unit | `pytest tests/test_settings_refine.py::test_refine_knobs_load_from_env_with_defaults -x` | ❌ W0 |
| V20 | CFG-08 | T-1-06 | `.gitignore` contains `.env` | unit | `pytest tests/test_precommit_hook.py::test_gitignore_excludes_dotenv -x` | ❌ W0 |
| V21 | CFG-08 | — | `.pre-commit-config.yaml` exists and references `Yelp/detect-secrets` + `.secrets.baseline` | unit | `pytest tests/test_precommit_hook.py::test_precommit_config_has_detect_secrets_hook -x` | ❌ W0 |
| V22 | CFG-08 | — | `.secrets.baseline` exists and is valid JSON | unit | `pytest tests/test_precommit_hook.py::test_secrets_baseline_is_valid_json -x` | ❌ W0 |
| V23 | CFG-08 (load-bearing) | T-1-06 | End-to-end: `pre-commit run detect-secrets --files <mock_key_file>` on a file with a 70-char high-entropy Anthropic-style key returns non-zero exit | subprocess integration | `pytest tests/test_precommit_hook.py::test_detect_secrets_blocks_mock_anthropic_key -x` | ❌ W0 |
| V24 | CFG-08 | — | `pre-commit run detect-secrets --all-files` against clean repo returns 0 (no false positives; guards a too-narrow or too-broad baseline) | subprocess integration | `pytest tests/test_precommit_hook.py::test_detect_secrets_does_not_false_positive_on_repo_code -x` | ❌ W0 |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky — populated by executor during phase run.*

---

## Wave 0 Requirements

Test modules that must exist before any feature task can verify:

- [ ] `tests/test_contracts_refine.py` — CTR-01..CTR-05 contract unit tests (covers V1–V4)
- [ ] `tests/test_contracts_roundtrip_regression.py` — D-02 fixture regression guard (covers V5)
- [ ] `tests/test_settings_refine.py` — CFG-03, CFG-05, CFG-07 Settings-level tests (covers V14–V19)
- [ ] `tests/test_jobs_refine.py` — CFG-02, CFG-04, CFG-06 HTTP-boundary tests (covers V9–V13)
- [ ] `tests/test_precommit_hook.py` — CFG-08 end-to-end verification (covers V20–V24)
- [ ] EXTEND `tests/test_pipeline_config.py` — CFG-01 plan insertion + byte-equal baseline (covers V7–V8)
- [ ] EXTEND engrave dispatch coverage (new file `tests/test_engrave_payload_dispatch.py` OR extend `tests/test_stages.py`) — D-07 (covers V6)
- [ ] CREATE `.pre-commit-config.yaml` with `Yelp/detect-secrets` hook entry referencing `.secrets.baseline`
- [ ] CREATE `.secrets.baseline` via `detect-secrets scan --exclude-files '^tests/fixtures/'`

> Not required: pytest, pytest-asyncio, pre-commit — already installed per pyproject.toml dev extras and global tooling.

---

## Manual-Only Verifications

*None — all phase behaviors have automated verification. See the Per-Task Verification Map above.*

Deliberately excluded per Nyquist (Dimension 8): no criterion is satisfied by "it compiles" or "manual check only."

---

## Zero-Behavior-Change Proof (derived from `01-RESEARCH.md` §"Zero-Behavior-Change Proof Approach")

Immutable baseline captured against pre-Phase-1 `contracts.py` (SCHEMA_VERSION="3.0.0") on 2026-04-13. These literal lists are embedded in the test file; drift fails the test.

```python
_BASELINE_PLANS_ENABLE_REFINE_FALSE = {
    ("full",         False, "arrange"):            ["ingest", "transcribe", "arrange", "humanize", "engrave"],
    ("audio_upload", False, "arrange"):            ["ingest", "transcribe", "arrange", "humanize", "engrave"],
    ("midi_upload",  False, "arrange"):            ["ingest",               "arrange", "humanize", "engrave"],
    ("sheet_only",   False, "arrange"):            ["ingest", "transcribe", "arrange",              "engrave"],
    ("midi_upload",  False, "condense_transform"): ["ingest", "condense", "transform", "humanize", "engrave"],
    ("sheet_only",   True,  "condense_transform"): ["ingest", "transcribe", "condense", "transform", "engrave"],
}
```

Post-Phase-1: the same calls with `enable_refine=False` MUST return identical lists.

---

## Threat-to-Test Cross-Reference (for `/gsd-secure-phase`)

| Threat ID (planner's `<threat_model>`) | Mitigation Test IDs | Notes |
|-----|---------------------|-------|
| T-1-01 (e.g., schema drift / unknown payload_type reaching engrave) | V1, V6 | Contract existence + engrave dispatch |
| T-1-02 (allowlist bypass) | V16, V17, V18 | Config-load validator + frozenset immutability |
| T-1-03 (HTTP boundary: enable_refine without key) | V9 | 400 with clear message |
| T-1-04 (refine enabled globally — cost blowout) | V12, V13 | Kill switch parity + log signal |
| T-1-05 (API key logged during debugging) | V14, V15 | SecretStr across 6 serialization paths |
| T-1-06 (API key committed to git) | V20, V21, V22, V23, V24 | gitignore + pre-commit + baseline + end-to-end hook |

> Threat IDs are populated by the planner's `<threat_model>` block. This cross-reference is canonical — planner must use these T-1-* IDs verbatim.

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s (quick) / < 300s (full)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
