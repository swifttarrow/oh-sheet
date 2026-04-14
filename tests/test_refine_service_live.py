"""Live Anthropic smoke test — gated by @pytest.mark.live_anthropic + OHSHEET_RUN_INTEGRATION_TESTS=1.

D-16, D-17: Manual dev-machine only. NEVER runs in CI.
D-18: On first successful run with --record-live-fixtures, capture the parsed
      response + trace into tests/fixtures/refine/live_sample_v1.json.
      Subsequent runs WITHOUT --record-live-fixtures assert shape-invariants
      against the captured fixture (no exact content match — the LLM varies
      across model ticks). The invariants include NON-EMPTY applied_edits:
      the live ship-gate requires at least one applied edit, because zero
      edits on the canonical Twinkle input indicates prompt / web_search /
      validator misalignment — capture the trace for diagnosis before
      relaxing this assertion.
D-19: Phase-3 ship-gate — one live end_turn happy-path works end-to-end
      through RefineService.run with non-empty applied_edits.

Developer workflow
------------------
First-run fixture capture (requires OHSHEET_ANTHROPIC_API_KEY with
web_search entitlement)::

    OHSHEET_RUN_INTEGRATION_TESTS=1 \\
    pytest tests/test_refine_service_live.py \\
           -m live_anthropic --record-live-fixtures

Subsequent runs assert shape-invariants against the captured fixture::

    OHSHEET_RUN_INTEGRATION_TESTS=1 \\
    pytest tests/test_refine_service_live.py -m live_anthropic

Default runs (no env var): test is SKIPPED — no network access, no cost.
"""
from __future__ import annotations

import json
from pathlib import Path

import anthropic
import pytest
from shared.contracts import SCHEMA_VERSION

from backend.config import settings
from backend.services.refine import RefineService
from backend.services.refine_prompt import REFINE_PROMPT_VERSION
from backend.services.refine_validate import RefineValidator
from tests.test_refine_service import _humanized  # reuse the canonical test performance

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "refine" / "live_sample_v1.json"


@pytest.mark.live_anthropic
@pytest.mark.asyncio
async def test_live_end_turn_happy_path(request: pytest.FixtureRequest) -> None:
    """D-19 ship gate — one live call with end_turn + schema-invariant shape
    INCLUDING non-empty applied_edits (D-18 locked invariant).

    Skipped unless OHSHEET_RUN_INTEGRATION_TESTS=1 (Task-2 conftest hook).
    Requires OHSHEET_ANTHROPIC_API_KEY set to a real key with web_search
    entitlement.
    """
    if settings.anthropic_api_key is None:
        pytest.skip("OHSHEET_ANTHROPIC_API_KEY not set — live test cannot run")

    # CFG-03 single-site secret access — mirrors backend/workers/refine.py exactly.
    # This is the ONE other place in the codebase that calls .get_secret_value()
    # because the live test needs a real AsyncAnthropic instance; all other
    # callers use the worker's client. No new logging sites.
    client = anthropic.AsyncAnthropic(
        api_key=settings.anthropic_api_key.get_secret_value(),
    )
    validator = RefineValidator(settings)
    svc = RefineService(client=client, validator=validator, settings=settings)

    performance = _humanized()
    metadata = {"title": "Twinkle Twinkle Little Star", "composer": "Traditional"}

    refined, trace = await svc.run(performance, metadata)

    # ------------------------------------------------------------
    # D-18 Fixture capture — when --record-live-fixtures is passed,
    # write the canonical sample and skip strict assertions this run.
    # ------------------------------------------------------------
    if request.config.getoption("--record-live-fixtures"):
        _FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _FIXTURE_PATH.write_text(
            json.dumps(
                {
                    "refined": refined.model_dump(mode="json"),
                    "trace": trace.model_dump(mode="json"),
                },
                indent=2,
                default=str,
            ),
        )
        return  # Recorded. Re-run without the flag to exercise assertions.

    # ------------------------------------------------------------
    # D-18 Shape invariants — exact content is NOT asserted because
    # the LLM's output varies across model ticks. These are the
    # structural facts the Phase-3 ship gate requires. D-18 locks
    # "non-empty applied_edits" as a mandatory invariant — a zero-
    # edit live response indicates prompt/web_search/validator
    # misalignment and must fail the ship gate until diagnosed.
    # ------------------------------------------------------------
    # refined schema anchor
    assert refined.schema_version == SCHEMA_VERSION
    assert SCHEMA_VERSION == "3.1.0"
    assert refined.model.startswith("claude-")

    # trace version + stop_reason + shape
    assert trace.prompt_version == REFINE_PROMPT_VERSION
    assert trace.stop_reason == "end_turn", (
        f"Live gate D-19: expected end_turn, got {trace.stop_reason!r} — "
        f"if pause_turn, that is a separate known failure surface tested via "
        f"FakeRefineClient (D-27). If observed in practice here, capture "
        f"the trace for diagnostics and re-run."
    )
    assert trace.usage.input_tokens > 0
    assert trace.usage.output_tokens > 0
    # D-18/D-19 LOCKED INVARIANT: non-empty applied_edits.
    # The user-locked ship gate requires at least one applied edit on
    # the canonical Twinkle input. Zero edits is NOT acceptable live
    # output — it indicates prompt / web_search / validator
    # misalignment that must be diagnosed (inspect llm_trace.json)
    # before the ship gate is cleared. Do NOT relax this to
    # isinstance(trace.applied_edits, list); that silently permits
    # zero edits and contradicts D-18.
    assert len(trace.applied_edits) > 0, (
        "D-18/D-19 ship gate: live Anthropic call on the canonical input must "
        "produce at least one applied edit. Zero edits indicates prompt, web_search, "
        "or validator misalignment — capture the trace for diagnosis before "
        "relaxing this assertion."
    )
    assert isinstance(trace.rejected_edits, list)

    # If a fixture exists, cross-check basic structure keys survived.
    if _FIXTURE_PATH.exists():
        fixture = json.loads(_FIXTURE_PATH.read_text())
        assert "refined" in fixture and "trace" in fixture
        assert fixture["trace"]["prompt_version"] == REFINE_PROMPT_VERSION, (
            "Fixture prompt_version drift — if REFINE_PROMPT_VERSION bumped "
            "intentionally (D-29 semantic change), re-record the fixture "
            "with --record-live-fixtures."
        )
