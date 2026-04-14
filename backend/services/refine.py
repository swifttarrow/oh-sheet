"""RefineService — orchestrates prompt build, LLM call, validation, and edit application.

Produces a (RefinedPerformance, RefineTrace) tuple. Plan 04's Celery worker
writes the RefinedPerformance to output.json and the RefineTrace to
llm_trace.json in the blob store.

Decision references (see .planning/phases/02-refine-service-and-pipeline-integration/02-CONTEXT.md):
  * D-08 — constructor DI (client, validator, settings); no Protocol layer.
  * D-10 — every trace stamps REFINE_PROMPT_VERSION.
  * D-12 — note IDs hand-prefixed sequential (via backend.services.refine_prompt).
  * STG-06 — service never mutates input; edits applied to a deep copy.
  * STG-07 — tenacity retry scope narrowed to the Anthropic SDK call only.
  * STG-08 — stop_reason != "end_turn" raises RefineLLMError (never auto-resume pause_turn).
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Protocol

import anthropic
from pydantic import BaseModel, Field
from shared.contracts import (
    ExpressiveNote,
    HumanizedPerformance,
    PianoScore,
    RefineCitation,
    RefinedPerformance,
    RefineEditOp,
    ScoreNote,
)
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from backend.services.refine_prompt import (
    REFINE_PROMPT_VERSION,
    _derive_note_id_map,
    build_prompt,
)
from backend.services.refine_validate import RefineValidator

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pricing (per-MTok USD; research/SUMMARY.md line 28)
# ---------------------------------------------------------------------------
# Keep hardcoded per Claude's Discretion note: "prices change rarely; a v2
# config knob can lift them later." Unknown models yield $0.0 + warning.
_PRICE_PER_MTOK_USD: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
}

# ---------------------------------------------------------------------------
# Retry policy (STG-07 — narrow scope: wraps only the SDK call)
# ---------------------------------------------------------------------------
_RETRYABLE_EXC: tuple[type[BaseException], ...] = (
    anthropic.RateLimitError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class RefineLLMError(Exception):
    """Raised on LLM response-level failure (non-end_turn stop_reason per STG-08).

    Caller (Plan 04 worker -> Plan 05 runner) translates this to
    `stage_completed` with `message="refine_skipped: <reason>"` (INT-03).
    """


# ---------------------------------------------------------------------------
# Structured output schema (Pitfall #4 — flat edit list, NOT nested score)
# ---------------------------------------------------------------------------
class RefinedEditOpList(BaseModel):
    """Target schema for anthropic.messages.parse(output_format=...).

    Flat per RESEARCH pitfall #4: nesting full HumanizedPerformance in the
    LLM output would hit Anthropic's constrained-decoding complexity limit.
    The LLM emits edits + citations; the service applies them locally.
    """
    edits: list[RefineEditOp] = Field(default_factory=list)
    citations: list[RefineCitation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Observability payload (STG-10 — llm_trace.json shape)
# ---------------------------------------------------------------------------
class RefineTraceRejectedEdit(BaseModel):
    edit: RefineEditOp
    reason: str


class RefineTraceUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class RefineTrace(BaseModel):
    """Structured observability payload; serialized to llm_trace.json.

    Schema is convention-only (not versioned with SCHEMA_VERSION) because
    it is an internal observability artifact, not a pipeline-boundary
    contract. The REFINE_PROMPT_VERSION field is the version hook (D-10)
    for future caching that keys on (song, prompt_version, model).
    """
    prompt_version: str = REFINE_PROMPT_VERSION
    prompt_system: str
    prompt_user: str
    model: str
    stop_reason: str
    raw_response_content: list[Any] = Field(default_factory=list)
    applied_edits: list[RefineEditOp] = Field(default_factory=list)
    rejected_edits: list[RefineTraceRejectedEdit] = Field(default_factory=list)
    citations: list[RefineCitation] = Field(default_factory=list)
    usage: RefineTraceUsage = Field(default_factory=RefineTraceUsage)
    estimated_cost_usd: float = 0.0
    anthropic_message_id: str = ""
    source_performance_digest: str = ""


# ---------------------------------------------------------------------------
# Settings protocol (duck-typed so tests can stub)
# ---------------------------------------------------------------------------
class _SettingsLike(Protocol):
    refine_model: str
    refine_max_tokens: int
    refine_web_search_max_uses: int
    refine_max_retries: int
    refine_ghost_velocity_max: int


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class RefineService:
    """D-08: constructor DI. No internal AsyncAnthropic construction.

    The worker (Plan 04) instantiates `AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())`
    and hands the client in. Tests pass a FakeRefineClient.
    """

    name = "refine"

    def __init__(
        self,
        *,
        client: Any,  # anthropic.AsyncAnthropic or FakeRefineClient — duck-typed on .messages.parse
        validator: RefineValidator,
        settings: _SettingsLike,
    ) -> None:
        self.client = client
        self.validator = validator
        self.settings = settings

    async def run(
        self,
        performance: HumanizedPerformance | PianoScore,
        metadata: dict[str, Any],
    ) -> tuple[RefinedPerformance, RefineTrace]:
        """Main entrypoint: build prompt, call LLM, validate, apply, trace."""
        digest = self._digest(performance)
        prompt = build_prompt(
            metadata, performance,
            web_search_max_uses=self.settings.refine_web_search_max_uses,
        )

        response = await self._call_llm(prompt)

        if response.stop_reason != "end_turn":
            # STG-08: never auto-resume pause_turn; any non-end_turn is failure.
            raise RefineLLMError(
                f"unexpected stop_reason from Anthropic: {response.stop_reason!r} "
                f"(expected 'end_turn')"
            )

        parsed: RefinedEditOpList = response.parsed
        if not isinstance(parsed, RefinedEditOpList):
            raise RefineLLMError(
                f"parsed output is not RefinedEditOpList: {type(parsed).__name__}"
            )

        applied, rejected = self.validator.validate(performance, parsed.edits)

        refined_inner = self._apply_edits(performance, applied)

        refined_perf = RefinedPerformance(
            refined_performance=refined_inner,
            edits=applied,
            citations=parsed.citations,
            model=response.model,
            source_performance_digest=digest,
        )

        trace = self._build_trace(
            prompt=prompt,
            response=response,
            applied=applied,
            rejected=rejected,
            citations=parsed.citations,
            digest=digest,
        )
        log.info(
            "refine: applied=%d rejected=%d stop_reason=%s model=%s tokens_in=%d tokens_out=%d",
            len(applied), len(rejected), response.stop_reason, response.model,
            response.usage.input_tokens, response.usage.output_tokens,
        )
        return (refined_perf, trace)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _digest(performance: HumanizedPerformance | PianoScore) -> str:
        """SHA-256 of canonical JSON — 64-char lowercase hex. Drift detector."""
        return hashlib.sha256(
            performance.model_dump_json(by_alias=False).encode("utf-8"),
        ).hexdigest()

    async def _call_llm(self, prompt: dict[str, Any]) -> Any:
        """Retry-wrapped Anthropic SDK call. NARROW SCOPE (STG-07).

        Retries only transient SDK errors; validator rejections and local
        apply-edits work are OUTSIDE this retry boundary.
        """
        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(_RETRYABLE_EXC),
                wait=wait_exponential_jitter(initial=1.0, max=20.0),
                stop=stop_after_attempt(self.settings.refine_max_retries),
                reraise=True,
            ):
                with attempt:
                    return await self.client.messages.parse(
                        model=self.settings.refine_model,
                        max_tokens=self.settings.refine_max_tokens,
                        system=prompt["system"],
                        messages=[
                            {"role": "user", "content": prompt["user"]},
                        ],
                        tools=[prompt["web_search_tool_spec"]],
                        output_format=RefinedEditOpList,
                    )
        except RetryError as exc:  # pragma: no cover — reraise=True unwraps; belt-and-braces.
            raise exc.last_attempt.exception() from exc
        # Unreachable — AsyncRetrying with reraise=True either returns from `with attempt`
        # or propagates. Explicit guard for type-checker sanity.
        raise RuntimeError("AsyncRetrying exited without value or exception")

    def _apply_edits(
        self,
        source: HumanizedPerformance | PianoScore,
        edits: list[RefineEditOp],
    ) -> HumanizedPerformance | PianoScore:
        """Deep-copy source, apply edits. STG-06 — input is never mutated.

        WR-01 fix: resolve every edit to its concrete target note object BEFORE
        applying any edits. ``_derive_note_id_map`` sorts by (onset_beat, pitch)
        and indexes sequentially, so any ``delete`` shifts the IDs of all
        subsequent notes. Looking up ``id_map[edit.target_note_id]`` after a
        delete therefore risks misdirecting a later ``modify`` at a different
        physical note than the validator/LLM approved. By binding ``target``
        up front we keep each edit pointed at the exact note the LLM saw,
        regardless of subsequent re-indexing.
        """
        working = source.model_copy(deep=True)
        # Re-derive map against the deep copy so we can mutate the copy's note objects directly.
        id_map = _derive_note_id_map(working)

        # WR-01: resolve all edit targets up front (immune to post-delete re-indexing).
        resolved: list[tuple[RefineEditOp, ExpressiveNote | ScoreNote]] = []
        for edit in edits:
            target = id_map.get(edit.target_note_id)
            if target is None:
                # Shouldn't happen — validator filtered unknown IDs. Defense-in-depth.
                log.warning("apply_edits: skipping unknown target_note_id %r", edit.target_note_id)
                continue
            resolved.append((edit, target))

        for edit, target in resolved:
            if edit.op == "delete":
                self._delete_note(working, target)
            elif edit.op == "modify":
                self._modify_note(target, edit)
        return working

    @staticmethod
    def _delete_note(
        working: HumanizedPerformance | PianoScore,
        target: ExpressiveNote | ScoreNote,
    ) -> None:
        if isinstance(working, HumanizedPerformance):
            working.expressive_notes = [
                n for n in working.expressive_notes if n is not target
            ]
        else:
            working.right_hand = [n for n in working.right_hand if n is not target]
            working.left_hand = [n for n in working.left_hand if n is not target]

    @staticmethod
    def _modify_note(target: ExpressiveNote | ScoreNote, edit: RefineEditOp) -> None:
        """Apply provided modify-payload fields to target IN-PLACE on the deep copy."""
        if edit.pitch is not None:
            target.pitch = edit.pitch
        if edit.velocity is not None:
            target.velocity = max(0, min(127, edit.velocity))
        if edit.velocity_offset is not None:
            target.velocity = max(0, min(127, target.velocity + edit.velocity_offset))
        if edit.duration_beat is not None:
            target.duration_beat = edit.duration_beat
        if edit.timing_offset_ms is not None and isinstance(target, ExpressiveNote):
            # ExpressiveNote has timing_offset_ms; ScoreNote does not.
            # For ScoreNote, we translate ms into an onset_beat adjustment using
            # a nominal 120 BPM — documented as approximate, not load-bearing
            # until Phase-3 empirical tuning.
            target.timing_offset_ms = max(
                -50.0, min(50.0, target.timing_offset_ms + edit.timing_offset_ms),
            )

    def _build_trace(
        self,
        *,
        prompt: dict[str, Any],
        response: Any,
        applied: list[RefineEditOp],
        rejected: list[tuple[RefineEditOp, str]],
        citations: list[RefineCitation],
        digest: str,
    ) -> RefineTrace:
        model = response.model
        usage = RefineTraceUsage(
            input_tokens=int(response.usage.input_tokens),
            output_tokens=int(response.usage.output_tokens),
        )
        prices = _PRICE_PER_MTOK_USD.get(model)
        if prices is None:
            log.warning("refine: no pricing for model %r; cost_estimate=0", model)
            cost = 0.0
        else:
            cost = (
                usage.input_tokens / 1_000_000.0 * prices["input"]
                + usage.output_tokens / 1_000_000.0 * prices["output"]
            )

        return RefineTrace(
            prompt_system=prompt["system"],
            prompt_user=prompt["user"],
            model=model,
            stop_reason=response.stop_reason,
            raw_response_content=list(response.content),
            applied_edits=list(applied),
            rejected_edits=[
                RefineTraceRejectedEdit(edit=e, reason=r) for e, r in rejected
            ],
            citations=list(citations),
            usage=usage,
            estimated_cost_usd=round(cost, 6),
            anthropic_message_id=str(getattr(response, "id", "")),
            source_performance_digest=digest,
        )
