"""Interpret stage — LLM-driven arrangement hint extraction.

Consumes a ``TranscriptionResult`` plus a free-form user arrangement prompt,
asks Claude to interpret the prompt relative to the song's musical context,
and returns an enriched ``TranscriptionResult`` with ``arrangement_hints``
populated.

Never raises on LLM failure: on any error / timeout / invalid response /
disabled configuration, the input is returned unchanged with a warning
appended to ``quality.warnings``.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError
from shared.contracts import ArrangementHints, QualitySignal, TranscriptionResult

from backend.config import settings
from backend.services.interpret_prompt import (
    PROMPT_VERSION as _PROMPT_VERSION,  # noqa: F401 — imported for version visibility
)
from backend.services.interpret_prompt import (
    SYSTEM_PROMPT,
    build_user_prompt,
    submit_arrangement_hints_tool_schema,
)
from backend.services.refine import _clamp_hint

log = logging.getLogger(__name__)

_HINT_MAX_LEN = 200


def _clamp_prompt(s: str, max_len: int) -> str:
    """Strip control chars, collapse whitespace, and truncate the user prompt."""
    cleaned = "".join(ch if ch.isprintable() or ch == " " else " " for ch in s)
    cleaned = " ".join(cleaned.split())
    return cleaned[:max_len]


def _build_txr_summary(txr: TranscriptionResult) -> dict[str, Any]:
    """Extract a compact musical summary from a TranscriptionResult for the prompt."""
    analysis = txr.analysis

    key = analysis.key
    time_sig = analysis.time_signature
    tempo_bpm: float | str = (
        analysis.tempo_map[0].bpm if analysis.tempo_map else "unknown"
    )

    # Duration: max offset_sec across all tracks
    duration_sec: float = 0.0
    for track in txr.midi_tracks:
        for note in track.notes:
            if note.offset_sec > duration_sec:
                duration_sec = note.offset_sec

    chord_count = len(analysis.chords)
    section_labels = [s.label.value for s in analysis.sections]

    return {
        "key": key,
        "time_signature": time_sig,
        "tempo_bpm": tempo_bpm,
        "duration_sec": duration_sec,
        "chord_count": chord_count,
        "section_labels": section_labels,
    }


class InterpretService:
    name = "interpret"

    def __init__(
        self,
        *,
        blob_store: Any | None = None,
        client: Any | None = None,
    ) -> None:
        self.blob_store = blob_store
        self._client = client

    # ---- public entrypoint -------------------------------------------------

    async def run(
        self,
        payload: TranscriptionResult,
        *,
        prompt: str,
        title_hint: str | None = None,
        artist_hint: str | None = None,
    ) -> TranscriptionResult:
        # Short-circuit when disabled or no API key
        if not settings.interpret_enabled:
            log.info("interpret: disabled via settings, passing through")
            return self._with_warning(payload, "interpret: disabled")

        if not settings.anthropic_api_key:
            log.info("interpret: no API key configured, passing through")
            return self._with_warning(payload, "interpret: disabled")

        # Sanitize user-controlled strings
        prompt = _clamp_prompt(prompt, settings.interpret_prompt_max_chars)
        if not prompt:
            log.info("interpret: empty prompt after clamping, passing through")
            return self._with_warning(payload, "interpret: disabled")

        title_hint = _clamp_hint(title_hint, _HINT_MAX_LEN)
        artist_hint = _clamp_hint(artist_hint, _HINT_MAX_LEN)

        log.info(
            "interpret: start prompt_len=%d title_hint=%r artist_hint=%r",
            len(prompt), title_hint, artist_hint,
        )

        try:
            hints = await self._call_llm(payload, prompt, title_hint, artist_hint)
        except Exception as exc:  # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}"
            log.warning("interpret: LLM call failed (%s), passing through", reason)
            return self._with_warning(payload, f"interpret: {reason}")

        if hints is None:
            log.warning("interpret: LLM returned no submit_arrangement_hints call")
            return self._with_warning(payload, "interpret: no hints returned")

        log.info("interpret: done hints=%r", hints)
        new_quality = QualitySignal(
            overall_confidence=payload.quality.overall_confidence,
            warnings=list(payload.quality.warnings),
        )
        return payload.model_copy(update={
            "arrangement_hints": hints,
            "quality": new_quality,
        })

    # ---- LLM plumbing ------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from anthropic import AsyncAnthropic  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(f"anthropic SDK not installed: {exc}") from exc
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        return self._client

    async def _call_llm(
        self,
        payload: TranscriptionResult,
        prompt: str,
        title_hint: str | None,
        artist_hint: str | None,
    ) -> ArrangementHints | None:
        txr_summary = _build_txr_summary(payload)
        user_prompt = build_user_prompt(prompt, txr_summary, title_hint, artist_hint)
        tool_schema = submit_arrangement_hints_tool_schema()
        client = self._get_client()

        response = await client.messages.create(
            model=settings.interpret_model,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=[tool_schema],
            tool_choice={"type": "tool", "name": "submit_arrangement_hints"},
            messages=[{"role": "user", "content": user_prompt}],
        )

        for block in response.content:
            if (
                getattr(block, "type", None) == "tool_use"
                and getattr(block, "name", None) == "submit_arrangement_hints"
            ):
                raw = block.input
                raw_dict = dict(raw) if isinstance(raw, dict) else json.loads(str(raw))
                try:
                    return ArrangementHints.model_validate(raw_dict)
                except ValidationError as exc:
                    log.warning("interpret: ArrangementHints validation failed: %s", exc)
                    return None

        log.warning("interpret: no submit_arrangement_hints tool_use block in response")
        return None

    # ---- helpers -----------------------------------------------------------

    def _with_warning(
        self,
        payload: TranscriptionResult,
        warning: str,
    ) -> TranscriptionResult:
        new_quality = QualitySignal(
            overall_confidence=payload.quality.overall_confidence,
            warnings=[*payload.quality.warnings, warning],
        )
        return payload.model_copy(update={"quality": new_quality})
