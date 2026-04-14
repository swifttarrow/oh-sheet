"""Refine stage — LLM-driven score metadata annotation.

Consumes a ``PianoScore`` or ``HumanizedPerformance`` envelope, asks
Claude to research the song and return metadata annotations (title,
composer, key, tempo marking, sections, repeats), and merges those
annotations into ``ScoreMetadata`` before handing the envelope
downstream to engrave.

Never raises on LLM failure: on any error / timeout / invalid response
the input is returned unchanged with a warning appended to
``quality.warnings`` (when the envelope is a HumanizedPerformance —
PianoScore has no warnings field).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from shared.contracts import (
    HumanizedPerformance,
    PianoScore,
    QualitySignal,
    Repeat,
    ScoreSection,
    SectionLabel,
)

from backend.config import settings
from backend.services.refine_prompt import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_user_prompt,
    submit_refinements_tool_schema,
    web_search_tool_schema,
)

log = logging.getLogger(__name__)


_VALID_SECTION_LABELS = {e.value for e in SectionLabel}
_VALID_REPEAT_KINDS = {"simple", "with_endings"}
_MAX_BEATS = 1_000_000
_VALID_TS_DENOMINATORS = {1, 2, 4, 8, 16, 32}
_HINT_MAX_LEN = 200


def _clamp_hint(s: str | None, max_len: int = _HINT_MAX_LEN) -> str | None:
    """Strip control chars and truncate a user-controlled hint before it
    reaches the LLM prompt. Bounds token cost and reduces the prompt-injection
    surface (``repr()`` alone is not a security boundary).
    """
    if s is None:
        return None
    cleaned = "".join(ch if ch.isprintable() or ch == " " else " " for ch in s)
    cleaned = " ".join(cleaned.split())
    return cleaned[:max_len] or None


class RefineService:
    name = "refine"

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
        payload: HumanizedPerformance | PianoScore,
        *,
        title_hint: str | None = None,
        artist_hint: str | None = None,
        filename_hint: str | None = None,
    ) -> HumanizedPerformance | PianoScore:
        title_hint = _clamp_hint(title_hint)
        artist_hint = _clamp_hint(artist_hint)
        filename_hint = _clamp_hint(filename_hint)

        log.info(
            "refine: start title_hint=%r artist_hint=%r filename_hint=%r humanized=%s",
            title_hint, artist_hint, filename_hint, isinstance(payload, HumanizedPerformance),
        )

        cache_key = self._cache_key(payload, title_hint, artist_hint, filename_hint)
        cached = self._cache_get(cache_key)
        if cached is not None:
            log.info("refine: cache hit key=%s", cache_key[:12])
            return self._merge(payload, cached)

        score = payload.score if isinstance(payload, HumanizedPerformance) else payload
        try:
            refinements = await asyncio.wait_for(
                self._call_llm(score, title_hint, artist_hint, filename_hint),
                timeout=settings.refine_budget_sec,
            )
        except TimeoutError:
            log.warning(
                "refine: budget exceeded (%ss), passing through",
                settings.refine_budget_sec,
            )
            return self._with_warning(
                payload, "refine: LLM budget exceeded, passing through",
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("refine: LLM call failed (%s), passing through", exc)
            return self._with_warning(
                payload,
                f"refine: LLM unavailable ({type(exc).__name__}), passing through",
            )

        if not refinements:
            log.warning("refine: LLM returned no submit_refinements call, passing through")
            return self._with_warning(
                payload, "refine: LLM produced no refinements, passing through",
            )

        self._cache_put(cache_key, refinements)
        merged = self._merge(payload, refinements)
        log.info("refine: done applied_fields=%d", len(refinements))
        return merged

    # ---- LLM plumbing ------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from anthropic import AsyncAnthropic  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(f"anthropic SDK not installed: {exc}") from exc
        api_key = settings.anthropic_api_key
        if not api_key:
            raise RuntimeError("OHSHEET_ANTHROPIC_API_KEY not set")
        self._client = AsyncAnthropic(
            api_key=api_key,
            timeout=settings.refine_call_timeout_sec,
        )
        return self._client

    async def _call_llm(
        self,
        score: PianoScore,
        title_hint: str | None,
        artist_hint: str | None,
        filename_hint: str | None,
    ) -> dict[str, Any] | None:
        user_prompt = build_user_prompt(
            title_hint=title_hint,
            artist_hint=artist_hint,
            filename_hint=filename_hint,
            score=score,
        )
        tools = [
            web_search_tool_schema(settings.refine_max_searches),
            submit_refinements_tool_schema(),
        ]
        client = self._get_client()

        # 3 attempts total — backoffs before attempts 2 and 3 only.
        backoffs = [0.0, 1.0, 4.0]
        last_exc: BaseException | None = None
        for attempt, delay in enumerate(backoffs):
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                response = await client.messages.create(
                    model=settings.refine_model,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=tools,
                    # Force a tool call on every turn — prevents the model from
                    # returning a plain text block instead of submit_refinements.
                    # Can't pin to ``submit_refinements`` specifically because
                    # that would block the preceding web_search calls.
                    tool_choice={"type": "any"},
                    messages=[{"role": "user", "content": user_prompt}],
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if not _is_transient(exc) or attempt == len(backoffs) - 1:
                    raise
                log.warning(
                    "refine: attempt %d failed (%s), retrying",
                    attempt + 1, exc,
                )
                continue

            for block in response.content:
                if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "submit_refinements":
                    raw = block.input
                    return dict(raw) if isinstance(raw, dict) else json.loads(str(raw))
            return None

        if last_exc is not None:
            raise last_exc
        return None

    # ---- caching -----------------------------------------------------------

    def _cache_key(
        self,
        payload: HumanizedPerformance | PianoScore,
        title_hint: str | None,
        artist_hint: str | None,
        filename_hint: str | None,
    ) -> str:
        canon = json.dumps(
            payload.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        h = hashlib.sha256()
        h.update(canon.encode("utf-8"))
        h.update(PROMPT_VERSION.encode("utf-8"))
        h.update(settings.refine_model.encode("utf-8"))
        # Hints shape the rendered prompt, so they must be part of the key —
        # otherwise a later run with a different filename/title hint returns
        # the stale cached response, silently defeating the filename-fallback
        # feature.
        h.update(f"|title={title_hint or ''}".encode())
        h.update(f"|artist={artist_hint or ''}".encode())
        h.update(f"|filename={filename_hint or ''}".encode())
        return h.hexdigest()

    def _cache_uri(self, key: str) -> str | None:
        """Build the cache URI for a LocalBlobStore-backed store only.

        Returns ``None`` for other store types (treated as cache-disabled).
        """
        root = getattr(self.blob_store, "root", None)
        if root is None:
            return None
        return (Path(root) / f"refine-cache/{key}.json").as_uri()

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        if self.blob_store is None:
            return None
        uri = self._cache_uri(key)
        if uri is None or not self.blob_store.exists(uri):
            return None
        try:
            return self.blob_store.get_json(uri)
        except Exception as exc:  # noqa: BLE001
            log.warning("refine: cache read failed (%s), ignoring", exc)
            return None

    def _cache_put(self, key: str, data: dict[str, Any]) -> None:
        if self.blob_store is None:
            return
        try:
            self.blob_store.put_json(f"refine-cache/{key}.json", data)
        except Exception as exc:  # noqa: BLE001
            log.warning("refine: cache write failed (%s), continuing", exc)

    # ---- merge + fallback --------------------------------------------------

    def _with_warning(
        self,
        payload: HumanizedPerformance | PianoScore,
        warning: str,
    ) -> HumanizedPerformance | PianoScore:
        if isinstance(payload, HumanizedPerformance):
            new_quality = QualitySignal(
                overall_confidence=payload.quality.overall_confidence,
                warnings=[*payload.quality.warnings, warning],
            )
            return payload.model_copy(update={"quality": new_quality})
        # PianoScore has no warnings field — just return unchanged.
        return payload

    def _merge(
        self,
        payload: HumanizedPerformance | PianoScore,
        refinements: dict[str, Any],
    ) -> HumanizedPerformance | PianoScore:
        score = payload.score if isinstance(payload, HumanizedPerformance) else payload
        md = score.metadata
        update: dict[str, Any] = {}

        if "title" in refinements and refinements["title"] is not None:
            update["title"] = str(refinements["title"])[:200]
        if "composer" in refinements and refinements["composer"] is not None:
            update["composer"] = str(refinements["composer"])[:200]
        if "arranger" in refinements and refinements["arranger"] is not None:
            update["arranger"] = str(refinements["arranger"])[:200]
        if "tempo_marking" in refinements and refinements["tempo_marking"] is not None:
            update["tempo_marking"] = str(refinements["tempo_marking"])[:100]
        if "staff_split_hint" in refinements:
            try:
                v = int(refinements["staff_split_hint"])
            except (TypeError, ValueError):
                v = None
            if v is not None and 0 <= v <= 127:
                update["staff_split_hint"] = v
        if "key_signature" in refinements and isinstance(refinements["key_signature"], str):
            update["key"] = refinements["key_signature"]
        if "time_signature" in refinements:
            ts = refinements["time_signature"]
            if isinstance(ts, (list, tuple)) and len(ts) == 2:
                try:
                    num, den = int(ts[0]), int(ts[1])
                except (TypeError, ValueError):
                    num = den = 0
                if 1 <= num <= 32 and den in _VALID_TS_DENOMINATORS:
                    update["time_signature"] = (num, den)
        if "tempo_bpm" in refinements and md.tempo_map:
            try:
                new_bpm = float(refinements["tempo_bpm"])
            except (TypeError, ValueError):
                new_bpm = None
            if new_bpm is not None and new_bpm > 0:
                first = md.tempo_map[0].model_copy(update={"bpm": new_bpm})
                update["tempo_map"] = [first, *md.tempo_map[1:]]
        if "sections" in refinements:
            parsed = _parse_sections(refinements["sections"])
            if parsed:
                update["sections"] = parsed
        if "repeats" in refinements:
            parsed_r = _parse_repeats(refinements["repeats"])
            update["repeats"] = parsed_r

        new_md = md.model_copy(update=update)
        new_score = score.model_copy(update={"metadata": new_md})
        if isinstance(payload, HumanizedPerformance):
            return payload.model_copy(update={"score": new_score})
        return new_score


def _parse_sections(items: Any) -> list[ScoreSection]:
    if not isinstance(items, list):
        return []
    out: list[ScoreSection] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            start = float(it["start_beat"])
            end = float(it["end_beat"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= start < end <= _MAX_BEATS):
            continue
        label_raw = str(it.get("label", "other")).lower()
        label = SectionLabel(label_raw) if label_raw in _VALID_SECTION_LABELS else SectionLabel.OTHER
        custom = it.get("custom_label")
        custom_str = str(custom)[:100] if isinstance(custom, str) and custom else None
        out.append(ScoreSection(
            start_beat=start,
            end_beat=end,
            label=label,
            custom_label=custom_str,
        ))
    return out


def _parse_repeats(items: Any) -> list[Repeat]:
    if not isinstance(items, list):
        return []
    out: list[Repeat] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        kind = str(it.get("kind", ""))
        if kind not in _VALID_REPEAT_KINDS:
            continue
        try:
            start = float(it["start_beat"])
            end = float(it["end_beat"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= start < end <= _MAX_BEATS):
            continue
        out.append(Repeat(start_beat=start, end_beat=end, kind=kind))  # type: ignore[arg-type]
    return out


def _is_transient(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if any(s in msg for s in ("timeout", "overloaded", "rate limit", "connection")):
        return True
    status = getattr(exc, "status_code", None)
    try:
        return status is not None and 500 <= int(status) < 600
    except (TypeError, ValueError):
        return False
