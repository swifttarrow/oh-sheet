# Stack Research — LLM-Augmented Music Notation Refinement (GAU-105)

**Domain:** LLM-powered post-processing stage for symbolic music pipeline
**Researched:** 2026-04-13
**Confidence:** HIGH (core picks verified against Anthropic docs and platform.claude.com on 2026-04-13; retry/music21 picks verified via official docs and existing codebase patterns)

## Context: Additions Only

Oh Sheet's baseline stack is documented in `.planning/codebase/STACK.md` and is **not under discussion** here. The refine stage must slot into:

- Python 3.10+ (3.12 in Docker) with `pyproject.toml`-managed deps
- Celery 5.3+ prefork workers dispatched via `PipelineRunner.apply_async`
- `backend/workers/*` pattern: sync Celery task wraps `asyncio.run(service.run(...))`
- Pydantic v2.5+ contracts (`PianoScore`, `HumanizedPerformance`) at every stage boundary
- `pydantic-settings` with `OHSHEET_*` env prefix, `.env` loading
- `BlobStore` claim-check for URI-based input/output between stages

Everything below is **net-new** for the `refine` stage.

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `anthropic` | `>=0.94.0,<1.0` | Official Python SDK for Claude API (Messages, tool use, web search, structured outputs, async client) | First-party, actively maintained (v0.94.0 released 2026-04-10), bundles web search and structured outputs natively. No abstraction layer — matches the "Anthropic only, no multi-provider" decision in PROJECT.md |
| `claude-sonnet-4-6` | API alias `claude-sonnet-4-6` | Default LLM model for refine calls | Current-generation Sonnet (GA as of Mar 2026). 1M-token context, 64k output, supports web search + structured outputs + extended thinking. $3/$15 per MTok — 5x cheaper than Opus 4.6 while remaining competent at structured reasoning over score data. Overkill-avoidance: we don't need Opus-grade frontier reasoning to respell accidentals and split hands |
| `claude-opus-4-6` | API alias `claude-opus-4-6` | Optional escalation model for hard cases / A/B harness comparison | Most intelligent broadly available model. 1M context, 128k output. Only use behind an `OHSHEET_REFINE_MODEL_OVERRIDE` config knob; default stays Sonnet. At $5/$25 per MTok, reserve for A/B experiments, not default path |
| `tenacity` | `>=8.2,<10` | Retry/backoff for refine-stage resilience above the SDK's built-in 2-retry default | Industry standard Python retry library (Apache 2.0). Decorator-based, composable wait/stop/retry policies, supports both sync and async. The Anthropic SDK already retries 429/5xx/timeouts twice with exponential backoff — tenacity is the layer *above* that for our skip-on-failure semantics (PROJECT.md: "if refine fails after configured retries, emit `refine_skipped`"). `backoff` is a viable alternative but tenacity has better ecosystem momentum in 2026 |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `music21` | already at `9.1+` | Enharmonic respelling helpers (`Pitch.simplifyEnharmonic(mostCommon=True)`, `EnharmonicSimplifier`) if refine needs deterministic post-processing of LLM suggestions | Optional. Use only if the LLM returns raw pitch numbers and we need to canonicalize spelling before writing `PianoScore`. Prefer having the LLM emit correct `pitch_name` + `pitch_class` directly via structured output schema |
| `pretty_midi` | already at `0.2.10+` | No new usage expected — refine operates on `PianoScore` (beat-domain Pydantic model), not MIDI bytes | Do NOT use for refine. Refine reads/writes `PianoScore`/`HumanizedPerformance` JSON via BlobStore; pretty_midi stays inside Engrave |
| `anthropic` async client (`AsyncAnthropic`) | bundled in `anthropic>=0.94.0` | Non-blocking HTTP to Anthropic from inside the `refine` service | Call it via `AsyncAnthropic(api_key=...).messages.create(...)` inside the async `RefineService.run()`. The Celery worker already wraps the service with `asyncio.run(service.run(...))` — same pattern as every other stage in `backend/workers/*.py`. No new bridging code needed |
| `pydantic-settings` | already at `2.1+` | Load `OHSHEET_ANTHROPIC_API_KEY` from env | Extend `backend/config.py` with a `SecretStr` field. Fail-fast at app startup if `enable_refine_default=True` but key missing |
| `pydantic.SecretStr` | part of `pydantic>=2.5` | API key type so it never shows up in logs, repr, or serialized config dumps | Use for `anthropic_api_key: SecretStr \| None = None`. Call `.get_secret_value()` only inside the refine service when constructing the `AsyncAnthropic` client |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `anthropic[vertex]`/`anthropic[bedrock]` extras | NOT USED | PROJECT.md constrains us to direct Claude API. Skip Vertex/Bedrock; install plain `anthropic` |
| `pytest-httpx` / `respx` | HTTP mocking for unit tests that must not hit the real API | Mock `AsyncAnthropic` at the HTTP layer (the SDK uses `httpx.AsyncClient` internally). Alternative: inject a fake client via DI, which matches the existing codebase's `backend/storage/_blob_store` DI pattern. DI injection is cleaner — prefer it |
| `responses` lib for cassette-style tests | NOT NEEDED | `pytest-httpx`/`respx` cover async httpx; `responses` only works for the sync `requests` library |

## Installation

```bash
# Add to pyproject.toml [project] dependencies (not an extra — refine is core path when enabled):
#   "anthropic>=0.94.0,<1.0",
#   "tenacity>=8.2,<10",

# After pyproject.toml update:
pip install -e ".[dev]"

# music21 and pretty_midi are already installed by the main install (see existing STACK.md).
```

Env additions to `.env.example`:
```
# Anthropic API — required when refine is enabled on a job
OHSHEET_ANTHROPIC_API_KEY=
# Optional knobs (sane defaults live in backend/config.py)
OHSHEET_REFINE_MODEL=claude-sonnet-4-6
OHSHEET_REFINE_MAX_TOKENS=8192
OHSHEET_REFINE_WEB_SEARCH_MAX_USES=5
OHSHEET_REFINE_TIMEOUT_SECONDS=90
OHSHEET_REFINE_MAX_ATTEMPTS=3
```

## Key Decision: Web Search Tool Version

Use **`web_search_20260209`** (the latest, with dynamic filtering) rather than `web_search_20250305`.

| Field | Recommended Value | Rationale |
|-------|------------------|-----------|
| `type` | `"web_search_20260209"` | Current version, supported on Sonnet 4.6 and Opus 4.6. Dynamic filtering reduces tokens-in-context by having Claude write code to filter raw HTML before reasoning — meaningful cost savings for the grounding use case where we scrape song-info pages |
| `name` | `"web_search"` | Conventional |
| `max_uses` | `5` | Protects against runaway costs. Each search = $0.01. A typical grounding loop needs 1–3 searches ("What's the key of [song]?", "What's the time signature of [song]?"). 5 is generous headroom |
| `allowed_domains` | Initially unset; consider restricting to `["genius.com", "musicnotes.com", "ultimate-guitar.com", "wikipedia.org"]` after A/B harness results | Leave open in v1 — Claude's judgement is better than a curated list for first pass. Revisit if it cites unreliable sources |
| `user_location` | Unset | English-language songs only per PROJECT.md Out of Scope — location bias isn't needed |

**Trade-off:** Dynamic filtering requires the code execution tool to also be enabled on your Anthropic Console. If the team has not enabled it, fall back to `web_search_20250305` (no dynamic filtering, but otherwise identical API).

**Cost:** $10 per 1,000 searches + standard token costs for content pulled into context. At `max_uses=5` per job and an assumed 30% search-utilization rate, expected search cost per refined job is ~$0.015.

## Key Decision: Structured Output Strategy

**Use `client.messages.parse(output_format=RefinedPianoScore)`** — not raw `tools=[...]` with `input_schema`, and not "please return JSON" prompting.

Why:

1. **Strict schema guarantees.** Structured outputs (`output_config.format` / `output_format`) are GA as of Apr 2026 on Sonnet 4.6 and Opus 4.6. The API performs constrained decoding — Claude *cannot* emit tokens that violate the schema. No retry loops for malformed JSON, no prompt-engineered "return valid JSON" hacks.
2. **Pydantic-native.** `messages.parse()` accepts a `BaseModel` subclass directly, generates the JSON schema, validates the response, and returns `response.parsed_output` as a typed Pydantic instance. Zero custom code.
3. **Web search + structured outputs are compatible.** You can enable the web search tool *and* constrain the final text output to a JSON schema in the same call. This is the exact shape refine needs: "go search the web, then emit a `RefinedPianoScore` that conforms to our contract."

Alternative considered — **tool use with `input_schema`**: Define a fake `propose_refined_score` tool and read the arguments from the tool-use block. Viable but clunkier: you're abusing tools-as-output-channel, and you lose the `parsed_output` convenience. The `strict: true` flag on tool definitions gives equivalent guarantees if you must go this route, but structured outputs are the cleaner idiom in 2026.

Alternative considered — **plain text JSON prompting**: Not acceptable. Even with careful prompting, Claude produces malformed JSON at non-trivial rates, and every retry is a billed web-search-enabled roundtrip. Structured outputs eliminate this.

**Schema choice:** Define `RefinedPianoScore` as a Pydantic model in `shared/shared/contracts.py` that is either (a) a strict subset of `PianoScore` with additional `provenance: list[Citation]` field, or (b) a shape-compatible twin of `PianoScore`. Decision deferred to plan-phase — research confirms both are feasible.

## Key Decision: Async Calls Inside Celery Workers

**Follow the existing pattern. Do not introduce `asgiref`, `celery[asyncio]`, or custom event-loop plumbing.**

Every existing worker (`backend/workers/{ingest,transcribe,arrange,humanize,engrave}.py`) uses this shape:

```python
import asyncio
from backend.services.refine import RefineService

@celery_app.task(name="stages.refine")
def refine_task(...):
    service = RefineService(...)
    # asyncio.run() is safe with Celery's default prefork pool; breaks with gevent/eventlet.
    result = asyncio.run(service.run(humanized, job_id=job_id))
    return result
```

Inside `RefineService.run()`, call `AsyncAnthropic(...).messages.parse(...)` with `await`.

Why this works:
- Celery's **prefork** pool (the default, confirmed in current production config per STACK.md) creates one Python process per worker. `asyncio.run()` creates a fresh event loop per task, tears it down on return. Safe.
- This is identical to every other stage — no special-casing for refine.
- **Known pitfall:** If the team ever moves to `gevent` or `eventlet` pools, `asyncio.run()` fights with greenlet-based I/O. Stay on prefork (or add `threads`), which is the current default.

**Rejected alternatives:**
- `asgiref.sync.async_to_sync()` — works, but creates a new loop every call anyway; no benefit over `asyncio.run()` for our use case, and adds a dep.
- Sync `Anthropic` client inside sync task — works, but defeats the SDK's `httpx.AsyncClient` benefits and is inconsistent with existing services that are all `async def run(...)`.
- Celery's experimental async task support (GH #3884, #6552, #9058 — still open, no GA as of 2026) — not production-ready.

## Key Decision: Retry & Timeout Strategy

Two-layer retry: **SDK handles transient HTTP failures; tenacity handles semantic/business retries.**

**Layer 1 — SDK built-in (already on by default):**
```python
AsyncAnthropic(
    api_key=settings.anthropic_api_key.get_secret_value(),
    max_retries=2,  # default; retries 429, 408, 409, 5xx, connection errors with exponential backoff
    timeout=90.0,   # seconds; APITimeoutError on expiry, counts toward max_retries
)
```

**Layer 2 — tenacity around the whole service call** (for retry on validation failure, empty output, refusal, semantic-wrong output):
```python
from tenacity import (
    retry, stop_after_attempt, wait_exponential_jitter,
    retry_if_exception_type, before_sleep_log,
)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=30),
    retry=retry_if_exception_type((RefineValidationError, anthropic.APIError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def _call_refine(...): ...
```

After the tenacity budget is exhausted, catch the final exception in `RefineService.run()`, emit `refine_skipped`, and return the unchanged `HumanizedPerformance` for downstream engrave (per the failure-semantics constraint in PROJECT.md).

**Why two layers:** The SDK retries on HTTP status codes and nothing else. Our retry budget for "Claude returned a valid-but-semantically-wrong refinement" (e.g., mutated a locked field, dropped a required section) is separate and belongs in application code.

**Why `wait_exponential_jitter`:** Prevents thundering-herd on rate-limit recovery. Jitter is the industry standard for LLM APIs in 2026.

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| `anthropic` SDK directly | LangChain / LlamaIndex | Never for this milestone. PROJECT.md says Anthropic-only, no multi-provider abstraction. LangChain's `ChatAnthropic` adds a 300ms import cost, a stale-by-design wrapper around Claude's tool/structured-output APIs, and a deep dep tree. Skip |
| `anthropic` SDK directly | Raw `httpx` POST to `/v1/messages` | Only if the SDK is unavailable (it isn't) or if a new Claude feature ships before SDK support lands (unlikely, SDK is first-party). SDK handles auth, retries, streaming, tool orchestration, and structured output parsing for free |
| `anthropic` SDK directly | `instructor` library on top of Anthropic | `instructor` (python.useinstructor.com) is elegant for structured output but predates Anthropic's first-party structured outputs GA (Apr 2026). Now that `messages.parse()` is native, `instructor` is a redundant layer |
| `claude-sonnet-4-6` default | `claude-haiku-4-5` ($1/$5 per MTok, faster) | If A/B harness shows Haiku is "good enough" for the modify/delete operations and we need to cut per-job cost. Haiku 4.5 supports web search + structured outputs but has smaller 200k context and Feb-2025 knowledge cutoff. Revisit post-milestone |
| `claude-sonnet-4-6` default | `claude-opus-4-6` default | Use when the refine reasoning proves too hard for Sonnet — sustained multi-step reasoning over long scores, or when A/B harness shows Opus meaningfully wins. Engineered for accuracy; worth the 5x price only if measurable quality delta |
| `tenacity` | `backoff` library | Also fine (both are decorator-based, both widely used). `tenacity` wins on ecosystem momentum, better async support, and composable policy objects |
| `tenacity` | `stamina` (newer retry lib) | Viable but less widely adopted in 2026 codebases. `tenacity` has larger install base and this codebase doesn't need stamina's type-first ergonomics |
| Plain env-var `OHSHEET_ANTHROPIC_API_KEY` | Secrets manager (GCP Secret Manager, AWS Secrets Manager) | In production on Cloud Run, injecting secrets from GCP Secret Manager as env vars at container startup is the right long-term move. For this milestone, env var via `.env` + Cloud Run environment variables is sufficient and matches the current `OHSHEET_*` convention. Can migrate later with zero code change |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| LangChain / LangGraph | Multi-provider abstraction explicitly out of scope (PROJECT.md). Adds indirection, slows debugging, lags Claude features by 1–3 releases | Direct `anthropic` SDK |
| `openai` SDK pointed at an Anthropic-compatible endpoint | Loses tool use shape, loses web search, loses structured outputs. Pure downside | Direct `anthropic` SDK |
| Claude Sonnet 4.5 / Opus 4.5 / Opus 4.1 (legacy models) | Superseded by 4.6 generation (Nov 2025 – Mar 2026). Equal or lower price for 4.6 models, bigger context, newer knowledge cutoff | `claude-sonnet-4-6` / `claude-opus-4-6` |
| Claude Haiku 3 (`claude-3-haiku-20240307`) | **Deprecated; retires April 19, 2026** (per platform.claude.com/docs/en/about-claude/model-deprecations). Also too small for multi-stage reasoning over a full score | `claude-haiku-4-5` if you want a cheap/fast tier |
| Prompting Claude to "return JSON with these fields" | Non-guaranteed compliance; forces retry loops; wastes web-search budget on retries | Structured outputs via `messages.parse()` |
| Celery gevent/eventlet pool for refine workers | Breaks `asyncio.run(service.run(...))` pattern used by every existing stage | Stick with default prefork (or `threads`) pool |
| `backoff` decorator with `backoff.expo` and no jitter | Thundering-herd risk on rate limits; missing retry-on-specific-exception ergonomics that tenacity has | `tenacity` with `wait_exponential_jitter` |
| `AsyncAnthropic` inside a sync Celery task without `asyncio.run` | Coroutine never awaited; silent no-op | `asyncio.run(service.run(...))` in the task wrapper |
| Token-counting libraries (`tokenator`, `tokencost`, `token-analyzer`) | Adds a dep for something the SDK response already contains (`response.usage.input_tokens`, `output_tokens`, `server_tool_use.web_search_requests`). Log those directly | Read `response.usage` off the message object; log to structured logs or Prometheus |
| Caching libraries (redis-cache, functools.lru_cache) for LLM responses | PROJECT.md explicitly lists "LLM response caching" as Out of Scope for v1. Revisit later | No cache in v1 |

## Stack Patterns by Variant

**If refine is enabled on a full/audio_upload/midi_upload job:**
- Sits between `humanize` and `engrave` in the execution plan
- Reads `HumanizedPerformance` from blob, returns `RefinedHumanizedPerformance` (name TBD in plan-phase) to blob
- `Engrave` reads refined output if present, falls back to humanized on `refine_skipped`

**If refine is enabled on a `sheet_only` variant:**
- Sits between `arrange` and `engrave` (no humanize stage in sheet_only)
- Reads `PianoScore` directly, returns `RefinedPianoScore`
- Engrave must accept both shapes (existing engrave already handles `HumanizedPerformance`-or-`PianoScore` dispatch — verify in plan-phase)

**If `enable_refine=false` (default):**
- Execution plan excludes the refine step entirely — no Celery dispatch, no Anthropic call, no cost
- `OHSHEET_ANTHROPIC_API_KEY` may be unset; app still boots. The fail-fast check only fires when a job with `enable_refine=true` is submitted (or at startup if you choose strict mode — recommend startup-fail only when the key is present-but-invalid, not when missing with refine disabled)

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `anthropic>=0.94.0` | Python `>=3.9` including 3.10/3.11/3.12/3.13/3.14 | Matches codebase's 3.10+ minimum; Docker uses 3.12; CI runs 3.13 — all supported |
| `anthropic>=0.94.0` | `pydantic>=2.5` (existing) | Compatible — the SDK uses Pydantic v2 internally. `messages.parse(output_format=YourPydanticModel)` requires Pydantic v2 |
| `anthropic>=0.94.0` | `httpx` (existing transitive dep) | SDK uses `httpx.AsyncClient` internally. Already in dep tree via FastAPI/pytest-httpx |
| `anthropic>=0.94.0` | `celery>=5.3` (existing) | No direct interaction — SDK is called from inside service, Celery wraps with `asyncio.run`. Version combos are decoupled |
| `tenacity>=8.2` | Python `>=3.8` | Compatible with all supported versions |
| `web_search_20260209` tool | `claude-sonnet-4-6`, `claude-opus-4-6`, Claude Mythos Preview | NOT available on Sonnet 4.5, Opus 4.5, or earlier. If fallback to older model needed, use `web_search_20250305` instead |
| Structured outputs (`output_config.format`) | `claude-sonnet-4-6`, `claude-opus-4-6`, `claude-sonnet-4-5`, `claude-opus-4-5`, `claude-haiku-4-5` | Wide compatibility. Note: previously required `anthropic-beta: structured-outputs-2025-11-13` header; that's now optional as of GA |
| `music21 9.1+` | existing | No change needed for refine. `simplifyEnharmonic()` has been stable since v7 |

## API Key Lifecycle (Recommended Pattern)

```python
# backend/config.py
from pydantic import SecretStr
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OHSHEET_", env_file=".env")

    anthropic_api_key: SecretStr | None = None
    refine_model: str = "claude-sonnet-4-6"
    refine_max_tokens: int = 8192
    refine_web_search_max_uses: int = 5
    refine_timeout_seconds: float = 90.0
    refine_max_attempts: int = 3
```

```python
# backend/services/refine.py
import anthropic
from anthropic import AsyncAnthropic

class RefineService:
    def __init__(self, settings: Settings, blob: BlobStore):
        if settings.anthropic_api_key is None:
            raise RefineConfigError("OHSHEET_ANTHROPIC_API_KEY not set")
        self._client = AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value(),
            max_retries=2,
            timeout=settings.refine_timeout_seconds,
        )
        self._settings = settings
        self._blob = blob

    async def run(self, humanized: HumanizedPerformance, *, job_id: str) -> RefinedHumanizedPerformance:
        response = await self._client.messages.parse(
            model=self._settings.refine_model,
            max_tokens=self._settings.refine_max_tokens,
            messages=[...],
            tools=[{
                "type": "web_search_20260209",
                "name": "web_search",
                "max_uses": self._settings.refine_web_search_max_uses,
            }],
            output_format=RefinedPianoScore,  # Pydantic model
        )
        return response.parsed_output
```

**Why `SecretStr | None` with a nullable default:** Allows the app to boot without the key (so existing jobs that don't use refine still work). The service's constructor raises if anyone tries to use refine without a key, which surfaces as `refine_skipped` per the failure-semantics constraint. Fail fast at job-submit time rather than at startup.

## Sources

- **Anthropic PyPI** — https://pypi.org/project/anthropic/ — confirmed v0.94.0 (2026-04-10) as current release; Python 3.9–3.14 supported. HIGH confidence
- **Anthropic GitHub releases** — https://github.com/anthropics/anthropic-sdk-python/releases — confirmed v0.94.0 release notes and changelog through Apr 2026. HIGH confidence
- **Anthropic models overview** — https://platform.claude.com/docs/en/about-claude/models/overview — confirmed current model IDs (`claude-sonnet-4-6`, `claude-opus-4-6`, `claude-haiku-4-5-20251001`), pricing ($3/$15 Sonnet, $5/$25 Opus, $1/$5 Haiku), context windows (1M for 4.6, 200k for Haiku), knowledge cutoffs, and deprecation of Haiku 3 on 2026-04-19. HIGH confidence
- **Anthropic web search tool docs** — https://platform.claude.com/docs/en/build-with-claude/tool-use/web-search-tool — confirmed `web_search_20260209` is current version with dynamic filtering, pricing at $10/1k searches, full parameter list (`max_uses`, `allowed_domains`, `blocked_domains`, `user_location`), and Python example. HIGH confidence
- **Anthropic structured outputs docs** — https://platform.claude.com/docs/en/build-with-claude/structured-outputs — confirmed GA on Sonnet 4.6 + Opus 4.6 + 4.5 + Haiku 4.5, `messages.parse()` with Pydantic models, `output_config.format` replacing old `output_format` beta param, compatibility with tool use. HIGH confidence
- **Anthropic SDK retry docs** — DeepWiki "Request Lifecycle and Error Handling" — confirmed `max_retries=2` default, `with_options(max_retries=N)`, automatic retry on 408/409/429/5xx/connection/timeout. HIGH confidence
- **Tenacity docs** — https://tenacity.readthedocs.io/ and https://github.com/jd/tenacity — confirmed API for `retry`, `stop_after_attempt`, `wait_exponential_jitter`, async support. HIGH confidence
- **Existing codebase** — `/Users/jackjiang/GitHub/oh-sheet/backend/workers/*.py` (grep confirms `asyncio.run(service.run(...))` pattern already in use across 7 workers with comment "safe with Celery's default prefork pool; breaks with gevent/eventlet"). HIGH confidence
- **Existing codebase** — `/Users/jackjiang/GitHub/oh-sheet/shared/shared/contracts.py` (confirms `PianoScore` and `HumanizedPerformance` are Pydantic v2 `BaseModel` classes with `schema_version` field). HIGH confidence
- **music21 docs** — https://music21.org/music21docs/moduleReference/modulePitch.html and moduleAnalysisEnharmonics — confirmed `Pitch.simplifyEnharmonic(mostCommon=True)` API and `EnharmonicSimplifier` class. MEDIUM confidence (docs verified; not yet tested against refine use case)
- **Celery async support discussions** — https://github.com/celery/celery/discussions/9058, https://github.com/celery/celery/issues/3884, #6552 — confirmed no native async task support in Celery 5.x as of 2026; `asyncio.run()` in prefork pool is the community-standard workaround. HIGH confidence
- **Pydantic SecretStr docs** — https://docs.pydantic.dev/latest/concepts/pydantic_settings/ — confirmed `SecretStr` masks values in `repr()` and serialization; `.get_secret_value()` is the escape hatch. HIGH confidence

---
*Stack research for: LLM-augmented music notation refinement (GAU-105)*
*Researched: 2026-04-13*
