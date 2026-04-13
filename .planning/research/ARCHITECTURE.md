# Architecture Research — LLM Refine Stage Integration

**Domain:** Brownfield integration — adding an LLM-augmented refinement stage to an existing 5-stage Celery pipeline (ingest → transcribe → arrange → humanize → engrave) for piano sheet music generation
**Researched:** 2026-04-13
**Confidence:** HIGH (integration decisions grounded in existing codebase; LLM-specific patterns verified against Anthropic Python SDK docs and Celery community guidance)

## Guiding Principle

This is a **brownfield integration**, not greenfield architecture. The existing pipeline is already a well-factored async Celery pipeline with claim-check blob storage, WebSocket fan-out, and per-variant execution plans. The refine stage must **slot into that pattern** — same shape as humanize, same dispatch mechanism, same event schema. The only novelty is (a) an external LLM dependency and (b) skip-on-failure semantics. Every architectural decision below optimizes for **matching existing conventions** over importing external LLM-pipeline best practices.

## System Overview

### The Refine Stage in Pipeline Context

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     API Layer (FastAPI + WebSocket)                          │
│  POST /v1/jobs          WS /v1/jobs/{id}/ws         GET /v1/artifacts/{id}   │
│  (JobCreateRequest      (JobEvent stream —          (PDF, MIDI, MusicXML,    │
│   +enable_refine)        includes "refine" stage)    +maybe refine.json)     │
└───────┬──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    JobManager  (in-memory record + pub/sub)                  │
│   emits JobEvent → fans out to asyncio.Queue subscribers                     │
└───────┬──────────────────────────────────────────────────────────────────────┘
        │  .submit(bundle, config)
        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│       PipelineRunner  (backend/jobs/runner.py — walks execution plan)        │
│                                                                              │
│   STEP_TO_TASK: { ..., "humanize": "humanize.run",                           │
│                   "refine":   "refine.run",   ← NEW                          │
│                   "engrave":  "engrave.run" }                                │
│                                                                              │
│   Per-variant plans include "refine" after humanize (or after arrange for    │
│   sheet_only) WHEN config.enable_refine is True.                             │
└───────┬──────────────────────────────────────────────────────────────────────┘
        │  apply_async(job_id, payload_uri)
        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                     Celery Workers (Redis broker)                            │
│                                                                              │
│  ingest.run → transcribe.run → arrange.run → humanize.run                    │
│                                                   │                          │
│                                                   ▼                          │
│                                           ┌───────────────┐                  │
│                                           │  refine.run   │ ← NEW queue      │
│                                           │   worker      │    "refine"      │
│                                           └───────┬───────┘                  │
│                                                   │                          │
│                                                   ▼ (or bypass on failure)   │
│                                              engrave.run                     │
└───────┬──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                  Anthropic API (Claude + web_search tool)                    │
│   Called synchronously via anthropic.Anthropic() client.messages.create()    │
│   inside the refine.run Celery task (same shape as asyncio.run in humanize)  │
└──────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      BlobStore (Claim-Check)                                 │
│                                                                              │
│  jobs/{job_id}/humanize/output.json   (HumanizedPerformance)                 │
│  jobs/{job_id}/refine/input.json      (HumanizedPerformance, copied in)      │
│  jobs/{job_id}/refine/output.json     (RefinedPerformance — see Contract     │
│                                        Decision below)                       │
│  jobs/{job_id}/refine/llm_trace.json  (raw LLM request/response + tool log)  │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Where It Lives |
|-----------|---------------|----------------|
| **RefineService** | Core logic: build prompt from score, call Claude with web_search tool, parse response, apply modify+delete edits, validate output | `backend/services/refine.py` (new) |
| **refine Celery task** | Claim-check wrapper: read input URI, call service, write output URI (+ trace artifact) | `backend/workers/refine.py` (new) |
| **RefinePrompt builder** | Turn HumanizedPerformance + song metadata into a structured prompt | `backend/services/refine_prompt.py` (new) |
| **RefineValidator** | Enforce modify+delete authority at the edit-operation level (no note additions) | `backend/services/refine_validate.py` (new) |
| **PipelineRunner** | Insert `refine` step into execution plan when `config.enable_refine`; handle skip-on-failure | `backend/jobs/runner.py` (modify) |
| **PipelineConfig** | New field `enable_refine: bool = False`; include `refine` in `get_execution_plan()` | `shared/shared/contracts.py` (modify) |
| **JobCreateRequest** | New field `enable_refine: bool = False`; thread through to `PipelineConfig` | `backend/api/routes/jobs.py` (modify) |
| **EngraveService/worker** | Accept `RefinedPerformance` in addition to `HumanizedPerformance` and `PianoScore` | `backend/services/engrave.py`, `backend/workers/engrave.py` (modify) |
| **Contracts** | New `RefinedPerformance` model (see Contract Decision); new `RefineEditOp` types | `shared/shared/contracts.py` (modify) |
| **Celery app config** | New queue routing: `"refine.run": {"queue": "refine"}` | `backend/workers/celery_app.py` (modify) |
| **Frontend toggle** | Opt-in checkbox on upload screen; stage label in progress screen | `frontend/lib/screens/upload_screen.dart`, `progress_screen.dart` (modify) |
| **A/B harness** | Script that submits N jobs with/without refine, diffs LilyPond outputs | `scripts/ab_refine.py` (new) |
| **Config** | `OHSHEET_ANTHROPIC_API_KEY`, `OHSHEET_REFINE_MODEL`, timeouts, retry counts | `backend/config.py` (modify) |

## Contract Decision — RefinedPerformance (recommended)

**Decision:** Introduce a new `RefinedPerformance` Pydantic model that **wraps** `HumanizedPerformance`, not a subclass and not an extension of `PianoScore`. Engrave accepts all three types via the existing envelope-with-payload_type pattern.

**Rationale:**

1. **Existing engrave already uses a discriminator.** `backend/workers/engrave.py` already dispatches on `payload_type`:
   ```python
   if payload_type == "HumanizedPerformance":
       payload = HumanizedPerformance.model_validate(payload_data)
   elif payload_type == "PianoScore":
       payload = PianoScore.model_validate(payload_data)
   ```
   Adding a third arm for `"RefinedPerformance"` is a 2-line change that matches the established pattern. A new type keeps the discriminator semantic and the envelope explicit.

2. **Subclassing `PianoScore` or `HumanizedPerformance` would hide provenance.** Refine outputs are meaningfully different — they carry edit operations, LLM-attributed changes, confidence signals, and potentially web-search citations. Burying these in an existing shape obscures what happened. A wrapper makes "this score was touched by an LLM" explicit at the type level.

3. **Extending `PianoScore` with optional refine fields contaminates every stage.** Arrange and humanize would then carry `refine_*` fields that are always None. The whole point of Pydantic contracts here is shape clarity — adding dead fields to upstream types violates that.

4. **Wrapping (not subclassing) keeps serialization simple.** `RefinedPerformance` contains the full `HumanizedPerformance` as a field, plus refine-specific provenance. Engrave can either unwrap to the performance for rendering or consume refine fields as needed.

**Recommended shape:**

```python
class RefineEditOp(BaseModel):
    """One edit the LLM made to the humanized performance."""
    op: Literal["modify", "delete"]         # enforced — no "add"
    target_note_id: str                     # references ExpressiveNote.score_note_id
    field: str | None = None                # e.g. "pitch", "onset_beat", "hand", "voice"
    before: Any | None = None
    after: Any | None = None
    rationale: str                          # short LLM-provided reason

class RefineCitation(BaseModel):
    """A web source the LLM used (from Anthropic's web_search tool)."""
    url: str
    title: str | None = None
    quoted_span: str | None = None

class RefinedPerformance(BaseModel):
    schema_version: str = SCHEMA_VERSION
    performance: HumanizedPerformance       # the refined performance ready for engrave
    source_performance_digest: str          # sha256 of the input HumanizedPerformance
    edits: list[RefineEditOp]               # what changed (for audit + rollback)
    citations: list[RefineCitation]         # web sources consulted
    model: str                              # e.g. "claude-sonnet-4-6"
    quality: QualitySignal                  # reuses existing pattern
```

**Engrave compatibility:** In `backend/workers/engrave.py`, add a third arm that unwraps `RefinedPerformance.performance` and passes the inner `HumanizedPerformance` to `EngraveService.run()`. Existing engrave code paths unchanged.

## Data Flow — Three Paths Through the Refine Stage

### Path A — Refine runs successfully (happy path)

```
humanize.run
   │
   │ output: HumanizedPerformance at blob://jobs/{id}/humanize/output.json
   ▼
PipelineRunner.run() — next step in plan is "refine"
   │
   │ emits stage_started(stage="refine", progress=i/n)
   │ serializes input to blob://jobs/{id}/refine/input.json
   │   envelope: { performance: HumanizedPerformance,
   │              metadata: { title, artist, source_url (if YouTube) } }
   │
   ▼
Celery: dispatch "refine.run" via apply_async(job_id, payload_uri)
   │
   ▼
refine.run worker  (backend/workers/refine.py)
   │
   │ 1. blob.get_json(payload_uri) → input envelope
   │ 2. hp = HumanizedPerformance.model_validate(envelope["performance"])
   │ 3. service = RefineService(blob_store=blob, anthropic_client=...)
   │ 4. asyncio.run(service.run(hp, metadata)) → RefinedPerformance
   │      (service internally calls anthropic SDK synchronously — see Sync/Async
   │       decision below; asyncio.run matches existing worker pattern)
   │ 5. blob.put_json("jobs/{id}/refine/output.json", refined.model_dump())
   │ 6. blob.put_json("jobs/{id}/refine/llm_trace.json", trace_dict)  ← artifact
   │ 7. return output_uri
   │
   ▼
PipelineRunner
   │
   │ refined_dict = blob.get_json(output_uri)
   │ stores in local state: refined_dict (new), perf_dict still populated
   │ emits stage_completed(stage="refine", progress=(i+1)/n)
   │
   ▼
engrave step: envelope built with payload_type="RefinedPerformance",
              payload=refined_dict  → engrave.run worker unwraps and renders
```

### Path B — Refine fails (skip-on-failure)

```
refine.run raises (Anthropic timeout / 500 / rate limit / validation failure)
   │
   ▼
Celery task returns failure; result.get() raises CeleryError
   │
   ▼
PipelineRunner.run() catches exception for step == "refine" specifically
   │
   │ log.warning("refine failed, skipping", exc_info=True)
   │ emits JobEvent(type="stage_completed", stage="refine",
   │                message="refine_skipped: {reason}")
   │   (using "stage_completed" with a message, NOT "stage_failed",
   │    keeps job status = "succeeded" — see Failure Semantics below)
   │
   │ Sets a flag: refine_did_not_run = True
   │ Continues to engrave with perf_dict (the unrefined HumanizedPerformance)
   ▼
engrave.run with payload_type="HumanizedPerformance" — identical to
current default path. Job succeeds with normal PDF, no refine artifact.
```

### Path C — User didn't opt in (enable_refine=False)

```
PipelineConfig.get_execution_plan() omits "refine" from the list.
Pipeline behaves exactly as it does today — zero LLM call, zero latency overhead.
```

## Sync vs Async — LLM Call Inside the Celery Worker

**Decision:** Use the **synchronous `anthropic.Anthropic()` client**, called from within an `async def` service method that is invoked via `asyncio.run()` from the Celery task.

**Pattern:**

```python
# backend/workers/refine.py
@celery_app.task(name="refine.run")
def run(job_id: str, payload_uri: str) -> str:
    blob = LocalBlobStore(settings.blob_root)
    envelope = blob.get_json(payload_uri)
    hp = HumanizedPerformance.model_validate(envelope["performance"])
    metadata = envelope.get("metadata", {})

    service = RefineService(blob_store=blob)   # constructs Anthropic client internally
    # asyncio.run() matches humanize.py / engrave.py — Celery prefork pool safe
    refined = asyncio.run(service.run(hp, metadata, job_id=job_id))

    output_uri = blob.put_json(
        f"jobs/{job_id}/refine/output.json",
        refined.model_dump(mode="json"),
    )
    return output_uri
```

```python
# backend/services/refine.py
class RefineService:
    def __init__(self, blob_store: BlobStore, client: anthropic.Anthropic | None = None):
        self.blob = blob_store
        self.client = client or anthropic.Anthropic(api_key=settings.anthropic_api_key)

    async def run(
        self,
        performance: HumanizedPerformance,
        metadata: dict,
        *,
        job_id: str,
    ) -> RefinedPerformance:
        prompt = build_refine_prompt(performance, metadata)
        # Synchronous SDK call wrapped in to_thread so we don't block the
        # asyncio.run() event loop for other awaits in this service.
        response = await asyncio.to_thread(
            self.client.messages.create,
            model=settings.refine_model,
            max_tokens=settings.refine_max_tokens,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            system=REFINE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        edits = parse_edits(response)
        validate_modify_delete_authority(edits, performance)   # raises on violation
        refined_perf = apply_edits(performance, edits)
        return RefinedPerformance(...)
```

**Rationale:**

1. **Matches existing pattern.** `humanize.py`, `engrave.py`, and `ingest.py` workers all use `asyncio.run(service.run(...))`. Introducing a different pattern for refine would be surprising. Celery's default prefork pool runs each task in a separate process with its own event loop — `asyncio.run()` is safe there.

2. **Sync SDK avoids async context conflicts.** The [Anthropic Python SDK docs](https://platform.claude.com/docs/en/api/sdks/python) note the synchronous `Anthropic` client blocks during I/O. That's exactly what we want inside a Celery worker (the task is already its own unit of concurrency — Celery scales by spawning more workers/processes).

3. **`to_thread` wrapping inside the service** lets the service be composable. If a future change wants to call refine from within an async FastAPI handler (e.g. a `/v1/refine/preview` endpoint for debugging), the service signature doesn't need to change — only the client choice.

4. **Alternative considered: use `AsyncAnthropic` throughout.** Rejected because (a) the existing workers use sync-inside-async pattern, (b) concurrency inside one refine call is unnecessary — Claude with web_search is one logical call — and (c) Celery task-level concurrency is the right layer to scale on.

5. **Retry semantics live in the Celery task**, not inside the SDK call. Use Celery's `autoretry_for=(anthropic.APITimeoutError, anthropic.RateLimitError)`, `retry_backoff=True`, `max_retries=settings.refine_max_retries`. On final failure, task raises and the runner hits Path B (skip-on-failure).

## Modify+Delete Authority — Layered Enforcement

**Decision:** Enforce in **all three layers** — schema, prompt, and post-validation — with post-validation as the authoritative gate.

**Why layered:**

| Layer | Enforcement | What It Catches |
|-------|-------------|-----------------|
| **Schema** (`RefineEditOp.op: Literal["modify", "delete"]`) | Pydantic validation rejects op=="add" at deserialization time | Mechanical correctness — any well-formed response can only name modify/delete |
| **Prompt** (system message + examples) | LLM is told it can only modify existing notes and delete notes, never add | Soft constraint — most LLM attempts to violate this will be structured as modifications anyway |
| **Post-validation** (`validate_modify_delete_authority`) | Scan LLM's final refined score; compare note IDs against input. Reject if new IDs appear that weren't in the input's `ExpressiveNote.score_note_id` set | Definitive check — even if the LLM returned prose or freeform JSON that passed schema, the actual output score must not contain notes that didn't exist before |

**Critical choice: validate on note IDs, not pitch+onset.** `HumanizedPerformance.expressive_notes` references `ScoreNote.id` from the upstream `PianoScore`. Refine operations reference these IDs. Any note in the output whose ID doesn't appear in the input's note ID set is a violation — this is unambiguous and cheap to check (O(n) set lookup).

**Output format:** Ask the LLM to return a list of `RefineEditOp` objects, not a fully-rewritten score. The service then applies edits locally. This has three wins:

1. The LLM only has to emit the diff, not the whole score — shorter output, fewer hallucination surfaces.
2. The validator can apply edits with authority assertions baked in (try to apply an "add" → raise immediately).
3. The audit trail (`edits` field on `RefinedPerformance`) is native — no separate diffing step.

**Example validation:**

```python
def validate_modify_delete_authority(
    edits: list[RefineEditOp],
    source: HumanizedPerformance,
) -> None:
    valid_ids = {n.score_note_id for n in source.expressive_notes}
    for edit in edits:
        if edit.op not in ("modify", "delete"):
            raise RefineViolation(f"forbidden op: {edit.op!r}")
        if edit.target_note_id not in valid_ids:
            raise RefineViolation(
                f"op targets unknown note id {edit.target_note_id!r} — "
                f"LLM may be trying to add a note"
            )
```

## Web-Search Tool Wiring

Anthropic's web_search is a server-side tool (the model invokes it and the API handles execution — the worker never makes HTTP calls itself). Wiring is trivial:

```python
tools = [{
    "type": "web_search_20250305",    # pin the tool version; current as of 2026-04
    "name": "web_search",
    "max_uses": settings.refine_web_search_max_uses,  # e.g. 3
}]
response = client.messages.create(
    model=settings.refine_model,
    tools=tools,
    system=REFINE_SYSTEM_PROMPT,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=settings.refine_max_tokens,
)
```

**Response handling:** Iterate `response.content` looking for:
- `ToolUseBlock` entries (what the LLM searched for)
- `ServerToolUseBlock` / `WebSearchResultBlock` entries (the results Anthropic returned)
- `TextBlock` entries for the LLM's final answer (the edit list)

**Citation capture:** Extract URLs from `web_search_result` blocks into `RefineCitation` entries and attach them to `RefinedPerformance.citations`. These ship in the refine output artifact so inspections can see what the LLM grounded against.

**No multi-turn.** For v1, a single `messages.create` call with web_search available is enough. Anthropic handles the tool call loop server-side — the worker gets one final response with all tool uses and the answer. No `tool_runner` loop needed.

**Failure modes specific to web_search:**
- Tool disabled in org/account → `BadRequestError` at call time → retry won't help → Path B (skip)
- Tool rate-limited → 429 → Celery retry handles it
- Tool returns empty results → model still answers; this is not a failure

## PipelineConfig + JobCreateRequest Plumbing

**Step-by-step additions (in dependency order — each step depends on the prior):**

1. **Contracts** (`shared/shared/contracts.py`):
   - Add `enable_refine: bool = False` to `PipelineConfig`.
   - Modify `PipelineConfig.get_execution_plan()` to insert `"refine"` after `"humanize"` when `enable_refine` is True. For `sheet_only` (which skips humanize), insert `"refine"` after `"arrange"` and before `"engrave"`.
   - Add new models: `RefineEditOp`, `RefineCitation`, `RefinedPerformance`.

2. **API route** (`backend/api/routes/jobs.py`):
   - Add `enable_refine: bool = False` to `JobCreateRequest`.
   - In `create_job()`, pass `enable_refine=body.enable_refine` when constructing `PipelineConfig`.

3. **Runner** (`backend/jobs/runner.py`):
   - Add `"refine": "refine.run"` to `STEP_TO_TASK`.
   - Add a new `elif step == "refine":` branch in the dispatch loop. Input is `perf_dict` (or `score_dict` for sheet_only); envelope wraps with song metadata from bundle.
   - Store output in a new `refined_dict` local and update the `engrave` branch to prefer it over `perf_dict` when available.
   - Wrap the `refine` step's dispatch in a skip-on-failure try/except (see Failure Semantics below).

4. **Celery app** (`backend/workers/celery_app.py`):
   - Add `"refine.run": {"queue": "refine"}` to `task_routes`.

5. **Service + Worker** (`backend/services/refine.py`, `backend/workers/refine.py`):
   - New files following the exact pattern of `humanize.py` service+worker pair.

6. **Engrave** (`backend/workers/engrave.py`):
   - Add a third `elif payload_type == "RefinedPerformance":` arm. Unwrap to `HumanizedPerformance` via `.performance` field. Call `EngraveService.run()` as usual.
   - (Optional) `EngraveService` could accept `RefinedPerformance` directly later if refine-specific rendering hints emerge (dynamics from refine citations, etc.). Not in scope for v1.

7. **Config** (`backend/config.py`):
   - Add settings: `anthropic_api_key: str | None = None`, `refine_model: str = "claude-sonnet-4-6"`, `refine_max_tokens: int = 8000`, `refine_max_retries: int = 2`, `refine_web_search_max_uses: int = 3`, `refine_timeout_sec: int = 120`.
   - App startup: if any job attempts refine with `anthropic_api_key is None`, fail fast in the worker (not at app startup — workers and API may be separate processes, and the API doesn't need the key).

8. **Frontend** (`frontend/lib/screens/upload_screen.dart`, `progress_screen.dart`, `api/client.dart`, `api/models.dart`):
   - Add toggle on upload screen.
   - Thread `enableRefine` through `OhSheetApi.createJob()`.
   - Add "refine" to the stage label map on the progress screen.

9. **A/B harness** (`scripts/ab_refine.py`):
   - Standalone Python script (not a Celery task, not a pytest test). Takes a list of reference song specs, submits each twice (with/without refine), downloads LilyPond output, writes a diff report.

## JobEvent Emission Cadence

**Decision:** Follow the existing per-stage pattern exactly — one `stage_started` and one `stage_completed` per refine invocation. No intra-LLM progress events.

**Rationale:**

1. **Consistency.** Every other stage emits exactly `stage_started` and `stage_completed`. Refine doing more would make it visually noisy on the frontend's progress screen (which expects a predictable cadence).

2. **No streaming in v1.** PROJECT.md "Out of Scope" explicitly excludes streaming partial refine results. The Anthropic call is atomic to the worker — no partial state to emit.

3. **Web-search events are LLM internal.** The model may call web_search 0-3 times per refine; emitting a JobEvent per tool call would expose an implementation detail that users don't care about and that would change cadence between otherwise-identical songs. Log tool calls to `llm_trace.json` for debugging instead.

4. **Skipped-refine is a completion event, not a failure.** `stage_completed` with a human-readable `message="refine_skipped: {reason}"` lets the frontend render a distinct visual state (e.g. a grey checkmark with "skipped" text) without breaking the progress bar advancement. Reserving `stage_failed` for real failures (which halt the pipeline) keeps the event schema's semantics clean.

**Progress reporting:** The runner already emits `progress=i/n` where `n` is plan length. When refine is in the plan, `n` grows by 1 and the bar advances normally. When skipped, the completion event still emits progress `(i+1)/n`, so the bar continues unless the frontend chooses to style it differently.

## Failure Semantics — Skip-on-Refine-Failure

**Decision:** Implement skip-on-failure **in the runner**, not in the worker.

**Why not in the worker:** If the worker swallowed the failure and returned a synthetic "refine didn't happen" artifact, the runner wouldn't know which envelope to pass to engrave. The worker would have to write a mock output identical to its input, which is confusing and makes debugging harder.

**Why in the runner:** The runner owns the execution plan and already handles per-stage outcomes. Wrapping the `refine` branch in a try/except is a minimal, localized change:

```python
elif step == "refine":
    if perf_dict is None and score_dict is None:
        raise RuntimeError("refine stage requires a score or performance")

    refine_input = {
        "performance": perf_dict or score_dict,
        "payload_type": "HumanizedPerformance" if perf_dict else "PianoScore",
        "metadata": {
            "title": title,
            "artist": composer,
            "source_url": bundle.audio.uri if bundle.audio else None,
        },
    }
    payload_uri = self._serialize_stage_input(job_id, step, refine_input)
    try:
        output_uri = await self._dispatch_task(
            task_name, job_id, payload_uri, config.stage_timeout_sec,
        )
        refined_dict = self.blob_store.get_json(output_uri)
    except Exception as exc:   # noqa: BLE001 — intentional: refine never fails the job
        log.warning(
            "refine skipped job_id=%s reason=%r",
            job_id, exc,
        )
        emit(step, "stage_completed",
             progress=(i + 1) / n,
             message=f"refine_skipped: {type(exc).__name__}")
        refined_dict = None
        # DO NOT re-raise. Continue to engrave with unrefined data.
        continue
```

**Engrave branch updated:** prefers `refined_dict → perf_dict → score_dict`. When refine was skipped, `refined_dict is None` and the original path remains.

**Important: the `continue` skips the normal `stage_completed` emit** at the bottom of the loop, which is why the except block emits the event manually.

## A/B Harness Architecture

**Decision:** Standalone Python script in `scripts/ab_refine.py`, not a Celery task and not a pytest.

**Rationale:**

- **Not a pytest:** A/B comparisons are not pass/fail (they're quality assessments that need human review). Pytest is the wrong ergonomic fit — runs on CI, wants green/red. Even if we wrote it as a pytest, it would have to be marked skipped in CI to avoid blowing budget on every PR.

- **Not a Celery task:** The harness runs N jobs; each job already fans out to Celery. Wrapping the harness itself in Celery just adds indirection. A script that posts to `/v1/jobs` and polls is simpler.

- **Script pattern matches `scripts/eval_transcription.py`:** That file (which I read during research) is the existing precedent for offline quality harnesses in this codebase. Follow the same shape: argparse CLI, results JSON, optional `--out` for CI-friendly diffs.

**Structure:**

```
scripts/ab_refine.py
  main(args):
    for spec in reference_specs:
      without_refine = submit_job(spec, enable_refine=False)
      with_refine    = submit_job(spec, enable_refine=True)
      wait_for_completion(without_refine, with_refine)
      ly_a = download_lilypond(without_refine)
      ly_b = download_lilypond(with_refine)
      diff = unified_diff(ly_a, ly_b)
      refined_artifact = download_refine_output(with_refine)
      record_result(spec, diff, refined_artifact.edits, refined_artifact.citations)
    write_report(results, args.out)
```

**Reference specs live in `eval/fixtures/refine/` as a small JSON manifest** (songs, YouTube URLs or MIDI file references, expected genre/complexity). Keeps the harness reproducible and easy to extend.

**The harness needs a downloadable LilyPond artifact.** This is not currently served by `/v1/artifacts/{id}/{kind}` (PDF only — LilyPond is intermediate). The harness either (a) reads LilyPond from the blob store directly via local filesystem path (development convenience) or (b) we add a new artifact kind `lilypond`. Recommend (b) as a one-line addition to `artifacts.py` — it's broadly useful for debugging the engrave stage anyway, not just refine.

## Build Order (Dependency Chain)

The roadmap should sequence phases so that each phase produces a working, testable slice:

```
Phase 1 — Contracts + config (no behavior change)
   │    Changes: contracts.py adds PipelineConfig.enable_refine field +
   │             RefinedPerformance/RefineEditOp/RefineCitation models;
   │             config.py adds anthropic_api_key and refine_* settings;
   │             JobCreateRequest.enable_refine field; runner.STEP_TO_TASK
   │             (maps to task, but task doesn't exist yet).
   │    Test: contract round-trip; JobCreateRequest with enable_refine=True
   │          builds a PipelineConfig whose get_execution_plan() includes
   │          "refine" — but the pipeline will fail to dispatch until Phase 2.
   │    Gate: new contracts serialize; unit tests for get_execution_plan variants.
   ▼
Phase 2 — Service + worker (isolated, no pipeline wiring)
   │    Changes: backend/services/refine.py (prompt, Anthropic call, validation,
   │             edit application); backend/services/refine_prompt.py;
   │             backend/services/refine_validate.py; backend/workers/refine.py;
   │             celery_app.py task_routes update.
   │    Test: Unit tests for prompt builder, validator, edit applier.
   │          Integration test with a mocked Anthropic client verifying the
   │          worker returns a valid RefinedPerformance URI.
   │          Does NOT yet run end-to-end through the runner.
   │    Gate: refine.run dispatched with fake input succeeds; validator rejects
   │          synthetic "add" ops; edit applier mutates performance correctly.
   ▼
Phase 3 — Runner wiring + engrave update (end-to-end)
   │    Changes: backend/jobs/runner.py refine branch with skip-on-failure;
   │             refined_dict state passed to engrave; backend/workers/engrave.py
   │             third arm for payload_type="RefinedPerformance";
   │             backend/api/routes/artifacts.py optional LilyPond endpoint.
   │    Test: Mocked-Anthropic end-to-end pytest submits a job with
   │          enable_refine=True and verifies the pipeline produces an
   │          EngravedOutput. Second test forces the worker to raise and
   │          verifies Path B (skip) works — job.status == "succeeded",
   │          event stream contains "refine_skipped" message.
   │    Gate: refine job completes end-to-end with mocked LLM; skip-on-failure
   │          verified; stage_started/stage_completed events emit.
   ▼
Phase 4 — Real Anthropic integration + live test song
   │    Changes: Real ANTHROPIC_API_KEY in dev env; integration test run against
   │             the real API with a canned song (gated by env var so CI doesn't
   │             pay the token bill); prompt iteration; web_search tool
   │             verification.
   │    Test: Manual run on 2-3 songs; verify refine output shape, citations
   │          populated, edit list reasonable. Budget-gated pytest-marked
   │          integration test.
   │    Gate: Refine produces plausible output on a real song; citations
   │          extracted; latency within expected 5-30s envelope.
   ▼
Phase 5 — Frontend toggle
   │    Changes: frontend/lib/screens/upload_screen.dart adds checkbox;
   │             api/models.dart adds enableRefine field; api/client.dart
   │             threads through; progress_screen.dart adds refine stage label
   │             + skipped state rendering.
   │    Test: Flutter widget test for toggle; end-to-end manual verification
   │          that enabling the toggle causes refine events to appear on the
   │          progress screen.
   │    Gate: User can toggle refine on the upload screen and see it run in
   │          the progress screen; skipped state renders distinctively.
   ▼
Phase 6 — A/B harness + baseline run
   │    Changes: scripts/ab_refine.py; eval/fixtures/refine/manifest.json
   │             with 5-10 reference songs; artifact LilyPond endpoint if not
   │             added in Phase 3.
   │    Test: Run harness against reference set; compare against expectations
   │          (human review of output diffs).
   │    Gate: Harness produces a reproducible diff report; baseline committed.
```

**Why this order matters:**

- **Contracts before service**: Service signatures reference `RefinedPerformance`; cannot be written until the type exists.
- **Service before runner**: Runner's `elif step == "refine":` branch needs a functioning `refine.run` task to dispatch to.
- **Engrave update with runner wiring, not earlier**: `payload_type="RefinedPerformance"` is only exercised once the runner produces one, so these changes land together.
- **Mocked Anthropic before real Anthropic**: Real API spends tokens. Phases 1-3 should use a fake `AnthropicClient` protocol so CI doesn't incur cost and doesn't depend on external availability.
- **Frontend last**: Backend behavior gated behind `enable_refine` defaults to False; frontend toggle is the final "turn it on for users" step. Landing frontend earlier would be safe but provides no value without working backend.
- **A/B harness last**: Needs a stable refine pipeline to evaluate.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Inlining the LLM call into the engrave stage

**What people do:** Tack the LLM call onto the front of engrave: "it's just one more preprocessing step."

**Why it's wrong:**
- Couples LLM latency (5-30s + retries) to engrave's deterministic ~2s runtime, making the stage's p99 wildly variable.
- Makes the refine output non-inspectable — it's not a separate artifact, just an intermediate variable inside the engrave worker.
- Breaks the opt-in/opt-out model: refine becomes a runtime branch inside engrave rather than a plan-level toggle.

**Do this instead:** Treat refine as a first-class pipeline stage with its own worker, queue, and artifact. Exactly the approach documented above.

### Anti-Pattern 2: Using AsyncAnthropic client + asyncio inside the Celery task

**What people do:** Import `AsyncAnthropic`, wrap everything in `async def`, call `asyncio.run` on the top-level task.

**Why it's wrong:** No concurrency benefit. A single refine call is one logical unit — there's no parallelism to exploit inside it. Celery scales by running more workers, not by running async inside one worker. The sync client is simpler and matches existing workers.

**Do this instead:** Synchronous `Anthropic()` client. If the service method is async for composition, wrap the sync call in `asyncio.to_thread`.

### Anti-Pattern 3: Validating modify+delete authority only in the prompt

**What people do:** "The prompt says 'only modify or delete notes,' so we trust the LLM."

**Why it's wrong:** LLMs periodically violate soft constraints, especially with complex prompts and tool use. A prompt-only constraint will fail an eval someday, produce a note that wasn't in the source audio, and a user will get sheet music for notes that never played.

**Do this instead:** Schema-level `Literal["modify", "delete"]` + post-validation that cross-references edit target IDs against source note IDs. Layered defense.

### Anti-Pattern 4: Making refine failure fail the job

**What people do:** Treat refine like any other stage — if Anthropic is down, the job fails.

**Why it's wrong:** Refine is an enhancement, not a requirement. Users who don't opt in get perfect PDFs today. Users who do opt in shouldn't have their pipeline gated on Anthropic's uptime — if the LLM can't help, they should still get the unrefined PDF. Failing jobs on Anthropic outages undercuts the opt-in value proposition.

**Do this instead:** Catch refine failure at the runner, emit `stage_completed` with a `refine_skipped` message, continue to engrave with the unrefined performance.

### Anti-Pattern 5: Embedding refine-specific fields into PianoScore / HumanizedPerformance

**What people do:** Add optional `refine_citations`, `refine_edits`, etc. to `HumanizedPerformance` so engrave can see them.

**Why it's wrong:** Pollutes upstream contracts with fields that are always None for non-refined flows. Makes schema versioning harder. Obscures the semantic boundary — a humanized performance is supposed to be the output of humanize, not a carrier for downstream metadata.

**Do this instead:** Separate `RefinedPerformance` wrapper that contains the performance. Clear semantic boundary, zero impact on upstream types.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Anthropic Claude API | Synchronous SDK (`anthropic.Anthropic()`) inside Celery worker; `messages.create()` with `web_search_20250305` tool | Pin tool version; pin model version (`claude-sonnet-4-6`). Rotate both consciously; treat model change like a schema change. |
| Anthropic web_search (server-side tool) | No additional integration — invoked via `tools=[...]` argument to `messages.create` | Tool execution happens server-side; worker never makes HTTP calls to external URLs directly. Citations captured from response blocks. |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| PipelineRunner ↔ refine worker | Celery task dispatch via `apply_async(job_id, payload_uri)`; result is output URI | Same pattern as every other stage. `task_routes` maps to the `refine` queue. |
| refine worker ↔ BlobStore | `put_json` / `get_json` for input envelope, output `RefinedPerformance`, and separate `llm_trace.json` artifact | Matches humanize / engrave worker I/O exactly. |
| refine worker ↔ Anthropic API | Direct SDK call inside `asyncio.to_thread`; retries via Celery task `autoretry_for` | Secret pulled from `settings.anthropic_api_key` (env: `OHSHEET_ANTHROPIC_API_KEY`). |
| engrave worker ↔ RefinedPerformance | Envelope `payload_type` discriminator on existing type switch | Engrave unwraps `.performance` and proceeds with existing `HumanizedPerformance` path — minimal change. |
| frontend ↔ JobEvent stream | Existing WebSocket; `refine` stage name added to stage label map | No new event types — just an additional stage value in existing `stage_started` / `stage_completed` events. |

## Scaling Considerations

This milestone is not a scaling milestone, but some points to keep in mind:

| Scale | Consideration |
|-------|---------------|
| Dev / single user | One Celery worker with the `refine` queue is sufficient. Set a reasonable `refine_timeout_sec` (120s) so a hung LLM call doesn't stall the whole pipeline indefinitely. |
| Small production (10s of jobs/day, e.g. current GCP Cloud Run deployment) | Dedicated `refine` queue with a single worker process matches current single-worker-per-queue pattern. Anthropic handles parallelism on their side. |
| Scale-up (100s+ jobs/day with refine) | Consider caching refine outputs keyed on content-hash of input `HumanizedPerformance` — a re-submission of the same song would skip the LLM. Explicitly deferred in PROJECT.md ("LLM response caching — revisit when cost or repeat-submit traffic justifies it"). |

### First bottleneck

The first thing to break, long before scaling concerns, will be **Anthropic rate limits**. Even moderate traffic bursts can exceed per-minute or per-day token caps. Celery retries with exponential backoff absorb transient 429s; beyond that, Path B (skip) gracefully degrades. Monitor the skip rate — a rising skip rate indicates either rate limit pressure or model quality issues, and both warrant action.

## Sources

- [Anthropic Python SDK — Synchronous and Asynchronous Clients (DeepWiki)](https://deepwiki.com/anthropics/anthropic-sdk-python/4.2-synchronous-and-asynchronous-clients)
- [Python SDK — Claude API Docs](https://platform.claude.com/docs/en/api/sdks/python)
- [anthropic-sdk-python GitHub](https://github.com/anthropics/anthropic-sdk-python)
- [Async LLM Tasks with Celery and Celery Beat (Medium — AlgoMart)](https://medium.com/algomart/async-llm-tasks-with-celery-and-celery-beat-31c824837f35)
- [Advanced Celery: mastering idempotency, retries & error handling (Vinta)](https://www.vintasoftware.com/blog/celery-wild-tips-and-tricks-run-async-tasks-real-world)
- [Handling Long-Running AI Jobs with Redis and Celery (Markaicode)](https://markaicode.com/redis-celery-long-running-ai-jobs/)
- Existing codebase:
  - `/Users/jackjiang/GitHub/oh-sheet/.planning/codebase/ARCHITECTURE.md` — the pipeline pattern this work integrates with
  - `/Users/jackjiang/GitHub/oh-sheet/backend/jobs/runner.py` — PipelineRunner, STEP_TO_TASK, per-variant plans
  - `/Users/jackjiang/GitHub/oh-sheet/backend/workers/humanize.py` — worker pattern to mirror
  - `/Users/jackjiang/GitHub/oh-sheet/backend/workers/engrave.py` — envelope/payload_type discriminator to extend
  - `/Users/jackjiang/GitHub/oh-sheet/shared/shared/contracts.py` — PipelineConfig, PianoScore, HumanizedPerformance to extend
  - `/Users/jackjiang/GitHub/oh-sheet/backend/api/routes/jobs.py` — JobCreateRequest plumbing precedent (prefer_clean_source)

---
*Architecture research for: LLM refine stage integration into existing Celery pipeline*
*Researched: 2026-04-13*
