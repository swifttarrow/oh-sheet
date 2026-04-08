# Dev Journal

## 2026-04-08: Celery Refactor — Approach Decision

### TL;DR

**2026-04-08:** Picked orchestrator-driven Celery (`PipelineRunner` dispatches tasks and waits without blocking the event loop) over Celery chains or callback-chained workers so variant routing and WebSocket stage events stay centralized, with all five stages as Celery tasks, new `svc-decomposer` / `svc-assembler` services, and a shared blob store.

### Context

The current pipeline runs all 5 stages (ingest, transcribe, arrange, humanize, engrave) in-process via `asyncio.Task` inside a single FastAPI process. Jobs live in an in-memory registry and don't survive restarts. The system design docs describe a Celery + Redis microservice architecture that doesn't match what's built.

Goal: scaffold decomposer (`svc-decomposer`) and assembler (`svc-assembler`) as separate top-level services with their own Docker containers, refactor all 5 pipeline stages to run as Celery tasks, and wire everything through Redis — while maintaining identical end-to-end functionality via stubs that delegate to the existing service logic.

### Approaches Considered

**Approach 1: Celery Chain with Shared Blob Store**
- Build a Celery `chain(ingest -> decomposer -> assembler -> humanizer -> engraver)` and kick it off as a single unit.
- Workers share Redis broker + blob store (shared volume or MinIO).
- *Pros:* Simple sequential model; Celery chains handle ordering natively.
- *Cons:* Tightly coupled to Celery's chain primitive. Hard to express conditional paths — the current system supports 4 pipeline variants (full, audio_upload, midi_upload, sheet_only) that skip stages. Chains don't handle that cleanly without dynamic chain construction, which gets ugly fast.

**Approach 2: Orchestrator-Driven Dispatch (CHOSEN)**
- `PipelineRunner` stays as the orchestrator brain. Instead of `await self.transcribe.run(bundle)`, it dispatches `celery_app.send_task("decomposer.run")` and waits for the result.
- Orchestrator owns execution plan logic (which stages, what order). Workers are stateless and dumb.
- Blocking `.get()` wrapped in `asyncio.to_thread()` to avoid blocking the event loop.
- *Pros:* Smallest diff against current `PipelineRunner`. Preserves existing variant routing. Workers stay simple — receive payload URI, run logic, write output, return. Orchestrator can still emit `JobEvent`s at stage boundaries for WebSocket streaming.
- *Cons:* Synchronous wait on Celery results (mitigated by `asyncio.to_thread` wrapper). Orchestrator is still a single point of coordination.

**Approach 3: Event-Driven Callbacks**
- Each task fires the next via `self.app.send_task()` on completion. No central runner.
- *Pros:* Fully decoupled, no blocking waits.
- *Cons:* Pipeline logic scattered across every worker. Variant routing (skip stages, conditional paths) becomes a nightmare. Hard to emit centralized progress events. Debugging pipeline failures requires tracing across multiple services with no single view.

### Why Approach 2

1. **Variant routing stays centralized.** Four pipeline variants with conditional stage skipping is complex enough to live in one place, not scattered across workers.
2. **Minimal risk to working system.** The refactor replaces the *dispatch mechanism* inside `PipelineRunner` without changing the stage logic itself. Stubs delegate to the existing `IngestService`, `TranscribeService`, `ArrangeService`, `HumanizeService`, and `EngraveService`.
3. **WebSocket events preserved.** The orchestrator still controls stage boundaries, so `stage_started`/`stage_completed` events flow through the existing `JobManager` pub/sub without changes.
4. **Workers are independently deployable.** Each Celery worker (including the two new top-level services) can be scaled, restarted, or replaced without touching the orchestrator.

### Key Decisions

- **All 5 stages become Celery tasks**, not just decomposer/assembler. Avoids a hybrid in-process + Celery execution model.
- **Decomposer and assembler get own top-level directories** (`svc-decomposer/`, `svc-assembler/`). Other 3 stages stay in the monolith codebase as Celery workers.
- **Stubs maintain current functionality.** No new logic — stubs call existing service classes.
- **Blob store must become shared.** Separate containers can't use local filesystem. Shared Docker volume for local dev; S3/MinIO for production.
