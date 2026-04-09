# Celery Refactor: Scaffold Decomposer + Assembler Services

**Date:** 2026-04-08
**Status:** Draft

## Goal

Refactor the Oh Sheet pipeline from in-process `asyncio.Task` execution to Celery + Redis task dispatch. Scaffold `svc-decomposer` and `svc-assembler` as independent top-level services. All 5 pipeline stages become Celery tasks. Stubs delegate to existing service logic — end-to-end functionality is unchanged.

## Non-Goals

- Implementing real decomposer/assembler logic (musicpy split, music21 arrangement rules from PRDs)
- Migrating from local blob store to S3/MinIO
- Persistent job storage (Postgres)
- Optimizing Docker images for workers vs orchestrator
- Changing the frontend or WebSocket event contract

## Repository Layout

```
oh-sheet/
├── backend/
│   ├── api/                        # Unchanged — routes, deps
│   ├── jobs/
│   │   ├── manager.py              # Unchanged — in-memory job registry + pub/sub
│   │   ├── runner.py               # MODIFIED — dispatch via Celery instead of direct calls
│   │   └── events.py               # Unchanged
│   ├── services/                   # Unchanged — existing service classes
│   ├── storage/                    # Unchanged — LocalBlobStore
│   ├── workers/                    # NEW — Celery task definitions for monolith stages
│   │   ├── __init__.py
│   │   ├── celery_app.py           # Celery app config (shared by monolith workers)
│   │   ├── ingest.py               # Task: calls IngestService.run()
│   │   ├── humanize.py             # Task: calls HumanizeService.run()
│   │   └── engrave.py              # Task: calls EngraveService.run()
│   ├── config.py                   # Add REDIS_URL, CELERY_BROKER_URL
│   └── main.py                     # Unchanged
│
├── shared/                         # NEW — contracts + storage protocol
│   ├── pyproject.toml
│   └── shared/
│       ├── __init__.py
│       ├── contracts.py            # Extracted from backend/contracts.py
│       └── storage/
│           ├── __init__.py
│           ├── base.py             # BlobStore protocol
│           └── local.py            # LocalBlobStore implementation
│
├── svc-decomposer/                 # NEW — independent service
│   ├── Dockerfile
│   ├── pyproject.toml              # Depends on shared
│   ├── decomposer/
│   │   ├── __init__.py
│   │   ├── celery_app.py           # Celery app config pointing to Redis
│   │   └── tasks.py                # Stub task: deserialize → call TranscribeService → serialize
│   └── tests/
│       └── test_tasks.py
│
├── svc-assembler/                  # NEW — independent service
│   ├── Dockerfile
│   ├── pyproject.toml              # Depends on shared
│   ├── assembler/
│   │   ├── __init__.py
│   │   ├── celery_app.py
│   │   └── tasks.py                # Stub task: deserialize → call ArrangeService → serialize
│   └── tests/
│       └── test_tasks.py
│
├── docker-compose.yml              # NEW — full topology
├── Dockerfile                      # Existing — used by orchestrator + monolith workers
└── pyproject.toml                  # Existing — backend package, add shared dependency
```

## Execution Flow

### Current (in-process)

```python
# backend/jobs/runner.py — PipelineRunner.run()
bundle = await self.ingest.run(bundle)
txr = await self.transcribe.run(bundle)
score = await self.arrange.run(txr)
perf = await self.humanize.run(score)
result = await self.engrave.run(perf, job_id=job_id, ...)
```

### Refactored (Celery dispatch)

```python
# backend/jobs/runner.py — PipelineRunner.run()
for i, step in enumerate(plan):
    emit(step, "stage_started", progress=i/n)

    payload_uri = self.blob_store.put_json(
        f"jobs/{job_id}/{step}/input.json", payload_dict
    )
    result_uri = await asyncio.to_thread(
        celery_app.send_task(
            f"{step}.run",
            args=[job_id, payload_uri],
        ).get,
        timeout=config.job_timeout_sec,
    )
    payload_dict = self.blob_store.get_json(result_uri)

    emit(step, "stage_completed", progress=(i+1)/n)
```

### Worker Task Pattern (identical for all 5 stages)

```python
# Example: svc-decomposer/decomposer/tasks.py
@celery_app.task(name="decomposer.run")
def run(job_id: str, payload_uri: str) -> str:
    blob = LocalBlobStore(settings.blob_root)

    # 1. Read input
    raw = blob.get_json(payload_uri)
    bundle = InputBundle.model_validate(raw)

    # 2. Run stub (delegates to existing service)
    service = TranscribeService()
    result = asyncio.run(service.run(bundle))

    # 3. Write output
    output_uri = blob.put_json(
        f"jobs/{job_id}/decomposer/output.json",
        result.model_dump(mode="json"),
    )
    return output_uri
```

### Task Name Mapping

| Execution Plan Step | Celery Task Name     | Worker              | Stub Delegates To       |
|---------------------|----------------------|---------------------|-------------------------|
| `ingest`            | `ingest.run`         | backend worker      | `IngestService.run()`   |
| `transcribe`        | `decomposer.run`     | svc-decomposer      | `TranscribeService.run()` |
| `arrange`           | `assembler.run`      | svc-assembler        | `ArrangeService.run()`  |
| `humanize`          | `humanize.run`       | backend worker      | `HumanizeService.run()` |
| `engrave`           | `engrave.run`        | backend worker      | `EngraveService.run()`  |

Note: The execution plan steps (`transcribe`, `arrange`) map to different Celery task names (`decomposer.run`, `assembler.run`). The runner needs a step-to-task-name mapping.

### Pipeline Variant Support

The orchestrator still owns variant routing. `PipelineConfig.get_execution_plan()` returns the list of steps, and the runner iterates through them. No changes to variant logic:

- `full` / `audio_upload`: `[ingest, transcribe, arrange, humanize, engrave]`
- `midi_upload`: `[ingest, arrange, humanize, engrave]`
- `sheet_only`: `[ingest, transcribe, arrange, engrave]`

## Shared Storage

All containers mount the same Docker volume at `/app/blob`. `LocalBlobStore` writes to this path. URI format stays `file:///app/blob/...`.

```yaml
volumes:
  blob-data:

services:
  orchestrator:
    volumes: [blob-data:/app/blob]
  worker-decomposer:
    volumes: [blob-data:/app/blob]
  # ... same for all workers
```

Production migration to S3 is a future concern — swap `LocalBlobStore` for `S3BlobStore` behind the same `BlobStore` protocol. No worker code changes.

## Docker-Compose Topology

7 containers total:

```yaml
services:
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  orchestrator:
    build: .
    command: uvicorn backend.main:app --host 0.0.0.0 --port 8000
    ports: ["8000:8000"]
    environment:
      OHSHEET_BLOB_ROOT: /app/blob
      OHSHEET_REDIS_URL: redis://redis:6379/0
    volumes: [blob-data:/app/blob]
    depends_on: [redis]

  worker-ingest:
    build: .
    command: celery -A backend.workers.celery_app worker -Q ingest -c 1
    environment:
      OHSHEET_BLOB_ROOT: /app/blob
      OHSHEET_REDIS_URL: redis://redis:6379/0
    volumes: [blob-data:/app/blob]
    depends_on: [redis]

  worker-decomposer:
    build: ./svc-decomposer
    command: celery -A decomposer.celery_app worker -Q decomposer -c 1
    environment:
      OHSHEET_BLOB_ROOT: /app/blob
      OHSHEET_REDIS_URL: redis://redis:6379/0
    volumes: [blob-data:/app/blob]
    depends_on: [redis]

  worker-assembler:
    build: ./svc-assembler
    command: celery -A assembler.celery_app worker -Q assembler -c 1
    environment:
      OHSHEET_BLOB_ROOT: /app/blob
      OHSHEET_REDIS_URL: redis://redis:6379/0
    volumes: [blob-data:/app/blob]
    depends_on: [redis]

  worker-humanize:
    build: .
    command: celery -A backend.workers.celery_app worker -Q humanize -c 1
    environment:
      OHSHEET_BLOB_ROOT: /app/blob
      OHSHEET_REDIS_URL: redis://redis:6379/0
    volumes: [blob-data:/app/blob]
    depends_on: [redis]

  worker-engrave:
    build: .
    command: celery -A backend.workers.celery_app worker -Q engrave -c 1
    environment:
      OHSHEET_BLOB_ROOT: /app/blob
      OHSHEET_REDIS_URL: redis://redis:6379/0
    volumes: [blob-data:/app/blob]
    depends_on: [redis]

volumes:
  blob-data:
```

Each worker listens on a dedicated queue (`-Q ingest`, `-Q decomposer`, etc.) so tasks route to the correct service. Concurrency is `-c 1` per worker for the MVP.

## Changes to Existing Code

### `backend/jobs/runner.py`
- Replace direct service calls with Celery `send_task()` + `asyncio.to_thread(.get())`
- Add step-to-task-name + step-to-queue mapping
- Serialization: `model_dump(mode="json")` before blob write, `model_validate()` after blob read

### `backend/config.py`
- Add `redis_url: str = "redis://localhost:6379/0"` to `Settings`

### `backend/contracts.py`
- Move to `shared/shared/contracts.py`
- `backend/contracts.py` becomes a re-export: `from shared.contracts import *`
- This keeps existing backend imports working while letting external services use `shared` directly
- All three `pyproject.toml` files (backend, svc-decomposer, svc-assembler) declare `shared` as a path dependency: `shared = {path = "../shared"}`

### `backend/storage/`
- Move `base.py` and `local.py` to `shared/shared/storage/`
- `backend/storage/` becomes re-exports
- No logic changes to `LocalBlobStore`

### `backend/workers/` (new)
- `celery_app.py`: Celery app instance configured from `settings.redis_url`
- `ingest.py`, `humanize.py`, `engrave.py`: task functions following the worker pattern above

### Existing tests
- Existing unit tests continue to pass — service classes are unchanged
- New integration test: submit a job via API, verify all 5 Celery tasks execute and artifacts appear in blob store

## Error Handling

- Celery task raises an exception → `.get()` re-raises in the orchestrator → `PipelineRunner` catches it and emits `job_failed` event (same as today)
- Celery task timeout → `celery.exceptions.TimeoutError` → same failure path
- No retries in stubs — the existing services don't retry, and adding retry semantics is a future concern

## Testing Strategy

- **Unit tests (per service):** Mock blob store, verify task reads input, calls service, writes output
- **Integration test (docker-compose):** `docker compose up`, POST a job, poll until succeeded, verify artifacts exist
- **Regression:** Existing `tests/` suite runs against backend with Celery workers active — same API contract, same results
