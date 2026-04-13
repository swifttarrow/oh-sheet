# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Oh Sheet is an automated pipeline that transforms songs (MP3, MIDI, or YouTube links) into playable piano sheet music. It uses a FastAPI backend with a 5-stage pipeline (Ingest → Transcribe → Arrange → Humanize → Engrave) and a Flutter cross-platform frontend.

## Common Commands

```bash
# Install
make install              # Everything (backend + ML deps + frontend)
make install-backend      # Backend + dev deps only (no ML)
make install-basic-pitch  # Audio transcription ML deps

# Run
make backend              # uvicorn on localhost:8000 (auto-reload)
make frontend             # Flutter run (default device)
make frontend DEVICE=chrome API_BASE_URL=http://192.168.1.42:8000

# Test & Lint
make test                 # pytest with coverage
make lint                 # ruff check + flutter analyze
make typecheck            # mypy

# Single test
pytest tests/test_uploads.py::test_upload_audio -v
```

## Architecture

### Backend (Python 3.10+, FastAPI)

**Pipeline stages** (`backend/workers/`, `svc-decomposer/`, `svc-assembler/`): `PipelineRunner` dispatches each stage as a Celery task via Redis. Monolith workers (ingest, humanize, engrave) live in `backend/workers/`; decomposer and assembler are separate services with their own Celery apps. Most stages are stubs returning shape-correct contracts; transcribe has a real Basic Pitch implementation.

**Data contracts** (`shared/shared/contracts.py`, re-exported by `backend/contracts.py`): Schema v3.0.0 Pydantic models define all inter-stage data. Key types: `InputBundle`, `TranscriptionResult`, `PianoScore`, `EngravedOutput`. Pipeline variants (`full`, `audio_upload`, `midi_upload`, `sheet_only`) determine which stages run.

**Job system** (`backend/jobs/`): `JobManager` tracks in-memory job state and fans out `JobEvent`s to WebSocket subscribers via asyncio Queues. `PipelineRunner` walks the execution plan from `PipelineConfig`, dispatching each stage via `celery_app.tasks[name].apply_async()` (or `send_task()` for remote-only workers).

**Storage** (`backend/storage/`): Claim-Check pattern — heavy files go to a blob store (local filesystem, S3 future), URIs passed between services.

**Config** (`backend/config.py`): Pydantic settings with `OHSHEET_` env prefix. Reads `.env` file.

### Frontend (Flutter/Dart)

Three-screen flow: upload → progress (WebSocket events) → result (PDF viewer + MIDI player). API base URL configured via `--dart-define=API_BASE_URL=<url>`.

### API Endpoints

- `POST /v1/uploads/{audio,midi}` — file upload
- `POST /v1/jobs` — create pipeline job
- `GET /v1/jobs/{id}` — poll status
- `WS /v1/jobs/{id}/ws` — live event stream
- `GET /v1/artifacts/{job_id}/{kind}` — download pdf/midi/musicxml
- `POST /v1/stages/{name}` — stateless worker endpoint
- OpenAPI docs at `/docs`

### Testing

Tests use `httpx.AsyncClient` via a pytest fixture (`client` in `tests/conftest.py`). Key fixtures:
- `isolated_blob_root`: Fresh blob dir per test, clears DI singleton cache
- `skip_real_transcription`: Monkeypatches Basic Pitch to trigger stub fallback

### Deployment

Multi-stage Dockerfile: Flutter web build → Python 3.12-slim runtime with ffmpeg. Deployed to GCP Cloud Run via GitHub Actions (`deploy.yml`, manual trigger). CI runs lint, typecheck, and test on PRs to main.

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Project planning lives in `.planning/` (PROJECT.md, REQUIREMENTS.md, ROADMAP.md, STATE.md, research/, codebase/). Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-plan-phase <n>` to plan a roadmap phase before execution
- `/gsd-execute-phase <n>` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->
