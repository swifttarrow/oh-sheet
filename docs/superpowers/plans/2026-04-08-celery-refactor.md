# Celery Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the pipeline from in-process asyncio execution to Celery + Redis task dispatch, with `svc-decomposer` and `svc-assembler` as independent services. All 5 stages become Celery workers. Stubs delegate to existing service logic — behavior unchanged.

**Architecture:** Orchestrator-driven dispatch. `PipelineRunner` builds the execution plan per variant, serializes each stage's input to blob store, dispatches a Celery task, and waits for the result URI via `asyncio.to_thread(result.get)`. Workers are stateless: read payload from blob, run existing service class, write output to blob, return URI.

**Tech Stack:** Celery 5.x, Redis 7, existing FastAPI orchestrator, existing Pydantic contracts, shared Docker volume for blob store.

**Spec:** `docs/superpowers/specs/2026-04-08-celery-refactor-design.md`

---

## File Structure

### New files

| Path | Responsibility |
|------|---------------|
| `shared/pyproject.toml` | Package metadata for shared contracts lib |
| `shared/shared/__init__.py` | Package init |
| `shared/shared/contracts.py` | Pydantic models (extracted from `backend/contracts.py`) |
| `shared/shared/storage/__init__.py` | Package init |
| `shared/shared/storage/base.py` | `BlobStore` protocol (extracted from `backend/storage/base.py`) |
| `shared/shared/storage/local.py` | `LocalBlobStore` (extracted from `backend/storage/local.py`) |
| `backend/workers/__init__.py` | Package init |
| `backend/workers/celery_app.py` | Celery app instance + config |
| `backend/workers/ingest.py` | Celery task wrapping `IngestService` |
| `backend/workers/humanize.py` | Celery task wrapping `HumanizeService` |
| `backend/workers/engrave.py` | Celery task wrapping `EngraveService` |
| `svc-decomposer/pyproject.toml` | Package metadata |
| `svc-decomposer/Dockerfile` | Worker image |
| `svc-decomposer/decomposer/__init__.py` | Package init |
| `svc-decomposer/decomposer/celery_app.py` | Celery app instance |
| `svc-decomposer/decomposer/tasks.py` | Celery task wrapping `TranscribeService` |
| `svc-decomposer/tests/__init__.py` | Package init |
| `svc-decomposer/tests/test_tasks.py` | Unit tests |
| `svc-assembler/pyproject.toml` | Package metadata |
| `svc-assembler/Dockerfile` | Worker image |
| `svc-assembler/assembler/__init__.py` | Package init |
| `svc-assembler/assembler/celery_app.py` | Celery app instance |
| `svc-assembler/assembler/tasks.py` | Celery task wrapping `ArrangeService` |
| `svc-assembler/tests/__init__.py` | Package init |
| `svc-assembler/tests/test_tasks.py` | Unit tests |
| `docker-compose.yml` | Full topology: Redis + orchestrator + 5 workers |
| `tests/test_celery_dispatch.py` | Integration test for Celery-based runner |

### Modified files

| Path | Change |
|------|--------|
| `backend/contracts.py` | Replace body with re-exports from `shared.contracts` |
| `backend/storage/base.py` | Replace body with re-export from `shared.storage.base` |
| `backend/storage/local.py` | Replace body with re-export from `shared.storage.local` |
| `backend/config.py` | Add `redis_url` setting |
| `backend/jobs/runner.py` | Replace direct service calls with Celery dispatch |
| `backend/api/deps.py` | Wire `PipelineRunner` with blob store + celery app (remove service deps) |
| `pyproject.toml` | Add `celery[redis]` dep + `shared` path dep |

---

### Task 1: Extract shared contracts package

**Files:**
- Create: `shared/pyproject.toml`, `shared/shared/__init__.py`, `shared/shared/contracts.py`, `shared/shared/storage/__init__.py`, `shared/shared/storage/base.py`, `shared/shared/storage/local.py`
- Modify: `backend/contracts.py`, `backend/storage/base.py`, `backend/storage/local.py`, `pyproject.toml`

- [ ] **Step 1: Create `shared/pyproject.toml`**

```toml
[project]
name = "ohsheet-shared"
version = "0.1.0"
description = "Shared contracts and storage protocol for Oh Sheet pipeline"
requires-python = ">=3.10"
dependencies = [
    "pydantic>=2.5",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["shared"]
```

- [ ] **Step 2: Copy contracts and storage into shared package**

Create `shared/shared/__init__.py`:
```python
"""Shared contracts and storage for the Oh Sheet pipeline."""
```

Copy `backend/contracts.py` to `shared/shared/contracts.py` — the file content is identical. No import changes needed since it only imports from `pydantic` and stdlib.

Create `shared/shared/storage/__init__.py`:
```python
"""Blob storage protocol and implementations."""
```

Copy `backend/storage/base.py` to `shared/shared/storage/base.py` — identical content.

Copy `backend/storage/local.py` to `shared/shared/storage/local.py` — identical content.

- [ ] **Step 3: Replace backend modules with re-exports**

Replace `backend/contracts.py` body with:
```python
"""Re-export from shared package — keeps all existing imports working."""
from shared.contracts import *  # noqa: F401, F403
from shared.contracts import (  # noqa: F401 — explicit re-exports for type checkers
    SCHEMA_VERSION,
    Articulation,
    Difficulty,
    DynamicMarking,
    EngravedOutput,
    EngravedScoreData,
    ExpressionMap,
    ExpressiveNote,
    HarmonicAnalysis,
    HumanizedPerformance,
    InputBundle,
    InputMetadata,
    InstrumentRole,
    MidiTrack,
    Note,
    OrchestratorCommand,
    PedalEvent,
    PianoScore,
    PipelineConfig,
    PipelineVariant,
    QualitySignal,
    RealtimeChordEvent,
    RemoteAudioFile,
    RemoteMidiFile,
    ScoreChordEvent,
    ScoreMetadata,
    ScoreNote,
    ScoreSection,
    Section,
    SectionLabel,
    TempoChange,
    TempoMapEntry,
    TranscriptionResult,
    WorkerResponse,
    beat_to_sec,
    sec_to_beat,
)
```

Replace `backend/storage/base.py` body with:
```python
"""Re-export from shared package."""
from shared.storage.base import *  # noqa: F401, F403
from shared.storage.base import BlobStore  # noqa: F401
```

Replace `backend/storage/local.py` body with:
```python
"""Re-export from shared package."""
from shared.storage.local import *  # noqa: F401, F403
from shared.storage.local import LocalBlobStore  # noqa: F401
```

- [ ] **Step 4: Add shared as path dependency to backend's pyproject.toml**

In `pyproject.toml`, add to the `dependencies` list:
```toml
"ohsheet-shared @ file:shared",
```

- [ ] **Step 5: Install and run existing tests**

Run: `pip install -e ./shared && pip install -e .`
Then: `pytest tests/ -v`
Expected: All existing tests pass — imports resolve through re-exports.

- [ ] **Step 6: Commit**

```bash
git add shared/ backend/contracts.py backend/storage/base.py backend/storage/local.py pyproject.toml
git commit -m "refactor: extract shared contracts and storage package"
```

---

### Task 2: Add Redis config and Celery app for backend workers

**Files:**
- Modify: `backend/config.py`
- Modify: `pyproject.toml`
- Create: `backend/workers/__init__.py`, `backend/workers/celery_app.py`
- Test: `tests/test_celery_app.py`

- [ ] **Step 1: Write failing test for celery app import**

Create `tests/test_celery_app.py`:
```python
"""Verify the Celery app can be imported and configured."""
from backend.workers.celery_app import celery_app


def test_celery_app_name():
    assert celery_app.main == "ohsheet"


def test_celery_app_broker_from_settings():
    """Broker URL should come from settings.redis_url."""
    # Default is redis://localhost:6379/0
    assert "redis" in celery_app.conf.broker_url
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_celery_app.py -v`
Expected: `ModuleNotFoundError: No module named 'backend.workers'`

- [ ] **Step 3: Add celery dependency to pyproject.toml**

Add to `dependencies` list in `pyproject.toml`:
```
"celery[redis]>=5.3",
```

Run: `pip install -e .`

- [ ] **Step 4: Add redis_url to Settings**

In `backend/config.py`, add to the `Settings` class after `blob_root`:
```python
    # Redis URL for Celery broker + result backend.
    redis_url: str = "redis://localhost:6379/0"
```

- [ ] **Step 5: Create backend/workers package and celery_app**

Create `backend/workers/__init__.py`:
```python
"""Celery workers for pipeline stages that live in the monolith."""
```

Create `backend/workers/celery_app.py`:
```python
"""Celery application instance shared by all monolith workers."""
from celery import Celery

from backend.config import settings

celery_app = Celery(
    "ohsheet",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_default_queue="default",
    task_routes={
        "ingest.run": {"queue": "ingest"},
        "humanize.run": {"queue": "humanize"},
        "engrave.run": {"queue": "engrave"},
        "decomposer.run": {"queue": "decomposer"},
        "assembler.run": {"queue": "assembler"},
    },
)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_celery_app.py -v`
Expected: PASS

- [ ] **Step 7: Run full test suite**

Run: `pytest tests/ -v`
Expected: All existing tests still pass.

- [ ] **Step 8: Commit**

```bash
git add backend/config.py backend/workers/ pyproject.toml tests/test_celery_app.py
git commit -m "feat: add Celery app and Redis config for pipeline workers"
```

---

### Task 3: Create monolith Celery worker tasks (ingest, humanize, engrave)

**Files:**
- Create: `backend/workers/ingest.py`, `backend/workers/humanize.py`, `backend/workers/engrave.py`
- Test: `tests/test_worker_tasks.py`

- [ ] **Step 1: Write failing tests for worker tasks**

Create `tests/test_worker_tasks.py`:
```python
"""Unit tests for monolith Celery worker tasks.

Each task follows the same pattern: read input from blob, run existing
service, write output to blob, return output URI.
"""
import json
from pathlib import Path

import pytest

from backend.config import settings
from shared.contracts import (
    SCHEMA_VERSION,
    HarmonicAnalysis,
    InputBundle,
    InputMetadata,
    QualitySignal,
    TempoMapEntry,
)
from shared.storage.local import LocalBlobStore


@pytest.fixture
def blob(tmp_path):
    root = tmp_path / "blob"
    root.mkdir()
    settings.blob_root = root
    return LocalBlobStore(root)


class TestIngestTask:
    def test_reads_blob_runs_service_writes_output(self, blob):
        from backend.workers.ingest import run as ingest_run

        bundle = InputBundle(
            metadata=InputMetadata(title="Test", source="audio_upload"),
        )
        payload_uri = blob.put_json(
            "jobs/test-job/ingest/input.json",
            bundle.model_dump(mode="json"),
        )
        output_uri = ingest_run("test-job", payload_uri)
        result = blob.get_json(output_uri)
        assert result["metadata"]["title"] == "Test"


class TestHumanizeTask:
    def test_reads_blob_runs_service_writes_output(self, blob):
        from backend.workers.humanize import run as humanize_run
        from shared.contracts import (
            PianoScore,
            ScoreMetadata,
            ScoreNote,
        )

        score = PianoScore(
            right_hand=[
                ScoreNote(id="rh-0001", pitch=60, onset_beat=0.0, duration_beat=1.0, velocity=80, voice=1),
            ],
            left_hand=[
                ScoreNote(id="lh-0001", pitch=48, onset_beat=0.0, duration_beat=1.0, velocity=70, voice=1),
            ],
            metadata=ScoreMetadata(
                key="C:major",
                time_signature=(4, 4),
                tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
                difficulty="intermediate",
            ),
        )
        payload_uri = blob.put_json(
            "jobs/test-job/humanize/input.json",
            score.model_dump(mode="json"),
        )
        output_uri = humanize_run("test-job", payload_uri)
        result = blob.get_json(output_uri)
        assert "expressive_notes" in result
        assert "expression" in result


class TestEngraveTask:
    def test_reads_blob_runs_service_writes_output(self, blob):
        from backend.workers.engrave import run as engrave_run
        from shared.contracts import (
            PianoScore,
            ScoreMetadata,
            ScoreNote,
        )

        score = PianoScore(
            right_hand=[
                ScoreNote(id="rh-0001", pitch=60, onset_beat=0.0, duration_beat=1.0, velocity=80, voice=1),
            ],
            left_hand=[
                ScoreNote(id="lh-0001", pitch=48, onset_beat=0.0, duration_beat=1.0, velocity=70, voice=1),
            ],
            metadata=ScoreMetadata(
                key="C:major",
                time_signature=(4, 4),
                tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
                difficulty="intermediate",
            ),
        )
        # Engrave accepts HumanizedPerformance or PianoScore. The task wraps
        # this and also needs job_id, title, composer passed through.
        payload_uri = blob.put_json(
            "jobs/test-job/engrave/input.json",
            {
                "payload": score.model_dump(mode="json"),
                "payload_type": "PianoScore",
                "job_id": "test-job",
                "title": "Test Song",
                "composer": "Test Artist",
            },
        )
        output_uri = engrave_run("test-job", payload_uri)
        result = blob.get_json(output_uri)
        assert "pdf_uri" in result
        assert "musicxml_uri" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_worker_tasks.py -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement ingest worker task**

Create `backend/workers/ingest.py`:
```python
"""Celery task for the ingest pipeline stage."""
import asyncio

from backend.config import settings
from backend.services.ingest import IngestService
from backend.workers.celery_app import celery_app
from shared.contracts import InputBundle
from shared.storage.local import LocalBlobStore


@celery_app.task(name="ingest.run")
def run(job_id: str, payload_uri: str) -> str:
    blob = LocalBlobStore(settings.blob_root)
    raw = blob.get_json(payload_uri)
    bundle = InputBundle.model_validate(raw)

    service = IngestService(blob_store=blob)
    result = asyncio.run(service.run(bundle))

    output_uri = blob.put_json(
        f"jobs/{job_id}/ingest/output.json",
        result.model_dump(mode="json"),
    )
    return output_uri
```

- [ ] **Step 4: Implement humanize worker task**

Create `backend/workers/humanize.py`:
```python
"""Celery task for the humanize pipeline stage."""
import asyncio

from backend.config import settings
from backend.services.humanize import HumanizeService
from backend.workers.celery_app import celery_app
from shared.contracts import PianoScore
from shared.storage.local import LocalBlobStore


@celery_app.task(name="humanize.run")
def run(job_id: str, payload_uri: str) -> str:
    blob = LocalBlobStore(settings.blob_root)
    raw = blob.get_json(payload_uri)
    score = PianoScore.model_validate(raw)

    service = HumanizeService()
    result = asyncio.run(service.run(score))

    output_uri = blob.put_json(
        f"jobs/{job_id}/humanize/output.json",
        result.model_dump(mode="json"),
    )
    return output_uri
```

- [ ] **Step 5: Implement engrave worker task**

Create `backend/workers/engrave.py`:
```python
"""Celery task for the engrave pipeline stage.

Engrave is unique: it accepts either HumanizedPerformance or PianoScore,
plus extra args (job_id, title, composer). The task envelope wraps these
as a JSON object with a `payload_type` discriminator.
"""
import asyncio

from backend.config import settings
from backend.services.engrave import EngraveService
from backend.workers.celery_app import celery_app
from shared.contracts import HumanizedPerformance, PianoScore
from shared.storage.local import LocalBlobStore


@celery_app.task(name="engrave.run")
def run(job_id: str, payload_uri: str) -> str:
    blob = LocalBlobStore(settings.blob_root)
    raw = blob.get_json(payload_uri)

    payload_type = raw["payload_type"]
    payload_data = raw["payload"]
    title = raw.get("title", "Untitled")
    composer = raw.get("composer", "Unknown")

    if payload_type == "HumanizedPerformance":
        payload = HumanizedPerformance.model_validate(payload_data)
    else:
        payload = PianoScore.model_validate(payload_data)

    service = EngraveService(blob_store=blob)
    result = asyncio.run(service.run(payload, job_id=job_id, title=title, composer=composer))

    output_uri = blob.put_json(
        f"jobs/{job_id}/engrave/output.json",
        result.model_dump(mode="json"),
    )
    return output_uri
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_worker_tasks.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 7: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 8: Commit**

```bash
git add backend/workers/ingest.py backend/workers/humanize.py backend/workers/engrave.py tests/test_worker_tasks.py
git commit -m "feat: add Celery worker tasks for ingest, humanize, engrave"
```

---

### Task 4: Scaffold svc-decomposer

**Files:**
- Create: `svc-decomposer/pyproject.toml`, `svc-decomposer/Dockerfile`, `svc-decomposer/decomposer/__init__.py`, `svc-decomposer/decomposer/celery_app.py`, `svc-decomposer/decomposer/tasks.py`, `svc-decomposer/tests/__init__.py`, `svc-decomposer/tests/test_tasks.py`

- [ ] **Step 1: Write failing test**

Create `svc-decomposer/tests/__init__.py` (empty).

Create `svc-decomposer/tests/test_tasks.py`:
```python
"""Unit tests for the decomposer Celery task."""
import sys
from pathlib import Path

import pytest

# Ensure the svc-decomposer package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.contracts import (
    InputBundle,
    InputMetadata,
    RemoteAudioFile,
    TempoMapEntry,
)
from shared.storage.local import LocalBlobStore


@pytest.fixture
def blob(tmp_path):
    root = tmp_path / "blob"
    root.mkdir()
    return LocalBlobStore(root)


def test_decomposer_task_reads_input_writes_output(blob, monkeypatch):
    """Stub delegates to TranscribeService which falls back to stub result."""
    import decomposer.tasks as task_module

    # Patch settings so the task uses our temp blob root
    monkeypatch.setattr(task_module, "_get_blob_store", lambda: blob)

    bundle = InputBundle(
        audio=RemoteAudioFile(
            uri="file:///fake/audio.wav",
            format="wav",
            sample_rate=44100,
            duration_sec=10.0,
            channels=1,
        ),
        metadata=InputMetadata(title="Test", source="audio_upload"),
    )
    payload_uri = blob.put_json(
        "jobs/test-job/decomposer/input.json",
        bundle.model_dump(mode="json"),
    )
    output_uri = task_module.run("test-job", payload_uri)
    result = blob.get_json(output_uri)
    assert "midi_tracks" in result
    assert "analysis" in result
    assert "quality" in result
```

- [ ] **Step 2: Create pyproject.toml**

Create `svc-decomposer/pyproject.toml`:
```toml
[project]
name = "ohsheet-decomposer"
version = "0.1.0"
description = "Decomposer worker — transcription stage for Oh Sheet pipeline"
requires-python = ">=3.10"
dependencies = [
    "celery[redis]>=5.3",
    "ohsheet-shared @ file:../shared",
]

[project.optional-dependencies]
# The real transcription deps — not needed for stub mode
transcription = [
    "ohsheet @ file:..",
]
dev = [
    "pytest>=8.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["decomposer"]
```

- [ ] **Step 3: Create celery_app.py**

Create `svc-decomposer/decomposer/__init__.py` (empty).

Create `svc-decomposer/decomposer/celery_app.py`:
```python
"""Celery application instance for the decomposer worker."""
import os

from celery import Celery

_redis_url = os.environ.get("OHSHEET_REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "decomposer",
    broker=_redis_url,
    backend=_redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
)
```

- [ ] **Step 4: Create tasks.py**

Create `svc-decomposer/decomposer/tasks.py`:
```python
"""Decomposer Celery task — stub delegates to TranscribeService."""
import asyncio
import os
from pathlib import Path

from decomposer.celery_app import celery_app
from shared.contracts import InputBundle
from shared.storage.local import LocalBlobStore

_BLOB_ROOT = Path(os.environ.get("OHSHEET_BLOB_ROOT", "./blob"))


def _get_blob_store() -> LocalBlobStore:
    return LocalBlobStore(_BLOB_ROOT)


@celery_app.task(name="decomposer.run")
def run(job_id: str, payload_uri: str) -> str:
    blob = _get_blob_store()
    raw = blob.get_json(payload_uri)
    bundle = InputBundle.model_validate(raw)

    # Stub: delegate to existing TranscribeService
    from backend.services.transcribe import TranscribeService

    service = TranscribeService()
    result = asyncio.run(service.run(bundle))

    output_uri = blob.put_json(
        f"jobs/{job_id}/decomposer/output.json",
        result.model_dump(mode="json"),
    )
    return output_uri
```

- [ ] **Step 5: Create Dockerfile**

Create `svc-decomposer/Dockerfile`:
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# System deps for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install shared package first (changes less often)
COPY shared/ /app/shared/
RUN pip install --no-cache-dir /app/shared

# Install the main backend package (needed for TranscribeService stub)
COPY pyproject.toml /app/pyproject.toml
COPY backend/ /app/backend/
RUN pip install --no-cache-dir /app

# Install decomposer package
COPY svc-decomposer/ /app/svc-decomposer/
RUN pip install --no-cache-dir /app/svc-decomposer

ENV OHSHEET_BLOB_ROOT=/app/blob

CMD ["celery", "-A", "decomposer.celery_app", "worker", "-Q", "decomposer", "-c", "1", "--loglevel=info"]
```

- [ ] **Step 6: Run test**

Run: `cd svc-decomposer && pip install -e ../shared && pip install -e .. && pip install -e . && pytest tests/ -v`
Expected: PASS (TranscribeService will hit stub path because Basic Pitch isn't installed or audio is fake).

- [ ] **Step 7: Commit**

```bash
git add svc-decomposer/
git commit -m "feat: scaffold svc-decomposer with Celery task stub"
```

---

### Task 5: Scaffold svc-assembler

**Files:**
- Create: `svc-assembler/pyproject.toml`, `svc-assembler/Dockerfile`, `svc-assembler/assembler/__init__.py`, `svc-assembler/assembler/celery_app.py`, `svc-assembler/assembler/tasks.py`, `svc-assembler/tests/__init__.py`, `svc-assembler/tests/test_tasks.py`

- [ ] **Step 1: Write failing test**

Create `svc-assembler/tests/__init__.py` (empty).

Create `svc-assembler/tests/test_tasks.py`:
```python
"""Unit tests for the assembler Celery task."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.contracts import (
    SCHEMA_VERSION,
    HarmonicAnalysis,
    InstrumentRole,
    MidiTrack,
    Note,
    QualitySignal,
    TempoMapEntry,
    TranscriptionResult,
)
from shared.storage.local import LocalBlobStore


@pytest.fixture
def blob(tmp_path):
    root = tmp_path / "blob"
    root.mkdir()
    return LocalBlobStore(root)


def test_assembler_task_reads_input_writes_output(blob, monkeypatch):
    import assembler.tasks as task_module

    monkeypatch.setattr(task_module, "_get_blob_store", lambda: blob)

    txr = TranscriptionResult(
        midi_tracks=[
            MidiTrack(
                notes=[
                    Note(pitch=60, onset_sec=0.0, offset_sec=0.5, velocity=80),
                    Note(pitch=62, onset_sec=0.5, offset_sec=1.0, velocity=80),
                    Note(pitch=48, onset_sec=0.0, offset_sec=1.0, velocity=70),
                ],
                instrument=InstrumentRole.PIANO,
                program=0,
                confidence=0.9,
            ),
        ],
        analysis=HarmonicAnalysis(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
        ),
        quality=QualitySignal(overall_confidence=0.9),
    )
    payload_uri = blob.put_json(
        "jobs/test-job/assembler/input.json",
        txr.model_dump(mode="json"),
    )
    output_uri = task_module.run("test-job", payload_uri)
    result = blob.get_json(output_uri)
    assert "right_hand" in result
    assert "left_hand" in result
    assert "metadata" in result
```

- [ ] **Step 2: Create pyproject.toml**

Create `svc-assembler/pyproject.toml`:
```toml
[project]
name = "ohsheet-assembler"
version = "0.1.0"
description = "Assembler worker — arrangement stage for Oh Sheet pipeline"
requires-python = ">=3.10"
dependencies = [
    "celery[redis]>=5.3",
    "ohsheet-shared @ file:../shared",
]

[project.optional-dependencies]
arrangement = [
    "ohsheet @ file:..",
]
dev = [
    "pytest>=8.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["assembler"]
```

- [ ] **Step 3: Create celery_app.py**

Create `svc-assembler/assembler/__init__.py` (empty).

Create `svc-assembler/assembler/celery_app.py`:
```python
"""Celery application instance for the assembler worker."""
import os

from celery import Celery

_redis_url = os.environ.get("OHSHEET_REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "assembler",
    broker=_redis_url,
    backend=_redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
)
```

- [ ] **Step 4: Create tasks.py**

Create `svc-assembler/assembler/tasks.py`:
```python
"""Assembler Celery task — stub delegates to ArrangeService."""
import asyncio
import os
from pathlib import Path

from assembler.celery_app import celery_app
from shared.contracts import TranscriptionResult
from shared.storage.local import LocalBlobStore

_BLOB_ROOT = Path(os.environ.get("OHSHEET_BLOB_ROOT", "./blob"))


def _get_blob_store() -> LocalBlobStore:
    return LocalBlobStore(_BLOB_ROOT)


@celery_app.task(name="assembler.run")
def run(job_id: str, payload_uri: str) -> str:
    blob = _get_blob_store()
    raw = blob.get_json(payload_uri)
    txr = TranscriptionResult.model_validate(raw)

    from backend.services.arrange import ArrangeService

    service = ArrangeService()
    result = asyncio.run(service.run(txr))

    output_uri = blob.put_json(
        f"jobs/{job_id}/assembler/output.json",
        result.model_dump(mode="json"),
    )
    return output_uri
```

- [ ] **Step 5: Create Dockerfile**

Create `svc-assembler/Dockerfile`:
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install shared package first
COPY shared/ /app/shared/
RUN pip install --no-cache-dir /app/shared

# Install the main backend package (needed for ArrangeService stub)
COPY pyproject.toml /app/pyproject.toml
COPY backend/ /app/backend/
RUN pip install --no-cache-dir /app

# Install assembler package
COPY svc-assembler/ /app/svc-assembler/
RUN pip install --no-cache-dir /app/svc-assembler

ENV OHSHEET_BLOB_ROOT=/app/blob

CMD ["celery", "-A", "assembler.celery_app", "worker", "-Q", "assembler", "-c", "1", "--loglevel=info"]
```

- [ ] **Step 6: Run test**

Run: `cd svc-assembler && pip install -e ../shared && pip install -e .. && pip install -e . && pytest tests/ -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add svc-assembler/
git commit -m "feat: scaffold svc-assembler with Celery task stub"
```

---

### Task 6: Refactor PipelineRunner to dispatch via Celery

**Files:**
- Modify: `backend/jobs/runner.py`
- Modify: `backend/api/deps.py`
- Test: `tests/test_celery_dispatch.py`

- [ ] **Step 1: Write failing test for Celery-dispatched runner**

Create `tests/test_celery_dispatch.py`:
```python
"""Test the refactored PipelineRunner with Celery dispatch.

Uses Celery's eager mode (task_always_eager=True) so tasks execute
in-process synchronously — no Redis needed for unit tests.
"""
from pathlib import Path

import pytest

from backend.config import settings
from backend.jobs.events import JobEvent
from backend.jobs.runner import PipelineRunner
from backend.workers.celery_app import celery_app
from shared.contracts import (
    InputBundle,
    InputMetadata,
    PipelineConfig,
    RemoteAudioFile,
)
from shared.storage.local import LocalBlobStore


@pytest.fixture(autouse=True)
def celery_eager():
    """Run all Celery tasks in-process for testing."""
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True
    yield
    celery_app.conf.task_always_eager = False
    celery_app.conf.task_eager_propagates = False


@pytest.fixture
def blob(tmp_path):
    root = tmp_path / "blob"
    root.mkdir()
    settings.blob_root = root
    return LocalBlobStore(root)


@pytest.fixture
def runner(blob):
    return PipelineRunner(blob_store=blob, celery_app=celery_app)


@pytest.mark.asyncio
async def test_full_pipeline_via_celery(runner):
    """An audio_upload job should traverse all 5 stages via Celery tasks."""
    events: list[JobEvent] = []

    bundle = InputBundle(
        audio=RemoteAudioFile(
            uri="file:///fake/audio.wav",
            format="wav",
            sample_rate=44100,
            duration_sec=10.0,
            channels=1,
        ),
        metadata=InputMetadata(title="Test", artist="Tester", source="audio_upload"),
    )
    config = PipelineConfig(variant="audio_upload")

    result = await runner.run(
        job_id="test-celery-001",
        bundle=bundle,
        config=config,
        on_event=events.append,
    )

    assert result.pdf_uri
    assert result.musicxml_uri
    assert result.humanized_midi_uri

    stage_names = [e.stage for e in events if e.type == "stage_completed"]
    assert stage_names == ["ingest", "transcribe", "arrange", "humanize", "engrave"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pip install pytest-asyncio && pytest tests/test_celery_dispatch.py -v`
Expected: FAIL — `PipelineRunner.__init__` doesn't accept `blob_store` and `celery_app` params yet.

- [ ] **Step 3: Rewrite PipelineRunner**

Replace `backend/jobs/runner.py` with:
```python
"""PipelineRunner — dispatches pipeline stages as Celery tasks.

The runner owns the execution plan (which stages run in what order)
and uses the claim-check pattern: serialize each stage's input to
blob storage, dispatch a Celery task with the payload URI, wait for
the result URI, and deserialize the output for the next stage.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from celery import Celery

from backend.contracts import (
    SCHEMA_VERSION,
    EngravedOutput,
    HarmonicAnalysis,
    HumanizedPerformance,
    InputBundle,
    InstrumentRole,
    MidiTrack,
    Note,
    PianoScore,
    PipelineConfig,
    QualitySignal,
    TempoMapEntry,
    TranscriptionResult,
)
from backend.jobs.events import JobEvent
from backend.storage.base import BlobStore

log = logging.getLogger(__name__)

EventCallback = Callable[[JobEvent], None]

# Maps execution plan step names to Celery task names.
STEP_TO_TASK: dict[str, str] = {
    "ingest": "ingest.run",
    "transcribe": "decomposer.run",
    "arrange": "assembler.run",
    "humanize": "humanize.run",
    "engrave": "engrave.run",
}


def _gm_program_to_role(program: int, is_drum: bool) -> InstrumentRole:
    if is_drum:
        return InstrumentRole.OTHER
    if program < 8:
        return InstrumentRole.PIANO
    if 32 <= program <= 39:
        return InstrumentRole.BASS
    if 72 <= program <= 79:
        return InstrumentRole.MELODY
    return InstrumentRole.CHORDS


def _stub_transcription(reason: str) -> TranscriptionResult:
    return TranscriptionResult(
        schema_version=SCHEMA_VERSION,
        midi_tracks=[
            MidiTrack(
                notes=[
                    Note(pitch=60, onset_sec=0.0, offset_sec=0.5, velocity=80),
                    Note(pitch=62, onset_sec=0.5, offset_sec=1.0, velocity=80),
                ],
                instrument=InstrumentRole.PIANO,
                program=0,
                confidence=0.9,
            ),
        ],
        analysis=HarmonicAnalysis(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
            chords=[],
            sections=[],
        ),
        quality=QualitySignal(
            overall_confidence=0.5,
            warnings=[f"midi-to-transcription stub: {reason}"],
        ),
    )


def _bundle_to_transcription(bundle: InputBundle) -> TranscriptionResult:
    """Build a TranscriptionResult from a midi_upload bundle.

    Real path: parse the MIDI file via pretty_midi, recover the tempo map,
    fold each instrument into a MidiTrack, infer key/time-signature.
    Fallback: a small shape-correct stub so downstream stages still run.
    """
    if bundle.midi is None:
        return _stub_transcription("no midi in bundle")

    parsed = urlparse(bundle.midi.uri)
    if parsed.scheme != "file":
        return _stub_transcription(f"unsupported midi URI scheme: {parsed.scheme!r}")
    midi_path = Path(parsed.path)
    if not midi_path.is_file():
        return _stub_transcription(f"midi file missing: {midi_path}")

    try:
        import pretty_midi  # noqa: PLC0415 — optional dep
    except ImportError:
        return _stub_transcription("pretty_midi not installed")

    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as exc:  # noqa: BLE001
        return _stub_transcription(f"pretty_midi parse failed: {exc}")

    midi_tracks: list[MidiTrack] = []
    for instrument in pm.instruments:
        notes = [
            Note(
                pitch=int(n.pitch),
                onset_sec=float(n.start),
                offset_sec=float(max(n.end, n.start + 0.01)),
                velocity=int(max(1, min(127, n.velocity))),
            )
            for n in instrument.notes
        ]
        if not notes:
            continue
        midi_tracks.append(MidiTrack(
            notes=notes,
            instrument=_gm_program_to_role(int(instrument.program), bool(instrument.is_drum)),
            program=None if instrument.is_drum else int(instrument.program),
            confidence=0.9,
        ))

    if not midi_tracks:
        return _stub_transcription("midi file contained no notes")

    tempo_times, tempo_bpms = pm.get_tempo_changes()
    tempo_map: list[TempoMapEntry] = []
    if len(tempo_times) > 0:
        beat_cursor = 0.0
        prev_time = 0.0
        prev_bpm = float(tempo_bpms[0])
        for t, bpm in zip(tempo_times, tempo_bpms):
            t = float(t)
            bpm = float(bpm)
            beat_cursor += (t - prev_time) * (prev_bpm / 60.0)
            tempo_map.append(TempoMapEntry(time_sec=t, beat=beat_cursor, bpm=bpm))
            prev_time = t
            prev_bpm = bpm
    if not tempo_map:
        tempo_map = [TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)]

    time_signature: tuple[int, int] = (4, 4)
    if pm.time_signature_changes:
        first = pm.time_signature_changes[0]
        time_signature = (int(first.numerator), int(first.denominator))

    key = "C:major"
    if pm.key_signature_changes:
        try:
            key_number = int(pm.key_signature_changes[0].key_number)
            key = pretty_midi.key_number_to_key_name(key_number).replace(" ", ":")
        except Exception:  # noqa: BLE001
            pass

    total_notes = sum(len(t.notes) for t in midi_tracks)
    log.info(
        "Parsed MIDI %s: %d tracks, %d notes",
        midi_path.name, len(midi_tracks), total_notes,
    )

    return TranscriptionResult(
        schema_version=SCHEMA_VERSION,
        midi_tracks=midi_tracks,
        analysis=HarmonicAnalysis(
            key=key,
            time_signature=time_signature,
            tempo_map=tempo_map,
            chords=[],
            sections=[],
        ),
        quality=QualitySignal(
            overall_confidence=0.95,
            warnings=["MIDI input — no harmonic analysis"],
        ),
    )


class PipelineRunner:
    def __init__(
        self,
        blob_store: BlobStore,
        celery_app: Celery,
    ) -> None:
        self.blob_store = blob_store
        self.celery_app = celery_app

    def _serialize_stage_input(
        self,
        job_id: str,
        step: str,
        payload: dict,
    ) -> str:
        """Write stage input to blob store, return URI."""
        return self.blob_store.put_json(
            f"jobs/{job_id}/{step}/input.json",
            payload,
        )

    async def _dispatch_task(
        self,
        task_name: str,
        job_id: str,
        payload_uri: str,
        timeout: int,
    ) -> str:
        """Send Celery task and wait for result URI without blocking event loop."""
        result = self.celery_app.send_task(task_name, args=[job_id, payload_uri])
        output_uri = await asyncio.to_thread(result.get, timeout=timeout)
        return output_uri

    async def run(
        self,
        *,
        job_id: str,
        bundle: InputBundle,
        config: PipelineConfig,
        on_event: EventCallback | None = None,
    ) -> EngravedOutput:
        plan = config.get_execution_plan()
        n = len(plan)

        def emit(stage: str, event_type, **kw) -> None:
            if on_event is None:
                return
            on_event(JobEvent(job_id=job_id, type=event_type, stage=stage, **kw))

        title = bundle.metadata.title or "Untitled"
        composer = bundle.metadata.artist or "Unknown"

        # Current state as we walk the pipeline — always a dict for JSON serialization.
        current_payload: dict = bundle.model_dump(mode="json")
        txr_dict: dict | None = None
        score_dict: dict | None = None
        perf_dict: dict | None = None

        for i, step in enumerate(plan):
            emit(step, "stage_started", progress=i / n)

            task_name = STEP_TO_TASK[step]

            if step == "ingest":
                payload_uri = self._serialize_stage_input(job_id, step, current_payload)
                output_uri = await self._dispatch_task(task_name, job_id, payload_uri, config.job_timeout_sec)
                current_payload = self.blob_store.get_json(output_uri)
                # current_payload is now the enriched InputBundle dict

            elif step == "transcribe":
                payload_uri = self._serialize_stage_input(job_id, step, current_payload)
                output_uri = await self._dispatch_task(task_name, job_id, payload_uri, config.job_timeout_sec)
                txr_dict = self.blob_store.get_json(output_uri)

            elif step == "arrange":
                if txr_dict is None:
                    # midi_upload variant: build transcription from bundle
                    bundle_obj = InputBundle.model_validate(current_payload)
                    txr_obj = _bundle_to_transcription(bundle_obj)
                    txr_dict = txr_obj.model_dump(mode="json")
                payload_uri = self._serialize_stage_input(job_id, step, txr_dict)
                output_uri = await self._dispatch_task(task_name, job_id, payload_uri, config.job_timeout_sec)
                score_dict = self.blob_store.get_json(output_uri)

            elif step == "humanize":
                if score_dict is None:
                    raise RuntimeError("humanize stage requires a PianoScore — none was produced")
                payload_uri = self._serialize_stage_input(job_id, step, score_dict)
                output_uri = await self._dispatch_task(task_name, job_id, payload_uri, config.job_timeout_sec)
                perf_dict = self.blob_store.get_json(output_uri)

            elif step == "engrave":
                # Engrave needs special envelope with payload_type discriminator
                if perf_dict is not None:
                    engrave_envelope = {
                        "payload": perf_dict,
                        "payload_type": "HumanizedPerformance",
                        "job_id": job_id,
                        "title": title,
                        "composer": composer,
                    }
                elif score_dict is not None:
                    engrave_envelope = {
                        "payload": score_dict,
                        "payload_type": "PianoScore",
                        "job_id": job_id,
                        "title": title,
                        "composer": composer,
                    }
                else:
                    raise RuntimeError("engrave stage requires a score or performance — none was produced")
                payload_uri = self._serialize_stage_input(job_id, step, engrave_envelope)
                output_uri = await self._dispatch_task(task_name, job_id, payload_uri, config.job_timeout_sec)
                result_dict = self.blob_store.get_json(output_uri)

            else:
                raise RuntimeError(f"unknown stage in execution plan: {step!r}")

            emit(step, "stage_completed", progress=(i + 1) / n)

        if step != "engrave":
            raise RuntimeError("pipeline finished without an engrave stage")

        return EngravedOutput.model_validate(result_dict)
```

- [ ] **Step 4: Update deps.py to wire new PipelineRunner**

Replace `backend/api/deps.py` with:
```python
"""Shared dependency providers — wire singletons here.

These are kept as ``lru_cache``-d module-level functions so:

  * FastAPI's ``Depends(...)`` resolves them once per process.
  * Tests can ``cache_clear()`` between cases (see tests/conftest.py).
"""
from __future__ import annotations

from functools import lru_cache

from backend.config import settings
from backend.jobs.manager import JobManager
from backend.jobs.runner import PipelineRunner
from backend.storage.local import LocalBlobStore
from backend.workers.celery_app import celery_app


@lru_cache(maxsize=1)
def get_blob_store() -> LocalBlobStore:
    return LocalBlobStore(settings.blob_root)


@lru_cache(maxsize=1)
def get_runner() -> PipelineRunner:
    return PipelineRunner(
        blob_store=get_blob_store(),
        celery_app=celery_app,
    )


@lru_cache(maxsize=1)
def get_job_manager() -> JobManager:
    return JobManager(runner=get_runner())
```

- [ ] **Step 5: Run the new dispatch test**

Run: `pytest tests/test_celery_dispatch.py -v`
Expected: PASS — Celery eager mode runs tasks in-process.

- [ ] **Step 6: Update conftest.py to enable Celery eager mode**

The existing tests use `TestClient` which runs synchronously. They need Celery eager mode enabled. Add to `tests/conftest.py`:

```python
from backend.workers.celery_app import celery_app as _celery_app


@pytest.fixture(autouse=True)
def celery_eager_mode():
    """Run Celery tasks in-process for all tests."""
    _celery_app.conf.task_always_eager = True
    _celery_app.conf.task_eager_propagates = True
    yield
    _celery_app.conf.task_always_eager = False
    _celery_app.conf.task_eager_propagates = False
```

- [ ] **Step 7: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass. Existing tests use the same API surface — `JobManager` → `PipelineRunner.run()` → now dispatches via Celery eager mode instead of direct calls.

- [ ] **Step 8: Commit**

```bash
git add backend/jobs/runner.py backend/api/deps.py tests/test_celery_dispatch.py tests/conftest.py
git commit -m "feat: refactor PipelineRunner to dispatch stages via Celery"
```

---

### Task 7: Create docker-compose.yml

**Files:**
- Create: `docker-compose.yml`

- [ ] **Step 1: Create docker-compose.yml**

```yaml
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  orchestrator:
    build: .
    command: uvicorn backend.main:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    environment:
      OHSHEET_BLOB_ROOT: /app/blob
      OHSHEET_REDIS_URL: redis://redis:6379/0
    volumes:
      - blob-data:/app/blob
    depends_on:
      redis:
        condition: service_healthy

  worker-ingest:
    build: .
    command: celery -A backend.workers.celery_app worker -Q ingest -c 1 --loglevel=info
    environment:
      OHSHEET_BLOB_ROOT: /app/blob
      OHSHEET_REDIS_URL: redis://redis:6379/0
    volumes:
      - blob-data:/app/blob
    depends_on:
      redis:
        condition: service_healthy

  worker-decomposer:
    build:
      context: .
      dockerfile: svc-decomposer/Dockerfile
    environment:
      OHSHEET_BLOB_ROOT: /app/blob
      OHSHEET_REDIS_URL: redis://redis:6379/0
    volumes:
      - blob-data:/app/blob
    depends_on:
      redis:
        condition: service_healthy

  worker-assembler:
    build:
      context: .
      dockerfile: svc-assembler/Dockerfile
    environment:
      OHSHEET_BLOB_ROOT: /app/blob
      OHSHEET_REDIS_URL: redis://redis:6379/0
    volumes:
      - blob-data:/app/blob
    depends_on:
      redis:
        condition: service_healthy

  worker-humanize:
    build: .
    command: celery -A backend.workers.celery_app worker -Q humanize -c 1 --loglevel=info
    environment:
      OHSHEET_BLOB_ROOT: /app/blob
      OHSHEET_REDIS_URL: redis://redis:6379/0
    volumes:
      - blob-data:/app/blob
    depends_on:
      redis:
        condition: service_healthy

  worker-engrave:
    build: .
    command: celery -A backend.workers.celery_app worker -Q engrave -c 1 --loglevel=info
    environment:
      OHSHEET_BLOB_ROOT: /app/blob
      OHSHEET_REDIS_URL: redis://redis:6379/0
    volumes:
      - blob-data:/app/blob
    depends_on:
      redis:
        condition: service_healthy

volumes:
  blob-data:
```

- [ ] **Step 2: Verify compose config is valid**

Run: `docker compose config --quiet`
Expected: Exit 0, no errors.

- [ ] **Step 3: Commit**

```bash
git add docker-compose.yml
git commit -m "infra: add docker-compose with Redis + orchestrator + 5 Celery workers"
```

---

### Task 8: Smoke test with docker-compose

- [ ] **Step 1: Build all images**

Run: `docker compose build`
Expected: All 4 images build successfully (backend, svc-decomposer, svc-assembler, redis is pulled).

- [ ] **Step 2: Start the stack**

Run: `docker compose up -d`
Expected: All 7 containers start. Verify with `docker compose ps` — all should be "running" or "healthy".

- [ ] **Step 3: Submit a test job**

Run:
```bash
# Upload a small test audio file (or use title-based job)
curl -s -X POST http://localhost:8000/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"title": "Test Song", "artist": "Test"}' | python -m json.tool
```
Expected: Returns a job JSON with `status: "pending"` or `"running"` and a `job_id`.

- [ ] **Step 4: Poll for completion**

Run:
```bash
# Replace JOB_ID with the actual ID from step 3
curl -s http://localhost:8000/v1/jobs/JOB_ID | python -m json.tool
```
Expected: After a few seconds, `status: "succeeded"` with `pdf_uri`, `musicxml_uri`, `humanized_midi_uri` in the result.

- [ ] **Step 5: Check worker logs**

Run: `docker compose logs worker-decomposer worker-assembler --tail=20`
Expected: Logs show tasks being received and completed.

- [ ] **Step 6: Tear down**

Run: `docker compose down -v`

- [ ] **Step 7: Commit any fixes needed**

If any issues were found during smoke testing, fix them and commit:
```bash
git add -A
git commit -m "fix: address smoke test issues from docker-compose run"
```
