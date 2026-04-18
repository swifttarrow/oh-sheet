from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import backend.workers.arrange  # noqa: F401
import backend.workers.condense  # noqa: F401
import backend.workers.humanize  # noqa: F401

# Import monolith worker modules so their tasks are registered on the celery_app.
import backend.workers.ingest  # noqa: F401
import backend.workers.refine  # noqa: F401
import backend.workers.transcribe  # noqa: F401
import backend.workers.transform  # noqa: F401
from backend.api import deps
from backend.config import settings
from backend.main import create_app
from backend.services import ml_engraver_client as ml_engraver_module
from backend.services import transcribe as transcribe_module
from backend.workers.celery_app import celery_app as _celery_app

_FAKE_ML_MUSICXML = (
    b'<?xml version="1.0" encoding="UTF-8" standalone="no"?>'
    b'<score-partwise version="3.1"><part id="P1"/></score-partwise>'
)


@pytest.fixture(autouse=True)
def isolated_blob_root(tmp_path: Path, monkeypatch):
    """Each test gets a fresh blob root and fresh DI singletons."""
    blob = tmp_path / "blob"
    blob.mkdir()
    monkeypatch.setattr(settings, "blob_root", blob)

    deps.get_blob_store.cache_clear()
    deps.get_runner.cache_clear()
    deps.get_job_manager.cache_clear()
    yield
    deps.get_blob_store.cache_clear()
    deps.get_runner.cache_clear()
    deps.get_job_manager.cache_clear()


@pytest.fixture(autouse=True)
def skip_real_transcription(monkeypatch):
    """Force TranscribeService onto its stub-fallback path in every test.

    The suite uses fake audio bytes to exercise pipeline orchestration — not
    transcription quality — and running real Basic Pitch inference on those
    bytes is both slow (cold-start CoreML/ONNX compilation) and flaky
    (librosa silently decodes garbage into zero-length audio). Raising an
    exception from the sync inference helper routes TranscribeService.run
    through its `except Exception -> _stub_result` branch, which is exactly
    what the pipeline tests need.
    """
    async def _fake_run(self, payload, *, job_id=None):
        stub = transcribe_module._stub_result("real transcription disabled in tests")
        if self.blob_store is not None and job_id is not None:
            fake_midi = b"MThd\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00"
            uri = self.blob_store.put_bytes(
                f"jobs/{job_id}/transcription/basic-pitch.mid", fake_midi,
            )
            stub = stub.model_copy(update={"transcription_midi_uri": uri})
        return stub

    monkeypatch.setattr(transcribe_module.TranscribeService, "run", _fake_run)


@pytest.fixture(autouse=True)
def celery_eager_mode():
    """Run Celery tasks in-process for all tests."""
    _celery_app.conf.task_always_eager = True
    _celery_app.conf.task_eager_propagates = True
    yield
    _celery_app.conf.task_always_eager = False
    _celery_app.conf.task_eager_propagates = False


@pytest.fixture(autouse=True)
def stub_ml_engraver(monkeypatch):
    """Replace the ML engraver HTTP client with a deterministic fake.

    The engrave stage always routes through this client now (no local
    fallback), so without this stub every pipeline test would try to
    reach a real ML service. Tests that want to exercise error paths
    override this within their own monkeypatch scope.
    """
    async def fake_engrave(midi_bytes: bytes) -> bytes:
        return _FAKE_ML_MUSICXML

    monkeypatch.setattr(
        ml_engraver_module, "engrave_midi_via_ml_service", fake_engrave,
    )


@pytest.fixture(autouse=True)
def disable_real_refine_llm(monkeypatch):
    """Null out the Anthropic API key for every test.

    No test may hit the real API. The service's fallback path (no key →
    raise → caught → pass-through with warning) is exactly what bare
    pipeline tests need. Tests that want to exercise the merge logic
    construct ``RefineService(..., client=fake)`` directly — that path
    bypasses the key check entirely.
    """
    monkeypatch.setattr(settings, "anthropic_api_key", None)


@pytest.fixture
def client():
    """TestClient inside a `with` block so the lifespan and ASGI portal stay alive
    for the whole test. Without this, background asyncio tasks created during
    a request never get a chance to progress between sync calls."""
    app = create_app()
    with TestClient(app) as c:
        yield c
