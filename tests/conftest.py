from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api import deps
from backend.config import settings
from backend.main import create_app
from backend.services import transcribe as transcribe_module


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
    through its `except Exception → _stub_result` branch, which is exactly
    what the pipeline tests need.
    """
    def _fail_fast(*_args, **_kwargs):
        raise RuntimeError("real transcription disabled in tests")

    monkeypatch.setattr(transcribe_module, "_run_basic_pitch_sync", _fail_fast)


@pytest.fixture
def client():
    """TestClient inside a `with` block so the lifespan and ASGI portal stay alive
    for the whole test. Without this, background asyncio tasks created during
    a request never get a chance to progress between sync calls."""
    app = create_app()
    with TestClient(app) as c:
        yield c
