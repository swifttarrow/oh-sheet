from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api import deps
from backend.config import settings
from backend.main import create_app


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


@pytest.fixture
def client():
    """TestClient inside a `with` block so the lifespan and ASGI portal stay alive
    for the whole test. Without this, background asyncio tasks created during
    a request never get a chance to progress between sync calls."""
    app = create_app()
    with TestClient(app) as c:
        yield c
