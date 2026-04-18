"""Title/composer precedence: refined ScoreMetadata > InputMetadata > defaults."""
from __future__ import annotations

import pytest
from shared.contracts import (
    InputBundle,
    InputMetadata,
    PipelineConfig,
)
from shared.storage.local import LocalBlobStore

from backend.config import settings
from backend.jobs.runner import PipelineRunner
from backend.services import ml_engraver_client
from backend.services import refine as refine_module
from backend.workers.celery_app import celery_app

_FAKE_MUSICXML = (
    b'<?xml version="1.0" encoding="UTF-8" standalone="no"?>'
    b'<score-partwise version="3.1"><part id="P1"/></score-partwise>'
)


@pytest.fixture(autouse=True)
def mock_ml_engraver(monkeypatch):
    """Stub the engraver service so these metadata-propagation tests don't
    require a live ML backend."""
    async def fake_engrave(midi_bytes: bytes) -> bytes:
        return _FAKE_MUSICXML

    monkeypatch.setattr(ml_engraver_client, "engrave_midi_via_ml_service", fake_engrave)


@pytest.mark.asyncio
async def test_engrave_prefers_refined_title_over_bundle(monkeypatch):
    """When refine populates ScoreMetadata.title, engrave uses it even if
    InputMetadata.title was supplied by the user."""
    async def _canned_refine(self, payload, *, title_hint=None, artist_hint=None, filename_hint=None):
        # Merge in known refined values.
        return self._merge(payload, {
            "title": "Canonical Title",
            "composer": "Canonical Composer",
        })

    monkeypatch.setattr(refine_module.RefineService, "run", _canned_refine)

    blob = LocalBlobStore(settings.blob_root)
    runner = PipelineRunner(blob_store=blob, celery_app=celery_app)
    bundle = InputBundle(
        metadata=InputMetadata(
            title="user-supplied typo",
            artist="user-supplied artist",
            source="audio_upload",
        ),
    )
    config = PipelineConfig(variant="audio_upload", enable_refine=True)
    result = await runner.run(job_id="t-meta-1", bundle=bundle, config=config)

    assert result.metadata.title == "Canonical Title"
    assert result.metadata.composer == "Canonical Composer"


@pytest.mark.asyncio
async def test_engrave_falls_back_to_bundle_when_refine_empty():
    """With refine disabled, engrave uses InputMetadata.title/artist."""
    blob = LocalBlobStore(settings.blob_root)
    runner = PipelineRunner(blob_store=blob, celery_app=celery_app)
    bundle = InputBundle(
        metadata=InputMetadata(
            title="My User Title",
            artist="My User Artist",
            source="audio_upload",
        ),
    )
    config = PipelineConfig(variant="audio_upload", enable_refine=False)
    result = await runner.run(job_id="t-meta-2", bundle=bundle, config=config)

    assert result.metadata.title == "My User Title"
    assert result.metadata.composer == "My User Artist"


@pytest.mark.asyncio
async def test_engrave_defaults_when_nothing_provided():
    blob = LocalBlobStore(settings.blob_root)
    runner = PipelineRunner(blob_store=blob, celery_app=celery_app)
    bundle = InputBundle(
        metadata=InputMetadata(title=None, artist=None, source="audio_upload"),
    )
    config = PipelineConfig(variant="audio_upload", enable_refine=False)
    result = await runner.run(job_id="t-meta-3", bundle=bundle, config=config)

    assert result.metadata.title == "Untitled"
    assert result.metadata.composer == "Unknown"
