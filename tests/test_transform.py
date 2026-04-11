from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

from backend.contracts import (
    SCHEMA_VERSION,
    PianoScore,
    ScoreMetadata,
    ScoreNote,
    ScoreSection,
    TempoMapEntry,
)
from backend.services import transform as transform_mod
from backend.services.transform import TransformService


def test_transform_passthrough_returns_same_instance() -> None:
    score = PianoScore(
        schema_version=SCHEMA_VERSION,
        right_hand=[],
        left_hand=[],
        metadata=ScoreMetadata(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
            difficulty="intermediate",
        ),
    )
    svc = TransformService()

    async def _run() -> PianoScore:
        return await svc.run(score)

    out = asyncio.run(_run())
    assert out is score


def test_build_lyria_prompt_mentions_intermediate_piano_arrangement() -> None:
    score = PianoScore(
        schema_version=SCHEMA_VERSION,
        right_hand=[],
        left_hand=[],
        metadata=ScoreMetadata(
            key="G:major",
            time_signature=(3, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=96.0)],
            difficulty="intermediate",
            sections=[
                ScoreSection(start_beat=0.0, end_beat=16.0, label="verse"),
                ScoreSection(start_beat=16.0, end_beat=32.0, label="chorus"),
            ],
        ),
    )

    prompt = transform_mod._build_lyria_prompt(score)

    assert "solo acoustic piano arrangement" in prompt
    assert "Target player: intermediate pianist." in prompt
    assert "No vocals, drums, synths, pads, or extra instruments." in prompt
    assert "[Verse] about 16 beats" in prompt
    assert "[Chorus] about 16 beats" in prompt


def test_transform_lyria_writes_sidecar_and_preserves_score(monkeypatch, blob, caplog) -> None:
    monkeypatch.setattr(transform_mod.settings, "transform_lyria_enabled", True)
    monkeypatch.setattr(transform_mod.settings, "transform_lyria_model", "lyria-3-pro-preview")
    monkeypatch.setattr(transform_mod.settings, "transform_lyria_api_key", "test-key")
    caplog.set_level(logging.INFO, logger="backend.services.transform")

    fake_response = SimpleNamespace(
        parts=[
            SimpleNamespace(text='{"structure":["intro","theme"]}', inline_data=None),
            SimpleNamespace(
                text=None,
                inline_data=SimpleNamespace(data=b"fake-mp3", mime_type="audio/mp3"),
            ),
        ],
    )

    class _FakeModels:
        def generate_content(self, **kwargs):
            assert kwargs["model"] == "lyria-3-pro-preview"
            assert "intermediate pianist" in kwargs["contents"]
            return fake_response

    class _FakeClient:
        def __init__(self, *, api_key=None):
            assert api_key == "test-key"
            self.models = _FakeModels()

    fake_genai = SimpleNamespace(
        Client=_FakeClient,
        types=SimpleNamespace(
            GenerateContentConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    )
    monkeypatch.setattr(transform_mod, "_import_google_genai", lambda: fake_genai)

    score = PianoScore(
        schema_version=SCHEMA_VERSION,
        right_hand=[
            ScoreNote(
                id="rh-0001",
                pitch=72,
                onset_beat=0.0,
                duration_beat=1.0,
                velocity=88,
                voice=1,
            ),
        ],
        left_hand=[
            ScoreNote(
                id="lh-0001",
                pitch=48,
                onset_beat=0.0,
                duration_beat=1.0,
                velocity=74,
                voice=1,
            ),
        ],
        metadata=ScoreMetadata(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
            difficulty="intermediate",
        ),
    )

    svc = TransformService(blob_store=blob)

    async def _run() -> PianoScore:
        return await svc.run(score, job_id="job-123")

    out = asyncio.run(_run())

    assert out is score
    meta_uri = (blob.root / "jobs/job-123/transform/lyria-response.json").as_uri()
    audio_uri = (blob.root / "jobs/job-123/transform/lyria-arrangement.mp3").as_uri()
    meta = blob.get_json(meta_uri)
    assert meta["model"] == "lyria-3-pro-preview"
    assert "solo acoustic piano arrangement" in meta["prompt"]
    assert meta["text_parts"] == ['{"structure":["intro","theme"]}']
    assert blob.get_bytes(audio_uri) == b"fake-mp3"
    assert "transform: invoking Lyria job_id=job-123 model=lyria-3-pro-preview" in caplog.text
    assert "transform: Lyria completed job_id=job-123 model=lyria-3-pro-preview" in caplog.text
    assert "transform: persisted Lyria sidecar job_id=job-123" in caplog.text
