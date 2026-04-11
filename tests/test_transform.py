from __future__ import annotations

import asyncio

from backend.contracts import (
    SCHEMA_VERSION,
    PianoScore,
    ScoreMetadata,
    TempoMapEntry,
)
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
