"""End-to-end integration test: canned LLM response → RefineService → merged envelope → engrave input."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from shared.contracts import (
    ExpressionMap,
    HumanizedPerformance,
    PianoScore,
    QualitySignal,
    ScoreMetadata,
    ScoreNote,
    TempoMapEntry,
)
from shared.storage.local import LocalBlobStore

from backend.config import settings
from backend.services.refine import RefineService


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "refine" / "canned_claude_response.json"


class _CannedToolUse:
    def __init__(self, name: str, input_: dict) -> None:
        self.type = "tool_use"
        self.name = name
        self.input = input_


class _CannedResponse:
    def __init__(self, content: list) -> None:
        self.content = content


class _CannedMessages:
    def __init__(self, raw: dict) -> None:
        self._raw = raw

    async def create(self, **_kwargs):
        blocks = [
            _CannedToolUse(b["name"], b["input"])
            for b in self._raw["content"]
            if b["type"] == "tool_use"
        ]
        return _CannedResponse(blocks)


class _CannedClient:
    def __init__(self, raw: dict) -> None:
        self.messages = _CannedMessages(raw)


@pytest.mark.asyncio
async def test_canned_clair_de_lune_merges_end_to_end():
    raw = json.loads(FIXTURE_PATH.read_text())
    client = _CannedClient(raw)
    blob = LocalBlobStore(settings.blob_root)

    score = PianoScore(
        right_hand=[
            ScoreNote(id="rh-1", pitch=73, onset_beat=0.0, duration_beat=2.0, velocity=70, voice=1),
        ],
        left_hand=[
            ScoreNote(id="lh-1", pitch=37, onset_beat=0.0, duration_beat=4.0, velocity=60, voice=1),
        ],
        metadata=ScoreMetadata(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
            difficulty="intermediate",
        ),
    )
    perf = HumanizedPerformance(
        expressive_notes=[],
        expression=ExpressionMap(),
        score=score,
        quality=QualitySignal(overall_confidence=0.9, warnings=[]),
    )

    svc = RefineService(blob_store=blob, client=client)
    result = await svc.run(perf, title_hint="claire de lune", artist_hint=None)

    assert isinstance(result, HumanizedPerformance)
    md = result.score.metadata
    assert md.title == "Clair de Lune"
    assert md.composer == "Claude Debussy"
    assert md.key == "Db:major"
    assert md.time_signature == (9, 8)
    assert md.tempo_map[0].bpm == pytest.approx(66.0)
    assert md.tempo_marking == "Andante très expressif"
    assert md.staff_split_hint == 60
    assert len(md.sections) == 3
    assert md.sections[0].custom_label == "A"
