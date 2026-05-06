"""Unit tests for InterpretService — uses a fake Anthropic client.

The fake client mimics AsyncAnthropic.messages.create() just enough to
drive the service through its happy-path, error, and disabled-key paths
without touching the network.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from shared.contracts import (
    HarmonicAnalysis,
    InstrumentRole,
    MidiTrack,
    Note,
    QualitySignal,
    TempoMapEntry,
    TranscriptionResult,
)

from backend.config import settings
from backend.contracts import SCHEMA_VERSION
from backend.services.interpret import InterpretService

# ---------------------------------------------------------------------------
# Minimal fixture helpers
# ---------------------------------------------------------------------------

def _make_txr() -> TranscriptionResult:
    """Return a minimal but valid TranscriptionResult for testing."""
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
            warnings=[],
        ),
    )


# ---------------------------------------------------------------------------
# Fake Anthropic client helpers (same pattern as test_refine_service.py)
# ---------------------------------------------------------------------------

class _FakeToolUseBlock:
    def __init__(self, name: str, input_: dict[str, Any]) -> None:
        self.type = "tool_use"
        self.name = name
        self.input = input_


class _FakeResponse:
    def __init__(self, content: list[Any]) -> None:
        self.content = content


class _FakeMessages:
    def __init__(self, side_effect: Any) -> None:
        self._side_effect = side_effect

    async def create(self, **kwargs: Any) -> Any:
        if isinstance(self._side_effect, BaseException):
            raise self._side_effect
        if callable(self._side_effect):
            result = self._side_effect()
            if hasattr(result, "__await__"):
                return await result
            return result
        return self._side_effect


class _FakeAnthropic:
    def __init__(self, side_effect: Any) -> None:
        self.messages = _FakeMessages(side_effect)


# ---------------------------------------------------------------------------
# Test 1: Happy path — valid tool_use block, hints populated, no warning
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_happy_path_populates_arrangement_hints(monkeypatch):
    """A valid submit_arrangement_hints tool_use response populates .arrangement_hints."""
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test-key")
    monkeypatch.setattr(settings, "interpret_enabled", True)

    hints_dict = {
        "difficulty": "beginner",
        "density": "sparse",
        "tempo_bias": -0.1,
        "style_tags": ["jazz"],
    }
    fake_response = _FakeResponse([
        _FakeToolUseBlock("submit_arrangement_hints", hints_dict),
    ])
    fake_client = _FakeAnthropic(fake_response)

    svc = InterpretService(client=fake_client)
    txr = _make_txr()
    result = await svc.run(txr, prompt="make it easy")

    assert result.arrangement_hints is not None
    assert result.arrangement_hints.difficulty == "beginner"
    assert result.arrangement_hints.density == "sparse"
    assert result.arrangement_hints.tempo_bias == pytest.approx(-0.1)
    assert result.arrangement_hints.style_tags == ["jazz"]
    # No warning should be appended on success
    assert not any("interpret" in w for w in result.quality.warnings)


# ---------------------------------------------------------------------------
# Test 2: Error path — exception from messages.create adds warning, no hints
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_error_path_adds_warning_and_no_hints(monkeypatch):
    """When the LLM call raises, arrangement_hints stays None and a warning is appended."""
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test-key")
    monkeypatch.setattr(settings, "interpret_enabled", True)

    fake_client = _FakeAnthropic(RuntimeError("boom"))

    svc = InterpretService(client=fake_client)
    txr = _make_txr()
    result = await svc.run(txr, prompt="make it easy")

    assert result.arrangement_hints is None
    interpret_warnings = [w for w in result.quality.warnings if w.startswith("interpret: ")]
    assert len(interpret_warnings) == 1, f"Expected one interpret warning, got: {result.quality.warnings}"


# ---------------------------------------------------------------------------
# Test 3: Disabled path — no API key → short-circuit, no Anthropic client built
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_api_key_short_circuits_without_calling_anthropic(monkeypatch):
    """When anthropic_api_key is None, the service short-circuits before any LLM call."""
    monkeypatch.setattr(settings, "anthropic_api_key", None)
    monkeypatch.setattr(settings, "interpret_enabled", True)

    constructed = []

    def _fake_async_anthropic(*, api_key: str) -> Any:
        constructed.append(api_key)
        raise AssertionError("AsyncAnthropic should not be constructed when API key is absent")

    svc = InterpretService()  # No pre-injected client

    # Patch the import inside _get_client so we detect if it's ever called
    with patch("backend.services.interpret.AsyncAnthropic", _fake_async_anthropic, create=True):
        txr = _make_txr()
        result = await svc.run(txr, prompt="make it easy")

    assert result.arrangement_hints is None
    interpret_warnings = [w for w in result.quality.warnings if w.startswith("interpret: ")]
    assert len(interpret_warnings) == 1, f"Expected one interpret warning, got: {result.quality.warnings}"
    assert not constructed, "AsyncAnthropic constructor should not have been called"


# ---------------------------------------------------------------------------
# Test 4: Rate limit — exceeding the per-process cap short-circuits the call
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rate_limit_short_circuits_after_cap(monkeypatch):
    """Once the sliding-window cap is exceeded, the service returns the input
    unchanged with a ``rate_limited`` warning and does not invoke the LLM."""
    from backend.services import interpret as interpret_mod

    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test-key")
    monkeypatch.setattr(settings, "interpret_enabled", True)
    monkeypatch.setattr(settings, "interpret_max_calls_per_minute", 2)
    # Reset the per-process counter so prior tests don't leak into this one.
    interpret_mod._call_times.clear()

    call_counter = {"n": 0}

    def _make_response() -> Any:
        call_counter["n"] += 1
        return _FakeResponse([
            _FakeToolUseBlock("submit_arrangement_hints", {"difficulty": "beginner"}),
        ])

    fake_client = _FakeAnthropic(_make_response)
    svc = InterpretService(client=fake_client)

    # First two calls succeed and consume the budget.
    for _ in range(2):
        result = await svc.run(_make_txr(), prompt="make it easy")
        assert result.arrangement_hints is not None

    # Third call hits the cap — no LLM call, ``rate_limited`` warning.
    result = await svc.run(_make_txr(), prompt="make it easy")
    assert result.arrangement_hints is None
    assert call_counter["n"] == 2, "LLM should not be invoked once the cap is reached"
    assert any("rate_limited" in w for w in result.quality.warnings)


# ---------------------------------------------------------------------------
# Test 5: Rate limit disabled when cap == 0
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rate_limit_disabled_with_zero_cap(monkeypatch):
    """A zero cap turns the rate limiter off even after many calls."""
    from backend.services import interpret as interpret_mod

    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test-key")
    monkeypatch.setattr(settings, "interpret_enabled", True)
    monkeypatch.setattr(settings, "interpret_max_calls_per_minute", 0)
    interpret_mod._call_times.clear()

    fake_client = _FakeAnthropic(
        _FakeResponse([
            _FakeToolUseBlock("submit_arrangement_hints", {"difficulty": "beginner"}),
        ])
    )
    svc = InterpretService(client=fake_client)

    for _ in range(5):
        result = await svc.run(_make_txr(), prompt="make it easy")
        assert result.arrangement_hints is not None
