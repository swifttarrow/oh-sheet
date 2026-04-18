"""Unit tests for backend.services.ml_engraver_client.

Covers the behavior-critical paths added during PR review:
  * stub-looking payloads raise (not warn-and-return)
  * transient errors (timeout / 5xx) retry up to 3 attempts
  * non-retryable errors (4xx, stub payload) surface immediately
"""
from __future__ import annotations

import pytest

from backend.services import ml_engraver_client
from backend.services.ml_engraver_client import (
    MLEngraverError,
    engrave_midi_via_ml_service,
)

_REAL_MUSICXML = (
    b'<?xml version="1.0" encoding="UTF-8" standalone="no"?>'
    b'<score-partwise version="3.1"><part id="P1">'
    # Padding to clear the stub-ceiling threshold.
    + (b'<measure number="1"><note><pitch><step>C</step><octave>4</octave>'
       b'</pitch><duration>4</duration><type>whole</type></note></measure>' * 10)
    + b'</part></score-partwise>'
)
assert len(_REAL_MUSICXML) > 500  # guard the guard

_STUB_MUSICXML = (
    b'<?xml version="1.0" encoding="UTF-8" standalone="no"?>'
    b'<score-partwise version="3.1"><part id="P1"/></score-partwise>'
)
assert len(_STUB_MUSICXML) < 500  # confirms this is the "looks like stub" case


class _FakeResponse:
    def __init__(self, status_code: int, content: bytes) -> None:
        self.status_code = status_code
        self.content = content
        self.text = content.decode("utf-8", errors="replace")


class _FakeClient:
    """Scriptable httpx.AsyncClient stand-in — pops a result per call."""

    def __init__(self, scripted) -> None:
        self._scripted = list(scripted)
        self.call_count = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def post(self, *args, **kwargs):
        self.call_count += 1
        result = self._scripted.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def patch_httpx(monkeypatch):
    """Wire up a scripted client so each test controls response shape."""
    def _install(scripted):
        client = _FakeClient(scripted)

        def _factory(*args, **kwargs):
            return client

        monkeypatch.setattr(ml_engraver_client.httpx, "AsyncClient", _factory)
        return client

    return _install


@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch):
    """Collapse the retry backoff so the suite runs instantly."""
    monkeypatch.setattr(ml_engraver_client, "_BACKOFF_BASE_SEC", 0.0)


async def test_stub_payload_raises_not_returns(patch_httpx):
    """A response below the stub-ceiling must raise MLEngraverError so
    callers never see a silently-masked blank score. This is the single
    most important invariant introduced by the PR review."""
    patch_httpx([_FakeResponse(200, _STUB_MUSICXML)])

    with pytest.raises(MLEngraverError, match="suspiciously small"):
        await engrave_midi_via_ml_service(b"fake midi")


async def test_happy_path_returns_bytes(patch_httpx):
    patch_httpx([_FakeResponse(200, _REAL_MUSICXML)])
    out = await engrave_midi_via_ml_service(b"fake midi")
    assert out == _REAL_MUSICXML


async def test_5xx_retries_then_succeeds(patch_httpx):
    """Transient 5xx should trigger retry; eventual success is returned."""
    client = patch_httpx([
        _FakeResponse(503, b"service unavailable"),
        _FakeResponse(502, b"bad gateway"),
        _FakeResponse(200, _REAL_MUSICXML),
    ])
    out = await engrave_midi_via_ml_service(b"fake midi")
    assert out == _REAL_MUSICXML
    assert client.call_count == 3


async def test_timeout_retries_then_succeeds(patch_httpx):
    import httpx  # noqa: PLC0415
    client = patch_httpx([
        httpx.TimeoutException("boom"),
        _FakeResponse(200, _REAL_MUSICXML),
    ])
    out = await engrave_midi_via_ml_service(b"fake midi")
    assert out == _REAL_MUSICXML
    assert client.call_count == 2


async def test_4xx_fails_without_retry(patch_httpx):
    """Client errors are deterministic — retrying won't change the outcome.
    First attempt surfaces the error immediately.
    """
    client = patch_httpx([_FakeResponse(400, b"bad midi")])
    with pytest.raises(MLEngraverError, match="HTTP 400"):
        await engrave_midi_via_ml_service(b"fake midi")
    assert client.call_count == 1


async def test_5xx_exhausts_attempts_and_raises(patch_httpx):
    client = patch_httpx([
        _FakeResponse(503, b"unavailable"),
        _FakeResponse(503, b"unavailable"),
        _FakeResponse(503, b"unavailable"),
    ])
    with pytest.raises(MLEngraverError, match="HTTP 503"):
        await engrave_midi_via_ml_service(b"fake midi")
    assert client.call_count == 3


async def test_stub_payload_does_not_retry(patch_httpx):
    """A stub response is deterministic: retrying won't produce a real
    model output. Surface the error on the first attempt.
    """
    client = patch_httpx([_FakeResponse(200, _STUB_MUSICXML)])
    with pytest.raises(MLEngraverError, match="suspiciously small"):
        await engrave_midi_via_ml_service(b"fake midi")
    assert client.call_count == 1
