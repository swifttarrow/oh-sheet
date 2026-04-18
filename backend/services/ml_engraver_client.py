"""HTTP client for the oh-sheet-ml-pipeline engraver service.

Oh Sheet POSTs MIDI bytes to the service's ``/engrave`` endpoint and
receives MusicXML bytes in response. This is the only engrave path —
there is no local fallback — so failures propagate as job errors.

Transient errors (timeouts, 5xx) retry a small number of times with
backoff before surfacing; this is a different failure-mode than a
fallback (still only the ML service, just one more chance) and keeps
the pipeline tolerant of brief upstream blips without masking real
outages.

The exception hierarchy is the single source of truth for retry
classification. ``_is_retryable`` inspects ``exc.retryable`` rather
than pattern-matching on error text; adding a new failure mode means
adding a subclass with the right flag, not editing a substring list.
"""
from __future__ import annotations

import asyncio
import logging

import httpx

from backend.config import settings

log = logging.getLogger(__name__)


class MLEngraverError(RuntimeError):
    """Base class — every engraver-client failure derives from this.

    Subclasses set ``retryable`` at the class level. Callers outside
    this module only need to catch ``MLEngraverError``; the subclass
    is there for the retry loop and for targeted tests.
    """
    retryable: bool = False


class MLEngraverTimeout(MLEngraverError):
    """Upstream request didn't respond within the per-call timeout."""
    retryable = True


class MLEngraverTransportError(MLEngraverError):
    """Connection refused, DNS failure, unexpected disconnect, etc.

    Treated as transient — Cloud Run / load balancers can drop
    connections briefly under load without the service being sick.
    """
    retryable = True


class MLEngraverUpstreamError(MLEngraverError):
    """Service returned a 5xx. Likely transient — retry."""
    retryable = True

    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text[:200]
        super().__init__(
            f"engraver service returned HTTP {status_code}: {self.text}"
        )


class MLEngraverClientError(MLEngraverError):
    """Service returned a 4xx. Deterministic — retrying won't help."""
    retryable = False

    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text[:200]
        super().__init__(
            f"engraver service returned HTTP {status_code}: {self.text}"
        )


class MLEngraverStub(MLEngraverError):
    """Response passed the ``_looks_like_stub`` size filter.

    A real seq2seq transcription's MusicXML is many KB; anything below
    the ceiling is almost certainly the in-tree placeholder skeleton.
    Surfacing that as a success would silently hand the user a blank
    score — the exact failure mode this PR set out to kill — so we
    raise here and let the job fail loudly.
    """
    retryable = False


# Threshold below which a 200-OK response is treated as the placeholder
# skeleton. A real transcription runs many KB; the stub is a few
# hundred bytes of boilerplate.
_STUB_MUSICXML_BYTE_CEILING = 500

# Retry policy for transient upstream failures. The full pipeline has
# already run ingest/transcribe/arrange/humanize by the time we get here,
# so a retry on a momentary timeout or 5xx is cheap insurance — and it's
# NOT a fallback (same service, same contract, just one more attempt).
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_SEC = 0.5


def _looks_like_stub(musicxml_bytes: bytes) -> bool:
    return len(musicxml_bytes) < _STUB_MUSICXML_BYTE_CEILING


async def engrave_midi_via_ml_service(midi_bytes: bytes) -> bytes:
    """POST MIDI bytes to the engraver service, return MusicXML bytes.

    Raises some subclass of ``MLEngraverError`` on transport failure,
    timeout, non-2xx, or a stub-sized response. Retryable subclasses
    (``MLEngraverTimeout``, ``MLEngraverTransportError``,
    ``MLEngraverUpstreamError``) retry up to ``_MAX_ATTEMPTS`` with
    exponential backoff. Non-retryable subclasses (4xx, stub) surface
    on the first attempt.
    """
    url = f"{settings.engraver_service_url.rstrip('/')}/engrave"
    timeout = settings.engraver_service_timeout_sec

    log.info("ml_engraver: POST %s bytes_in=%d timeout=%ds", url, len(midi_bytes), timeout)

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            musicxml = await _post_once(url, midi_bytes, timeout)
        except MLEngraverError as exc:
            if not exc.retryable or attempt == _MAX_ATTEMPTS:
                raise
            backoff = _BACKOFF_BASE_SEC * (2 ** (attempt - 1))
            log.warning(
                "ml_engraver: attempt %d/%d failed (%s); retrying in %.1fs",
                attempt, _MAX_ATTEMPTS, exc, backoff,
            )
            await asyncio.sleep(backoff)
            continue
        log.info("ml_engraver: success bytes_out=%d attempt=%d", len(musicxml), attempt)
        return musicxml

    # Unreachable: the loop always returns on success or re-raises on
    # the final attempt. Kept to satisfy static return-type checking.
    raise AssertionError("retry loop fell through without returning or raising")


async def _post_once(url: str, midi_bytes: bytes, timeout: int) -> bytes:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                content=midi_bytes,
                headers={"Content-Type": "application/octet-stream"},
            )
    except httpx.TimeoutException as exc:
        raise MLEngraverTimeout(
            f"engraver service timed out after {timeout}s"
        ) from exc
    except httpx.HTTPError as exc:
        raise MLEngraverTransportError(
            f"engraver service transport error: {exc}"
        ) from exc

    if 500 <= response.status_code < 600:
        raise MLEngraverUpstreamError(response.status_code, response.text)
    if response.status_code != 200:
        raise MLEngraverClientError(response.status_code, response.text)

    musicxml = response.content
    if _looks_like_stub(musicxml):
        raise MLEngraverStub(
            f"engraver service returned suspiciously small payload "
            f"(bytes_out={len(musicxml)} < {_STUB_MUSICXML_BYTE_CEILING}); "
            f"service is likely running the in-tree placeholder rather "
            f"than a real model. Refusing to surface a blank score."
        )
    return musicxml
