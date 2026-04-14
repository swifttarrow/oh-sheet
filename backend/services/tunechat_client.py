"""HTTP client for TuneChat's transcription API.

Oh Sheet sends audio bytes to TuneChat and receives back a job ID
(for the "Open in TuneChat" deep link) and a preview image URL
(a first-page PNG of the rendered score for display in Oh Sheet's
result screen).

Silent-failure contract: any error returns ``None`` so the caller
can fall back to Oh Sheet's own pipeline. This function NEVER raises.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from backend.config import settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TuneChatResult:
    """The two things Oh Sheet needs from TuneChat."""

    job_id: str
    preview_image_url: str | None


async def transcribe_via_tunechat(
    audio_bytes: bytes,
    filename: str,
    title: str | None = None,
    artist: str | None = None,
) -> TuneChatResult | None:
    """Send audio to TuneChat, get back a job ID + preview image.

    Returns ``None`` on any failure. The caller should fall back to
    Oh Sheet's own pipeline result. Gated by ``settings.tunechat_enabled``.
    """
    if not settings.tunechat_enabled:
        return None

    if not settings.tunechat_api_key:
        log.warning("tunechat: enabled but OHSHEET_TUNECHAT_API_KEY is empty — skipping")
        return None

    url = f"{settings.tunechat_url.rstrip('/')}/api/v1/transcribe"

    try:
        async with httpx.AsyncClient(timeout=settings.tunechat_timeout_sec) as client:
            log.info("tunechat: sending %d bytes (%s) to %s", len(audio_bytes), filename, url)

            form_data: dict[str, str] = {}
            if title:
                form_data["title"] = title
            if artist:
                form_data["artist"] = artist
            response = await client.post(
                url,
                headers={"Authorization": f"Bearer {settings.tunechat_api_key}"},
                files={"file": (filename, audio_bytes, "application/octet-stream")},
                data=form_data,
            )

            if response.status_code != 200:
                log.warning("tunechat: HTTP %d: %s", response.status_code, response.text[:200])
                return None

            data = response.json()
            result = TuneChatResult(
                job_id=data["jobId"],
                preview_image_url=data.get("previewImageUrl"),
            )
            log.info("tunechat: success job_id=%s has_image=%s", result.job_id, result.preview_image_url is not None)
            return result

    except httpx.TimeoutException:
        log.warning("tunechat: timed out after %ds", settings.tunechat_timeout_sec)
        return None
    except httpx.RequestError as exc:
        log.warning("tunechat: request error: %s", exc)
        return None
    except (KeyError, ValueError) as exc:
        log.warning("tunechat: bad response: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001
        log.warning("tunechat: unexpected error: %s", exc)
        return None
