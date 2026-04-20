"""GET /v1/artifacts/{job_id}/{kind} — download finished pipeline outputs.

The pipeline contracts return ``file://`` URIs in ``EngravedOutput`` so the
worker layer can stay storage-agnostic. Browsers and mobile apps can't fetch
those URIs directly, so this route resolves the kind → URI for a finished
job, reads the bytes via the BlobStore, and streams them back over HTTP.

For the TuneChat-fast-path (title_lookup jobs routed through the external
TuneChat engraver), artifacts live on TuneChat's server rather than in our
blob store — ``EngravedOutput.tunechat_*_url`` holds the full https URL.
This endpoint proxies those downloads so the client-facing URL shape
(``/v1/artifacts/{job}/{kind}``) stays consistent regardless of which
engraver produced the bytes.
"""
from __future__ import annotations

import logging
from typing import Annotated, Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from backend.api.deps import get_blob_store, get_job_manager
from backend.jobs.manager import JobManager
from backend.storage.local import LocalBlobStore

log = logging.getLogger(__name__)

router = APIRouter()

ArtifactKind = Literal["pdf", "musicxml", "midi", "transcription_midi"]

# kind → (local-URI attribute, TuneChat-URL attribute, media type, filename suffix)
# The TuneChat URL field is checked FIRST — when populated, we proxy the
# download from TuneChat's server. Falls back to the local blob-store URI
# when the TuneChat field is empty (e.g. audio_upload / midi_upload paths
# that never touch TuneChat). ``transcription_midi`` has no TuneChat
# counterpart — it's the pre-engrave MIDI from Oh Sheet's own pipeline.
_MUSICXML_MIME = "application/vnd.recordare.musicxml+xml"
_KIND_INFO: dict[str, tuple[str, str | None, str, str]] = {
    "pdf":                ("pdf_uri",                "tunechat_pdf_url",      "application/pdf", "sheet.pdf"),
    "musicxml":           ("musicxml_uri",           "tunechat_musicxml_url", _MUSICXML_MIME,    "score.musicxml"),
    "midi":               ("humanized_midi_uri",     "tunechat_midi_url",     "audio/midi",      "humanized.mid"),
    "transcription_midi": ("transcription_midi_uri", None,                    "audio/midi",      "transcription.mid"),
}

# httpx timeout for the TuneChat proxy fetch. Short because the file
# already exists on TuneChat's disk — this is just a LAN-ish transfer.
_PROXY_TIMEOUT_SEC = 30.0


@router.get("/artifacts/{job_id}/{kind}")
def download_artifact(
    job_id: str,
    kind: str,
    manager: Annotated[JobManager, Depends(get_job_manager)],
    blob: Annotated[LocalBlobStore, Depends(get_blob_store)],
    inline: bool = False,
) -> Response:
    if kind not in _KIND_INFO:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown artifact kind: {kind!r}. Allowed: {sorted(_KIND_INFO)}",
        )

    record = manager.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if record.status != "succeeded" or record.result is None:
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is {record.status}; artifacts unavailable.",
        )

    local_attr, tunechat_attr, media_type, suffix = _KIND_INFO[kind]
    filename = f"{job_id}-{suffix}"
    disposition = "inline" if inline else f'attachment; filename="{filename}"'

    # TuneChat-first: when the job was engraved by TuneChat the local
    # blob URI is empty — fall through to the hosted URL.
    tunechat_url = getattr(record.result, tunechat_attr) if tunechat_attr else None
    if tunechat_url:
        try:
            with httpx.Client(timeout=_PROXY_TIMEOUT_SEC, follow_redirects=True) as client:
                response = client.get(tunechat_url)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning(
                "artifacts: TuneChat proxy fetch failed for %s/%s (%s): %s",
                job_id, kind, tunechat_url, exc,
            )
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch {kind!r} artifact from upstream.",
            ) from exc
        return Response(
            content=response.content,
            media_type=media_type,
            headers={"Content-Disposition": disposition},
        )

    # Local-blob-store path (audio_upload / midi_upload pipelines).
    uri = getattr(record.result, local_attr)
    if not uri:
        # Optional artifacts won't exist for every variant:
        #   * transcription_midi — absent on midi_upload / stub paths
        #   * pdf — the ML engraver returns MusicXML only; rendering is
        #     a client-side responsibility
        # Return 404 so clients can distinguish "never produced" from
        # "job failed".
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} has no {kind!r} artifact.",
        )

    try:
        data = blob.get_bytes(uri)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read artifact: {exc}",
        ) from exc

    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": disposition},
    )
