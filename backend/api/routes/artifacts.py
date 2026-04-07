"""GET /v1/artifacts/{job_id}/{kind} — download finished pipeline outputs.

The pipeline contracts return ``file://`` URIs in ``EngravedOutput`` so the
worker layer can stay storage-agnostic. Browsers and mobile apps can't fetch
those URIs directly, so this route resolves the kind → URI for a finished
job, reads the bytes via the BlobStore, and streams them back over HTTP.
"""
from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from backend.api.deps import get_blob_store, get_job_manager
from backend.jobs.manager import JobManager
from backend.storage.local import LocalBlobStore

router = APIRouter()

ArtifactKind = Literal["pdf", "musicxml", "midi"]

# kind → (uri attribute on EngravedOutput, media type, downloaded filename suffix)
_KIND_INFO: dict[str, tuple[str, str, str]] = {
    "pdf":      ("pdf_uri",            "application/pdf",                       "sheet.pdf"),
    "musicxml": ("musicxml_uri",       "application/vnd.recordare.musicxml+xml", "score.musicxml"),
    "midi":     ("humanized_midi_uri", "audio/midi",                            "humanized.mid"),
}


@router.get("/artifacts/{job_id}/{kind}")
def download_artifact(
    job_id: str,
    kind: str,
    manager: Annotated[JobManager, Depends(get_job_manager)],
    blob: Annotated[LocalBlobStore, Depends(get_blob_store)],
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

    attr, media_type, suffix = _KIND_INFO[kind]
    uri = getattr(record.result, attr)

    try:
        data = blob.get_bytes(uri)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read artifact: {exc}",
        ) from exc

    filename = f"{job_id}-{suffix}"
    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
