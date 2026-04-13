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

ArtifactKind = Literal["pdf", "musicxml", "midi", "transcription_midi"]

# kind → (uri attribute on EngravedOutput, media type, downloaded filename suffix)
_KIND_INFO: dict[str, tuple[str, str, str]] = {
    "pdf":                ("pdf_uri",                "application/pdf",                       "sheet.pdf"),
    "musicxml":           ("musicxml_uri",           "application/vnd.recordare.musicxml+xml", "score.musicxml"),
    "midi":               ("humanized_midi_uri",     "audio/midi",                            "humanized.mid"),
    "transcription_midi": ("transcription_midi_uri", "audio/midi",                            "transcription.mid"),
}


@router.get("/artifacts/{job_id}/refine-trace")
def download_refine_trace(
    job_id: str,
    manager: Annotated[JobManager, Depends(get_job_manager)],
    blob: Annotated[LocalBlobStore, Depends(get_blob_store)],
    inline: bool = False,
) -> Response:
    """INT-06: return llm_trace.json for a refined job.

    Uses a convention-derived path (jobs/{job_id}/refine/llm_trace.json)
    that the refine.run Celery task (Plan 02-04) writes on every
    successful run. 404 when the job did not run refine (enable_refine=False)
    or when refine was skipped via INT-03 before the llm_trace.json write
    (e.g., validator all-rejected at service level — though Plan 04 writes
    the trace even on validator rejections, so in practice only
    pre-service failures like envelope deserialization produce a missing
    trace blob).

    Registered BEFORE the catch-all /{kind} route so FastAPI matches
    /refine-trace here rather than handing it to download_artifact (which
    would reject it as an unknown kind).
    """
    record = manager.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if record.status != "succeeded":
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is {record.status}; artifacts unavailable.",
        )

    # CRITICAL (B1 fix): LocalBlobStore.get_bytes REQUIRES a file:// URI.
    # Passing a plain key raises ValueError. Construct the URI via
    # (settings.blob_root / blob_key).as_uri().
    from backend.config import settings  # local import avoids circular deps
    blob_key = f"jobs/{job_id}/refine/llm_trace.json"
    blob_uri = (settings.blob_root / blob_key).as_uri()
    try:
        data = blob.get_bytes(blob_uri)
    except (FileNotFoundError, ValueError):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Job {job_id} has no refine trace. The job did not run the "
                "refine stage (enable_refine=false) or refine was skipped "
                "before the trace was persisted."
            ),
        ) from None

    filename = f"{job_id}-llm_trace.json"
    disposition = "inline" if inline else f'attachment; filename="{filename}"'
    return Response(
        content=data,
        media_type="application/json",
        headers={"Content-Disposition": disposition},
    )


@router.get("/artifacts/{job_id}/lilypond")
def download_lilypond_source(
    job_id: str,
    manager: Annotated[JobManager, Depends(get_job_manager)],
    blob: Annotated[LocalBlobStore, Depends(get_blob_store)],
    inline: bool = False,
) -> Response:
    """INT-07: return the LilyPond .ly source persisted by EngraveService.

    Uses a convention-derived path (jobs/{job_id}/engrave/score.ly) rather
    than a URI on EngravedOutput to avoid a schema bump. 404 semantics fall
    out naturally when the blob is missing: the engrave stage only writes
    this artifact when LilyPond rendering succeeded (MuseScore fallback
    does NOT produce .ly; stub fallback does NOT produce .ly; refine-less
    jobs DO produce .ly if LilyPond rendered successfully — .ly availability
    is orthogonal to refine).

    Registered BEFORE the catch-all /{kind} route so FastAPI matches
    /lilypond here rather than handing it to download_artifact (which
    would reject it as an unknown kind).
    """
    record = manager.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if record.status != "succeeded":
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is {record.status}; artifacts unavailable.",
        )

    # CRITICAL (B1 fix): LocalBlobStore.get_bytes REQUIRES a file:// URI.
    # Passing a plain key raises ValueError. Construct the URI via
    # (settings.blob_root / blob_key).as_uri() — this is the ONLY supported form.
    from backend.config import settings  # local import avoids circular deps
    blob_key = f"jobs/{job_id}/engrave/score.ly"
    blob_uri = (settings.blob_root / blob_key).as_uri()
    try:
        data = blob.get_bytes(blob_uri)
    except (FileNotFoundError, ValueError):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Job {job_id} has no LilyPond source. LilyPond rendering "
                "may have fallen back to MuseScore or the stub renderer."
            ),
        ) from None

    filename = f"{job_id}-score.ly"
    disposition = "inline" if inline else f'attachment; filename="{filename}"'
    return Response(
        content=data,
        media_type="application/x-lilypond",
        headers={"Content-Disposition": disposition},
    )


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

    attr, media_type, suffix = _KIND_INFO[kind]
    uri = getattr(record.result, attr)
    if uri is None:
        # Optional artifacts (e.g. transcription_midi) won't exist for
        # every variant — midi_upload skips transcription, and the stub
        # fallback doesn't persist a MIDI file. Return 404 so clients can
        # distinguish "never produced" from "job failed".
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

    filename = f"{job_id}-{suffix}"
    disposition = "inline" if inline else f'attachment; filename="{filename}"'
    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": disposition},
    )
