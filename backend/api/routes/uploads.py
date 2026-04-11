"""POST /v1/uploads/{audio,midi} — Claim-Check pattern.

Clients upload bytes; we hash them, write to blob storage, and return a
Remote*File reference that can be passed to /v1/jobs. The real ingestion
service should probe the audio for sample_rate / duration / channels and
parse the MIDI for ticks_per_beat — for now we return placeholder metadata
and let the ingest stub fill in the rest.
"""
from __future__ import annotations

import hashlib
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from backend.api.deps import get_blob_store
from backend.contracts import RemoteAudioFile, RemoteMidiFile
from backend.storage.local import LocalBlobStore

router = APIRouter()

AUDIO_FORMATS = {"mp3", "wav", "flac", "m4a"}
MIDI_FORMATS = {"mid", "midi"}

# Standard MIDI File magic header. Every SMF (regardless of format 0/1/2)
# begins with the 4-byte "MThd" chunk type. Checking these four bytes
# is a cheap content-integrity gate — it blocks the "rename anything
# to .mid and upload" attack without pulling in a MIDI parser at the
# HTTP layer. Deeper structural validation is ingest's job.
_MIDI_MAGIC = b"MThd"


def _ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


@router.post("/uploads/audio", response_model=RemoteAudioFile)
async def upload_audio(
    file: Annotated[UploadFile, File()],
    blob: Annotated[LocalBlobStore, Depends(get_blob_store)],
) -> RemoteAudioFile:
    ext = _ext(file.filename or "")
    if ext not in AUDIO_FORMATS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio format: {ext!r}. Allowed: {sorted(AUDIO_FORMATS)}",
        )

    data = await file.read()
    digest = hashlib.sha256(data).hexdigest()
    uri = blob.put_bytes(f"uploads/audio/{digest}.{ext}", data)

    # Real ingestion would probe these via librosa/soundfile; placeholders for now.
    return RemoteAudioFile(
        uri=uri,
        format=ext,  # type: ignore[arg-type]
        sample_rate=44100,
        duration_sec=0.0,
        channels=2,
        content_hash=digest,
    )


@router.post("/uploads/midi", response_model=RemoteMidiFile)
async def upload_midi(
    file: Annotated[UploadFile, File()],
    blob: Annotated[LocalBlobStore, Depends(get_blob_store)],
) -> RemoteMidiFile:
    ext = _ext(file.filename or "")
    if ext not in MIDI_FORMATS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported MIDI format: {ext!r}. Allowed: {sorted(MIDI_FORMATS)}",
        )

    data = await file.read()

    # Content-integrity check: the .mid/.midi extension is a hint, not
    # a guarantee. Reject anything that doesn't begin with the Standard
    # MIDI File magic header so garbage bytes can't land in blob
    # storage and get passed to the ingest stage.
    if not data.startswith(_MIDI_MAGIC):
        raise HTTPException(
            status_code=415,
            detail=(
                "Uploaded bytes are not a valid MIDI file "
                "(missing 'MThd' header). Please upload a Standard MIDI File."
            ),
        )

    digest = hashlib.sha256(data).hexdigest()
    uri = blob.put_bytes(f"uploads/midi/{digest}.mid", data)

    return RemoteMidiFile(uri=uri, ticks_per_beat=480, content_hash=digest)
