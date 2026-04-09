"""Assembler Celery task — arrangement stage stub.

Returns a shape-correct PianoScore using only the shared contracts
package.  When real arrangement logic is wired up, the stub body will be
replaced with actual arrangement code that lives inside this service —
no backend import needed.
"""
import os
from pathlib import Path

from assembler.celery_app import celery_app
from shared.contracts import (
    SCHEMA_VERSION,
    PianoScore,
    ScoreMetadata,
    ScoreNote,
    TempoMapEntry,
    TranscriptionResult,
)
from shared.storage.local import LocalBlobStore

_BLOB_ROOT = Path(os.environ.get("OHSHEET_BLOB_ROOT", "./blob"))


def _get_blob_store() -> LocalBlobStore:
    return LocalBlobStore(_BLOB_ROOT)


def _stub_arrangement(txr: TranscriptionResult) -> PianoScore:
    """Tiny shape-correct fallback so downstream stages still run."""
    tempo_map = txr.analysis.tempo_map or [
        TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)
    ]
    return PianoScore(
        schema_version=SCHEMA_VERSION,
        right_hand=[
            ScoreNote(
                id="rh-0001", pitch=60, onset_beat=0.0,
                duration_beat=1.0, velocity=80, voice=1,
            ),
        ],
        left_hand=[
            ScoreNote(
                id="lh-0001", pitch=48, onset_beat=0.0,
                duration_beat=1.0, velocity=70, voice=1,
            ),
        ],
        metadata=ScoreMetadata(
            key=txr.analysis.key,
            time_signature=txr.analysis.time_signature,
            tempo_map=tempo_map,
            difficulty="intermediate",
        ),
    )


@celery_app.task(name="assembler.run")
def run(job_id: str, payload_uri: str) -> str:
    blob = _get_blob_store()
    raw = blob.get_json(payload_uri)
    txr = TranscriptionResult.model_validate(raw)

    result = _stub_arrangement(txr)

    output_uri = blob.put_json(
        f"jobs/{job_id}/assembler/output.json",
        result.model_dump(mode="json"),
    )
    return output_uri
