"""Decomposer Celery task — transcription stage stub.

Returns a shape-correct TranscriptionResult using only the shared
contracts package.  When real transcription is wired up, the stub body
will be replaced with actual inference logic that lives inside this
service — no backend import needed.
"""
import os
from pathlib import Path

from decomposer.celery_app import celery_app
from shared.contracts import (
    SCHEMA_VERSION,
    HarmonicAnalysis,
    InputBundle,
    InstrumentRole,
    MidiTrack,
    Note,
    QualitySignal,
    TempoMapEntry,
    TranscriptionResult,
)
from shared.storage.local import LocalBlobStore

_BLOB_ROOT = Path(os.environ.get("OHSHEET_BLOB_ROOT", "./blob"))


def _get_blob_store() -> LocalBlobStore:
    return LocalBlobStore(_BLOB_ROOT)


def _stub_transcription() -> TranscriptionResult:
    """Tiny shape-correct fallback so downstream stages still run."""
    return TranscriptionResult(
        schema_version=SCHEMA_VERSION,
        midi_tracks=[
            MidiTrack(
                notes=[
                    Note(pitch=60, onset_sec=0.0, offset_sec=0.5, velocity=80),
                    Note(pitch=64, onset_sec=0.5, offset_sec=1.0, velocity=80),
                    Note(pitch=67, onset_sec=1.0, offset_sec=1.5, velocity=80),
                    Note(pitch=72, onset_sec=1.5, offset_sec=2.0, velocity=80),
                ],
                instrument=InstrumentRole.MELODY,
                program=None,
                confidence=0.7,
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
            overall_confidence=0.3,
            warnings=["decomposer stub — real transcription not wired yet"],
        ),
    )


@celery_app.task(name="decomposer.run")
def run(job_id: str, payload_uri: str) -> str:
    blob = _get_blob_store()
    raw = blob.get_json(payload_uri)
    InputBundle.model_validate(raw)  # validate input contract

    result = _stub_transcription()

    output_uri = blob.put_json(
        f"jobs/{job_id}/decomposer/output.json",
        result.model_dump(mode="json"),
    )
    return output_uri
