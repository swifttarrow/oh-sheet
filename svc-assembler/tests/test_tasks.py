"""Unit tests for the assembler Celery task."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.contracts import (
    SCHEMA_VERSION,
    HarmonicAnalysis,
    InstrumentRole,
    MidiTrack,
    Note,
    QualitySignal,
    TempoMapEntry,
    TranscriptionResult,
)
from shared.storage.local import LocalBlobStore


@pytest.fixture
def blob(tmp_path):
    root = tmp_path / "blob"
    root.mkdir()
    return LocalBlobStore(root)


def test_assembler_task_reads_input_writes_output(blob, monkeypatch):
    import assembler.tasks as task_module

    monkeypatch.setattr(task_module, "_get_blob_store", lambda: blob)

    txr = TranscriptionResult(
        midi_tracks=[
            MidiTrack(
                notes=[
                    Note(pitch=60, onset_sec=0.0, offset_sec=0.5, velocity=80),
                    Note(pitch=62, onset_sec=0.5, offset_sec=1.0, velocity=80),
                    Note(pitch=48, onset_sec=0.0, offset_sec=1.0, velocity=70),
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
        ),
        quality=QualitySignal(overall_confidence=0.9),
    )
    payload_uri = blob.put_json(
        "jobs/test-job/assembler/input.json",
        txr.model_dump(mode="json"),
    )
    output_uri = task_module.run("test-job", payload_uri)
    result = blob.get_json(output_uri)
    assert "right_hand" in result
    assert "left_hand" in result
    assert "metadata" in result
