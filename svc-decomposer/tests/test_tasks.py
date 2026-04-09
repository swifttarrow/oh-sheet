"""Unit tests for the decomposer Celery task."""
import sys
from pathlib import Path

import pytest

# Ensure the svc-decomposer package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.contracts import (
    InputBundle,
    InputMetadata,
    RemoteAudioFile,
    TempoMapEntry,
)
from shared.storage.local import LocalBlobStore


@pytest.fixture
def blob(tmp_path):
    root = tmp_path / "blob"
    root.mkdir()
    return LocalBlobStore(root)


def test_decomposer_task_reads_input_writes_output(blob, monkeypatch):
    """Stub delegates to TranscribeService which falls back to stub result."""
    import decomposer.tasks as task_module

    # Patch settings so the task uses our temp blob root
    monkeypatch.setattr(task_module, "_get_blob_store", lambda: blob)

    bundle = InputBundle(
        audio=RemoteAudioFile(
            uri="file:///fake/audio.wav",
            format="wav",
            sample_rate=44100,
            duration_sec=10.0,
            channels=1,
        ),
        metadata=InputMetadata(title="Test", source="audio_upload"),
    )
    payload_uri = blob.put_json(
        "jobs/test-job/decomposer/input.json",
        bundle.model_dump(mode="json"),
    )
    output_uri = task_module.run("test-job", payload_uri)
    result = blob.get_json(output_uri)
    assert "midi_tracks" in result
    assert "analysis" in result
    assert "quality" in result
