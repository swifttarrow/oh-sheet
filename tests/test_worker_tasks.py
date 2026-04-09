"""Unit tests for monolith Celery worker tasks.

Each task follows the same pattern: read input from blob, run existing
service, write output to blob, return output URI.
"""

import pytest
from shared.contracts import (
    InputBundle,
    InputMetadata,
    TempoMapEntry,
)
from shared.storage.local import LocalBlobStore

from backend.config import settings


@pytest.fixture
def blob():
    """Return a LocalBlobStore rooted at the isolated_blob_root path."""
    return LocalBlobStore(settings.blob_root)


class TestIngestTask:
    def test_reads_blob_runs_service_writes_output(self, blob):
        from backend.workers.ingest import run as ingest_run

        bundle = InputBundle(
            metadata=InputMetadata(title="Test", source="audio_upload"),
        )
        payload_uri = blob.put_json(
            "jobs/test-job/ingest/input.json",
            bundle.model_dump(mode="json"),
        )
        output_uri = ingest_run("test-job", payload_uri)
        result = blob.get_json(output_uri)
        assert result["metadata"]["title"] == "Test"


class TestHumanizeTask:
    def test_reads_blob_runs_service_writes_output(self, blob):
        from shared.contracts import (
            PianoScore,
            ScoreMetadata,
            ScoreNote,
        )

        from backend.workers.humanize import run as humanize_run

        score = PianoScore(
            right_hand=[
                ScoreNote(id="rh-0001", pitch=60, onset_beat=0.0, duration_beat=1.0, velocity=80, voice=1),
            ],
            left_hand=[
                ScoreNote(id="lh-0001", pitch=48, onset_beat=0.0, duration_beat=1.0, velocity=70, voice=1),
            ],
            metadata=ScoreMetadata(
                key="C:major",
                time_signature=(4, 4),
                tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
                difficulty="intermediate",
            ),
        )
        payload_uri = blob.put_json(
            "jobs/test-job/humanize/input.json",
            score.model_dump(mode="json"),
        )
        output_uri = humanize_run("test-job", payload_uri)
        result = blob.get_json(output_uri)
        assert "expressive_notes" in result
        assert "expression" in result


class TestEngraveTask:
    def test_reads_blob_runs_service_writes_output(self, blob):
        from shared.contracts import (
            PianoScore,
            ScoreMetadata,
            ScoreNote,
        )

        from backend.workers.engrave import run as engrave_run

        score = PianoScore(
            right_hand=[
                ScoreNote(id="rh-0001", pitch=60, onset_beat=0.0, duration_beat=1.0, velocity=80, voice=1),
            ],
            left_hand=[
                ScoreNote(id="lh-0001", pitch=48, onset_beat=0.0, duration_beat=1.0, velocity=70, voice=1),
            ],
            metadata=ScoreMetadata(
                key="C:major",
                time_signature=(4, 4),
                tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
                difficulty="intermediate",
            ),
        )
        payload_uri = blob.put_json(
            "jobs/test-job/engrave/input.json",
            {
                "payload": score.model_dump(mode="json"),
                "payload_type": "PianoScore",
                "job_id": "test-job",
                "title": "Test Song",
                "composer": "Test Artist",
            },
        )
        output_uri = engrave_run("test-job", payload_uri)
        result = blob.get_json(output_uri)
        assert "pdf_uri" in result
        assert "musicxml_uri" in result
