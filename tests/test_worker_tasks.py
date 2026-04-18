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


class TestRefineTask:
    def test_passes_through_on_llm_failure(self, blob, monkeypatch):
        """The worker wraps the input envelope and calls RefineService,
        which returns the input unchanged when the LLM is unavailable
        (the default for tests — no API key)."""
        from shared.contracts import (
            PianoScore,
            ScoreMetadata,
            ScoreNote,
        )

        from backend.workers.refine import run as refine_run

        score = PianoScore(
            right_hand=[
                ScoreNote(id="rh-1", pitch=60, onset_beat=0.0, duration_beat=1.0, velocity=80, voice=1),
            ],
            left_hand=[
                ScoreNote(id="lh-1", pitch=48, onset_beat=0.0, duration_beat=1.0, velocity=70, voice=1),
            ],
            metadata=ScoreMetadata(
                key="C:major",
                time_signature=(4, 4),
                tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
                difficulty="intermediate",
            ),
        )
        payload_uri = blob.put_json(
            "jobs/test-job/refine/input.json",
            {
                "payload": score.model_dump(mode="json"),
                "payload_type": "PianoScore",
                "title_hint": "test",
                "artist_hint": None,
            },
        )
        output_uri = refine_run("test-job", payload_uri)
        result = blob.get_json(output_uri)
        assert result["payload_type"] == "PianoScore"
        assert result["payload"]["metadata"]["key"] == "C:major"

    def test_humanized_performance_envelope_roundtrip(self, blob):
        from shared.contracts import (
            ExpressionMap,
            HumanizedPerformance,
            PianoScore,
            QualitySignal,
            ScoreMetadata,
            ScoreNote,
        )

        from backend.workers.refine import run as refine_run

        score = PianoScore(
            right_hand=[
                ScoreNote(id="rh-1", pitch=60, onset_beat=0.0, duration_beat=1.0, velocity=80, voice=1),
            ],
            left_hand=[
                ScoreNote(id="lh-1", pitch=48, onset_beat=0.0, duration_beat=1.0, velocity=70, voice=1),
            ],
            metadata=ScoreMetadata(
                key="C:major",
                time_signature=(4, 4),
                tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
                difficulty="intermediate",
            ),
        )
        perf = HumanizedPerformance(
            expressive_notes=[],
            expression=ExpressionMap(),
            score=score,
            quality=QualitySignal(overall_confidence=0.9, warnings=[]),
        )
        payload_uri = blob.put_json(
            "jobs/test-job/refine/input.json",
            {
                "payload": perf.model_dump(mode="json"),
                "payload_type": "HumanizedPerformance",
                "title_hint": None,
                "artist_hint": None,
            },
        )
        output_uri = refine_run("test-job", payload_uri)
        result = blob.get_json(output_uri)
        assert result["payload_type"] == "HumanizedPerformance"
        assert "expressive_notes" in result["payload"]
        assert "score" in result["payload"]
