"""INT-06 smoke tests for GET /v1/artifacts/{job_id}/refine-trace."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from shared.storage.local import LocalBlobStore

from backend.api import deps
from backend.config import settings


def _audio_payload(blob_root: Path) -> dict:
    audio_file = blob_root / "jobs" / "test" / "uploads" / "audio" / "test.mp3"
    audio_file.parent.mkdir(parents=True, exist_ok=True)
    audio_file.write_bytes(b"\x00" * 64)
    return {
        "audio": {
            "uri": f"file://{audio_file}",
            "format": "mp3",
            "sample_rate": 44100,
            "duration_sec": 1.0,
            "channels": 2,
        },
        "title": "Test Song",
        "artist": "Tester",
    }


def test_refine_trace_404_for_unknown_job(client: TestClient) -> None:
    """Unknown job_id -> 404 with 'Job not found' detail."""
    resp = client.get("/v1/artifacts/nonexistent-job-id/refine-trace")
    assert resp.status_code == 404
    assert "Job not found" in resp.json()["detail"]


def test_refine_trace_404_for_non_refined_job(client: TestClient) -> None:
    """Succeeded job with enable_refine=False -> 404 because the trace blob was never written."""
    body = _audio_payload(settings.blob_root)
    body["enable_refine"] = False
    r = client.post("/v1/jobs", json=body)
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    # Wait for completion
    deadline = time.time() + 5.0
    while time.time() < deadline:
        status = client.get(f"/v1/jobs/{job_id}").json()["status"]
        if status in ("succeeded", "failed"):
            break
        time.sleep(0.05)

    resp = client.get(f"/v1/artifacts/{job_id}/refine-trace")
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert "refine" in detail.lower()


def test_refine_trace_200_when_blob_exists(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Blob exists at convention path -> 200 with application/json."""
    # Directly write a synthetic trace blob (avoids orchestrating a real refine run).
    # Create a job record so the 404/409 gates pass.
    from backend.contracts import (
        SCHEMA_VERSION,
        EngravedOutput,
        EngravedScoreData,
        InputBundle,
        InputMetadata,
        PipelineConfig,
        RemoteAudioFile,
    )
    from backend.jobs.manager import JobRecord

    job_id = "refined-job-synthetic"
    blob_root = settings.blob_root
    audio_file = blob_root / "jobs" / "synthetic" / "uploads" / "audio" / "test.mp3"
    audio_file.parent.mkdir(parents=True, exist_ok=True)
    audio_file.write_bytes(b"\x00" * 64)

    manager = deps.get_job_manager()
    manager._jobs[job_id] = JobRecord(
        job_id=job_id,
        status="succeeded",
        config=PipelineConfig(variant="audio_upload", enable_refine=True),
        bundle=InputBundle(
            schema_version=SCHEMA_VERSION,
            audio=RemoteAudioFile(
                uri=f"file://{audio_file}",
                format="mp3",
                sample_rate=44100,
                duration_sec=1.0,
                channels=2,
            ),
            midi=None,
            metadata=InputMetadata(title="x", artist="y", source="audio_upload"),
        ),
        result=EngravedOutput(
            schema_version=SCHEMA_VERSION,
            metadata=EngravedScoreData(
                includes_dynamics=False,
                includes_pedal_marks=False,
                includes_fingering=False,
                includes_chord_symbols=False,
                title="x",
                composer="y",
            ),
            pdf_uri="file://fake.pdf",
            musicxml_uri="file://fake.xml",
            humanized_midi_uri="file://fake.mid",
        ),
    )

    # Write synthetic trace at the convention path.
    blob = LocalBlobStore(settings.blob_root)
    trace_payload = {
        "prompt_version": "2026.04-v1",
        "prompt_system": "...",
        "prompt_user": "...",
        "model": "claude-sonnet-4-6",
        "stop_reason": "end_turn",
        "applied_edits": [],
        "rejected_edits": [],
        "citations": [],
        "usage": {"input_tokens": 100, "output_tokens": 50},
        "estimated_cost_usd": 0.0105,
    }
    blob.put_json(f"jobs/{job_id}/refine/llm_trace.json", trace_payload)

    resp = client.get(f"/v1/artifacts/{job_id}/refine-trace")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/json"
    returned = json.loads(resp.content)
    assert returned["prompt_version"] == "2026.04-v1"
    # Content-Disposition default: attachment
    assert "attachment" in resp.headers.get("content-disposition", "")

    # Cleanup for next test
    del manager._jobs[job_id]


def test_refine_trace_inline_query_param(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """?inline=true -> Content-Disposition: inline."""
    from backend.contracts import (
        SCHEMA_VERSION,
        EngravedOutput,
        EngravedScoreData,
        InputBundle,
        InputMetadata,
        PipelineConfig,
        RemoteAudioFile,
    )
    from backend.jobs.manager import JobRecord

    job_id = "refined-inline-synthetic"
    blob_root = settings.blob_root
    audio_file = blob_root / "jobs" / "inline" / "uploads" / "audio" / "test.mp3"
    audio_file.parent.mkdir(parents=True, exist_ok=True)
    audio_file.write_bytes(b"\x00" * 64)

    manager = deps.get_job_manager()
    manager._jobs[job_id] = JobRecord(
        job_id=job_id,
        status="succeeded",
        config=PipelineConfig(variant="audio_upload", enable_refine=True),
        bundle=InputBundle(
            schema_version=SCHEMA_VERSION,
            audio=RemoteAudioFile(
                uri=f"file://{audio_file}",
                format="mp3",
                sample_rate=44100,
                duration_sec=1.0,
                channels=2,
            ),
            midi=None,
            metadata=InputMetadata(title="x", artist="y", source="audio_upload"),
        ),
        result=EngravedOutput(
            schema_version=SCHEMA_VERSION,
            metadata=EngravedScoreData(
                includes_dynamics=False,
                includes_pedal_marks=False,
                includes_fingering=False,
                includes_chord_symbols=False,
                title="x",
                composer="y",
            ),
            pdf_uri="file://x",
            musicxml_uri="file://x",
            humanized_midi_uri="file://x",
        ),
    )

    blob = LocalBlobStore(settings.blob_root)
    blob.put_json(
        f"jobs/{job_id}/refine/llm_trace.json",
        {"prompt_version": "2026.04-v1"},
    )

    resp = client.get(f"/v1/artifacts/{job_id}/refine-trace?inline=true")
    assert resp.status_code == 200
    assert "inline" in resp.headers.get("content-disposition", "")

    del manager._jobs[job_id]


def test_refine_trace_409_when_job_running(client: TestClient) -> None:
    """Job exists but status != succeeded -> 409."""
    from backend.contracts import (
        SCHEMA_VERSION,
        InputBundle,
        InputMetadata,
        PipelineConfig,
        RemoteAudioFile,
    )
    from backend.jobs.manager import JobRecord

    blob_root = settings.blob_root
    audio_file = blob_root / "jobs" / "running" / "uploads" / "audio" / "test.mp3"
    audio_file.parent.mkdir(parents=True, exist_ok=True)
    audio_file.write_bytes(b"\x00" * 64)

    manager = deps.get_job_manager()
    manager._jobs["running-job"] = JobRecord(
        job_id="running-job",
        status="running",
        config=PipelineConfig(variant="audio_upload", enable_refine=True),
        bundle=InputBundle(
            schema_version=SCHEMA_VERSION,
            audio=RemoteAudioFile(
                uri=f"file://{audio_file}",
                format="mp3",
                sample_rate=44100,
                duration_sec=1.0,
                channels=2,
            ),
            midi=None,
            metadata=InputMetadata(title="x", artist="y", source="audio_upload"),
        ),
    )
    resp = client.get("/v1/artifacts/running-job/refine-trace")
    assert resp.status_code == 409
    assert "running" in resp.json()["detail"]
    del manager._jobs["running-job"]
