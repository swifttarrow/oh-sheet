"""Tests for /v1/stages/* worker endpoints (OrchestratorCommand envelope)."""
from __future__ import annotations

import json

from backend.api.deps import get_blob_store
from backend.contracts import SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blob(client):
    """Return the blob store wired into the running app."""
    return get_blob_store()


def _cmd(blob, payload_dict: dict, *, job_id: str = "test-job") -> dict:
    """Seed *payload_dict* into the blob store and return an OrchestratorCommand dict."""
    uri = blob.put_json(f"jobs/{job_id}/input", payload_dict)
    return {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "step_id": "step-1",
        "payload_uri": uri,
        "timeout_sec": 30,
    }


def _input_bundle() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "audio": None,
        "midi": None,
        "metadata": {"title": "Test", "artist": "QA", "source": "title_lookup"},
    }


def _transcription_result() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "midi_tracks": [
            {
                "notes": [
                    {"pitch": 60, "onset_sec": 0.0, "offset_sec": 0.5, "velocity": 80},
                    {"pitch": 64, "onset_sec": 0.5, "offset_sec": 1.0, "velocity": 75},
                ],
                "instrument": "melody",
                "program": 0,
                "confidence": 0.9,
            }
        ],
        "analysis": {
            "key": "C:major",
            "time_signature": [4, 4],
            "tempo_map": [{"time_sec": 0.0, "beat": 0.0, "bpm": 120.0}],
            "chords": [],
            "sections": [],
        },
        "quality": {"overall_confidence": 0.8, "warnings": []},
        "transcription_midi_uri": None,
    }


def _piano_score() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "right_hand": [
            {
                "id": "rh-0000",
                "pitch": 60,
                "onset_beat": 0.0,
                "duration_beat": 1.0,
                "velocity": 80,
                "voice": 1,
            },
        ],
        "left_hand": [
            {
                "id": "lh-0000",
                "pitch": 48,
                "onset_beat": 0.0,
                "duration_beat": 1.0,
                "velocity": 70,
                "voice": 1,
            },
        ],
        "metadata": {
            "key": "C:major",
            "time_signature": [4, 4],
            "tempo_map": [{"time_sec": 0.0, "beat": 0.0, "bpm": 120.0}],
            "difficulty": "intermediate",
            "sections": [],
            "chord_symbols": [],
        },
    }


def _humanized_performance() -> dict:
    score = _piano_score()
    return {
        "schema_version": SCHEMA_VERSION,
        "expressive_notes": [
            {
                "score_note_id": "rh-0000",
                "pitch": 60,
                "onset_beat": 0.0,
                "duration_beat": 1.0,
                "velocity": 80,
                "hand": "rh",
                "voice": 1,
                "timing_offset_ms": 0.0,
                "velocity_offset": 0,
            },
            {
                "score_note_id": "lh-0000",
                "pitch": 48,
                "onset_beat": 0.0,
                "duration_beat": 1.0,
                "velocity": 70,
                "hand": "lh",
                "voice": 1,
                "timing_offset_ms": 0.0,
                "velocity_offset": 0,
            },
        ],
        "expression": {
            "dynamics": [],
            "articulations": [],
            "pedal_events": [],
            "tempo_changes": [],
        },
        "score": score,
        "quality": {"overall_confidence": 0.7, "warnings": []},
    }


# ---------------------------------------------------------------------------
# Happy-path tests — one per stage
# ---------------------------------------------------------------------------

def test_stage_ingest_success(client):
    blob = _blob(client)
    cmd = _cmd(blob, _input_bundle())

    resp = client.post("/v1/stages/ingest", json=cmd)

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["job_id"] == "test-job"
    assert body["schema_version"] == SCHEMA_VERSION
    assert body["output_uri"] is not None
    # Verify output is valid JSON in the blob store
    output = blob.get_json(body["output_uri"])
    assert output["schema_version"] == SCHEMA_VERSION


def test_stage_transcribe_success(client):
    blob = _blob(client)
    cmd = _cmd(blob, _input_bundle())

    resp = client.post("/v1/stages/transcribe", json=cmd)

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["output_uri"] is not None
    output = blob.get_json(body["output_uri"])
    assert output["schema_version"] == SCHEMA_VERSION
    assert "midi_tracks" in output


def test_stage_arrange_success(client):
    blob = _blob(client)
    cmd = _cmd(blob, _transcription_result())

    resp = client.post("/v1/stages/arrange", json=cmd)

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["output_uri"] is not None
    output = blob.get_json(body["output_uri"])
    assert output["schema_version"] == SCHEMA_VERSION
    assert "right_hand" in output
    assert "left_hand" in output


def test_stage_humanize_success(client):
    blob = _blob(client)
    cmd = _cmd(blob, _piano_score())

    resp = client.post("/v1/stages/humanize", json=cmd)

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["output_uri"] is not None
    output = blob.get_json(body["output_uri"])
    assert output["schema_version"] == SCHEMA_VERSION
    assert "expressive_notes" in output


def test_stage_engrave_success(client):
    blob = _blob(client)
    cmd = _cmd(blob, _humanized_performance())

    resp = client.post("/v1/stages/engrave", json=cmd)

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["output_uri"] is not None
    output = blob.get_json(body["output_uri"])
    assert output["schema_version"] == SCHEMA_VERSION
    assert "pdf_uri" in output
    assert "musicxml_uri" in output
    assert "humanized_midi_uri" in output


# ---------------------------------------------------------------------------
# Schema version mismatch → 409
# ---------------------------------------------------------------------------

def test_stage_rejects_schema_version_mismatch(client):
    blob = _blob(client)
    uri = blob.put_json("jobs/mismatch/input", _input_bundle())
    cmd = {
        "schema_version": "0.0.0-wrong",
        "job_id": "mismatch",
        "step_id": "step-1",
        "payload_uri": uri,
        "timeout_sec": 30,
    }

    resp = client.post("/v1/stages/ingest", json=cmd)

    assert resp.status_code == 409
    assert "schema_version mismatch" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Error handling — bad payload → fatal_error response
# ---------------------------------------------------------------------------

def test_stage_returns_fatal_error_on_bad_payload(client):
    blob = _blob(client)
    # Store invalid JSON that won't parse as InputBundle
    uri = blob.put_json("jobs/bad/input", {"not": "an InputBundle"})
    cmd = {
        "schema_version": SCHEMA_VERSION,
        "job_id": "bad",
        "step_id": "step-1",
        "payload_uri": uri,
        "timeout_sec": 30,
    }

    resp = client.post("/v1/stages/ingest", json=cmd)

    assert resp.status_code == 200  # envelope always 200
    body = resp.json()
    assert body["status"] == "fatal_error"
    assert body["output_uri"] is None
    assert body["logs"] is not None


def test_stage_returns_fatal_error_on_missing_payload(client):
    cmd = {
        "schema_version": SCHEMA_VERSION,
        "job_id": "missing",
        "step_id": "step-1",
        "payload_uri": "file:///nonexistent/path",
        "timeout_sec": 30,
    }

    resp = client.post("/v1/stages/ingest", json=cmd)

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "fatal_error"
    assert body["logs"] is not None
