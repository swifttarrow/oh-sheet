"""Tests for GET /v1/artifacts/{job_id}/{kind}."""
from __future__ import annotations

import time


def _wait_for_succeeded(client, job_id: str, deadline_sec: float = 5.0) -> dict:
    deadline = time.time() + deadline_sec
    status: dict | None = None
    while time.time() < deadline:
        status = client.get(f"/v1/jobs/{job_id}").json()
        if status["status"] in ("succeeded", "failed"):
            return status
        time.sleep(0.05)
    assert status is not None, "job never returned a status"
    return status


def _submit_midi_job(client) -> str:
    midi = client.post(
        "/v1/uploads/midi",
        files={"file": ("a.mid", b"MThd\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00", "audio/midi")},
    ).json()
    create = client.post("/v1/jobs", json={"midi": midi, "title": "Artifact Test"}).json()
    return create["job_id"]


def test_artifact_pdf_not_available(client):
    """The ML engraver returns MusicXML only — PDF rendering is a client
    responsibility now, so the /pdf endpoint 404s for every job."""
    job_id = _submit_midi_job(client)
    status = _wait_for_succeeded(client, job_id)
    assert status["status"] == "succeeded", status

    response = client.get(f"/v1/artifacts/{job_id}/pdf")
    assert response.status_code == 404


def test_artifact_download_musicxml(client):
    job_id = _submit_midi_job(client)
    _wait_for_succeeded(client, job_id)

    response = client.get(f"/v1/artifacts/{job_id}/musicxml")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/vnd.recordare.musicxml+xml"
    assert b"score-partwise" in response.content


def test_artifact_download_midi(client):
    job_id = _submit_midi_job(client)
    _wait_for_succeeded(client, job_id)

    response = client.get(f"/v1/artifacts/{job_id}/midi")
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/midi"
    assert response.content.startswith(b"MThd")


def test_artifact_unknown_kind(client):
    job_id = _submit_midi_job(client)
    _wait_for_succeeded(client, job_id)

    response = client.get(f"/v1/artifacts/{job_id}/wav")
    assert response.status_code == 400
    assert "Unknown artifact kind" in response.json()["detail"]


def test_artifact_unknown_job(client):
    response = client.get("/v1/artifacts/does-not-exist/pdf")
    assert response.status_code == 404


def test_artifact_job_not_yet_succeeded(client):
    """Submit a job and immediately try to download — should be 409 until it finishes.

    We query musicxml because pdf is a permanent 404 now (ML engraver path
    doesn't produce PDFs). If we hit the window between submit and finish,
    we see 409; if we miss it, a successful job returns 200.
    """
    job_id = _submit_midi_job(client)
    response = client.get(f"/v1/artifacts/{job_id}/musicxml")
    assert response.status_code in (200, 409)
    if response.status_code == 409:
        assert "artifacts unavailable" in response.json()["detail"]
