import time

from backend.config import settings


def _upload_audio(client):
    return client.post(
        "/v1/uploads/audio",
        files={"file": ("song.wav", b"RIFFfake wav data", "audio/wav")},
    ).json()


def test_create_job_from_audio_runs_to_completion(client):
    audio = _upload_audio(client)

    create = client.post(
        "/v1/jobs",
        json={"audio": audio, "title": "Test Song", "artist": "QA"},
    )
    assert create.status_code == 202, create.text
    job = create.json()
    assert job["job_id"]
    assert job["variant"] == "audio_upload"
    job_id = job["job_id"]

    # Stub services finish in well under a second; poll briefly.
    deadline = time.time() + 5
    status = None
    while time.time() < deadline:
        status = client.get(f"/v1/jobs/{job_id}").json()
        if status["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.05)

    assert status is not None
    assert status["status"] == "succeeded", status
    assert status["result"] is not None
    assert status["result"]["pdf_uri"].startswith("file://")
    assert status["result"]["musicxml_uri"]
    assert status["result"]["humanized_midi_uri"]


def test_create_job_rejects_no_source(client):
    response = client.post("/v1/jobs", json={})
    assert response.status_code == 400


def test_create_job_rejects_both_audio_and_midi(client):
    audio = _upload_audio(client)
    midi = client.post(
        "/v1/uploads/midi",
        files={"file": ("a.mid", b"MThd\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00", "audio/midi")},
    ).json()
    response = client.post("/v1/jobs", json={"audio": audio, "midi": midi})
    assert response.status_code == 400


def test_create_job_title_lookup_only(client):
    response = client.post("/v1/jobs", json={"title": "Yesterday", "artist": "The Beatles"})
    assert response.status_code == 202
    job = response.json()
    assert job["variant"] == "full"


def test_get_job_returns_404_for_unknown(client):
    response = client.get("/v1/jobs/does-not-exist")
    assert response.status_code == 404


def test_websocket_streams_events_to_completion(monkeypatch, client):
    # Default arrange path; local .env may set condense_transform for manual QA.
    monkeypatch.setattr(settings, "score_pipeline", "arrange")

    midi = client.post(
        "/v1/uploads/midi",
        files={"file": ("a.mid", b"MThd\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00", "audio/midi")},
    ).json()
    create = client.post("/v1/jobs", json={"midi": midi, "title": "WS"}).json()
    job_id = create["job_id"]

    with client.websocket_connect(f"/v1/jobs/{job_id}/ws") as ws:
        events = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] in ("job_succeeded", "job_failed"):
                break

    types = [e["type"] for e in events]
    assert "job_created" in types
    assert "job_started" in types
    assert "job_succeeded" in types
    assert any(e["type"] == "stage_completed" for e in events)
    # midi_upload variant skips transcription, so the plan is ingest→arrange→humanize→engrave
    completed_stages = [e["stage"] for e in events if e["type"] == "stage_completed"]
    assert "ingest" in completed_stages
    assert "arrange" in completed_stages
    assert "humanize" in completed_stages
    assert "engrave" in completed_stages
    assert "transcribe" not in completed_stages


def test_midi_job_condense_pipeline_emits_condense_and_transform(monkeypatch, client):
    monkeypatch.setattr(settings, "score_pipeline", "condense_transform")

    midi = client.post(
        "/v1/uploads/midi",
        files={"file": ("a.mid", b"MThd\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00", "audio/midi")},
    ).json()
    create = client.post("/v1/jobs", json={"midi": midi, "title": "Condense path"}).json()
    job_id = create["job_id"]

    deadline = time.time() + 5
    status = None
    while time.time() < deadline:
        status = client.get(f"/v1/jobs/{job_id}").json()
        if status["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.05)

    assert status is not None
    assert status["status"] == "succeeded", status

    with client.websocket_connect(f"/v1/jobs/{job_id}/ws") as ws:
        events = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] in ("job_succeeded", "job_failed"):
                break

    completed = [e["stage"] for e in events if e["type"] == "stage_completed"]
    assert "condense" in completed
    assert "transform" in completed
    assert "arrange" not in completed
