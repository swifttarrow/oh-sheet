import time
from unittest.mock import patch

import pytest

from backend.config import settings
from backend.contracts import RemoteAudioFile


def _upload_audio(client):
    return client.post(
        "/v1/uploads/audio",
        files={"file": ("song.wav", b"RIFFfake wav data", "audio/wav")},
    ).json()


@pytest.fixture
def mock_youtube_download():
    """Stub out _download_youtube_sync so YouTube-URL tests don't hit the
    network. Returns a minimal RemoteAudioFile + title/uploader tuple so
    the ingest stage completes normally."""
    with patch("backend.services.ingest._download_youtube_sync") as mock_dl:
        mock_dl.return_value = (
            RemoteAudioFile(
                uri="file:///tmp/fake.wav",
                format="wav",
                sample_rate=44100,
                duration_sec=60.0,
                channels=2,
            ),
            "Mock Song Title",
            "Mock Uploader",
        )
        yield mock_dl


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


# ---------------------------------------------------------------------------
# prefer_clean_source: per-job opt-in to the cover_search fast path
# ---------------------------------------------------------------------------
#
# The upload screen has a "find a clean piano cover" toggle. When the user
# flips it on, the frontend must be able to pass prefer_clean_source=true
# in the POST /v1/jobs body, and that flag must land on the InputBundle's
# metadata so the ingest stage can read it. These tests exercise the route
# wiring; the ingest-side consumption is tested in test_ingest_cover_search.


def test_prefer_clean_source_defaults_to_false_when_omitted(client):
    # Omitting the flag must NOT break the request (backward compat with
    # existing frontends that don't know about this field yet).
    response = client.post("/v1/jobs", json={"title": "Yesterday"})
    assert response.status_code == 202


def test_prefer_clean_source_true_is_accepted_by_create_job(client, mock_youtube_download):
    response = client.post(
        "/v1/jobs",
        json={"title": "https://youtu.be/fJ9rUzIMcZQ", "prefer_clean_source": True},
    )
    assert response.status_code == 202, response.text


def test_prefer_clean_source_is_rejected_on_bad_type(client):
    # Pydantic should reject a non-bool value cleanly — documented contract.
    response = client.post(
        "/v1/jobs",
        json={"title": "Yesterday", "prefer_clean_source": "sometimes"},
    )
    assert response.status_code == 422


def test_prefer_clean_source_lands_on_bundle_metadata(client, mock_youtube_download):
    # The submitted flag must survive the trip into the JobManager's stored
    # InputBundle — otherwise the ingest stage never sees it.
    from backend.api.deps import get_job_manager
    from backend.main import app

    response = client.post(
        "/v1/jobs",
        json={"title": "https://youtu.be/fJ9rUzIMcZQ", "prefer_clean_source": True},
    )
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]

    # Reach into the in-memory JobManager to verify the stored bundle.
    # Same DI singleton the route uses, so the record is guaranteed to exist.
    manager = app.dependency_overrides.get(get_job_manager, get_job_manager)()
    record = manager.get(job_id)
    assert record is not None
    assert record.bundle.metadata.prefer_clean_source is True


def test_prefer_clean_source_false_lands_on_bundle_metadata(client, mock_youtube_download):
    from backend.api.deps import get_job_manager
    from backend.main import app

    response = client.post(
        "/v1/jobs",
        json={"title": "https://youtu.be/fJ9rUzIMcZQ", "prefer_clean_source": False},
    )
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]

    manager = app.dependency_overrides.get(get_job_manager, get_job_manager)()
    record = manager.get(job_id)
    assert record is not None
    assert record.bundle.metadata.prefer_clean_source is False


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
