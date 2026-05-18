import time
from unittest.mock import patch

import pytest

from backend.config import settings
from backend.contracts import RemoteAudioFile


@pytest.fixture(autouse=True)
def _title_lookup_needs_tunechat(monkeypatch):
    """Infrastructure fixture — do not rely on its name in test code.

    ``POST /v1/jobs`` rejects title_lookup submissions when
    ``tunechat_enabled`` is False (that path can't produce a score
    without a fallback engraver). Several tests in this file exercise
    title-only job creation, cover_search flags, and YouTube URL flows
    that all land on the title_lookup code path. Flipping the toggle
    on by default lets those assertions run; tests that specifically
    want to exercise the rejection path override locally with
    ``monkeypatch.setattr(settings, "tunechat_enabled", False)``.

    CAUTION: adding a non-title-lookup test to this file inherits the
    enabled default silently. If you're asserting behavior that
    depends on tunechat being off, opt out explicitly.
    """
    monkeypatch.setattr(settings, "tunechat_enabled", True)


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


@pytest.fixture
def mock_tunechat_success():
    """Stub TuneChat client to return a fake successful TranscribeResult.

    Without this, the runner attempts a real httpx call against
    ``http://localhost:3000`` (the dev default), errors out, and falls
    through to the local pipeline — which produces NO tunechat_* URLs
    and so isn't cacheable. Tests that exercise the cache hit path
    must populate those URLs.
    """
    from backend.services.tunechat_client import TuneChatResult

    fake = TuneChatResult(
        job_id="tc-fake-job",
        preview_image_url="https://tunechat.example/p.png",
        midi_url="https://tunechat.example/notation.mid",
        musicxml_url="https://tunechat.example/notation.musicxml",
        pdf_url="https://tunechat.example/notation.pdf",
    )

    async def _fake(*args, **kwargs):  # noqa: ARG001
        return fake

    with patch("backend.services.tunechat_client.transcribe_via_tunechat", new=_fake):
        # Patch the symbol the runner actually resolved — it imports
        # ``transcribe_via_tunechat`` inside _execute(), so we have to
        # patch at the source module, which the runner re-imports each
        # call. Single patch is sufficient.
        yield fake


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
    # pdf_uri is intentionally empty — the ML engraver returns MusicXML only.
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


def test_create_job_rejects_audio_when_youtube_only_mode(client, monkeypatch):
    """When ``OHSHEET_YOUTUBE_ONLY_MODE`` is on (Railway demo deployment),
    audio uploads are refused at the API layer even though the underlying
    pipeline still works in tests with the flag off. This is the
    operator-visible kill switch — flip the env var to bring uploads back.

    See `coaching-mode.md` rationale: env-gated feature flag chosen over
    permanent deletion so the code path stays exercised in CI and reversal
    is one env var, not a re-architecture.
    """
    monkeypatch.setattr(settings, "youtube_only_mode", True)
    audio = _upload_audio(client)
    response = client.post(
        "/v1/jobs",
        json={"audio": audio, "title": "Test Song", "artist": "QA"},
    )
    assert response.status_code == 400
    # User-facing copy: friendly tone + actionable next step. If you
    # change the wording, also update frontend-v2/src/views.js error
    # handling — the SPA surfaces this string verbatim today.
    detail = response.json()["detail"].lower()
    assert "youtube" in detail


def test_create_job_rejects_midi_when_youtube_only_mode(client, monkeypatch):
    monkeypatch.setattr(settings, "youtube_only_mode", True)
    midi = client.post(
        "/v1/uploads/midi",
        files={"file": ("a.mid", b"MThd\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00", "audio/midi")},
    ).json()
    response = client.post("/v1/jobs", json={"midi": midi})
    assert response.status_code == 400


def test_create_job_allows_title_lookup_when_youtube_only_mode(
    client, monkeypatch, mock_youtube_download
):
    """The whole point of YouTube-only mode is to keep title_lookup working.
    Belt-and-suspenders test against a regression where the guard
    accidentally rejects the path it's meant to allow."""
    monkeypatch.setattr(settings, "youtube_only_mode", True)
    response = client.post(
        "/v1/jobs",
        json={"title": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
    )
    assert response.status_code == 202, response.text


def test_resubmitting_youtube_url_returns_cached_job(
    client, monkeypatch, mock_tunechat_success
):
    """End-to-end cache behavior: a second POST for the same YouTube
    URL returns the same job_id immediately, without re-running yt-dlp.

    We bypass ``mock_youtube_download`` here because it points at
    ``/tmp/fake.wav`` (outside the test blob root) — that's fine for
    most tests, but the TuneChat-fast-path inside the runner reads the
    audio bytes from the blob store, which fails for that URI and
    causes the runner to fall through to the local pipeline (which
    produces no tunechat_* URLs, and therefore nothing cacheable).

    Instead we upload a real audio blob via /v1/uploads/audio and
    point the yt-dlp stub at that URI so the runner's TuneChat block
    can read the bytes and the cache write fires on success.
    """
    audio = _upload_audio(client)
    fake_remote = RemoteAudioFile(
        uri=audio["uri"],
        format=audio["format"],
        sample_rate=audio["sample_rate"],
        duration_sec=audio["duration_sec"],
        channels=audio["channels"],
    )

    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    with patch("backend.services.ingest._download_youtube_sync") as mock_dl:
        mock_dl.return_value = (fake_remote, "Mock Song Title", "Mock Uploader")

        first = client.post("/v1/jobs", json={"title": url})
        assert first.status_code == 202, first.text
        first_job_id = first.json()["job_id"]

        # Poll until succeeded — cache write happens on success.
        deadline = time.time() + 5
        while time.time() < deadline:
            status = client.get(f"/v1/jobs/{first_job_id}").json()
            if status["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.05)
        assert status["status"] == "succeeded", status

        mock_dl.reset_mock()

        second = client.post("/v1/jobs", json={"title": url})
        assert second.status_code == 202, second.text
        assert second.json()["job_id"] == first_job_id, (
            "Cache hit must return the original job_id, not a new one"
        )
        assert mock_dl.call_count == 0, (
            "Cache hit must NOT re-trigger yt-dlp download"
        )


def test_resubmitting_after_cache_disabled_runs_pipeline_again(
    client, monkeypatch, mock_youtube_download
):
    """Sanity: with the cache disabled (or unreachable), duplicate
    submits create distinct jobs. Belt-and-suspenders against a
    regression where the cache is accidentally always-on.

    Both jobs are polled to completion before the test returns —
    leaving asyncio pipeline tasks running past teardown causes the
    ``celery_eager_mode`` autouse fixture to flip off mid-task, which
    surfaces as "Never call result.get() within a task!" in the NEXT
    test (cross-test pollution).
    """
    monkeypatch.setattr(settings, "youtube_cache_enabled", False)
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    first = client.post("/v1/jobs", json={"title": url})
    second = client.post("/v1/jobs", json={"title": url})
    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["job_id"] != second.json()["job_id"]

    # Drain both pipelines so the asyncio tasks finish before teardown.
    for job_id in (first.json()["job_id"], second.json()["job_id"]):
        deadline = time.time() + 5
        while time.time() < deadline:
            if client.get(f"/v1/jobs/{job_id}").json()["status"] in (
                "succeeded",
                "failed",
            ):
                break
            time.sleep(0.05)


def test_create_job_rejects_audio_with_nonexistent_uri(client):
    # Integrity: clients must not be able to forge a RemoteAudioFile
    # pointing at a blob URI that was never uploaded. Without this
    # check the route accepted the request, pushed it through stub
    # stages, and reported a successful job — the user only saw the
    # problem when they tried to play the resulting "audio" (which
    # was nothing at all).
    bogus_audio = {
        "uri": "file:///tmp/this-blob-was-never-uploaded.wav",
        "format": "wav",
        "sample_rate": 44100,
        "duration_sec": 1.0,
        "channels": 2,
        "content_hash": "0" * 64,
    }
    response = client.post(
        "/v1/jobs",
        json={"audio": bogus_audio, "title": "Forged", "artist": "Ghost"},
    )
    assert response.status_code == 400
    assert "uri" in response.json()["detail"].lower()


def test_create_job_rejects_midi_with_nonexistent_uri(client):
    bogus_midi = {
        "uri": "file:///tmp/never-uploaded.mid",
        "ticks_per_beat": 480,
        "content_hash": "0" * 64,
    }
    response = client.post(
        "/v1/jobs",
        json={"midi": bogus_midi, "title": "Forged", "artist": "Ghost"},
    )
    assert response.status_code == 400
    assert "uri" in response.json()["detail"].lower()


def test_create_job_title_lookup_only(client):
    response = client.post("/v1/jobs", json={"title": "Yesterday", "artist": "The Beatles"})
    assert response.status_code == 202
    job = response.json()
    assert job["variant"] == "full"


def test_create_job_title_lookup_rejected_when_tunechat_disabled(client, monkeypatch):
    """title_lookup requires TuneChat — when disabled, fail fast at the
    route boundary instead of burning ingest/transcribe/arrange/humanize
    to hit the hard-fail guard in runner.py."""
    monkeypatch.setattr(settings, "tunechat_enabled", False)
    response = client.post("/v1/jobs", json={"title": "Yesterday"})
    assert response.status_code == 400
    assert "tunechat" in response.json()["detail"].lower()


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
    # Default arrange path; local .env may set condense_only for manual QA.
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
    monkeypatch.setattr(settings, "score_pipeline", "condense_only")

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
    assert "arrange" not in completed
    assert "transform" not in completed
