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


# ---------------------------------------------------------------------------
# TuneChat proxy — used for jobs that went through the TuneChat-only fast
# path. Local blob URIs are empty strings in that case; the endpoint must
# fall through to the tunechat_*_url field and stream the file back with
# Content-Disposition so the client's <a download> works reliably.
# ---------------------------------------------------------------------------


def _install_tunechat_job(client, *, pdf: str | None, musicxml: str | None, midi: str | None) -> str:
    """Submit a midi job and patch its EngravedOutput to look like a
    TuneChat-fast-path result: empty blob URIs + populated tunechat URLs.
    Returns the job_id."""
    from backend.api.deps import get_job_manager
    from backend.jobs.manager import JobManager

    job_id = _submit_midi_job(client)
    _wait_for_succeeded(client, job_id)

    # Reach into the in-memory JobManager used by this TestClient app.
    manager: JobManager = client.app.dependency_overrides.get(
        get_job_manager, get_job_manager
    )()
    record = manager.get(job_id)
    assert record is not None and record.result is not None
    record.result = record.result.model_copy(update={
        "musicxml_uri": "",
        "humanized_midi_uri": "",
        "pdf_uri": None,
        "tunechat_job_id": "tc-fake-123",
        "tunechat_pdf_url": pdf,
        "tunechat_musicxml_url": musicxml,
        "tunechat_midi_url": midi,
    })
    return job_id


def test_artifact_proxies_musicxml_from_tunechat(client, monkeypatch):
    """When tunechat_musicxml_url is set, the endpoint proxies the fetch
    and returns the bytes with Content-Disposition: attachment."""
    upstream_body = b"<?xml version='1.0'?><score-partwise/>"

    class FakeResponse:
        status_code = 200
        content = upstream_body
        def raise_for_status(self): pass

    class FakeClient:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, url):
            assert url == "http://localhost:3000/pipeline/tc-fake-123/score.musicxml"
            return FakeResponse()

    monkeypatch.setattr("backend.api.routes.artifacts.httpx.Client", FakeClient)

    job_id = _install_tunechat_job(
        client,
        pdf="http://localhost:3000/pipeline/tc-fake-123/sheet.pdf",
        musicxml="http://localhost:3000/pipeline/tc-fake-123/score.musicxml",
        midi="http://localhost:3000/pipeline/tc-fake-123/notation.mid",
    )
    response = client.get(f"/v1/artifacts/{job_id}/musicxml")
    assert response.status_code == 200
    assert response.content == upstream_body
    assert response.headers["content-type"] == "application/vnd.recordare.musicxml+xml"
    # Critical: the <a download> attribute is unreliable across cross-origin
    # redirects, so we set Content-Disposition server-side to force the
    # download regardless of the user's browser.
    assert "attachment" in response.headers["content-disposition"]
    assert ".musicxml" in response.headers["content-disposition"]


def test_artifact_proxies_pdf_from_tunechat(client, monkeypatch):
    class FakeResponse:
        status_code = 200
        content = b"%PDF-1.4 fake"
        def raise_for_status(self): pass

    class FakeClient:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, url): return FakeResponse()

    monkeypatch.setattr("backend.api.routes.artifacts.httpx.Client", FakeClient)

    job_id = _install_tunechat_job(
        client,
        pdf="http://localhost:3000/pipeline/tc-fake-123/sheet.pdf",
        musicxml=None,
        midi=None,
    )
    response = client.get(f"/v1/artifacts/{job_id}/pdf")
    assert response.status_code == 200
    assert response.content.startswith(b"%PDF")
    assert response.headers["content-type"] == "application/pdf"


def test_artifact_falls_back_to_blob_when_no_tunechat_url(client):
    """Audio/midi-upload paths never set tunechat_*_url — the old blob-
    store code path must still work unchanged."""
    job_id = _submit_midi_job(client)
    _wait_for_succeeded(client, job_id)
    response = client.get(f"/v1/artifacts/{job_id}/musicxml")
    assert response.status_code == 200
    assert b"score-partwise" in response.content


def test_artifact_upstream_failure_returns_502(client, monkeypatch):
    """If the TuneChat proxy fetch fails (network, 5xx, timeout), the
    endpoint returns 502 rather than 500 — distinguishes upstream
    trouble from a missing/corrupt local artifact."""
    import httpx

    class FakeClient:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, url):
            raise httpx.ConnectError("boom")

    monkeypatch.setattr("backend.api.routes.artifacts.httpx.Client", FakeClient)

    job_id = _install_tunechat_job(
        client,
        pdf=None,
        musicxml="http://localhost:3000/pipeline/tc-fake-123/score.musicxml",
        midi=None,
    )
    response = client.get(f"/v1/artifacts/{job_id}/musicxml")
    assert response.status_code == 502


def test_artifact_refuses_proxy_to_foreign_host(client, monkeypatch):
    """SSRF guard: if the tunechat_*_url points somewhere other than the
    configured TuneChat host, refuse with 502 BEFORE firing the request.

    Simulates a TuneChat response that (whether via compromise or
    misconfiguration) returns an internal/external URL we shouldn't be
    proxying. The httpx.Client mock raises on any .get() so the test
    fails loudly if the guard is bypassed.
    """
    class ExplodeClient:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, url):
            raise AssertionError(f"SSRF guard bypassed — fetched {url}")

    monkeypatch.setattr("backend.api.routes.artifacts.httpx.Client", ExplodeClient)

    job_id = _install_tunechat_job(
        client,
        pdf=None,
        # settings.tunechat_url is http://localhost:3000 by default; this
        # URL points at a host NOT in the allowlist (classic SSRF
        # attempt targeting AWS IMDS).
        musicxml="http://169.254.169.254/latest/meta-data/iam/security-credentials/",
        midi=None,
    )
    response = client.get(f"/v1/artifacts/{job_id}/musicxml")
    assert response.status_code == 502
    assert "not allowed" in response.json()["detail"].lower()


def test_artifact_proxy_scheme_mismatch_is_rejected(client, monkeypatch):
    """Scheme mismatch (https target vs http configured) also blocked —
    prevents downgrade/upgrade attacks where an attacker swaps
    http://internal.svc for https://internal.svc expecting only host
    matching."""
    class ExplodeClient:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, url):
            raise AssertionError("SSRF guard bypassed")

    monkeypatch.setattr("backend.api.routes.artifacts.httpx.Client", ExplodeClient)

    job_id = _install_tunechat_job(
        client,
        pdf=None,
        musicxml="https://localhost:3000/pipeline/tc-fake-123/score.musicxml",  # https, not http
        midi=None,
    )
    response = client.get(f"/v1/artifacts/{job_id}/musicxml")
    assert response.status_code == 502
