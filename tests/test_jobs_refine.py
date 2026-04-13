"""V9-V13 HTTP-boundary tests for CFG-02, CFG-04, CFG-06.

Covers:
  - V9: POST /v1/jobs with enable_refine=true + no key -> 400
  - V10: POST with enable_refine=false + key set -> 202, plan has no refine
  - V11: POST with enable_refine=true + key set -> 202, plan includes refine
  - V12: kill switch ON + enable_refine=true -> 202, plan identical to enable_refine=false
  - V13: kill switch coercion emits single log.warning per job
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from backend.api import deps
from backend.config import settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _audio_payload(blob_root: Path) -> dict:
    """Create a tiny audio blob at a known URI and return a JobCreateRequest dict."""
    audio_file = blob_root / "jobs" / "test" / "uploads" / "audio" / "test.mp3"
    audio_file.parent.mkdir(parents=True, exist_ok=True)
    audio_file.write_bytes(b"\x00" * 64)  # nonzero bytes - exists() is all we need
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


def _get_job_record(job_id: str):
    """Fetch the in-memory JobRecord so we can inspect the PipelineConfig."""
    manager = deps.get_job_manager()
    return manager.get(job_id)


# ---------------------------------------------------------------------------
# V9 - CFG-02 / T-1-03: 400 when enable_refine=true without API key
# ---------------------------------------------------------------------------


def test_create_job_400_when_enable_refine_true_and_key_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CFG-04: Submitting with enable_refine=true + no key -> 400 with clear message."""
    monkeypatch.setattr(settings, "anthropic_api_key", None)
    monkeypatch.setattr(settings, "refine_kill_switch", False)

    body = _audio_payload(settings.blob_root)
    body["enable_refine"] = True

    response = client.post("/v1/jobs", json=body)

    assert response.status_code == 400, response.text
    detail = response.json().get("detail", "")
    assert "OHSHEET_ANTHROPIC_API_KEY" in detail, detail
    assert "enable_refine" in detail, detail


# ---------------------------------------------------------------------------
# V10 - CFG-02: 202 with enable_refine=false, plan has NO refine
# ---------------------------------------------------------------------------


def test_create_job_202_when_enable_refine_false_plan_has_no_refine(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CFG-02: enable_refine=false (or omitted) + key set -> 202, plan has no refine step."""
    monkeypatch.setattr(settings, "anthropic_api_key", SecretStr("sk-ant-api03-dummy"))
    monkeypatch.setattr(settings, "refine_kill_switch", False)

    body = _audio_payload(settings.blob_root)
    body["enable_refine"] = False

    response = client.post("/v1/jobs", json=body)
    assert response.status_code == 202, response.text

    job_id = response.json()["job_id"]
    record = _get_job_record(job_id)
    assert record is not None
    assert record.config.enable_refine is False
    assert "refine" not in record.config.get_execution_plan()


# ---------------------------------------------------------------------------
# V11 - CFG-04: 202 with enable_refine=true + key, plan INCLUDES refine
# ---------------------------------------------------------------------------


def test_create_job_202_when_enable_refine_true_plan_includes_refine(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CFG-04: enable_refine=true + key set + kill-switch off -> 202, plan has refine."""
    monkeypatch.setattr(settings, "anthropic_api_key", SecretStr("sk-ant-api03-dummy"))
    monkeypatch.setattr(settings, "refine_kill_switch", False)

    body = _audio_payload(settings.blob_root)
    body["enable_refine"] = True

    response = client.post("/v1/jobs", json=body)
    assert response.status_code == 202, response.text

    job_id = response.json()["job_id"]
    record = _get_job_record(job_id)
    assert record is not None
    assert record.config.enable_refine is True
    assert "refine" in record.config.get_execution_plan()


# ---------------------------------------------------------------------------
# V12 - CFG-06 / T-1-04: kill switch produces identical plan to enable_refine=false
# ---------------------------------------------------------------------------


def test_kill_switch_produces_identical_plan_to_enable_refine_false(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CFG-06: OHSHEET_REFINE_KILL_SWITCH=true + enable_refine=true -> plan == enable_refine=false plan."""
    monkeypatch.setattr(settings, "anthropic_api_key", SecretStr("sk-ant-api03-dummy"))
    monkeypatch.setattr(settings, "refine_kill_switch", True)

    body = _audio_payload(settings.blob_root)
    body["enable_refine"] = True

    response = client.post("/v1/jobs", json=body)
    assert response.status_code == 202, response.text

    job_id = response.json()["job_id"]
    record = _get_job_record(job_id)
    assert record is not None
    # Kill switch coerces effective flag to False - the stored config
    # reflects that coercion, not the original request body.
    assert record.config.enable_refine is False, (
        "Kill switch must coerce enable_refine to False at the route boundary "
        "before PipelineConfig construction (CFG-06)."
    )
    assert "refine" not in record.config.get_execution_plan()


# ---------------------------------------------------------------------------
# V13 - CFG-06: kill switch coercion emits a single log.warning with job_id
# ---------------------------------------------------------------------------


def _enable_backend_log_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make pytest's caplog fixture capture logs from the ``backend.*`` tree.

    ``backend.main._configure_app_logging()`` sets ``backend.propagate = False``
    during the FastAPI lifespan startup so library-level loggers don't double-
    print through the uvicorn/root handlers in production. That flip also
    prevents caplog (whose handler is on the ROOT logger) from seeing any
    ``backend.*`` records. Flip propagate back ON for the test — monkeypatch
    restores the original value at teardown so the production config is
    preserved across the suite.
    """
    backend_logger = logging.getLogger("backend")
    monkeypatch.setattr(backend_logger, "propagate", True)


def test_kill_switch_emits_warning_log(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """CFG-06: kill-switch coercion emits exactly one log.warning per job."""
    monkeypatch.setattr(settings, "anthropic_api_key", SecretStr("sk-ant-api03-dummy"))
    monkeypatch.setattr(settings, "refine_kill_switch", True)
    _enable_backend_log_capture(monkeypatch)

    body = _audio_payload(settings.blob_root)
    body["enable_refine"] = True

    with caplog.at_level(logging.WARNING, logger="backend.api.routes.jobs"):
        response = client.post("/v1/jobs", json=body)

    assert response.status_code == 202
    job_id = response.json()["job_id"]

    # Expect exactly one matching warning record
    kill_switch_records = [
        r for r in caplog.records
        if "refine kill switch active" in r.getMessage()
    ]
    assert len(kill_switch_records) == 1, (
        f"expected exactly 1 kill-switch warning, got {len(kill_switch_records)}: "
        f"{[r.getMessage() for r in kill_switch_records]}"
    )
    msg = kill_switch_records[0].getMessage()
    assert "stripping refine from plan" in msg, msg
    assert job_id in msg, f"job_id {job_id} not found in log message: {msg}"


def test_kill_switch_does_not_log_when_enable_refine_false(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """CFG-06 safety: no log.warning when the request did NOT request refine."""
    monkeypatch.setattr(settings, "anthropic_api_key", SecretStr("sk-ant-api03-dummy"))
    monkeypatch.setattr(settings, "refine_kill_switch", True)  # kill switch ON
    _enable_backend_log_capture(monkeypatch)

    body = _audio_payload(settings.blob_root)
    body["enable_refine"] = False  # but request does NOT opt in

    with caplog.at_level(logging.WARNING, logger="backend.api.routes.jobs"):
        response = client.post("/v1/jobs", json=body)

    assert response.status_code == 202
    kill_switch_records = [
        r for r in caplog.records
        if "refine kill switch active" in r.getMessage()
    ]
    assert kill_switch_records == [], (
        "kill switch must NOT log when enable_refine was already False - "
        f"spurious log would create noise: {[r.getMessage() for r in kill_switch_records]}"
    )
