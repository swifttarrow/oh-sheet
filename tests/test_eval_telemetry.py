"""Phase 7 production-telemetry tests.

Acceptance per Phase 7 plan:

* :func:`compute_production_quality_report` produces a composite Q
  in ``[0, 1]`` that combines Tier 3 + Tier 2-lite weights at 0.7
  and 0.3.
* :class:`TelemetryClient` no-ops cleanly when
  ``OHSHEET_EVAL_TELEMETRY_DSN`` is unset (the dev / CI default) and
  doesn't raise even when ``psycopg`` is missing or the DB is down.
* :func:`emit_production_quality` returns a report and never raises
  (telemetry must NEVER fail a production job).
* The :class:`EvaluationReport` Pydantic field on ``EngravedOutput``
  accepts the report shape produced by ``as_evaluation_report``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.contracts import (  # noqa: E402
    EngravedOutput,
    EngravedScoreData,
    EvaluationReport,
    PianoScore,
    ScoreChordEvent,
    ScoreMetadata,
    ScoreNote,
    TempoMapEntry,
)

from backend.eval.telemetry import (  # noqa: E402
    PRODUCTION_Q_VERSION,
    TelemetryClient,
    compute_production_quality_report,
    emit_production_quality,
    get_telemetry_client,
    reset_telemetry_client,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Each test gets a fresh telemetry client singleton."""
    reset_telemetry_client()
    yield
    reset_telemetry_client()


def _make_score(*, with_chords: bool = True, with_key: bool = True) -> PianoScore:
    return PianoScore(
        right_hand=[
            ScoreNote(id="rh-0", pitch=60, onset_beat=0.0, duration_beat=1.0,
                      velocity=80, voice=1),
            ScoreNote(id="rh-1", pitch=64, onset_beat=0.0, duration_beat=1.0,
                      velocity=80, voice=1),
            ScoreNote(id="rh-2", pitch=67, onset_beat=0.0, duration_beat=1.0,
                      velocity=80, voice=1),
            ScoreNote(id="rh-3", pitch=62, onset_beat=1.0, duration_beat=1.0,
                      velocity=80, voice=1),
            ScoreNote(id="rh-4", pitch=65, onset_beat=1.0, duration_beat=1.0,
                      velocity=80, voice=1),
        ],
        left_hand=[
            ScoreNote(id="lh-0", pitch=48, onset_beat=0.0, duration_beat=1.0,
                      velocity=80, voice=1),
            ScoreNote(id="lh-1", pitch=43, onset_beat=1.0, duration_beat=1.0,
                      velocity=80, voice=1),
        ],
        metadata=ScoreMetadata(
            key="C:major" if with_key else "",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
            difficulty="intermediate",
            chord_symbols=(
                [ScoreChordEvent(beat=0.0, duration_beat=2.0, label="C:maj", root=60)]
                if with_chords else []
            ),
        ),
    )


# ---------------------------------------------------------------------------
# compute_production_quality_report
# ---------------------------------------------------------------------------

def test_compute_production_q_returns_unit_range():
    score = _make_score()
    report = compute_production_quality_report(score=score)
    assert 0.0 <= report.composite_q <= 1.0
    assert report.q_version == PRODUCTION_Q_VERSION


def test_compute_production_q_includes_tier3_and_tier2_lite():
    score = _make_score()
    report = compute_production_quality_report(score=score)
    assert "tier3_composite" in report.contributing_terms
    assert "tier2_lite" in report.contributing_terms
    assert report.tier3_composite is not None
    assert report.tier3_playability_fraction is not None
    assert report.tier2_chord_symbol_count == 1
    assert report.tier2_has_key is True


def test_compute_production_q_drops_score_with_no_chords_or_key():
    score = _make_score(with_chords=False, with_key=False)
    report = compute_production_quality_report(score=score)
    # Tier 2 lite term still runs (just scores 0); Tier 3 still runs.
    assert report.tier2_chord_symbol_count == 0
    assert report.tier2_has_key is False
    # Composite drops because Tier 2 lite contributes 0.
    assert report.composite_q < 1.0


def test_compute_production_q_handles_empty_score():
    score = PianoScore(
        right_hand=[],
        left_hand=[],
        metadata=ScoreMetadata(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
            difficulty="beginner",
        ),
    )
    # Must not raise; composite ends up low but valid.
    report = compute_production_quality_report(score=score)
    assert 0.0 <= report.composite_q <= 1.0


# ---------------------------------------------------------------------------
# as_evaluation_report — round-trips into EngravedOutput
# ---------------------------------------------------------------------------

def test_as_evaluation_report_round_trips_through_pydantic():
    score = _make_score()
    report = compute_production_quality_report(score=score)
    payload = report.as_evaluation_report()
    eval_report = EvaluationReport.model_validate(payload)
    assert eval_report.composite_q == pytest.approx(payload["composite_q"])
    assert eval_report.q_version == PRODUCTION_Q_VERSION


def test_engraved_output_accepts_evaluation_report():
    score = _make_score()
    report = compute_production_quality_report(score=score)
    eo = EngravedOutput(
        metadata=EngravedScoreData(
            includes_dynamics=True,
            includes_pedal_marks=True,
            includes_fingering=False,
            includes_chord_symbols=True,
            title="t",
            composer="c",
        ),
        musicxml_uri="file:///tmp/x.xml",
        humanized_midi_uri="file:///tmp/x.mid",
        evaluation_report=EvaluationReport.model_validate(report.as_evaluation_report()),
    )
    assert eo.evaluation_report is not None
    assert 0.0 <= eo.evaluation_report.composite_q <= 1.0


def test_engraved_output_accepts_none_evaluation_report():
    eo = EngravedOutput(
        metadata=EngravedScoreData(
            includes_dynamics=False, includes_pedal_marks=False,
            includes_fingering=False, includes_chord_symbols=False,
            title="t", composer="c",
        ),
        musicxml_uri="file:///tmp/x.xml",
        humanized_midi_uri="file:///tmp/x.mid",
    )
    assert eo.evaluation_report is None


# ---------------------------------------------------------------------------
# TelemetryClient — no-op when DSN unset
# ---------------------------------------------------------------------------

def test_telemetry_client_disabled_when_dsn_unset(monkeypatch):
    monkeypatch.delenv("OHSHEET_EVAL_TELEMETRY_DSN", raising=False)
    client = TelemetryClient()
    assert not client.enabled


def test_telemetry_client_disabled_with_empty_dsn():
    client = TelemetryClient(dsn="")
    assert not client.enabled


def test_telemetry_client_record_returns_none_when_disabled():
    client = TelemetryClient(dsn="")
    score = _make_score()
    report = compute_production_quality_report(score=score)
    result = client.record(report, job_id="job-1", user_audio_hash="abc123")
    assert result is None


def test_telemetry_client_singleton_resets():
    c1 = get_telemetry_client()
    reset_telemetry_client()
    c2 = get_telemetry_client()
    assert c1 is not c2


def test_telemetry_client_swallows_psycopg_failure(monkeypatch):
    """Even with DSN set, a connection failure must not raise.

    Telemetry is fire-and-forget; the runner can't have its job
    blocked by a transient DB outage.
    """
    monkeypatch.setenv("OHSHEET_EVAL_TELEMETRY_DSN", "postgresql://nowhere:1/x")
    client = TelemetryClient()
    if not client.enabled:
        pytest.skip("psycopg not installed in this environment")
    score = _make_score()
    report = compute_production_quality_report(score=score)
    # If psycopg is installed, this will fail to connect; should
    # return None silently without raising.
    result = client.record(report, job_id="job-1", user_audio_hash="abc")
    assert result is None


# ---------------------------------------------------------------------------
# emit_production_quality — fire-and-forget contract
# ---------------------------------------------------------------------------

def test_emit_production_quality_returns_report_when_enabled(monkeypatch):
    monkeypatch.delenv("OHSHEET_EVAL_TELEMETRY_DSN", raising=False)
    score = _make_score()
    report = emit_production_quality(
        score=score, job_id="j1", user_audio_hash="hash1",
        engrave_route="local", title="Test",
    )
    assert report is not None
    assert 0.0 <= report.composite_q <= 1.0


def test_emit_production_quality_never_raises_on_compute_failure(monkeypatch):
    """If compute fails (e.g. score is None), emit returns None, no raise."""
    from backend.eval import telemetry as tel  # noqa: PLC0415

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic compute failure")

    monkeypatch.setattr(tel, "compute_production_quality_report", boom)
    score = _make_score()
    result = emit_production_quality(
        score=score, job_id="j1", user_audio_hash="hash1",
    )
    assert result is None


# ---------------------------------------------------------------------------
# Build-insert-sql lock — schema column drift detector
# ---------------------------------------------------------------------------

def test_db_row_columns_match_migration():
    """Lock the column set against the SQL migration so a schema drift
    fails this test rather than the production INSERT.
    """
    score = _make_score()
    report = compute_production_quality_report(score=score)
    row = report.as_db_row(
        job_id="j", user_audio_hash="h", engrave_route="local", title="t",
    )
    expected_cols = {
        "job_id", "created_at", "user_audio_hash", "composite_quality_score",
        "q_version", "tier3_playability_fraction", "tier3_voice_leading_smoothness",
        "tier3_polyphony_in_target_range", "tier3_sight_readability",
        "tier3_engraving_warning_count", "tier3_composite",
        "tier2_has_key", "tier2_chord_symbol_count", "engrave_route", "title",
    }
    assert set(row.keys()) == expected_cols
