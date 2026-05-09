"""Phase 7 unified-harness tests.

Acceptance per Phase 7 plan:

* :class:`TierSelection` factory methods produce the right shape
  for ci / nightly / arrange-only / round-trip-only.
* :func:`apply_ci_gates` correctly identifies regressions per
  strategy doc §5.1 thresholds and skips gates whose metrics are
  missing on either side.
* :func:`run_eval_set` orchestrates per-song scoring, writes
  ``aggregate.json`` with the schema-versioned payload, and the
  composite-Q computation drops missing tiers.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pretty_midi
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.harness import (  # noqa: E402
    DEFAULT_CI_GATES,
    GateOutcome,
    GateReport,
    SongScore,
    TierSelection,
    aggregate_rows,
    apply_ci_gates,
    render_gate_summary,
    run_eval_set,
)

# ---------------------------------------------------------------------------
# TierSelection factories
# ---------------------------------------------------------------------------

def test_tier_selection_ci_skips_tier4():
    s = TierSelection.ci()
    assert s.tier_rf and s.tier2 and s.tier3
    assert not s.tier4
    assert s.composite_q


def test_tier_selection_nightly_runs_everything():
    s = TierSelection.nightly()
    assert s.tier_rf and s.tier2 and s.tier3 and s.tier4
    assert s.tier4_round_trip
    assert s.composite_q


def test_tier_selection_arrange_only_is_tier3():
    s = TierSelection.arrange_only()
    assert not s.tier_rf
    assert not s.tier2
    assert s.tier3
    assert not s.tier4
    assert not s.composite_q


def test_tier_selection_round_trip_only():
    s = TierSelection.round_trip_only()
    assert not s.tier3
    assert s.tier4
    assert s.tier4_round_trip
    assert not s.tier4_clap


# ---------------------------------------------------------------------------
# apply_ci_gates — strategy doc §5.1
# ---------------------------------------------------------------------------

def _payload(metrics: dict[str, float], run_id: str = "test") -> dict:
    return {"run_id": run_id, "aggregate": dict(metrics)}


def test_ci_gates_pass_when_head_matches_baseline():
    base = _payload({
        "mean_tier2_chord_score": 0.50,
        "mean_tier3_playability_fraction": 0.70,
        "mean_tier4_round_trip_f1_no_offset": 0.80,
        "mean_tier4_clap_cosine": 0.60,
    }, run_id="base")
    head = _payload({
        "mean_tier2_chord_score": 0.50,
        "mean_tier3_playability_fraction": 0.70,
        "mean_tier4_round_trip_f1_no_offset": 0.80,
        "mean_tier4_clap_cosine": 0.60,
    }, run_id="head")
    report = apply_ci_gates(head, base)
    assert report.all_passed
    assert all(o.passed for o in report.outcomes)


def test_ci_gates_fail_on_chord_regression_above_threshold():
    base = _payload({"mean_tier2_chord_score": 0.50})
    head = _payload({"mean_tier2_chord_score": 0.45})  # -5 ppt > 3 ppt threshold
    report = apply_ci_gates(head, base)
    assert not report.all_passed
    failing = [o for o in report.outcomes if not o.passed]
    assert any(o.name == "chord_mirex_regression" for o in failing)


def test_ci_gates_fail_on_playability_regression():
    base = _payload({"mean_tier3_playability_fraction": 0.80})
    head = _payload({"mean_tier3_playability_fraction": 0.70})  # -10 ppt > 5 ppt
    report = apply_ci_gates(head, base)
    assert not report.all_passed
    assert any(o.name == "playability_regression" and not o.passed for o in report.outcomes)


def test_ci_gates_pass_when_drop_below_threshold():
    base = _payload({"mean_tier3_playability_fraction": 0.80})
    head = _payload({"mean_tier3_playability_fraction": 0.78})  # -2 ppt < 5 ppt
    report = apply_ci_gates(head, base)
    play = next(o for o in report.outcomes if o.name == "playability_regression")
    assert play.passed


def test_ci_gates_skip_when_metric_missing():
    """A gate whose metric isn't in either payload should pass with a skip note."""
    base = _payload({"mean_tier2_chord_score": 0.50})
    head = _payload({"mean_tier2_chord_score": 0.50})
    report = apply_ci_gates(head, base)
    play = next(o for o in report.outcomes if o.name == "playability_regression")
    assert play.passed
    assert "skipped" in play.message.lower()


def test_ci_gates_pass_when_head_improves():
    base = _payload({"mean_tier2_chord_score": 0.50})
    head = _payload({"mean_tier2_chord_score": 0.65})
    report = apply_ci_gates(head, base)
    chord = next(o for o in report.outcomes if o.name == "chord_mirex_regression")
    assert chord.passed
    assert chord.delta == pytest.approx(0.15)


def test_render_gate_summary_marks_pass_fail():
    base = _payload({"mean_tier2_chord_score": 0.50})
    head = _payload({"mean_tier2_chord_score": 0.40})
    report = apply_ci_gates(head, base)
    summary = render_gate_summary(report)
    assert "FAIL" in summary
    assert "chord_mirex_regression" in summary


def test_ci_gates_fail_when_baseline_missing_active_tier_key():
    """Stale baseline (head has tier2 metric, baseline doesn't) must fail
    the gate loud — silent skip would let real regressions merge green.
    """
    base = _payload({})
    head = _payload({"mean_tier2_chord_score": 0.50})
    report = apply_ci_gates(head, base, selected_tiers=TierSelection.ci())
    chord = next(o for o in report.outcomes if o.name == "chord_mirex_regression")
    assert not chord.passed
    assert "stale" in chord.message.lower()


def test_ci_gates_fail_when_head_missing_active_tier_key():
    """Head missing a metric that the selected tier should have produced
    points at a harness regression — fail loud.
    """
    base = _payload({"mean_tier2_chord_score": 0.50})
    head = _payload({})
    report = apply_ci_gates(head, base, selected_tiers=TierSelection.ci())
    chord = next(o for o in report.outcomes if o.name == "chord_mirex_regression")
    assert not chord.passed
    assert "head missing" in chord.message.lower()


def test_ci_gates_skip_when_tier_inactive_for_run():
    """A gate whose tier was deliberately disabled (e.g. tier4 in a CI
    run) skips cleanly with the selected-tier hint.
    """
    base = _payload({"mean_tier2_chord_score": 0.50})
    head = _payload({"mean_tier2_chord_score": 0.50})
    # CI selection has tier4=False, so the round-trip / clap gates are
    # tier-inactive and skip cleanly even though the keys are missing.
    report = apply_ci_gates(head, base, selected_tiers=TierSelection.ci())
    rt = next(o for o in report.outcomes if o.name == "round_trip_regression")
    assert rt.passed
    assert "tier inactive" in rt.message.lower()


def test_ci_gates_skip_when_head_missing_but_tier_inactive():
    """Regression: when CI's TierSelection has tier4 off but the
    baseline payload has tier4 metrics (because it was captured by an
    earlier nightly/all-tiers run), the gate's head_v is None but
    base_v is a real number. Pre-fix, this surfaced as a "FAIL: head
    missing" gate even though the run never asked for tier4 — a false
    positive that would block PRs every time CI compared against a
    nightly-captured baseline.
    """
    base = _payload({
        "mean_tier2_chord_score": 0.50,
        "mean_tier4_round_trip_f1_no_offset": 0.80,
        "mean_tier4_clap_cosine": 0.60,
    })
    head = _payload({"mean_tier2_chord_score": 0.50})
    report = apply_ci_gates(head, base, selected_tiers=TierSelection.ci())
    rt = next(o for o in report.outcomes if o.name == "round_trip_regression")
    assert rt.passed
    assert "tier inactive" in rt.message.lower()
    # The baseline value should still be reported on the skip outcome
    # so reviewers can see what was on the other side of the gate.
    assert rt.baseline_value == pytest.approx(0.80)


# ---------------------------------------------------------------------------
# aggregate_rows
# ---------------------------------------------------------------------------

def test_aggregate_rows_excludes_errors():
    rows = [
        SongScore(slug="ok-1", tier_rf={"chord_rf": 0.4, "playability_rf": 0.7, "chroma_rf": 0.5}),
        SongScore(slug="ok-2", tier_rf={"chord_rf": 0.6, "playability_rf": 0.9, "chroma_rf": 0.7}),
        SongScore(slug="bad", error="OOPS"),
    ]
    agg = aggregate_rows(rows, TierSelection(tier_rf=True, tier2=False, tier3=False))
    assert agg["n_songs_total"] == 3
    assert agg["n_songs_scored"] == 2
    assert agg["n_songs_errored"] == 1
    assert agg["mean_tier_rf_chord_rf"] == pytest.approx(0.5)
    assert agg["mean_tier_rf_playability_rf"] == pytest.approx(0.8)


def test_aggregate_rows_handles_no_successful_songs():
    rows = [SongScore(slug="bad", error="FAIL")]
    agg = aggregate_rows(rows, TierSelection())
    assert agg["n_songs_scored"] == 0
    assert agg["n_songs_errored"] == 1
    # No mean_* keys when nothing scored.
    assert all(not k.startswith("mean_") for k in agg if k != "mean_wall_sec")


# ---------------------------------------------------------------------------
# run_eval_set — orchestration smoke test
# ---------------------------------------------------------------------------

def _build_chord_midi(pitches: list[int], duration_sec: float = 4.0) -> bytes:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0, name="Piano")
    for p in pitches:
        inst.notes.append(
            pretty_midi.Note(velocity=90, pitch=p, start=0.0, end=duration_sec)
        )
    pm.instruments.append(inst)
    buf = io.BytesIO()
    pm.write(buf)
    return buf.getvalue()


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    import soundfile as sf
    sf.write(str(path), audio, sr)


def test_run_eval_set_writes_aggregate_with_tier3(tmp_path, monkeypatch):
    """End-to-end orchestration with a stubbed pipeline.

    Verifies:
    * aggregate.json is written.
    * Each song row carries tier_rf and tier3 dicts when those tiers are on.
    * The schema-version field is bumped to 2 (Phase 7).
    """
    from shared.contracts import (  # noqa: PLC0415
        HarmonicAnalysis,
        PianoScore,
        QualitySignal,
        ScoreMetadata,
        ScoreNote,
        TempoMapEntry,
        TranscriptionResult,
    )

    from scripts import eval_mini  # noqa: PLC0415

    cmaj7 = [60, 64, 67, 71]
    midi_bytes = _build_chord_midi(cmaj7)
    from eval.tier_rf import fluidsynth_resynth  # noqa: PLC0415
    audio, sr = fluidsynth_resynth(midi_bytes)

    eval_set_path = tmp_path / "set"
    songs_dir = eval_set_path / "songs"
    songs_dir.mkdir(parents=True)
    audio_path = songs_dir / "source.wav"
    _write_wav(audio_path, audio, sr)
    manifest = {
        "schema_version": 1,
        "eval_set": "harness_test",
        "target_duration_sec": 5.0,
        "chord_recognition_key": "C:major",
        "songs": [
            {
                "slug": "harness_001",
                "title": "Harness chord",
                "artist": "test",
                "genre": "test",
                "source": {
                    "kind": "audio_file",
                    "path": "songs/source.wav",
                },
            },
        ],
    }
    (eval_set_path / "manifest.yaml").write_text(yaml.safe_dump(manifest))

    score = PianoScore(
        right_hand=[
            ScoreNote(id=f"rh-{i}", pitch=p, onset_beat=0.0, duration_beat=1.0,
                      velocity=80, voice=1)
            for i, p in enumerate(cmaj7)
        ],
        left_hand=[
            ScoreNote(id="lh-0", pitch=48, onset_beat=0.0, duration_beat=1.0,
                      velocity=80, voice=1),
        ],
        metadata=ScoreMetadata(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
            difficulty="intermediate",
        ),
    )

    def fake_pipeline(_audio_path: Path):
        txr = TranscriptionResult(
            midi_tracks=[],
            analysis=HarmonicAnalysis(
                key="C:major",
                time_signature=(4, 4),
                tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
            ),
            quality=QualitySignal(overall_confidence=1.0, warnings=[]),
        )
        return eval_mini.PipelineArtifacts(
            score=score, midi_bytes=midi_bytes, key_label="C:major", transcription=txr,
        )

    monkeypatch.setattr(eval_mini, "_run_pipeline", fake_pipeline)

    output_dir = tmp_path / "run"
    payload = run_eval_set(
        eval_set_path=eval_set_path,
        output_dir=output_dir,
        tiers=TierSelection(tier_rf=True, tier3=True, tier2=False, composite_q=True),
    )

    assert payload["schema_version"] == 2
    assert payload["aggregate"]["n_songs_scored"] == 1
    song = payload["songs"][0]
    assert song["slug"] == "harness_001"
    assert song["error"] is None
    assert "tier_rf" in song
    assert "tier3" in song
    assert "composite_q" in song
    out_path = output_dir / "aggregate.json"
    assert out_path.is_file()
    on_disk = json.loads(out_path.read_text())
    assert on_disk["schema_version"] == 2


# ---------------------------------------------------------------------------
# composite_q computation
# ---------------------------------------------------------------------------

def test_composite_q_drops_missing_tiers():
    """Composite Q should re-average over the present terms.

    With only Tier 3 present, the Q should equal the Tier 3 composite
    (weight gets re-normalized from 0.30 to 1.0).
    """
    from eval.harness import _compute_composite_q  # noqa: PLC0415

    row = SongScore(slug="t", tier3={"composite": 0.7})
    q = _compute_composite_q(row)
    assert q == pytest.approx(0.7)


def test_composite_q_blends_three_tiers():
    from eval.harness import _compute_composite_q  # noqa: PLC0415

    row = SongScore(
        slug="t",
        tier2={"mean_score": 0.6},
        tier3={"composite": 0.8},
        tier4={"composite": 0.5},
    )
    q = _compute_composite_q(row)
    expected = (0.30 * 0.6 + 0.30 * 0.8 + 0.40 * 0.5) / 1.0
    assert q == pytest.approx(expected)


def test_composite_q_returns_none_for_empty_row():
    from eval.harness import _compute_composite_q  # noqa: PLC0415

    row = SongScore(slug="t")
    assert _compute_composite_q(row) is None


# ---------------------------------------------------------------------------
# GateReport serialization
# ---------------------------------------------------------------------------

def test_gate_report_as_dict_round_trips():
    outcome = GateOutcome(
        name="chord_mirex_regression",
        passed=False,
        head_value=0.40,
        baseline_value=0.50,
        delta=-0.10,
        threshold=0.03,
        direction="regression_if_drop_gt",
        message="head=0.40 baseline=0.50 delta=-0.10 threshold=0.03 FAIL",
    )
    report = GateReport(
        outcomes=[outcome],
        all_passed=False,
        head_run_id="h",
        baseline_run_id="b",
    )
    payload = report.as_dict()
    assert payload["all_passed"] is False
    assert payload["outcomes"][0]["name"] == "chord_mirex_regression"
    assert payload["outcomes"][0]["passed"] is False


def test_default_ci_gates_match_strategy_doc():
    """Lock the §5.1 thresholds — drift here is a strategy-doc deviation."""
    by_name = {gate[0]: gate[2] for gate in DEFAULT_CI_GATES}
    assert by_name["chord_mirex_regression"] == pytest.approx(0.03)
    assert by_name["playability_regression"] == pytest.approx(0.05)
    assert by_name["round_trip_regression"] == pytest.approx(0.05)
    assert by_name["clap_regression"] == pytest.approx(0.05)
