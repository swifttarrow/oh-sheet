from __future__ import annotations

from backend.contracts import PipelineConfig


def test_default_plan_includes_refine_before_engrave() -> None:
    """enable_refine defaults to True — refine slots immediately before engrave."""
    cfg = PipelineConfig(variant="audio_upload", enable_refine=True)
    assert cfg.get_execution_plan() == [
        "ingest",
        "transcribe",
        "arrange",
        "humanize",
        "refine",
        "engrave",
    ]


def test_enable_refine_false_omits_stage() -> None:
    cfg = PipelineConfig(variant="audio_upload", enable_refine=False)
    assert cfg.get_execution_plan() == [
        "ingest",
        "transcribe",
        "arrange",
        "humanize",
        "engrave",
    ]


def test_sheet_only_refine_runs_after_arrange() -> None:
    cfg = PipelineConfig(variant="sheet_only", enable_refine=True)
    assert cfg.get_execution_plan() == [
        "ingest",
        "transcribe",
        "arrange",
        "refine",
        "engrave",
    ]


def test_midi_upload_plan_includes_refine() -> None:
    cfg = PipelineConfig(variant="midi_upload", enable_refine=True)
    assert cfg.get_execution_plan() == [
        "ingest",
        "arrange",
        "humanize",
        "refine",
        "engrave",
    ]


def test_condense_only_replaces_arrange() -> None:
    """condense_only pipeline uses condense instead of arrange."""
    cfg = PipelineConfig(
        variant="midi_upload",
        score_pipeline="condense_only",
        enable_refine=False,
    )
    assert cfg.get_execution_plan() == [
        "ingest",
        "condense",
        "humanize",
        "engrave",
    ]


def test_condense_only_with_skip_humanizer_and_refine() -> None:
    cfg = PipelineConfig(
        variant="sheet_only",
        score_pipeline="condense_only",
        skip_humanizer=True,
        enable_refine=True,
    )
    assert cfg.get_execution_plan() == [
        "ingest",
        "transcribe",
        "condense",
        "refine",
        "engrave",
    ]
