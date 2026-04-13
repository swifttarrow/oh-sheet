from __future__ import annotations

from backend.contracts import PipelineConfig


def test_default_plan_includes_refine_before_engrave() -> None:
    """enable_refine defaults to True — refine slots immediately before engrave."""
    cfg = PipelineConfig(variant="audio_upload")
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


def test_condense_transform_plan_places_refine_before_engrave() -> None:
    cfg = PipelineConfig(
        variant="midi_upload",
        score_pipeline="condense_transform",
    )
    assert cfg.get_execution_plan() == [
        "ingest",
        "condense",
        "transform",
        "humanize",
        "refine",
        "engrave",
    ]


def test_condense_transform_with_skip_humanizer_includes_refine() -> None:
    cfg = PipelineConfig(
        variant="sheet_only",
        score_pipeline="condense_transform",
        skip_humanizer=True,
    )
    assert cfg.get_execution_plan() == [
        "ingest",
        "transcribe",
        "condense",
        "transform",
        "refine",
        "engrave",
    ]
