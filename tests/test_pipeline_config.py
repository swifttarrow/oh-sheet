from __future__ import annotations

from backend.contracts import PipelineConfig


def test_default_plan_uses_arrange() -> None:
    cfg = PipelineConfig(variant="audio_upload")
    assert cfg.get_execution_plan() == [
        "ingest",
        "transcribe",
        "arrange",
        "humanize",
        "engrave",
    ]


def test_condense_transform_replaces_arrange() -> None:
    cfg = PipelineConfig(variant="midi_upload", score_pipeline="condense_transform")
    assert cfg.get_execution_plan() == [
        "ingest",
        "condense",
        "transform",
        "humanize",
        "engrave",
    ]


def test_condense_transform_with_skip_humanizer() -> None:
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
        "engrave",
    ]
