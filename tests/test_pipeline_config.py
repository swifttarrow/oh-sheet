from __future__ import annotations

from backend.contracts import PipelineConfig


def test_default_plan_uses_arrange() -> None:
    """Default audio_upload with skip_humanizer=True skips humanize."""
    cfg = PipelineConfig(variant="audio_upload")
    assert cfg.get_execution_plan() == [
        "ingest",
        "transcribe",
        "arrange",
        "engrave",
    ]


def test_default_plan_with_humanizer() -> None:
    """audio_upload with skip_humanizer=False includes humanize."""
    cfg = PipelineConfig(variant="audio_upload", skip_humanizer=False)
    assert cfg.get_execution_plan() == [
        "ingest",
        "transcribe",
        "arrange",
        "humanize",
        "engrave",
    ]


def test_condense_transform_replaces_arrange() -> None:
    """condense_transform pipeline uses condense (no transform) and skips humanize by default."""
    cfg = PipelineConfig(variant="midi_upload", score_pipeline="condense_transform")
    assert cfg.get_execution_plan() == [
        "ingest",
        "condense",
        "engrave",
    ]


def test_condense_transform_with_humanizer() -> None:
    """condense_transform with humanizer enabled includes humanize."""
    cfg = PipelineConfig(
        variant="midi_upload",
        score_pipeline="condense_transform",
        skip_humanizer=False,
    )
    assert cfg.get_execution_plan() == [
        "ingest",
        "condense",
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
        "engrave",
    ]
