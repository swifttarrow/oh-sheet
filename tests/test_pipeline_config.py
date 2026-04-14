from __future__ import annotations

from backend.contracts import PipelineConfig


def test_default_plan_uses_arrange() -> None:
    """Default audio_upload includes humanize (skip_humanizer defaults to False)."""
    cfg = PipelineConfig(variant="audio_upload")
    assert cfg.get_execution_plan() == [
        "ingest",
        "transcribe",
        "arrange",
        "humanize",
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


def test_condense_only_replaces_arrange() -> None:
    """condense_only pipeline uses condense instead of arrange, includes humanize by default."""
    cfg = PipelineConfig(variant="midi_upload", score_pipeline="condense_only")
    assert cfg.get_execution_plan() == [
        "ingest",
        "condense",
        "humanize",
        "engrave",
    ]


def test_condense_only_with_humanizer() -> None:
    """condense_only with humanizer enabled includes humanize."""
    cfg = PipelineConfig(
        variant="midi_upload",
        score_pipeline="condense_only",
        skip_humanizer=False,
    )
    assert cfg.get_execution_plan() == [
        "ingest",
        "condense",
        "humanize",
        "engrave",
    ]


def test_condense_only_with_skip_humanizer() -> None:
    cfg = PipelineConfig(
        variant="sheet_only",
        score_pipeline="condense_only",
        skip_humanizer=True,
    )
    assert cfg.get_execution_plan() == [
        "ingest",
        "transcribe",
        "condense",
        "engrave",
    ]
