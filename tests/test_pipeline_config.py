from __future__ import annotations

from backend.contracts import PipelineConfig


def test_default_plan_includes_refine_before_engrave() -> None:
    """enable_refine defaults to True — refine slots immediately before engrave."""
    cfg = PipelineConfig(variant="audio_upload", enable_refine=True)
    assert cfg.get_execution_plan() == [
        "ingest",
        "separate",
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
        "separate",
        "transcribe",
        "arrange",
        "humanize",
        "engrave",
    ]


def test_sheet_only_refine_runs_after_arrange() -> None:
    cfg = PipelineConfig(variant="sheet_only", enable_refine=True)
    assert cfg.get_execution_plan() == [
        "ingest",
        "separate",
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
        "separate",
        "transcribe",
        "condense",
        "refine",
        "engrave",
    ]


# ---------------------------------------------------------------------------
# Phase 8: pop_cover variant
# ---------------------------------------------------------------------------


def test_pop_cover_skips_arrange_humanize_and_refine() -> None:
    """AMT-APC emits arrangement-ready piano output, so the cover-mode plan
    skips arrange and humanize entirely. Refine also drops out — it
    dispatches on PianoScore / HumanizedPerformance, neither of which
    exists in cover mode. Separate still runs (instrumental stem feeds
    AMT-APC); engrave still runs."""
    cfg = PipelineConfig(variant="pop_cover", enable_refine=True)
    assert cfg.get_execution_plan() == [
        "ingest",
        "separate",
        "transcribe",
        "engrave",
    ]


def test_pop_cover_without_refine() -> None:
    cfg = PipelineConfig(variant="pop_cover", enable_refine=False)
    assert cfg.get_execution_plan() == [
        "ingest",
        "separate",
        "transcribe",
        "engrave",
    ]


def test_pop_cover_without_separator() -> None:
    """When the operator disables Demucs globally, pop_cover still runs —
    AMT-APC will fall back to the full mix instead of the instrumental
    stem (less ideal but still produces a cover)."""
    cfg = PipelineConfig(
        variant="pop_cover",
        separator="off",
        enable_refine=False,
    )
    assert cfg.get_execution_plan() == [
        "ingest",
        "transcribe",
        "engrave",
    ]


def test_pop_cover_skip_humanizer_is_noop() -> None:
    """Cover mode never had humanize in its plan, so skip_humanizer is a no-op
    rather than an error — keeps API callers simple (they can pass the same
    skip_humanizer value across variants)."""
    cfg = PipelineConfig(
        variant="pop_cover",
        skip_humanizer=True,
        enable_refine=False,
    )
    assert cfg.get_execution_plan() == [
        "ingest",
        "separate",
        "transcribe",
        "engrave",
    ]
