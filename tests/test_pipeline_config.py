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


# ===========================================================================
# Phase 1 (CFG-01) — enable_refine execution plan tests
# ===========================================================================
#
# V7: refine insertion correctness across all 4 variants.
# V8: byte-equal baseline guard — enable_refine=False is completely
#     transparent to pre-Phase-1 behavior.
#
# Baseline captured by tool run 2026-04-13 against
# shared/shared/contracts.py SCHEMA_VERSION="3.0.0" (pre-bump). These
# values are IMMUTABLE — any Phase 1 edit that alters the enable_refine=False
# output for any (variant, skip_humanizer, score_pipeline) combination fails
# this test. Copied VERBATIM from 01-VALIDATION.md §"Zero-Behavior-Change Proof".

import pytest

# Immutable pre-Phase-1 baseline. DO NOT edit these values to make a test
# pass — that defeats the entire point. If a legitimate routing change
# ships in a later phase, add a new test for that behavior; leave these
# alone as the "Phase 1 was zero-behavior-change" proof.
_BASELINE_PLANS_ENABLE_REFINE_FALSE = {
    ("full",         False, "arrange"):            ["ingest", "transcribe", "arrange", "humanize", "engrave"],
    ("audio_upload", False, "arrange"):            ["ingest", "transcribe", "arrange", "humanize", "engrave"],
    ("midi_upload",  False, "arrange"):            ["ingest",               "arrange", "humanize", "engrave"],
    ("sheet_only",   False, "arrange"):            ["ingest", "transcribe", "arrange",              "engrave"],
    ("midi_upload",  False, "condense_transform"): ["ingest", "condense", "transform", "humanize", "engrave"],
    ("sheet_only",   True,  "condense_transform"): ["ingest", "transcribe", "condense", "transform", "engrave"],
}


@pytest.mark.parametrize(
    "variant,skip_humanizer,score_pipeline,expected",
    [
        (v, s, p, plan)
        for (v, s, p), plan in _BASELINE_PLANS_ENABLE_REFINE_FALSE.items()
    ],
)
def test_enable_refine_false_preserves_pre_phase_1_baseline(
    variant: str, skip_humanizer: bool, score_pipeline: str, expected: list[str]
) -> None:
    """V8 (CFG-01 frozen baseline): enable_refine=False is byte-equal to pre-Phase-1 output.

    Any drift here is a Phase 1 bug. Phase 2 can expand the allowed plans
    for enable_refine=True, but the enable_refine=False case stays frozen.
    """
    cfg = PipelineConfig(
        variant=variant,
        skip_humanizer=skip_humanizer,
        score_pipeline=score_pipeline,
        enable_refine=False,
    )
    assert cfg.get_execution_plan() == expected


@pytest.mark.parametrize(
    "variant,expected_insert_index,expected_insert_after",
    [
        ("full",         4, "humanize"),
        ("audio_upload", 4, "humanize"),
        ("midi_upload",  3, "humanize"),
        ("sheet_only",   3, "arrange"),  # no humanize — refine goes after arrange
    ],
)
def test_enable_refine_true_inserts_refine_at_correct_position(
    variant: str, expected_insert_index: int, expected_insert_after: str
) -> None:
    """V7 (CFG-01): enable_refine=True inserts refine directly after humanize.

    For sheet_only (no humanize), refine goes after arrange. This places it
    immediately before engrave in every variant, which is the natural slot
    for an LLM pass that operates on the humanized/arranged score.
    """
    cfg = PipelineConfig(variant=variant, enable_refine=True)
    plan = cfg.get_execution_plan()
    assert "refine" in plan, plan
    assert plan[expected_insert_index] == "refine", plan
    assert plan[expected_insert_index - 1] == expected_insert_after, plan
    # Refine sits immediately before engrave in every variant
    assert plan[expected_insert_index + 1] == "engrave", plan


def test_enable_refine_true_with_condense_transform_inserts_refine_after_transform() -> None:
    """V7 corner case: enable_refine + condense_transform + sheet_only puts refine after transform."""
    cfg = PipelineConfig(
        variant="sheet_only",
        score_pipeline="condense_transform",
        skip_humanizer=True,
        enable_refine=True,
    )
    plan = cfg.get_execution_plan()
    assert plan == ["ingest", "transcribe", "condense", "transform", "refine", "engrave"], plan


def test_enable_refine_default_is_false() -> None:
    """Regression guard: default PipelineConfig must NOT include refine."""
    cfg = PipelineConfig(variant="full")
    assert cfg.enable_refine is False
    assert "refine" not in cfg.get_execution_plan()
