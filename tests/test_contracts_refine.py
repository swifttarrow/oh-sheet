"""Schema-level tests for the refinement contract additions (SCHEMA_VERSION 3.2.0)."""
from __future__ import annotations

from shared.contracts import (
    SCHEMA_VERSION,
    Repeat,
    ScoreMetadata,
    ScoreSection,
    SectionLabel,
    TempoMapEntry,
)


def test_schema_version_bumped_to_3_2_0() -> None:
    assert SCHEMA_VERSION == "3.2.0"


def test_repeat_model_requires_beat_range_and_kind() -> None:
    r = Repeat(start_beat=0.0, end_beat=16.0, kind="simple")
    assert r.kind == "simple"
    r2 = Repeat(start_beat=32.0, end_beat=48.0, kind="with_endings")
    assert r2.kind == "with_endings"


def test_score_section_allows_custom_label() -> None:
    s = ScoreSection(
        start_beat=0.0,
        end_beat=16.0,
        label=SectionLabel.OTHER,
        custom_label="Bridge → Solo",
    )
    assert s.custom_label == "Bridge → Solo"


def test_score_section_custom_label_defaults_to_none() -> None:
    s = ScoreSection(start_beat=0.0, end_beat=16.0, label=SectionLabel.INTRO)
    assert s.custom_label is None


def test_score_metadata_accepts_refinement_fields() -> None:
    md = ScoreMetadata(
        key="Db:major",
        time_signature=(4, 4),
        tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=66.0)],
        difficulty="intermediate",
        title="Clair de Lune",
        composer="Claude Debussy",
        arranger=None,
        tempo_marking="Andante",
        staff_split_hint=60,
        repeats=[Repeat(start_beat=0.0, end_beat=16.0, kind="simple")],
    )
    assert md.title == "Clair de Lune"
    assert md.composer == "Claude Debussy"
    assert md.tempo_marking == "Andante"
    assert md.staff_split_hint == 60
    assert len(md.repeats) == 1


def test_repeat_rejects_unsupported_kind() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Repeat(start_beat=0.0, end_beat=16.0, kind="da_capo")  # type: ignore[arg-type]


def test_staff_split_hint_rejects_out_of_range() -> None:
    import pytest
    from pydantic import ValidationError

    for bad in (-1, 128, 200):
        with pytest.raises(ValidationError):
            ScoreMetadata(
                key="C:major",
                time_signature=(4, 4),
                tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
                difficulty="intermediate",
                staff_split_hint=bad,
            )


def test_score_metadata_refinement_fields_all_optional() -> None:
    md = ScoreMetadata(
        key="C:major",
        time_signature=(4, 4),
        tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
        difficulty="intermediate",
    )
    assert md.title is None
    assert md.composer is None
    assert md.arranger is None
    assert md.tempo_marking is None
    assert md.staff_split_hint is None
    assert md.repeats == []


def test_backwards_compatible_with_v3_0_0_payloads() -> None:
    """A ScoreMetadata JSON with no refinement fields still validates."""
    payload = {
        "key": "G:major",
        "time_signature": [3, 4],
        "tempo_map": [{"time_sec": 0.0, "beat": 0.0, "bpm": 90.0}],
        "difficulty": "beginner",
        "sections": [],
        "chord_symbols": [],
    }
    md = ScoreMetadata.model_validate(payload)
    assert md.title is None
    assert md.repeats == []


def test_backend_reexports_repeat() -> None:
    from backend.contracts import Repeat as BackendRepeat
    assert BackendRepeat is Repeat
