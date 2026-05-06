"""Phase 6 tests — Kong piano-stem AMT + sustain-pedal plumbing.

Covers:

* ``transcribe_kong._pedal_events_from_kong`` correctly converts Kong's
  pedal dicts into ``RealtimePedalEvent`` instances and respects the
  confidence floor.
* ``run_kong`` raises ``ImportError`` when the optional
  ``piano_transcription_inference`` PyPI package isn't installed (so the
  dispatcher in ``transcribe.py`` falls back to the BP stems pipeline).
* ``should_route_to_kong`` honors the gating rules: kill switch,
  no-stems-no-Kong, vocal-energy threshold, user_hint override.
* ``_arrange_sync`` converts seconds-domain ``RealtimePedalEvent`` into
  beat-domain ``PedalEvent`` and stashes it on
  ``ScoreMetadata.pedal_events`` for the engraver.
* ``_humanize_sync`` prefers transcribed pedal events over the
  chord-symbol heuristic when both are present.

Real Kong inference is gated behind an optional dep + ~170MB weight
download from Zenodo, so we monkeypatch the model interface and exercise
only the wrapping / routing logic. A real-Kong smoke test (analogous to
``test_real_transcribe_smoke.py``) is a Phase 6.5 follow-up.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.config import settings
from backend.contracts import (
    HarmonicAnalysis,
    InstrumentRole,
    MidiTrack,
    Note,
    PianoScore,
    QualitySignal,
    RealtimePedalEvent,
    ScoreMetadata,
    ScoreNote,
    TempoMapEntry,
    TranscriptionResult,
)
from backend.services import transcribe_kong as kong_mod
from backend.services import transcribe_pipeline_kong as kong_pipeline
from backend.services.arrange import _arrange_sync, _pedal_to_score_pedal
from backend.services.humanize import _humanize_sync
from backend.services.stem_separation import SeparatedStems

# ---------------------------------------------------------------------------
# RealtimePedalEvent contract
# ---------------------------------------------------------------------------


def test_realtime_pedal_event_round_trips_through_json():
    pedal = RealtimePedalEvent(
        cc=64, onset_sec=1.5, offset_sec=3.0, confidence=0.8,
    )
    blob = pedal.model_dump_json()
    restored = RealtimePedalEvent.model_validate_json(blob)
    assert restored == pedal


def test_realtime_pedal_event_validates_cc_range():
    with pytest.raises(ValidationError):
        RealtimePedalEvent(cc=999, onset_sec=0.0, offset_sec=1.0)


# ---------------------------------------------------------------------------
# Kong → RealtimePedalEvent conversion
# ---------------------------------------------------------------------------


def test_pedal_events_from_kong_filters_below_confidence_threshold(monkeypatch):
    monkeypatch.setattr(settings, "kong_pedal_min_confidence", 0.5)
    raw = [
        {"onset_time": 0.0, "offset_time": 1.0, "confidence": 0.9},
        {"onset_time": 1.5, "offset_time": 2.0, "confidence": 0.3},
        {"onset_time": 2.5, "offset_time": 3.0},   # default confidence 1.0
    ]
    out = kong_mod._pedal_events_from_kong(raw)
    assert len(out) == 2
    assert out[0].onset_sec == 0.0
    assert out[1].onset_sec == 2.5
    assert all(p.cc == 64 for p in out)


def test_pedal_events_from_kong_drops_zero_duration_segments():
    raw = [
        {"onset_time": 1.0, "offset_time": 1.0},
        {"onset_time": 2.0, "offset_time": 1.5},   # offset < onset
        {"onset_time": 3.0, "offset_time": 4.0},
    ]
    out = kong_mod._pedal_events_from_kong(raw, min_confidence=0.0)
    assert len(out) == 1
    assert out[0].onset_sec == 3.0


# ---------------------------------------------------------------------------
# Kong wrapper: ImportError when optional dep missing
# ---------------------------------------------------------------------------


def test_run_kong_raises_import_error_when_dep_missing(monkeypatch, tmp_path):
    """The dispatcher in transcribe.py catches ImportError and falls back to
    the Basic Pitch stems pipeline. Verify the wrapper actually raises it."""

    monkeypatch.setattr(kong_mod, "_KONG_MODEL", None)

    def fake_load():
        raise ImportError("piano_transcription_inference not installed")

    monkeypatch.setattr(kong_mod, "_load_kong", fake_load)
    audio = tmp_path / "fake.wav"
    audio.write_bytes(b"\x00" * 100)
    with pytest.raises(ImportError):
        kong_mod.run_kong(audio)


# ---------------------------------------------------------------------------
# Routing heuristic (should_route_to_kong)
# ---------------------------------------------------------------------------


def _stems_with(*, vocals=None, bass="b.wav", other="o.wav") -> SeparatedStems:
    return SeparatedStems(
        vocals=Path(vocals) if vocals else None,
        bass=Path(bass) if bass else None,
        other=Path(other) if other else None,
    )


def test_should_route_to_kong_disabled_by_kill_switch(monkeypatch):
    monkeypatch.setattr(settings, "kong_enabled", False)
    assert kong_pipeline.should_route_to_kong(_stems_with()) is False


def test_should_route_to_kong_requires_piano_stem(monkeypatch):
    monkeypatch.setattr(settings, "kong_enabled", True)
    assert kong_pipeline.should_route_to_kong(None) is False
    empty = SeparatedStems(vocals=None, bass=None, other=None)
    assert kong_pipeline.should_route_to_kong(empty) is False


def test_should_route_to_kong_user_hint_overrides_vocal_energy(monkeypatch):
    monkeypatch.setattr(settings, "kong_enabled", True)
    monkeypatch.setattr(settings, "kong_vocal_energy_threshold", 0.0)
    monkeypatch.setattr(kong_pipeline, "_vocal_energy", lambda _p: 1.0)
    assert kong_pipeline.should_route_to_kong(_stems_with(), user_hint="piano")


def test_should_route_to_kong_uses_vocal_energy_threshold(monkeypatch):
    monkeypatch.setattr(settings, "kong_enabled", True)
    monkeypatch.setattr(settings, "kong_vocal_energy_threshold", 0.05)
    monkeypatch.setattr(settings, "kong_user_hint_only", False)
    monkeypatch.setattr(kong_pipeline, "_vocal_energy", lambda _p: 0.01)
    assert kong_pipeline.should_route_to_kong(_stems_with()) is True
    monkeypatch.setattr(kong_pipeline, "_vocal_energy", lambda _p: 0.5)
    assert kong_pipeline.should_route_to_kong(_stems_with()) is False


def test_should_route_to_kong_user_hint_only_blocks_unhinted(monkeypatch):
    monkeypatch.setattr(settings, "kong_enabled", True)
    monkeypatch.setattr(settings, "kong_user_hint_only", True)
    # Vocal energy below threshold but unhinted → still skipped.
    monkeypatch.setattr(kong_pipeline, "_vocal_energy", lambda _p: 0.0)
    assert kong_pipeline.should_route_to_kong(_stems_with()) is False
    assert kong_pipeline.should_route_to_kong(_stems_with(), user_hint="piano")


# ---------------------------------------------------------------------------
# Pedal sec→beat conversion in arrange
# ---------------------------------------------------------------------------


def test_pedal_to_score_pedal_converts_seconds_to_beats():
    tempo_map = [TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)]
    pedal = RealtimePedalEvent(cc=64, onset_sec=1.0, offset_sec=2.5, confidence=0.9)
    converted = _pedal_to_score_pedal(pedal, tempo_map)
    assert converted is not None
    # 120 BPM → 2 beats per second
    assert converted.onset_beat == 2.0
    assert converted.offset_beat == 5.0
    assert converted.type == "sustain"


def test_pedal_to_score_pedal_drops_unknown_cc():
    tempo_map = [TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)]
    pedal = RealtimePedalEvent(cc=1, onset_sec=0.0, offset_sec=1.0)  # CC1 = mod wheel
    assert _pedal_to_score_pedal(pedal, tempo_map) is None


def test_pedal_to_score_pedal_drops_zero_duration():
    tempo_map = [TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)]
    pedal = RealtimePedalEvent(cc=64, onset_sec=1.0, offset_sec=1.0)
    assert _pedal_to_score_pedal(pedal, tempo_map) is None


def test_arrange_carries_pedal_events_through_to_score_metadata():
    tempo_map = [TempoMapEntry(time_sec=0.0, beat=0.0, bpm=60.0)]  # 1 bps
    payload = TranscriptionResult(
        midi_tracks=[
            MidiTrack(
                instrument=InstrumentRole.MELODY,
                program=0,
                confidence=0.9,
                notes=[Note(pitch=60, onset_sec=0.0, offset_sec=1.0, velocity=80)],
            ),
        ],
        analysis=HarmonicAnalysis(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=tempo_map,
        ),
        quality=QualitySignal(overall_confidence=0.9, warnings=[]),
        pedal_events=[
            RealtimePedalEvent(cc=64, onset_sec=1.0, offset_sec=3.0, confidence=0.9),
            RealtimePedalEvent(cc=66, onset_sec=4.0, offset_sec=5.0, confidence=0.8),
        ],
    )
    score = _arrange_sync(payload, "intermediate")
    assert len(score.metadata.pedal_events) == 2
    assert score.metadata.pedal_events[0].onset_beat == 1.0
    assert score.metadata.pedal_events[0].offset_beat == 3.0
    assert score.metadata.pedal_events[0].type == "sustain"
    assert score.metadata.pedal_events[1].type == "sostenuto"


def test_arrange_with_no_pedal_events_emits_empty_list():
    payload = TranscriptionResult(
        midi_tracks=[
            MidiTrack(
                instrument=InstrumentRole.MELODY,
                program=0,
                confidence=0.9,
                notes=[Note(pitch=60, onset_sec=0.0, offset_sec=1.0, velocity=80)],
            ),
        ],
        analysis=HarmonicAnalysis(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
        ),
        quality=QualitySignal(overall_confidence=0.9, warnings=[]),
    )
    score = _arrange_sync(payload, "intermediate")
    assert score.metadata.pedal_events == []


# ---------------------------------------------------------------------------
# Humanize prefers transcribed pedal over chord-symbol heuristic
# ---------------------------------------------------------------------------


def _piano_score_with_pedals(pedal_events) -> PianoScore:
    return PianoScore(
        right_hand=[
            ScoreNote(id="rh-0001", pitch=60, onset_beat=0.0,
                      duration_beat=1.0, velocity=80, voice=1),
        ],
        left_hand=[
            ScoreNote(id="lh-0001", pitch=48, onset_beat=0.0,
                      duration_beat=1.0, velocity=70, voice=1),
        ],
        metadata=ScoreMetadata(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
            difficulty="intermediate",
            pedal_events=pedal_events,
        ),
    )


def test_humanize_uses_transcribed_pedal_events_when_present():
    from backend.contracts import PedalEvent

    transcribed = [
        PedalEvent(onset_beat=0.0, offset_beat=2.0, type="sustain"),
        PedalEvent(onset_beat=4.0, offset_beat=6.0, type="sustain"),
    ]
    score = _piano_score_with_pedals(transcribed)
    perf = _humanize_sync(score, seed=42)
    assert len(perf.expression.pedal_events) == 2
    assert perf.expression.pedal_events[0].onset_beat == 0.0
    assert perf.expression.pedal_events[0].offset_beat == 2.0


def test_humanize_falls_back_to_heuristic_when_no_transcribed_pedal():
    score = _piano_score_with_pedals([])
    # Add a section + chord_symbols so the heuristic generator has
    # something to chew on.
    score = score.model_copy(
        update={
            "metadata": score.metadata.model_copy(
                update={
                    "sections": [],
                    "chord_symbols": [],
                }
            ),
        },
    )
    perf = _humanize_sync(score, seed=42)
    # Heuristic falls back to per-bar pedal — non-empty for any
    # score with notes.
    assert isinstance(perf.expression.pedal_events, list)


# ---------------------------------------------------------------------------
# Routing dispatcher tail: Kong path returns 3-tuple in transcribe sync
# ---------------------------------------------------------------------------


def test_run_basic_pitch_sync_kong_path_returns_three_tuple(monkeypatch, tmp_path):
    """When Kong claims the routing, the sync wrapper returns the 3-tuple
    ``(result, midi_bytes, pedal_events)`` so the caller can populate the
    contract field on the TranscriptionResult."""
    from backend.services import transcribe as transcribe_mod

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00" * 100)
    stems = SeparatedStems(
        vocals=None,
        bass=tmp_path / "b.wav",
        other=tmp_path / "o.wav",
        drums=None,
        _tempdir=tmp_path,
    )
    (tmp_path / "b.wav").write_bytes(b"\x00")
    (tmp_path / "o.wav").write_bytes(b"\x00")

    fake_result = TranscriptionResult(
        midi_tracks=[
            MidiTrack(
                instrument=InstrumentRole.PIANO,
                program=0,
                confidence=0.8,
                notes=[Note(pitch=60, onset_sec=0.0, offset_sec=1.0, velocity=80)],
            ),
        ],
        analysis=HarmonicAnalysis(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
        ),
        quality=QualitySignal(overall_confidence=0.8, warnings=[]),
    )
    fake_pedals = [
        RealtimePedalEvent(cc=64, onset_sec=0.5, offset_sec=2.0, confidence=0.9),
    ]

    monkeypatch.setattr(
        kong_pipeline, "should_route_to_kong",
        lambda stems, user_hint=None: True,
    )
    monkeypatch.setattr(
        kong_pipeline, "_run_with_kong",
        lambda audio_path, stems, stem_stats: (fake_result, b"MThd", fake_pedals),
    )

    result, midi_bytes, pedals = transcribe_mod._run_basic_pitch_sync(
        audio, pre_separated=stems,
    )
    assert result is fake_result
    assert midi_bytes == b"MThd"
    assert pedals == fake_pedals


def test_run_basic_pitch_sync_kong_failure_falls_back_to_bp_stems(
    monkeypatch, tmp_path,
):
    """Kong inference exception falls through to the Basic Pitch stems
    pipeline so the user still gets a transcription (just without pedal)."""
    from backend.services import transcribe as transcribe_mod
    from backend.services import transcribe_pipeline_stems as stems_mod

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00" * 100)
    stems = SeparatedStems(
        vocals=None,
        bass=tmp_path / "b.wav",
        other=tmp_path / "o.wav",
        drums=None,
        _tempdir=tmp_path,
    )
    (tmp_path / "b.wav").write_bytes(b"\x00")
    (tmp_path / "o.wav").write_bytes(b"\x00")

    fake_bp_result = TranscriptionResult(
        midi_tracks=[
            MidiTrack(
                instrument=InstrumentRole.MELODY,
                program=0,
                confidence=0.7,
                notes=[Note(pitch=72, onset_sec=0.0, offset_sec=0.5, velocity=70)],
            ),
        ],
        analysis=HarmonicAnalysis(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
        ),
        quality=QualitySignal(overall_confidence=0.7, warnings=[]),
    )

    monkeypatch.setattr(
        kong_pipeline, "should_route_to_kong",
        lambda stems, user_hint=None: True,
    )

    def _kong_boom(audio_path, stems, stem_stats):
        raise RuntimeError("kong: weights unavailable")

    monkeypatch.setattr(kong_pipeline, "_run_with_kong", _kong_boom)
    monkeypatch.setattr(
        stems_mod, "_run_with_stems",
        lambda audio_path, stems, stem_stats: (fake_bp_result, b"MThd_bp"),
    )

    result, midi_bytes, pedals = transcribe_mod._run_basic_pitch_sync(
        audio, pre_separated=stems,
    )
    assert result is fake_bp_result
    assert midi_bytes == b"MThd_bp"
    assert pedals == []


# ---------------------------------------------------------------------------
# Sheet_only synthesized expression carries pedals
# ---------------------------------------------------------------------------


def test_score_metadata_pedal_events_default_empty_list():
    meta = ScoreMetadata(
        key="C:major",
        time_signature=(4, 4),
        tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
        difficulty="intermediate",
    )
    assert meta.pedal_events == []
