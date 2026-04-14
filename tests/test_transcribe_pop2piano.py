"""Unit tests for the Pop2Piano transcription path.

Pop2Piano's model + processor are heavy (~1 GB) and require
essentia + torch + transformers, so these tests monkeypatch the
inference layer and exercise:

* ``run_pop2piano`` → pretty_midi → NoteEvent conversion
* ``_run_with_pop2piano`` pipeline wiring (post-processing chain)
* ``_run_basic_pitch_sync`` dispatch: Pop2Piano → Demucs+BP fallback
* Graceful fallback when Pop2Piano deps are missing
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

pretty_midi = pytest.importorskip("pretty_midi")

from backend.contracts import InstrumentRole, TranscriptionResult  # noqa: E402
from backend.services import transcribe as transcribe_mod  # noqa: E402
from backend.services import (  # noqa: E402
    transcribe_pipeline_pop2piano as p2p_pipeline_mod,
)
from backend.services.transcribe_pop2piano import (  # noqa: E402
    Pop2PianoStats,
    _pretty_midi_to_note_events,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_pretty_midi_with_notes(notes: list[tuple[int, float, float, int]]) -> Any:
    """Build a pretty_midi.PrettyMIDI with the given notes.

    Each tuple is (pitch, start, end, velocity).
    """
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    for pitch, start, end, velocity in notes:
        instrument.notes.append(
            pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)
        )
    pm.instruments.append(instrument)
    return pm


def _fake_run_pop2piano(audio_path: Path):
    """Fake Pop2Piano inference returning a simple 4-note piano passage."""
    pm = _make_pretty_midi_with_notes([
        (60, 0.0, 0.5, 80),   # C4
        (64, 0.5, 1.0, 90),   # E4
        (67, 1.0, 1.5, 85),   # G4
        (72, 1.5, 2.0, 95),   # C5
    ])
    events = _pretty_midi_to_note_events(pm)
    stats = Pop2PianoStats(
        model_id="sweetcocoa/pop2piano",
        note_count=len(events),
        audio_duration_sec=2.0,
    )
    return events, pm, stats


# ---------------------------------------------------------------------------
# _pretty_midi_to_note_events
# ---------------------------------------------------------------------------

def test_pretty_midi_to_note_events_basic():
    """Converts pretty_midi notes to NoteEvent tuples correctly."""
    pm = _make_pretty_midi_with_notes([
        (60, 0.0, 1.0, 100),
        (72, 0.5, 1.5, 64),
    ])
    events = _pretty_midi_to_note_events(pm)

    assert len(events) == 2
    # Sorted by (onset, pitch)
    assert events[0][2] == 60  # pitch
    assert events[1][2] == 72

    # Check amplitude = velocity / 127
    assert abs(events[0][3] - 100 / 127) < 0.01
    assert abs(events[1][3] - 64 / 127) < 0.01

    # Check timing
    assert events[0][0] == 0.0   # start
    assert events[0][1] == 1.0   # end
    assert events[1][0] == 0.5
    assert events[1][1] == 1.5


def test_pretty_midi_to_note_events_empty():
    """Empty pretty_midi → empty event list."""
    pm = pretty_midi.PrettyMIDI()
    events = _pretty_midi_to_note_events(pm)
    assert events == []


def test_pretty_midi_to_note_events_multi_instrument():
    """Notes from all instruments are merged and sorted."""
    pm = pretty_midi.PrettyMIDI()
    inst1 = pretty_midi.Instrument(program=0)
    inst1.notes.append(pretty_midi.Note(velocity=80, pitch=72, start=1.0, end=2.0))
    inst2 = pretty_midi.Instrument(program=0)
    inst2.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=1.0))
    pm.instruments.extend([inst1, inst2])

    events = _pretty_midi_to_note_events(pm)
    assert len(events) == 2
    assert events[0][2] == 60  # lower onset first
    assert events[1][2] == 72


# ---------------------------------------------------------------------------
# _run_with_pop2piano pipeline
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_pop2piano_pipeline(monkeypatch):
    """Stub out the heavy dependencies for _run_with_pop2piano."""
    monkeypatch.setattr(p2p_pipeline_mod, "run_pop2piano", _fake_run_pop2piano)
    monkeypatch.setattr(
        p2p_pipeline_mod, "tempo_map_from_audio_path", lambda _path, **_kw: None,
    )

    from backend.services import transcribe_audio as audio_mod
    monkeypatch.setattr(audio_mod, "_audio_duration_sec", lambda _path: 2.0)

    # Key/meter estimation returns defaults
    from backend.services.key_estimation import KeyEstimationStats, MeterEstimationStats
    monkeypatch.setattr(
        audio_mod,
        "_maybe_analyze_key_and_meter",
        lambda _path, **_kw: (
            "C:major",
            (4, 4),
            KeyEstimationStats(key_label="C:major", confidence=0.8, skipped=False),
            MeterEstimationStats(time_signature=(4, 4), confidence=0.8, skipped=False),
        ),
    )

    # Chord recognition — skip
    def fake_recognize_chords(_path, **_kwargs):
        from backend.services.chord_recognition import ChordRecognitionStats
        return [], ChordRecognitionStats(skipped=True)

    monkeypatch.setattr(p2p_pipeline_mod, "recognize_chords", fake_recognize_chords)


def test_run_with_pop2piano_produces_valid_result(stub_pop2piano_pipeline, tmp_path):
    """The Pop2Piano pipeline returns a valid TranscriptionResult."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"\x00")

    result, midi_bytes = p2p_pipeline_mod._run_with_pop2piano(audio)

    assert isinstance(result, TranscriptionResult)
    assert len(result.midi_tracks) >= 1
    total_notes = sum(len(t.notes) for t in result.midi_tracks)
    assert total_notes == 4

    # All notes should end up in PIANO since contour is None
    # (melody/bass extraction skips)
    assert result.midi_tracks[0].instrument == InstrumentRole.PIANO

    # Pop2Piano banner should be in warnings
    assert any("Pop2Piano" in w for w in result.quality.warnings)
    assert not any("Basic Pitch baseline" in w for w in result.quality.warnings)


def test_run_with_pop2piano_returns_midi_bytes(stub_pop2piano_pipeline, tmp_path):
    """The pipeline returns serialized MIDI bytes for blob storage."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"\x00")

    _result, midi_bytes = p2p_pipeline_mod._run_with_pop2piano(audio)
    assert midi_bytes is not None
    assert len(midi_bytes) > 0


# ---------------------------------------------------------------------------
# _run_basic_pitch_sync dispatch
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_dispatch_pop2piano(monkeypatch, stub_pop2piano_pipeline):
    """Make _run_basic_pitch_sync dispatch to the (stubbed) Pop2Piano path."""
    monkeypatch.setattr("backend.config.settings.pop2piano_enabled", True)


def test_dispatch_uses_pop2piano_when_enabled(stub_dispatch_pop2piano, tmp_path):
    """_run_basic_pitch_sync dispatches to Pop2Piano when enabled."""
    audio = tmp_path / "test.mp3"
    audio.write_bytes(b"\x00")

    result, _midi_bytes = transcribe_mod._run_basic_pitch_sync(audio)

    assert isinstance(result, TranscriptionResult)
    assert any("Pop2Piano" in w for w in result.quality.warnings)


def _make_fake_fallback_result() -> TranscriptionResult:
    """Build a distinguishable TranscriptionResult for fallback assertions."""
    from backend.contracts import (
        HarmonicAnalysis,
        MidiTrack,
        Note,
        QualitySignal,
        TempoMapEntry,
    )

    return TranscriptionResult(
        midi_tracks=[
            MidiTrack(
                notes=[Note(pitch=42, onset_sec=0.0, offset_sec=0.5, velocity=64)],
                instrument=InstrumentRole.PIANO,
                program=0,
                confidence=0.5,
            ),
        ],
        analysis=HarmonicAnalysis(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
        ),
        quality=QualitySignal(
            overall_confidence=0.5,
            warnings=["fallback-marker"],
        ),
    )


def test_dispatch_falls_back_on_import_error(monkeypatch, tmp_path):
    """Pop2Piano ImportError falls back to the Demucs+BP path."""
    monkeypatch.setattr("backend.config.settings.pop2piano_enabled", True)
    monkeypatch.setattr("backend.config.settings.demucs_enabled", False)

    from backend.services import transcribe_pipeline_pop2piano as p2p_mod
    monkeypatch.setattr(
        p2p_mod,
        "_run_with_pop2piano",
        MagicMock(side_effect=ImportError("no essentia")),
    )

    from backend.services import transcribe_pipeline_single as single_mod
    fake_result = _make_fake_fallback_result()
    monkeypatch.setattr(
        single_mod,
        "_run_without_stems",
        lambda _audio, _stats: (fake_result, None),
    )

    result, _ = transcribe_mod._run_basic_pitch_sync(tmp_path / "test.mp3")
    assert result is fake_result


def test_dispatch_falls_back_on_runtime_error(monkeypatch, tmp_path):
    """Pop2Piano runtime crash falls back to the Demucs+BP path."""
    monkeypatch.setattr("backend.config.settings.pop2piano_enabled", True)
    monkeypatch.setattr("backend.config.settings.demucs_enabled", False)

    from backend.services import transcribe_pipeline_pop2piano as p2p_mod
    monkeypatch.setattr(
        p2p_mod,
        "_run_with_pop2piano",
        MagicMock(side_effect=RuntimeError("CUDA OOM")),
    )

    from backend.services import transcribe_pipeline_single as single_mod
    fake_result = _make_fake_fallback_result()
    monkeypatch.setattr(
        single_mod,
        "_run_without_stems",
        lambda _audio, _stats: (fake_result, None),
    )

    result, _ = transcribe_mod._run_basic_pitch_sync(tmp_path / "test.mp3")
    assert result is fake_result


def test_dispatch_skips_pop2piano_when_disabled(monkeypatch, tmp_path):
    """When pop2piano_enabled=False, goes straight to Demucs+BP path."""
    monkeypatch.setattr("backend.config.settings.pop2piano_enabled", False)
    monkeypatch.setattr("backend.config.settings.demucs_enabled", False)

    from backend.services import transcribe_pipeline_single as single_mod
    fake_result = _make_fake_fallback_result()
    monkeypatch.setattr(
        single_mod,
        "_run_without_stems",
        lambda _audio, _stats: (fake_result, None),
    )

    from backend.services import transcribe_pipeline_pop2piano as p2p_mod
    p2p_mock = MagicMock(side_effect=AssertionError("should not be called"))
    monkeypatch.setattr(p2p_mod, "_run_with_pop2piano", p2p_mock)

    result, _ = transcribe_mod._run_basic_pitch_sync(tmp_path / "test.mp3")
    assert result is fake_result
    p2p_mock.assert_not_called()
