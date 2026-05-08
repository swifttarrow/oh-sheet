"""Phase 8 tests — AMT-APC cover-mode pipeline + dispatcher routing.

Covers:

* ``transcribe_amt_apc.run_amt_apc`` raises ``ImportError`` when the
  optional ``amt_apc`` package isn't installed (so the dispatcher in
  ``transcribe.py`` falls back to the Kong/BP path).
* ``should_route_to_amt_apc`` honors the gating rules: kill switch,
  pop_cover variant override, "cover" user_hint, and the
  ``amt_apc_enabled`` flag.
* ``_run_basic_pitch_sync`` invokes the AMT-APC pipeline first when
  cover mode is requested, and falls through to Kong / BP on failure
  so cover-mode jobs still produce a transcription.
* ``_cover_score_from_transcription`` (in jobs/runner.py) splits notes
  onto staves at middle C and converts seconds → beats correctly.

Real AMT-APC inference is gated behind an optional dep + weight
download, so we monkeypatch the model interface and exercise the
wrapping / routing logic only.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.config import settings
from backend.contracts import (
    HarmonicAnalysis,
    InstrumentRole,
    MidiTrack,
    Note,
    QualitySignal,
    TempoMapEntry,
    TranscriptionResult,
)
from backend.services import transcribe_amt_apc as amt_mod
from backend.services import transcribe_pipeline_amt_apc as amt_pipeline
from backend.services.stem_separation import SeparatedStems

# ---------------------------------------------------------------------------
# AmtApcStats wiring
# ---------------------------------------------------------------------------


def test_amt_apc_stats_emits_zero_notes_warning():
    stats = amt_mod.AmtApcStats(model_id="amt_apc", note_count=0)
    out = stats.as_warnings()
    assert any("zero notes" in w for w in out)


def test_amt_apc_stats_skipped_suppresses_zero_notes_warning():
    stats = amt_mod.AmtApcStats(skipped=True)
    assert "zero notes" not in " ".join(stats.as_warnings())


# ---------------------------------------------------------------------------
# AMT-APC wrapper: ImportError when optional dep is missing
# ---------------------------------------------------------------------------


def test_run_amt_apc_raises_import_error_when_dep_missing(monkeypatch, tmp_path):
    """The dispatcher in transcribe.py catches ImportError and falls back to
    the Kong/BP path. Verify the wrapper actually raises it."""
    monkeypatch.setattr(amt_mod, "_AMT_APC_MODEL", None)

    def fake_load():
        raise ImportError("amt_apc not installed")

    monkeypatch.setattr(amt_mod, "_load_amt_apc", fake_load)
    audio = tmp_path / "fake.wav"
    audio.write_bytes(b"\x00" * 100)
    with pytest.raises(ImportError):
        amt_mod.run_amt_apc(audio)


# ---------------------------------------------------------------------------
# Routing heuristic (should_route_to_amt_apc)
# ---------------------------------------------------------------------------


def test_should_route_to_amt_apc_disabled_by_kill_switch(monkeypatch):
    monkeypatch.setattr(settings, "amt_apc_enabled", False)
    assert amt_pipeline.should_route_to_amt_apc(variant="pop_cover") is False
    assert amt_pipeline.should_route_to_amt_apc(user_hint="cover") is False


def test_should_route_to_amt_apc_pop_cover_variant_triggers(monkeypatch):
    monkeypatch.setattr(settings, "amt_apc_enabled", True)
    assert amt_pipeline.should_route_to_amt_apc(variant="pop_cover") is True


def test_should_route_to_amt_apc_user_hint_triggers(monkeypatch):
    monkeypatch.setattr(settings, "amt_apc_enabled", True)
    assert amt_pipeline.should_route_to_amt_apc(user_hint="cover") is True


def test_should_route_to_amt_apc_unrelated_variants_skip(monkeypatch):
    monkeypatch.setattr(settings, "amt_apc_enabled", True)
    assert amt_pipeline.should_route_to_amt_apc(variant="audio_upload") is False
    assert amt_pipeline.should_route_to_amt_apc(variant="midi_upload") is False
    assert amt_pipeline.should_route_to_amt_apc(user_hint="piano") is False


# ---------------------------------------------------------------------------
# Dispatcher tail: AMT-APC path returns 3-tuple in transcribe sync
# ---------------------------------------------------------------------------


def _stems_with_files(tmp_path: Path) -> SeparatedStems:
    bass = tmp_path / "b.wav"
    other = tmp_path / "o.wav"
    bass.write_bytes(b"\x00" * 64)
    other.write_bytes(b"\x00" * 64)
    return SeparatedStems(
        vocals=None, bass=bass, other=other, drums=None, _tempdir=tmp_path,
    )


def test_run_basic_pitch_sync_amt_apc_path_runs_for_pop_cover(monkeypatch, tmp_path):
    """When variant=='pop_cover', the dispatcher invokes AMT-APC first
    and returns its result. Stems are passed through but Kong is never
    consulted."""
    from backend.services import transcribe as transcribe_mod

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00" * 100)
    stems = _stems_with_files(tmp_path)

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

    monkeypatch.setattr(settings, "amt_apc_enabled", True)
    monkeypatch.setattr(
        amt_pipeline, "_run_with_amt_apc",
        lambda audio_path, stems, stem_stats: (fake_result, b"MThd_amt"),
    )

    result, midi_bytes, pedals = transcribe_mod._run_basic_pitch_sync(
        audio, pre_separated=stems, variant="pop_cover",
    )
    assert result is fake_result
    assert midi_bytes == b"MThd_amt"
    assert pedals == []


def test_run_basic_pitch_sync_amt_apc_failure_falls_back_to_kong_or_bp(
    monkeypatch, tmp_path,
):
    """AMT-APC inference exception falls through so the user still gets a
    transcription. Verifies the cover-mode → faithful fallback chain."""
    from backend.services import transcribe as transcribe_mod
    from backend.services import transcribe_pipeline_kong as kong_pipeline
    from backend.services import transcribe_pipeline_stems as stems_mod

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00" * 100)
    stems = _stems_with_files(tmp_path)

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

    monkeypatch.setattr(settings, "amt_apc_enabled", True)

    def _amt_boom(audio_path, stems, stem_stats):
        raise RuntimeError("amt_apc: weights unavailable")

    monkeypatch.setattr(amt_pipeline, "_run_with_amt_apc", _amt_boom)
    # Disable Kong so we land on BP stems (ensures fallback chain works).
    monkeypatch.setattr(
        kong_pipeline, "should_route_to_kong",
        lambda stems, user_hint=None: False,
    )
    monkeypatch.setattr(
        stems_mod, "_run_with_stems",
        lambda audio_path, stems, stem_stats: (fake_bp_result, b"MThd_bp"),
    )

    result, midi_bytes, pedals = transcribe_mod._run_basic_pitch_sync(
        audio, pre_separated=stems, variant="pop_cover",
    )
    assert result is fake_bp_result
    assert midi_bytes == b"MThd_bp"
    assert pedals == []


def test_run_basic_pitch_sync_skips_amt_apc_when_not_cover_mode(
    monkeypatch, tmp_path,
):
    """Non-cover variants don't invoke AMT-APC, even with the kill switch on."""
    from backend.services import transcribe as transcribe_mod
    from backend.services import transcribe_pipeline_kong as kong_pipeline
    from backend.services import transcribe_pipeline_stems as stems_mod

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00" * 100)
    stems = _stems_with_files(tmp_path)

    monkeypatch.setattr(settings, "amt_apc_enabled", True)

    amt_called = {"flag": False}

    def _amt_should_not_run(audio_path, stems, stem_stats):
        amt_called["flag"] = True
        raise AssertionError("AMT-APC should not run on faithful path")

    monkeypatch.setattr(amt_pipeline, "_run_with_amt_apc", _amt_should_not_run)
    monkeypatch.setattr(
        kong_pipeline, "should_route_to_kong",
        lambda stems, user_hint=None: False,
    )

    fake_bp = TranscriptionResult(
        midi_tracks=[
            MidiTrack(
                instrument=InstrumentRole.MELODY,
                program=0,
                confidence=0.7,
                notes=[Note(pitch=60, onset_sec=0.0, offset_sec=1.0, velocity=80)],
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
        stems_mod, "_run_with_stems",
        lambda audio_path, stems, stem_stats: (fake_bp, b"MThd_bp"),
    )

    transcribe_mod._run_basic_pitch_sync(
        audio, pre_separated=stems, variant="audio_upload",
    )
    assert amt_called["flag"] is False


# ---------------------------------------------------------------------------
# _cover_score_from_transcription — runner-side conversion helper
# ---------------------------------------------------------------------------


def test_cover_score_splits_notes_at_middle_c():
    """Pitches < 60 land on the left hand; pitches >= 60 land on the right.
    The cover model emits arrangement-ready notes, so the split is just
    a staff cut at middle C — no overlap resolution / voice assignment."""
    from backend.jobs.runner import _cover_score_from_transcription

    txr = TranscriptionResult(
        midi_tracks=[
            MidiTrack(
                instrument=InstrumentRole.PIANO,
                program=0,
                confidence=0.8,
                notes=[
                    Note(pitch=48, onset_sec=0.0, offset_sec=1.0, velocity=70),  # LH
                    Note(pitch=60, onset_sec=0.0, offset_sec=1.0, velocity=80),  # RH (boundary)
                    Note(pitch=72, onset_sec=1.0, offset_sec=2.0, velocity=90),  # RH
                    Note(pitch=36, onset_sec=1.0, offset_sec=2.0, velocity=65),  # LH
                ],
            ),
        ],
        analysis=HarmonicAnalysis(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=60.0)],  # 1 bps
        ),
        quality=QualitySignal(overall_confidence=0.8, warnings=[]),
    )
    score = _cover_score_from_transcription(txr)
    rh_pitches = sorted(n.pitch for n in score.right_hand)
    lh_pitches = sorted(n.pitch for n in score.left_hand)
    assert rh_pitches == [60, 72]
    assert lh_pitches == [36, 48]


def test_cover_score_converts_seconds_to_beats():
    """Onsets/offsets are converted via the analysis tempo_map. At 60 BPM,
    1 sec == 1 beat — verifies sec_to_beat plumbing is correct."""
    from backend.jobs.runner import _cover_score_from_transcription

    txr = TranscriptionResult(
        midi_tracks=[
            MidiTrack(
                instrument=InstrumentRole.PIANO,
                program=0,
                confidence=0.8,
                notes=[
                    Note(pitch=72, onset_sec=2.0, offset_sec=4.5, velocity=80),
                ],
            ),
        ],
        analysis=HarmonicAnalysis(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=60.0)],
        ),
        quality=QualitySignal(overall_confidence=0.8, warnings=[]),
    )
    score = _cover_score_from_transcription(txr)
    assert len(score.right_hand) == 1
    note = score.right_hand[0]
    assert note.onset_beat == 2.0
    assert note.duration_beat == 2.5


def test_cover_score_carries_analysis_metadata():
    """Key, time signature, downbeats, and chord_symbols flow from
    HarmonicAnalysis into ScoreMetadata so the engraver still gets
    the audio-side analysis even though arrange is skipped."""
    from backend.contracts import RealtimeChordEvent
    from backend.jobs.runner import _cover_score_from_transcription

    txr = TranscriptionResult(
        midi_tracks=[
            MidiTrack(
                instrument=InstrumentRole.PIANO,
                program=0,
                confidence=0.8,
                notes=[Note(pitch=60, onset_sec=0.0, offset_sec=1.0, velocity=80)],
            ),
        ],
        analysis=HarmonicAnalysis(
            key="A:minor",
            time_signature=(3, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=90.0)],
            downbeats=[0.0, 2.0, 4.0],
            chords=[
                RealtimeChordEvent(
                    time_sec=0.0, duration_sec=2.0,
                    label="A:min", root=9, confidence=0.9,
                ),
            ],
        ),
        quality=QualitySignal(overall_confidence=0.8, warnings=[]),
    )
    score = _cover_score_from_transcription(txr)
    assert score.metadata.key == "A:minor"
    assert score.metadata.time_signature == (3, 4)
    assert score.metadata.downbeats == [0.0, 2.0, 4.0]
    assert len(score.metadata.chord_symbols) == 1
    assert score.metadata.chord_symbols[0].label == "A:min"
    # AMT-APC does not emit pedal — pedal_events stay empty.
    assert score.metadata.pedal_events == []


def test_cover_score_handles_empty_tracks():
    """Edge case — zero notes produces an empty score, not a crash."""
    from backend.jobs.runner import _cover_score_from_transcription

    txr = TranscriptionResult(
        midi_tracks=[],
        analysis=HarmonicAnalysis(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
        ),
        quality=QualitySignal(overall_confidence=0.1, warnings=[]),
    )
    score = _cover_score_from_transcription(txr)
    assert score.right_hand == []
    assert score.left_hand == []
