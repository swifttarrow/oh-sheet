"""Unit tests for the stems-path orchestration in ``TranscribeService``.

The real Basic Pitch inference is too heavy to exercise in the unit
suite (see ``test_stem_separation.py`` for the same rationale applied
to Demucs), so these tests monkeypatch ``_basic_pitch_single_pass``
and the audio-facing helpers (``tempo_map_from_audio_path``,
``recognize_chords``) to drive ``_run_with_stems`` with deterministic
fakes.

Covered behaviors:

* ``_basic_pitch_single_pass(path, keep_model_output=False)`` drops
  the contour tensor before returning — the review pointed out that
  the stems path kept three live ``model_output`` dicts alive
  simultaneously, and this is the guardrail that proves we fixed it.
* ``_run_with_stems`` parallel path succeeds, populates all three
  per-role event lists, and never consults ``model_output``
  downstream (we prove this by passing an empty dict — a real
  ``model_output.get("note")`` would raise on a plain dict, but the
  ``events_by_role`` guard makes the branch unreachable).
* One stem raising inside the worker does not sink the other two —
  exception isolation matches the pre-parallel behavior.
* ``demucs_parallel_stems=False`` produces the same result as the
  parallel path (serial fallback is a 1:1 substitute).

We do **not** test Basic Pitch thread-safety directly here — ONNX
Runtime / CoreML session thread-safety is an upstream contract and
would require a real model to exercise. The parallel-path test just
verifies that the orchestrator correctly submits jobs, gathers
results in submission order, and survives per-worker exceptions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pretty_midi
import pytest

from backend.contracts import InstrumentRole
from backend.services import transcribe as transcribe_mod
from backend.services.stem_separation import SeparatedStems, StemSeparationStats
from backend.services.transcribe import (
    _BasicPitchPass,
    _run_with_stems,
)
from backend.services.transcription_cleanup import CleanupStats

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_pass(
    label: str,
    *,
    model_output: dict[str, Any] | None = None,
) -> _BasicPitchPass:
    """Build a fake ``_BasicPitchPass`` shaped like the real thing.

    ``cleaned_events`` carries one note per label so the assertions
    can tell the three stems apart. The note amplitudes (0.5) keep
    ``overall_conf`` away from both saturation rails so a confidence
    assertion stays meaningful if anything downstream uses it.
    """
    # (start_sec, end_sec, pitch_midi, amplitude, pitch_bend)
    pitch = {"vocals": 72, "bass": 36, "other": 60}[label]
    note = (0.0, 0.5, pitch, 0.5, None)
    pm = pretty_midi.PrettyMIDI()
    return _BasicPitchPass(
        cleaned_events=[note],
        model_output=model_output if model_output is not None else {},
        midi_data=pm,
        preprocess_stats=None,
        cleanup_stats=CleanupStats(input_count=1, output_count=1),
    )


def _make_stems(tmp_path: Path) -> SeparatedStems:
    """A ``SeparatedStems`` with four placeholder wav files.

    The files don't need to be valid audio — our monkeypatched
    ``_basic_pitch_single_pass`` never actually opens them. The real
    paths exist so ``Path`` attribute access is happy.
    """
    tempdir = tmp_path / "stems"
    tempdir.mkdir()
    paths = {}
    for name in ("vocals", "bass", "other", "drums"):
        p = tempdir / f"{name}.wav"
        p.write_bytes(b"\x00")
        paths[name] = p
    return SeparatedStems(
        vocals=paths["vocals"],
        bass=paths["bass"],
        other=paths["other"],
        drums=paths["drums"],
        _tempdir=tempdir,
    )


@pytest.fixture
def stub_audio_helpers(monkeypatch):
    """Silence the audio-only stages — tempo + chord recog.

    Both are invoked unconditionally on the stems path (gated only
    by ``chord_recognition_enabled``), and both hit librosa on a
    real file. We stub them so the test stays pure.
    """
    monkeypatch.setattr(
        transcribe_mod, "tempo_map_from_audio_path", lambda _path: None
    )

    def fake_recognize_chords(_path, **_kwargs):
        from backend.services.chord_recognition import ChordRecognitionStats
        return [], ChordRecognitionStats(skipped=True)

    monkeypatch.setattr(transcribe_mod, "recognize_chords", fake_recognize_chords)


# ---------------------------------------------------------------------------
# _basic_pitch_single_pass — keep_model_output kwarg
# ---------------------------------------------------------------------------

def test_basic_pitch_single_pass_drops_model_output_when_asked(monkeypatch, tmp_path):
    """``keep_model_output=False`` replaces the dict contents with nothing.

    We stub ``predict`` and the cleanup helpers so we don't need a
    real Basic Pitch model. The assertion is specifically that the
    returned ``_BasicPitchPass.model_output`` is empty (``.clear()``
    has run) — this is the "stems path doesn't keep the contour
    tensor alive" guarantee the review asked for.
    """
    import numpy as np
    big_contour = np.zeros((100, 88), dtype=np.float32)
    mock_output = {
        "note": np.zeros((100, 88), dtype=np.float32),
        "onset": np.zeros((100, 88), dtype=np.float32),
        "contour": big_contour,
    }

    class _FakePM:
        pass

    # Stub basic_pitch.inference.predict to return our fake output
    # without touching disk. Basic Pitch's package is installed in
    # this environment so we can import + patch directly.
    import basic_pitch.inference as bp_inf
    import basic_pitch.note_creation as bp_nc
    monkeypatch.setattr(
        bp_inf,
        "predict",
        lambda *_args, **_kwargs: (mock_output, _FakePM(), []),
    )
    monkeypatch.setattr(
        bp_nc,
        "note_events_to_midi",
        lambda _events, *_a, **_kw: _FakePM(),
    )
    # Short-circuit the preprocess stage — the real one reads audio.
    from backend.config import settings
    monkeypatch.setattr(settings, "audio_preprocess_enabled", False)
    # Skip the model load — ``predict`` is mocked so the model is
    # never consulted, but ``_load_basic_pitch_model`` would still
    # try to import and build the real one.
    monkeypatch.setattr(transcribe_mod, "_load_basic_pitch_model", lambda: object())

    audio_path = tmp_path / "fake.wav"
    audio_path.write_bytes(b"\x00")

    pass_kept = transcribe_mod._basic_pitch_single_pass(
        audio_path, keep_model_output=True,
    )
    assert pass_kept.model_output is mock_output  # identity: no copy
    assert "contour" in pass_kept.model_output

    # Reset — the previous call called .clear() paths are exercised
    # only when keep_model_output=False, but the same mock_output
    # dict was mutated above if we hit the clear path. Rebuild it.
    mock_output2 = {
        "note": np.zeros((10, 88), dtype=np.float32),
        "onset": np.zeros((10, 88), dtype=np.float32),
        "contour": np.zeros((10, 88), dtype=np.float32),
    }
    monkeypatch.setattr(
        bp_inf,
        "predict",
        lambda *_args, **_kwargs: (mock_output2, _FakePM(), []),
    )
    pass_dropped = transcribe_mod._basic_pitch_single_pass(
        audio_path, keep_model_output=False,
    )
    assert pass_dropped.model_output == {}  # contour tensor unreferenced
    # And the original dict was cleared in place (local name), so
    # nothing held a reference to the contour array.
    assert mock_output2 == {}


# ---------------------------------------------------------------------------
# _run_with_stems — parallel happy path
# ---------------------------------------------------------------------------

def test_run_with_stems_parallel_populates_all_three_roles(
    monkeypatch, tmp_path, stub_audio_helpers,
):
    """Parallel path: three stems in, three roles out."""
    stems = _make_stems(tmp_path)
    stem_stats = StemSeparationStats(
        model_name="htdemucs",
        device="cpu",
        wall_time_sec=0.0,
        stems_written=["vocals", "bass", "other", "drums"],
    )

    call_log: list[str] = []

    def fake_pass(stem_path: Path, *, keep_model_output: bool = True):
        # The stems path must always pass keep_model_output=False so
        # the contour tensor can be GC'd as soon as each stem
        # finishes. The review explicitly called this out.
        assert keep_model_output is False, (
            "stems path must call _basic_pitch_single_pass with "
            "keep_model_output=False so the contour tensor doesn't "
            "pin memory across all three concurrent passes"
        )
        label = stem_path.stem
        call_log.append(label)
        return _make_pass(label)

    monkeypatch.setattr(transcribe_mod, "_basic_pitch_single_pass", fake_pass)

    from backend.config import settings
    monkeypatch.setattr(settings, "demucs_parallel_stems", True)
    monkeypatch.setattr(settings, "chord_recognition_enabled", False)

    audio_path = tmp_path / "mix.wav"
    audio_path.write_bytes(b"\x00")

    result, _midi_bytes = _run_with_stems(audio_path, stems, stem_stats)

    # All three consumer stems were invoked — order is
    # submission-order (vocals/bass/other) but we don't assert on
    # order because ThreadPoolExecutor.map() schedules eagerly and
    # the test would be flaky on that axis.
    assert set(call_log) == {"vocals", "bass", "other"}

    roles = {t.instrument for t in result.midi_tracks}
    assert InstrumentRole.MELODY in roles
    assert InstrumentRole.BASS in roles
    assert InstrumentRole.CHORDS in roles
    # No PIANO fallback — the stems path routes every note directly.
    assert InstrumentRole.PIANO not in roles


# ---------------------------------------------------------------------------
# _run_with_stems — one stem raising must not poison the others
# ---------------------------------------------------------------------------

def test_run_with_stems_isolates_per_stem_failures(
    monkeypatch, tmp_path, stub_audio_helpers,
):
    """One worker raising leaves the other two roles intact."""
    stems = _make_stems(tmp_path)
    stem_stats = StemSeparationStats(stems_written=["vocals", "bass", "other", "drums"])

    def fake_pass(stem_path: Path, *, keep_model_output: bool = True):
        if stem_path.stem == "bass":
            raise RuntimeError("simulated BP OOM on bass stem")
        return _make_pass(stem_path.stem)

    monkeypatch.setattr(transcribe_mod, "_basic_pitch_single_pass", fake_pass)

    from backend.config import settings
    monkeypatch.setattr(settings, "demucs_parallel_stems", True)
    monkeypatch.setattr(settings, "chord_recognition_enabled", False)

    audio_path = tmp_path / "mix.wav"
    audio_path.write_bytes(b"\x00")

    result, _ = _run_with_stems(audio_path, stems, stem_stats)

    roles = {t.instrument for t in result.midi_tracks}
    assert InstrumentRole.MELODY in roles   # vocals survived
    assert InstrumentRole.CHORDS in roles   # other survived
    assert InstrumentRole.BASS not in roles  # bass worker raised


# ---------------------------------------------------------------------------
# _run_with_stems — serial fallback matches parallel output
# ---------------------------------------------------------------------------

def test_run_with_stems_serial_mode_matches_parallel(
    monkeypatch, tmp_path, stub_audio_helpers,
):
    """``demucs_parallel_stems=False`` still produces the same roles.

    This is the escape hatch for debugging single-thread traces.
    It should be a behavioral no-op — only the scheduling differs.
    """
    stems = _make_stems(tmp_path)
    stem_stats = StemSeparationStats(stems_written=["vocals", "bass", "other", "drums"])

    def fake_pass(stem_path: Path, *, keep_model_output: bool = True):
        return _make_pass(stem_path.stem)

    monkeypatch.setattr(transcribe_mod, "_basic_pitch_single_pass", fake_pass)

    from backend.config import settings
    monkeypatch.setattr(settings, "demucs_parallel_stems", False)
    monkeypatch.setattr(settings, "chord_recognition_enabled", False)

    audio_path = tmp_path / "mix.wav"
    audio_path.write_bytes(b"\x00")

    result, _ = _run_with_stems(audio_path, stems, stem_stats)

    roles = {t.instrument for t in result.midi_tracks}
    assert roles == {InstrumentRole.MELODY, InstrumentRole.BASS, InstrumentRole.CHORDS}


# ---------------------------------------------------------------------------
# _run_with_stems — all stems empty should fall back to single-mix
# ---------------------------------------------------------------------------

def test_run_with_stems_all_empty_falls_back_to_single_mix(
    monkeypatch, tmp_path, stub_audio_helpers,
):
    """If every stem returns zero notes, fall back to the legacy pipeline.

    This is the existing ``all stems empty`` guard — we verify the
    parallel refactor didn't accidentally remove it.
    """
    stems = _make_stems(tmp_path)
    stem_stats = StemSeparationStats(stems_written=["vocals", "bass", "other", "drums"])

    def fake_pass(stem_path: Path, *, keep_model_output: bool = True):
        p = _make_pass(stem_path.stem)
        p.cleaned_events = []  # empty — forces the fallback branch
        return p

    monkeypatch.setattr(transcribe_mod, "_basic_pitch_single_pass", fake_pass)

    # Intercept the fallback so we can verify it was called without
    # actually running the single-mix Basic Pitch pipeline.
    called = {"fallback": False}

    def fake_single_mix(audio_path, stem_stats):
        called["fallback"] = True
        from backend.contracts import (
            SCHEMA_VERSION,
            HarmonicAnalysis,
            QualitySignal,
            TempoMapEntry,
            TranscriptionResult,
        )
        return (
            TranscriptionResult(
                schema_version=SCHEMA_VERSION,
                midi_tracks=[],
                analysis=HarmonicAnalysis(
                    key="C:major",
                    time_signature=(4, 4),
                    tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
                    chords=[],
                    sections=[],
                ),
                quality=QualitySignal(overall_confidence=0.1, warnings=["fallback stub"]),
            ),
            None,
        )

    monkeypatch.setattr(transcribe_mod, "_run_without_stems", fake_single_mix)

    from backend.config import settings
    monkeypatch.setattr(settings, "demucs_parallel_stems", True)
    monkeypatch.setattr(settings, "chord_recognition_enabled", False)

    audio_path = tmp_path / "mix.wav"
    audio_path.write_bytes(b"\x00")

    _run_with_stems(audio_path, stems, stem_stats)
    assert called["fallback"] is True
    # The stem_stats should carry the warning marker so the
    # QualitySignal explains why the stems path bailed.
    assert any("all stems empty" in w for w in stem_stats.warnings)
