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

import pytest

# pretty_midi ships with the ``basic-pitch`` extra, not with ``dev``, so
# CI runs that only install ``.[dev]`` must skip this whole module
# rather than error out during collection. Matches the importorskip
# convention used in test_bass_extraction.py, test_melody_extraction.py,
# etc.
pretty_midi = pytest.importorskip("pretty_midi")

from backend.contracts import InstrumentRole  # noqa: E402
from backend.services import transcribe as transcribe_mod  # noqa: E402
from backend.services.stem_separation import SeparatedStems, StemSeparationStats  # noqa: E402
from backend.services.transcribe import (  # noqa: E402
    _BasicPitchPass,
    _run_with_stems,
)
from backend.services.transcription_cleanup import CleanupStats  # noqa: E402

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
    """Silence the audio-only stages — tempo, chord recog, duration probe.

    All three are invoked unconditionally on the stems path (``tempo``
    and ``duration`` always, ``chord`` gated only by
    ``chord_recognition_enabled``), and all three would otherwise hit
    librosa/soundfile on a placeholder wav. We stub them so the test
    stays pure and the "tried audioread, failed" deprecation warning
    doesn't clutter the log.
    """
    monkeypatch.setattr(
        transcribe_mod, "tempo_map_from_audio_path", lambda _path: None
    )
    monkeypatch.setattr(
        transcribe_mod, "_audio_duration_sec", lambda _path: None
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
    # without touching disk. Skip if basic_pitch isn't installed —
    # pretty_midi can technically be present without basic_pitch even
    # though the extras group bundles them together.
    bp_inf = pytest.importorskip("basic_pitch.inference")
    bp_nc = pytest.importorskip("basic_pitch.note_creation")
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
        # Per-stem expectation: only the vocals pass keeps its
        # ``model_output`` alive, because the Phase-2 Viterbi
        # melody extractor needs the vocals contour to re-score
        # BP's vocals note events. Bass and other drop theirs
        # immediately so the three concurrent contour tensors
        # don't all pin memory at once.
        label = stem_path.stem
        if label == "vocals":
            assert keep_model_output is True, (
                "stems path must keep the vocals model_output so "
                "the post-pass Viterbi melody extractor can read "
                "the vocals contour"
            )
        else:
            assert keep_model_output is False, (
                "stems path must drop bass/other model_output so "
                "the contour tensors don't pin memory across all "
                "three concurrent passes"
            )
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


# ---------------------------------------------------------------------------
# _run_with_stems — Viterbi melody extractor runs on the vocals contour
# ---------------------------------------------------------------------------

def test_run_with_stems_runs_viterbi_on_vocals_contour(
    monkeypatch, tmp_path, stub_audio_helpers,
):
    """When the vocals pass carries a contour, ``backfill_melody_notes`` runs.

    This is the "Viterbi-on-stems" wiring: the vocals stem's Basic
    Pitch pass keeps ``model_output`` alive, and we re-score its
    note events against the vocals contour via the Phase-2 Viterbi
    back-fill wrapper before routing them to MELODY. The bass/other
    stems are still routed raw — only vocals gets the rescoring.
    """
    import numpy as np

    from backend.services import melody_extraction as melody_mod

    stems = _make_stems(tmp_path)
    stem_stats = StemSeparationStats(stems_written=["vocals", "bass", "other", "drums"])

    # Fake vocals contour: shape (T, N_CONTOUR_BINS) of zeros. The
    # real extractor is monkeypatched below, so the contents don't
    # matter — what matters is that ``backfill_melody_notes`` receives
    # the same numpy array we planted on the vocals ``_BasicPitchPass``.
    fake_contour = np.zeros((20, melody_mod.N_CONTOUR_BINS), dtype=np.float32)

    def fake_pass(stem_path: Path, *, keep_model_output: bool = True):
        label = stem_path.stem
        mo = {"contour": fake_contour} if label == "vocals" else {}
        return _make_pass(label, model_output=mo)

    monkeypatch.setattr(transcribe_mod, "_basic_pitch_single_pass", fake_pass)

    # Distinctive "Viterbi-promoted" event so the assertion can
    # tell the rescored list apart from the raw vocals events.
    viterbi_event = (0.1, 0.4, 74, 0.4, None)
    captured: dict[str, Any] = {}

    def fake_backfill_melody_notes(contour, note_events, **kwargs):
        captured["contour"] = contour
        captured["note_events"] = list(note_events)
        captured["kwargs"] = kwargs
        stats = melody_mod.MelodyExtractionStats(
            input_note_count=len(note_events),
            melody_note_count=1,
            chord_note_count=0,
            voiced_frame_fraction=1.0,
        )
        return [viterbi_event], stats

    monkeypatch.setattr(
        transcribe_mod, "backfill_melody_notes", fake_backfill_melody_notes
    )

    from backend.config import settings
    monkeypatch.setattr(settings, "demucs_parallel_stems", True)
    monkeypatch.setattr(settings, "chord_recognition_enabled", False)
    monkeypatch.setattr(settings, "melody_extraction_enabled", True)

    audio_path = tmp_path / "mix.wav"
    audio_path.write_bytes(b"\x00")

    result, _ = _run_with_stems(audio_path, stems, stem_stats)

    # ``backfill_melody_notes`` saw the exact contour array we planted
    # on the vocals pass (identity, not just equality — we never want
    # to silently materialize a copy of a tens-of-MB tensor).
    assert captured.get("contour") is fake_contour
    # And it received the raw vocals events — one note at pitch 72
    # from ``_make_pass("vocals")``.
    assert len(captured["note_events"]) == 1
    assert captured["note_events"][0][2] == 72

    # The MELODY track carries the Viterbi-promoted event, not the
    # raw vocals event — i.e. the rescored list was actually used.
    melody_tracks = [t for t in result.midi_tracks if t.instrument == InstrumentRole.MELODY]
    assert len(melody_tracks) == 1
    melody_pitches = {n.pitch for n in melody_tracks[0].notes}
    assert melody_pitches == {74}

    # The extraction stats were surfaced as a warning so the
    # QualitySignal can explain what the Viterbi did.
    warnings_joined = " ".join(result.quality.warnings)
    assert "melody split" in warnings_joined


def test_run_with_stems_falls_back_when_viterbi_returns_empty(
    monkeypatch, tmp_path, stub_audio_helpers,
):
    """Empty Viterbi output must not silently drop the MELODY track.

    A Viterbi that returns an empty melody list (e.g. the contour
    salience never cleared the voicing floor) should fall back to
    the raw Basic Pitch vocals events — otherwise the stems path
    would regress relative to the pre-Viterbi wiring. The same
    fallback covers ``skipped=True`` due to numpy / contour-shape
    errors so the caller never has to reason about them.
    """
    import numpy as np

    from backend.services import melody_extraction as melody_mod

    stems = _make_stems(tmp_path)
    stem_stats = StemSeparationStats(stems_written=["vocals", "bass", "other", "drums"])

    fake_contour = np.zeros((20, melody_mod.N_CONTOUR_BINS), dtype=np.float32)

    def fake_pass(stem_path: Path, *, keep_model_output: bool = True):
        label = stem_path.stem
        mo = {"contour": fake_contour} if label == "vocals" else {}
        return _make_pass(label, model_output=mo)

    monkeypatch.setattr(transcribe_mod, "_basic_pitch_single_pass", fake_pass)

    def fake_backfill_melody_notes(contour, note_events, **kwargs):
        stats = melody_mod.MelodyExtractionStats(
            input_note_count=len(note_events),
            melody_note_count=0,
            chord_note_count=0,
            voiced_frame_fraction=0.0,
        )
        return [], stats

    monkeypatch.setattr(
        transcribe_mod, "backfill_melody_notes", fake_backfill_melody_notes
    )

    from backend.config import settings
    monkeypatch.setattr(settings, "demucs_parallel_stems", True)
    monkeypatch.setattr(settings, "chord_recognition_enabled", False)
    monkeypatch.setattr(settings, "melody_extraction_enabled", True)

    audio_path = tmp_path / "mix.wav"
    audio_path.write_bytes(b"\x00")

    result, _ = _run_with_stems(audio_path, stems, stem_stats)

    # MELODY still carries the raw vocals event from ``_make_pass``
    # (pitch 72) — the empty Viterbi output was ignored so we didn't
    # silently lose a track.
    melody_tracks = [t for t in result.midi_tracks if t.instrument == InstrumentRole.MELODY]
    assert len(melody_tracks) == 1
    assert {n.pitch for n in melody_tracks[0].notes} == {72}


# ---------------------------------------------------------------------------
# _run_with_stems — Viterbi runs in additive-only mode on the vocals stem
# ---------------------------------------------------------------------------

def test_crepe_owns_melody_skips_basic_pitch_vocals_pass(
    monkeypatch, tmp_path, stub_audio_helpers,
):
    """When CREPE returns events, the vocals stem skips Basic Pitch.

    This is the "crepe_owns_melody" wiring: if
    ``crepe_vocal_melody_enabled`` is on and
    ``extract_vocal_melody_crepe`` returns a non-empty note list, the
    stems path drops the vocals job from ``stem_jobs`` entirely and
    routes the CREPE events straight to MELODY. Basic Pitch still
    runs on the bass/other stems. This test pins all three guarantees
    so a future refactor can't silently re-enable the duplicate pass.
    """
    from backend.services.crepe_melody import CrepeMelodyStats

    stems = _make_stems(tmp_path)
    stem_stats = StemSeparationStats(stems_written=["vocals", "bass", "other", "drums"])

    call_log: list[str] = []

    def fake_pass(stem_path: Path, *, keep_model_output: bool = True):
        call_log.append(stem_path.stem)
        return _make_pass(stem_path.stem)

    # CREPE returns a single note at pitch 74 — distinct from
    # ``_make_pass("vocals")``'s 72 so the assertion can prove
    # the MELODY track carries the CREPE output, not the BP output.
    crepe_note = (0.0, 1.0, 74, 0.6, None)
    crepe_stats = CrepeMelodyStats(
        skipped=False,
        model="full",
        device="cpu",
        n_frames=100,
        n_voiced_frames=60,
        n_notes=1,
        wall_sec=0.1,
        warnings=["crepe test warn"],
    )

    monkeypatch.setattr(
        transcribe_mod,
        "extract_vocal_melody_crepe",
        lambda *_args, **_kwargs: ([crepe_note], crepe_stats),
    )
    monkeypatch.setattr(transcribe_mod, "_basic_pitch_single_pass", fake_pass)

    from backend.config import settings
    monkeypatch.setattr(settings, "crepe_vocal_melody_enabled", True)
    monkeypatch.setattr(settings, "demucs_parallel_stems", True)
    monkeypatch.setattr(settings, "chord_recognition_enabled", False)

    audio_path = tmp_path / "mix.wav"
    audio_path.write_bytes(b"\x00")

    result, _ = _run_with_stems(audio_path, stems, stem_stats)

    # (a) crepe_owns_melody must skip the BP vocals pass entirely.
    assert "vocals" not in call_log, (
        "crepe_owns_melody must skip the BP vocals pass"
    )
    # The other two stems still ran through Basic Pitch.
    assert "bass" in call_log
    assert "other" in call_log

    # (b) The MELODY track carries the CREPE pitch (74), not the
    # pitch (72) that ``_make_pass("vocals")`` would have emitted.
    melody_tracks = [
        t for t in result.midi_tracks if t.instrument == InstrumentRole.MELODY
    ]
    assert len(melody_tracks) == 1
    assert {n.pitch for n in melody_tracks[0].notes} == {74}

    # (c) The CREPE-side warning surfaces on the QualitySignal, so
    # operators can tell which melody path ran.
    warnings_joined = " ".join(result.quality.warnings)
    assert "crepe test warn" in warnings_joined


def test_run_with_stems_uses_backfill_melody_notes_wrapper(
    monkeypatch, tmp_path, stub_audio_helpers,
):
    """On the stems path Viterbi runs via the ``backfill_melody_notes`` wrapper.

    The per-stem cleanup already drops BP ghosts on the vocals stem,
    so re-running the path-agreement split at this layer just throws
    away legitimate vocal harmonies / ornaments whose pitches don't
    match the dominant melodic Viterbi line. The
    ``backfill_melody_notes`` wrapper bakes ``split_enabled=False``
    into its contract and returns ``(events, stats)`` — this test
    pins the wiring so a future refactor can't quietly swap back to
    ``extract_melody`` directly and re-introduce the dropped-vocals
    regression.

    Also verifies that ``extract_melody`` (the full split pipeline)
    is *never* called from the stems path — a lingering direct call
    would be dead code hidden behind the wrapper, which is the bug
    the review caught in the previous round.

    ``max_time_sec`` must also be threaded through — it clamps the
    back-fill against the audio duration so synthesized notes never
    extend past the end of the song.
    """
    import numpy as np

    from backend.services import melody_extraction as melody_mod

    stems = _make_stems(tmp_path)
    stem_stats = StemSeparationStats(stems_written=["vocals", "bass", "other", "drums"])
    fake_contour = np.zeros((20, melody_mod.N_CONTOUR_BINS), dtype=np.float32)

    def fake_pass(stem_path: Path, *, keep_model_output: bool = True):
        label = stem_path.stem
        mo = {"contour": fake_contour} if label == "vocals" else {}
        return _make_pass(label, model_output=mo)

    monkeypatch.setattr(transcribe_mod, "_basic_pitch_single_pass", fake_pass)

    captured: dict[str, Any] = {}

    def fake_backfill_melody_notes(contour, note_events, **kwargs):
        captured["kwargs"] = kwargs
        stats = melody_mod.MelodyExtractionStats(
            input_note_count=len(note_events),
            melody_note_count=len(note_events),
            chord_note_count=0,
            voiced_frame_fraction=1.0,
        )
        return list(note_events), stats

    monkeypatch.setattr(
        transcribe_mod, "backfill_melody_notes", fake_backfill_melody_notes
    )

    def fail_extract_melody(*_args, **_kwargs):
        raise AssertionError(
            "stems path must route through backfill_melody_notes, "
            "not extract_melody"
        )

    monkeypatch.setattr(transcribe_mod, "extract_melody", fail_extract_melody)
    # A real duration probe would hit librosa on a 1-byte placeholder
    # wav; the fixture stub returns None, but for this test we want a
    # concrete value to assert that it's threaded through to the call.
    monkeypatch.setattr(transcribe_mod, "_audio_duration_sec", lambda _p: 12.5)

    from backend.config import settings
    monkeypatch.setattr(settings, "demucs_parallel_stems", True)
    monkeypatch.setattr(settings, "chord_recognition_enabled", False)
    monkeypatch.setattr(settings, "melody_extraction_enabled", True)

    audio_path = tmp_path / "mix.wav"
    audio_path.write_bytes(b"\x00")

    _run_with_stems(audio_path, stems, stem_stats)

    kwargs = captured["kwargs"]
    # The wrapper's signature intentionally omits ``split_enabled`` —
    # asserting its *absence* is the cheapest regression guard.
    assert "split_enabled" not in kwargs
    assert kwargs.get("max_time_sec") == pytest.approx(12.5)


# ---------------------------------------------------------------------------
# _run_with_stems — basic-pitch.mid blob MIDI carries the audio tempo
# ---------------------------------------------------------------------------

def test_run_with_stems_blob_midi_inherits_audio_tempo_and_4_4(
    monkeypatch, tmp_path, stub_audio_helpers,
):
    """The blob ``basic-pitch.mid`` must declare the real audio tempo.

    basic-pitch's own ``note_events_to_midi`` hard-codes 120 BPM,
    which mismatches every song that isn't at 120 — and notation
    importers (MuseScore's MIDI wizard especially) treat the declared
    tempo as a hint and re-infer metric structure from note density
    when it disagrees with the notes, occasionally landing on the
    wrong time signature in the process.

    This test overrides ``tempo_map_from_audio_path`` so the stems
    pipeline sees a waveform-derived tempo of 123 BPM, runs the
    parallel stems pipeline, re-parses the serialized MIDI bytes,
    and verifies the blob file declares 123 BPM + explicit 4/4.
    """
    import io

    stems = _make_stems(tmp_path)
    stem_stats = StemSeparationStats(stems_written=["vocals", "bass", "other", "drums"])

    def fake_pass(stem_path: Path, *, keep_model_output: bool = True):
        # All three stems drop ``model_output`` — we're testing the
        # blob-MIDI tempo wiring, not the Viterbi path.
        return _make_pass(stem_path.stem)

    monkeypatch.setattr(transcribe_mod, "_basic_pitch_single_pass", fake_pass)

    from backend.contracts import TempoMapEntry
    fake_tempo_map = [
        TempoMapEntry(time_sec=0.0, beat=0.0, bpm=123.0),
        TempoMapEntry(time_sec=0.5, beat=1.0, bpm=123.5),
    ]
    monkeypatch.setattr(
        transcribe_mod, "tempo_map_from_audio_path", lambda _path: fake_tempo_map,
    )

    from backend.config import settings
    monkeypatch.setattr(settings, "demucs_parallel_stems", True)
    monkeypatch.setattr(settings, "chord_recognition_enabled", False)
    # Melody extraction off so the test doesn't also depend on the
    # (separately-tested) Viterbi wiring.
    monkeypatch.setattr(settings, "melody_extraction_enabled", False)

    audio_path = tmp_path / "mix.wav"
    audio_path.write_bytes(b"\x00")

    _result, midi_bytes = _run_with_stems(audio_path, stems, stem_stats)
    assert midi_bytes is not None

    blob = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))

    # Tempo: single ``set_tempo`` event at t=0 that matches the first
    # entry of the fake tempo map (not basic-pitch's 120 default).
    tempo_times, tempo_bpms = blob.get_tempo_changes()
    assert len(tempo_bpms) == 1
    # pretty_midi encodes tempo as seconds-per-tick internally, so a
    # round-trip write/read introduces sub-millibpm quantization noise
    # (123.0 → 123.00022 at 220 ticks/beat). A 0.01 BPM tolerance is
    # tight enough to catch a real drift while ignoring the encoding
    # floor.
    assert float(tempo_bpms[0]) == pytest.approx(123.0, abs=0.01)
    assert float(tempo_times[0]) == pytest.approx(0.0)

    # Time signature: explicit 4/4 at t=0, regardless of what
    # pretty_midi's write defaults would emit. The explicit event
    # is the belt-and-suspenders against any future pretty_midi
    # change that stops emitting the default TS meta.
    assert len(blob.time_signature_changes) == 1
    ts = blob.time_signature_changes[0]
    assert ts.numerator == 4 and ts.denominator == 4
    assert ts.time == 0.0


def test_run_with_stems_blob_midi_falls_back_to_120_without_tempo_map(
    monkeypatch, tmp_path, stub_audio_helpers,
):
    """When beat tracking fails, the blob MIDI stays at BP's 120 BPM default.

    The stub fixture already returns ``None`` from
    ``tempo_map_from_audio_path``, so this test just verifies the
    default-BPM branch in ``_run_with_stems`` picks 120 and the
    serialized MIDI matches. Guards against a future refactor
    accidentally routing the ``None`` case to ``pretty_midi.estimate_tempo``
    or some other source that could drift from 120.
    """
    import io

    stems = _make_stems(tmp_path)
    stem_stats = StemSeparationStats(stems_written=["vocals", "bass", "other", "drums"])

    def fake_pass(stem_path: Path, *, keep_model_output: bool = True):
        return _make_pass(stem_path.stem)

    monkeypatch.setattr(transcribe_mod, "_basic_pitch_single_pass", fake_pass)

    from backend.config import settings
    monkeypatch.setattr(settings, "demucs_parallel_stems", True)
    monkeypatch.setattr(settings, "chord_recognition_enabled", False)
    monkeypatch.setattr(settings, "melody_extraction_enabled", False)

    audio_path = tmp_path / "mix.wav"
    audio_path.write_bytes(b"\x00")

    _result, midi_bytes = _run_with_stems(audio_path, stems, stem_stats)
    assert midi_bytes is not None

    blob = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    _tempo_times, tempo_bpms = blob.get_tempo_changes()
    assert len(tempo_bpms) == 1
    assert float(tempo_bpms[0]) == pytest.approx(120.0)
    assert len(blob.time_signature_changes) == 1
    assert (blob.time_signature_changes[0].numerator,
            blob.time_signature_changes[0].denominator) == (4, 4)


# ---------------------------------------------------------------------------
# _rebuild_blob_midi — empty-events fallback
# ---------------------------------------------------------------------------

def test_rebuild_blob_midi_empty_events_returns_none():
    """``_rebuild_blob_midi([])`` must return ``None`` so the caller's
    fallback branch (``if blob_midi is None: blob_midi = midi_data``)
    picks up the pre-existing pretty_midi instead of serializing an
    empty pretty_midi (which MuseScore treats as a broken file).
    """
    assert transcribe_mod._rebuild_blob_midi([], initial_bpm=120.0) is None


def test_run_without_stems_falls_back_to_midi_data_when_cleaned_events_empty(
    monkeypatch, tmp_path,
):
    """Empty cleaned_events → ``blob_midi`` falls back to ``midi_data``.

    Pins the fallback wiring at ``transcribe.py:_run_without_stems``:
    when the Basic Pitch cleanup drops every note, ``_rebuild_blob_midi``
    returns ``None`` and the pre-existing ``midi_data`` (Basic Pitch's
    own pretty_midi) is used to build ``midi_bytes`` — otherwise the
    blob serializer would crash on a ``None`` pretty_midi.
    """
    import io

    # Stub audio helpers — we're not exercising Viterbi / chords /
    # tempo here, so shut them all off to keep the test focused on the
    # blob-MIDI fallback path.
    monkeypatch.setattr(
        transcribe_mod, "tempo_map_from_audio_path", lambda _path: None
    )
    monkeypatch.setattr(
        transcribe_mod, "_audio_duration_sec", lambda _path: None
    )
    from backend.config import settings
    monkeypatch.setattr(settings, "melody_extraction_enabled", False)
    monkeypatch.setattr(settings, "bass_extraction_enabled", False)
    monkeypatch.setattr(settings, "chord_recognition_enabled", False)

    # Fake ``_basic_pitch_single_pass`` that returns an empty event
    # list with a valid ``midi_data`` pretty_midi. This is the
    # "cleanup dropped everything" shape.
    empty_pm = pretty_midi.PrettyMIDI()

    def fake_pass(audio_path: Path, *, keep_model_output: bool = True):
        return _BasicPitchPass(
            cleaned_events=[],
            model_output={},
            midi_data=empty_pm,
            preprocess_stats=None,
            cleanup_stats=CleanupStats(input_count=0, output_count=0),
        )

    monkeypatch.setattr(transcribe_mod, "_basic_pitch_single_pass", fake_pass)

    audio_path = tmp_path / "mix.wav"
    audio_path.write_bytes(b"\x00")

    result, midi_bytes = transcribe_mod._run_without_stems(audio_path, None)

    # ``midi_bytes`` is non-None — the fallback branch picked up
    # ``midi_data`` rather than crashing on a None pretty_midi.
    assert midi_bytes is not None and len(midi_bytes) > 0
    # And the bytes round-trip through pretty_midi cleanly.
    _roundtrip = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    assert result is not None


def test_run_with_stems_falls_back_to_midi_data_when_rebuild_returns_none(
    monkeypatch, tmp_path, stub_audio_helpers,
):
    """Stems path: ``_rebuild_blob_midi`` → ``None`` falls back to ``midi_data``.

    Pins the fallback wiring at
    ``transcribe.py:_combined_midi_from_events``: when the rebuild
    step returns ``None`` (e.g. empty events or a missing pretty_midi
    import), the caller must substitute the representative per-stem
    ``midi_data`` pretty_midi so blob serialization still produces
    real bytes rather than crashing on ``None``.

    Mirrors ``test_run_without_stems_falls_back_to_midi_data_when_cleaned_events_empty``
    for the Demucs-driven pipeline — the single-mix and stems paths
    both reach the same ``if pm is not None else fallback`` guard via
    different callers, and both need regression coverage.
    """
    import io

    stems = _make_stems(tmp_path)
    stem_stats = StemSeparationStats(
        stems_written=["vocals", "bass", "other", "drums"],
    )

    # At least one stem must return non-empty events so we don't hit
    # the outer "all stems empty → fall back to single-mix" guard —
    # this test is specifically about the *blob rebuild* fallback
    # inside the stems-path happy route.
    def fake_pass(stem_path: Path, *, keep_model_output: bool = True):
        return _make_pass(stem_path.stem)

    monkeypatch.setattr(transcribe_mod, "_basic_pitch_single_pass", fake_pass)

    # Force ``_rebuild_blob_midi`` to return ``None`` so
    # ``_combined_midi_from_events`` takes the fallback branch at
    # ``transcribe.py:612``. Without this monkeypatch the rebuild
    # would succeed on the fake events and the fallback branch
    # would stay uncovered.
    monkeypatch.setattr(
        transcribe_mod, "_rebuild_blob_midi", lambda _events, *, initial_bpm: None
    )

    from backend.config import settings
    monkeypatch.setattr(settings, "chord_recognition_enabled", False)
    monkeypatch.setattr(settings, "melody_extraction_enabled", False)

    audio_path = tmp_path / "mix.wav"
    audio_path.write_bytes(b"\x00")

    result, midi_bytes = _run_with_stems(audio_path, stems, stem_stats)

    # ``midi_bytes`` is non-None — the fallback branch picked up the
    # representative per-stem ``midi_data`` pretty_midi rather than
    # crashing on a ``None`` pretty_midi in ``_serialize_pretty_midi``.
    assert midi_bytes is not None and len(midi_bytes) > 0
    _roundtrip = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    assert _roundtrip is not None
    # The result is still populated from the per-stem events — the
    # blob fallback is independent of the track-level routing.
    roles = {t.instrument for t in result.midi_tracks}
    assert InstrumentRole.MELODY in roles
    assert InstrumentRole.BASS in roles
    assert InstrumentRole.CHORDS in roles
