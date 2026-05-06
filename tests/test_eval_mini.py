"""Phase 0 mini-eval harness tests.

Acceptance per
``docs/research/transcription-improvement-implementation-plan.md`` §Phase 0:

* ``pytest`` covering ``scripts/eval_mini.py`` exists and the suite runs
  in <60 s.
* All three metrics produce values in ``[0, 1]`` (no NaNs, no exceptions).
* Smoke A/B regression: deliberately injecting a 5-semitone transpose on
  the engraved MIDI must drop ``chroma_rf`` by ≥0.10. Otherwise the
  metric has decoupled from harmonic identity and the plan's mitigation
  (switch ``mir_eval.chord`` to ``tetrads``) kicks in.

The test module patches the heavy pipeline surface (``_run_basic_pitch_sync``,
``arrange``, ``humanize``) with hand-crafted contracts so the suite stays
deterministic and fast — the *harness wiring* is what's under test, not
Basic Pitch's transcription quality.
"""
from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pretty_midi
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.contracts import (  # noqa: E402
    PianoScore,
    ScoreMetadata,
    ScoreNote,
    TempoMapEntry,
)

from eval.tier_rf import (  # noqa: E402
    PLAYABILITY_MAX_NOTES_PER_HAND,
    PLAYABILITY_MAX_SPAN_SEMITONES,
    chord_rf_score,
    chroma_rf_score,
    compute_tier_rf,
    fluidsynth_resynth,
    playability_rf_score,
)
from scripts import eval_mini  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers — synthetic fixtures so tests don't depend on Basic Pitch / Demucs
# ---------------------------------------------------------------------------

def _make_score(rh_notes, lh_notes) -> PianoScore:
    """Build a minimal PianoScore from ``(pitch, onset_beat)`` tuples.

    Voice/velocity/duration are filled with sane defaults — the
    playability metric only inspects ``onset_beat`` and ``pitch``.
    """
    return PianoScore(
        right_hand=[
            ScoreNote(
                id=f"rh-{i:04d}",
                pitch=p,
                onset_beat=ob,
                duration_beat=1.0,
                velocity=80,
                voice=1,
            )
            for i, (p, ob) in enumerate(rh_notes)
        ],
        left_hand=[
            ScoreNote(
                id=f"lh-{i:04d}",
                pitch=p,
                onset_beat=ob,
                duration_beat=1.0,
                velocity=80,
                voice=1,
            )
            for i, (p, ob) in enumerate(lh_notes)
        ],
        metadata=ScoreMetadata(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
            difficulty="intermediate",
        ),
    )


def _build_chord_midi(
    pitches: list[int],
    *,
    duration_sec: float = 4.0,
) -> bytes:
    """Build a MIDI of a sustained chord (notes held simultaneously).

    Used to drive both sides of the chord/chroma metrics with identical
    harmonic content for the round-trip sanity tests.
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0, name="Piano")
    for p in pitches:
        inst.notes.append(
            pretty_midi.Note(velocity=90, pitch=p, start=0.0, end=duration_sec)
        )
    pm.instruments.append(inst)
    buf = io.BytesIO()
    pm.write(buf)
    return buf.getvalue()


def _build_arpeggio_midi(
    pitches: list[int],
    *,
    note_dur_sec: float = 0.4,
) -> bytes:
    """Build a MIDI of an arpeggio (notes one after another)."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0, name="Piano")
    t = 0.0
    for p in pitches:
        inst.notes.append(
            pretty_midi.Note(velocity=90, pitch=p, start=t, end=t + note_dur_sec)
        )
        t += note_dur_sec
    pm.instruments.append(inst)
    buf = io.BytesIO()
    pm.write(buf)
    return buf.getvalue()


def _transpose_midi(midi_bytes: bytes, *, semitones: int) -> bytes:
    """Transpose every note in a MIDI by ``semitones`` (clamped to 0..127)."""
    pm = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    for inst in pm.instruments:
        for n in inst.notes:
            n.pitch = max(0, min(127, n.pitch + semitones))
    buf = io.BytesIO()
    pm.write(buf)
    return buf.getvalue()


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    import soundfile as sf
    sf.write(str(path), audio, sr)


# ---------------------------------------------------------------------------
# Tier 3 RF — pure-data tests, no audio dependency
# ---------------------------------------------------------------------------

def test_playability_all_playable_when_chords_within_limits():
    # Two RH chords (4-note chord, 6-semitone span); two LH chords.
    score = _make_score(
        rh_notes=[(60, 0.0), (64, 0.0), (67, 0.0), (72, 0.0),
                  (62, 1.0), (65, 1.0), (69, 1.0)],
        lh_notes=[(36, 0.0), (43, 0.0),
                  (38, 1.0), (45, 1.0)],
    )
    fraction, n_play, n_total = playability_rf_score(score)
    assert n_total == 4  # 2 RH groups + 2 LH groups
    assert n_play == 4
    assert fraction == 1.0


def test_playability_fails_on_oversized_chord():
    # RH chord with 6 notes — fails on max_notes_per_hand=5.
    rh = [(60 + 2 * i, 0.0) for i in range(6)]
    # LH chord spanning 18 semitones — fails on max_span_semitones=14.
    lh = [(36, 0.0), (54, 0.0)]
    score = _make_score(rh_notes=rh, lh_notes=lh)
    fraction, n_play, n_total = playability_rf_score(score)
    assert n_total == 2
    assert n_play == 0
    assert fraction == 0.0


def test_playability_returns_zero_for_empty_score():
    score = _make_score(rh_notes=[], lh_notes=[])
    fraction, n_play, n_total = playability_rf_score(score)
    assert (fraction, n_play, n_total) == (0.0, 0, 0)


def test_playability_constants_match_plan():
    # The plan §Phase 0 fixes these thresholds; locking them avoids a
    # silent drift if someone bumps the constants without updating the
    # acceptance criteria.
    assert PLAYABILITY_MAX_SPAN_SEMITONES == 14
    assert PLAYABILITY_MAX_NOTES_PER_HAND == 5


# ---------------------------------------------------------------------------
# Tier 2 + Tier 4 RF — round-trip on FluidSynth-rendered audio
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def chord_midi_audio_pair(tmp_path_factory):
    """A C-major-7 sustained chord rendered to audio + MIDI bytes.

    Module-scoped so the FluidSynth render cost is paid once across the
    Tier 2/Tier 4 round-trip tests (FluidSynth ~200 ms per render).
    """
    cmaj7 = [60, 64, 67, 71]            # C E G B
    midi_bytes = _build_chord_midi(cmaj7, duration_sec=4.0)
    audio, sr = fluidsynth_resynth(midi_bytes)
    return midi_bytes, audio, sr


def test_chroma_rf_high_for_identical_inputs(chord_midi_audio_pair):
    midi_bytes, audio, sr = chord_midi_audio_pair
    # Same audio on both sides — chroma vectors should match almost
    # perfectly, modulo floating point noise.
    score, n_beats, notes = chroma_rf_score((audio, sr), (audio, sr))
    assert 0.0 <= score <= 1.0, f"out-of-range chroma_rf={score} notes={notes}"
    assert score > 0.95, f"identical audio should score near 1.0, got {score}"


def test_chord_rf_high_for_identical_inputs(chord_midi_audio_pair):
    midi_bytes, audio, sr = chord_midi_audio_pair
    chord_score, n_in, n_rs, notes = chord_rf_score(
        (audio, sr), (audio, sr), key_label="C:major",
    )
    assert 0.0 <= chord_score <= 1.0, f"out-of-range chord_rf={chord_score} notes={notes}"
    # Both sides recognize the same chord, so MIREX should land at 1.0.
    # If the recognizer can't pick anything up at all (n_in=0), the
    # chord_rf path returns 0.0 — that's an honest answer for short
    # silent fixtures, but our 4-second sustained C maj7 produces clear
    # chroma so we expect detection.
    if n_in > 0 and n_rs > 0:
        assert chord_score >= 0.99, f"identical chord audio should score 1.0, got {chord_score}"


def test_chroma_rf_drops_under_5_semitone_transpose(chord_midi_audio_pair):
    """The Risks-section smoke test from the plan.

    A 5-semitone transpose on the engraved MIDI must drop chroma_rf by
    ≥0.10 vs. the un-transposed baseline. If it doesn't, chroma cosine
    has decoupled from harmonic identity and the plan's mitigation
    (switch ``mir_eval.chord`` to ``tetrads``) is required.
    """
    midi_bytes, in_audio, in_sr = chord_midi_audio_pair

    rs_audio_baseline, rs_sr = fluidsynth_resynth(midi_bytes)
    score_baseline, _, _ = chroma_rf_score(
        (in_audio, in_sr), (rs_audio_baseline, rs_sr),
    )

    transposed = _transpose_midi(midi_bytes, semitones=5)
    rs_audio_transposed, rs_sr2 = fluidsynth_resynth(transposed)
    score_transposed, _, _ = chroma_rf_score(
        (in_audio, in_sr), (rs_audio_transposed, rs_sr2),
    )

    delta = score_baseline - score_transposed
    assert delta >= 0.10, (
        f"5-semitone transpose only dropped chroma_rf by {delta:.3f} "
        f"(baseline={score_baseline:.3f}, transposed={score_transposed:.3f}). "
        "If this fires, the metric has decoupled from harmonic identity — "
        "see the Risks section of Phase 0 in the implementation plan."
    )


# ---------------------------------------------------------------------------
# Top-level compute_tier_rf — wiring smoke test
# ---------------------------------------------------------------------------

def test_compute_tier_rf_returns_metrics_in_unit_range(
    chord_midi_audio_pair, tmp_path,
):
    """End-to-end ``compute_tier_rf`` produces three metrics in ``[0, 1]``.

    Phase 0 acceptance: "Three metrics all in [0, 1] for all 5 songs
    (no NaNs, no exceptions)." Single-fixture smoke against that bound.
    """
    midi_bytes, audio, sr = chord_midi_audio_pair

    audio_path = tmp_path / "input.wav"
    _write_wav(audio_path, audio, sr)

    # Score consistent with the audio (a C major chord in the RH).
    score = _make_score(
        rh_notes=[(60, 0.0), (64, 0.0), (67, 0.0)],
        lh_notes=[(48, 0.0), (52, 0.0)],
    )

    result = compute_tier_rf(
        audio_path, score, midi_bytes, key_label="C:major",
    )

    for name, value in (
        ("chord_rf", result.chord_rf),
        ("playability_rf", result.playability_rf),
        ("chroma_rf", result.chroma_rf),
    ):
        assert 0.0 <= value <= 1.0, (
            f"{name}={value} out of [0, 1]; notes={result.notes}"
        )
        assert not math.isnan(value), f"{name} is NaN"


# ---------------------------------------------------------------------------
# CLI orchestration — eval_mini.run() against a synthetic 1-song manifest
# ---------------------------------------------------------------------------

def _stub_pipeline_artifacts(score: PianoScore, midi_bytes: bytes):
    """Build the same 4-tuple ``_run_pipeline`` would return."""
    from backend.contracts import (
        HarmonicAnalysis,
        QualitySignal,
        TempoMapEntry,
        TranscriptionResult,
    )
    txr = TranscriptionResult(
        midi_tracks=[],
        analysis=HarmonicAnalysis(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
        ),
        quality=QualitySignal(overall_confidence=1.0, warnings=[]),
    )
    return eval_mini.PipelineArtifacts(
        score=score, midi_bytes=midi_bytes, key_label=txr.analysis.key,
        transcription=txr,
    )


def test_eval_mini_run_writes_aggregate_json(
    chord_midi_audio_pair, tmp_path, monkeypatch,
):
    """``eval_mini.run`` writes ``aggregate.json`` with metrics in ``[0, 1]``.

    Mocks ``_run_pipeline`` so this test exercises the orchestration —
    manifest loading, audio resolution, tier_rf wiring, JSON writing —
    without paying the Basic Pitch cold-start cost.
    """
    midi_bytes, audio, sr = chord_midi_audio_pair

    # Build a one-song synthetic manifest.
    eval_set_path = tmp_path / "synthetic_set"
    songs_dir = eval_set_path / "songs"
    songs_dir.mkdir(parents=True)
    audio_path = songs_dir / "source.wav"
    _write_wav(audio_path, audio, sr)

    manifest = {
        "schema_version": 1,
        "eval_set": "synthetic_test",
        "target_duration_sec": 5.0,
        "chord_recognition_key": "C:major",
        "songs": [
            {
                "slug": "synth_001",
                "title": "C major 7",
                "artist": "tier_rf_test",
                "genre": "test",
                "source": {
                    "kind": "audio_file",
                    "path": "songs/source.wav",
                    "license": "test",
                    "bootstrap": False,
                },
            },
        ],
    }
    import yaml as _yaml
    (eval_set_path / "manifest.yaml").write_text(_yaml.safe_dump(manifest))

    score = _make_score(
        rh_notes=[(60, 0.0), (64, 0.0), (67, 0.0), (71, 0.0)],
        lh_notes=[(48, 0.0), (52, 0.0)],
    )

    def fake_run_pipeline(_audio_path):
        return _stub_pipeline_artifacts(score, midi_bytes)

    monkeypatch.setattr(eval_mini, "_run_pipeline", fake_run_pipeline)

    output_dir = tmp_path / "run"
    payload = eval_mini.run(eval_set_path, output_dir)

    out_path = output_dir / "aggregate.json"
    assert out_path.is_file(), "aggregate.json missing"
    on_disk = json.loads(out_path.read_text())
    assert on_disk["aggregate"]["n_songs_scored"] == 1
    assert on_disk["aggregate"]["n_songs_errored"] == 0
    song = on_disk["songs"][0]
    assert song["slug"] == "synth_001"
    assert song["error"] is None
    for k in ("chord_rf", "playability_rf", "chroma_rf"):
        v = song[k]
        assert 0.0 <= v <= 1.0, f"{k}={v} out of [0, 1]"
        assert not math.isnan(v), f"{k} is NaN"

    # The same payload is returned in-memory.
    assert payload["aggregate"]["n_songs_scored"] == 1


def test_eval_mini_run_marks_errors_per_song(
    tmp_path, monkeypatch,
):
    """A failing per-song run is captured in the JSON, not propagated."""
    eval_set_path = tmp_path / "set"
    eval_set_path.mkdir()
    # Reference a non-existent audio_file source so resolution fails.
    manifest = {
        "schema_version": 1,
        "eval_set": "broken_test",
        "target_duration_sec": 5.0,
        "chord_recognition_key": "C:major",
        "songs": [
            {
                "slug": "missing_001",
                "source": {
                    "kind": "audio_file",
                    "path": "does_not_exist.wav",
                },
            },
        ],
    }
    import yaml as _yaml
    (eval_set_path / "manifest.yaml").write_text(_yaml.safe_dump(manifest))

    output_dir = tmp_path / "run"
    payload = eval_mini.run(eval_set_path, output_dir)

    assert payload["aggregate"]["n_songs_total"] == 1
    assert payload["aggregate"]["n_songs_scored"] == 0
    assert payload["aggregate"]["n_songs_errored"] == 1
    song = payload["songs"][0]
    assert song["error"] is not None
    assert "FileNotFoundError" in song["error"]
