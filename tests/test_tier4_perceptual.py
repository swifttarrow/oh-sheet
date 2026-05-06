"""Phase 7 Tier 4 perceptual-metric tests.

Acceptance per Phase 7 plan:

* Chroma cosine re-export matches ``tier_rf.chroma_rf_score`` byte-for-byte.
* Round-trip self-consistency F1 returns ``1.0`` (or near it) when
  the transcribe callable is the identity function — the engraved
  MIDI bytes round-trip to themselves.
* CLAP / MERT skip cleanly when their heavy deps aren't installed
  (returns ``None`` + a note in the result; never raises).
* ``compute_tier4`` honors the ``enable_*`` toggles — Tier 4 with
  no transcribe callable returns chroma only; with it returns
  chroma + round_trip; CLAP/MERT only when explicitly enabled.
* The composite drops missing terms and re-averages.
"""
from __future__ import annotations

import io
import math
import sys
from pathlib import Path

import numpy as np
import pretty_midi
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.tier4_perceptual import (  # noqa: E402
    Tier4Result,
    chroma_cosine_score,
    clap_cosine_score,
    compute_tier4,
    fluidsynth_available,
    mert_cosine_score,
    round_trip_f1_score,
)
from eval.tier_rf import fluidsynth_resynth  # noqa: E402

# All Tier 4 metrics depend on FluidSynth for the resynth side. Skip
# the suite cleanly on environments without the binary (mostly
# minimal CI containers — the eval-ci.yml workflow installs it).
pytestmark = pytest.mark.skipif(
    not fluidsynth_available(),
    reason="fluidsynth binary not installed; Tier 4 tests need it for resynth.",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_chord_midi(pitches, duration_sec: float = 4.0) -> bytes:
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


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    import soundfile as sf
    sf.write(str(path), audio, sr)


@pytest.fixture(scope="module")
def chord_audio_pair(tmp_path_factory):
    """A C-major-7 chord rendered to (audio, midi_bytes, audio_path).

    Module-scoped so we pay the FluidSynth render cost once across
    the Tier 4 suite.
    """
    tmpdir = tmp_path_factory.mktemp("tier4")
    cmaj7 = [60, 64, 67, 71]
    midi_bytes = _build_chord_midi(cmaj7, duration_sec=4.0)
    audio, sr = fluidsynth_resynth(midi_bytes)
    audio_path = tmpdir / "input.wav"
    _write_wav(audio_path, audio, sr)
    return audio, sr, midi_bytes, audio_path


# ---------------------------------------------------------------------------
# chroma_cosine_score — re-export parity
# ---------------------------------------------------------------------------

def test_chroma_cosine_score_re_export_matches_tier_rf(chord_audio_pair):
    """Re-export must produce identical numbers to tier_rf."""
    from eval.tier_rf import chroma_rf_score

    audio, sr, _, _ = chord_audio_pair
    a, n_a, _ = chroma_cosine_score((audio, sr), (audio, sr))
    b, n_b, _ = chroma_rf_score((audio, sr), (audio, sr))
    assert a == b
    assert n_a == n_b


def test_chroma_cosine_score_is_high_for_identical_audio(chord_audio_pair):
    audio, sr, _, _ = chord_audio_pair
    score, n_beats, notes = chroma_cosine_score((audio, sr), (audio, sr))
    assert 0.0 <= score <= 1.0, f"out-of-range chroma cosine={score}"
    assert score > 0.95, f"identical audio should score near 1.0, got {score}"


# ---------------------------------------------------------------------------
# round_trip_f1_score — identity transcribe should yield ~1.0 F1
# ---------------------------------------------------------------------------

def test_round_trip_f1_identity_transcribe_close_to_one(chord_audio_pair):
    """When the transcribe callable returns the same MIDI bytes for both
    sides, mir_eval F1 should be ~1.0. Catches a wiring bug where
    midi1 / midi2 get swapped or the bytes are re-encoded in a lossy
    way before comparison.
    """
    _, _, engraved_midi_bytes, audio_path = chord_audio_pair

    captured: list[Path] = []

    def fake_transcribe(path: Path) -> bytes:
        captured.append(path)
        return engraved_midi_bytes

    f1_no, f1_w, n_in, n_rs, notes = round_trip_f1_score(
        audio_path, engraved_midi_bytes, fake_transcribe,
    )
    assert n_in > 0 and n_rs > 0
    assert f1_no is not None and f1_w is not None
    assert f1_no >= 0.95, f"identity transcribe should F1 ~ 1.0; got {f1_no}"
    assert len(captured) == 2, "should call transcribe twice (input + resynth)"


def test_round_trip_f1_handles_transcribe_failure(chord_audio_pair):
    """A transcribe failure on either side returns ``(None, None, …)`` with notes."""
    _, _, engraved_midi_bytes, audio_path = chord_audio_pair

    def explode(path: Path) -> bytes:
        raise RuntimeError("synthetic transcribe failure")

    f1_no, f1_w, n_in, n_rs, notes = round_trip_f1_score(
        audio_path, engraved_midi_bytes, explode,
    )
    assert f1_no is None and f1_w is None
    assert any("transcribe(input) failed" in n for n in notes)


def test_round_trip_f1_handles_empty_midi(chord_audio_pair, tmp_path):
    """An empty MIDI on the resynth side returns None with a note."""
    _, _, engraved_midi_bytes, audio_path = chord_audio_pair

    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    pm.instruments.append(pretty_midi.Instrument(program=0, name="Piano"))
    buf = io.BytesIO()
    pm.write(buf)
    empty_midi = buf.getvalue()

    def transcribe(path: Path) -> bytes:
        return empty_midi

    f1_no, _, _, _, notes = round_trip_f1_score(
        audio_path, engraved_midi_bytes, transcribe,
    )
    assert f1_no is None
    assert any("empty note set" in n for n in notes)


# ---------------------------------------------------------------------------
# CLAP / MERT — graceful skip when deps missing
# ---------------------------------------------------------------------------

def test_clap_cosine_skips_gracefully_when_dep_missing(chord_audio_pair):
    """The CLI tests run in an env without ``laion_clap``; verify the
    metric returns ``None`` + a note rather than raising.
    """
    audio, sr, _, audio_path = chord_audio_pair
    score, notes = clap_cosine_score(audio_path, (audio, sr))
    if score is None:
        assert any("clap_cosine" in n for n in notes)
    else:
        assert 0.0 <= score <= 1.0


def test_mert_cosine_skips_gracefully_when_dep_missing(chord_audio_pair):
    audio, sr, _, audio_path = chord_audio_pair
    score, notes = mert_cosine_score(audio_path, (audio, sr))
    if score is None:
        assert any("mert_cosine" in n for n in notes)
    else:
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# compute_tier4 — top-level wiring
# ---------------------------------------------------------------------------

def test_compute_tier4_chroma_only_returns_unit_range(chord_audio_pair):
    audio, sr, midi_bytes, audio_path = chord_audio_pair
    result = compute_tier4(audio_path, midi_bytes, transcribe_callable=None)
    assert isinstance(result, Tier4Result)
    assert result.chroma_cosine is not None
    assert 0.0 <= result.chroma_cosine <= 1.0
    assert result.round_trip_f1_no_offset is None
    assert result.clap_cosine is None
    assert result.mert_cosine is None


def test_compute_tier4_with_round_trip_runs_self_consistency(chord_audio_pair):
    audio, sr, midi_bytes, audio_path = chord_audio_pair

    def fake_transcribe(path: Path) -> bytes:
        return midi_bytes

    result = compute_tier4(
        audio_path, midi_bytes, transcribe_callable=fake_transcribe,
    )
    assert result.round_trip_f1_no_offset is not None
    assert result.round_trip_f1_no_offset >= 0.95


def test_compute_tier4_composite_drops_missing_terms(chord_audio_pair):
    """The composite must filter ``None`` entries and re-average."""
    audio, sr, midi_bytes, audio_path = chord_audio_pair
    result = compute_tier4(audio_path, midi_bytes, transcribe_callable=None)
    # Only chroma_cosine populated, so composite == chroma_cosine.
    assert result.composite == pytest.approx(result.chroma_cosine, abs=1e-6)


def test_compute_tier4_as_dict_round_trips(chord_audio_pair):
    audio, sr, midi_bytes, audio_path = chord_audio_pair
    result = compute_tier4(audio_path, midi_bytes, transcribe_callable=None)
    payload = result.as_dict()
    for key in (
        "chroma_cosine", "round_trip_f1_no_offset", "clap_cosine",
        "mert_cosine", "composite", "n_chroma_beats",
    ):
        assert key in payload


def test_compute_tier4_handles_unreadable_audio(tmp_path):
    """A bad audio path returns a Tier4Result with an error note rather than raising."""
    bogus = tmp_path / "does_not_exist.wav"
    midi_bytes = _build_chord_midi([60, 64, 67])
    result = compute_tier4(bogus, midi_bytes, transcribe_callable=None)
    assert result.chroma_cosine is None
    assert any("load input audio" in n for n in result.notes)


def test_tier4_composite_clamped_to_unit_range(chord_audio_pair):
    """Even with synthetic perfect inputs, composite stays within [0, 1]."""
    audio, sr, midi_bytes, audio_path = chord_audio_pair

    def fake_transcribe(path: Path) -> bytes:
        return midi_bytes

    result = compute_tier4(
        audio_path, midi_bytes, transcribe_callable=fake_transcribe,
    )
    assert 0.0 <= result.composite <= 1.0
    assert not math.isnan(result.composite)
