"""Real Basic Pitch smoke test — guards against regressions in the
inference path that the auto-stubbed pipeline tests no longer cover.

Synthesizes a 5-second monophonic piano clip with three known notes
via FluidSynth, runs the full ``TranscribeService`` (no stub fixture),
and asserts a mid-bar note-detection F1 ≥ 0.4. This is intentionally a
loose floor: the assertion is "real Basic Pitch produced something
note-like from a clean synth source," not a quality benchmark — the
quality benchmarks live in ``scripts/eval_transcription.py`` and the
mini-eval harness.

Skipped when ``basic-pitch`` or a ``fluidsynth`` binary is unavailable
on the test host (e.g. CI's ``.[dev]`` install). The marker
``real_transcribe`` opts the test out of the auto-applied stub
fixture in ``tests/conftest.py``.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.real_transcribe


def _have_basic_pitch() -> bool:
    try:
        import basic_pitch  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def _have_fluidsynth_binary() -> bool:
    return shutil.which("fluidsynth") is not None


def _bundled_soundfont() -> Path | None:
    """Locate the TimGM6mb soundfont bundled with pretty_midi (used by the
    eval harness too). Returns None when pretty_midi isn't installed."""
    try:
        import pretty_midi  # noqa: PLC0415
    except ImportError:
        return None
    sf2 = Path(pretty_midi.__file__).parent / "TimGM6mb.sf2"
    return sf2 if sf2.is_file() else None


def _synthesize_piano_clip(out_wav: Path, soundfont: Path) -> list[int]:
    """Render three half-second piano notes (C4, E4, G4) into a WAV file.

    Returns the list of MIDI pitches synthesized so the caller can score
    Basic Pitch's predictions against ground truth.
    """
    import pretty_midi  # noqa: PLC0415

    pitches = [60, 64, 67]  # C4 E4 G4 — root-position triad arpeggio
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0)  # Acoustic grand
    for i, p in enumerate(pitches):
        inst.notes.append(
            pretty_midi.Note(
                velocity=100,
                pitch=p,
                start=0.5 + i * 1.0,
                end=0.5 + i * 1.0 + 0.6,
            )
        )
    pm.instruments.append(inst)

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as midi_tmp:
        midi_path = Path(midi_tmp.name)
    try:
        pm.write(str(midi_path))
        subprocess.run(
            [
                "fluidsynth",
                "-ni",
                "-F", str(out_wav),
                "-r", "44100",
                str(soundfont),
                str(midi_path),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
    finally:
        midi_path.unlink(missing_ok=True)

    return pitches


def _f1_pitch_only(predicted_pitches: list[int], reference_pitches: list[int]) -> float:
    """Multiset F1 over pitch only (timing-agnostic).

    Each reference pitch must be matched exactly once. Order doesn't
    matter, multiplicities do — `[60,60,64]` vs `[60,64]` scores 2/3
    matched on each side rather than 3/3.
    """
    ref = list(reference_pitches)
    matched = 0
    for p in predicted_pitches:
        if p in ref:
            ref.remove(p)
            matched += 1
    if matched == 0:
        return 0.0
    precision = matched / max(1, len(predicted_pitches))
    recall = matched / max(1, len(reference_pitches))
    return 2 * precision * recall / (precision + recall)


@pytest.mark.skipif(not _have_basic_pitch(), reason="basic-pitch not installed")
@pytest.mark.skipif(not _have_fluidsynth_binary(), reason="fluidsynth binary not on PATH")
def test_real_basic_pitch_recovers_clean_synth_triad(tmp_path: Path) -> None:
    sf2 = _bundled_soundfont()
    if sf2 is None:
        pytest.skip("TimGM6mb.sf2 soundfont not bundled with pretty_midi")

    wav_path = tmp_path / "synth_piano.wav"
    truth = _synthesize_piano_clip(wav_path, sf2)
    assert wav_path.is_file() and wav_path.stat().st_size > 1024

    from basic_pitch import ICASSP_2022_MODEL_PATH  # noqa: PLC0415
    from basic_pitch.inference import predict  # noqa: PLC0415

    _model_output, _midi, note_events = predict(
        str(wav_path),
        model_or_model_path=ICASSP_2022_MODEL_PATH,
    )

    predicted = [int(ev[2]) for ev in note_events]
    f1 = _f1_pitch_only(predicted, truth)
    assert f1 >= 0.4, (
        f"Real Basic Pitch F1 {f1:.2f} below 0.4 floor "
        f"(predicted={predicted}, truth={truth})"
    )
