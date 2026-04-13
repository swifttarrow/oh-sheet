#!/usr/bin/env python3
"""
Pop2Piano (sweetcocoa/pop2piano) — pop **audio** (e.g. MP3) → piano MIDI.

**Golden input:** ``golden.mp3`` (audio-native encoder). Use ``--from-midi golden.mid`` only
if you want the old sine-synthesis path from a score file.

Default input is ``golden.mp3`` at the repo root. Override with ``--audio``.

MP3 decoding uses **librosa** (often via ``audioread`` + **ffmpeg**). If loading fails,
install a decoder stack, e.g. ``pip install soundfile audioread`` and ensure ``ffmpeg``
is on your ``PATH``.

Optional: ``--from-midi path.mid`` ignores ``--audio`` and instead synthesizes the MIDI
to a waveform with ``pretty_midi`` (same as the old script behavior).

Install (``Pop2PianoProcessor`` is gated on **essentia** in current ``transformers``; the
feature extractor also uses **librosa** resampling, which lazily needs **resampy**):
  pip install torch transformers librosa resampy scipy pretty_midi essentia==2.1b6.dev1389

The pin ``2.1b6.dev1034`` appears in some Transformers messages but is **no longer on PyPI**;
install a published ``2.1b6.dev*`` build (e.g. ``dev1389``) or the latest your index lists.

Python **3.13** often has no prebuilt ``essentia`` wheels; use **3.10–3.11** (e.g.
``conda create -n pop2piano python=3.11``) if ``pip install essentia`` fails.

No Hugging Face credentials required for public weights (HF_TOKEN optional for rate limits).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

_HF_DIR = Path(__file__).resolve().parent
if str(_HF_DIR) not in sys.path:
    sys.path.insert(0, str(_HF_DIR))

import _common  # noqa: E402

_REPO_ID = "sweetcocoa/pop2piano"

# Pop2PianoProcessor / Pop2PianoFeatureExtractor are decorated with
# ``@requires(backends=("essentia", "librosa", "pretty_midi", "scipy", "torch"))``
# in current ``transformers`` — all must be importable before ``from_pretrained``.
# ``resampy`` is not listed there but librosa loads it lazily for ``resample(..., res_type="kaiser_best")``
# inside ``Pop2PianoFeatureExtractor``; without it, ``processor(...)`` crashes at runtime.
_PREREQ_PKGS = ("essentia", "librosa", "resampy", "scipy", "pretty_midi", "torch")


def _missing_prereq_modules() -> list[str]:
    """Match what ``transformers`` will require at ``Pop2PianoProcessor`` load time."""
    missing: list[str] = []
    for name in _PREREQ_PKGS:
        if importlib.util.find_spec(name) is None:
            missing.append(name)
            continue
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    return missing


def _load_pop2piano_processor(repo_id: str):
    from transformers import Pop2PianoProcessor

    try:
        return Pop2PianoProcessor.from_pretrained(repo_id)
    except ImportError as e:
        print(
            "\nPop2PianoProcessor still failed to load (often missing or broken ``essentia``):\n"
            "  pip install essentia==2.1b6.dev1389\n"
            "On Python 3.13, prefer a 3.10–3.11 venv — essentia wheels are rarely published for 3.13.\n",
            file=sys.stderr,
        )
        raise e


def main() -> int:
    p = argparse.ArgumentParser(description="Run Pop2Piano on an audio file (default: golden.mp3).")
    p.add_argument(
        "--audio",
        type=Path,
        default=_common.default_audio_path(),
        help="Input audio (wav/mp3/flac …); default: golden.mp3 at repo root",
    )
    p.add_argument(
        "--from-midi",
        type=Path,
        default=None,
        metavar="PATH.mid",
        help="Synthesize this MIDI (e.g. golden.mid) with pretty_midi instead of --audio",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output MIDI path (default: hf_model_outputs/pop2piano_output.mid)",
    )
    p.add_argument("--composer", type=str, default="composer1")
    p.add_argument("--sr", type=int, default=44100, help="Target sample rate for the model")
    args = p.parse_args()

    missing = _missing_prereq_modules()
    if missing:
        print(
            "Pop2Piano requires these importable packages (Transformers gates + librosa resampling):\n"
            f"  missing: {', '.join(missing)}\n"
            "Install with:\n"
            "  pip install torch transformers librosa resampy scipy pretty_midi essentia==2.1b6.dev1389\n"
            "If essentia fails to build on Python 3.13, use Python 3.10–3.11 (e.g. conda env).",
            file=sys.stderr,
        )
        return 1

    out = args.out or (_common.repo_root() / "hf_model_outputs" / "pop2piano_output.mid")
    out = Path(out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import librosa
        import numpy as np
        import torch
        from transformers import Pop2PianoForConditionalGeneration
    except ImportError as e:
        print(
            "Import failed after prereq check. Try:\n"
            "  pip install torch transformers librosa resampy scipy pretty_midi essentia==2.1b6.dev1389",
            file=sys.stderr,
        )
        raise e

    if args.from_midi is not None:
        import pretty_midi

        midi_path = _common.require_midi(args.from_midi)
        print(f"Synthesizing MIDI to audio (pretty_midi): {midi_path}")
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        audio = pm.synthesize(fs=args.sr)
        if audio.ndim != 1:
            audio = audio[0]
        audio = np.asarray(audio, dtype=np.float32)
        sr = args.sr
    else:
        audio_path = _common.require_existing_file(args.audio, "Audio")
        print(f"Loading audio: {audio_path}")
        audio, sr = librosa.load(str(audio_path), sr=args.sr, mono=True)
        audio = np.asarray(audio, dtype=np.float32)

    print(f"Loading {_REPO_ID} …")
    model = Pop2PianoForConditionalGeneration.from_pretrained(_REPO_ID)
    processor = _load_pop2piano_processor(_REPO_ID)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    print("Generating …")
    with torch.no_grad():
        model_output = model.generate(
            input_features=inputs["input_features"],
            composer=args.composer,
        )

    decoded = processor.batch_decode(
        token_ids=model_output,
        feature_extractor_output=inputs,
    )["pretty_midi_objects"][0]
    decoded.write(str(out))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
