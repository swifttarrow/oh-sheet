#!/usr/bin/env python3
"""
Monster Piano Transformer (HF hub: asigalov61/Monster-Piano-Transformer).

**Golden input:** ``golden.mid`` (MIDI → internal tokens). Not an audio model.

Optional **piano audio → MIDI** step uses weights from Hugging Face
``Genius-Society/piano_trans`` (ByteDance-style CRNN checkpoint) via the
``piano_transcription_inference`` package, which matches the paper
"High-resolution Piano Transcription with Pedals…" (Kong et al.).

Pipeline:

1. If ``--audio`` is set: transcribe audio → MIDI (seed), also written as
   ``<out_stem>_transcribed.mid``.
2. Else: seed MIDI from ``--midi`` (default: ``golden.mid``).
3. Monster Piano continues from the seed → main ``.mid`` output.
4. If ``--no-pdf`` is not set: render **sheet PDF** from the continuation MIDI
   (MusicXML via ``music21``, then MuseScore or LilyPond — same approach as
   ``backend/services/engrave.py``). MuseScore is resolved from ``PATH`` or the
   standard macOS ``.app`` bundle. If nothing is available, writes
   ``<out_stem>.musicxml`` instead and prints a hint.

Uses the upstream pip package which bundles inference and checkpoints.

Install:
  pip install monsterpianotransformer

Optional transcription + PDF path:
  pip install piano_transcription_inference huggingface_hub librosa music21

Use ``--max-audio-seconds N`` with ``--audio`` to transcribe only the opening ``N`` seconds (faster tests).

For PDF: install **MuseScore** or **LilyPond** (``musicxml2ly`` + ``lilypond``).
MuseScore is picked up from ``PATH`` or ``/Applications/MuseScore*.app`` on macOS.

Checkpoints are often saved on CUDA; ``monsterpianotransformer`` calls ``torch.load`` without
``map_location``. This script temporarily forces ``map_location=cpu`` during ``load_model()``
so CPU-only machines (e.g. Mac without CUDA) can load weights.

``load_model`` defaults to ``device="cuda"``; this script passes ``device="cpu"`` when
``torch.cuda.is_available()`` is false so ``.to(device)`` does not trigger a CUDA build error.

No Hugging Face token required unless you configure gated downloads.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

_HF_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT, _HF_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _common  # noqa: E402

_PIANO_TRANS_REPO = "Genius-Society/piano_trans"


def _ensure_monster_piano_package() -> int | None:
    """Return exit code 1 if ``monsterpianotransformer`` is not installed."""
    if importlib.util.find_spec("monsterpianotransformer") is None:
        print(
            "The Monster Piano stack is shipped as a separate PyPI package.\n"
            "Install with:\n"
            "  pip install monsterpianotransformer\n",
            file=sys.stderr,
        )
        return 1
    return None


def _ensure_transcription_packages() -> int | None:
    missing: list[str] = []
    for name in ("huggingface_hub", "librosa", "piano_transcription_inference"):
        if importlib.util.find_spec(name) is None:
            missing.append(name)
            continue
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        print(
            "``--audio`` transcription needs extra packages:\n"
            f"  Missing: {', '.join(missing)}\n"
            "Install with:\n"
            "  pip install piano_transcription_inference huggingface_hub librosa\n",
            file=sys.stderr,
        )
        return 1
    return None


def _ensure_music21() -> int | None:
    if importlib.util.find_spec("music21") is None:
        print(
            "Sheet export needs ``music21``:\n"
            "  pip install music21\n",
            file=sys.stderr,
        )
        return 1
    return None


@contextmanager
def _torch_load_force_cpu():
    """``monsterpianotransformer.model_loader`` uses ``torch.load`` without ``map_location``."""
    import torch

    real_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs.setdefault("map_location", torch.device("cpu"))
        return real_load(*args, **kwargs)

    torch.load = patched_load  # type: ignore[method-assign]
    try:
        yield
    finally:
        torch.load = real_load  # type: ignore[method-assign]


def _hf_piano_trans_checkpoint(repo_id: str = _PIANO_TRANS_REPO) -> Path:
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(repo_id))
    paths = sorted(root.glob("*.pth"))
    if not paths:
        raise FileNotFoundError(f"No .pth weights under snapshot: {root}")
    return paths[0]


def _transcribe_audio_to_midi(
    *,
    audio_path: Path,
    checkpoint: Path,
    midi_out: Path,
    device: "torch.device",
    max_seconds: float | None,
) -> None:
    import librosa
    from piano_transcription_inference import PianoTranscription, sample_rate

    audio, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    if max_seconds is not None and max_seconds > 0:
        n = int(sample_rate * max_seconds)
        if len(audio) > n:
            audio = audio[:n]
    transcriptor = PianoTranscription(device=device, checkpoint_path=str(checkpoint))
    midi_out.parent.mkdir(parents=True, exist_ok=True)
    transcriptor.transcribe(audio, str(midi_out))


def _midi_to_pdf_or_musicxml(midi_path: Path, pdf_path: Path) -> str:
    """Return ``'pdf'``, ``'musicxml'``, or ``'failed'``. Mirrors ``backend/services/engrave.py``."""
    from music21 import converter

    from shared.musescore_cli import musescore_executable_paths

    score = converter.parse(str(midi_path))
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        base = tmp / "score"
        musicxml_written = Path(score.write("musicxml", fp=str(base)))
        if not musicxml_written.is_file():
            return "failed"

        pdf_tmp = tmp / "sheet.pdf"
        for exe in musescore_executable_paths():
            try:
                subprocess.run(
                    [exe, "-o", str(pdf_tmp), str(musicxml_written)],
                    check=True,
                    capture_output=True,
                    timeout=120,
                )
                if pdf_tmp.is_file():
                    pdf_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(pdf_tmp, pdf_path)
                    return "pdf"
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                continue

        if shutil.which("musicxml2ly") and shutil.which("lilypond"):
            try:
                ly_path = tmp / "score.ly"
                subprocess.run(
                    ["musicxml2ly", "-o", str(ly_path), str(musicxml_written)],
                    check=True,
                    capture_output=True,
                    timeout=60,
                )
                subprocess.run(
                    ["lilypond", "-o", str(tmp / "sheet"), str(ly_path)],
                    check=True,
                    capture_output=True,
                    timeout=120,
                )
                if pdf_tmp.is_file():
                    pdf_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(pdf_tmp, pdf_path)
                    return "pdf"
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                pass

        # No PDF renderer: keep MusicXML next to the intended PDF path.
        fallback = pdf_path.with_suffix(".musicxml")
        shutil.copyfile(musicxml_written, fallback)
        return "musicxml"


def main() -> int:
    p = _common.midi_arg()
    p.add_argument("--num-gen-tokens", type=int, default=600)
    p.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Optional piano recording (e.g. golden.mp3). When set, transcribe with "
        f"{_PIANO_TRANS_REPO} weights first; the result seeds Monster Piano.",
    )
    p.add_argument(
        "--hf-piano-trans-repo",
        default=_PIANO_TRANS_REPO,
        help="Hugging Face repo id for transcription weights (default: Genius-Society/piano_trans).",
    )
    p.add_argument(
        "--max-audio-seconds",
        type=float,
        default=None,
        metavar="SEC",
        help="With --audio, transcribe only the first SEC seconds (handy for smoke tests).",
    )
    p.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip sheet export (PDF or fallback MusicXML).",
    )
    p.add_argument(
        "--pdf-out",
        type=Path,
        default=None,
        help="PDF path (default: same stem as --out with .pdf).",
    )
    args = p.parse_args()

    if (code := _ensure_monster_piano_package()) is not None:
        return code

    if args.audio is not None and (code := _ensure_transcription_packages()) is not None:
        return code

    if not args.no_pdf and (code := _ensure_music21()) is not None:
        return code

    import torch

    import monsterpianotransformer as mpt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out = args.out or (_common.repo_root() / "hf_model_outputs" / "monster_piano_continuation.mid")
    out = Path(out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    seed_midi: Path
    if args.audio is not None:
        audio_in = _common.require_existing_file(Path(args.audio), "Audio")
        print(f"Resolving transcription checkpoint ({args.hf_piano_trans_repo}) …")
        ck = _hf_piano_trans_checkpoint(repo_id=args.hf_piano_trans_repo)
        transcribed = out.parent / f"{out.stem}_transcribed.mid"
        print(f"Transcribing: {audio_in} → {transcribed}")
        _transcribe_audio_to_midi(
            audio_path=audio_in,
            checkpoint=ck,
            midi_out=transcribed,
            device=device,
            max_seconds=args.max_audio_seconds,
        )
        seed_midi = transcribed
    else:
        seed_midi = _common.require_midi(args.midi)

    print(f"Loading default Monster Piano model (device={device}) …")
    with _torch_load_force_cpu():
        model = mpt.load_model(device=str(device))

    print(f"Tokenizing seed MIDI: {seed_midi}")
    input_tokens = mpt.midi_to_tokens(str(seed_midi))

    print(f"Generating ({args.num_gen_tokens} tokens) …")
    output_tokens = mpt.generate(
        model,
        input_tokens,
        num_gen_tokens=args.num_gen_tokens,
        return_prime=True,
    )

    base = str(out.parent / out.stem)
    mpt.tokens_to_midi(output_tokens[0], output_midi_name=base)
    print(f"Wrote {out}")

    if not args.no_pdf:
        pdf_out = args.pdf_out or out.with_suffix(".pdf")
        pdf_out = Path(pdf_out).expanduser().resolve()
        print(f"Rendering sheet music → {pdf_out} …")
        kind = _midi_to_pdf_or_musicxml(out, pdf_out)
        if kind == "pdf":
            print(f"Wrote PDF: {pdf_out}")
        elif kind == "musicxml":
            print(
                "No MuseScore CLI or LilyPond found — wrote MusicXML instead:\n"
                f"  {pdf_out.with_suffix('.musicxml')}\n"
                "Install MuseScore (macOS: drag to /Applications) or LilyPond for direct PDF export.",
            )
        else:
            print("Sheet export failed after MusicXML generation.", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
