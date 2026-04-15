#!/usr/bin/env python3
"""
Run the three HF smoke scripts as a **strict sequential chain** (each stage's output
path is the next stage's ``--midi`` input):

1. **Pop2Piano** → writes ``01_pop2piano.mid``
2. **Monster Piano Transformer** — ``--midi`` = that file → writes ``02_monster_piano.mid``
3. **Pianist Transformer** — ``--midi`` = Monster Piano's file → writes ``03_pianist.mid`` (or ``--out``)

There is no parallel mix: step *n* does not start until step *n−1* exits successfully.
Intermediate ``.mid`` files are only the handoff between subprocesses.

Each step invokes the matching ``run_*.py`` in this directory with ``subprocess`` so
their isolated installs and side effects (e.g. Pianist's ``chdir``) stay contained.

Example::

  python scripts/hf_models/run_combo_pop2piano_mpt_pianist.py --audio ./golden.mp3

Install everything each runner documents (Pop2Piano + essentia, ``monsterpianotransformer``,
Pianist deps + git clone on first run).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_HF_DIR = Path(__file__).resolve().parent
for _p in (_HF_DIR,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _common  # noqa: E402


def _safe_stem(path: Path) -> str:
    raw = path.stem or "combo"
    out = "".join(c if c.isalnum() or c in "._-" else "_" for c in raw)
    return (out[:80] or "combo").strip("_") or "combo"


def _run_step(title: str, argv: list[str]) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}", flush=True)
    subprocess.run(argv, check=True)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Chain Pop2Piano → Monster Piano Transformer → Pianist Transformer.",
    )
    p.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Input audio for Pop2Piano (wav/mp3/…). Ignored if --from-midi is set.",
    )
    p.add_argument(
        "--from-midi",
        type=Path,
        default=None,
        metavar="PATH.mid",
        help="Pop2Piano: synthesize this MIDI to audio instead of --audio.",
    )
    p.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory for intermediate MIDIs (default: hf_model_outputs/combo_<stem>/).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Final MIDI path (default: <work-dir>/03_pianist.mid).",
    )
    p.add_argument("--composer", type=str, default="composer1", help="Pop2Piano --composer")
    p.add_argument("--sr", type=int, default=44100, help="Pop2Piano target sample rate")
    p.add_argument(
        "--num-gen-tokens",
        type=int,
        default=600,
        help="Monster Piano continuation length",
    )
    args = p.parse_args()

    exe = sys.executable
    if args.from_midi is not None:
        midi_src = _common.require_midi(args.from_midi)
        stem = _safe_stem(midi_src)
    else:
        if args.audio is None:
            audio = _common.default_audio_path()
        else:
            audio = Path(args.audio).expanduser().resolve()
        audio = _common.require_existing_file(audio, "Audio")
        stem = _safe_stem(audio)

    work = args.work_dir or (_common.repo_root() / "hf_model_outputs" / f"combo_{stem}")
    work = Path(work).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)

    out_final = args.out or (work / "03_pianist.mid")
    out_final = Path(out_final).expanduser().resolve()
    out_final.parent.mkdir(parents=True, exist_ok=True)

    p2p = work / "01_pop2piano.mid"
    mpt = work / "02_monster_piano.mid"

    pop_argv = [
        exe,
        str(_HF_DIR / "run_pop2piano.py"),
        "--out",
        str(p2p),
        "--composer",
        args.composer,
        "--sr",
        str(args.sr),
    ]
    if args.from_midi is not None:
        pop_argv.extend(["--from-midi", str(midi_src)])
    else:
        pop_argv.extend(["--audio", str(audio)])

    mpt_argv = [
        exe,
        str(_HF_DIR / "run_monster_piano_transformer.py"),
        "--midi",
        str(p2p),
        "--out",
        str(mpt),
        "--no-pdf",
        "--num-gen-tokens",
        str(args.num_gen_tokens),
    ]

    pt_argv = [
        exe,
        str(_HF_DIR / "run_pianist_transformer.py"),
        "--midi",
        str(mpt),
        "--out",
        str(out_final),
    ]

    try:
        _run_step("1/3 Pop2Piano", pop_argv)
        _run_step("2/3 Monster Piano Transformer", mpt_argv)
        _run_step("3/3 Pianist Transformer", pt_argv)
    except subprocess.CalledProcessError as e:
        print(f"\nCombo pipeline stopped: step failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode or 1

    print(f"\nDone. Final MIDI: {out_final}")
    print(f"Intermediates: {p2p} , {mpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
