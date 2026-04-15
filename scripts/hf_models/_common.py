"""Shared helpers for one-off Hugging Face model smoke scripts.

**Golden files** (both live at :func:`repo_root`):

- ``golden.mid`` — Use for **symbolic** pipelines: anything that tokenizes, continues, or
  renders from a **score/performance MIDI** (Aria, Monster Piano, Pianist, Orpheus,
  PianoBART, RC prompt helper).

- ``golden.mp3`` — Use only when the model is **audio-native** (Pop2Piano: mel / encoder
  expects a waveform). Other runners do not read ``golden.mp3`` unless you wire a custom
  path or transcribe audio → MIDI yourself first.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_midi_path() -> Path:
    """Repo-root ``golden.mid`` (symbolic / score default)."""
    return repo_root() / "golden.mid"


def default_audio_path() -> Path:
    """Repo-root ``golden.mp3`` (audio-native default, e.g. Pop2Piano)."""
    return repo_root() / "golden.mp3"


def midi_arg(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    p = parser or argparse.ArgumentParser()
    p.add_argument(
        "--midi",
        type=Path,
        default=default_midi_path(),
        help="Input MIDI (default: golden.mid at repo root)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: model-specific under repo root)",
    )
    return p


def require_existing_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def require_midi(path: Path) -> Path:
    return require_existing_file(path, "MIDI")
