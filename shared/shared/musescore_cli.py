"""Resolve MuseScore CLI for headless PDF export.

The macOS ``.app`` bundle ships ``Contents/MacOS/mscore``, which is usually **not** on
``$PATH``. We probe ``shutil.which`` first, then standard install locations.
"""

from __future__ import annotations

import functools
import os
import shutil
import sys
from pathlib import Path

_WHICH_NAMES = ("musescore4", "musescore3", "mscore", "MuseScore4")


def _macos_bundle_candidates() -> list[Path]:
    roots = (Path("/Applications"), Path.home() / "Applications")
    # (bundle folder name, binary names inside Contents/MacOS — order matters)
    app_bins: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("MuseScore 4.app", ("mscore", "MuseScore4")),
        ("MuseScore 3.app", ("mscore", "MuseScore3")),
        ("MuseScore.app", ("mscore", "MuseScore4")),
    )
    found: list[Path] = []
    for root in roots:
        for app, bins in app_bins:
            macos = root / app / "Contents" / "MacOS"
            if not macos.is_dir():
                continue
            for name in bins:
                p = macos / name
                if p.is_file() and os.access(p, os.X_OK):
                    found.append(p)
    return found


@functools.lru_cache(maxsize=1)
def musescore_executable_paths() -> list[str]:
    """Absolute paths to MuseScore-style CLIs, in preference order (deduplicated).

    Set ``MUSESCORE_PATH`` to a full path to force a specific binary (e.g. a non-default
    install or a wrapper script).
    """
    out: list[str] = []
    seen: set[str] = set()
    override = os.environ.get("MUSESCORE_PATH", "").strip()
    if override:
        p = Path(override).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            s = str(p.resolve())
            out.append(s)
            seen.add(s)
    for name in _WHICH_NAMES:
        w = shutil.which(name)
        if w and w not in seen:
            out.append(w)
            seen.add(w)
    if sys.platform == "darwin":
        for p in _macos_bundle_candidates():
            s = str(p.resolve())
            if s not in seen:
                out.append(s)
                seen.add(s)
    return out
