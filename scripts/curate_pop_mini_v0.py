"""Curate a real-audio source for a ``pop_mini_v0`` manifest slot.

Replaces the bootstrap ``synthetic_from_midi`` source for one slot in
``eval/pop_mini_v0/manifest.yaml`` with a curated ``audio_file`` source.
Each invocation handles one slug; rerun the script per slug as you
collect candidates.

Usage::

    # YouTube / SoundCloud / FMA / any yt-dlp-supported URL.
    # The script extracts audio, postprocesses to MP3, and drops it at
    # eval/pop_mini_v0/songs/<slug>/source.audio.mp3.
    python scripts/curate_pop_mini_v0.py mini_kpop_001 \\
        https://www.youtube.com/watch?v=... \\
        --license cc-by

    # Direct HTTP(S) audio URL (FMA mp3, etc.) — yt-dlp handles via
    # generic extractor; falls back to plain requests if extraction fails.
    python scripts/curate_pop_mini_v0.py mini_pop_001 \\
        https://freemusicarchive.org/file/.../Track.mp3 \\
        --license cc-by-4.0 \\
        --title "Track Title" \\
        --artist "Artist Name"

    # Local file you've already downloaded.
    python scripts/curate_pop_mini_v0.py mini_ballad_001 \\
        ~/Downloads/sparse_ballad.flac \\
        --license cc-by-4.0

The script:

  1. Downloads / copies the source to
     ``eval/pop_mini_v0/songs/<slug>/source.audio.<ext>`` (creates the
     directory tree if missing).
  2. Computes the SHA-256 of the resulting file.
  3. Rewrites ``manifest.yaml`` in place: replaces only the
     ``source:`` subblock for the matching slug. Comments, blank lines,
     ``intended_source``, and every other song entry are preserved
     verbatim — the rewrite is line-level surgery, not a YAML round-trip.
  4. Re-running the eval CLI now picks up the real audio:
     ``python scripts/eval_mini.py eval/pop_mini_v0/ eval/runs/<id>/``.

After all 5 slots are curated, snapshot a fresh baseline and commit it
as ``eval/baselines/pop_mini_v0__main_<sha>.json`` next to the bootstrap
baseline so you can compare bootstrap vs. real-audio numbers.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import logging
import shutil
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

log = logging.getLogger(__name__)

# Audio formats we accept on the input side. yt-dlp normalises everything to
# MP3 via the FFmpegExtractAudio postprocessor; local files are copied as-is
# so a user-provided WAV / FLAC stays lossless.
_LOSSLESS_EXTS = {".wav", ".flac"}
_LOSSY_EXTS = {".mp3", ".m4a", ".aac", ".ogg", ".opus"}
_ACCEPTED_EXTS = _LOSSLESS_EXTS | _LOSSY_EXTS

_HASH_CHUNK_BYTES = 1 << 20  # 1 MiB


# ---------------------------------------------------------------------------
# Source resolution: URL → yt-dlp → local mp3, OR local file → copy
# ---------------------------------------------------------------------------

def _is_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"}


def _download_via_ytdlp(url: str, target_dir: Path, target_stem: str) -> Path:
    """Use yt-dlp to download + transcode to MP3 at ``<target_dir>/<stem>.mp3``.

    yt-dlp's generic extractor handles direct MP3 URLs (e.g. FMA) as well
    as platform URLs (YouTube, SoundCloud, Bandcamp, …) — one code path
    covers both. Postprocessor pins the output container to MP3 so the
    suffix is predictable downstream.
    """
    try:
        import yt_dlp  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlp not installed. Install with `pip install yt-dlp` or "
            "`make install` (it's already pinned in pyproject.toml)."
        ) from exc

    target_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(target_dir / f"{target_stem}.%(ext)s")

    ydl_opts: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        # Don't write thumbnails / description sidecars — we only want
        # the audio file at the curation path.
        "writethumbnail": False,
        "writedescription": False,
        "writeinfojson": False,
    }

    log.info("yt-dlp → %s", url)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)

    out_path = target_dir / f"{target_stem}.mp3"
    if not out_path.is_file():
        # FFmpeg postprocess can occasionally land at a different ext
        # if the source was already mp3; fall back to glob.
        for candidate in target_dir.glob(f"{target_stem}.*"):
            if candidate.suffix.lower() in _ACCEPTED_EXTS:
                return candidate
        raise RuntimeError(
            f"yt-dlp completed but no audio landed at {out_path} "
            f"(target_dir contents: {list(target_dir.iterdir())})"
        )
    return out_path


def _copy_local_file(source_path: Path, target_dir: Path, target_stem: str) -> Path:
    """Copy a local audio file into the manifest tree, preserving extension."""
    if not source_path.is_file():
        raise FileNotFoundError(f"Local source missing: {source_path}")
    ext = source_path.suffix.lower()
    if ext not in _ACCEPTED_EXTS:
        raise ValueError(
            f"Unsupported audio extension: {ext!r}. "
            f"Accepted: {sorted(_ACCEPTED_EXTS)}"
        )
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{target_stem}{ext}"
    shutil.copy2(source_path, target_path)
    log.info("copied %s -> %s", source_path, target_path)
    return target_path


def _resolve_source(
    source: str,
    target_dir: Path,
    target_stem: str,
) -> Path:
    if _is_url(source):
        return _download_via_ytdlp(source, target_dir, target_stem)
    return _copy_local_file(Path(source).expanduser().resolve(), target_dir, target_stem)


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_HASH_CHUNK_BYTES):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Manifest rewrite — line-level surgery so comments + intended_source survive
# ---------------------------------------------------------------------------

def _find_song_slug_line(lines: list[str], slug: str) -> int:
    """Return the 0-based index of ``  - slug: <slug>`` in the manifest.

    Tolerates double / single quoted slug variants because PyYAML may
    write them either way during a future round-trip.
    """
    candidates = (
        f"  - slug: {slug}",
        f'  - slug: "{slug}"',
        f"  - slug: '{slug}'",
    )
    for i, line in enumerate(lines):
        stripped = line.rstrip("\n").rstrip()
        if stripped in candidates:
            return i
    raise ValueError(f"slug {slug!r} not found in manifest")


def _next_song_index(lines: list[str], from_index: int) -> int:
    """Return the index of the next ``  - slug:`` line, or ``len(lines)``."""
    for j in range(from_index + 1, len(lines)):
        if lines[j].startswith("  - slug:"):
            return j
    return len(lines)


def _find_source_block(
    lines: list[str], song_start: int, song_end: int,
) -> tuple[int, int]:
    """Locate the ``    source:`` block within a song.

    Returns ``(start_index, end_index)`` such that
    ``lines[start_index:end_index]`` covers the entire source block AND
    any trailing blank line that visually separates it from the next
    song. This lets the caller splice in a replacement that ends with
    a single blank line and keeps the manifest's between-songs spacing
    intact.
    """
    source_start = -1
    for k in range(song_start + 1, song_end):
        if lines[k].rstrip("\n").rstrip() == "    source:":
            source_start = k
            break
    if source_start == -1:
        raise ValueError("source: block not found within song range")

    # Walk forward consuming 6+ space-indented lines and any blank lines
    # that may be embedded inside source (multi-line scalar bodies use
    # 8-space indent but blank lines inside YAML scalars are unusual;
    # treat blanks as continuation until we see a 4-space sibling or the
    # next song).
    source_end = song_end
    for m in range(source_start + 1, song_end):
        line = lines[m]
        if not line.strip():
            continue
        if not line.startswith("      "):
            # First sibling at <=4 indent → end of source block.
            source_end = m
            break
    return source_start, source_end


def _format_audio_source_block(
    *,
    relpath: str,
    sha256_hex: str,
    license_str: str,
    source_url: str | None,
    curated_at_iso: str,
    notes: str | None,
) -> str:
    """Render the new ``source:`` subblock with 4-/6-space indentation.

    Returns the block including a trailing blank line so the splice
    preserves the visual separator between songs in the manifest.
    """
    lines = [
        "    source:\n",
        "      kind: audio_file\n",
        f"      path: {relpath}\n",
        f"      content_hash: sha256:{sha256_hex}\n",
        f"      license: {license_str}\n",
    ]
    if source_url:
        # YAML accepts bare URLs (no scheme prefix that conflicts with
        # tags like '!!') so a plain assignment is round-trip safe.
        lines.append(f"      source_url: {source_url}\n")
    lines.append(f"      curated_at: {curated_at_iso}\n")
    lines.append("      curated_via: scripts/curate_pop_mini_v0.py\n")
    if notes:
        lines.append("      notes: |\n")
        for note_line in notes.splitlines():
            lines.append(f"        {note_line}\n")
    lines.append("      bootstrap: false\n")
    lines.append("\n")  # trailing blank line — preserves between-songs spacing
    return "".join(lines)


def _rewrite_manifest_source(
    manifest_path: Path,
    slug: str,
    new_source_block: str,
) -> None:
    """In-place replace the ``source:`` subblock for ``slug``.

    Validates that the manifest still parses as YAML and that the song
    is now ``kind: audio_file`` after the rewrite.
    """
    text = manifest_path.read_text()
    lines = text.splitlines(keepends=True)

    song_start = _find_song_slug_line(lines, slug)
    song_end = _next_song_index(lines, song_start)
    source_start, source_end = _find_source_block(lines, song_start, song_end)

    new_lines = lines[:source_start] + [new_source_block] + lines[source_end:]
    new_text = "".join(new_lines)

    # Validate the result parses cleanly before committing it to disk.
    parsed = yaml.safe_load(new_text)
    if not isinstance(parsed, dict) or "songs" not in parsed:
        raise RuntimeError(
            "post-rewrite manifest failed to parse as a mapping with `songs`"
        )
    matched = next(
        (s for s in parsed["songs"] if s.get("slug") == slug), None,
    )
    if matched is None:
        raise RuntimeError(f"slug {slug!r} disappeared after rewrite")
    if matched.get("source", {}).get("kind") != "audio_file":
        raise RuntimeError(
            f"slug {slug!r} source.kind is {matched.get('source', {}).get('kind')!r} "
            f"after rewrite (expected 'audio_file')"
        )

    manifest_path.write_text(new_text)


# ---------------------------------------------------------------------------
# Top-level curate() — what main() and tests both call
# ---------------------------------------------------------------------------

def curate(
    slug: str,
    source: str,
    *,
    manifest_path: Path,
    license_str: str,
    title: str | None = None,
    artist: str | None = None,
    notes: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Resolve, install, hash, and patch the manifest for one slug.

    Returns the new ``source`` dict that landed in the manifest, plus
    the absolute path of the curated audio file. The returned dict is
    handy for tests asserting the post-rewrite shape without re-reading
    the manifest.
    """
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    manifest = yaml.safe_load(manifest_path.read_text())
    songs = manifest.get("songs") or []
    matching = next((s for s in songs if s.get("slug") == slug), None)
    if matching is None:
        known = [s.get("slug") for s in songs]
        raise ValueError(
            f"slug {slug!r} not found in manifest. Known slugs: {known}"
        )

    manifest_dir = manifest_path.parent
    songs_dir = manifest_dir / "songs" / slug
    target_stem = "source.audio"

    # Refuse to clobber a previously-curated audio unless --force.
    if songs_dir.is_dir():
        existing = [
            p for p in songs_dir.iterdir()
            if p.is_file() and p.stem == target_stem and p.suffix.lower() in _ACCEPTED_EXTS
        ]
        if existing and not force:
            raise FileExistsError(
                f"audio already curated at {existing[0]} — pass --force to overwrite"
            )
        for p in existing:
            p.unlink()

    audio_path = _resolve_source(source, songs_dir, target_stem)
    sha256_hex = _sha256_of(audio_path)

    # Manifest paths are relative to the manifest directory so an
    # eval-set folder is relocatable.
    rel_path = audio_path.relative_to(manifest_dir)

    # Compose notes — preserve provenance even when the user doesn't
    # pass --notes explicitly.
    composed_notes = []
    if title:
        composed_notes.append(f"title: {title}")
    if artist:
        composed_notes.append(f"artist: {artist}")
    if notes:
        composed_notes.append(notes)
    notes_block = "\n".join(composed_notes) if composed_notes else None

    source_url = source if _is_url(source) else None
    curated_at_iso = (
        dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
    )

    new_block = _format_audio_source_block(
        relpath=str(rel_path),
        sha256_hex=sha256_hex,
        license_str=license_str,
        source_url=source_url,
        curated_at_iso=curated_at_iso,
        notes=notes_block,
    )

    _rewrite_manifest_source(manifest_path, slug, new_block)

    # Build the dict-shape the test asserts against.
    new_source: dict[str, Any] = {
        "kind": "audio_file",
        "path": str(rel_path),
        "content_hash": f"sha256:{sha256_hex}",
        "license": license_str,
        "curated_at": curated_at_iso,
        "curated_via": "scripts/curate_pop_mini_v0.py",
        "bootstrap": False,
    }
    if source_url:
        new_source["source_url"] = source_url
    if notes_block:
        new_source["notes"] = notes_block + "\n"  # PyYAML's | scalar adds trailing nl
    return {
        "audio_path": audio_path,
        "source": new_source,
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("slug", type=str, help="Manifest slug to curate (e.g. mini_pop_001).")
    parser.add_argument(
        "source",
        type=str,
        help="URL (yt-dlp-supported) or local audio file path.",
    )
    parser.add_argument(
        "--license",
        dest="license_str",
        type=str,
        required=True,
        help=(
            "License identifier for the curated audio (e.g. cc-by-4.0, cc0, "
            "cc-by-sa-3.0, public-domain, sync-licensed). Required so the "
            "manifest never carries an unlicensed source."
        ),
    )
    parser.add_argument(
        "--title", type=str, default=None,
        help="Optional title; recorded in the source.notes block.",
    )
    parser.add_argument(
        "--artist", type=str, default=None,
        help="Optional artist; recorded in the source.notes block.",
    )
    parser.add_argument(
        "--notes", type=str, default=None,
        help="Optional free-form provenance notes.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "eval" / "pop_mini_v0" / "manifest.yaml",
        help="Path to manifest.yaml (default: eval/pop_mini_v0/manifest.yaml).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite a previously-curated audio file at the slug's path.",
    )
    args = parser.parse_args()

    result = curate(
        slug=args.slug,
        source=args.source,
        manifest_path=args.manifest,
        license_str=args.license_str,
        title=args.title,
        artist=args.artist,
        notes=args.notes,
        force=args.force,
    )

    print()
    print(f"Curated {args.slug}:")
    print(f"  audio:        {result['audio_path']}")
    print(f"  content_hash: {result['source']['content_hash']}")
    print(f"  license:      {result['source']['license']}")
    print(f"  manifest:     {args.manifest}")
    print()
    print("Next: re-run the eval and snapshot a fresh baseline:")
    print(
        f"  python scripts/eval_mini.py {args.manifest.parent} "
        f"eval/runs/$(date -u +%Y%m%dT%H%M%SZ)/"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
