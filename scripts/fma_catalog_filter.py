"""Filter the FMA metadata catalog for ``pop_mini_v0`` candidate tracks.

The Free Music Archive (https://freemusicarchive.org) ships a metadata
zip containing ``tracks.csv`` (~106k tracks). This script downloads the
zip once (cached under ``.cache/fma_metadata/``), reads ``tracks.csv``,
filters by license + genre + duration, and prints a markdown table of
candidates suitable for hand-picking.

The filter targets the Phase 0 manifest slots:

  * ``mini_pop_001`` — ``--genre Pop`` / mainstream
  * ``mini_pop_002`` — ``--genre Electronic`` (synth-pop)
  * ``mini_pop_003`` — ``--genre Soul-RnB``
  * ``mini_kpop_001`` — best-effort: filter by ``--genre Pop`` and search
    track tags client-side; FMA has limited K-pop coverage so YouTube
    CC-BY is usually the better source for that slot.
  * ``mini_ballad_001`` — ``--genre Folk`` or ``--genre Classical`` plus
    ``--max-tempo 90`` if you need a sparse-feel filter.

Commercial-use is detected by inspecting the license URL for the
``-nc`` modifier — anything containing ``/by-nc`` or ``/nc/`` is dropped
even if the user didn't pass ``--commercial-only`` (the only sane
default for the plan's ``commercial=true`` requirement).

Output is a markdown table that you copy URLs out of and feed to
``scripts/curate_pop_mini_v0.py``::

    python scripts/fma_catalog_filter.py --genre Pop --top 20
    python scripts/curate_pop_mini_v0.py mini_pop_001 \\
        https://freemusicarchive.org/track/<id>/ --license cc-by-4.0

Run ``--no-download`` if the zip is already cached but the host is
unreachable (or you've moved the zip in manually). Run ``--list-genres``
to print the top-genres histogram before filtering.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / ".cache" / "fma_metadata"

# Canonical metadata zip — Switch / EPFL host. If it 404s, the FMA
# README at https://github.com/mdeff/fma lists current mirrors. The
# script prints the README URL on download failure so the user can
# pick an active mirror without grepping the script.
DEFAULT_FMA_ZIP_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
DEFAULT_README_URL = "https://github.com/mdeff/fma#data"

# License URLs containing any of these substrings are non-commercial.
# Filtering on substrings is robust to the URL versioning differences
# (1.0 / 2.0 / 2.5 / 3.0 / 4.0) that FMA tracks span.
_NON_COMMERCIAL_MARKERS = ("/by-nc", "/nc/", "noncommercial")

# Default top-genres to surface in --list-genres if the user doesn't
# specify --genre. Keeps the histogram readable.
_TOP_GENRES_LIMIT = 25

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Download + cache
# ---------------------------------------------------------------------------

def _download_zip(url: str, target_path: Path) -> None:
    """Download the FMA metadata zip with a progress-aware fetcher.

    Uses ``requests`` if available (better progress + retry semantics);
    falls back to ``urllib.request`` from the stdlib so the script runs
    in a minimal environment.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".part")

    try:
        import requests  # noqa: PLC0415

        log.info("requests GET %s", url)
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length") or 0)
            written = 0
            with tmp_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if not chunk:
                        continue
                    f.write(chunk)
                    written += len(chunk)
                    if total:
                        pct = 100.0 * written / total
                        print(
                            f"\r  ↓ {written / 1e6:6.1f}MB / {total / 1e6:.1f}MB "
                            f"({pct:5.1f}%)", end="", flush=True,
                        )
            print()
    except ImportError:
        from urllib.request import urlopen  # noqa: PLC0415

        log.info("urllib GET %s (requests unavailable)", url)
        with urlopen(url, timeout=60) as resp, tmp_path.open("wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)

    tmp_path.replace(target_path)


def _ensure_metadata(*, no_download: bool, url: str) -> Path:
    """Return the path of an extracted ``tracks.csv``, downloading + unzipping if needed."""
    zip_path = CACHE_DIR / "fma_metadata.zip"
    extract_root = CACHE_DIR / "fma_metadata"
    tracks_csv = extract_root / "tracks.csv"

    if tracks_csv.is_file():
        return tracks_csv

    if not zip_path.is_file():
        if no_download:
            raise FileNotFoundError(
                f"--no-download set but {zip_path} not present. Manually drop "
                f"the FMA metadata zip there and rerun."
            )
        try:
            _download_zip(url, zip_path)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"FMA metadata download failed ({exc}). Mirrors are listed at "
                f"{DEFAULT_README_URL} — fetch manually and place at {zip_path}."
            ) from exc

    log.info("extracting %s -> %s", zip_path, extract_root)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(CACHE_DIR)

    if not tracks_csv.is_file():
        raise RuntimeError(
            f"Expected {tracks_csv} after extraction but it's missing. "
            f"The FMA zip schema may have changed; see {DEFAULT_README_URL}."
        )
    return tracks_csv


# ---------------------------------------------------------------------------
# Schema inspection — flatten FMA's multi-row CSV header
# ---------------------------------------------------------------------------

def _read_header(reader: csv.reader) -> list[str]:
    """Read FMA's two header rows and flatten to ``"<level>.<field>"`` keys.

    FMA's ``tracks.csv`` is a pandas multi-index ``header=[0, 1]`` dump:
    row 0 = section (``album`` / ``artist`` / ``set`` / ``track``), row 1 =
    field name (``id`` / ``title`` / ``listens`` / …). We collapse the two
    with a dot so column names match the FMA paper's documentation
    (``track.duration``, ``track.license``, ``artist.name``).

    The first column is the row index and has empty header cells in
    both rows; we name it ``track_id``.
    """
    row0 = next(reader)
    row1 = next(reader)
    if len(row0) != len(row1):
        raise RuntimeError(
            f"FMA header rows have mismatched widths: "
            f"row0={len(row0)} vs row1={len(row1)}"
        )

    headers: list[str] = []
    for level, field in zip(row0, row1):
        level_clean = level.strip().lower()
        field_clean = field.strip().lower().replace(" ", "_")
        if not field_clean and not level_clean:
            headers.append("track_id")
        elif not level_clean:
            headers.append(field_clean)
        else:
            headers.append(f"{level_clean}.{field_clean}")
    # The first column of FMA tracks.csv carries the integer track id —
    # both header rows are blank there, so our flattening produces an
    # empty key. Force it to 'track_id' so downstream lookups work.
    if headers[0] != "track_id":
        headers[0] = "track_id"
    return headers


@dataclass
class TrackRow:
    track_id: int
    title: str
    artist: str
    genre_top: str
    duration_sec: int
    license_url: str
    listens: int
    fma_track_url: str

    def commercial_ok(self) -> bool:
        return not any(m in self.license_url.lower() for m in _NON_COMMERCIAL_MARKERS)

    def license_short(self) -> str:
        """Return a compact license tag for the markdown table.

        ``http://creativecommons.org/licenses/by/4.0/`` → ``CC BY 4.0``.
        Public-domain CC0 → ``CC0``. Unknown URLs fall through verbatim.
        """
        u = self.license_url.lower()
        if "publicdomain/zero" in u or "creativecommons.org/publicdomain" in u:
            return "CC0"
        if "/by-nc-sa/" in u:
            return _with_version("CC BY-NC-SA", u)
        if "/by-nc-nd/" in u:
            return _with_version("CC BY-NC-ND", u)
        if "/by-nc/" in u:
            return _with_version("CC BY-NC", u)
        if "/by-sa/" in u:
            return _with_version("CC BY-SA", u)
        if "/by-nd/" in u:
            return _with_version("CC BY-ND", u)
        if "/by/" in u:
            return _with_version("CC BY", u)
        return self.license_url or "unknown"


def _with_version(prefix: str, url: str) -> str:
    """Append the trailing ``X.Y`` version segment if the URL has one."""
    parts = url.rstrip("/").split("/")
    for token in reversed(parts):
        if token and token[0].isdigit() and "." in token:
            return f"{prefix} {token}"
    return prefix


def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_track_row(headers: list[str], row: list[str]) -> TrackRow | None:
    """Build a TrackRow, skipping rows missing must-have fields."""
    record = dict(zip(headers, row))
    track_id_raw = record.get("track_id", "")
    if not track_id_raw or not track_id_raw.strip():
        return None
    try:
        track_id = int(track_id_raw)
    except ValueError:
        return None

    title = record.get("track.title", "").strip()
    artist = record.get("artist.name", "").strip()
    genre_top = record.get("track.genre_top", "").strip()
    duration = _safe_int(record.get("track.duration", "0"))
    license_url = record.get("track.license", "").strip()
    listens = _safe_int(record.get("track.listens", "0"))

    return TrackRow(
        track_id=track_id,
        title=title,
        artist=artist,
        genre_top=genre_top,
        duration_sec=duration,
        license_url=license_url,
        listens=listens,
        fma_track_url=f"https://freemusicarchive.org/track/{track_id}/",
    )


# ---------------------------------------------------------------------------
# Filter pipeline
# ---------------------------------------------------------------------------

def iter_tracks(tracks_csv: Path):
    """Yield :class:`TrackRow` records, skipping malformed lines."""
    with tracks_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = _read_header(reader)
        for row in reader:
            track = _to_track_row(headers, row)
            if track is not None:
                yield track


def _matches_genre(track: TrackRow, genre: str | None) -> bool:
    if genre is None:
        return True
    return track.genre_top.casefold() == genre.casefold()


def _matches_duration(track: TrackRow, min_sec: int, max_sec: int) -> bool:
    return min_sec <= track.duration_sec <= max_sec


def _matches_query(track: TrackRow, query: str | None) -> bool:
    if not query:
        return True
    q = query.casefold()
    return (
        q in track.title.casefold()
        or q in track.artist.casefold()
        or q in track.genre_top.casefold()
    )


def filter_tracks(
    tracks_csv: Path,
    *,
    genre: str | None,
    min_duration_sec: int,
    max_duration_sec: int,
    query: str | None,
    commercial_only: bool,
) -> list[TrackRow]:
    out: list[TrackRow] = []
    for track in iter_tracks(tracks_csv):
        if commercial_only and not track.commercial_ok():
            continue
        if not _matches_genre(track, genre):
            continue
        if not _matches_duration(track, min_duration_sec, max_duration_sec):
            continue
        if not _matches_query(track, query):
            continue
        out.append(track)
    return out


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def render_markdown(tracks: list[TrackRow], *, top_n: int) -> str:
    """Render a markdown table sorted by listens desc, capped at ``top_n``."""
    sorted_tracks = sorted(tracks, key=lambda t: -t.listens)[:top_n]
    if not sorted_tracks:
        return "_(no tracks matched the filters)_\n"
    lines = [
        "| listens | duration | genre | license | artist | title | url |",
        "|--:|--:|---|---|---|---|---|",
    ]
    for t in sorted_tracks:
        m, s = divmod(t.duration_sec, 60)
        lines.append(
            "| {listens} | {dur} | {genre} | {lic} | {artist} | {title} | {url} |".format(
                listens=f"{t.listens:,}",
                dur=f"{m}:{s:02d}",
                genre=_md_escape(t.genre_top),
                lic=_md_escape(t.license_short()),
                artist=_md_escape(t.artist),
                title=_md_escape(t.title),
                url=t.fma_track_url,
            )
        )
    return "\n".join(lines) + "\n"


def _md_escape(value: str) -> str:
    """Escape pipe characters that would break the markdown table row."""
    return value.replace("|", r"\|")


def list_top_genres(tracks_csv: Path, *, limit: int = _TOP_GENRES_LIMIT) -> str:
    """Print the top-N genre histogram. Useful before deciding ``--genre``."""
    counter: Counter[str] = Counter()
    for track in iter_tracks(tracks_csv):
        if track.genre_top:
            counter[track.genre_top] += 1
    pairs = counter.most_common(limit)
    width = max(len(g) for g, _ in pairs) if pairs else 0
    lines = [f"  {g:<{width}}  {n:>7,}" for g, n in pairs]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--genre", type=str, default=None,
        help="Filter to this top-genre exactly (case-insensitive). e.g. Pop, Electronic, Folk.",
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Substring match against title/artist/genre.",
    )
    parser.add_argument(
        "--min-duration", type=int, default=60,
        help="Minimum track duration in seconds (default: 60).",
    )
    parser.add_argument(
        "--max-duration", type=int, default=360,
        help="Maximum track duration in seconds (default: 360).",
    )
    parser.add_argument(
        "--top", type=int, default=20,
        help="Cap output to the N most-listened candidates after filtering (default: 20).",
    )
    parser.add_argument(
        "--allow-non-commercial", action="store_true",
        help="Include CC BY-NC* and similar licenses in the output (off by default).",
    )
    parser.add_argument(
        "--list-genres", action="store_true",
        help="Print the top-genre histogram and exit (no track table).",
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Don't fetch the FMA zip — fail if the cache is empty.",
    )
    parser.add_argument(
        "--url", type=str, default=DEFAULT_FMA_ZIP_URL,
        help=f"Override the FMA metadata zip URL (default: {DEFAULT_FMA_ZIP_URL}).",
    )
    args = parser.parse_args()

    tracks_csv = _ensure_metadata(no_download=args.no_download, url=args.url)

    if args.list_genres:
        print(list_top_genres(tracks_csv))
        return 0

    print(f"# FMA candidates ({tracks_csv})\n")
    print(f"Filters: genre={args.genre!r} duration={args.min_duration}-{args.max_duration}s "
          f"commercial_only={not args.allow_non_commercial}\n")

    matches = filter_tracks(
        tracks_csv,
        genre=args.genre,
        min_duration_sec=args.min_duration,
        max_duration_sec=args.max_duration,
        query=args.query,
        commercial_only=not args.allow_non_commercial,
    )
    print(f"Matched {len(matches)} tracks; showing top {min(args.top, len(matches))} by listens.\n")
    print(render_markdown(matches, top_n=args.top))

    print(
        "\nFeed a candidate URL to the curate script:\n"
        "  python scripts/curate_pop_mini_v0.py <slug> <url> --license cc-by-4.0\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
