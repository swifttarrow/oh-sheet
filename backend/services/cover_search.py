"""Piano cover search — the ingest stage's "fast path" data source router.

When the user submits a YouTube URL with ``prefer_clean_source=True``,
this module searches YouTube for a clean piano cover of the same song
and returns the best candidate (if any). The ingest stage then swaps
the user's original URL for the cover's URL so Basic Pitch receives a
monophonic piano recording instead of a polyphonic full-band mix.

Why this helps transcription quality:
    Basic Pitch is a polyphonic pitch tracker, but it transcribes every
    audible pitch — including drum fundamentals, vocal harmonics, and
    bass subharmonics — as piano notes. On a pop-song mix this yields
    a dense, unplayable result. A piano cover is already monophonic
    (or polyphonic but piano-only), so Basic Pitch's output maps
    cleanly to sheet music without the instrument-confusion problem.

Scoring policy (see ``score_candidate``):
    +50  channel is in ``COVER_CHANNEL_ALLOWLIST``
    +30  title contains "piano cover" / "piano arrangement" / "solo piano"
    +20  wanted song title is a substring of the found video title
    +10  artist name appears in the found title or the channel name
    -20  title contains "karaoke" / "tutorial" / "how to play" / "lesson"

Default threshold: score >= 60 triggers a URL swap. Dry-run testing
against real YouTube (see scripts/dryrun_cover_search.py) showed the
original 70 was too strict — legitimate pop covers from non-allowlist
channels cap at exactly 60 (+30 piano cover + +20 title + +10 artist),
so 70 silently rejected them. At 60, junk content (karaoke/tutorial)
still can't clear because the -20 penalty brings them to 40 or below.
The threshold is passed as a parameter so callers (or config) can
tune strictness without editing this module.

Silent failure contract:
    ``find_piano_cover`` returns ``None`` on any failure (network error,
    yt-dlp exception, no candidates, all below threshold). The caller
    must interpret ``None`` as "fall back to direct transcription of the
    original URL." No exceptions propagate out — this is a hint, not a
    hard dependency.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Channel allowlist
# ---------------------------------------------------------------------------
#
# These channels post high-quality piano covers that Basic Pitch transcribes
# well. Seeded from manual inspection of the piano-cover YouTube ecosystem.
# All entries are lowercase and matched case-insensitively as substrings of
# the candidate's channel field, so "Rousseau" matches "Rousseau - Official".
#
# Tune this list in one place to change the "trusted" signal weight.

COVER_CHANNEL_ALLOWLIST: tuple[str, ...] = (
    # Cinematic / classical piano covers — full arrangements with bass,
    # melody, and harmony. Best match for Oh Sheet's two-hand output.
    "rousseau",
    "patrik pietschmann",
    "kyle landry",
    "peter buka",
    "lord vinheteiro",
    "david solís",
    "david solis",  # ASCII fallback
    "the piano guys",
    # Pop / contemporary piano covers — Billboard hits and TikTok viral.
    "jacob's piano",
    "jacobs piano",
    "akmigone",
    "francesco parrino",
    "martin walsh",
    "adam chen",
    "dotted8th",
    "pianote",
)


# ---------------------------------------------------------------------------
# Title normalization
# ---------------------------------------------------------------------------


# Noise patterns to strip from titles before comparing. Order matters:
# more-specific patterns first so they don't get accidentally matched by
# a generic one.
_NOISE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "(Official Music Video)", "[Official Video]", " - Official Video"
    re.compile(r"\s*[\[\(]\s*official\s+(music\s+)?video\s*[\]\)]", re.IGNORECASE),
    re.compile(r"\s*-\s*official\s+(music\s+)?video", re.IGNORECASE),
    # "(Lyrics)", "[Lyric Video]"
    re.compile(r"\s*[\[\(]\s*lyrics?\s*(video)?\s*[\]\)]", re.IGNORECASE),
    # "(HD)", "[4K Remaster]", "(Remastered 2020)"
    re.compile(r"\s*[\[\(]\s*\d*k?\s*remaster(ed)?\s*\d*\s*[\]\)]", re.IGNORECASE),
    re.compile(r"\s*[\[\(]\s*\d+k\s*[\]\)]", re.IGNORECASE),
    re.compile(r"\s*[\[\(]\s*hd\s*[\]\)]", re.IGNORECASE),
    # "(Live at Wembley 1988)"
    re.compile(r"\s*[\[\(]\s*live[^\]\)]*[\]\)]", re.IGNORECASE),
    # "feat. Jay-Z", "(feat. Jay-Z)", "ft. Jay-Z"
    re.compile(r"\s*[\[\(]\s*(feat\.?|ft\.?)[^\]\)]*[\]\)]", re.IGNORECASE),
    re.compile(r"\s+(feat\.?|ft\.?)\s+.*$", re.IGNORECASE),
)


def normalize_title(raw: str | None) -> str:
    """Strip noise tokens from a song title and lowercase the result.

    Returns ``""`` for ``None`` or empty input. The output is safe to use
    as a substring-comparison key against other normalized titles.
    """
    if not raw:
        return ""
    cleaned = raw
    for pattern in _NOISE_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    # Collapse whitespace runs and strip leading/trailing space.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.lower()


# ---------------------------------------------------------------------------
# Scoring rules
# ---------------------------------------------------------------------------


# Positive-signal keywords that indicate "this really is a piano cover."
_PIANO_COVER_KEYWORDS: tuple[str, ...] = (
    "piano cover",
    "piano arrangement",
    "solo piano",
)

# Negative-signal keywords that indicate "this is NOT what we want."
_BAD_KEYWORDS: tuple[str, ...] = (
    "karaoke",
    "tutorial",
    "how to play",
    "lesson",
)


def score_candidate(
    entry: dict[str, Any],
    wanted_title: str,
    wanted_artist: str | None,
) -> int:
    """Score a yt-dlp search result entry against what we're looking for.

    See the module docstring for the canonical rule list. Returns an
    integer score; callers compare this against a minimum threshold.
    """
    title_norm = normalize_title(entry.get("title", ""))
    channel_norm = (entry.get("channel") or "").lower()
    wanted_title_norm = normalize_title(wanted_title)
    wanted_artist_norm = (wanted_artist or "").lower().strip()

    score = 0

    # +50 if the channel is in the allowlist (substring match so
    # "Rousseau - Official" still matches "rousseau").
    if any(trusted in channel_norm for trusted in COVER_CHANNEL_ALLOWLIST):
        score += 50

    # +30 if any piano-cover keyword appears in the title.
    if any(kw in title_norm for kw in _PIANO_COVER_KEYWORDS):
        score += 30

    # +20 if the wanted song title appears in the found title.
    if wanted_title_norm and wanted_title_norm in title_norm:
        score += 20

    # +10 if the artist name appears in the title or channel.
    if wanted_artist_norm and (
        wanted_artist_norm in title_norm or wanted_artist_norm in channel_norm
    ):
        score += 10

    # -20 for any bad keyword — karaoke, tutorials, and lessons are not
    # cover recordings and will confuse Basic Pitch worse than the mix.
    if any(bad in title_norm for bad in _BAD_KEYWORDS):
        score -= 20

    return score


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverSearchResult:
    """Successful cover match. The URL is ready to feed into yt-dlp for
    download; the score is included so callers can log or display it."""

    url: str
    score: int
    channel: str
    title: str


def find_piano_cover(
    title: str,
    artist: str | None,
    *,
    min_score: int = 60,
    top_k: int = 5,
) -> CoverSearchResult | None:
    """Search YouTube for a piano cover of ``title`` (optionally by ``artist``).

    Returns the highest-scoring candidate whose score >= ``min_score``, or
    ``None`` if nothing clears the threshold or the search itself fails.
    This function NEVER raises — silent failure is the contract so the
    caller can fall back to direct transcription.
    """
    # Build the search query. Include the artist when we have it — it
    # narrows results without making scoring depend on it.
    if artist:
        query = f"{title} {artist} piano cover"
    else:
        query = f"{title} piano cover"

    try:
        entries = _yt_dlp_search(query, top_k=top_k)
    except Exception as exc:  # noqa: BLE001 — this is the silent-failure boundary
        log.warning("cover_search: yt-dlp search failed for %r: %s", query, exc)
        return None

    if not entries:
        log.info("cover_search: no search results for %r", query)
        return None

    # Score every candidate, pick the best, check the threshold.
    scored: list[tuple[int, dict[str, Any]]] = [
        (score_candidate(e, wanted_title=title, wanted_artist=artist), e)
        for e in entries
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    best_score, best_entry = scored[0]
    if best_score < min_score:
        log.info(
            "cover_search: best candidate for %r scored %d, below threshold %d",
            query, best_score, min_score,
        )
        return None

    return CoverSearchResult(
        url=best_entry.get("url", ""),
        score=best_score,
        channel=best_entry.get("channel", ""),
        title=best_entry.get("title", ""),
    )


def _yt_dlp_search(query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
    """Run yt-dlp in search mode and return up to ``top_k`` entries.

    Uses ytsearch{N}: prefix — no YouTube Data API key required.
    Metadata only; does not download any video. This is the single
    boundary where network I/O happens, so tests mock this function.
    """
    import yt_dlp  # local import so tests can patch _yt_dlp_search without
                   # forcing yt-dlp into the test environment.

    ydl_opts = {
        "default_search": f"ytsearch{top_k}",
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": True,  # metadata only, don't resolve each result
        "socket_timeout": 15,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(f"ytsearch{top_k}:{query}", download=False)

    entries = (result or {}).get("entries", []) or []
    # Normalize the URL field — extract_flat returns it as "url" but some
    # entries use "webpage_url" instead.
    for e in entries:
        if "url" not in e and "webpage_url" in e:
            e["url"] = e["webpage_url"]
    return entries


# ---------------------------------------------------------------------------
# Metadata probe for a single URL
# ---------------------------------------------------------------------------


def probe_youtube_metadata(url: str) -> tuple[str, str | None] | None:
    """Fetch (song_title, artist) for a YouTube URL without downloading audio.

    Used by the ingest stage to resolve the user's submitted URL into a
    searchable song identity before calling ``find_piano_cover``. The
    returned title is lowercased and noise-stripped via ``normalize_title``
    so it can feed straight into the search query.

    Field precedence:
      * title: ``track`` → ``title`` (noise-stripped)
      * artist: ``artist`` → ``creator`` → ``uploader``

    Returns ``None`` on any failure (network error, invalid URL,
    metadata dict has no title-shaped field). Like ``find_piano_cover``,
    this function follows the silent-failure contract.
    """
    try:
        info = _yt_dlp_extract_info(url)
    except Exception as exc:  # noqa: BLE001 — silent-failure boundary
        log.warning("cover_search: metadata probe failed for %r: %s", url, exc)
        return None

    if not info:
        return None

    # Title: prefer 'track' (structured music metadata) over 'title'
    # (human-readable video title, often has noise tags). Normalize
    # whichever we pick so downstream callers get a clean search key.
    raw_title = info.get("track") or info.get("title")
    if not raw_title:
        return None
    title = normalize_title(raw_title)
    if not title:
        return None

    # Artist: precedence artist > creator > uploader.
    artist = info.get("artist") or info.get("creator") or info.get("uploader")

    return title, artist


def _yt_dlp_extract_info(url: str) -> dict[str, Any] | None:
    """Fetch a single video's metadata via yt-dlp without downloading.

    The thin I/O wrapper that tests mock. Uses the same suppress-noise
    flags as ``_yt_dlp_search`` but targets a single URL instead of a
    search query.
    """
    import yt_dlp  # local import — keeps yt-dlp out of the test import path

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "socket_timeout": 15,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        return ydl.extract_info(url, download=False)
