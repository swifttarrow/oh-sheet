"""Clean-source search — the ingest stage's "fast path" data source router.

When the user submits a YouTube URL with ``prefer_clean_source=True``,
this module searches YouTube for a clean alternative version of the same
song (piano cover, 8-bit/chiptune cover, …) and returns the best
candidate across all variants. The ingest stage then swaps the user's
original URL for the match's URL so Basic Pitch receives something it
can actually transcribe well.

Why this helps transcription quality:
    Basic Pitch is a polyphonic pitch tracker, but it transcribes every
    audible pitch — including drum fundamentals, vocal harmonics, and
    bass subharmonics — as piano notes. On a pop-song mix this yields a
    dense, unplayable result. A **piano cover** is already monophonic
    (or polyphonic but piano-only). An **8-bit cover** is even cleaner
    because chiptune channels are pure square/triangle waves with zero
    reverb and no drums mixed into the pitched content. Either path is
    dramatically easier to transcribe than a full-band mix.

Multi-variant architecture:
    ``find_clean_source`` runs the search once per ``_SourceVariant``
    (piano + chiptune by default) and returns the single highest-scoring
    result across ALL variants. Each variant has its own channel
    allowlist and positive-keyword list so the scorer rewards the right
    kind of signal for the right kind of source.

Scoring policy (see ``score_candidate_for_variant``):
    +50  channel is in the variant's channel allowlist
    +30  title contains one of the variant's positive keywords
    +20  wanted song title is a substring of the found video title
    +10  artist name appears in the found title or the channel name
    -20  title contains "karaoke" / "tutorial" / "how to play" / "lesson"

The +50 and +30 weights come from the *variant's* lists, so a Rousseau
result scores 50 against the piano variant and 0 against the chiptune
variant — exactly the isolation we want.

Default threshold: score >= 60 triggers a URL swap. Dry-run testing
against real YouTube (see scripts/dryrun_cover_search.py) showed the
original 70 was too strict — legitimate pop covers from non-allowlist
channels cap at exactly 60 (+30 piano cover + +20 title + +10 artist),
so 70 silently rejected them. At 60, junk content (karaoke/tutorial)
still can't clear because the -20 penalty brings them to 40 or below.
The threshold is passed as a parameter so callers (or config) can
tune strictness without editing this module.

Silent failure contract:
    ``find_clean_source`` / ``find_piano_cover`` return ``None`` on any
    failure (network error, yt-dlp exception, no candidates, all below
    threshold). If ONE variant's search crashes but another succeeds,
    the successful variant's best result is still returned. The caller
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
# Channel allowlist — tiered by playability
# ---------------------------------------------------------------------------
#
# Piano covers on YouTube span a huge difficulty range. Rousseau's Bohemian
# Rhapsody has 2000+ notes across full 10-finger chords — transcribes to
# MIDI beautifully but the resulting sheet music is unreadable for anyone
# short of a concert pianist. Pianote's beginner arrangements of the same
# song have ~400 notes and fit comfortably on two staves.
#
# For the MVP we target **easy + moderate** only. Dropping the advanced
# tier is a scope decision, not a quality issue: virtuoso covers still
# exist and work fine for MIDI playback, but the sheet music output is
# not a fit for the target user (casual / beginner / intermediate player
# learning a song). When we add a difficulty selector to the upload
# screen that meaningfully affects cover selection, the advanced tier
# can be reactivated by adding PIANO_ADVANCED_CHANNELS to the active
# allowlist below.
#
# Entries are lowercase and matched case-insensitively as substrings of
# the candidate's channel field, so "Rousseau" matches "Rousseau - Official".

# Tier 1 — easy / beginner arrangements. Explicitly branded as easy or
# known for simplified versions. Gets a soft preference bonus in scoring
# so when an easy-tier match and a moderate-tier match both exist, easy
# wins the tie.
PIANO_EASY_CHANNELS: tuple[str, ...] = (
    "pianote",
    "peter plutax",
    "everynote",
    "phianonize",           # frequent "easy version" releases
    "onepianoheart",
    "easy piano tutorials",
    "simple piano",
    "atlantic notes",
    "tutorialsbyhugo",
    "bitesize piano",
    "rainbow piano tuto",
)

# Tier 2 — moderate / intermediate arrangements. Playable for a
# developing player without being watered down. Many of these channels
# occasionally post advanced tracks too; scoring uses the channel tier,
# not the individual video's complexity.
PIANO_MODERATE_CHANNELS: tuple[str, ...] = (
    "jacob's piano",
    "jacobs piano",         # ASCII fallback
    "akmigone",
    "peter buka",
    "pianella piano",
    "dotted8th",
    "francesco parrino",
    "martin walsh",
    "adam chen",
    "aaronastro",
    # Kesh Piano Music (@keshpianomusic) — see TRUSTED_TUTORIAL_CHANNELS
    # below. yt-dlp returns ``channel='Kesh'`` which is too broad for
    # substring matching (risks false positives against the pop artist
    # Kesha), so we match on the unambiguous uploader_id handle.
    "keshpianomusic",
    # Added from the 8-song audit (2026-04-11): channels that produced
    # good matches but were missing from the allowlist, scoring only
    # on title/artist/keyword without the +50 bonus.
    "yifanmusic",
    # "sheet music boss",  # removed: duet arrangements break music21 engrave
    "jova musique",         # also publishes as "Pianella Piano"
    "kassia",
    "piano by number",
    "learn piano live",
    "the theorist",
    "littletranscriber",
    "riyandi kusuma",
    "naor yadid",
    "katherine cordova",
    "costantino carrara",
    "we artplay",
    "ya boi carter",
    "akiva broder",
    "cleminova",
    "pollonuel",
    "matty on the keys",
    "flying fingers",
    "piano covers - topic",
)

# Tier 3 — virtuoso / concert-level arrangements. Defined here for
# documentation and future re-enablement, but NOT part of the active
# allowlist. These channels produce dense 10-finger arrangements that
# are great to listen to but not readable as sheet music for the target
# audience. To reactivate, include this tuple in COVER_CHANNEL_ALLOWLIST
# below and add a difficulty selector to the UI.
PIANO_ADVANCED_CHANNELS: tuple[str, ...] = (
    "rousseau",
    "patrik pietschmann",
    "kyle landry",
    "lord vinheteiro",
    "david solís",
    "david solis",          # ASCII fallback
    "the piano guys",
)

# Active piano allowlist = easy + moderate + advanced. All three tiers
# receive the +50 channel bonus. Easy channels get an additional +10
# bias via _EASY_TIER_BONUS below. Advanced channels (Rousseau, Kyle
# Landry, etc.) produce complex transcriptions that may be hard for
# beginners, but a complex cover transcribed accurately is dramatically
# better than no cover at all — the Spread Thin A/B test (2026-04-11)
# proved this conclusively.
COVER_CHANNEL_ALLOWLIST: tuple[str, ...] = (
    PIANO_EASY_CHANNELS + PIANO_MODERATE_CHANNELS + PIANO_ADVANCED_CHANNELS
)


# ---------------------------------------------------------------------------
# Trusted tutorial channels — exempt from the "tutorial" keyword penalty
# ---------------------------------------------------------------------------
#
# Some channels explicitly label their videos as "Piano Tutorial" but
# produce clean transcription-quality audio — Synthesia-style rendered
# piano with no voiceover or backing track. For those channels the
# ``-20`` tutorial penalty in _BAD_KEYWORDS is the wrong signal: it
# marks them down for what they CALL themselves, not what they actually
# sound like. Basic Pitch can transcribe their audio fine.
#
# Channels listed here bypass the tutorial penalty. Matched against
# yt-dlp's ``uploader_id`` (the ``@handle``, with the ``@`` stripped)
# so the identifier is unambiguous even for channels whose display
# name is a common word. "keshpianomusic" matches ``@keshpianomusic``
# cleanly without colliding with the pop artist Kesha.
#
# Add new entries here as dry-runs identify other trustworthy
# tutorial-labelled channels. Keep it narrow — the penalty exists for
# a reason and most tutorials ARE noisy (voiceover, slowdowns, etc.).

TRUSTED_TUTORIAL_CHANNELS: tuple[str, ...] = (
    "keshpianomusic",
)


# 8-bit / chiptune covers. These channels publish pure square+triangle
# arrangements of popular songs (and game themes). Chiptune audio is the
# easiest possible input for Basic Pitch — monophonic channels, zero
# reverb, no drums mixed into pitched content — so when one of these
# channels has a cover of the user's song, it's usually a better source
# than any piano cover. The list is intentionally shorter than the piano
# allowlist because the chiptune ecosystem is narrower; tune as needed.
#
# Must be disjoint from COVER_CHANNEL_ALLOWLIST so a matching channel
# only scores for ONE variant — enforced by a test.

CHIPTUNE_CHANNEL_ALLOWLIST: tuple[str, ...] = (
    "8-bit universe",
    "8 bit universe",         # common spelling variant w/o hyphen
    "button masher",
    "press start",
    "noize 8-bit",
    "noize 8 bit",
    "vgmpire",
    "arcade player",
    "pixelord",
    "8-bit arcade",
    "8 bit arcade",
    "chipzel",
    "inverse phase",
)


# ---------------------------------------------------------------------------
# Title normalization
# ---------------------------------------------------------------------------


# Noise patterns to strip from titles before comparing. Order matters:
# more-specific patterns first so they don't get accidentally matched by
# a generic one.
_NOISE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Catch-all for "[Official ...]" and "(Official ...)" — covers video,
    # visualizer, audio, lyric video, and any future YouTube label variants.
    re.compile(r"\s*[\[\(]\s*official\s+[^\]\)]*[\]\)]", re.IGNORECASE),
    re.compile(r"\s*-\s*official\s+\w+(\s+\w+)?", re.IGNORECASE),
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
    "piano tutorial",
    "piano version",
)

# Positive-signal keywords for chiptune/8-bit covers. "8 bit" is the most
# common phrasing on YouTube; "8-bit" gets hit by the same substring check
# because we match on normalized lowercase text.
_CHIPTUNE_KEYWORDS: tuple[str, ...] = (
    "8 bit",
    "8-bit",
    "chiptune",
    "nes version",
    "famicom",
    "8bit",
)

# Negative-signal keywords. Karaoke tracks have backing vocals and no
# piano — always bad. Tutorial/lesson penalty removed (2026-04-12):
# too many clean Synthesia-style piano tutorials were being rejected
# (Omar Blyde, Piano and Sing, Top Piano Tutorials, etc). The positive
# keyword "piano tutorial" now gives +30 instead.
_BAD_KEYWORDS: tuple[str, ...] = (
    "karaoke",
)

# Channels whose arrangements consistently break music21's MusicXML
# export (duet layouts, complex subdivisions). Any candidate from these
# channels is filtered out entirely before scoring.
_BLOCKED_CHANNELS: tuple[str, ...] = (
    "sheet music boss",
)

# Additional bonus for piano channels in the "easy" tier on top of the
# normal +50 allowlist bonus. When both an easy-tier and a moderate-tier
# candidate score equally on title/artist, the easy-tier one wins by +10.
# Not enough to let a weak easy candidate beat a strong moderate one;
# just enough to break ties in the easy tier's favor.
_EASY_TIER_BONUS: int = 10


# ---------------------------------------------------------------------------
# Source variants
# ---------------------------------------------------------------------------
#
# A _SourceVariant describes one "kind of clean source" we search for.
# Adding a new variant (e.g. "orchestral cover", "acoustic guitar cover")
# is a data change: append a new _SourceVariant to DEFAULT_VARIANTS and
# write its allowlist + keywords. No scoring logic change required.


@dataclass(frozen=True)
class _SourceVariant:
    """One search strategy: a query suffix, a channel allowlist, and
    a positive-keyword list. Scoring is variant-scoped, so a Rousseau
    entry gets +50 against the piano variant and +0 against the chiptune
    variant."""

    name: str
    query_suffix: str
    channel_allowlist: tuple[str, ...]
    keywords: tuple[str, ...]


PIANO_VARIANT = _SourceVariant(
    name="piano",
    query_suffix="piano cover",
    channel_allowlist=COVER_CHANNEL_ALLOWLIST,
    keywords=_PIANO_COVER_KEYWORDS,
)

CHIPTUNE_VARIANT = _SourceVariant(
    name="chiptune",
    query_suffix="8 bit cover",
    channel_allowlist=CHIPTUNE_CHANNEL_ALLOWLIST,
    keywords=_CHIPTUNE_KEYWORDS,
)

# Default variant set used by ``find_clean_source``. Order matters only
# for logging — the scorer always picks the globally highest across all
# variants, regardless of list position.
# Chiptune variant is paused: the downstream transcription pipeline is
# trained for piano only. Chiptune results that win the cross-variant
# comparison become dead ends — they fall back to the basic pipeline
# with no enhanced output. Worse, chiptune can beat piano on score
# (e.g. Flashing Lights: chiptune 90 > piano 80), stealing a song
# that WOULD route to the enhanced pipeline. Reactivate post-demo if
# a chiptune-capable transcription path is added.
DEFAULT_VARIANTS: tuple[_SourceVariant, ...] = (PIANO_VARIANT,)


def score_candidate_for_variant(
    entry: dict[str, Any],
    *,
    wanted_title: str,
    wanted_artist: str | None,
    variant: _SourceVariant,
) -> int:
    """Score a yt-dlp entry against a SPECIFIC source variant.

    This is the scorer the multi-variant orchestrator uses. See the
    module docstring for the canonical rule list. The +50 allowlist
    bonus and +30 keyword bonus come from the variant; everything else
    (title match, artist match, bad-keyword penalty) is shared.
    """
    title_norm = normalize_title(entry.get("title", ""))
    channel_norm = (entry.get("channel") or "").lower()
    # uploader_id is yt-dlp's "@handle" form (e.g. "@keshpianomusic").
    # Strip the leading "@" so allowlist entries can be plain strings.
    uploader_id_norm = (entry.get("uploader_id") or "").lower().lstrip("@")
    wanted_title_norm = normalize_title(wanted_title)
    wanted_artist_norm = (wanted_artist or "").lower().strip()

    # Blocked channels are filtered out entirely — return -1 so they
    # never pass the min_score threshold.
    if any(b in channel_norm or b in uploader_id_norm for b in _BLOCKED_CHANNELS):
        return -1

    # Skip videos shorter than 60 seconds — these are snippets/shorts,
    # not full covers. Produces thin, incomplete transcriptions.
    duration = entry.get("duration") or 0
    if duration and duration < 60:
        return -1

    # Helper: does any allowlist entry match either the channel name
    # OR the uploader_id handle? Checking both fields lets us pin
    # specific channels via their unambiguous handle when their
    # display name is too broad (e.g. "Kesh" would collide with the
    # pop artist "Kesha").
    def _matches_any(allowlist: tuple[str, ...]) -> bool:
        return any(
            entry_key in channel_norm or entry_key in uploader_id_norm
            for entry_key in allowlist
        )

    score = 0

    # +50 if the channel is in this variant's allowlist (substring match
    # against channel OR uploader_id so "Rousseau - Official" still
    # matches "rousseau" and "@keshpianomusic" still matches
    # "keshpianomusic").
    if _matches_any(variant.channel_allowlist):
        score += 50

    # Piano-only extra: +10 if the channel is in the "easy" tier of the
    # piano allowlist. Soft preference so easy arrangements win ties
    # against moderate arrangements. Chiptune variant doesn't have
    # difficulty sub-tiers so this check is a no-op there.
    if variant.name == "piano" and _matches_any(PIANO_EASY_CHANNELS):
        score += _EASY_TIER_BONUS

    # +30 if any of this variant's positive keywords appear in the title.
    if any(kw in title_norm for kw in variant.keywords):
        score += 30

    # +20 if the wanted song title appears in the found title. Shared
    # across all variants.
    if wanted_title_norm and wanted_title_norm in title_norm:
        score += 20

    # +10 if the artist name appears in the title or channel. Shared
    # across all variants.
    if wanted_artist_norm and (
        wanted_artist_norm in title_norm or wanted_artist_norm in channel_norm
    ):
        score += 10

    # -20 for any bad keyword — karaoke, tutorials, and lessons are not
    # cover recordings and will confuse the transcriber worse than the
    # original mix.
    #
    # EXEMPTION: any channel in the variant's allowlist OR in the
    # TRUSTED_TUTORIAL_CHANNELS list bypasses the penalty. The reasoning:
    # if a channel is good enough to be on the allowlist, their "tutorial"
    # videos produce clean transcription-quality audio (Synthesia-style
    # rendered piano, no voiceover). The penalty exists for UNKNOWN
    # tutorial uploaders — channels we've never vetted and whose
    # "tutorial" videos might have talking, slowdowns, metronome clicks.
    is_allowlisted = _matches_any(variant.channel_allowlist)
    is_trusted_tutorial = _matches_any(TRUSTED_TUTORIAL_CHANNELS)
    is_exempt = is_allowlisted or is_trusted_tutorial
    if not is_exempt and any(bad in title_norm for bad in _BAD_KEYWORDS):
        score -= 20

    return score


def score_candidate(
    entry: dict[str, Any],
    wanted_title: str,
    wanted_artist: str | None,
) -> int:
    """Score a yt-dlp search result entry for the piano variant.

    Thin wrapper around ``score_candidate_for_variant`` with the piano
    variant baked in. Kept for backward compatibility with older tests
    and callers that still use the piano-specific rule list.
    """
    return score_candidate_for_variant(
        entry,
        wanted_title=wanted_title,
        wanted_artist=wanted_artist,
        variant=PIANO_VARIANT,
    )


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


def _search_one_variant(
    title: str,
    artist: str | None,
    *,
    variant: _SourceVariant,
    top_k: int,
) -> tuple[int, dict[str, Any]] | None:
    """Run the search for ONE variant and return its best (score, entry).

    Returns ``None`` if the search failed, returned nothing, or crashed.
    Does NOT apply the min_score threshold — the orchestrator compares
    raw best scores across variants before applying the threshold.
    """
    if artist:
        query = f"{title} {artist} {variant.query_suffix}"
    else:
        query = f"{title} {variant.query_suffix}"

    try:
        entries = _yt_dlp_search(query, top_k=top_k)
    except Exception as exc:  # noqa: BLE001 — silent-failure boundary
        log.warning(
            "cover_search[%s]: yt-dlp search failed for %r: %s",
            variant.name, query, exc,
        )
        return None

    if not entries:
        log.info("cover_search[%s]: no results for %r", variant.name, query)
        return None

    scored = [
        (
            score_candidate_for_variant(
                e, wanted_title=title, wanted_artist=artist, variant=variant,
            ),
            e,
        )
        for e in entries
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[0]


def find_clean_source(
    title: str,
    artist: str | None,
    *,
    min_score: int = 60,
    top_k: int = 10,
    variants: tuple[_SourceVariant, ...] = DEFAULT_VARIANTS,
) -> CoverSearchResult | None:
    """Search every variant for a clean alternative source of the song,
    return the highest-scoring match across all variants.

    Runs the query once per variant (so "piano cover" AND "8 bit cover"
    both get searched), scores each variant's best candidate against
    its OWN allowlist + keywords, and returns the single best result
    across all of them. Silent-failure contract: any exception in one
    variant is logged and the others still run; the function never
    raises.

    Returns ``None`` if no variant's best candidate clears ``min_score``,
    so the caller can fall back to direct transcription of the original
    URL. This is a hint, not a hard dependency.
    """
    best_overall: tuple[int, dict[str, Any], _SourceVariant] | None = None

    for variant in variants:
        result = _search_one_variant(
            title, artist, variant=variant, top_k=top_k,
        )
        if result is None:
            continue
        score, entry = result
        if best_overall is None or score > best_overall[0]:
            best_overall = (score, entry, variant)

    if best_overall is None:
        return None

    best_score, best_entry, best_variant = best_overall
    if best_score < min_score:
        log.info(
            "cover_search: best candidate across %d variant(s) for %r "
            "scored %d, below threshold %d",
            len(variants), title, best_score, min_score,
        )
        return None

    log.info(
        "cover_search: winning variant=%s score=%d channel=%r",
        best_variant.name, best_score, best_entry.get("channel", ""),
    )
    return CoverSearchResult(
        url=best_entry.get("url", ""),
        score=best_score,
        channel=best_entry.get("channel", ""),
        title=best_entry.get("title", ""),
    )


def find_piano_cover(
    title: str,
    artist: str | None,
    *,
    min_score: int = 60,
    top_k: int = 10,
) -> CoverSearchResult | None:
    """Piano-only search — thin backward-compat wrapper around
    ``find_clean_source``.

    Runs only the piano variant, not the chiptune variant. Kept so
    callers and tests that specifically want piano-only behavior don't
    have to construct a one-element ``variants`` tuple. New code should
    prefer ``find_clean_source`` to get the full multi-variant benefit.
    """
    return find_clean_source(
        title,
        artist,
        min_score=min_score,
        top_k=top_k,
        variants=(PIANO_VARIANT,),
    )


def _normalize_entry_url(entry: dict[str, Any]) -> str:
    """Resolve a yt-dlp entry into a canonical ``https://www.youtube.com/watch?v=...`` URL.

    yt-dlp's ``extract_flat=True`` mode is inconsistent about where the
    watch URL ends up. In practice we see three variants:

      1. ``url`` field holds a full ``http(s)://...`` URL (most common)
      2. ``url`` field holds the bare 11-char video ID (observed on
         recent yt-dlp builds — "flat" is literal there)
      3. ``url`` field is missing and ``webpage_url`` carries the URL

    Variant #2 is the dangerous one: a bare ID flows unchecked through
    CoverSearchResult → _maybe_swap_for_cover_sync → _download_youtube_sync,
    where ``urlparse`` returns an empty hostname and _download_youtube_sync
    raises ValueError — crashing the ingest job (PR #47 review, Critical).

    Preference order, highest to lowest:
      1. ``webpage_url`` if it's a valid ``http(s)://`` URL
      2. ``url`` if it's a valid ``http(s)://`` URL
      3. Constructed ``https://www.youtube.com/watch?v={id}`` from ``id``
      4. ``""`` — caller must drop this entry

    webpage_url wins over url because when yt-dlp populates it at all,
    it's always the full canonical form; url may be bare.
    """
    webpage_url = (entry.get("webpage_url") or "").strip()
    if webpage_url.startswith(("http://", "https://")):
        return webpage_url

    existing_url = (entry.get("url") or "").strip()
    if existing_url.startswith(("http://", "https://")):
        return existing_url

    video_id = (entry.get("id") or "").strip()
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"

    return ""


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

    raw_entries = (result or {}).get("entries", []) or []
    # Normalize each entry's URL through _normalize_entry_url, which
    # handles the extract_flat quirks (bare IDs, webpage_url fallback,
    # id-based reconstruction). Drop any entry we can't resolve at all;
    # passing a bad URL downstream crashes _download_youtube_sync.
    entries: list[dict[str, Any]] = []
    for e in raw_entries:
        normalized = _normalize_entry_url(e)
        if not normalized:
            continue
        e["url"] = normalized
        entries.append(e)
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
