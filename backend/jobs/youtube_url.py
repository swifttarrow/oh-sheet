"""YouTube URL normalization for the job cache.

The cache key is the video ID, not the raw URL — so we collapse the
many URL shapes YouTube hands out (desktop, mobile, short, embed, with
playlist params, with timestamps) into one canonical key. Non-URL
inputs (search queries, plain song titles, malformed URLs) return
``None`` to signal "not cacheable, dispatch normally."

Pure logic — no I/O, no Redis. Tested via ``tests/test_youtube_url.py``.
"""
from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

# Video IDs in the wild are 11 chars / [A-Za-z0-9_-], but we don't pin
# length here — YouTube has changed conventions before, and matching
# the character class is enough to keep cache keys honest.
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# Hosts where a path/query carries a YouTube video ID. Anything outside
# this list → return None even if the URL looks watch-like (a malicious
# or copy-pasted URL on a phishing domain should never hit our cache).
_YOUTUBE_HOSTS = frozenset(
    {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
)


def extract_video_id(url: str | None) -> str | None:
    """Return the canonical YouTube video ID for ``url``, or None.

    Returns None for: empty/None input, non-YouTube hosts, search/channel/
    playlist-root URLs (no specific video), and anything that fails to
    parse as a URL. NEVER raises — callers may pass arbitrary user input.
    """
    if not url or not isinstance(url, str):
        return None

    try:
        parsed = urlparse(url.strip())
    except (ValueError, AttributeError):
        return None

    # urlparse on "Hey Jude" returns scheme='' netloc=''. Filter that out
    # before trying anything else — saves us from accidentally treating a
    # song title as a relative URL.
    if not parsed.netloc and not parsed.scheme:
        # Allow scheme-less "youtube.com/watch?v=..." — urlparse puts
        # those in path, not netloc. Re-parse with a fake scheme.
        if "youtube.com/" in url or "youtu.be/" in url:
            try:
                parsed = urlparse("https://" + url.strip())
            except (ValueError, AttributeError):
                return None
        else:
            return None

    host = parsed.netloc.lower()
    if host not in _YOUTUBE_HOSTS:
        return None

    # youtu.be/<id> — ID is the path component.
    if host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/", 1)[0]
        return candidate if _VIDEO_ID_RE.match(candidate) else None

    # youtube.com/watch?v=<id> — ID is the ``v`` query param.
    if parsed.path == "/watch":
        v_values = parse_qs(parsed.query).get("v", [])
        if v_values and _VIDEO_ID_RE.match(v_values[0]):
            return v_values[0]
        return None

    # youtube.com/embed/<id> — ID is the second path segment.
    if parsed.path.startswith("/embed/"):
        candidate = parsed.path[len("/embed/") :].split("/", 1)[0]
        return candidate if _VIDEO_ID_RE.match(candidate) else None

    # Everything else (search pages, channels, playlist roots, the
    # bare home page) doesn't reference a single video — None.
    return None
