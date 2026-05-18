"""Pure-logic tests for YouTube URL → video-ID extraction.

The video-ID is the canonical cache key for the YouTube job cache: any
two URLs that resolve to the same video MUST extract to the same ID
(so re-submitting the same song hits cache), and any non-URL or
non-video submission MUST return None (so search queries don't get
falsely cached against the wrong video).

These tests cover the URL shapes the wild internet actually delivers
(desktop, mobile, short, embed, with timestamps, with playlist refs).
They are deliberately stateless — no fixtures, no Redis, no FastAPI.
"""
from __future__ import annotations

import pytest

from backend.jobs.youtube_url import extract_video_id


class TestExtractVideoIdHappyPaths:
    """All of these must extract to the same canonical video ID."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "http://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10s",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLabc",
            "https://www.youtube.com/watch?t=10s&v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ?t=10",
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "youtube.com/watch?v=dQw4w9WgXcQ",  # no scheme
            "  https://www.youtube.com/watch?v=dQw4w9WgXcQ  ",  # whitespace
        ],
    )
    def test_extracts_canonical_video_id(self, url):
        assert extract_video_id(url) == "dQw4w9WgXcQ"


class TestExtractVideoIdReturnsNone:
    """Non-video and non-URL inputs MUST return None to keep the cache
    key honest. False positives here would cache distinct songs against
    the same key — much worse than a cache miss.
    """

    @pytest.mark.parametrize(
        "url",
        [
            "",
            "Hey Jude",  # plain song title
            "Beatles - Hey Jude",
            "https://www.google.com/",
            "https://www.youtube.com/",  # bare home
            "https://www.youtube.com/results?search_query=hey+jude",  # search page
            "https://www.youtube.com/c/SomeChannel",  # channel
            "https://www.youtube.com/playlist?list=PLabc",  # playlist root
            "https://example.com/watch?v=dQw4w9WgXcQ",  # not YouTube
            None,  # defensive — caller may pass None title
        ],
    )
    def test_non_video_inputs_return_none(self, url):
        assert extract_video_id(url) is None
