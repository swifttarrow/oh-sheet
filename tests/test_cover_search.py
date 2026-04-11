"""TDD tests for the piano cover search fast path.

Module under test: ``backend.services.cover_search`` (doesn't exist yet —
these tests drive the design).

The feature takes a song title + optional artist and searches YouTube
for a clean piano cover of that song. If it finds a high-quality match,
the ingest stage swaps the user's original URL for the cover's URL so
Basic Pitch gets a monophonic piano recording to transcribe instead of
a full-band mix.

Tested in three layers:

1. ``normalize_title`` — strips "[Official Video]", "(Lyrics)", etc.
2. ``score_candidate`` — the scoring rules and threshold policy.
3. ``find_piano_cover`` — the orchestrator, with yt-dlp mocked.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Layer 1a: title normalization
# ---------------------------------------------------------------------------


class TestNormalizeTitle:
    """Strip noise tokens that obscure title comparison."""

    def test_plain_title_unchanged(self):
        from backend.services.cover_search import normalize_title
        assert normalize_title("Bohemian Rhapsody") == "bohemian rhapsody"

    def test_lowercase_output(self):
        from backend.services.cover_search import normalize_title
        # Normalization must lowercase so case-insensitive substring match works.
        assert normalize_title("Hotel California") == "hotel california"

    def test_strips_official_video_tag(self):
        from backend.services.cover_search import normalize_title
        assert normalize_title("Hotel California [Official Video]") == "hotel california"
        assert normalize_title("Hotel California (Official Music Video)") == "hotel california"
        assert normalize_title("Hotel California - Official Video") == "hotel california"

    def test_strips_lyrics_tag(self):
        from backend.services.cover_search import normalize_title
        assert normalize_title("Someone Like You (Lyrics)") == "someone like you"
        assert normalize_title("Someone Like You [Lyric Video]") == "someone like you"

    def test_strips_quality_tags(self):
        from backend.services.cover_search import normalize_title
        assert normalize_title("Imagine (HD)") == "imagine"
        assert normalize_title("Imagine [4K Remaster]") == "imagine"
        assert normalize_title("Imagine (Remastered 2020)") == "imagine"

    def test_strips_live_tag(self):
        from backend.services.cover_search import normalize_title
        assert normalize_title("Hey Jude (Live at Wembley 1988)") == "hey jude"

    def test_collapses_whitespace(self):
        from backend.services.cover_search import normalize_title
        # After stripping a tag, leftover whitespace should be normalized.
        assert normalize_title("Yesterday  [HD]   ") == "yesterday"

    def test_removes_featured_artists(self):
        from backend.services.cover_search import normalize_title
        # "feat." is noise for title matching (the artist is a separate field).
        assert normalize_title("Umbrella feat. Jay-Z") == "umbrella"
        assert normalize_title("Umbrella (feat. Jay-Z)") == "umbrella"
        assert normalize_title("Umbrella ft. Jay-Z") == "umbrella"

    def test_empty_or_none_safe(self):
        from backend.services.cover_search import normalize_title
        assert normalize_title("") == ""
        assert normalize_title(None) == ""


# ---------------------------------------------------------------------------
# Layer 1b: channel allowlist presence check
# ---------------------------------------------------------------------------


class TestChannelAllowlist:
    """The seeded list of known-good piano cover channels."""

    def test_allowlist_exists_and_nonempty(self):
        from backend.services.cover_search import COVER_CHANNEL_ALLOWLIST
        assert isinstance(COVER_CHANNEL_ALLOWLIST, (list, tuple, frozenset, set))
        assert len(COVER_CHANNEL_ALLOWLIST) >= 5  # Seeded with at least a handful

    def test_allowlist_entries_are_lowercase(self):
        # Match is case-insensitive so the list stores lowercased names.
        from backend.services.cover_search import COVER_CHANNEL_ALLOWLIST
        for channel in COVER_CHANNEL_ALLOWLIST:
            assert channel == channel.lower(), f"{channel!r} should be lowercase"


# ---------------------------------------------------------------------------
# Layer 2: scoring rules
# ---------------------------------------------------------------------------


def _entry(
    title: str,
    channel: str = "Random Channel",
    duration: int = 240,
    view_count: int = 10_000,
) -> dict:
    """Build a yt-dlp search result entry for testing."""
    return {
        "id": "abc123",
        "title": title,
        "channel": channel,
        "duration": duration,
        "view_count": view_count,
        "url": "https://www.youtube.com/watch?v=abc123",
    }


class TestScoreCandidate:
    """Scoring rules drive the match quality gate. See module docstring
    in cover_search.py for the canonical rule list.

    Reference weights (from design):
      +50 allowlist channel
      +30 "piano cover" / "piano arrangement" / "solo piano" in title
      +20 wanted title is substring of found title (after normalize)
      +10 wanted artist appears in title or channel
      -20 "karaoke" / "tutorial" / "how to play" / "lesson" in title
    """

    def test_allowlist_channel_adds_50(self):
        from backend.services.cover_search import score_candidate
        # "rousseau" is in the allowlist — channel bonus only, nothing else matches.
        entry = _entry(title="My Song", channel="Rousseau")
        assert score_candidate(entry, wanted_title="unrelated song", wanted_artist=None) == 50

    def test_piano_keyword_adds_30(self):
        from backend.services.cover_search import score_candidate
        entry = _entry(title="Some Song Piano Cover", channel="No One")
        assert score_candidate(entry, wanted_title="unrelated", wanted_artist=None) == 30

    def test_piano_arrangement_also_counts(self):
        from backend.services.cover_search import score_candidate
        entry = _entry(title="My Song Piano Arrangement")
        assert score_candidate(entry, wanted_title="unrelated", wanted_artist=None) == 30

    def test_solo_piano_also_counts(self):
        from backend.services.cover_search import score_candidate
        entry = _entry(title="My Song - Solo Piano")
        assert score_candidate(entry, wanted_title="unrelated", wanted_artist=None) == 30

    def test_title_substring_adds_20(self):
        from backend.services.cover_search import score_candidate
        entry = _entry(title="Hotel California Random Thing")
        assert score_candidate(entry, wanted_title="Hotel California", wanted_artist=None) == 20

    def test_title_substring_is_case_insensitive(self):
        from backend.services.cover_search import score_candidate
        entry = _entry(title="HOTEL CALIFORNIA - something")
        assert score_candidate(entry, wanted_title="Hotel California", wanted_artist=None) == 20

    def test_artist_in_title_adds_10(self):
        from backend.services.cover_search import score_candidate
        entry = _entry(title="A song by Queen", channel="No One")
        assert score_candidate(entry, wanted_title="unrelated", wanted_artist="Queen") == 10

    def test_artist_in_channel_adds_10(self):
        from backend.services.cover_search import score_candidate
        entry = _entry(title="Unrelated", channel="Adele Official")
        assert score_candidate(entry, wanted_title="unrelated-nope", wanted_artist="Adele") == 10

    def test_karaoke_subtracts_20(self):
        from backend.services.cover_search import score_candidate
        entry = _entry(title="Bohemian Rhapsody Karaoke Version")
        # title match (+20) + karaoke penalty (-20) = 0
        assert score_candidate(entry, wanted_title="Bohemian Rhapsody", wanted_artist=None) == 0

    def test_tutorial_subtracts_20(self):
        from backend.services.cover_search import score_candidate
        entry = _entry(title="How to Play Hotel California on Piano Tutorial")
        # Has "piano" keyword (+30?) — actually no, "how to play" is a penalty.
        # Let's check: "tutorial" in title → -20
        # "Hotel California" substring → +20
        # No "piano cover" (it's "piano tutorial" which is a tutorial, not a cover)
        # Expected: 20 - 20 = 0. This test forces us to think about the token
        # rules carefully: "piano cover" matches, but "piano tutorial" doesn't.
        result = score_candidate(
            entry, wanted_title="Hotel California", wanted_artist=None
        )
        assert result == 0

    def test_perfect_match_max_score(self):
        from backend.services.cover_search import score_candidate
        entry = _entry(
            title="Bohemian Rhapsody - Queen (Piano Cover)",
            channel="Rousseau",
        )
        # allowlist +50, piano cover +30, title match +20, artist match +10 = 110
        assert score_candidate(
            entry, wanted_title="Bohemian Rhapsody", wanted_artist="Queen"
        ) == 110

    def test_realistic_good_match_clears_70_threshold(self):
        from backend.services.cover_search import score_candidate
        # Random piano cover channel, clear piano cover title, title matches.
        # 30 (piano cover) + 20 (title) + 10 (artist) = 60. Below 70.
        # Real policy: needs either allowlist OR two strong signals AND
        # title match. This test documents the threshold behavior.
        entry = _entry(
            title="Bohemian Rhapsody - Queen Piano Cover",
            channel="Some Random Channel",
        )
        assert score_candidate(
            entry, wanted_title="Bohemian Rhapsody", wanted_artist="Queen"
        ) == 60

    def test_rousseau_with_title_only_clears_70(self):
        from backend.services.cover_search import score_candidate
        # Rousseau channel (+50) + title substring (+20) = 70, exactly at threshold.
        entry = _entry(
            title="Bohemian Rhapsody",
            channel="Rousseau",
        )
        assert score_candidate(
            entry, wanted_title="Bohemian Rhapsody", wanted_artist=None
        ) == 70


# ---------------------------------------------------------------------------
# Layer 3: find_piano_cover orchestrator
# ---------------------------------------------------------------------------


class TestFindPianoCover:
    """Mocks yt-dlp's search to exercise the candidate selection logic."""

    def test_returns_best_candidate_above_threshold(self):
        from backend.services.cover_search import find_piano_cover

        search_results = [
            _entry(title="Bohemian Rhapsody Tutorial", channel="Other"),  # 0
            _entry(title="Bohemian Rhapsody Piano Cover", channel="Rousseau"),  # 100
            _entry(title="Bohemian Rhapsody Karaoke", channel="Other"),  # 0
        ]

        with patch("backend.services.cover_search._yt_dlp_search") as mock_search:
            mock_search.return_value = search_results
            result = find_piano_cover(
                title="Bohemian Rhapsody", artist="Queen", min_score=70
            )

        assert result is not None
        assert result.url == search_results[1]["url"]
        assert result.score == 100
        assert result.channel == "Rousseau"

    def test_returns_none_when_all_candidates_below_threshold(self):
        from backend.services.cover_search import find_piano_cover

        search_results = [
            _entry(title="Some Other Thing"),   # 0
            _entry(title="Karaoke Version"),    # -20
        ]

        with patch("backend.services.cover_search._yt_dlp_search") as mock_search:
            mock_search.return_value = search_results
            result = find_piano_cover(
                title="Bohemian Rhapsody", artist="Queen", min_score=70
            )

        assert result is None

    def test_empty_search_results_returns_none(self):
        from backend.services.cover_search import find_piano_cover

        with patch("backend.services.cover_search._yt_dlp_search") as mock_search:
            mock_search.return_value = []
            result = find_piano_cover(title="Obscure Song", artist=None)

        assert result is None

    def test_yt_dlp_failure_returns_none(self):
        from backend.services.cover_search import find_piano_cover

        with patch("backend.services.cover_search._yt_dlp_search") as mock_search:
            mock_search.side_effect = RuntimeError("network down")
            result = find_piano_cover(title="Hotel California", artist="Eagles")

        assert result is None  # Silent failure — caller falls back to direct transcription

    def test_query_includes_artist_when_provided(self):
        from backend.services.cover_search import find_piano_cover

        with patch("backend.services.cover_search._yt_dlp_search") as mock_search:
            mock_search.return_value = []
            find_piano_cover(title="Yesterday", artist="The Beatles")

        # Should have called with a query containing both title and artist.
        called_query = mock_search.call_args.args[0]
        assert "yesterday" in called_query.lower()
        assert "the beatles" in called_query.lower()
        assert "piano cover" in called_query.lower()

    def test_query_works_without_artist(self):
        from backend.services.cover_search import find_piano_cover

        with patch("backend.services.cover_search._yt_dlp_search") as mock_search:
            mock_search.return_value = []
            find_piano_cover(title="River Flows In You", artist=None)

        called_query = mock_search.call_args.args[0]
        assert "river flows in you" in called_query.lower()
        assert "piano cover" in called_query.lower()


# ---------------------------------------------------------------------------
# Layer 4: metadata probe for the original URL
# ---------------------------------------------------------------------------
#
# Before we can search for a cover, we need to know the song title + artist
# of whatever the user pasted. yt-dlp can fetch video metadata (title,
# uploader, track, artist) without downloading any audio. probe_youtube_metadata
# wraps that call and normalizes the output into a (title, artist) tuple that
# feeds directly into find_piano_cover().


class TestProbeYoutubeMetadata:
    """Resolves a YouTube URL to (song_title, artist) using yt-dlp's
    extract_info(download=False). The title should come from the
    'track' or 'title' field; the artist from 'artist', 'creator', or
    'uploader' depending on which the video has."""

    def test_extracts_music_metadata_when_present(self):
        from backend.services.cover_search import probe_youtube_metadata

        # Official music videos often have structured 'track' and 'artist'
        # fields that YouTube Music populates.
        with patch("backend.services.cover_search._yt_dlp_extract_info") as mock_ei:
            mock_ei.return_value = {
                "title": "Bohemian Rhapsody (Official Video)",
                "track": "Bohemian Rhapsody",
                "artist": "Queen",
                "uploader": "Queen Official",
            }
            title, artist = probe_youtube_metadata("https://youtu.be/fJ9rUzIMcZQ")

        # Title is lowercased by normalize_title so it can feed directly
        # into find_piano_cover's substring matching.
        assert title == "bohemian rhapsody"
        assert artist == "Queen"

    def test_falls_back_to_title_when_track_missing(self):
        from backend.services.cover_search import probe_youtube_metadata

        # User uploads without music metadata — fall back to the 'title'
        # field (which normalize_title will clean up downstream).
        with patch("backend.services.cover_search._yt_dlp_extract_info") as mock_ei:
            mock_ei.return_value = {
                "title": "Hotel California [Official Video]",
                "uploader": "Eagles VEVO",
            }
            title, artist = probe_youtube_metadata("https://youtu.be/BciS5krYL80")

        # normalize_title strips the "[Official Video]" noise tag
        assert title == "hotel california"
        # artist falls back to uploader
        assert artist == "Eagles VEVO"

    def test_returns_none_on_yt_dlp_failure(self):
        from backend.services.cover_search import probe_youtube_metadata

        # Network failure, invalid URL, geo-blocked — all silent.
        with patch("backend.services.cover_search._yt_dlp_extract_info") as mock_ei:
            mock_ei.side_effect = RuntimeError("network down")
            result = probe_youtube_metadata("https://youtu.be/bad")

        assert result is None

    def test_returns_none_when_metadata_has_no_title(self):
        from backend.services.cover_search import probe_youtube_metadata

        # Without any title-shaped field, we can't form a search query.
        with patch("backend.services.cover_search._yt_dlp_extract_info") as mock_ei:
            mock_ei.return_value = {"uploader": "Random Channel"}
            result = probe_youtube_metadata("https://youtu.be/weird")

        assert result is None

    def test_prefers_artist_field_over_creator_and_uploader(self):
        from backend.services.cover_search import probe_youtube_metadata

        # Precedence: artist > creator > uploader. Make sure the function
        # picks the most-specific one available.
        with patch("backend.services.cover_search._yt_dlp_extract_info") as mock_ei:
            mock_ei.return_value = {
                "track": "Clocks",
                "artist": "Coldplay",
                "creator": "Coldplay VEVO",
                "uploader": "Coldplay - Topic",
            }
            _, artist = probe_youtube_metadata("https://youtu.be/clocks")

        assert artist == "Coldplay"


# ---------------------------------------------------------------------------
# Layer 5: contract field for per-job opt-in
# ---------------------------------------------------------------------------
#
# The user toggles "try to find a clean piano cover" on the upload screen.
# That choice travels through the pipeline as a field on InputMetadata so
# the ingest stage can read it and decide whether to run cover_search.


class TestPreferCleanSourceField:
    """InputMetadata must carry a ``prefer_clean_source`` flag so the
    user's per-request choice survives the trip from POST /v1/jobs
    through Celery dispatch into the ingest worker."""

    def test_prefer_clean_source_defaults_to_false(self):
        from shared.contracts import InputMetadata
        meta = InputMetadata(source="title_lookup")
        assert meta.prefer_clean_source is False

    def test_prefer_clean_source_accepts_true(self):
        from shared.contracts import InputMetadata
        meta = InputMetadata(
            source="title_lookup",
            title="https://youtu.be/abc",
            prefer_clean_source=True,
        )
        assert meta.prefer_clean_source is True

    def test_prefer_clean_source_round_trips_json(self):
        # The contract is serialized via model_dump()/model_validate() between
        # Celery tasks, so the field must survive JSON encoding.
        from shared.contracts import InputMetadata
        original = InputMetadata(source="title_lookup", prefer_clean_source=True)
        roundtripped = InputMetadata.model_validate(original.model_dump(mode="json"))
        assert roundtripped.prefer_clean_source is True
