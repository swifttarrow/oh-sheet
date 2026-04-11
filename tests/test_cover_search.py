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

from unittest.mock import patch

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
    uploader_id: str | None = None,
) -> dict:
    """Build a yt-dlp search result entry for testing.

    ``uploader_id`` is the ``@handle`` string yt-dlp returns for the
    channel (e.g. ``@keshpianomusic``). Optional — most tests don't
    care about it, but the trusted-tutorial exemption path reads it
    for unambiguous channel identification.
    """
    entry = {
        "id": "abc123",
        "title": title,
        "channel": channel,
        "duration": duration,
        "view_count": view_count,
        "url": "https://www.youtube.com/watch?v=abc123",
    }
    if uploader_id is not None:
        entry["uploader_id"] = uploader_id
    return entry


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
        # "jacob's piano" is in the moderate tier of the piano allowlist.
        # Channel bonus only, nothing else matches. Moderate tier gets
        # exactly +50 (no easy-tier bonus), which is the "baseline" we
        # test here. Easy-tier channels get +60 — covered by
        # TestPianoTierPreference below.
        entry = _entry(title="My Song", channel="Jacob's Piano")
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
            channel="Jacob's Piano",
        )
        # Moderate-tier allowlist +50, piano cover +30, title match +20,
        # artist match +10 = 110. Easy-tier would add +10 more (see
        # TestPianoTierPreference::test_easy_tier_perfect_match_scores_120).
        assert score_candidate(
            entry, wanted_title="Bohemian Rhapsody", wanted_artist="Queen"
        ) == 110

    def test_realistic_good_match_scores_60(self):
        from backend.services.cover_search import score_candidate
        # Random piano cover channel, clear piano cover title, title matches.
        # 30 (piano cover) + 20 (title) + 10 (artist) = 60.
        # This exactly meets the default threshold of 60 — the dry-run
        # showed most legitimate pop covers land here, so we want them to
        # clear. Bumping the scorer must not drop this below 60.
        entry = _entry(
            title="Bohemian Rhapsody - Queen Piano Cover",
            channel="Some Random Channel",
        )
        assert score_candidate(
            entry, wanted_title="Bohemian Rhapsody", wanted_artist="Queen"
        ) == 60

    def test_moderate_tier_with_title_only_scores_70(self):
        from backend.services.cover_search import score_candidate
        # Moderate-tier channel (+50) + title substring (+20) = 70.
        # This is the minimum "clearly a hit" shape: allowlisted channel
        # posts a video that mentions the target song in its title but
        # doesn't say "piano cover" in the title. Exactly clears the
        # legacy 70 threshold and sits above the current 60 threshold.
        entry = _entry(
            title="Bohemian Rhapsody",
            channel="Jacob's Piano",
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
            _entry(title="Bohemian Rhapsody Piano Cover", channel="Jacob's Piano"),  # 100
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
        assert result.channel == "Jacob's Piano"

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

    def test_default_threshold_is_60_accepts_triple_signal_match(self):
        from backend.services.cover_search import find_piano_cover

        # A non-allowlist channel with the full "piano cover + title +
        # artist" triple scores exactly 60. The default threshold must
        # let this through — this is the common case for pop covers,
        # surfaced by the dry-run against real YouTube.
        search_results = [
            _entry(
                title="Someone Like You - Adele Piano Cover",
                channel="Random Pianist",
            ),
        ]

        with patch("backend.services.cover_search._yt_dlp_search") as mock_search:
            mock_search.return_value = search_results
            # No min_score override — must use the module default.
            result = find_piano_cover(title="Someone Like You", artist="Adele")

        assert result is not None, "score 60 must clear the default threshold"
        assert result.score == 60


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


# ---------------------------------------------------------------------------
# Layer 6: Settings — feature toggle + tunable threshold
# ---------------------------------------------------------------------------
#
# The cover search runs inside the ingest worker, so its behavior needs to
# be controllable at deploy time via environment variables. Two knobs:
#
#   OHSHEET_COVER_SEARCH_ENABLED       — kill switch for the whole feature
#   OHSHEET_COVER_SEARCH_MIN_SCORE     — per-env threshold override
#
# These let operators flip the feature off on a bad-release day without
# touching code, and let them crank strictness up on quality-sensitive
# environments while the default holds elsewhere.


class TestCoverSearchSettings:
    """Config must expose cover_search knobs so operators can tune the
    feature without editing code."""

    def test_cover_search_enabled_defaults_true(self):
        # Feature-flag default: on. The user's per-job prefer_clean_source
        # still gates whether the search runs per request, but operators
        # need a global kill switch for emergency disablement.
        from backend.config import Settings
        s = Settings()
        assert s.cover_search_enabled is True

    def test_cover_search_min_score_defaults_to_60(self):
        # Global default must match find_piano_cover's hardcoded default
        # so tests and runtime agree. When this drifts from cover_search.py,
        # one of them has been updated without the other and reviewers
        # should catch it.
        from backend.config import Settings
        s = Settings()
        assert s.cover_search_min_score == 60

    def test_cover_search_settings_overridable_via_env(self, monkeypatch):
        # pydantic-settings reads OHSHEET_* at Settings() construction,
        # so each Settings() call reflects the current env.
        from backend.config import Settings
        monkeypatch.setenv("OHSHEET_COVER_SEARCH_ENABLED", "false")
        monkeypatch.setenv("OHSHEET_COVER_SEARCH_MIN_SCORE", "80")
        s = Settings()
        assert s.cover_search_enabled is False
        assert s.cover_search_min_score == 80


# ---------------------------------------------------------------------------
# PR #47 review: yt-dlp entry URL normalization (Critical)
# ---------------------------------------------------------------------------
#
# yt-dlp's ``extract_flat=True`` mode is inconsistent about where the
# watch URL ends up — see _normalize_entry_url's docstring for details.
# The dangerous case is a bare 11-char video ID landing in the ``url``
# field: it flows unchecked through CoverSearchResult.url →
# _maybe_swap_for_cover_sync → _download_youtube_sync, where
# extract_youtube_id rejects it and _download_youtube_sync raises
# ValueError — crashing the ingest job.


class TestNormalizeEntryUrl:
    """Normalize yt-dlp entries into canonical watch URLs."""

    def test_full_https_url_in_url_field_returned_as_is(self):
        from backend.services.cover_search import _normalize_entry_url
        entry = {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                 "id": "dQw4w9WgXcQ"}
        assert _normalize_entry_url(entry) == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_full_http_url_accepted(self):
        from backend.services.cover_search import _normalize_entry_url
        entry = {"url": "http://youtube.com/watch?v=dQw4w9WgXcQ",
                 "id": "dQw4w9WgXcQ"}
        assert _normalize_entry_url(entry) == "http://youtube.com/watch?v=dQw4w9WgXcQ"

    def test_bare_video_id_in_url_field_reconstructed_from_id(self):
        # The (Critical) case: extract_flat sometimes puts just the
        # 11-char ID in the "url" field. Without normalization, this
        # flows into _download_youtube_sync and crashes the job.
        from backend.services.cover_search import _normalize_entry_url
        entry = {"url": "dQw4w9WgXcQ", "id": "dQw4w9WgXcQ"}
        assert _normalize_entry_url(entry) == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_webpage_url_wins_when_url_field_is_bare_id(self):
        # Prefer webpage_url — it's always the full canonical form when
        # yt-dlp populates it at all.
        from backend.services.cover_search import _normalize_entry_url
        entry = {
            "url": "dQw4w9WgXcQ",
            "webpage_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "id": "dQw4w9WgXcQ",
        }
        assert _normalize_entry_url(entry) == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_webpage_url_used_when_url_field_missing(self):
        from backend.services.cover_search import _normalize_entry_url
        entry = {
            "webpage_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "id": "dQw4w9WgXcQ",
        }
        assert _normalize_entry_url(entry) == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_empty_url_field_falls_back_to_id(self):
        from backend.services.cover_search import _normalize_entry_url
        entry = {"url": "", "id": "dQw4w9WgXcQ"}
        assert _normalize_entry_url(entry) == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_nothing_resolvable_returns_empty_string(self):
        # Caller must drop this entry.
        from backend.services.cover_search import _normalize_entry_url
        entry = {"title": "some video", "channel": "some channel"}
        assert _normalize_entry_url(entry) == ""

    def test_yt_dlp_search_drops_entries_with_unresolvable_url(self):
        # Integration: _yt_dlp_search must drop unresolvable entries
        # rather than let bare IDs pass through.
        import types

        import backend.services.cover_search as cs

        fake_response = {
            "entries": [
                {"url": "dQw4w9WgXcQ", "id": "dQw4w9WgXcQ",
                 "title": "t1", "channel": "c1"},
                {"title": "t2", "channel": "c2"},  # dropped — no URL
                {"webpage_url": "https://www.youtube.com/watch?v=abc12345678",
                 "id": "abc12345678", "title": "t3", "channel": "c3"},
            ]
        }

        class _FakeYDL:
            def __init__(self, opts): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def extract_info(self, query, download): return fake_response

        fake_yt_dlp = types.SimpleNamespace(YoutubeDL=_FakeYDL)
        with patch.dict("sys.modules", {"yt_dlp": fake_yt_dlp}):
            entries = cs._yt_dlp_search("anything", top_k=5)

        assert len(entries) == 2
        assert entries[0]["url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert entries[1]["url"] == "https://www.youtube.com/watch?v=abc12345678"


# ---------------------------------------------------------------------------
# Layer 7: multi-variant search — piano + chiptune
# ---------------------------------------------------------------------------
#
# The scorer was originally piano-only. Full-band pop mixes are a nightmare
# for Basic Pitch, but so are many "piano covers" on YouTube — the non-
# allowlist ones cap at 60 and half the time there IS a cleaner alternative:
# 8-bit / chiptune covers. Chiptune audio is dramatically easier to transcribe
# (monophonic channels, pure square/triangle waves, no reverb, no drums
# mixed into pitched content) so when a chiptune cover of a song exists,
# it's a BETTER source than a piano cover for our pipeline.
#
# The multi-variant refactor: ``find_clean_source`` runs the search once per
# variant (piano + chiptune by default), scores each variant's candidates
# against its own channel allowlist + keywords, and returns the single
# highest-scoring result across all variants. Backward compatibility:
# ``find_piano_cover`` stays as a thin wrapper that runs the piano variant
# only so existing callers keep their narrow behavior.


class TestPianoTierPreference:
    """The piano allowlist is tiered: easy beats moderate beats (disabled)
    advanced. Advanced channels score 0 from channel alone — they only get
    points from title/artist matches and keywords, which caps them below
    the 60 threshold unless a chiptune variant rescues the song."""

    def test_easy_tier_channel_gets_60(self):
        # Easy-tier gets +50 base + +10 tier bonus = 60.
        from backend.services.cover_search import score_candidate
        entry = _entry(title="My Song", channel="Pianote")
        assert score_candidate(entry, wanted_title="unrelated", wanted_artist=None) == 60

    def test_moderate_tier_channel_gets_50(self):
        # Moderate-tier gets +50 base, no tier bonus.
        from backend.services.cover_search import score_candidate
        entry = _entry(title="My Song", channel="Jacob's Piano")
        assert score_candidate(entry, wanted_title="unrelated", wanted_artist=None) == 50

    def test_advanced_tier_channel_gets_zero_from_channel(self):
        # Advanced channels (Rousseau, Pietschmann, Kyle Landry, etc.) are
        # defined but NOT in the active allowlist. They get no channel
        # bonus — title/artist/keyword matches only.
        from backend.services.cover_search import score_candidate
        for channel in ("Rousseau", "Patrik Pietschmann", "Kyle Landry"):
            entry = _entry(title="My Song", channel=channel)
            score = score_candidate(
                entry, wanted_title="unrelated", wanted_artist=None,
            )
            assert score == 0, f"{channel!r} should score 0 from channel alone"

    def test_advanced_channel_scores_below_threshold_for_typical_match(self):
        # A Rousseau Bohemian Rhapsody Piano Cover match under the
        # CURRENT active allowlist scores: 0 (advanced not active) +
        # 30 (piano cover keyword) + 20 (title) + 10 (artist) = 60.
        # Exactly at the default threshold. One bad keyword (live
        # recording, karaoke) would drop it. Title-only (no "piano
        # cover" keyword in the title) would drop it below. This
        # documents the intended "advanced rarely wins" behavior.
        from backend.services.cover_search import score_candidate
        entry = _entry(
            title="Bohemian Rhapsody - Queen (Piano Cover)",
            channel="Rousseau",
        )
        score = score_candidate(
            entry, wanted_title="Bohemian Rhapsody", wanted_artist="Queen",
        )
        assert score == 60

    def test_easy_tier_perfect_match_scores_120(self):
        # Easy-tier gets the full stack: 50 + 10 easy bonus + 30 piano
        # cover keyword + 20 title + 10 artist = 120. This is the new
        # ceiling for scoring under the easy/moderate-only active
        # allowlist.
        from backend.services.cover_search import score_candidate
        entry = _entry(
            title="Bohemian Rhapsody - Queen (Easy Piano Cover)",
            channel="Pianote",
        )
        score = score_candidate(
            entry, wanted_title="Bohemian Rhapsody", wanted_artist="Queen",
        )
        assert score == 120

    def test_easy_beats_moderate_when_tied_elsewhere(self):
        # Two candidates with identical title/artist/keyword scoring:
        # the easy-tier one wins because of the +10 easy bonus.
        from backend.services.cover_search import find_piano_cover

        search_results = [
            _entry(
                title="Song X - Piano Cover",
                channel="Jacob's Piano",   # moderate: 50 + 30 + 20 = 100
            ),
            _entry(
                title="Song X - Piano Cover",
                channel="Pianote",          # easy: 50 + 10 + 30 + 20 = 110
            ),
        ]

        with patch("backend.services.cover_search._yt_dlp_search") as mock_search:
            mock_search.return_value = search_results
            result = find_piano_cover(title="Song X", artist=None)

        assert result is not None
        assert result.channel == "Pianote"
        assert result.score == 110

    def test_advanced_tier_exported_for_future_reactivation(self):
        # PIANO_ADVANCED_CHANNELS should remain defined in the module so
        # a future PR can reactivate the tier by adding it to the active
        # allowlist. This test locks in the contract and catches a
        # silent deletion of the tier data.
        from backend.services.cover_search import (
            COVER_CHANNEL_ALLOWLIST,
            PIANO_ADVANCED_CHANNELS,
            PIANO_EASY_CHANNELS,
            PIANO_MODERATE_CHANNELS,
        )
        # Advanced list is defined and nonempty.
        assert len(PIANO_ADVANCED_CHANNELS) >= 3
        # But it is NOT in the active allowlist.
        for ch in PIANO_ADVANCED_CHANNELS:
            assert ch not in COVER_CHANNEL_ALLOWLIST, (
                f"{ch!r} should not be in the active allowlist"
            )
        # The active allowlist equals easy + moderate, nothing else.
        assert set(COVER_CHANNEL_ALLOWLIST) == (
            set(PIANO_EASY_CHANNELS) | set(PIANO_MODERATE_CHANNELS)
        )


# ---------------------------------------------------------------------------
# Layer 7b: trusted tutorial channels
# ---------------------------------------------------------------------------
#
# Some channels brand themselves as "tutorial" but produce clean,
# transcription-quality audio — Synthesia-style rendered piano with no
# voiceover or backing track. For those channels the ``-20`` tutorial
# keyword penalty is the wrong signal: it marks them down for what they
# call themselves rather than what they actually sound like.
#
# We maintain a small ``TRUSTED_TUTORIAL_CHANNELS`` exemption list.
# Channels in it score the normal allowlist + title + artist bonuses
# without the tutorial penalty deducted. Matches against yt-dlp's
# ``uploader_id`` (the ``@handle`` form) so the identifier is unique
# even for channels whose display names are common words.
#
# First entry: Kesh Piano Music (``@keshpianomusic``) — their entire
# catalog is "Piano Tutorial + MIDI" but the audio is pure rendered
# piano which Basic Pitch transcribes beautifully.


class TestTrustedTutorialChannels:
    """Trusted-tutorial channels bypass the ``-20`` tutorial penalty.

    The canonical first member is Kesh (``@keshpianomusic``) whose
    catalog is all "Piano Tutorial" videos but whose audio is
    transcription-quality piano synthesis.
    """

    def test_trusted_tutorial_list_exists_and_contains_kesh(self):
        from backend.services.cover_search import TRUSTED_TUTORIAL_CHANNELS
        assert isinstance(TRUSTED_TUTORIAL_CHANNELS, (list, tuple, frozenset, set))
        assert any("kesh" in entry.lower() for entry in TRUSTED_TUTORIAL_CHANNELS), (
            f"Kesh should seed the trusted-tutorial list: {TRUSTED_TUTORIAL_CHANNELS}"
        )

    def test_trusted_tutorial_entries_are_lowercase(self):
        from backend.services.cover_search import TRUSTED_TUTORIAL_CHANNELS
        for entry in TRUSTED_TUTORIAL_CHANNELS:
            assert entry == entry.lower(), f"{entry!r} should be lowercase"

    def test_kesh_in_piano_moderate_allowlist(self):
        # Kesh is on the piano moderate tier (not easy — his tutorials
        # are for learners but aren't "beginner" simplified).
        from backend.services.cover_search import PIANO_MODERATE_CHANNELS
        assert any(
            "kesh" in entry.lower() for entry in PIANO_MODERATE_CHANNELS
        ), f"Kesh should be in PIANO_MODERATE_CHANNELS: {PIANO_MODERATE_CHANNELS}"

    def test_kesh_match_via_uploader_id_handle(self):
        # yt-dlp returns channel="Kesh" (too broad for substring match
        # because of the "Kesha" collision risk) but also exposes
        # uploader_id="@keshpianomusic" which is unambiguous. The
        # scorer must check uploader_id alongside channel so matching
        # on the handle works.
        from backend.services.cover_search import score_candidate
        entry = _entry(
            title="River Flows In You - Yiruma - Piano Tutorial + MIDI",
            channel="Kesh",
            uploader_id="@keshpianomusic",
        )
        score = score_candidate(
            entry, wanted_title="River Flows In You", wanted_artist="Yiruma",
        )
        # allowlist +50 (via uploader_id match) + title match +20 +
        # artist match +10 - tutorial penalty (SKIPPED for trusted
        # tutorial) = 80
        assert score == 80, (
            f"Kesh + title + artist should score 80 with tutorial penalty "
            f"exempted; got {score}"
        )

    def test_kesh_tutorial_penalty_exempt(self):
        # Specific regression: the -20 tutorial penalty must NOT apply
        # when the channel is trusted, even though "tutorial" appears
        # in the title.
        from backend.services.cover_search import score_candidate
        entry = _entry(
            title="Some Song - Piano Tutorial",
            channel="Kesh",
            uploader_id="@keshpianomusic",
        )
        # allowlist +50 only; no title/artist match; tutorial penalty
        # skipped because trusted. Should score 50.
        score = score_candidate(
            entry, wanted_title="unrelated song", wanted_artist=None,
        )
        assert score == 50

    def test_non_trusted_tutorial_still_penalized(self):
        # Regression: the exemption must be SPECIFIC to the trusted list.
        # A random channel with "tutorial" in the title still gets -20.
        from backend.services.cover_search import score_candidate
        entry = _entry(
            title="River Flows In You - Piano Tutorial",
            channel="Random Piano Channel",
            uploader_id="@randomchannel",
        )
        # No allowlist, no piano-cover keyword ("piano tutorial"
        # doesn't match), title match +20, no artist, tutorial -20.
        # Net: 0.
        score = score_candidate(
            entry, wanted_title="River Flows In You", wanted_artist=None,
        )
        assert score == 0

    def test_kesh_allowlist_does_not_false_match_kesha_pop_channel(self):
        # Defensive regression: the pop artist "Kesha" must not trip
        # the Kesh allowlist. We match on uploader_id handle, not on
        # channel name substring, so a Kesha-named channel without
        # the @keshpianomusic handle should score 0.
        from backend.services.cover_search import score_candidate
        entry = _entry(
            title="Kesha Tik Tok Piano Cover",
            channel="Kesha Fan Covers",
            uploader_id="@keshafancovers",
        )
        # No allowlist match (Kesha != keshpianomusic), piano cover
        # keyword +30, title/artist +10 (if matched). Should NOT get
        # the +50 allowlist bonus.
        score = score_candidate(
            entry, wanted_title="Tik Tok", wanted_artist="Kesha",
        )
        # piano cover keyword +30, title +20, artist +10 = 60 (no
        # allowlist match = no +50). Confirms the handle-based lookup
        # is specific enough.
        assert score == 60


class TestChiptuneChannelAllowlist:
    """Companion to the piano allowlist — a lowercase list of YouTube
    channels that reliably post clean 8-bit covers of popular songs."""

    def test_chiptune_allowlist_exists_and_nonempty(self):
        from backend.services.cover_search import CHIPTUNE_CHANNEL_ALLOWLIST
        assert isinstance(CHIPTUNE_CHANNEL_ALLOWLIST, (list, tuple, frozenset, set))
        assert len(CHIPTUNE_CHANNEL_ALLOWLIST) >= 3  # Seeded with a few handpicked channels.

    def test_chiptune_allowlist_entries_are_lowercase(self):
        from backend.services.cover_search import CHIPTUNE_CHANNEL_ALLOWLIST
        for channel in CHIPTUNE_CHANNEL_ALLOWLIST:
            assert channel == channel.lower(), f"{channel!r} should be lowercase"

    def test_chiptune_allowlist_disjoint_from_piano_allowlist(self):
        # Same channel should never be in both lists — it would confuse
        # the scoring (which variant did the match come from?).
        from backend.services.cover_search import (
            CHIPTUNE_CHANNEL_ALLOWLIST,
            COVER_CHANNEL_ALLOWLIST,
        )
        overlap = set(CHIPTUNE_CHANNEL_ALLOWLIST) & set(COVER_CHANNEL_ALLOWLIST)
        assert overlap == set(), f"channels in both lists: {overlap}"


class TestScoreCandidateChiptuneVariant:
    """When scoring for the chiptune variant, the +50 / +30 weights come
    from CHIPTUNE_CHANNEL_ALLOWLIST and chiptune keywords instead of the
    piano defaults."""

    def test_chiptune_allowlist_channel_adds_50(self):
        from backend.services.cover_search import (
            CHIPTUNE_VARIANT,
            score_candidate_for_variant,
        )
        entry = _entry(title="Bohemian Rhapsody", channel="8-Bit Universe")
        score = score_candidate_for_variant(
            entry,
            wanted_title="unrelated song",
            wanted_artist=None,
            variant=CHIPTUNE_VARIANT,
        )
        assert score == 50

    def test_chiptune_keyword_adds_30(self):
        from backend.services.cover_search import (
            CHIPTUNE_VARIANT,
            score_candidate_for_variant,
        )
        entry = _entry(title="Bohemian Rhapsody (8 Bit Cover)", channel="Random")
        score = score_candidate_for_variant(
            entry,
            wanted_title="unrelated",
            wanted_artist=None,
            variant=CHIPTUNE_VARIANT,
        )
        assert score == 30

    def test_chiptune_chiptune_keyword_also_counts(self):
        from backend.services.cover_search import (
            CHIPTUNE_VARIANT,
            score_candidate_for_variant,
        )
        entry = _entry(title="Some Song - Chiptune Version", channel="Random")
        score = score_candidate_for_variant(
            entry,
            wanted_title="unrelated",
            wanted_artist=None,
            variant=CHIPTUNE_VARIANT,
        )
        assert score == 30

    def test_chiptune_scoring_ignores_piano_keywords(self):
        # A "piano cover" result should NOT get the +30 boost when scored
        # against the chiptune variant — the variants are independent.
        from backend.services.cover_search import (
            CHIPTUNE_VARIANT,
            score_candidate_for_variant,
        )
        entry = _entry(title="Bohemian Rhapsody Piano Cover", channel="Random")
        score = score_candidate_for_variant(
            entry,
            wanted_title="Bohemian Rhapsody",
            wanted_artist=None,
            variant=CHIPTUNE_VARIANT,
        )
        # +20 title match only. No chiptune keywords, no chiptune allowlist.
        assert score == 20

    def test_chiptune_allowlist_does_not_score_piano_variant(self):
        from backend.services.cover_search import (
            PIANO_VARIANT,
            score_candidate_for_variant,
        )
        entry = _entry(title="Bohemian Rhapsody", channel="8-Bit Universe")
        score = score_candidate_for_variant(
            entry,
            wanted_title="Bohemian Rhapsody",
            wanted_artist=None,
            variant=PIANO_VARIANT,
        )
        # Only +20 title match — 8-Bit Universe isn't in the piano allowlist
        # and the title has no piano keywords.
        assert score == 20

    def test_chiptune_perfect_match_max_score(self):
        from backend.services.cover_search import (
            CHIPTUNE_VARIANT,
            score_candidate_for_variant,
        )
        entry = _entry(
            title="Bohemian Rhapsody - Queen (8 Bit Cover)",
            channel="8-Bit Universe",
        )
        score = score_candidate_for_variant(
            entry,
            wanted_title="Bohemian Rhapsody",
            wanted_artist="Queen",
            variant=CHIPTUNE_VARIANT,
        )
        # allowlist +50, chiptune keyword +30, title match +20, artist +10 = 110
        assert score == 110


class TestFindCleanSource:
    """The multi-variant orchestrator. Runs the search once per variant
    and returns the highest-scoring candidate across all of them."""

    def test_returns_piano_match_when_it_outscores_chiptune(self):
        from backend.services.cover_search import find_clean_source

        # piano query returns a Jacob's Piano (moderate tier) match → score 110
        # chiptune query returns a Random channel with only title+keyword → 50
        def _search(query, *, top_k=5):
            if "piano cover" in query:
                return [_entry(
                    title="Bohemian Rhapsody - Queen (Piano Cover)",
                    channel="Jacob's Piano",
                )]
            if "8 bit" in query:
                return [_entry(
                    title="Bohemian Rhapsody 8 bit cover",
                    channel="Random Chip Channel",
                )]
            return []

        with patch("backend.services.cover_search._yt_dlp_search", side_effect=_search):
            result = find_clean_source("Bohemian Rhapsody", "Queen")

        assert result is not None
        assert result.channel == "Jacob's Piano"
        assert result.score == 110

    def test_returns_chiptune_match_when_it_outscores_piano(self):
        from backend.services.cover_search import find_clean_source

        # piano query returns a non-allowlist result with weak match → 20
        # chiptune query returns a strong allowlist match → 110
        def _search(query, *, top_k=5):
            if "piano cover" in query:
                return [_entry(title="Bohemian Rhapsody weak", channel="Random")]
            if "8 bit" in query:
                return [_entry(
                    title="Bohemian Rhapsody - Queen (8 Bit Cover)",
                    channel="8-Bit Universe",
                )]
            return []

        with patch("backend.services.cover_search._yt_dlp_search", side_effect=_search):
            result = find_clean_source("Bohemian Rhapsody", "Queen")

        assert result is not None
        assert result.channel == "8-Bit Universe"
        assert result.score == 110

    def test_returns_none_when_both_variants_below_threshold(self):
        from backend.services.cover_search import find_clean_source

        def _search(query, *, top_k=5):
            return [_entry(title="Unrelated", channel="Random")]

        with patch("backend.services.cover_search._yt_dlp_search", side_effect=_search):
            result = find_clean_source("Bohemian Rhapsody", "Queen", min_score=60)

        assert result is None

    def test_one_variant_failing_does_not_block_the_other(self):
        # If piano search crashes but chiptune succeeds, we should still
        # return the chiptune result instead of silently returning None.
        from backend.services.cover_search import find_clean_source

        def _search(query, *, top_k=5):
            if "piano cover" in query:
                raise RuntimeError("piano search hiccup")
            return [_entry(
                title="Bohemian Rhapsody - Queen 8 bit cover",
                channel="8-Bit Universe",
            )]

        with patch("backend.services.cover_search._yt_dlp_search", side_effect=_search):
            result = find_clean_source("Bohemian Rhapsody", "Queen")

        assert result is not None
        assert result.channel == "8-Bit Universe"

    def test_queries_every_variant_even_when_first_yields_match(self):
        # We don't short-circuit: every variant runs so the scorer can
        # pick the absolute best across all of them. This test guards
        # against a "first-match-wins" regression.
        from backend.services.cover_search import find_clean_source

        calls: list[str] = []

        def _search(query, *, top_k=5):
            calls.append(query)
            return [_entry(title="Generic", channel="Random")]

        with patch("backend.services.cover_search._yt_dlp_search", side_effect=_search):
            find_clean_source("Bohemian Rhapsody", "Queen")

        # Both "piano cover" and "8 bit cover" queries should have fired.
        assert any("piano cover" in q for q in calls), calls
        assert any("8 bit" in q for q in calls), calls

    def test_find_piano_cover_remains_piano_only_for_backward_compat(self):
        # Explicit regression test: find_piano_cover MUST keep running
        # the piano variant only. Downstream mocks (test_ingest_cover_search)
        # still patch this name and count on the narrow semantics.
        from backend.services.cover_search import find_piano_cover

        calls: list[str] = []

        def _search(query, *, top_k=5):
            calls.append(query)
            return []

        with patch("backend.services.cover_search._yt_dlp_search", side_effect=_search):
            find_piano_cover("Bohemian Rhapsody", "Queen")

        # Only the piano query should have fired.
        assert len(calls) == 1
        assert "piano cover" in calls[0]
        assert "8 bit" not in calls[0]
