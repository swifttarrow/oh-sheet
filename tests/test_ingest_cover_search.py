"""Tests for the cover_search integration inside IngestService.

When a job arrives with ``prefer_clean_source=True``, the ingest stage
should:

  1. Probe the user's YouTube URL for (title, artist)
  2. Search YouTube for a piano cover of that song
  3. If a candidate clears the score threshold, SWAP the URL that
     gets passed to ``_download_youtube_sync`` for the cover's URL
  4. Otherwise, fall back silently to downloading the original URL

The whole feature is gated by ``cover_search_enabled`` in Settings so
operators can flip it off globally. The silent-failure contract is
preserved — any cover_search exception must fall back to the original
URL, never crash the job.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from backend.contracts import (
    SCHEMA_VERSION,
    InputBundle,
    InputMetadata,
    RemoteAudioFile,
)
from backend.services.cover_search import CoverSearchResult
from backend.services.ingest import IngestService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def blob_store(tmp_path):
    from backend.storage.local import LocalBlobStore
    return LocalBlobStore(tmp_path / "blob")


@pytest.fixture
def service(blob_store):
    return IngestService(blob_store=blob_store)


def _make_bundle(
    url: str = "https://youtube.com/watch?v=dQw4w9WgXcQ",
    *,
    prefer_clean_source: bool = True,
) -> InputBundle:
    return InputBundle(
        schema_version=SCHEMA_VERSION,
        audio=None,
        midi=None,
        metadata=InputMetadata(
            title=url,
            artist=None,
            source="title_lookup",
            prefer_clean_source=prefer_clean_source,
        ),
    )


def _fake_downloaded_audio(source_url: str) -> tuple[RemoteAudioFile, str, str]:
    """Build the tuple _download_youtube_sync normally returns.

    ``source_url`` is embedded in the content_hash so tests can assert
    which URL the downloader was actually called with.
    """
    return (
        RemoteAudioFile(
            uri="file:///tmp/fake.wav",
            format="wav",
            sample_rate=44100,
            duration_sec=60.0,
            channels=2,
            content_hash=f"hash-for-{source_url}",
        ),
        "Downloaded Video Title",
        "Downloaded Uploader",
    )


# ---------------------------------------------------------------------------
# Happy path: cover found, URL swapped
# ---------------------------------------------------------------------------


class TestCoverSearchSwapsUrl:
    """When prefer_clean_source=True and a high-scoring cover exists,
    the ingest stage should pass the cover URL to _download_youtube_sync."""

    def test_swaps_to_cover_url_when_match_clears_threshold(self, service):
        bundle = _make_bundle(prefer_clean_source=True)
        cover_url = "https://youtube.com/watch?v=rouss12cvr1"

        with (
            patch("backend.services.ingest.probe_youtube_metadata") as mock_probe,
            patch("backend.services.ingest.find_clean_source") as mock_find,
            patch("backend.services.ingest._download_youtube_sync") as mock_dl,
        ):
            mock_probe.return_value = ("bohemian rhapsody", "Queen")
            mock_find.return_value = CoverSearchResult(
                url=cover_url,
                score=110,
                channel="Rousseau",
                title="Bohemian Rhapsody - Queen (Piano Cover)",
            )
            mock_dl.side_effect = lambda url, _bs: _fake_downloaded_audio(url)

            result = asyncio.run(service.run(bundle))

        # _download_youtube_sync must have received the COVER url, not the user's.
        mock_dl.assert_called_once()
        called_url = mock_dl.call_args.args[0]
        assert called_url == cover_url
        assert result.audio is not None
        assert result.audio.content_hash == f"hash-for-{cover_url}"

    def test_probe_and_search_receive_correct_args(self, service):
        bundle = _make_bundle(
            url="https://youtu.be/fJ9rUzIMcZQ",
            prefer_clean_source=True,
        )

        with (
            patch("backend.services.ingest.probe_youtube_metadata") as mock_probe,
            patch("backend.services.ingest.find_clean_source") as mock_find,
            patch("backend.services.ingest._download_youtube_sync") as mock_dl,
        ):
            mock_probe.return_value = ("bohemian rhapsody", "Queen")
            mock_find.return_value = None  # below threshold → no swap
            mock_dl.side_effect = lambda url, _bs: _fake_downloaded_audio(url)

            asyncio.run(service.run(bundle))

        # probe is called with the original URL
        mock_probe.assert_called_once_with("https://youtu.be/fJ9rUzIMcZQ")
        # find_clean_source gets the probed title + artist
        mock_find.assert_called_once()
        call = mock_find.call_args
        # Probed title has no " - " separator, so the full title is
        # passed as-is and the YouTube uploader is ignored (unreliable).
        assert call.args[0] == "bohemian rhapsody"
        assert call.args[1] is None


# ---------------------------------------------------------------------------
# Fallback path: no cover / probe failed / search failed
# ---------------------------------------------------------------------------


class TestCoverSearchFallsBackToOriginal:
    """Any failure in the cover-search chain must leave the original
    URL in place — the silent-failure contract."""

    def test_falls_back_when_find_clean_source_returns_none(self, service):
        original_url = "https://youtube.com/watch?v=origvid1111"
        bundle = _make_bundle(original_url, prefer_clean_source=True)

        with (
            patch("backend.services.ingest.probe_youtube_metadata") as mock_probe,
            patch("backend.services.ingest.find_clean_source") as mock_find,
            patch("backend.services.ingest._download_youtube_sync") as mock_dl,
        ):
            mock_probe.return_value = ("some title", "some artist")
            mock_find.return_value = None
            mock_dl.side_effect = lambda url, _bs: _fake_downloaded_audio(url)

            asyncio.run(service.run(bundle))

        # Original URL is what actually gets downloaded.
        called_url = mock_dl.call_args.args[0]
        assert called_url == original_url

    def test_falls_back_when_probe_returns_none(self, service):
        original_url = "https://youtube.com/watch?v=origvid1111"
        bundle = _make_bundle(original_url, prefer_clean_source=True)

        with (
            patch("backend.services.ingest.probe_youtube_metadata") as mock_probe,
            patch("backend.services.ingest.find_clean_source") as mock_find,
            patch("backend.services.ingest._download_youtube_sync") as mock_dl,
        ):
            mock_probe.return_value = None  # probe failed
            mock_dl.side_effect = lambda url, _bs: _fake_downloaded_audio(url)

            asyncio.run(service.run(bundle))

        # find_clean_source should NOT even be called if the probe failed —
        # no point searching for a song we couldn't identify.
        mock_find.assert_not_called()
        called_url = mock_dl.call_args.args[0]
        assert called_url == original_url

    def test_falls_back_when_find_clean_source_raises(self, service):
        # find_clean_source is documented as silent-failure, but defensive
        # depth: if it somehow does raise, ingest still must not crash.
        original_url = "https://youtube.com/watch?v=origvid1111"
        bundle = _make_bundle(original_url, prefer_clean_source=True)

        with (
            patch("backend.services.ingest.probe_youtube_metadata") as mock_probe,
            patch("backend.services.ingest.find_clean_source") as mock_find,
            patch("backend.services.ingest._download_youtube_sync") as mock_dl,
        ):
            mock_probe.return_value = ("title", "artist")
            mock_find.side_effect = RuntimeError("unexpected crash")
            mock_dl.side_effect = lambda url, _bs: _fake_downloaded_audio(url)

            # Must not propagate
            result = asyncio.run(service.run(bundle))

        called_url = mock_dl.call_args.args[0]
        assert called_url == original_url
        assert result.audio is not None

    def test_falls_back_when_cover_result_has_non_youtube_parseable_url(self, service):
        # Defense-in-depth for the (Critical) PR #47 review finding: even
        # if _yt_dlp_search's URL normalization misses a pathological entry
        # and cover_search returns a CoverSearchResult with a bare video ID
        # (or any other string that urlparse can't turn into a YouTube URL),
        # _maybe_swap_for_cover_sync must refuse to propagate it downstream.
        # Otherwise _download_youtube_sync raises ValueError and crashes
        # the whole ingest job — exactly what the silent-failure contract
        # is meant to prevent.
        original_url = "https://youtube.com/watch?v=origvid1111"
        bundle = _make_bundle(original_url, prefer_clean_source=True)

        with (
            patch("backend.services.ingest.probe_youtube_metadata") as mock_probe,
            patch("backend.services.ingest.find_clean_source") as mock_find,
            patch("backend.services.ingest._download_youtube_sync") as mock_dl,
        ):
            mock_probe.return_value = ("bohemian rhapsody", "Queen")
            mock_find.return_value = CoverSearchResult(
                url="dQw4w9WgXcQ",  # ← bare video ID — extract_youtube_id rejects it
                score=110,
                channel="Rousseau",
                title="Bohemian Rhapsody (Piano Cover)",
            )
            mock_dl.side_effect = lambda url, _bs: _fake_downloaded_audio(url)

            # Must not crash — the bad URL is rejected and we fall back.
            asyncio.run(service.run(bundle))

        called_url = mock_dl.call_args.args[0]
        assert called_url == original_url


# ---------------------------------------------------------------------------
# Gating: prefer_clean_source=False and cover_search_enabled=False
# ---------------------------------------------------------------------------


class TestCoverSearchGating:
    """cover_search should only run when BOTH the per-job flag and the
    global setting are on."""

    def test_prefer_clean_source_false_skips_cover_search(self, service):
        bundle = _make_bundle(prefer_clean_source=False)

        with (
            patch("backend.services.ingest.probe_youtube_metadata") as mock_probe,
            patch("backend.services.ingest.find_clean_source") as mock_find,
            patch("backend.services.ingest._download_youtube_sync") as mock_dl,
        ):
            mock_dl.side_effect = lambda url, _bs: _fake_downloaded_audio(url)
            asyncio.run(service.run(bundle))

        mock_probe.assert_not_called()
        mock_find.assert_not_called()

    def test_global_kill_switch_skips_cover_search(self, service, monkeypatch):
        # Simulate OHSHEET_COVER_SEARCH_ENABLED=false via the settings
        # singleton. Ingest reads this at run-time so a fresh setting
        # takes effect without restarting the service.
        import backend.services.ingest as ingest_mod
        monkeypatch.setattr(ingest_mod.settings, "cover_search_enabled", False)

        bundle = _make_bundle(prefer_clean_source=True)

        with (
            patch("backend.services.ingest.probe_youtube_metadata") as mock_probe,
            patch("backend.services.ingest.find_clean_source") as mock_find,
            patch("backend.services.ingest._download_youtube_sync") as mock_dl,
        ):
            mock_dl.side_effect = lambda url, _bs: _fake_downloaded_audio(url)
            asyncio.run(service.run(bundle))

        mock_probe.assert_not_called()
        mock_find.assert_not_called()

    def test_uses_configured_min_score_threshold(self, service, monkeypatch):
        # If operator bumps OHSHEET_COVER_SEARCH_MIN_SCORE=90, that value
        # must propagate into find_clean_source's min_score parameter.
        import backend.services.ingest as ingest_mod
        monkeypatch.setattr(ingest_mod.settings, "cover_search_min_score", 90)

        bundle = _make_bundle(prefer_clean_source=True)

        with (
            patch("backend.services.ingest.probe_youtube_metadata") as mock_probe,
            patch("backend.services.ingest.find_clean_source") as mock_find,
            patch("backend.services.ingest._download_youtube_sync") as mock_dl,
        ):
            mock_probe.return_value = ("title", "artist")
            mock_find.return_value = None
            mock_dl.side_effect = lambda url, _bs: _fake_downloaded_audio(url)
            asyncio.run(service.run(bundle))

        mock_find.assert_called_once()
        # min_score should be passed as kwarg
        assert mock_find.call_args.kwargs.get("min_score") == 90


# ---------------------------------------------------------------------------
# Non-applicable paths: audio upload, non-YouTube title
# ---------------------------------------------------------------------------


class TestCoverSearchSkipsNonYoutubeInputs:
    """Cover search should never run on inputs that don't match the
    YouTube-URL-via-title-lookup shape."""

    def test_non_youtube_title_skips_cover_search(self, service):
        bundle = InputBundle(
            schema_version=SCHEMA_VERSION,
            audio=None,
            midi=None,
            metadata=InputMetadata(
                title="Yesterday",  # plain song title, not a URL
                artist="The Beatles",
                source="title_lookup",
                prefer_clean_source=True,
            ),
        )

        with (
            patch("backend.services.ingest.probe_youtube_metadata") as mock_probe,
            patch("backend.services.ingest.find_clean_source") as mock_find,
            patch("backend.services.ingest._download_youtube_sync") as mock_dl,
        ):
            asyncio.run(service.run(bundle))

        mock_probe.assert_not_called()
        mock_find.assert_not_called()
        mock_dl.assert_not_called()

    def test_audio_upload_skips_cover_search(self, service):
        existing_audio = RemoteAudioFile(
            uri="file:///tmp/uploaded.wav",
            format="wav",
            sample_rate=44100,
            duration_sec=120.0,
            channels=2,
        )
        bundle = InputBundle(
            schema_version=SCHEMA_VERSION,
            audio=existing_audio,
            midi=None,
            metadata=InputMetadata(
                title="https://youtube.com/watch?v=dQw4w9WgXcQ",
                artist=None,
                source="audio_upload",
                prefer_clean_source=True,
            ),
        )

        with (
            patch("backend.services.ingest.probe_youtube_metadata") as mock_probe,
            patch("backend.services.ingest.find_clean_source") as mock_find,
        ):
            asyncio.run(service.run(bundle))

        mock_probe.assert_not_called()
        mock_find.assert_not_called()


# ---------------------------------------------------------------------------
# Static bundle builder: from_title_lookup must accept prefer_clean_source
# ---------------------------------------------------------------------------


class TestFromTitleLookupBuilder:
    """PR #47 review (Important) #3: the static
    IngestService.from_title_lookup builder is used by scripts, tests,
    and any future caller that doesn't hit the API. Before the fix it
    silently dropped the prefer_clean_source flag because the builder
    didn't accept it, so any non-API caller would see the cover_search
    fast path skipped without knowing why. The live API route
    constructs InputBundle directly, so production is unaffected."""

    def test_from_title_lookup_defaults_to_prefer_clean_source_false(self):
        bundle = IngestService.from_title_lookup("Yesterday", artist="The Beatles")
        assert bundle.metadata.prefer_clean_source is False

    def test_from_title_lookup_accepts_prefer_clean_source_true(self):
        bundle = IngestService.from_title_lookup(
            "Yesterday",
            artist="The Beatles",
            prefer_clean_source=True,
        )
        assert bundle.metadata.prefer_clean_source is True

    def test_from_title_lookup_prefer_clean_source_is_keyword_only(self):
        # prefer_clean_source must be keyword-only to prevent positional
        # shadowing of artist (a common mistake with a 2-arg signature).
        import pytest
        with pytest.raises(TypeError):
            IngestService.from_title_lookup("Yesterday", "The Beatles", True)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Metadata threading: cover_search's split-extracted title/artist win
# over the downloaded video's raw yt_title/yt_uploader.
# ---------------------------------------------------------------------------
#
# The engraved sheet's <work-title> and <creator type="composer">
# are filled from whatever ends up in ``payload.metadata.title`` and
# ``.artist`` after ingest. Before this change, both fields came from
# ``_download_youtube_sync`` — which returns the DOWNLOADED video's
# raw title. For a user who pastes the official "Michael Jackson -
# Beat It (Official Video)" URL and gets their job routed to Peter
# Bence's cover ("BEAT IT - Michael Jackson x Peter Bence (Piano
# Cover)"), the resulting sheet read "BEAT IT - Michael Jackson x
# Peter Bence (Piano Cover)" as the title and "Peter Bence" as the
# composer — neither of which is what the user searched for.
#
# After the change, ``_maybe_swap_for_cover_sync`` returns the
# ORIGINAL URL's probed title/artist (split on the first " - " / " – "
# / " | " separator, with trailing parens stripped). The caller
# prefers those values over the raw yt_title/yt_uploader.


class TestMetadataFromCoverSearchWinsOverDownload:
    """When cover_search extracts clean title/artist from the original
    URL, those values must appear in the final metadata — not the
    downloaded cover's raw yt_title/yt_uploader."""

    def test_extracted_title_and_artist_populate_metadata(self, service):
        # Scenario: user pastes MJ's official Beat It URL. Probe returns
        # "Michael Jackson - Beat It (Official Video)" which splits to
        # artist="Michael Jackson" + title="Beat It (Official Video)",
        # then trailing paren strips to "Beat It". Cover_search swaps to
        # Peter Bence's cover. Download returns the cover's raw title,
        # which we must IGNORE in favour of the extracted values.
        original_url = "https://youtu.be/kOn-HdEg6AQ"  # official MJ Beat It
        cover_url = "https://www.youtube.com/watch?v=TjouNZZNE3g"  # Peter Bence
        bundle = _make_bundle(original_url, prefer_clean_source=True)

        with (
            patch("backend.services.ingest.probe_youtube_metadata") as mock_probe,
            patch("backend.services.ingest.find_clean_source") as mock_find,
            patch("backend.services.ingest._download_youtube_sync") as mock_dl,
        ):
            # Probe returns "Artist - Song (Official Video)" shape
            mock_probe.return_value = (
                "Michael Jackson - Beat It (Official Video)",
                "Michael Jackson",
            )
            mock_find.return_value = CoverSearchResult(
                url=cover_url,
                score=100,
                channel="Peter Bence",
                title="BEAT IT - Michael Jackson x Peter Bence (Piano Cover)",
            )
            # Download returns the COVER's messy raw title — which must
            # NOT end up in the final metadata.
            mock_dl.return_value = (
                RemoteAudioFile(
                    uri="file:///tmp/fake.wav",
                    format="wav",
                    sample_rate=44100,
                    duration_sec=193.0,
                    channels=2,
                    content_hash="hash-for-cover",
                ),
                "BEAT IT - Michael Jackson x Peter Bence (Piano Cover)",
                "Peter Bence",
            )

            result = asyncio.run(service.run(bundle))

        # Extracted (split + trailing-paren-stripped) values win
        assert result.metadata.title == "Beat It"
        assert result.metadata.artist == "Michael Jackson"

    def test_no_dash_separator_passes_full_title_as_title(self, service):
        # Video title without an "Artist - Song" separator — the full
        # string becomes the title, artist stays None, and the
        # downloaded video's yt_uploader is used as the fallback artist
        # (only because cover_search's extracted artist is None).
        bundle = _make_bundle(prefer_clean_source=True)

        with (
            patch("backend.services.ingest.probe_youtube_metadata") as mock_probe,
            patch("backend.services.ingest.find_clean_source") as mock_find,
            patch("backend.services.ingest._download_youtube_sync") as mock_dl,
        ):
            # No " - " separator anywhere in the title
            mock_probe.return_value = ("Someday", "the beatles vevo")
            mock_find.return_value = None  # no swap
            mock_dl.return_value = (
                RemoteAudioFile(
                    uri="file:///tmp/fake.wav",
                    format="wav",
                    sample_rate=44100,
                    duration_sec=60.0,
                    channels=2,
                    content_hash="hash",
                ),
                "yt_title_from_download",  # would be wrong
                "yt_uploader_from_download",  # fallback when extracted artist is None
            )

            result = asyncio.run(service.run(bundle))

        # Full probed title is used as title (no split possible)
        assert result.metadata.title == "Someday"
        # Cover_search's extracted artist is None, so caller falls back
        # to the downloaded uploader — acceptable floor behaviour.
        assert result.metadata.artist == "yt_uploader_from_download"

    def test_trailing_bracket_is_stripped(self, service):
        # "(Official Video)" / "[Remastered]" / "(Live at Wembley)" are
        # parenthetical provenance tags that pollute sheet music headers
        # but don't change what song it is. They should be stripped.
        bundle = _make_bundle(prefer_clean_source=True)

        with (
            patch("backend.services.ingest.probe_youtube_metadata") as mock_probe,
            patch("backend.services.ingest.find_clean_source") as mock_find,
            patch("backend.services.ingest._download_youtube_sync") as mock_dl,
        ):
            mock_probe.return_value = (
                "Queen - Bohemian Rhapsody (Official Video)",
                "Queen",
            )
            mock_find.return_value = None
            mock_dl.side_effect = lambda url, _bs: _fake_downloaded_audio(url)

            result = asyncio.run(service.run(bundle))

        assert result.metadata.title == "Bohemian Rhapsody"
        assert result.metadata.artist == "Queen"

    def test_extracted_values_survive_when_no_swap_happens(self, service):
        # Even when cover_search finds no match (returns None), the
        # extracted title/artist should still propagate — they're
        # derived from the probe, not from the swap, and are always a
        # better sheet header than the raw download title.
        bundle = _make_bundle(prefer_clean_source=True)

        with (
            patch("backend.services.ingest.probe_youtube_metadata") as mock_probe,
            patch("backend.services.ingest.find_clean_source") as mock_find,
            patch("backend.services.ingest._download_youtube_sync") as mock_dl,
        ):
            mock_probe.return_value = (
                "Coldplay - Yellow (Official Audio)",
                "Coldplay",
            )
            mock_find.return_value = None  # no swap
            mock_dl.return_value = (
                RemoteAudioFile(
                    uri="file:///tmp/fake.wav",
                    format="wav",
                    sample_rate=44100,
                    duration_sec=60.0,
                    channels=2,
                    content_hash="hash",
                ),
                "Coldplay - Yellow (Official Audio) 4K UHD",  # even messier raw
                "Coldplay",
            )

            result = asyncio.run(service.run(bundle))

        # Extracted clean values, not the raw 4K UHD nonsense from download
        assert result.metadata.title == "Yellow"
        assert result.metadata.artist == "Coldplay"
