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
        cover_url = "https://youtube.com/watch?v=ROUSSEAU_COVER"

        with (
            patch("backend.services.ingest.probe_youtube_metadata") as mock_probe,
            patch("backend.services.ingest.find_piano_cover") as mock_find,
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
            patch("backend.services.ingest.find_piano_cover") as mock_find,
            patch("backend.services.ingest._download_youtube_sync") as mock_dl,
        ):
            mock_probe.return_value = ("bohemian rhapsody", "Queen")
            mock_find.return_value = None  # below threshold → no swap
            mock_dl.side_effect = lambda url, _bs: _fake_downloaded_audio(url)

            asyncio.run(service.run(bundle))

        # probe is called with the original URL
        mock_probe.assert_called_once_with("https://youtu.be/fJ9rUzIMcZQ")
        # find_piano_cover gets the probed title + artist
        mock_find.assert_called_once()
        call = mock_find.call_args
        # title is first positional arg, artist is second
        assert call.args[0] == "bohemian rhapsody"
        assert call.args[1] == "Queen"


# ---------------------------------------------------------------------------
# Fallback path: no cover / probe failed / search failed
# ---------------------------------------------------------------------------


class TestCoverSearchFallsBackToOriginal:
    """Any failure in the cover-search chain must leave the original
    URL in place — the silent-failure contract."""

    def test_falls_back_when_find_piano_cover_returns_none(self, service):
        original_url = "https://youtube.com/watch?v=originalVID"
        bundle = _make_bundle(original_url, prefer_clean_source=True)

        with (
            patch("backend.services.ingest.probe_youtube_metadata") as mock_probe,
            patch("backend.services.ingest.find_piano_cover") as mock_find,
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
        original_url = "https://youtube.com/watch?v=originalVID"
        bundle = _make_bundle(original_url, prefer_clean_source=True)

        with (
            patch("backend.services.ingest.probe_youtube_metadata") as mock_probe,
            patch("backend.services.ingest.find_piano_cover") as mock_find,
            patch("backend.services.ingest._download_youtube_sync") as mock_dl,
        ):
            mock_probe.return_value = None  # probe failed
            mock_dl.side_effect = lambda url, _bs: _fake_downloaded_audio(url)

            asyncio.run(service.run(bundle))

        # find_piano_cover should NOT even be called if the probe failed —
        # no point searching for a song we couldn't identify.
        mock_find.assert_not_called()
        called_url = mock_dl.call_args.args[0]
        assert called_url == original_url

    def test_falls_back_when_find_piano_cover_raises(self, service):
        # find_piano_cover is documented as silent-failure, but defensive
        # depth: if it somehow does raise, ingest still must not crash.
        original_url = "https://youtube.com/watch?v=originalVID"
        bundle = _make_bundle(original_url, prefer_clean_source=True)

        with (
            patch("backend.services.ingest.probe_youtube_metadata") as mock_probe,
            patch("backend.services.ingest.find_piano_cover") as mock_find,
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
            patch("backend.services.ingest.find_piano_cover") as mock_find,
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
            patch("backend.services.ingest.find_piano_cover") as mock_find,
            patch("backend.services.ingest._download_youtube_sync") as mock_dl,
        ):
            mock_dl.side_effect = lambda url, _bs: _fake_downloaded_audio(url)
            asyncio.run(service.run(bundle))

        mock_probe.assert_not_called()
        mock_find.assert_not_called()

    def test_uses_configured_min_score_threshold(self, service, monkeypatch):
        # If operator bumps OHSHEET_COVER_SEARCH_MIN_SCORE=90, that value
        # must propagate into find_piano_cover's min_score parameter.
        import backend.services.ingest as ingest_mod
        monkeypatch.setattr(ingest_mod.settings, "cover_search_min_score", 90)

        bundle = _make_bundle(prefer_clean_source=True)

        with (
            patch("backend.services.ingest.probe_youtube_metadata") as mock_probe,
            patch("backend.services.ingest.find_piano_cover") as mock_find,
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
            patch("backend.services.ingest.find_piano_cover") as mock_find,
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
            patch("backend.services.ingest.find_piano_cover") as mock_find,
        ):
            asyncio.run(service.run(bundle))

        mock_probe.assert_not_called()
        mock_find.assert_not_called()
