"""Tests for YouTube URL ingestion in IngestService.

TDD: these tests are written FIRST, before the implementation.
They define the expected behavior for YouTube URL detection and download.
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
from backend.services.ingest import IngestService, extract_youtube_id, is_youtube_url

# ---------------------------------------------------------------------------
# Unit tests: YouTube URL detection
# ---------------------------------------------------------------------------


class TestIsYoutubeUrl:
    """is_youtube_url() should match common YouTube URL formats and reject
    everything else — plain song titles, Spotify links, empty strings, etc."""

    def test_standard_watch_url(self):
        assert is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is True

    def test_short_url(self):
        assert is_youtube_url("https://youtu.be/dQw4w9WgXcQ") is True

    def test_no_www(self):
        assert is_youtube_url("https://youtube.com/watch?v=dQw4w9WgXcQ") is True

    def test_http_without_s(self):
        assert is_youtube_url("http://youtube.com/watch?v=dQw4w9WgXcQ") is True

    def test_with_extra_params(self):
        assert is_youtube_url("https://youtube.com/watch?v=dQw4w9WgXcQ&t=42") is True

    def test_music_youtube(self):
        assert is_youtube_url("https://music.youtube.com/watch?v=dQw4w9WgXcQ") is True

    def test_plain_title_is_not_youtube(self):
        assert is_youtube_url("Yesterday") is False

    def test_empty_string_is_not_youtube(self):
        assert is_youtube_url("") is False

    def test_spotify_url_is_not_youtube(self):
        assert is_youtube_url("https://open.spotify.com/track/abc123") is False

    def test_none_is_not_youtube(self):
        assert is_youtube_url(None) is False


class TestExtractYoutubeId:
    """extract_youtube_id() should pull the 11-character video ID from any
    supported YouTube URL format."""

    def test_standard_watch_url(self):
        assert extract_youtube_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_short_url(self):
        assert extract_youtube_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_with_extra_params(self):
        assert extract_youtube_id("https://youtube.com/watch?v=dQw4w9WgXcQ&t=42") == "dQw4w9WgXcQ"

    def test_music_youtube(self):
        assert extract_youtube_id("https://music.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_returns_none_for_non_youtube(self):
        assert extract_youtube_id("Yesterday") is None


# ---------------------------------------------------------------------------
# Integration tests: IngestService with YouTube URLs
# ---------------------------------------------------------------------------


class TestIngestServiceYoutube:
    """When IngestService.run() receives a title_lookup bundle where the title
    is a YouTube URL, it should download the audio and return a bundle with
    audio populated."""

    def _make_youtube_bundle(self, url: str = "https://youtube.com/watch?v=dQw4w9WgXcQ") -> InputBundle:
        return InputBundle(
            schema_version=SCHEMA_VERSION,
            audio=None,
            midi=None,
            metadata=InputMetadata(title=url, artist=None, source="title_lookup"),
        )

    @pytest.fixture
    def blob_store(self, tmp_path):
        from backend.storage.local import LocalBlobStore
        return LocalBlobStore(tmp_path / "blob")

    @pytest.fixture
    def service(self, blob_store):
        return IngestService(blob_store=blob_store)

    def test_youtube_url_populates_audio_on_bundle(self, service, blob_store):
        """After ingesting a YouTube URL, the returned bundle should have
        audio set to a RemoteAudioFile with a valid BlobStore URI."""
        bundle = self._make_youtube_bundle()

        with patch("backend.services.ingest._download_youtube_sync") as mock_dl:
            # Simulate yt-dlp returning a WAV file + metadata tuple
            mock_dl.return_value = (
                RemoteAudioFile(
                    uri="file:///tmp/blob/yt_dQw4w9WgXcQ.wav",
                    format="wav",
                    sample_rate=44100,
                    duration_sec=180.0,
                    channels=2,
                    content_hash="abc123",
                ),
                "Never Gonna Give You Up",
                "Rick Astley",
            )

            result = asyncio.run(service.run(bundle))

        assert result.audio is not None
        assert result.audio.format == "wav"
        assert result.audio.duration_sec == 180.0
        mock_dl.assert_called_once()

    def test_youtube_url_extracts_video_title(self, service):
        """After ingesting a YouTube URL, metadata.title should be the
        video title from yt-dlp, not the raw URL."""
        url = "https://youtube.com/watch?v=dQw4w9WgXcQ"
        bundle = self._make_youtube_bundle(url)

        with patch("backend.services.ingest._download_youtube_sync") as mock_dl:
            mock_dl.return_value = (
                RemoteAudioFile(
                    uri="file:///tmp/fake.wav",
                    format="wav",
                    sample_rate=44100,
                    duration_sec=60.0,
                    channels=2,
                ),
                "Never Gonna Give You Up",
                "Rick Astley",
            )
            result = asyncio.run(service.run(bundle))

        assert result.metadata.title == "Never Gonna Give You Up"
        assert result.metadata.artist == "Rick Astley"
        assert result.metadata.source == "title_lookup"

    def test_non_youtube_title_passes_through_unchanged(self, service):
        """A plain song title (not a YouTube URL) should pass through
        without attempting a download — audio stays None."""
        bundle = InputBundle(
            schema_version=SCHEMA_VERSION,
            audio=None,
            midi=None,
            metadata=InputMetadata(title="Yesterday", artist="The Beatles", source="title_lookup"),
        )

        with patch("backend.services.ingest._download_youtube_sync") as mock_dl:
            result = asyncio.run(service.run(bundle))

        mock_dl.assert_not_called()
        assert result.audio is None

    def test_audio_upload_skips_youtube_detection(self, service):
        """If the bundle already has audio (audio_upload variant), YouTube
        detection should not run — even if title looks like a URL."""
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
            ),
        )

        with patch("backend.services.ingest._download_youtube_sync") as mock_dl:
            result = asyncio.run(service.run(bundle))

        mock_dl.assert_not_called()
        assert result.audio.uri == "file:///tmp/uploaded.wav"


# ---------------------------------------------------------------------------
# E2E test: YouTube URL through full job API
# ---------------------------------------------------------------------------


def test_youtube_url_job_runs_full_variant(client):
    """Submitting a YouTube URL as the title should trigger the 'full'
    pipeline variant and run to completion (with mocked yt-dlp)."""
    with patch("backend.services.ingest._download_youtube_sync") as mock_dl:
        mock_dl.return_value = (
            RemoteAudioFile(
                uri="file:///tmp/fake.wav",
                format="wav",
                sample_rate=44100,
                duration_sec=60.0,
                channels=2,
            ),
            "Rick Astley - Never Gonna Give You Up",
            "Rick Astley",
        )

        resp = client.post("/v1/jobs", json={
            "title": "https://youtube.com/watch?v=dQw4w9WgXcQ",
        })
        assert resp.status_code == 202
        job = resp.json()
        assert job["variant"] == "full"

        import time
        deadline = time.time() + 5
        status = None
        while time.time() < deadline:
            status = client.get(f"/v1/jobs/{job['job_id']}").json()
            if status["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.05)

        assert status["status"] == "succeeded", status
