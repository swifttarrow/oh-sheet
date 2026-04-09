"""Ingestion stage — best-effort metadata probe.

Runs on whatever's already in the InputBundle:

  * audio  → librosa probe to fill duration_sec / sample_rate / channels
  * midi   → pretty_midi parse to fill ticks_per_beat
  * title  → pass-through (real title-lookup is a future stage)

Both probes are *best effort*: if librosa / pretty_midi aren't installed,
the URI isn't a local file, or the bytes don't parse, we log and pass the
bundle through unchanged so downstream stages still run.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from urllib.parse import urlparse

from backend.contracts import (
    SCHEMA_VERSION,
    InputBundle,
    InputMetadata,
    RemoteAudioFile,
    RemoteMidiFile,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# YouTube URL helpers — stubs, implementation TBD
# ---------------------------------------------------------------------------

_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}


def is_youtube_url(value: str | None) -> bool:
    """Return True if *value* looks like a YouTube video URL."""
    if not value:
        return False
    return extract_youtube_id(value) is not None


def extract_youtube_id(url: str) -> str | None:
    """Extract the 11-character video ID from a YouTube URL, or None."""
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    host = (parsed.hostname or "").lower()
    if host not in _YOUTUBE_HOSTS:
        return None
    # youtu.be/<id>
    video_id: str | None
    if host == "youtu.be":
        video_id = parsed.path.lstrip("/").split("/")[0]
    else:
        # youtube.com/watch?v=<id>
        from urllib.parse import parse_qs
        video_id = parse_qs(parsed.query).get("v", [None])[0]
    if video_id and len(video_id) == 11:
        return video_id
    return None


def _download_youtube_sync(url: str, blob_store) -> RemoteAudioFile:
    """Download audio from a YouTube URL via yt-dlp, store in blob_store.

    Returns a RemoteAudioFile pointing to the stored WAV.
    This is called via asyncio.to_thread() so it can block.
    """
    import hashlib
    import tempfile

    try:
        import yt_dlp
    except ImportError as err:
        raise RuntimeError(
            "yt-dlp is required for YouTube downloads. "
            "Install with: pip install ohsheet[youtube]"
        ) from err

    video_id = extract_youtube_id(url)
    if video_id is None:
        raise ValueError(f"Could not extract video ID from URL: {url}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_template = str(Path(tmp_dir) / "%(id)s.%(ext)s")

        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }],
            "outtmpl": output_template,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "socket_timeout": 30,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
        except Exception as exc:
            raise RuntimeError(f"yt-dlp download failed for {url}: {exc}") from exc

        # Find the downloaded WAV file
        wav_path = Path(tmp_dir) / f"{info['id']}.wav"
        if not wav_path.exists():
            # yt-dlp may use a different naming — find any .wav in the dir
            wav_files = list(Path(tmp_dir).glob("*.wav"))
            if not wav_files:
                raise RuntimeError(f"No WAV file produced by yt-dlp for {url}")
            wav_path = wav_files[0]

        audio_bytes = wav_path.read_bytes()
        content_hash = hashlib.sha256(audio_bytes).hexdigest()

        blob_key = f"youtube/{video_id}.wav"
        uri = blob_store.put_bytes(blob_key, audio_bytes)

        duration = float(info.get("duration", 0))
        sample_rate = int(info.get("asr", 44100) or 44100)

        return RemoteAudioFile(
            uri=uri,
            format="wav",
            sample_rate=sample_rate,
            duration_sec=duration,
            channels=2,
            content_hash=content_hash,
        ), info.get("title"), info.get("uploader") or info.get("channel")


def _file_path(uri: str) -> Path | None:
    """Return a local Path for ``file://`` URIs, or None for anything else."""
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return None
    return Path(parsed.path)


def _probe_audio_sync(audio: RemoteAudioFile) -> RemoteAudioFile:
    """Best-effort soundfile probe. Returns the original on any failure.

    Uses ``soundfile.info`` (libsndfile) instead of librosa because librosa
    falls back to audioread on unparseable bytes — a slow, deprecated path
    that takes several seconds on small fake test inputs. soundfile fails
    fast and only handles formats libsndfile understands, which is exactly
    the behaviour we want for a best-effort metadata probe.
    """
    path = _file_path(audio.uri)
    if path is None or not path.is_file():
        return audio
    try:
        import soundfile  # noqa: PLC0415 — optional dep (ships with librosa)
    except ImportError:
        log.debug("soundfile unavailable; skipping audio probe")
        return audio
    try:
        info = soundfile.info(str(path))
    except Exception as exc:  # noqa: BLE001 — bad bytes shouldn't crash the pipeline
        log.warning("soundfile probe failed for %s: %s", path, exc)
        return audio
    return audio.model_copy(update={
        "duration_sec": float(info.duration),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
    })


def _probe_midi_sync(midi: RemoteMidiFile) -> RemoteMidiFile:
    """Best-effort pretty_midi probe. Returns the original on any failure."""
    path = _file_path(midi.uri)
    if path is None or not path.is_file():
        return midi
    try:
        import pretty_midi  # noqa: PLC0415 — optional dep
    except ImportError:
        log.debug("pretty_midi unavailable; skipping MIDI probe")
        return midi
    try:
        pm = pretty_midi.PrettyMIDI(str(path))
    except Exception as exc:  # noqa: BLE001
        log.warning("pretty_midi probe failed for %s: %s", path, exc)
        return midi
    return midi.model_copy(update={"ticks_per_beat": int(pm.resolution)})


class IngestService:
    name = "ingest"

    def __init__(self, blob_store=None):
        self._blob_store = blob_store

    async def run(self, payload: InputBundle) -> InputBundle:
        log.info(
            "ingest: start source=%s has_audio=%s has_midi=%s",
            payload.metadata.source,
            payload.audio is not None,
            payload.midi is not None,
        )
        audio = payload.audio
        midi = payload.midi

        # YouTube URL detection: if no audio/midi and title is a YouTube URL,
        # download the audio and attach it to the bundle.
        metadata_update: dict = {}
        if (
            audio is None
            and midi is None
            and payload.metadata.source == "title_lookup"
            and payload.metadata.title is not None
            and is_youtube_url(payload.metadata.title)
        ):
            audio, yt_title, yt_artist = await asyncio.to_thread(
                _download_youtube_sync, payload.metadata.title, self._blob_store,
            )
            if yt_title:
                metadata_update["title"] = yt_title
            if yt_artist and not payload.metadata.artist:
                metadata_update["artist"] = yt_artist

        if audio is not None:
            audio = await asyncio.to_thread(_probe_audio_sync, audio)
        if midi is not None:
            midi = await asyncio.to_thread(_probe_midi_sync, midi)

        updated_metadata = payload.metadata
        if metadata_update:
            updated_metadata = payload.metadata.model_copy(update=metadata_update)

        out = payload.model_copy(update={
            "schema_version": SCHEMA_VERSION,
            "audio": audio,
            "midi": midi,
            "metadata": updated_metadata,
        })
        if out.audio is not None:
            log.info(
                "ingest: audio probed duration_sec=%.2f sample_rate=%d channels=%d",
                out.audio.duration_sec,
                out.audio.sample_rate,
                out.audio.channels,
            )
        if out.midi is not None:
            log.info("ingest: midi probed ticks_per_beat=%d", out.midi.ticks_per_beat)
        log.info("ingest: done")
        return out

    # ---- bundle constructors -------------------------------------------------

    @staticmethod
    def from_audio(
        audio: RemoteAudioFile,
        *,
        title: str | None = None,
        artist: str | None = None,
    ) -> InputBundle:
        return InputBundle(
            schema_version=SCHEMA_VERSION,
            audio=audio,
            midi=None,
            metadata=InputMetadata(title=title, artist=artist, source="audio_upload"),
        )

    @staticmethod
    def from_midi(
        midi: RemoteMidiFile,
        *,
        title: str | None = None,
        artist: str | None = None,
    ) -> InputBundle:
        return InputBundle(
            schema_version=SCHEMA_VERSION,
            audio=None,
            midi=midi,
            metadata=InputMetadata(title=title, artist=artist, source="midi_upload"),
        )

    @staticmethod
    def from_title_lookup(title: str, artist: str | None = None) -> InputBundle:
        return InputBundle(
            schema_version=SCHEMA_VERSION,
            audio=None,
            midi=None,
            metadata=InputMetadata(title=title, artist=artist, source="title_lookup"),
        )
