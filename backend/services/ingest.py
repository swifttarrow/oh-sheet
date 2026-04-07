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

    async def run(self, payload: InputBundle) -> InputBundle:
        audio = payload.audio
        midi = payload.midi
        if audio is not None:
            audio = await asyncio.to_thread(_probe_audio_sync, audio)
        if midi is not None:
            midi = await asyncio.to_thread(_probe_midi_sync, midi)
        return payload.model_copy(update={
            "schema_version": SCHEMA_VERSION,
            "audio": audio,
            "midi": midi,
        })

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
