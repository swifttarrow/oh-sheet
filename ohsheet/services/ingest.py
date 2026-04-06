"""Ingestion stage — STUB.

Real implementation: temp1/ingest.py. Responsible for resolving the input
(title lookup → fetch audio, audio_upload → probe metadata, midi_upload →
parse ticks_per_beat) and emitting an InputBundle.

This stub passes the bundle through unchanged. Use the static helpers below
when constructing bundles in tests / API handlers.
"""
from __future__ import annotations

import asyncio

from ohsheet.contracts import (
    SCHEMA_VERSION,
    InputBundle,
    InputMetadata,
    RemoteAudioFile,
    RemoteMidiFile,
)


class IngestService:
    name = "ingest"

    async def run(self, payload: InputBundle) -> InputBundle:
        await asyncio.sleep(0.05)  # simulate I/O latency
        return payload.model_copy(update={"schema_version": SCHEMA_VERSION})

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
