"""Engraving stage — STUB.

Real implementation: temp1/engrave.py — emits MusicXML via music21, MIDI from
the humanized performance, optional audio preview via fluidsynth, and renders
a PDF via LilyPond. This stub writes placeholder bytes into the configured
blob store and returns the URIs in the same shape as the real worker.
"""
from __future__ import annotations

import asyncio

from ohsheet.contracts import (
    SCHEMA_VERSION,
    EngravedOutput,
    EngravedScoreData,
    HumanizedPerformance,
    PianoScore,
)
from ohsheet.storage.base import BlobStore


class EngraveService:
    name = "engrave"

    def __init__(self, blob_store: BlobStore) -> None:
        self.blob_store = blob_store

    async def run(
        self,
        payload: HumanizedPerformance | PianoScore,
        *,
        job_id: str,
        title: str = "Untitled",
        composer: str = "Unknown",
    ) -> EngravedOutput:
        await asyncio.sleep(0.1)

        is_humanized = isinstance(payload, HumanizedPerformance)
        prefix = f"jobs/{job_id}/output"

        pdf_uri = self.blob_store.put_bytes(
            f"{prefix}/sheet.pdf",
            b"%PDF-1.4\n% stub PDF emitted by ohsheet engrave service\n",
        )
        musicxml_uri = self.blob_store.put_bytes(
            f"{prefix}/score.musicxml",
            b'<?xml version="1.0" encoding="UTF-8"?><score-partwise version="3.1"/>\n',
        )
        midi_uri = self.blob_store.put_bytes(
            f"{prefix}/humanized.mid",
            b"MThd\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00",
        )

        return EngravedOutput(
            schema_version=SCHEMA_VERSION,
            metadata=EngravedScoreData(
                includes_dynamics=is_humanized,
                includes_pedal_marks=is_humanized,
                includes_fingering=False,
                includes_chord_symbols=False,
                title=title,
                composer=composer,
            ),
            pdf_uri=pdf_uri,
            musicxml_uri=musicxml_uri,
            humanized_midi_uri=midi_uri,
            audio_preview_uri=None,
        )
