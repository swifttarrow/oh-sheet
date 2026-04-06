"""Transcription stage — STUB.

Real implementation: MT4 v5 (custom Conformer + frame-level onset/frame/velocity
heads, 14 GM families × 128 pitches, 50fps). See temp1/orchestrator.py
``_run_transcribe_mt4`` for the wrapping logic — when wiring this up:

  1. Move the MT4 inference into a thread via ``asyncio.to_thread`` (it's
     CPU/GPU-bound and synchronous).
  2. Use the v5 calibrated thresholds (onset=0.45, frame=0.30) — see
     feedback_overreg_calibration.md.
  3. Reuse the family→InstrumentRole mapping from temp1/orchestrator.py.

This stub returns a small fake TranscriptionResult so downstream stages can be
exercised end-to-end without touching the model.
"""
from __future__ import annotations

import asyncio

from ohsheet.contracts import (
    SCHEMA_VERSION,
    HarmonicAnalysis,
    InputBundle,
    InstrumentRole,
    MidiTrack,
    Note,
    QualitySignal,
    SeparatedStems,
    TempoMapEntry,
    TranscriptionResult,
)


class TranscribeService:
    name = "transcribe"

    async def run(self, payload: InputBundle) -> TranscriptionResult:
        await asyncio.sleep(0.1)
        return TranscriptionResult(
            schema_version=SCHEMA_VERSION,
            stems=SeparatedStems(),
            midi_tracks=[
                MidiTrack(
                    notes=[
                        Note(pitch=60, onset_sec=0.0, offset_sec=0.5, velocity=80),
                        Note(pitch=64, onset_sec=0.5, offset_sec=1.0, velocity=80),
                        Note(pitch=67, onset_sec=1.0, offset_sec=1.5, velocity=80),
                        Note(pitch=72, onset_sec=1.5, offset_sec=2.0, velocity=80),
                    ],
                    instrument=InstrumentRole.MELODY,
                    source_stem="stub",
                    confidence=0.7,
                ),
            ],
            analysis=HarmonicAnalysis(
                key="C:major",
                time_signature=(4, 4),
                tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
                chords=[],
                sections=[],
            ),
            quality=QualitySignal(
                overall_confidence=0.7,
                warnings=["stub transcription — replace with MT4 v5 wrapper"],
            ),
        )
