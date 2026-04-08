"""Transcription stage — Basic Pitch baseline.

Wraps Spotify's `basic-pitch`_ polyphonic pitch-tracker into the async
pipeline. Basic Pitch is a lightweight CNN that consumes arbitrary mixed
audio and emits polyphonic note events with per-note amplitudes. It
produces a single un-instrumented pitch stream — we collapse the whole
prediction into one ``PIANO`` track, which is the right shape for a
piano-reduction pipeline anyway.

Backend selection is left to basic-pitch's auto-pick order via
``ICASSP_2022_MODEL_PATH``: on Darwin this resolves to the CoreML
model (fastest on Apple Silicon), on Linux CI it falls through to
ONNX/TFLite. The model is cached at module scope so the runtime only
loads once per process.

If basic-pitch isn't installed, or inference fails on a specific audio
file, the service falls back to a tiny stub ``TranscriptionResult`` so
the rest of the pipeline can still be exercised end-to-end.

.. _basic-pitch: https://github.com/spotify/basic-pitch
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from backend.config import settings
from backend.contracts import (
    SCHEMA_VERSION,
    HarmonicAnalysis,
    InputBundle,
    InstrumentRole,
    MidiTrack,
    Note,
    QualitySignal,
    TempoMapEntry,
    TranscriptionResult,
)
from backend.services.audio_timing import tempo_map_from_audio_path

log = logging.getLogger(__name__)


# Cached Basic Pitch model — building the inference session (CoreML on
# Darwin, ONNX/TFLite elsewhere) costs ~1s, so we load it once per process.
# Held as Any to avoid importing basic_pitch at module import time
# (optional dep, stub path needs to work without it).
_BP_MODEL: Any = None


def _pretty_midi_to_transcription_result(
    pm: Any,
    note_events: list[tuple[float, float, int, float, Any]],
    model_output: dict[str, Any],
    default_bpm: float = 120.0,
    *,
    tempo_map_override: list[TempoMapEntry] | None = None,
) -> TranscriptionResult:
    """Convert Basic Pitch's output into our pydantic TranscriptionResult.

    Basic Pitch emits a single polyphonic pitch stream (no instrument
    separation), so we collapse everything into one ``PIANO`` MidiTrack.
    Per-note confidence comes straight from the model's sigmoid output
    (note_events[i][3] is the amplitude at the onset frame, which is the
    same scalar basic_pitch uses to derive the MIDI velocity).

    If ``tempo_map_override`` is provided (e.g. from waveform beat
    tracking), it replaces the single-anchor map we'd otherwise build
    from ``pm.estimate_tempo`` so arrange's ``sec_to_beat`` aligns
    quantization to the real pulse of the recording.
    """
    import numpy as np  # noqa: PLC0415 — heavy/optional dep

    contract_notes: list[Note] = []
    amplitudes: list[float] = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for n in instrument.notes:
            contract_notes.append(
                Note(
                    pitch=int(n.pitch),
                    onset_sec=float(n.start),
                    offset_sec=float(n.end),
                    velocity=int(n.velocity),
                )
            )
    # note_events is the source of truth for per-note confidence; amplitudes
    # are sigmoid probabilities in [0, 1] from model_output["note"] at each
    # detected onset frame.
    for _start, _end, _pitch, amplitude, _bends in note_events:
        amplitudes.append(float(amplitude))

    contract_notes.sort(key=lambda n: (n.onset_sec, n.pitch))

    # Overall confidence — mean of per-note amplitudes, scaled and clamped.
    # Fall back to model_output["note"] mean if note_events is empty.
    if amplitudes:
        per_note_conf = float(np.mean(amplitudes))
    else:
        note_grid = model_output.get("note")
        per_note_conf = float(np.mean(note_grid)) if note_grid is not None else 0.3
    track_conf = round(min(max(per_note_conf, 0.1), 1.0), 2)

    midi_tracks: list[MidiTrack] = []
    if contract_notes:
        midi_tracks.append(
            MidiTrack(
                notes=contract_notes,
                instrument=InstrumentRole.PIANO,
                program=0,
                confidence=track_conf,
            )
        )

    # Tempo map — prefer the waveform-derived beat grid when available.
    # Basic Pitch itself does not estimate tempo, so without the override
    # we fall back to pretty_midi's estimate (single global BPM).
    if tempo_map_override:
        tempo_map = tempo_map_override
    else:
        bpm = default_bpm
        try:
            estimated = float(pm.estimate_tempo())
            if 40.0 <= estimated <= 240.0:
                bpm = estimated
        except Exception:  # noqa: BLE001 — estimate_tempo can raise on sparse input
            pass
        tempo_map = [TempoMapEntry(time_sec=0.0, beat=0.0, bpm=bpm)]

    analysis = HarmonicAnalysis(
        key="C:major",
        time_signature=(4, 4),
        tempo_map=tempo_map,
        chords=[],
        sections=[],
    )

    total_notes = len(contract_notes)
    warnings: list[str] = [
        "Basic Pitch baseline (polyphonic pitch tracker, no instrument separation)"
    ]
    if tempo_map_override:
        warnings.append("tempo_map from audio beat tracking (librosa)")
    if total_notes < 20:
        warnings.append(f"Low note count ({total_notes}) — possible quality issue")
    quality = QualitySignal(
        overall_confidence=track_conf if midi_tracks else 0.1,
        warnings=warnings,
    )

    return TranscriptionResult(
        schema_version=SCHEMA_VERSION,
        midi_tracks=midi_tracks,
        analysis=analysis,
        quality=quality,
    )


def _stub_result(reason: str) -> TranscriptionResult:
    """Tiny shape-correct fallback so downstream stages still run."""
    return TranscriptionResult(
        schema_version=SCHEMA_VERSION,
        midi_tracks=[
            MidiTrack(
                notes=[
                    Note(pitch=60, onset_sec=0.0, offset_sec=0.5, velocity=80),
                    Note(pitch=64, onset_sec=0.5, offset_sec=1.0, velocity=80),
                    Note(pitch=67, onset_sec=1.0, offset_sec=1.5, velocity=80),
                    Note(pitch=72, onset_sec=1.5, offset_sec=2.0, velocity=80),
                ],
                instrument=InstrumentRole.MELODY,
                program=None,
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
            overall_confidence=0.3,
            warnings=[f"Basic Pitch fallback stub: {reason}"],
        ),
    )


def _load_basic_pitch_model() -> Any:
    """Lazy-load the Basic Pitch model. Cached for the process lifetime."""
    global _BP_MODEL
    if _BP_MODEL is not None:
        return _BP_MODEL

    from basic_pitch import ICASSP_2022_MODEL_PATH  # noqa: PLC0415
    from basic_pitch.inference import Model  # noqa: PLC0415

    log.info("Loading Basic Pitch model from %s", ICASSP_2022_MODEL_PATH)
    _BP_MODEL = Model(ICASSP_2022_MODEL_PATH)
    return _BP_MODEL


def _audio_path_from_uri(uri: str) -> Path:
    """Resolve a Remote*File URI to a real path on disk.

    Today we only handle file:// URIs (LocalBlobStore). When the S3 store
    lands, this should download to a temp file instead.
    """
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"TranscribeService can only read file:// URIs, got {uri!r}")
    return Path(parsed.path)


def _run_basic_pitch_sync(audio_path: Path) -> TranscriptionResult:
    """Synchronous Basic Pitch inference. Run inside asyncio.to_thread."""
    model = _load_basic_pitch_model()
    from basic_pitch.inference import predict  # noqa: PLC0415

    model_output, midi_data, note_events = predict(
        str(audio_path),
        model_or_model_path=model,
        onset_threshold=settings.basic_pitch_onset_threshold,
        frame_threshold=settings.basic_pitch_frame_threshold,
        minimum_note_length=settings.basic_pitch_minimum_note_length_ms,
    )
    # Waveform-derived tempo_map (best-effort; None on failure).
    audio_tempo_map = tempo_map_from_audio_path(audio_path)
    return _pretty_midi_to_transcription_result(
        midi_data,
        note_events,
        model_output,
        tempo_map_override=audio_tempo_map,
    )


class TranscribeService:
    name = "transcribe"

    async def run(self, payload: InputBundle) -> TranscriptionResult:
        if payload.audio is None:
            return _stub_result("no audio in InputBundle")

        # Resolve the audio URI to a local path. For file:// URIs we can read
        # directly; otherwise we'd need to stage to a temp file via the blob
        # store. The blob store import is local to keep the stub path light.
        try:
            audio_path = _audio_path_from_uri(payload.audio.uri)
        except ValueError as exc:
            # Non-file URI: stage via the blob store into a temp file. The
            # blob store is imported lazily here because backend.api.deps
            # imports this module — pulling it in at module load would create
            # a cycle.
            from backend.api.deps import get_blob_store  # noqa: PLC0415
            try:
                blob = get_blob_store()
                data = blob.get_bytes(payload.audio.uri)
            except Exception as fetch_exc:  # noqa: BLE001
                log.warning("Could not fetch audio for Basic Pitch: %s", fetch_exc)
                return _stub_result(f"audio fetch failed: {fetch_exc}")
            suffix = f".{payload.audio.format}"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp.write(data)
            tmp.close()
            audio_path = Path(tmp.name)
            log.debug(
                "Staged %s → %s for Basic Pitch (%s)",
                payload.audio.uri, audio_path, exc,
            )

        if not audio_path.is_file():
            return _stub_result(f"audio file missing: {audio_path}")

        try:
            return await asyncio.to_thread(_run_basic_pitch_sync, audio_path)
        except ImportError as exc:
            log.warning("Basic Pitch deps unavailable (%s) — using stub", exc)
            return _stub_result(f"missing dependency: {exc}")
        except Exception as exc:  # noqa: BLE001 — boundary; we don't want one bad audio file to crash the worker
            log.exception("Basic Pitch inference failed for %s", audio_path)
            return _stub_result(f"inference failed: {exc}")
