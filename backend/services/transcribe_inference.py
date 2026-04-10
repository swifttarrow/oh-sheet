"""Basic Pitch model loading and single-pass inference wrapper.

Wraps Spotify's ``basic-pitch`` polyphonic pitch-tracker into a
reusable ``_basic_pitch_single_pass()`` function that handles model
caching, optional audio preprocessing, inference, and Phase-1 cleanup.
Extracted from ``transcribe.py``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.config import settings
from backend.services.audio_preprocess import (
    PreprocessStats,
    preprocess_audio_file,
)
from backend.services.transcription_cleanup import (
    AmplitudeEnvelope,
    CleanupStats,
    NoteEvent,
    cleanup_note_events,
)

log = logging.getLogger(__name__)


# Cached Basic Pitch model — building the inference session (CoreML on
# Darwin, ONNX/TFLite elsewhere) costs ~1s, so we load it once per process.
# Held as Any to avoid importing basic_pitch at module import time
# (optional dep, stub path needs to work without it).
_BP_MODEL: Any = None


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


@dataclass
class _BasicPitchPass:
    """Captured outputs of one Basic Pitch inference run on a single audio file.

    Returned by :func:`_basic_pitch_single_pass` so the orchestrator can
    either use one pass directly (no-Demucs path) or stitch several
    passes together (one per stem).
    """
    cleaned_events: list[NoteEvent]
    model_output: dict[str, Any]
    midi_data: Any  # pretty_midi.PrettyMIDI rebuilt from cleaned_events
    preprocess_stats: PreprocessStats | None
    cleanup_stats: CleanupStats


def _basic_pitch_single_pass(
    audio_path: Path,
    *,
    keep_model_output: bool = True,
    onset_threshold: float | None = None,
    frame_threshold: float | None = None,
    cleanup_octave_amp_ratio: float | None = None,
    cleanup_ghost_max_duration_sec: float | None = None,
    amplitude_envelope: AmplitudeEnvelope | None = None,
) -> _BasicPitchPass:
    """Run preprocess -> Basic Pitch -> cleanup for one audio file.

    Factored out so the Demucs path can call it once per stem
    (vocals / bass / other) without duplicating boilerplate. The
    returned :class:`_BasicPitchPass` carries everything downstream
    consumers might want.

    ``keep_model_output`` defaults to True for the single-mix path,
    where the downstream Viterbi melody/bass extractors read
    ``model_output["contour"]``. The stems path passes False so the
    contour tensor (tens of MB per stem) can be garbage-collected as
    soon as Basic Pitch's note events are extracted.
    """
    model = _load_basic_pitch_model()
    from basic_pitch.inference import predict  # noqa: PLC0415
    from basic_pitch.note_creation import note_events_to_midi  # noqa: PLC0415

    # Audio pre-processing (feature-flagged). Any failure falls back
    # to the original path with stats.skipped=True, so the rest of the
    # pipeline runs unchanged. The preprocessed temp file is cleaned up
    # after predict() regardless of success.
    preprocess_stats: PreprocessStats | None = None
    inference_path = audio_path
    preprocessed_tempfile: Path | None = None
    if settings.audio_preprocess_enabled:
        inference_path, preprocess_stats = preprocess_audio_file(
            audio_path,
            hpss_enabled=settings.audio_preprocess_hpss_enabled,
            hpss_margin=settings.audio_preprocess_hpss_margin,
            normalize_enabled=settings.audio_preprocess_normalize_enabled,
            target_rms_dbfs=settings.audio_preprocess_target_rms_dbfs,
            peak_ceiling_dbfs=settings.audio_preprocess_peak_ceiling_dbfs,
        )
        if inference_path != audio_path:
            preprocessed_tempfile = inference_path

    try:
        model_output, midi_data, note_events = predict(
            str(inference_path),
            model_or_model_path=model,
            onset_threshold=onset_threshold if onset_threshold is not None else settings.basic_pitch_onset_threshold,
            frame_threshold=frame_threshold if frame_threshold is not None else settings.basic_pitch_frame_threshold,
            minimum_note_length=settings.basic_pitch_minimum_note_length_ms,
        )
    finally:
        if preprocessed_tempfile is not None:
            try:
                preprocessed_tempfile.unlink(missing_ok=True)
            except Exception as exc:  # noqa: BLE001 — best-effort cleanup
                log.warning(
                    "Failed to unlink preprocessed temp file %s: %s",
                    preprocessed_tempfile, exc,
                )

    # Phase 1 post-processing — merge fragmented sustains, drop octave
    # ghosts, quiet ghost-tail notes, and energy-gate long offsets. Rebuild
    # pretty_midi from the cleaned list so the blob-stored .mid matches
    # the contract.
    cleaned_events, cleanup_stats = cleanup_note_events(
        note_events,
        merge_gap_sec=settings.cleanup_merge_gap_sec,
        octave_amp_ratio=(
            cleanup_octave_amp_ratio
            if cleanup_octave_amp_ratio is not None
            else settings.cleanup_octave_amp_ratio
        ),
        octave_onset_tol_sec=settings.cleanup_octave_onset_tol_sec,
        ghost_max_duration_sec=(
            cleanup_ghost_max_duration_sec
            if cleanup_ghost_max_duration_sec is not None
            else settings.cleanup_ghost_max_duration_sec
        ),
        ghost_amp_median_scale=settings.cleanup_ghost_amp_median_scale,
        amplitude_envelope=amplitude_envelope,
        energy_gate_max_sustain_sec=settings.cleanup_energy_gate_max_sustain_sec,
        energy_gate_floor_ratio=settings.cleanup_energy_gate_floor_ratio,
        energy_gate_tail_sec=settings.cleanup_energy_gate_tail_sec,
        energy_gate_enabled=settings.cleanup_energy_gate_enabled,
    )
    if cleanup_stats.output_count != cleanup_stats.input_count:
        try:
            midi_data = note_events_to_midi(cleaned_events)
        except Exception as exc:  # noqa: BLE001 — never let cleanup sink the job
            log.warning("note_events_to_midi rebuild failed, using raw pm: %s", exc)
            cleaned_events = note_events  # fall back to raw
            cleanup_stats.warnings.append("cleanup: rebuild failed, using raw pm")

    # Drop the shared reference to ``model_output`` before the
    # function returns when the caller doesn't need it.
    if not keep_model_output:
        model_output.clear()
    return _BasicPitchPass(
        cleaned_events=cleaned_events,
        model_output=model_output,
        midi_data=midi_data,
        preprocess_stats=preprocess_stats,
        cleanup_stats=cleanup_stats,
    )
