"""Kong-driven transcription pipeline (piano-stem AMT with sustain pedal).

Phase 6 transcription path. Routes a piano-ish *summed* stem
(``bass + other`` from Demucs) into ByteDance Kong's CRNN-Regress
piano transcription model. Kong is the only mature, pip-installable,
commercially-licensed transcriber that emits sustain-pedal CC64
events — the difference between a "blizzard of staccato eighths" and
a readable score on piano-cover material.

Routing policy (decided in ``transcribe.py``):

  * Pre-separated stems must exist (Phase 5 Demucs ran).
  * Vocal energy < threshold OR user_hint == "piano".

This module mirrors ``transcribe_pipeline_stems``'s post-processing
chain so the rest of the pipeline (chord recognition, key/meter,
tempo map, downbeats) behaves identically — only the per-note
inference and pedal output differ.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from backend.config import settings
from backend.contracts import (
    InstrumentRole,
    RealtimeChordEvent,
    RealtimePedalEvent,
    TranscriptionResult,
)
from backend.services import transcribe_audio as _audio_mod
from backend.services import transcribe_midi as _midi_mod
from backend.services import transcribe_result as _result_mod
from backend.services.audio_timing import tempo_map_and_downbeats_from_audio_path
from backend.services.chord_recognition import (
    ChordRecognitionStats,
    recognize_chords,
)
from backend.services.key_estimation import refine_key_with_chords
from backend.services.stem_separation import (
    SeparatedStems,
    StemSeparationStats,
)
from backend.services.transcribe_kong import run_kong
from backend.services.transcription_cleanup import NoteEvent

log = logging.getLogger(__name__)


def _sum_piano_stems(stems: SeparatedStems) -> Path | None:
    """Sum Demucs ``bass + other`` into a piano-ish mono WAV.

    Returns the new tempfile path, or ``None`` when neither stem is
    available. The caller owns cleanup of the returned file (best-effort
    delete after Kong is done with it). When only one of the two stems
    exists we use it directly without re-writing — Kong's loader
    handles a single WAV the same way.
    """
    candidates = [p for p in (stems.bass, stems.other) if p is not None]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    try:
        import numpy as np  # noqa: PLC0415
        import soundfile as sf  # noqa: PLC0415
    except ImportError:
        # soundfile is a librosa transitive dep — should always be
        # available in environments that have the basic-pitch extra.
        # Fall back to the louder of the two stems instead of summing.
        log.warning("kong: soundfile unavailable; falling back to single stem")
        return candidates[0]

    wavs: list[tuple[Any, int]] = []
    for path in candidates:
        try:
            data, sr = sf.read(str(path), always_2d=False)
        except Exception as exc:  # noqa: BLE001
            log.warning("kong: failed to read stem %s: %s", path, exc)
            continue
        if data.ndim > 1:
            data = data.mean(axis=1)
        wavs.append((data, sr))

    if not wavs:
        return None
    if len(wavs) == 1:
        return candidates[0]

    target_sr = wavs[0][1]
    if any(sr != target_sr for _, sr in wavs):
        # Mismatched stem sample rates would require resampling to
        # sum cleanly — skip the sum and use the first stem rather
        # than pull in librosa's resampler for a fast-path.
        log.warning("kong: stem sample rates differ; using first stem only")
        return candidates[0]

    min_len = min(len(d) for d, _ in wavs)
    summed = np.zeros(min_len, dtype=np.float32)
    for data, _ in wavs:
        summed += data[:min_len].astype(np.float32)
    summed /= float(len(wavs))

    tmp = tempfile.NamedTemporaryFile(
        suffix=".wav", prefix="kong_piano_", delete=False,
    )
    tmp.close()
    sf.write(tmp.name, summed, target_sr)
    return Path(tmp.name)


def _run_with_kong(
    audio_path: Path,
    stems: SeparatedStems,
    stem_stats: StemSeparationStats,
) -> tuple[TranscriptionResult, bytes | None, list[RealtimePedalEvent]]:
    """Kong pipeline — one inference pass on summed piano-ish stems.

    Kong emits a single piano stream (no melody / bass split), so all
    transcribed notes land on a single ``PIANO`` track. Pedal events
    come out as ``RealtimePedalEvent`` instances and are returned
    alongside the result so the caller (``transcribe.py``) can attach
    them to ``TranscriptionResult.pedal_events`` after stem-URI
    enrichment.

    Mirrors the post-processing chain from
    :func:`transcribe_pipeline_stems._run_with_stems`: tempo map +
    downbeats from the drums stem (or mix), key/meter from the mix,
    chord recognition from the ``other`` stem (or mix). The summed
    piano stem is fed only to Kong — chord/key/tempo analysis still
    benefits from access to the broader spectrum.
    """
    piano_audio = _sum_piano_stems(stems)
    cleanup_piano: Path | None = None
    # Track whether we own the summed file (delete on exit) vs. it's
    # an alias to a stem path (don't touch — owned by SeparatedStems).
    if piano_audio is not None and piano_audio not in (stems.bass, stems.other):
        cleanup_piano = piano_audio

    if piano_audio is None:
        log.warning("kong: no bass/other stems available, cannot run")
        # Caller will fall back to the Basic Pitch stems pipeline.
        raise RuntimeError("kong: no piano-ish stems available")

    try:
        note_events, pedal_events, kong_stats = run_kong(piano_audio)
    finally:
        if cleanup_piano is not None:
            try:
                cleanup_piano.unlink(missing_ok=True)
            except OSError:
                log.debug("failed to clean up kong piano tempfile %s", cleanup_piano)

    if not note_events:
        # Empty Kong output — push the empty-pedal warning into stem_stats
        # and fail loudly so the dispatcher knows to fall through.
        stem_stats.warnings.append("kong: empty notes; falling back to BP stems")
        raise RuntimeError("kong: empty note output")

    events_by_role: dict[InstrumentRole, list[NoteEvent]] = {
        InstrumentRole.PIANO: note_events,
    }

    # ── Pre-load mix once for mix-side audio analyzers ─────────────────
    def _preload(path: Path | None) -> tuple | None:
        if path is None:
            return None
        try:
            import librosa as _lib  # noqa: PLC0415
            y, sr = _lib.load(str(path), sr=22050, mono=True)
            return (y, sr)
        except Exception as exc:  # noqa: BLE001
            log.debug("kong: pre-load failed for %s: %s", path, exc)
            return None

    mix_audio = _preload(audio_path)

    # Tempo map + downbeats — drums stem when available, mix otherwise.
    tempo_src: Path = audio_path
    tempo_preloaded = mix_audio
    if settings.demucs_use_drums_for_beats and stems.drums is not None:
        tempo_src = stems.drums
        tempo_preloaded = _preload(stems.drums)
    audio_tempo_map: list | None = None
    audio_downbeats: list[float] = []
    tempo_result = tempo_map_and_downbeats_from_audio_path(
        tempo_src, preloaded_audio=tempo_preloaded,
    )
    if tempo_result is None and tempo_src != audio_path:
        log.debug("kong: drums-stem beat tracking returned None, retrying with mix")
        tempo_result = tempo_map_and_downbeats_from_audio_path(
            audio_path, preloaded_audio=mix_audio,
        )
    if tempo_result is not None:
        audio_tempo_map, audio_downbeats = tempo_result

    # Key + meter on the mix.
    key_label, time_signature, key_stats, meter_stats = (
        _audio_mod._maybe_analyze_key_and_meter(
            audio_path, preloaded_audio=mix_audio,
        )
    )

    # Chord recognition — prefer the "other" stem when enabled.
    chord_labels: list[RealtimeChordEvent] = []
    chord_stats: ChordRecognitionStats | None = None
    if settings.chord_recognition_enabled:
        chord_src: Path = audio_path
        chord_preloaded = mix_audio
        if settings.demucs_use_other_for_chords and stems.other is not None:
            chord_src = stems.other
            chord_preloaded = _preload(stems.other)
        try:
            chord_labels, chord_stats = recognize_chords(
                chord_src,
                min_score=settings.chord_min_template_score,
                hpss_margin=settings.chord_hpss_margin,
                seventh_enabled=settings.chord_seventh_templates_enabled,
                hmm_enabled=settings.chord_hmm_enabled,
                hmm_self_transition=settings.chord_hmm_self_transition,
                hmm_temperature=settings.chord_hmm_temperature,
                key_label=key_label,
                preloaded_audio=chord_preloaded,
            )
            if chord_stats.skipped and chord_src != audio_path:
                chord_labels, chord_stats = recognize_chords(
                    audio_path,
                    min_score=settings.chord_min_template_score,
                    hpss_margin=settings.chord_hpss_margin,
                    seventh_enabled=settings.chord_seventh_templates_enabled,
                    hmm_enabled=settings.chord_hmm_enabled,
                    hmm_self_transition=settings.chord_hmm_self_transition,
                    hmm_temperature=settings.chord_hmm_temperature,
                    key_label=key_label,
                    preloaded_audio=mix_audio,
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("kong: chord recognition raised: %s", exc)
            chord_labels = []
            chord_stats = ChordRecognitionStats(skipped=True)
            chord_stats.warnings.append(f"chord recognition failed: {exc}")

    if (
        settings.key_chord_validation_enabled
        and key_stats is not None
        and not key_stats.skipped
        and chord_labels
    ):
        key_label, key_stats = refine_key_with_chords(
            key_label, key_stats.confidence,
            key_stats.runner_up_label, key_stats.runner_up_confidence,
            chord_labels,
            diatonic_threshold=settings.key_chord_diatonic_threshold,
            flip_margin=settings.key_chord_flip_margin,
        )

    initial_bpm = (
        float(audio_tempo_map[0].bpm)
        if audio_tempo_map
        else 120.0
    )
    combined_midi = _midi_mod._combined_midi_from_events(
        events_by_role, None, initial_bpm=initial_bpm,
    )

    stem_stats.warnings.extend(kong_stats.as_warnings())

    result = _result_mod._pretty_midi_to_transcription_result(
        combined_midi,
        events_by_role,
        {},
        tempo_map_override=audio_tempo_map,
        downbeats_override=audio_downbeats,
        key_label=key_label,
        time_signature=time_signature,
        key_stats=key_stats,
        meter_stats=meter_stats,
        chord_stats=chord_stats,
        chord_labels=chord_labels,
        stem_stats=stem_stats,
    )
    midi_bytes = (
        _midi_mod._serialize_pretty_midi(combined_midi) if combined_midi else None
    )
    return result, midi_bytes, pedal_events


def _vocal_energy(stem_path: Path | None) -> float:
    """Compute a normalized RMS energy estimate for the vocals stem.

    Returns ``0.0`` when the stem is missing or the read fails — caller
    treats missing/zero as "below threshold" (favors Kong on
    instrumental material). The value is the mean absolute amplitude
    of the waveform, which lands roughly in [0, 0.3] for typical pop;
    we expose it as a fraction of the per-sample peak so the threshold
    is intuitive to set in config.
    """
    if stem_path is None or not stem_path.is_file():
        return 0.0
    try:
        import numpy as np  # noqa: PLC0415
        import soundfile as sf  # noqa: PLC0415
        data, _ = sf.read(str(stem_path), always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if data.size == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))
        return min(1.0, rms)
    except Exception as exc:  # noqa: BLE001
        log.debug("kong: vocal energy read failed for %s: %s", stem_path, exc)
        return 0.0


def should_route_to_kong(
    stems: SeparatedStems | None,
    *,
    user_hint: str | None = None,
) -> bool:
    """Decide whether the Kong path should run for this job.

    Kong is gated on three conditions, all of which must hold:

      1. ``settings.kong_enabled`` is true (operator opt-in / kill switch).
      2. Pre-separated stems exist with at least one piano-ish source
         (``bass`` or ``other``). Kong is MAESTRO-overfit; raw audio
         hurts F1 by ~20pp on pitch-shifted / reverbed material per
         Edwards et al. 2024. Demucs gives us a clean enough piano
         stem to play to Kong's strengths.
      3. Either the user has hinted ``piano`` OR the vocals-stem RMS
         energy is below ``kong_vocal_energy_threshold`` — i.e. the
         input is piano-dominant. On vocal-heavy pop, BP on the
         ``other`` stem still wins.
    """
    if not settings.kong_enabled:
        return False
    if stems is None or (stems.bass is None and stems.other is None):
        return False
    if user_hint and user_hint.lower() == "piano":
        return True
    if settings.kong_user_hint_only:
        return False
    energy = _vocal_energy(stems.vocals)
    if energy < settings.kong_vocal_energy_threshold:
        log.info(
            "kong: routing chosen — vocal_energy=%.3f < threshold=%.3f",
            energy, settings.kong_vocal_energy_threshold,
        )
        return True
    log.info(
        "kong: routing skipped — vocal_energy=%.3f >= threshold=%.3f",
        energy, settings.kong_vocal_energy_threshold,
    )
    return False
