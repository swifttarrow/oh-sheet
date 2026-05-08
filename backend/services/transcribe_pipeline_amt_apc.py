"""AMT-APC piano-cover transcription pipeline (Phase 8).

Phase 8 cover-mode transcription path. Routes audio (or the Demucs
*instrumental* stem when separation has run) through AMT-APC to produce
a pianistic cover — idiomatic LH accompaniment patterns and melody
re-voicings — then attaches the same audio-side analysis (tempo, key,
chord recognition, downbeats) the other pipelines emit so the engraver
has the metadata it needs.

Routing policy (decided in ``transcribe.py``):

  * Bound to the ``pop_cover`` :class:`PipelineVariant`.
  * Runs on the summed instrumental stem (``bass + other`` from Demucs)
    when stems are present, falling back to the full mix when separation
    didn't run. AMT-APC was trained on full-mix audio so the fallback
    is acceptable; the stem path just biases the model away from
    melody-only output by suppressing vocals.

Unlike :mod:`transcribe_pipeline_pop2piano` and
:mod:`transcribe_pipeline_single`, this pipeline **deliberately skips**
melody/bass extraction. AMT-APC already emits an arrangement-ready
piano stream — splitting it into MELODY/BASS/CHORDS roles for a
downstream rules-based arranger would only fight the model. Cover-mode
output flows into a single :class:`InstrumentRole.PIANO` track and
the runner skips ``arrange``/``humanize`` entirely (see the
``pop_cover`` execution plan in :class:`PipelineConfig`).
"""
from __future__ import annotations

import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from backend.config import settings
from backend.contracts import InstrumentRole, RealtimeChordEvent, TranscriptionResult
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
from backend.services.transcribe_amt_apc import run_amt_apc
from backend.services.transcription_cleanup import NoteEvent

log = logging.getLogger(__name__)


def _sum_instrumental_stems(stems: SeparatedStems) -> tuple[Path | None, bool]:
    """Sum Demucs ``bass + other`` into a vocal-suppressed mono WAV.

    Returns ``(path, owned)``. ``owned`` is ``True`` when we created a
    new tempfile that the caller must clean up; ``False`` when the
    returned path aliases an existing stem (don't delete it — owned by
    :class:`SeparatedStems`).

    Returns ``(None, False)`` when neither stem is available.
    """
    candidates = [p for p in (stems.bass, stems.other) if p is not None]
    if not candidates:
        return None, False
    if len(candidates) == 1:
        return candidates[0], False

    try:
        import numpy as np  # noqa: PLC0415
        import soundfile as sf  # noqa: PLC0415
    except ImportError:
        log.warning(
            "amt_apc: soundfile unavailable; falling back to single stem",
        )
        return candidates[0], False

    wavs: list[tuple[Any, int]] = []
    for path in candidates:
        try:
            data, sr = sf.read(str(path), always_2d=False)
        except Exception as exc:  # noqa: BLE001
            log.warning("amt_apc: failed to read stem %s: %s", path, exc)
            continue
        if data.ndim > 1:
            data = data.mean(axis=1)
        wavs.append((data, sr))

    if not wavs:
        return None, False
    if len(wavs) == 1:
        return candidates[0], False

    target_sr = wavs[0][1]
    if any(sr != target_sr for _, sr in wavs):
        log.warning(
            "amt_apc: stem sample rates differ; using first stem only",
        )
        return candidates[0], False

    min_len = min(len(d) for d, _ in wavs)
    summed = np.zeros(min_len, dtype=np.float32)
    for data, _ in wavs:
        summed += data[:min_len].astype(np.float32)
    summed /= float(len(wavs))

    tmp = tempfile.NamedTemporaryFile(
        suffix=".wav", prefix="amt_apc_inst_", delete=False,
    )
    tmp.close()
    sf.write(tmp.name, summed, target_sr)
    return Path(tmp.name), True


def _run_with_amt_apc(
    audio_path: Path,
    stems: SeparatedStems | None,
    stem_stats: StemSeparationStats | None,
) -> tuple[TranscriptionResult, bytes | None]:
    """AMT-APC pipeline — one cover-mode inference + audio-side analysis.

    The model emits a single piano stream attributed to ``InstrumentRole.PIANO``;
    no melody/bass split runs because the cover model has already made
    those arrangement decisions internally. Tempo / key / chord
    analysis still runs on the *mix* (not the instrumental stem) so
    chord recognition gets the full harmonic context.

    Returns ``(TranscriptionResult, midi_bytes)``. The caller (the
    Phase 8 dispatch in ``transcribe.py``) wraps the return into the
    standard 3-tuple by appending ``[]`` for pedal events — AMT-APC
    does not model pedal.
    """
    # ── Pick the inference source ────────────────────────────────────
    inference_audio: Path = audio_path
    cleanup_owned: Path | None = None
    if stems is not None:
        summed, owned = _sum_instrumental_stems(stems)
        if summed is not None:
            inference_audio = summed
            if owned:
                cleanup_owned = summed

    try:
        note_events, pm, amt_stats = run_amt_apc(inference_audio)
    finally:
        if cleanup_owned is not None:
            try:
                cleanup_owned.unlink(missing_ok=True)
            except OSError:
                log.debug(
                    "failed to clean up amt_apc tempfile %s", cleanup_owned,
                )

    if not note_events:
        # Empty AMT-APC output — caller falls back to Kong / BP.
        if stem_stats is not None:
            stem_stats.warnings.append(
                "amt_apc: empty notes; falling back to faithful path",
            )
        raise RuntimeError("amt_apc: empty note output")

    # All AMT-APC notes land on a single PIANO track. The cover model's
    # output is arrangement-ready; we don't run melody/bass extraction.
    events_by_role: dict[InstrumentRole, list[NoteEvent]] = {
        InstrumentRole.PIANO: note_events,
    }

    # ── Pre-load mix once for mix-side audio analyzers ───────────────
    def _preload(path: Path | None) -> tuple | None:
        if path is None:
            return None
        try:
            import librosa as _lib  # noqa: PLC0415
            y, sr = _lib.load(str(path), sr=22050, mono=True)
            return (y, sr)
        except Exception as exc:  # noqa: BLE001
            log.debug("amt_apc: pre-load failed for %s: %s", path, exc)
            return None

    mix_audio = _preload(audio_path)

    # ── Audio-side analysis (tempo / key / chords) ───────────────────
    audio_tempo_map = None
    audio_downbeats: list[float] = []
    key_label = "C:major"
    time_signature: tuple[int, int] = (4, 4)
    key_stats = None
    meter_stats = None
    chord_labels: list[RealtimeChordEvent] = []
    chord_stats: ChordRecognitionStats | None = None

    def _run_tempo() -> tuple[list, list[float]] | None:
        # Prefer drums stem when available, fall back to mix.
        if (
            stems is not None
            and settings.demucs_use_drums_for_beats
            and stems.drums is not None
        ):
            tempo_src = stems.drums
            tempo_pre = _preload(stems.drums)
            r = tempo_map_and_downbeats_from_audio_path(
                tempo_src, preloaded_audio=tempo_pre,
            )
            if r is not None:
                return r
        return tempo_map_and_downbeats_from_audio_path(
            audio_path, preloaded_audio=mix_audio,
        )

    def _run_key_meter() -> tuple:
        return _audio_mod._maybe_analyze_key_and_meter(
            audio_path, preloaded_audio=mix_audio,
        )

    def _run_chords(kl: str) -> tuple[list[RealtimeChordEvent], ChordRecognitionStats | None]:
        if not settings.chord_recognition_enabled:
            return [], None
        try:
            return recognize_chords(
                audio_path,
                min_score=settings.chord_min_template_score,
                hpss_margin=settings.chord_hpss_margin,
                seventh_enabled=settings.chord_seventh_templates_enabled,
                hmm_enabled=settings.chord_hmm_enabled,
                hmm_self_transition=settings.chord_hmm_self_transition,
                hmm_temperature=settings.chord_hmm_temperature,
                key_label=kl,
                preloaded_audio=mix_audio,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("amt_apc: chord recognition raised: %s", exc)
            cs = ChordRecognitionStats(skipped=True)
            cs.warnings.append(f"chord recognition failed: {exc}")
            return [], cs

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="amt-apc-post") as pool:
        fut_tempo = pool.submit(_run_tempo)
        fut_key = pool.submit(_run_key_meter)
        tempo_result = fut_tempo.result()
        if tempo_result is not None:
            audio_tempo_map, audio_downbeats = tempo_result
        key_label, time_signature, key_stats, meter_stats = fut_key.result()
        fut_chords = pool.submit(_run_chords, key_label)
        chord_labels, chord_stats = fut_chords.result()

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
        float(audio_tempo_map[0].bpm) if audio_tempo_map else 120.0
    )
    blob_midi = _midi_mod._rebuild_blob_midi(note_events, initial_bpm=initial_bpm)
    if blob_midi is None:
        blob_midi = pm

    if stem_stats is not None:
        stem_stats.warnings.extend(amt_stats.as_warnings())

    result = _result_mod._pretty_midi_to_transcription_result(
        pm,
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

    # Replace the generic "Basic Pitch baseline" banner with an AMT-APC one
    existing_warnings = [
        w for w in result.quality.warnings
        if "Basic Pitch baseline" not in w
    ]
    amt_banner = (
        f"AMT-APC cover transcription (style={amt_stats.style}, "
        f"notes={amt_stats.note_count}, "
        f"audio={amt_stats.audio_duration_sec:.1f}s)"
    )
    result = result.model_copy(
        update={
            "quality": result.quality.model_copy(
                update={
                    "warnings": [amt_banner, *amt_stats.as_warnings(), *existing_warnings],
                },
            ),
        },
    )

    midi_bytes = _midi_mod._serialize_pretty_midi(blob_midi)
    return result, midi_bytes


def should_route_to_amt_apc(
    *,
    user_hint: str | None = None,
    variant: str | None = None,
) -> bool:
    """Decide whether the AMT-APC path should run for this job.

    AMT-APC is gated on either:

      1. The bundle's :class:`PipelineVariant` is ``pop_cover`` — the
         user explicitly opted into cover mode via the frontend toggle.
      2. ``user_hint == "cover"`` — programmatic override path (used
         in tests + future API extensions for callers that want cover
         output without flipping the variant).
      3. ``settings.amt_apc_enabled`` — operator-level kill switch.
         When False, the dispatcher skips AMT-APC even when the variant
         would otherwise select it (used to roll the model back from a
         deploy without re-pushing API consumers).
    """
    if not settings.amt_apc_enabled:
        return False
    if variant == "pop_cover":
        return True
    return bool(user_hint and user_hint.lower() == "cover")
