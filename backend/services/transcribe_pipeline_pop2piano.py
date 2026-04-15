"""Pop2Piano transcription pipeline — audio → piano MIDI → TranscriptionResult.

Runs Pop2Piano to get a single piano MIDI stream, then feeds the result
through the same post-processing chain as the single-mix Basic Pitch
path: melody/bass extraction (Viterbi on note events since there's no
contour matrix), onset/duration refinement, key/chord/tempo estimation
from the original audio waveform.

This module is the Pop2Piano counterpart to ``transcribe_pipeline_single``
(no-Demucs Basic Pitch) and ``transcribe_pipeline_stems`` (Demucs+BP).

Performance
-----------
Audio is loaded from disk **once** at sr=22050 and the waveform is shared
across all post-processing consumers (onset refinement, duration
refinement, tempo map, key/meter estimation, chord recognition).
Audio-only analysis steps (tempo, key/meter, chords) run in parallel
with note-dependent steps (melody/bass extraction, onset/duration
refinement) via a :class:`~concurrent.futures.ThreadPoolExecutor`.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from backend.config import settings
from backend.contracts import InstrumentRole, RealtimeChordEvent, TranscriptionResult
from backend.services import transcribe_audio as _audio_mod
from backend.services import transcribe_midi as _midi_mod
from backend.services import transcribe_result as _result_mod
from backend.services.audio_timing import tempo_map_from_audio_path
from backend.services.bass_extraction import BassExtractionStats, extract_bass
from backend.services.chord_recognition import ChordRecognitionStats, recognize_chords
from backend.services.duration_refine import DurationRefineStats, refine_durations
from backend.services.key_estimation import refine_key_with_chords
from backend.services.melody_extraction import MelodyExtractionStats, extract_melody
from backend.services.onset_refine import OnsetRefineStats, refine_onsets
from backend.services.transcribe_pop2piano import run_pop2piano
from backend.services.transcription_cleanup import NoteEvent

log = logging.getLogger(__name__)


def _run_with_pop2piano(
    audio_path: Path,
) -> tuple[TranscriptionResult, bytes | None]:
    """Pop2Piano pipeline — one transformer pass + shared post-processing.

    The structure mirrors ``_run_without_stems`` in
    ``transcribe_pipeline_single.py`` closely: Pop2Piano replaces the
    Basic Pitch inference, then the same melody/bass extraction, onset
    refinement, duration refinement, tempo/key/chord analysis runs on
    the result + original audio waveform.
    """
    events, pm, pop2piano_stats = run_pop2piano(audio_path)

    # ── Load audio once for all post-processing ──────────────────────
    # Every downstream consumer (onset refine, duration refine, tempo
    # map, key/meter, chords) wants mono 22050 Hz.  Loading once here
    # eliminates 5+ redundant librosa.load() calls.
    preloaded_audio: tuple | None = None
    try:
        import librosa  # noqa: PLC0415
        y, actual_sr = librosa.load(str(audio_path), sr=22050, mono=True)
        preloaded_audio = (y, actual_sr)
    except Exception as exc:  # noqa: BLE001 — best-effort; consumers fall back to loading themselves
        log.debug("pre-load for post-processing failed: %s", exc)

    # Pop2Piano produces a single piano stream — no contour matrix
    # for Viterbi, so melody/bass extraction works on note pitch ranges only.
    # We pass contour=None; the extractors handle this gracefully.
    contour = None
    remaining: list[NoteEvent] = list(events)
    melody_events: list[NoteEvent] = []
    bass_events: list[NoteEvent] = []

    melody_stats: MelodyExtractionStats | None = None
    bass_stats: BassExtractionStats | None = None

    audio_duration_sec = _audio_mod._audio_duration_sec(audio_path)

    if settings.melody_extraction_enabled:
        melody_events, remaining, melody_stats = extract_melody(
            contour,
            remaining,
            melody_low_midi=settings.melody_low_midi,
            melody_high_midi=settings.melody_high_midi,
            voicing_floor=settings.melody_voicing_floor,
            transition_weight=settings.melody_transition_weight,
            max_transition_bins=settings.melody_max_transition_bins,
            match_fraction=settings.melody_match_fraction,
            backfill_enabled=settings.melody_backfill_enabled,
            backfill_min_duration_sec=settings.melody_backfill_min_duration_sec,
            backfill_overlap_fraction=settings.melody_backfill_overlap_fraction,
            backfill_min_amp=settings.melody_backfill_min_amp,
            backfill_max_amp=settings.melody_backfill_max_amp,
            max_time_sec=audio_duration_sec,
        )

    if settings.bass_extraction_enabled:
        bass_events, remaining, bass_stats = extract_bass(
            contour,
            remaining,
            bass_low_midi=settings.bass_low_midi,
            bass_high_midi=settings.bass_high_midi,
            voicing_floor=settings.bass_voicing_floor,
            transition_weight=settings.bass_transition_weight,
            max_transition_bins=settings.bass_max_transition_bins,
            match_fraction=settings.bass_match_fraction,
        )

    melody_skipped = melody_stats is None or melody_stats.skipped
    bass_skipped = bass_stats is None or bass_stats.skipped

    events_by_role: dict[InstrumentRole, list[NoteEvent]]
    if melody_skipped and bass_skipped:
        events_by_role = {InstrumentRole.PIANO: events}
    else:
        events_by_role = {}
        if melody_events:
            events_by_role[InstrumentRole.MELODY] = melody_events
        if bass_events:
            events_by_role[InstrumentRole.BASS] = bass_events
        if remaining:
            events_by_role[InstrumentRole.CHORDS] = remaining
        if not events_by_role:
            events_by_role = {InstrumentRole.PIANO: events}

    # ── Parallel post-processing ─────────────────────────────────────
    # Audio-analysis steps (tempo, key/meter, chords) are independent
    # of note events.  Run them concurrently with onset/duration
    # refinement to cut wall-clock time.  All share the pre-loaded
    # waveform so there is no redundant disk I/O.
    #
    # Results are collected from futures after the note-processing
    # steps complete (melody/bass split → onset → duration).

    # Kick off audio-analysis work in background threads.
    audio_tempo_map = None
    key_label = "C:major"
    time_signature: tuple[int, int] = (4, 4)
    key_stats = None
    meter_stats = None
    chord_labels: list[RealtimeChordEvent] = []
    chord_stats: ChordRecognitionStats | None = None

    def _run_tempo() -> list | None:
        return tempo_map_from_audio_path(audio_path, preloaded_audio=preloaded_audio)

    def _run_key_meter() -> tuple:
        return _audio_mod._maybe_analyze_key_and_meter(
            audio_path, preloaded_audio=preloaded_audio,
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
                preloaded_audio=preloaded_audio,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("chord recognition raised: %s", exc)
            cs = ChordRecognitionStats(skipped=True)
            cs.warnings.append(f"chord recognition failed: {exc}")
            return [], cs

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="p2p-post") as pool:
        fut_tempo = pool.submit(_run_tempo)
        fut_key = pool.submit(_run_key_meter)

        # ── Note-dependent refinement (runs on main thread) ──────────
        # Onset refinement
        onset_refine_stats: OnsetRefineStats | None = None
        if settings.onset_refine_enabled:
            all_events: list[NoteEvent] = []
            for evts in events_by_role.values():
                all_events.extend(evts)
            try:
                refined_all, onset_refine_stats = refine_onsets(
                    all_events,
                    audio_path,
                    sr=22050,
                    hop_length=settings.onset_refine_hop_length,
                    max_shift_sec=settings.onset_refine_max_shift_sec,
                    preloaded_audio=preloaded_audio,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("onset refinement raised: %s", exc)
                onset_refine_stats = OnsetRefineStats(
                    total_notes=len(all_events), skipped=True,
                )
                onset_refine_stats.warnings.append(f"onset-refine: exception: {exc}")
            else:
                offset = 0
                for role in list(events_by_role.keys()):
                    role_len = len(events_by_role[role])
                    events_by_role[role] = refined_all[offset : offset + role_len]
                    offset += role_len

        # Duration refinement
        duration_refine_stats: DurationRefineStats | None = None
        if settings.duration_refine_enabled:
            all_events_dur: list[NoteEvent] = []
            for evts in events_by_role.values():
                all_events_dur.extend(evts)
            try:
                refined_dur, duration_refine_stats = refine_durations(
                    all_events_dur,
                    audio_path,
                    sr=22050,
                    hop_length=settings.duration_refine_hop_length,
                    floor_ratio=settings.duration_refine_floor_ratio,
                    tail_sec=settings.duration_refine_tail_sec,
                    min_duration_sec=settings.duration_refine_min_duration_sec,
                    preloaded_audio=preloaded_audio,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("duration refinement raised: %s", exc)
                duration_refine_stats = DurationRefineStats(total_notes=len(all_events_dur))
            else:
                offset = 0
                for role in list(events_by_role.keys()):
                    role_len = len(events_by_role[role])
                    events_by_role[role] = refined_dur[offset : offset + role_len]
                    offset += role_len

        # ── Collect audio-analysis results ───────────────────────────
        audio_tempo_map = fut_tempo.result()
        key_label, time_signature, key_stats, meter_stats = fut_key.result()

        # Chord recognition needs key_label from the key estimator, so
        # it must be submitted after fut_key completes.
        fut_chords = pool.submit(_run_chords, key_label)
        chord_labels, chord_stats = fut_chords.result()

    # Cross-validate key estimate against detected chords
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

    # Build blob MIDI with the audio-derived tempo
    initial_bpm = float(audio_tempo_map[0].bpm) if audio_tempo_map else 120.0
    blob_midi = _midi_mod._rebuild_blob_midi(events, initial_bpm=initial_bpm)
    if blob_midi is None:
        blob_midi = pm

    # Build a minimal model_output dict — _pretty_midi_to_transcription_result
    # uses model_output["note"] only as a confidence fallback when events are
    # empty, so an empty dict is fine for the normal path.
    model_output: dict = {}

    result = _result_mod._pretty_midi_to_transcription_result(
        pm,
        events_by_role,
        model_output,
        tempo_map_override=audio_tempo_map,
        key_label=key_label,
        time_signature=time_signature,
        key_stats=key_stats,
        meter_stats=meter_stats,
        melody_stats=melody_stats,
        bass_stats=bass_stats,
        chord_stats=chord_stats,
        chord_labels=chord_labels,
        onset_refine_stats=onset_refine_stats,
        duration_refine_stats=duration_refine_stats,
    )

    # Inject Pop2Piano-specific warnings into quality signal
    pop2piano_warnings = pop2piano_stats.as_warnings()
    existing_warnings = list(result.quality.warnings)
    # Replace the "Basic Pitch baseline" banner with a Pop2Piano one
    existing_warnings = [
        w for w in existing_warnings
        if "Basic Pitch baseline" not in w
    ]
    pop2piano_banner = (
        f"Pop2Piano transcription (model={pop2piano_stats.model_id}, "
        f"notes={pop2piano_stats.note_count}, "
        f"audio={pop2piano_stats.audio_duration_sec:.1f}s)"
    )
    result = result.model_copy(
        update={
            "quality": result.quality.model_copy(
                update={"warnings": [pop2piano_banner] + pop2piano_warnings + existing_warnings}
            )
        }
    )

    midi_bytes = _midi_mod._serialize_pretty_midi(blob_midi)
    return result, midi_bytes
