"""Single-mix transcription pipeline (no Demucs stems).

One Basic Pitch pass on the full mix, then Viterbi melody/bass split,
chord recognition, onset/duration refinement, and result assembly.
This is the legacy path and also the fallback when stem separation
fails. Extracted from ``transcribe.py``.
"""
from __future__ import annotations

import logging
from pathlib import Path

from backend.config import settings
from backend.contracts import InstrumentRole, RealtimeChordEvent, TranscriptionResult
from backend.services import transcribe_audio as _audio_mod
from backend.services import transcribe_inference as _bp_mod
from backend.services import transcribe_midi as _midi_mod
from backend.services import transcribe_result as _result_mod
from backend.services.audio_timing import tempo_map_from_audio_path
from backend.services.bass_extraction import (
    BassExtractionStats,
    extract_bass,
)
from backend.services.chord_recognition import (
    ChordRecognitionStats,
    recognize_chords,
)
from backend.services.duration_refine import (
    DurationRefineStats,
    refine_durations,
)
from backend.services.key_estimation import refine_key_with_chords
from backend.services.melody_extraction import (
    MelodyExtractionStats,
    extract_melody,
)
from backend.services.onset_refine import (
    OnsetRefineStats,
    refine_onsets,
)
from backend.services.stem_separation import StemSeparationStats
from backend.services.transcription_cleanup import (
    AmplitudeEnvelope,
    NoteEvent,
)

log = logging.getLogger(__name__)


def _run_without_stems(
    audio_path: Path,
    stem_stats: StemSeparationStats | None,
) -> tuple[TranscriptionResult, bytes | None]:
    """Legacy single-mix pipeline — one Basic Pitch pass + Viterbi splits.

    This is what ``_run_basic_pitch_sync`` falls back to when Demucs
    is disabled *or* stem separation failed. ``stem_stats`` may
    carry a "skipped: ..." message from a failed Demucs attempt;
    threading it through lets the QualitySignal explain why the
    per-stem code path didn't run.
    """
    # Compute amplitude envelope for energy gating — best-effort.
    envelope: AmplitudeEnvelope | None = None
    if settings.cleanup_energy_gate_enabled:
        try:
            envelope = _audio_mod._compute_amplitude_envelope(audio_path)
        except Exception as exc:  # noqa: BLE001
            log.debug("envelope computation failed for %s: %s", audio_path, exc)

    pass_result = _bp_mod._basic_pitch_single_pass(
        audio_path, amplitude_envelope=envelope,
    )
    cleaned_events = pass_result.cleaned_events
    model_output = pass_result.model_output
    midi_data = pass_result.midi_data

    # Audio duration is used to clamp the Viterbi melody back-fill so
    # synthesized notes don't extend past the real end of the audio
    # (see :func:`_backfill_missed_melody_notes`). Best-effort: ``None``
    # leaves back-fill unclamped, which matches the pre-fix behaviour.
    audio_duration_sec = _audio_mod._audio_duration_sec(audio_path)

    # Phase 2+3 post-processing — waveform-guided voice split via a
    # Viterbi path over ``model_output["contour"]``, then bass on the
    # low-register slice, then chord recognition from the source
    # waveform. Each phase is independently feature-flagged. Skipped
    # extractors leave their inputs unchanged so the chain degrades
    # gracefully to single-track PIANO when everything is off.
    contour = model_output.get("contour")
    melody_events: list[NoteEvent] = []
    bass_events: list[NoteEvent] = []
    remaining: list[NoteEvent] = list(cleaned_events)

    melody_stats: MelodyExtractionStats | None = None
    bass_stats: BassExtractionStats | None = None
    chord_stats: ChordRecognitionStats | None = None
    chord_labels: list[RealtimeChordEvent] = []

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
        # Legacy single-track fallback — both Phase 2 and Phase 3 voice
        # splits were disabled or failed. Merge everything into PIANO
        # so the arrange stage still gets the full pitch stream.
        events_by_role = {InstrumentRole.PIANO: cleaned_events}
    else:
        events_by_role = {}
        if melody_events:
            events_by_role[InstrumentRole.MELODY] = melody_events
        if bass_events:
            events_by_role[InstrumentRole.BASS] = bass_events
        if remaining:
            events_by_role[InstrumentRole.CHORDS] = remaining
        if not events_by_role:
            # Edge case: both extractors ran but all notes ended up
            # outside every band. Keep the raw stream under PIANO.
            events_by_role = {InstrumentRole.PIANO: cleaned_events}

    # Onset refinement — snap note onsets to spectral onset-strength peaks.
    # Runs after cleanup + voice split but before _event_to_note() conversion.
    # Each role's events are refined independently so the ODF peak lookup
    # uses a single cached computation per audio file.
    onset_refine_stats_single: OnsetRefineStats | None = None
    if settings.onset_refine_enabled:
        all_events: list[NoteEvent] = []
        for evts in events_by_role.values():
            all_events.extend(evts)
        try:
            refined_all, onset_refine_stats_single = refine_onsets(
                all_events,
                audio_path,
                sr=22050,
                hop_length=settings.onset_refine_hop_length,
                max_shift_sec=settings.onset_refine_max_shift_sec,
            )
        except Exception as exc:  # noqa: BLE001 — never let onset refine sink transcribe
            log.warning("onset refinement raised: %s", exc)
            onset_refine_stats_single = OnsetRefineStats(
                total_notes=len(all_events), skipped=True,
            )
            onset_refine_stats_single.warnings.append(f"onset-refine: exception: {exc}")
        else:
            # Scatter the refined events back into their per-role buckets.
            # The concatenation order is deterministic (dict insertion order)
            # so we can slice the flat refined list by the same lengths.
            offset = 0
            for role in list(events_by_role.keys()):
                role_len = len(events_by_role[role])
                events_by_role[role] = refined_all[offset : offset + role_len]
                offset += role_len

    # Duration refinement — trim note offsets using per-pitch CQT energy.
    duration_refine_stats_single: DurationRefineStats | None = None
    if settings.duration_refine_enabled:
        all_events_dur: list[NoteEvent] = []
        for evts in events_by_role.values():
            all_events_dur.extend(evts)
        try:
            refined_dur, duration_refine_stats_single = refine_durations(
                all_events_dur,
                audio_path,
                sr=22050,
                hop_length=settings.duration_refine_hop_length,
                floor_ratio=settings.duration_refine_floor_ratio,
                tail_sec=settings.duration_refine_tail_sec,
                min_duration_sec=settings.duration_refine_min_duration_sec,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("duration refinement raised: %s", exc)
            duration_refine_stats_single = DurationRefineStats(total_notes=len(all_events_dur))
        else:
            offset = 0
            for role in list(events_by_role.keys()):
                role_len = len(events_by_role[role])
                events_by_role[role] = refined_dur[offset : offset + role_len]
                offset += role_len

    # Waveform-derived tempo_map (best-effort; None on failure). This
    # lands before chord recognition so if we ever decide to share the
    # beat grid, the sequencing is already correct.
    audio_tempo_map = tempo_map_from_audio_path(audio_path)

    # Key + meter estimation on the source waveform. Fills the two
    # ``HarmonicAnalysis`` fields Basic Pitch does not predict. Both
    # estimators are best-effort and fall back to ``C:major`` / ``(4,4)``
    # on any failure, so the result is never worse than the old
    # hardcoded defaults — see :func:`~backend.services.key_estimation.analyze_audio`.
    key_label, time_signature, key_stats, meter_stats = _audio_mod._maybe_analyze_key_and_meter(
        audio_path,
    )

    # Chord recognition is audio-only and independent of the event
    # pipeline — it labels the waveform, and the labels attach to
    # HarmonicAnalysis.chords. Best-effort: any failure yields an empty
    # label list and a "skipped" stats marker.
    if settings.chord_recognition_enabled:
        try:
            chord_labels, chord_stats = recognize_chords(
                audio_path,
                min_score=settings.chord_min_template_score,
                hpss_margin=settings.chord_hpss_margin,
                seventh_enabled=settings.chord_seventh_templates_enabled,
                hmm_enabled=settings.chord_hmm_enabled,
                hmm_self_transition=settings.chord_hmm_self_transition,
                hmm_temperature=settings.chord_hmm_temperature,
                key_label="C:major",
            )
        except Exception as exc:  # noqa: BLE001 — never let chord recog sink transcribe
            log.warning("chord recognition raised: %s", exc)
            chord_labels = []
            chord_stats = ChordRecognitionStats(skipped=True)
            chord_stats.warnings.append(f"chord recognition failed: {exc}")

    # Cross-validate the KS key estimate against detected chords.
    # This helps resolve relative major/minor confusions (Am vs C)
    # by checking whether the chord progression better supports the
    # runner-up key. Best-effort: any failure returns the KS key
    # unchanged.
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

    # Rebuild the blob MIDI so its ``set_tempo`` meta event matches the
    # waveform-derived tempo. Fall back to the BP-default ``midi_data``
    # on any rebuild failure so blob persistence stays best-effort.
    initial_bpm = (
        float(audio_tempo_map[0].bpm)
        if audio_tempo_map
        else 120.0
    )
    blob_midi = _midi_mod._rebuild_blob_midi(cleaned_events, initial_bpm=initial_bpm)
    if blob_midi is None:
        blob_midi = midi_data

    result = _result_mod._pretty_midi_to_transcription_result(
        midi_data,
        events_by_role,
        model_output,
        tempo_map_override=audio_tempo_map,
        key_label=key_label,
        time_signature=time_signature,
        key_stats=key_stats,
        meter_stats=meter_stats,
        preprocess_stats=pass_result.preprocess_stats,
        cleanup_stats=pass_result.cleanup_stats,
        melody_stats=melody_stats,
        bass_stats=bass_stats,
        chord_stats=chord_stats,
        chord_labels=chord_labels,
        stem_stats=stem_stats,
        onset_refine_stats=onset_refine_stats_single,
        duration_refine_stats=duration_refine_stats_single,
    )
    midi_bytes = _midi_mod._serialize_pretty_midi(blob_midi)
    return result, midi_bytes
