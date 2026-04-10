"""Demucs-driven transcription pipeline (per-stem Basic Pitch passes).

Routing:

  * ``vocals.wav`` -> Basic Pitch -> MELODY events
  * ``bass.wav``   -> Basic Pitch -> BASS events
  * ``other.wav``  -> Basic Pitch -> CHORDS events
  * ``drums.wav``  -> ``tempo_map_from_audio_path`` (beat tracking)
  * ``other.wav``  -> ``recognize_chords`` (chroma + triad templates)

Each routing is gated by a ``demucs_use_*`` flag so an operator
can disable a single stem without touching the others. When a stem is
missing or its consumer returns empty, the stage falls back to the
original mixed audio. Extracted from ``transcribe.py``.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from backend.config import settings
from backend.contracts import InstrumentRole, RealtimeChordEvent, TranscriptionResult
from backend.services import transcribe_audio as _audio_mod
from backend.services import transcribe_inference as _bp_mod
from backend.services import transcribe_midi as _midi_mod
from backend.services import transcribe_pipeline_single as _single_mod
from backend.services import transcribe_result as _result_mod
from backend.services.audio_timing import tempo_map_from_audio_path
from backend.services.chord_recognition import (
    ChordRecognitionStats,
    recognize_chords,
)
from backend.services.crepe_melody import (
    CrepeMelodyStats,
    extract_vocal_melody_crepe,
    fuse_crepe_and_bp_melody,
)
from backend.services.duration_refine import (
    DurationRefineStats,
    refine_durations,
)
from backend.services.key_estimation import refine_key_with_chords
from backend.services.melody_extraction import (
    MelodyExtractionStats,
    backfill_melody_notes,
)
from backend.services.onset_refine import (
    OnsetRefineStats,
    refine_onsets,
)
from backend.services.stem_separation import (
    SeparatedStems,
    StemSeparationStats,
)
from backend.services.transcription_cleanup import (
    AmplitudeEnvelope,
    NoteEvent,
    cleanup_for_role,
)

log = logging.getLogger(__name__)


def _run_with_stems(
    audio_path: Path,
    stems: SeparatedStems,
    stem_stats: StemSeparationStats,
) -> tuple[TranscriptionResult, bytes | None]:
    """Demucs-driven pipeline — one Basic Pitch pass per stem.

    Routing:

      * ``vocals.wav`` -> Basic Pitch -> MELODY events
      * ``bass.wav``   -> Basic Pitch -> BASS events
      * ``other.wav``  -> Basic Pitch -> CHORDS events
      * ``drums.wav``  -> ``tempo_map_from_audio_path`` (beat tracking)
      * ``other.wav``  -> ``recognize_chords`` (chroma + triad templates)

    Each routing is gated by a ``demucs_use_*`` flag so an operator
    can disable a single stem without touching the others (e.g. turn
    off drums-for-beats on material where Demucs drums is noisy, and
    let beat tracking fall back to the mix). When a stem is missing
    or its consumer returns empty, we fall back to the *original*
    mixed audio for that stage.

    The Phase-2 Viterbi **melody** extractor *does* run on this path
    — but only on the vocals stem, using the vocals contour from
    that stem's own Basic Pitch pass to re-score BP's vocals note
    events (dropping octave ghosts / consonant false positives and
    back-filling soft sustained notes the polyphonic tracker lost
    under accompaniment bleed). The Phase-3 Viterbi **bass**
    extractor stays off here because the bass stem already fixes
    what it was compensating for in the single-mix path.

    Parallelism
    -----------
    The three Basic Pitch passes (vocals / bass / other) are
    independent, so by default they run concurrently in a
    :class:`~concurrent.futures.ThreadPoolExecutor` gated by
    :attr:`Settings.demucs_parallel_stems`. The shared cached
    ``basic_pitch.inference.Model`` is safe to call from multiple
    threads because its backing session — ONNX Runtime on Linux/CI,
    CoreML on Darwin — is documented thread-safe for concurrent
    ``.run()`` / ``.predict()`` calls, and ``basic_pitch.inference``
    itself has no module-level mutable state to race on. Most of the
    wall time (session inference, numpy window packing, ``librosa``
    resampling) happens in C extensions that release the GIL, so on
    a multi-core host the three passes overlap to ~1x Basic Pitch
    cost instead of ~3x.

    Set ``OHSHEET_DEMUCS_PARALLEL_STEMS=0`` to force serial
    execution (useful for debugging single-thread traces or
    reproducing sequential memory behavior).
    """
    events_by_role: dict[InstrumentRole, list[NoteEvent]] = {}
    per_stem_preprocess_stats: dict[str, Any] = {}
    per_stem_cleanup_stats: dict[str, Any] = {}
    per_stem_passes: dict[str, _bp_mod._BasicPitchPass] = {}

    # Try CREPE on the vocals stem first. If it returns events, it
    # owns MELODY and the Basic Pitch vocals pass is skipped entirely
    # — CREPE is SOTA for monophonic singing F0 and Basic Pitch adds
    # noise more than signal on this stem (see crepe_melody.py
    # module docstring). If CREPE is disabled or fails, we fall
    # through to the legacy Basic Pitch vocals path.
    crepe_events: list[NoteEvent] = []
    crepe_stats: CrepeMelodyStats | None = None
    if (
        settings.crepe_vocal_melody_enabled
        and settings.demucs_use_vocals_for_melody
        and stems.vocals is not None
    ):
        try:
            crepe_events, crepe_stats = extract_vocal_melody_crepe(
                stems.vocals,
                model=settings.crepe_model,
                device=settings.crepe_device,
                voicing_threshold=settings.crepe_voicing_threshold,
                median_filter_frames=settings.crepe_median_filter_frames,
                min_note_duration_sec=settings.crepe_min_note_duration_sec,
                merge_gap_sec=settings.crepe_merge_gap_sec,
                amp_min=settings.crepe_amp_min,
                amp_max=settings.crepe_amp_max,
                max_pitch_leap=settings.crepe_max_pitch_leap,
            )
        except Exception as exc:  # noqa: BLE001 — CREPE must not sink transcribe
            log.warning("crepe vocal melody raised: %s", exc)
            crepe_events = []
            crepe_stats = CrepeMelodyStats(skipped=True)
            crepe_stats.warnings.append(f"crepe-melody: exception: {exc}")
    crepe_owns_melody = bool(crepe_events)
    # In hybrid mode, we always run BP on vocals even when CREPE
    # succeeded, because the fusion step needs both event lists.
    # In non-hybrid mode, CREPE success still skips the BP vocals pass
    # (the pre-hybrid behavior).
    hybrid_mode = (
        crepe_owns_melody
        and settings.crepe_hybrid_enabled
    )

    # Build the list of stems we actually need to run Basic Pitch on,
    # honoring the per-consumer escape hatches. An absent stem (e.g.
    # a 2-source bag that didn't emit vocals) naturally drops out
    # here too. Vocals are skipped when CREPE already supplied the
    # melody track — unless hybrid mode is on, in which case we need
    # BP vocals events for the fusion step.
    stem_jobs: list[tuple[str, Path]] = []
    if (
        settings.demucs_use_vocals_for_melody
        and stems.vocals is not None
        and (not crepe_owns_melody or hybrid_mode)
    ):
        stem_jobs.append(("vocals", stems.vocals))
    if settings.demucs_use_bass_stem and stems.bass is not None:
        stem_jobs.append(("bass", stems.bass))
    if settings.demucs_use_other_for_chords and stems.other is not None:
        stem_jobs.append(("other", stems.other))

    # Per-stem threshold lookup — each stem gets its own tuned
    # thresholds. Precedence: stem-specific setting > generic
    # per-stem fallback > None (which lets _basic_pitch_single_pass
    # fall through to the global default).
    def _get_stem_bp_thresholds(label: str) -> tuple[float | None, float | None]:
        """Return (onset_threshold, frame_threshold) for a given stem label."""
        onset: float | None = None
        frame: float | None = None
        if label == "vocals":
            onset = settings.basic_pitch_stem_onset_threshold_vocals
            frame = settings.basic_pitch_stem_frame_threshold_vocals
        elif label == "bass":
            onset = settings.basic_pitch_stem_onset_threshold_bass
            frame = settings.basic_pitch_stem_frame_threshold_bass
        elif label == "other":
            onset = settings.basic_pitch_stem_onset_threshold_other
            frame = settings.basic_pitch_stem_frame_threshold_other
        # Fall back to generic per-stem overrides if stem-specific is None.
        if onset is None:
            onset = settings.basic_pitch_stem_onset_threshold
        if frame is None:
            frame = settings.basic_pitch_stem_frame_threshold
        return onset, frame

    def _get_stem_cleanup_thresholds() -> tuple[float | None, float | None]:
        """Return (octave_amp_ratio, ghost_max_duration_sec) for a stem."""
        return (
            settings.cleanup_stem_octave_amp_ratio,
            settings.cleanup_stem_ghost_max_duration_sec,
        )

    def _run_stem(job: tuple[str, Path]) -> tuple[str, _bp_mod._BasicPitchPass | None]:
        label, stem_path = job
        onset_thr, frame_thr = _get_stem_bp_thresholds(label)
        oct_ratio, ghost_dur = _get_stem_cleanup_thresholds()
        try:
            bp = _bp_mod._basic_pitch_single_pass(
                stem_path,
                keep_model_output=(label == "vocals"),
                onset_threshold=onset_thr,
                frame_threshold=frame_thr,
                cleanup_octave_amp_ratio=oct_ratio,
                cleanup_ghost_max_duration_sec=ghost_dur,
            )
        except Exception as exc:  # noqa: BLE001 — one bad stem must not sink the job
            log.warning("Basic Pitch failed on %s stem (%s): %s", label, stem_path, exc)
            return label, None
        return label, bp

    if stem_jobs and settings.demucs_parallel_stems and len(stem_jobs) > 1:
        # The cached Basic Pitch model is thread-safe to call
        # concurrently (see docstring) and most of the work happens
        # in GIL-releasing C extensions, so a ThreadPoolExecutor is
        # the right tool here — no pickling, no process spawn cost,
        # no duplicated model weights.
        max_workers = min(settings.demucs_parallel_max_workers, len(stem_jobs))
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="bp-stem",
        ) as ex:
            results = list(ex.map(_run_stem, stem_jobs))
    else:
        results = [_run_stem(job) for job in stem_jobs]

    for label, bp in results:
        if bp is None:
            continue
        per_stem_passes[label] = bp
        if bp.preprocess_stats is not None:
            per_stem_preprocess_stats[label] = bp.preprocess_stats
        per_stem_cleanup_stats[label] = bp.cleanup_stats

    # Per-role cleanup re-run: apply role-specific thresholds + energy
    # gating on each stem's events. The initial cleanup in
    # _basic_pitch_single_pass used global thresholds; this second pass
    # re-cleans with tighter/looser settings matched to each stem's
    # instrument role (melody = tighter merge, bass = looser sustain,
    # etc.) and optionally gates offsets using the stem's amplitude
    # envelope. The role map: vocals -> melody, bass -> bass, other -> chords.
    _stem_role_map = {"vocals": "melody", "bass": "bass", "other": "chords"}
    for label, bp in per_stem_passes.items():
        if not bp.cleaned_events:
            continue
        role = _stem_role_map.get(label)
        if role is None:
            continue
        # Compute amplitude envelope for energy gating — best-effort.
        stem_path = getattr(stems, label, None)
        envelope: AmplitudeEnvelope | None = None
        if stem_path is not None:
            try:
                envelope = _audio_mod._compute_amplitude_envelope(stem_path)
            except Exception as exc:  # noqa: BLE001
                log.debug("envelope computation failed for %s: %s", label, exc)
        recleaned, role_stats = cleanup_for_role(
            bp.cleaned_events, role, settings,
            amplitude_envelope=envelope,
        )
        bp.cleaned_events = recleaned
        per_stem_cleanup_stats[label] = role_stats

    vocals_bp = per_stem_passes.get("vocals")
    bass_bp = per_stem_passes.get("bass")
    other_bp = per_stem_passes.get("other")

    # MELODY precedence: CREPE > Viterbi-augmented Basic-Pitch-on-vocals
    # > raw Basic-Pitch-on-vocals. CREPE and Basic Pitch are mutually
    # exclusive (CREPE success keeps vocals out of stem_jobs above),
    # but checking both defensively keeps the assignment site readable
    # and survives any future refactor that re-enables simultaneous
    # Basic Pitch passes for A/B comparison.
    #
    # When Basic Pitch ran on the vocals stem, we run the Phase-2
    # Viterbi extractor in *additive-only* mode (``split_enabled=False``)
    # over the vocals contour. The per-stem ``cleanup_note_events``
    # pass has already dropped octave ghosts / consonant false
    # positives, so the Viterbi's role here is strictly to *add* soft
    # sustained notes BP missed under light accompaniment bleed — not
    # to re-filter. The earlier implementation also re-routed the
    # split's "chords" bucket into oblivion on this path, silently
    # dropping legitimate vocal harmonies / ornaments whenever the
    # main melodic Viterbi line rejected them.
    #
    # ``max_time_sec=vocals_duration_sec`` clamps back-filled notes to
    # the real end of the vocals stem. Basic Pitch's contour is
    # zero-padded past the audio end (block-size alignment) and the
    # tracer can otherwise emit a multi-second "note" in the padding.
    #
    # Any failure or empty Viterbi output falls through to the raw
    # BP events so the stems path never *loses* a melody track to
    # this pass.
    melody_stats: MelodyExtractionStats | None = None
    if crepe_owns_melody and hybrid_mode and vocals_bp is not None and vocals_bp.cleaned_events:
        # Hybrid CREPE+BP fusion: combine CREPE's pitch accuracy with
        # BP's onset/offset timing. Both ran on the vocals stem.
        try:
            fused_events = fuse_crepe_and_bp_melody(
                crepe_events,
                vocals_bp.cleaned_events,
                bp_min_amp=settings.crepe_hybrid_bp_min_amp,
                overlap_threshold=settings.crepe_hybrid_overlap_threshold,
            )
        except Exception as exc:  # noqa: BLE001 — fusion must not sink transcribe
            log.warning("crepe+bp fusion raised: %s; falling back to CREPE-only", exc)
            fused_events = crepe_events
        events_by_role[InstrumentRole.MELODY] = fused_events if fused_events else crepe_events
    elif crepe_owns_melody:
        # Non-hybrid or BP vocals had no events — use CREPE directly.
        events_by_role[InstrumentRole.MELODY] = crepe_events
    elif vocals_bp is not None and vocals_bp.cleaned_events:
        vocals_melody_events = vocals_bp.cleaned_events
        vocals_contour = vocals_bp.model_output.get("contour")
        vocals_duration_sec: float | None = None
        if stems.vocals is not None:
            vocals_duration_sec = _audio_mod._audio_duration_sec(stems.vocals)
        if (
            settings.melody_extraction_enabled
            and settings.melody_backfill_enabled
            and vocals_contour is not None
        ):
            try:
                extracted, melody_stats = backfill_melody_notes(
                    vocals_contour,
                    vocals_bp.cleaned_events,
                    melody_low_midi=settings.melody_low_midi,
                    melody_high_midi=settings.melody_high_midi,
                    voicing_floor=settings.melody_voicing_floor,
                    transition_weight=settings.melody_transition_weight,
                    max_transition_bins=settings.melody_max_transition_bins,
                    match_fraction=settings.melody_match_fraction,
                    backfill_min_duration_sec=settings.melody_backfill_min_duration_sec,
                    backfill_overlap_fraction=settings.melody_backfill_overlap_fraction,
                    backfill_min_amp=settings.melody_backfill_min_amp,
                    backfill_max_amp=settings.melody_backfill_max_amp,
                    max_time_sec=vocals_duration_sec,
                )
            except Exception as exc:  # noqa: BLE001 — never let Viterbi sink transcribe
                log.warning("vocals-stem melody Viterbi raised: %s", exc)
            else:
                if not melody_stats.skipped and extracted:
                    vocals_melody_events = extracted
        # Release the contour tensor unconditionally so it can be GC'd
        # before the result is built — mirrors ``keep_model_output=False``
        # for bass/other stems.
        vocals_bp.model_output.clear()
        events_by_role[InstrumentRole.MELODY] = vocals_melody_events
    if bass_bp is not None and bass_bp.cleaned_events:
        events_by_role[InstrumentRole.BASS] = bass_bp.cleaned_events
    if other_bp is not None and other_bp.cleaned_events:
        events_by_role[InstrumentRole.CHORDS] = other_bp.cleaned_events

    # If every per-stem pass came back empty we can't ship a useful
    # stem-driven result. Fall back to the single-mix pipeline so the
    # caller still gets notes, and tag stem_stats so the QualitySignal
    # explains why we bailed.
    if not events_by_role:
        log.warning(
            "all Demucs stems produced empty Basic Pitch output for %s; "
            "falling back to single-mix path",
            audio_path,
        )
        stem_stats.warnings.append("all stems empty; fell back to single-mix")
        return _single_mod._run_without_stems(audio_path, stem_stats)

    # Onset refinement — snap note onsets to spectral onset-strength peaks.
    # On the stems path we refine each role's events using the corresponding
    # stem's audio, which gives a cleaner ODF than the full mix.
    _role_stem_map = {
        InstrumentRole.MELODY: getattr(stems, "vocals", None),
        InstrumentRole.BASS: getattr(stems, "bass", None),
        InstrumentRole.CHORDS: getattr(stems, "other", None),
    }
    per_stem_onset_stats: dict[str, OnsetRefineStats] = {}
    if settings.onset_refine_enabled:
        _role_label_map = {
            InstrumentRole.MELODY: "vocals",
            InstrumentRole.BASS: "bass",
            InstrumentRole.CHORDS: "other",
        }
        for role, role_events in list(events_by_role.items()):
            stem_audio = _role_stem_map.get(role)
            refine_audio = stem_audio if stem_audio is not None else audio_path
            label = _role_label_map.get(role, str(role))
            try:
                refined_events, role_or_stats = refine_onsets(
                    role_events,
                    refine_audio,
                    sr=22050,
                    hop_length=settings.onset_refine_hop_length,
                    max_shift_sec=settings.onset_refine_max_shift_sec,
                )
            except Exception as exc:  # noqa: BLE001 — never let onset refine sink transcribe
                log.warning("onset refinement raised for %s: %s", label, exc)
                role_or_stats = OnsetRefineStats(
                    total_notes=len(role_events), skipped=True,
                )
                role_or_stats.warnings.append(f"onset-refine: exception: {exc}")
            else:
                events_by_role[role] = refined_events
            per_stem_onset_stats[label] = role_or_stats

    # Duration refinement — per-pitch CQT energy gating on stems.
    per_stem_duration_stats: dict[str, DurationRefineStats] = {}
    if settings.duration_refine_enabled:
        _dur_label_map = {
            InstrumentRole.MELODY: "vocals",
            InstrumentRole.BASS: "bass",
            InstrumentRole.CHORDS: "other",
        }
        for role, role_events in list(events_by_role.items()):
            stem_audio = _role_stem_map.get(role)
            refine_audio = stem_audio if stem_audio is not None else audio_path
            label = _dur_label_map.get(role, str(role))
            try:
                refined_dur_events, dur_stats = refine_durations(
                    role_events,
                    refine_audio,
                    sr=22050,
                    hop_length=settings.duration_refine_hop_length,
                    floor_ratio=settings.duration_refine_floor_ratio,
                    tail_sec=settings.duration_refine_tail_sec,
                    min_duration_sec=settings.duration_refine_min_duration_sec,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("duration refinement raised for %s: %s", label, exc)
                dur_stats = DurationRefineStats(total_notes=len(role_events))
            else:
                events_by_role[role] = refined_dur_events
            per_stem_duration_stats[label] = dur_stats

    # Tempo map — prefer the drums stem when enabled and available.
    # A drums-stem beat track that returns no beats (possible on
    # cappella / ambient material) falls back to the mix so the
    # downstream tempo_map is still waveform-derived.
    tempo_src: Path = audio_path
    if settings.demucs_use_drums_for_beats and stems.drums is not None:
        tempo_src = stems.drums
    audio_tempo_map = tempo_map_from_audio_path(tempo_src)
    if audio_tempo_map is None and tempo_src != audio_path:
        log.debug("drums-stem beat tracking returned None, retrying with mix")
        audio_tempo_map = tempo_map_from_audio_path(audio_path)

    # Key + meter estimation — always run against the original mix
    # rather than any single stem. Vocals alone lose the harmonic
    # support the KS profile keys on; "other" alone loses the melody
    # leading tones; drums alone obviously has no pitch content.
    # The mix sees all of it and is what downstream engrave will
    # render the key signature for anyway.
    key_label, time_signature, key_stats, meter_stats = _audio_mod._maybe_analyze_key_and_meter(
        audio_path,
    )

    # Chord recognition — prefer the "other" stem when enabled and
    # available. Chord labeling on the other stem is cleaner because
    # vocal sibilance and drum spectral leakage don't pollute the
    # chroma vectors. Same mix-fallback pattern as beat tracking.
    chord_labels: list[RealtimeChordEvent] = []
    chord_stats: ChordRecognitionStats | None = None
    if settings.chord_recognition_enabled:
        chord_src: Path = audio_path
        if settings.demucs_use_other_for_chords and stems.other is not None:
            chord_src = stems.other
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
            )
            if chord_stats.skipped and chord_src != audio_path:
                log.debug(
                    "chord recognition on other-stem skipped, retrying with mix"
                )
                chord_labels, chord_stats = recognize_chords(
                    audio_path,
                    min_score=settings.chord_min_template_score,
                    hpss_margin=settings.chord_hpss_margin,
                    seventh_enabled=settings.chord_seventh_templates_enabled,
                    hmm_enabled=settings.chord_hmm_enabled,
                    hmm_self_transition=settings.chord_hmm_self_transition,
                    hmm_temperature=settings.chord_hmm_temperature,
                    key_label=key_label,
                )
        except Exception as exc:  # noqa: BLE001 — chord recog must not sink transcribe
            log.warning("chord recognition raised: %s", exc)
            chord_labels = []
            chord_stats = ChordRecognitionStats(skipped=True)
            chord_stats.warnings.append(f"chord recognition failed: {exc}")

    # Cross-validate the KS key estimate against detected chords.
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

    # Pick a representative pretty_midi for the blob artifact — we
    # prefer the "other" pass because it carries the largest note
    # count on most material, and fall back to whichever stem did
    # return something. The combined builder flattens every stem's
    # notes into one list for serialization so the blob is still a
    # complete debugging artifact (just without role tags, which
    # pretty_midi can't represent losslessly anyway).
    #
    # This is reached only after the ``not events_by_role`` guard
    # above, so at least one of the three passes is non-None — the
    # ``or``-chain always returns a real _BasicPitchPass here. The
    # explicit binding is for type narrowing.
    representative_pass: _bp_mod._BasicPitchPass | None = other_bp or bass_bp or vocals_bp
    fallback_midi = representative_pass.midi_data if representative_pass else None
    # Use the librosa-derived tempo for the blob MIDI's ``set_tempo``
    # meta event so the declared tempo agrees with the notes. Without
    # this, basic-pitch.mid lands on a hard-coded 120 BPM and
    # MuseScore's import wizard re-infers tempo (and occasionally
    # time signature) from note density — see :func:`_rebuild_blob_midi`.
    initial_bpm = (
        float(audio_tempo_map[0].bpm)
        if audio_tempo_map
        else 120.0
    )
    combined_midi = _midi_mod._combined_midi_from_events(
        events_by_role, fallback_midi, initial_bpm=initial_bpm,
    )

    # ``model_output`` is intentionally empty on the stems path. The
    # ``_pretty_midi_to_transcription_result`` consumer only reads it
    # as a ``note_grid`` confidence fallback when ``all_amplitudes``
    # is empty, and the ``if not events_by_role`` guard above
    # guarantees at least one non-empty per-stem event list, so that
    # branch is unreachable here. Carrying a live reference would
    # just pin the (unused) contour tensors until the function
    # returns, defeating the ``keep_model_output=False`` memory win.
    stems_model_output: dict[str, Any] = {}

    result = _result_mod._pretty_midi_to_transcription_result(
        combined_midi,
        events_by_role,
        stems_model_output,
        tempo_map_override=audio_tempo_map,
        key_label=key_label,
        time_signature=time_signature,
        key_stats=key_stats,
        meter_stats=meter_stats,
        preprocess_stats=None,  # surfaced per-stem below
        cleanup_stats=None,     # surfaced per-stem below
        melody_stats=melody_stats,
        bass_stats=None,
        chord_stats=chord_stats,
        chord_labels=chord_labels,
        stem_stats=stem_stats,
        per_stem_preprocess_stats=per_stem_preprocess_stats,
        per_stem_cleanup_stats=per_stem_cleanup_stats,
        crepe_melody_stats=crepe_stats,
        per_stem_onset_refine_stats=per_stem_onset_stats if per_stem_onset_stats else None,
        per_stem_duration_refine_stats=per_stem_duration_stats if per_stem_duration_stats else None,
    )
    midi_bytes = _midi_mod._serialize_pretty_midi(combined_midi) if combined_midi else None
    return result, midi_bytes
