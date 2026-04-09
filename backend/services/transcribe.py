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
import io
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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
    RealtimeChordEvent,
    TempoMapEntry,
    TranscriptionResult,
)
from backend.services.audio_preprocess import (
    PreprocessStats,
    preprocess_audio_file,
)
from backend.services.audio_timing import tempo_map_from_audio_path
from backend.services.bass_extraction import (
    BassExtractionStats,
    extract_bass,
)
from backend.services.chord_recognition import (
    ChordRecognitionStats,
    recognize_chords,
)
from backend.services.crepe_melody import (
    CrepeMelodyStats,
    extract_vocal_melody_crepe,
)
from backend.services.melody_extraction import (
    MelodyExtractionStats,
    extract_melody,
)
from backend.services.stem_separation import (
    SeparatedStems,
    StemSeparationStats,
    separate_stems,
)
from backend.services.transcription_cleanup import (
    CleanupStats,
    NoteEvent,
    cleanup_note_events,
)
from backend.storage.base import BlobStore

log = logging.getLogger(__name__)


# Cached Basic Pitch model — building the inference session (CoreML on
# Darwin, ONNX/TFLite elsewhere) costs ~1s, so we load it once per process.
# Held as Any to avoid importing basic_pitch at module import time
# (optional dep, stub path needs to work without it).
_BP_MODEL: Any = None


def _event_to_note(event: NoteEvent) -> Note:
    """Convert a Basic Pitch ``note_events`` tuple to a contract ``Note``.

    Basic Pitch's own velocity formula is ``int(round(127 * amplitude))``
    (see ``basic_pitch.note_creation.note_events_to_midi``); we replicate
    it here so the contract notes match what the rebuilt pretty_midi
    would contain, without needing to cross-reference ``pm.instruments``.
    """
    start, end, pitch, amplitude, _bends = event
    velocity = int(round(127 * float(amplitude)))
    velocity = max(1, min(127, velocity))
    return Note(
        pitch=int(pitch),
        onset_sec=float(start),
        offset_sec=float(end),
        velocity=velocity,
    )


def _prefixed_warnings(label: str, warnings: list[str]) -> list[str]:
    """Tag each warning line with a ``[label]`` prefix.

    Used when a stage runs once per Demucs stem — the base warning
    string comes from ``Stats.as_warnings()`` and doesn't know which
    stem produced it, so we decorate at the assembly site instead of
    mutating the stats objects themselves (which would make them
    harder to diff across re-runs).
    """
    if not label:
        return list(warnings)
    return [f"[{label}] {w}" for w in warnings]


def _pretty_midi_to_transcription_result(
    pm: Any,
    events_by_role: dict[InstrumentRole, list[NoteEvent]],
    model_output: dict[str, Any],
    default_bpm: float = 120.0,
    *,
    tempo_map_override: list[TempoMapEntry] | None = None,
    preprocess_stats: PreprocessStats | None = None,
    cleanup_stats: CleanupStats | None = None,
    melody_stats: MelodyExtractionStats | None = None,
    bass_stats: BassExtractionStats | None = None,
    chord_stats: ChordRecognitionStats | None = None,
    chord_labels: list[RealtimeChordEvent] | None = None,
    stem_stats: StemSeparationStats | None = None,
    per_stem_preprocess_stats: dict[str, PreprocessStats] | None = None,
    per_stem_cleanup_stats: dict[str, CleanupStats] | None = None,
    crepe_melody_stats: CrepeMelodyStats | None = None,
) -> TranscriptionResult:
    """Convert Basic Pitch's output into our pydantic TranscriptionResult.

    ``events_by_role`` maps a ``InstrumentRole`` (MELODY / BASS / CHORDS
    after Phase 2+3 extraction, or a single PIANO fallback) to the list
    of ``NoteEvent`` tuples belonging to that role. One ``MidiTrack`` is
    emitted per non-empty role. Per-track confidence is the mean
    amplitude of that role's events, clamped to [0.1, 1.0].

    ``pm`` is retained only so we can fall back to ``pm.estimate_tempo``
    when the waveform-derived tempo map is unavailable — the contract
    notes are built from ``events_by_role`` directly.

    If ``tempo_map_override`` is provided (e.g. from waveform beat
    tracking), it replaces the single-anchor map we'd otherwise build
    from ``pm.estimate_tempo`` so arrange's ``sec_to_beat`` aligns
    quantization to the real pulse of the recording.

    ``chord_labels`` (when provided) are attached to
    ``HarmonicAnalysis.chords``. The labels come from
    :func:`recognize_chords` — a chroma + triad template pass over the
    source waveform. Empty list means "no chord recognition ran or
    nothing scored above threshold".
    """
    import numpy as np  # noqa: PLC0415 — heavy/optional dep

    midi_tracks: list[MidiTrack] = []
    all_amplitudes: list[float] = []

    # Deterministic track order so test output is stable — MELODY first
    # (it's the arrange right-hand target), then the rest.
    _order = [
        InstrumentRole.MELODY,
        InstrumentRole.BASS,
        InstrumentRole.CHORDS,
        InstrumentRole.PIANO,
        InstrumentRole.OTHER,
    ]
    ordered_roles = [r for r in _order if r in events_by_role] + [
        r for r in events_by_role if r not in _order
    ]

    for role in ordered_roles:
        events = events_by_role.get(role, [])
        if not events:
            continue
        contract_notes = [_event_to_note(ev) for ev in events]
        contract_notes.sort(key=lambda n: (n.onset_sec, n.pitch))

        amps = [float(ev[3]) for ev in events]
        all_amplitudes.extend(amps)
        role_conf = float(np.mean(amps)) if amps else 0.3
        role_conf = round(min(max(role_conf, 0.1), 1.0), 2)

        midi_tracks.append(
            MidiTrack(
                notes=contract_notes,
                instrument=role,
                program=0,
                confidence=role_conf,
            )
        )

    # Overall confidence — mean of per-note amplitudes across all roles.
    # Fall back to model_output["note"] mean if everything was empty.
    if all_amplitudes:
        overall_conf = float(np.mean(all_amplitudes))
    else:
        note_grid = model_output.get("note")
        overall_conf = float(np.mean(note_grid)) if note_grid is not None else 0.3
    overall_conf = round(min(max(overall_conf, 0.1), 1.0), 2)

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
        chords=list(chord_labels) if chord_labels else [],
        sections=[],
    )

    total_notes = sum(len(t.notes) for t in midi_tracks)
    warnings: list[str] = [
        "Basic Pitch baseline (polyphonic pitch tracker, no instrument separation)"
    ]
    if tempo_map_override:
        warnings.append("tempo_map from audio beat tracking (librosa)")
    if stem_stats is not None:
        warnings.extend(stem_stats.as_warnings())
    # Single-pass stats (no Demucs). When the Demucs path is active
    # these stay None and the per_stem_* dicts carry the equivalents.
    if preprocess_stats is not None:
        warnings.extend(preprocess_stats.as_warnings())
    if cleanup_stats is not None:
        warnings.extend(cleanup_stats.as_warnings())
    # Per-stem stats — one entry per active stem, prefixed so the
    # reader can tell them apart (e.g. ``[vocals] cleanup: 12 merged``).
    if per_stem_preprocess_stats:
        for label, pps in per_stem_preprocess_stats.items():
            warnings.extend(_prefixed_warnings(label, pps.as_warnings()))
    if per_stem_cleanup_stats:
        for label, cps in per_stem_cleanup_stats.items():
            warnings.extend(_prefixed_warnings(label, cps.as_warnings()))
    if melody_stats is not None:
        warnings.extend(melody_stats.as_warnings())
    if bass_stats is not None:
        warnings.extend(bass_stats.as_warnings())
    if chord_stats is not None:
        warnings.extend(chord_stats.as_warnings())
    if crepe_melody_stats is not None:
        warnings.extend(crepe_melody_stats.as_warnings())
    if total_notes < 20:
        warnings.append(f"Low note count ({total_notes}) — possible quality issue")
    quality = QualitySignal(
        overall_confidence=overall_conf if midi_tracks else 0.1,
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
    log.info("transcribe: stub result — %s", reason)
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


def _audio_duration_sec(path: Path) -> float | None:
    """Return the duration of an audio file in seconds, or None on failure.

    Used to clamp the Viterbi melody back-fill so synthesized notes can't
    extend past the real end of the audio. Basic Pitch's contour tensor
    is zero-padded past the audio end (the mel spectrogram is rounded up
    to a model block size), and the back-fill tracer can otherwise pick
    up low-salience runs in that padding and emit ghost notes beyond the
    song. See :func:`_backfill_missed_melody_notes`.

    ``soundfile`` is tried first because it reads headers only for WAV
    (the Demucs stems format), avoiding a full decode. Failure falls
    through to ``librosa.get_duration`` which handles m4a/mp3. Any
    exception returns ``None`` so callers can proceed without clamping.
    """
    try:
        import soundfile as sf  # noqa: PLC0415 — optional
        info = sf.info(str(path))
        if info.samplerate > 0 and info.frames > 0:
            return float(info.frames) / float(info.samplerate)
    except Exception:  # noqa: BLE001 — fall through to librosa
        log.debug("soundfile header read failed for %s", path, exc_info=True)
    try:
        import librosa  # noqa: PLC0415 — ships with the basic-pitch extra
        return float(librosa.get_duration(path=str(path)))
    except Exception as exc:  # noqa: BLE001 — best-effort
        log.debug("audio duration probe failed for %s: %s", path, exc)
        return None


def _serialize_pretty_midi(pm: Any) -> bytes | None:
    """Serialize a pretty_midi.PrettyMIDI to raw .mid bytes.

    pretty_midi >= 0.2.10 accepts a file-like object in ``write()`` and
    forwards it to mido as ``file=...``, so we can avoid a temp file.
    Returns None on any failure — blob persistence is best-effort and must
    never break transcription.
    """
    try:
        buf = io.BytesIO()
        pm.write(buf)
        return buf.getvalue()
    except Exception as exc:  # noqa: BLE001 — best-effort serialization
        log.warning("Failed to serialize pretty_midi for blob storage: %s", exc)
        return None


@dataclass
class _BasicPitchPass:
    """Captured outputs of one Basic Pitch inference run on a single audio file.

    Returned by :func:`_basic_pitch_single_pass` so the orchestrator in
    :func:`_run_basic_pitch_sync` can either use one pass directly
    (no-Demucs path) or stitch several passes together (one per stem).
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
) -> _BasicPitchPass:
    """Run preprocess → Basic Pitch → cleanup for one audio file.

    Factored out of :func:`_run_basic_pitch_sync` so the Demucs path
    can call it once per stem (vocals / bass / other) without
    duplicating the preprocess + predict + cleanup boilerplate. The
    returned :class:`_BasicPitchPass` carries everything downstream
    consumers might want: the cleaned note events, Basic Pitch's
    ``model_output`` (for contour access in Viterbi fallbacks),
    the rebuilt pretty_midi for blob serialization, and the
    per-pass stats objects.

    Honors ``settings.audio_preprocess_enabled`` exactly like the
    old inline code did — the preprocessed tempfile (if any) is
    unlinked in a ``finally`` regardless of whether ``predict``
    succeeded, so we never leak temp WAVs on the inference path.

    ``keep_model_output`` defaults to True for the single-mix path,
    where the downstream Viterbi melody/bass extractors read
    ``model_output["contour"]``. The stems path passes False so the
    contour tensor (tens of MB per stem — three copies in the
    parallel path) can be garbage-collected as soon as Basic Pitch's
    note events are extracted. On the stems path that matrix is
    dead data: the ``if not events_by_role`` guard in
    :func:`_run_with_stems` makes the ``model_output.get("note")``
    confidence fallback in :func:`_pretty_midi_to_transcription_result`
    unreachable.
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
            onset_threshold=settings.basic_pitch_onset_threshold,
            frame_threshold=settings.basic_pitch_frame_threshold,
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
    # ghosts and quiet ghost-tail notes. Rebuild pretty_midi from the
    # cleaned list so the blob-stored .mid matches the contract.
    cleaned_events, cleanup_stats = cleanup_note_events(
        note_events,
        merge_gap_sec=settings.cleanup_merge_gap_sec,
        octave_amp_ratio=settings.cleanup_octave_amp_ratio,
        octave_onset_tol_sec=settings.cleanup_octave_onset_tol_sec,
        ghost_max_duration_sec=settings.cleanup_ghost_max_duration_sec,
        ghost_amp_median_scale=settings.cleanup_ghost_amp_median_scale,
    )
    if cleanup_stats.output_count != cleanup_stats.input_count:
        try:
            midi_data = note_events_to_midi(cleaned_events)
        except Exception as exc:  # noqa: BLE001 — never let cleanup sink the job
            log.warning("note_events_to_midi rebuild failed, using raw pm: %s", exc)
            cleaned_events = note_events  # fall back to raw
            cleanup_stats.warnings.append("cleanup: rebuild failed, using raw pm")

    # Drop the shared reference to ``model_output`` before the
    # function returns when the caller doesn't need it. The local
    # binding above goes out of scope regardless, but keeping this
    # explicit means the stems path never hands a live reference
    # to the contour tensor back to the orchestrator — the C-backed
    # numpy arrays can be reclaimed before the next stem's
    # ``predict()`` allocates its own.
    if not keep_model_output:
        model_output.clear()
    return _BasicPitchPass(
        cleaned_events=cleaned_events,
        model_output=model_output,
        midi_data=midi_data,
        preprocess_stats=preprocess_stats,
        cleanup_stats=cleanup_stats,
    )


def _rebuild_blob_midi(
    events: list[NoteEvent],
    *,
    initial_bpm: float,
) -> Any:
    """Build a fresh pretty_midi from a flat event list for blob storage.

    ``initial_bpm`` becomes the MIDI ``set_tempo`` meta-event on
    track 0. basic-pitch's own ``note_events_to_midi`` hard-codes
    120 BPM, which mismatches every piece of music that isn't at
    120, and notation importers (MuseScore's MIDI wizard in
    particular) treat the declared tempo as a hint and re-infer
    metric structure from note density when it looks wrong —
    occasionally landing on the wrong time signature in the
    process. Wiring the librosa-derived tempo through here makes
    the blob MIDI agree with its notes, so importers have no
    reason to re-infer. Callers pass ``audio_tempo_map[0].bpm``
    when available; the 120 default matches basic-pitch for the
    (rare) case where beat tracking failed.

    The time-signature meta event is also set explicitly — pretty_midi
    would emit a default 4/4 anyway, but making it explicit documents
    intent and keeps any future pretty_midi behavior change from
    silently dropping the event.

    Returns ``None`` on any failure (missing optional deps, empty
    events, ``note_events_to_midi`` blowup) so callers can fall back
    to a pre-existing pretty_midi without sinking blob persistence.
    """
    if not events:
        return None
    try:
        import pretty_midi  # noqa: PLC0415 — optional dep
        from basic_pitch.note_creation import note_events_to_midi  # noqa: PLC0415
    except ImportError:
        return None
    try:
        pm = note_events_to_midi(events, midi_tempo=initial_bpm)
    except Exception as exc:  # noqa: BLE001 — best-effort
        log.warning("note_events_to_midi rebuild failed: %s", exc)
        return None
    pm.time_signature_changes = [
        pretty_midi.TimeSignature(numerator=4, denominator=4, time=0.0)
    ]
    return pm


def _combined_midi_from_events(
    events_by_role: dict[InstrumentRole, list[NoteEvent]],
    fallback: Any,
    *,
    initial_bpm: float = 120.0,
) -> Any:
    """Build a single pretty_midi from the concatenated per-role events.

    Used on the Demucs path where we have three independent Basic
    Pitch passes (vocals / bass / other) and need *one* ``.mid``
    artifact for blob storage. We flatten the per-role note lists
    into time-sorted order and delegate to :func:`_rebuild_blob_midi`,
    which handles the pretty_midi construction + tempo / TS meta
    event wiring. The blob MIDI is a debugging artifact, not
    authoritative, so collapsing the role split there is fine (the
    contract ``TranscriptionResult`` keeps the split intact).

    On any failure we fall back to ``fallback`` (typically the last
    pretty_midi we produced in the loop) so blob persistence stays
    best-effort.
    """
    combined: list[NoteEvent] = []
    for events in events_by_role.values():
        combined.extend(events)
    if not combined:
        return fallback
    combined.sort(key=lambda e: (e[0], e[2]))
    pm = _rebuild_blob_midi(combined, initial_bpm=initial_bpm)
    return pm if pm is not None else fallback


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
    pass_result = _basic_pitch_single_pass(audio_path)
    cleaned_events = pass_result.cleaned_events
    model_output = pass_result.model_output
    midi_data = pass_result.midi_data

    # Audio duration is used to clamp the Viterbi melody back-fill so
    # synthesized notes don't extend past the real end of the audio
    # (see :func:`_backfill_missed_melody_notes`). Best-effort: ``None``
    # leaves back-fill unclamped, which matches the pre-fix behaviour.
    audio_duration_sec = _audio_duration_sec(audio_path)

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

    # Waveform-derived tempo_map (best-effort; None on failure). This
    # lands before chord recognition so if we ever decide to share the
    # beat grid, the sequencing is already correct.
    audio_tempo_map = tempo_map_from_audio_path(audio_path)

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
            )
        except Exception as exc:  # noqa: BLE001 — never let chord recog sink transcribe
            log.warning("chord recognition raised: %s", exc)
            chord_labels = []
            chord_stats = ChordRecognitionStats(skipped=True)
            chord_stats.warnings.append(f"chord recognition failed: {exc}")

    # Rebuild the blob MIDI so its ``set_tempo`` meta event matches the
    # waveform-derived tempo. basic-pitch's own ``note_events_to_midi``
    # hard-codes 120 BPM, which notation importers (MuseScore's MIDI
    # wizard especially) treat as a hint and re-infer from note density
    # when it contradicts the note pattern — occasionally landing on the
    # wrong time signature in the process. Fall back to the BP-default
    # ``midi_data`` on any rebuild failure so blob persistence stays
    # best-effort.
    initial_bpm = (
        float(audio_tempo_map[0].bpm)
        if audio_tempo_map
        else 120.0
    )
    blob_midi = _rebuild_blob_midi(cleaned_events, initial_bpm=initial_bpm)
    if blob_midi is None:
        blob_midi = midi_data

    # NOTE: ``midi_data`` below is Basic Pitch's original pretty_midi
    # (fixed 120 BPM) and feeds ``TranscriptionResult.midi_data`` /
    # note-grid construction, which only cares about pitches + times.
    # The serialized ``midi_bytes`` further down comes from ``blob_midi``
    # — the tempo-aware rebuild — so the on-disk .mid carries the
    # librosa-derived tempo. See ``_rebuild_blob_midi`` docstring.
    result = _pretty_midi_to_transcription_result(
        midi_data,
        events_by_role,
        model_output,
        tempo_map_override=audio_tempo_map,
        preprocess_stats=pass_result.preprocess_stats,
        cleanup_stats=pass_result.cleanup_stats,
        melody_stats=melody_stats,
        bass_stats=bass_stats,
        chord_stats=chord_stats,
        chord_labels=chord_labels,
        stem_stats=stem_stats,
    )
    midi_bytes = _serialize_pretty_midi(blob_midi)
    return result, midi_bytes


def _run_with_stems(
    audio_path: Path,
    stems: SeparatedStems,
    stem_stats: StemSeparationStats,
) -> tuple[TranscriptionResult, bytes | None]:
    """Demucs-driven pipeline — one Basic Pitch pass per stem.

    Routing:

      * ``vocals.wav`` → Basic Pitch → MELODY events
      * ``bass.wav``   → Basic Pitch → BASS events
      * ``other.wav``  → Basic Pitch → CHORDS events
      * ``drums.wav``  → ``tempo_map_from_audio_path`` (beat tracking)
      * ``other.wav``  → ``recognize_chords`` (chroma + triad templates)

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
    per_stem_preprocess_stats: dict[str, PreprocessStats] = {}
    per_stem_cleanup_stats: dict[str, CleanupStats] = {}
    per_stem_passes: dict[str, _BasicPitchPass] = {}

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
            )
        except Exception as exc:  # noqa: BLE001 — CREPE must not sink transcribe
            log.warning("crepe vocal melody raised: %s", exc)
            crepe_events = []
            crepe_stats = CrepeMelodyStats(skipped=True)
            crepe_stats.warnings.append(f"crepe-melody: exception: {exc}")
    crepe_owns_melody = bool(crepe_events)

    # Build the list of stems we actually need to run Basic Pitch on,
    # honoring the per-consumer escape hatches. An absent stem (e.g.
    # a 2-source bag that didn't emit vocals) naturally drops out
    # here too. Vocals are skipped when CREPE already supplied the
    # melody track.
    stem_jobs: list[tuple[str, Path]] = []
    if (
        settings.demucs_use_vocals_for_melody
        and stems.vocals is not None
        and not crepe_owns_melody
    ):
        stem_jobs.append(("vocals", stems.vocals))
    if settings.demucs_use_bass_stem and stems.bass is not None:
        stem_jobs.append(("bass", stems.bass))
    if settings.demucs_use_other_for_chords and stems.other is not None:
        stem_jobs.append(("other", stems.other))

    def _run_stem(job: tuple[str, Path]) -> tuple[str, _BasicPitchPass | None]:
        label, stem_path = job
        try:
            # Keep ``model_output`` on the vocals stem so the Phase-2
            # Viterbi melody extractor can re-score the BP vocals
            # events against the vocals contour below. Bass / other
            # drop it immediately: on the parallel path this matters
            # twice, since the three passes overlap in time (peak
            # memory would otherwise carry three live contour
            # matrices), and neither of those two stems feeds a
            # downstream ``model_output`` consumer — the bass
            # Viterbi fallback stays disabled here because the bass
            # stem already fixes what it was compensating for.
            bp = _basic_pitch_single_pass(
                stem_path, keep_model_output=(label == "vocals"),
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
    if crepe_owns_melody:
        events_by_role[InstrumentRole.MELODY] = crepe_events
    elif vocals_bp is not None and vocals_bp.cleaned_events:
        vocals_melody_events = vocals_bp.cleaned_events
        vocals_contour = vocals_bp.model_output.get("contour")
        vocals_duration_sec: float | None = None
        if stems.vocals is not None:
            vocals_duration_sec = _audio_duration_sec(stems.vocals)
        if settings.melody_extraction_enabled and vocals_contour is not None:
            try:
                extracted, _chords, melody_stats = extract_melody(
                    vocals_contour,
                    vocals_bp.cleaned_events,
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
                    max_time_sec=vocals_duration_sec,
                    split_enabled=False,
                )
            except Exception as exc:  # noqa: BLE001 — never let Viterbi sink transcribe
                log.warning("vocals-stem melody Viterbi raised: %s", exc)
            else:
                # Only promote the Viterbi output when it actually
                # ran and found voiced content. ``skipped=True``
                # means numpy/contour-shape issues; an empty
                # ``extracted`` with no skip means the path was
                # entirely unvoiced — either way, fall back to the
                # raw BP events rather than silently drop the track.
                # (``_chords`` is always ``[]`` in ``split_enabled=False``
                # mode — the split step is bypassed — but we still
                # unpack the triple to keep the call site honest
                # about the public ``extract_melody`` return shape.)
                del _chords
                if not melody_stats.skipped and extracted:
                    vocals_melody_events = extracted
            # Release the contour reference now that the Viterbi is done
            # with it — mirrors the ``keep_model_output=False`` memory
            # optimization for the bass/other stems so the vocals
            # contour tensor can be GC'd before the result is built.
            # Kept inside the ``melody_extraction_enabled`` guard so a
            # future downstream consumer that reads ``model_output``
            # past this point isn't silently handed an empty dict.
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
        return _run_without_stems(audio_path, stem_stats)

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
            )
            if chord_stats.skipped and chord_src != audio_path:
                log.debug(
                    "chord recognition on other-stem skipped, retrying with mix"
                )
                chord_labels, chord_stats = recognize_chords(
                    audio_path,
                    min_score=settings.chord_min_template_score,
                    hpss_margin=settings.chord_hpss_margin,
                )
        except Exception as exc:  # noqa: BLE001 — chord recog must not sink transcribe
            log.warning("chord recognition raised: %s", exc)
            chord_labels = []
            chord_stats = ChordRecognitionStats(skipped=True)
            chord_stats.warnings.append(f"chord recognition failed: {exc}")

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
    representative_pass: _BasicPitchPass | None = other_bp or bass_bp or vocals_bp
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
    combined_midi = _combined_midi_from_events(
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

    result = _pretty_midi_to_transcription_result(
        combined_midi,
        events_by_role,
        stems_model_output,
        tempo_map_override=audio_tempo_map,
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
    )
    midi_bytes = _serialize_pretty_midi(combined_midi) if combined_midi else None
    return result, midi_bytes


def _run_basic_pitch_sync(audio_path: Path) -> tuple[TranscriptionResult, bytes | None]:
    """Synchronous Basic Pitch inference. Run inside asyncio.to_thread.

    Returns both the parsed ``TranscriptionResult`` and the raw MIDI bytes
    (if serialization succeeded) so the async caller can persist the MIDI
    to blob storage without blocking on disk I/O in the worker thread.

    Dispatches between two pipelines:

      * **Demucs path** (``settings.demucs_enabled``): split the source
        into 4 stems, run Basic Pitch once per stem (vocals / bass /
        other), and route drums + other to the beat-track and chord
        recognizers. See :func:`_run_with_stems`.
      * **Single-mix path** (default): one Basic Pitch pass on the
        whole mix + Viterbi melody/bass split + chord recognition on
        the original waveform. See :func:`_run_without_stems`.

    Any failure in the Demucs path (separation crash, all-stems-empty,
    missing torch) transparently falls back to the single-mix path so
    flipping ``demucs_enabled`` is always safe — the worst case is that
    the user pays the Demucs load cost for nothing.
    """
    stems: SeparatedStems | None = None
    stem_stats: StemSeparationStats | None = None

    if settings.demucs_enabled:
        stems, stem_stats = separate_stems(
            audio_path,
            model_name=settings.demucs_model,
            device=settings.demucs_device,
            segment_sec=settings.demucs_segment_sec,
            shifts=settings.demucs_shifts,
            overlap=settings.demucs_overlap,
            split=settings.demucs_split,
        )

    try:
        if stems is not None and stem_stats is not None:
            return _run_with_stems(audio_path, stems, stem_stats)
        return _run_without_stems(audio_path, stem_stats)
    finally:
        if stems is not None:
            stems.cleanup()


class TranscribeService:
    name = "transcribe"

    def __init__(self, blob_store: BlobStore | None = None) -> None:
        # Optional so the service can still be constructed in bare unit tests
        # that don't exercise the persistence path. In production (via
        # backend.api.deps.get_runner) it's always injected.
        self.blob_store = blob_store

    async def run(
        self,
        payload: InputBundle,
        *,
        job_id: str | None = None,
    ) -> TranscriptionResult:
        log.info(
            "transcribe: start job_id=%s audio_uri=%s",
            job_id or "—",
            payload.audio.uri if payload.audio else None,
        )
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
            result, midi_bytes = await asyncio.to_thread(
                _run_basic_pitch_sync, audio_path,
            )
        except ImportError as exc:
            log.warning("Basic Pitch deps unavailable (%s) — using stub", exc)
            return _stub_result(f"missing dependency: {exc}")
        except Exception as exc:  # noqa: BLE001 — boundary; we don't want one bad audio file to crash the worker
            log.exception("Basic Pitch inference failed for %s", audio_path)
            return _stub_result(f"inference failed: {exc}")

        # Persist the raw transcription MIDI to blob storage so it's
        # retrievable alongside the engraved output. Best-effort: a storage
        # failure shouldn't sink the job, since the downstream pipeline only
        # needs the parsed notes in ``result``.
        if midi_bytes and self.blob_store is not None and job_id is not None:
            try:
                uri = self.blob_store.put_bytes(
                    f"jobs/{job_id}/transcription/basic-pitch.mid",
                    midi_bytes,
                )
                result = result.model_copy(update={"transcription_midi_uri": uri})
            except Exception as exc:  # noqa: BLE001 — best-effort persistence
                log.warning("Failed to persist transcription MIDI for %s: %s", job_id, exc)

        n_notes = sum(len(t.notes) for t in result.midi_tracks)
        log.info(
            "transcribe: done job_id=%s tracks=%d notes=%d transcription_midi=%s",
            job_id or "—",
            len(result.midi_tracks),
            n_notes,
            "yes" if result.transcription_midi_uri else "no",
        )
        return result
