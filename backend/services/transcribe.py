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
from backend.services.audio_timing import tempo_map_from_audio_path
from backend.services.bass_extraction import (
    BassExtractionStats,
    extract_bass,
)
from backend.services.chord_recognition import (
    ChordRecognitionStats,
    recognize_chords,
)
from backend.services.melody_extraction import (
    MelodyExtractionStats,
    extract_melody,
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


def _pretty_midi_to_transcription_result(
    pm: Any,
    events_by_role: dict[InstrumentRole, list[NoteEvent]],
    model_output: dict[str, Any],
    default_bpm: float = 120.0,
    *,
    tempo_map_override: list[TempoMapEntry] | None = None,
    cleanup_stats: CleanupStats | None = None,
    melody_stats: MelodyExtractionStats | None = None,
    bass_stats: BassExtractionStats | None = None,
    chord_stats: ChordRecognitionStats | None = None,
    chord_labels: list[RealtimeChordEvent] | None = None,
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
    if cleanup_stats is not None:
        warnings.extend(cleanup_stats.as_warnings())
    if melody_stats is not None:
        warnings.extend(melody_stats.as_warnings())
    if bass_stats is not None:
        warnings.extend(bass_stats.as_warnings())
    if chord_stats is not None:
        warnings.extend(chord_stats.as_warnings())
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


def _run_basic_pitch_sync(audio_path: Path) -> tuple[TranscriptionResult, bytes | None]:
    """Synchronous Basic Pitch inference. Run inside asyncio.to_thread.

    Returns both the parsed ``TranscriptionResult`` and the raw MIDI bytes
    (if serialization succeeded) so the async caller can persist the MIDI
    to blob storage without blocking on disk I/O in the worker thread.

    The note events go through :func:`cleanup_note_events` before we
    rebuild ``pretty_midi``, so both the contract notes and the
    serialized ``.mid`` blob reflect the cleaned set. Cleanup is
    pure-Python and deterministic — see transcription_cleanup.py for the
    heuristics.
    """
    model = _load_basic_pitch_model()
    from basic_pitch.inference import predict  # noqa: PLC0415
    from basic_pitch.note_creation import note_events_to_midi  # noqa: PLC0415

    model_output, midi_data, note_events = predict(
        str(audio_path),
        model_or_model_path=model,
        onset_threshold=settings.basic_pitch_onset_threshold,
        frame_threshold=settings.basic_pitch_frame_threshold,
        minimum_note_length=settings.basic_pitch_minimum_note_length_ms,
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
    if cleanup_stats.output_count != cleanup_stats.input_count or cleanup_stats.merged:
        try:
            midi_data = note_events_to_midi(cleaned_events)
        except Exception as exc:  # noqa: BLE001 — never let cleanup sink the job
            log.warning("note_events_to_midi rebuild failed, using raw pm: %s", exc)
            cleaned_events = note_events  # fall back to raw
            cleanup_stats.warnings.append("cleanup: rebuild failed, using raw pm")

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

    result = _pretty_midi_to_transcription_result(
        midi_data,
        events_by_role,
        model_output,
        tempo_map_override=audio_tempo_map,
        cleanup_stats=cleanup_stats,
        melody_stats=melody_stats,
        bass_stats=bass_stats,
        chord_stats=chord_stats,
        chord_labels=chord_labels,
    )
    midi_bytes = _serialize_pretty_midi(midi_data)
    return result, midi_bytes


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

        return result
