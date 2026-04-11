"""TranscriptionResult assembly and stub fallback.

Converts internal ``NoteEvent`` lists and per-stage stats into the
pydantic ``TranscriptionResult`` contract, or returns a shape-correct
stub when real inference is unavailable. Extracted from ``transcribe.py``.
"""
from __future__ import annotations

import logging
from typing import Any

from backend.contracts import (
    SCHEMA_VERSION,
    HarmonicAnalysis,
    InstrumentRole,
    MidiTrack,
    Note,
    QualitySignal,
    RealtimeChordEvent,
    TempoMapEntry,
    TranscriptionResult,
)
from backend.services.audio_preprocess import PreprocessStats
from backend.services.bass_extraction import BassExtractionStats
from backend.services.chord_recognition import ChordRecognitionStats
from backend.services.crepe_melody import CrepeMelodyStats
from backend.services.duration_refine import DurationRefineStats
from backend.services.key_estimation import KeyEstimationStats, MeterEstimationStats
from backend.services.melody_extraction import MelodyExtractionStats
from backend.services.onset_refine import OnsetRefineStats
from backend.services.stem_separation import StemSeparationStats
from backend.services.transcription_cleanup import CleanupStats, NoteEvent

log = logging.getLogger(__name__)


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
    key_label: str = "C:major",
    time_signature: tuple[int, int] = (4, 4),
    key_stats: KeyEstimationStats | None = None,
    meter_stats: MeterEstimationStats | None = None,
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
    onset_refine_stats: OnsetRefineStats | None = None,
    per_stem_onset_refine_stats: dict[str, OnsetRefineStats] | None = None,
    duration_refine_stats: DurationRefineStats | None = None,
    per_stem_duration_refine_stats: dict[str, DurationRefineStats] | None = None,
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
        key=key_label,
        time_signature=time_signature,
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
    if key_stats is not None:
        warnings.extend(key_stats.as_warnings())
    if meter_stats is not None:
        warnings.extend(meter_stats.as_warnings())
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
    if onset_refine_stats is not None:
        warnings.extend(onset_refine_stats.as_warnings())
    if per_stem_onset_refine_stats:
        for label, ors in per_stem_onset_refine_stats.items():
            warnings.extend(_prefixed_warnings(label, ors.as_warnings()))
    if duration_refine_stats is not None:
        warnings.extend(duration_refine_stats.as_warnings())
    if per_stem_duration_refine_stats:
        for label, drs in per_stem_duration_refine_stats.items():
            warnings.extend(_prefixed_warnings(label, drs.as_warnings()))
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
