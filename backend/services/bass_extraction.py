"""Phase 3 post-processing — waveform-guided bass-line extraction.

Mirrors :mod:`backend.services.melody_extraction` but operates on the
low-register slice of Basic Pitch's ``model_output["contour"]`` matrix.
The same Viterbi F0 tracer (``_trace_f0_contour``) is reused with a
different band mask, stricter transition penalty, and bass-tuned
thresholds — real bass lines are more stepwise than leads and rarely
leap far in a single 11.6 ms frame.

The pipeline is:

    cleaned_events
        → extract_melody()        -> (melody, non_melody, …)
        → extract_bass(non_melody) -> (bass,   remaining,  …)
        → remaining is tagged CHORDS by the transcribe wiring

No new model weights, no new dependencies — everything here runs on
``numpy`` which is already pulled in by the ``basic-pitch`` extra.

This module deliberately imports the private tracer helpers from
``melody_extraction`` rather than duplicating them: they're already
voice-agnostic (band mask + transition weight are parameters), and a
separate ``_viterbi_tracer`` module would just spread the Viterbi
implementation across two places for no benefit.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from backend.services.melody_extraction import (
    N_CONTOUR_BINS,
    _split_notes_by_path_agreement,
    _trace_f0_contour,
    midi_to_bin,
)
from backend.services.transcription_cleanup import NoteEvent

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults — tuned for real bass-line material. Wider voicing floor than
# melody because low-register contour bins are typically dimmer, and a
# stronger transition weight because bass rarely leaps.
# ---------------------------------------------------------------------------

DEFAULT_BASS_LOW_MIDI = 28            # E1
DEFAULT_BASS_HIGH_MIDI = 55           # G3  (overlaps melody_low_midi=55)
DEFAULT_BASS_VOICING_FLOOR = 0.12
DEFAULT_BASS_TRANSITION_WEIGHT = 0.40
DEFAULT_BASS_MAX_TRANSITION_BINS = 9  # ≈ 3 semitones / frame
DEFAULT_BASS_VOICED_ENTER_COST = 1.0
DEFAULT_BASS_UNVOICED_ENTER_COST = 1.0
DEFAULT_BASS_MATCH_TOL_SEMITONES = 1.0
DEFAULT_BASS_MATCH_FRACTION = 0.55

# Bass and melody bands intentionally share their boundary bin (G3 / 55).
# If a note straddles both, the first extractor to claim it wins; bass
# runs *after* melody, so this degenerates to "melody always keeps its
# G3 claim". That matches the usual musical case — melody leads routinely
# dip to G3, bass lines rarely climb there.


# ---------------------------------------------------------------------------
# Stats — telemetry surfaced back to the caller for QualitySignal warnings
# ---------------------------------------------------------------------------

@dataclass
class BassExtractionStats:
    """Per-run summary of what the bass extractor decided."""
    input_note_count: int = 0
    bass_note_count: int = 0
    remaining_note_count: int = 0
    voiced_frame_fraction: float = 0.0
    warnings: list[str] = field(default_factory=list)
    skipped: bool = False

    def as_warnings(self) -> list[str]:
        if self.skipped:
            return ["bass extraction skipped (numpy or contour unavailable)"]
        out: list[str] = []
        if self.bass_note_count or self.remaining_note_count:
            out.append(
                f"bass split: {self.bass_note_count} bass / "
                f"{self.remaining_note_count} chord notes "
                f"({self.voiced_frame_fraction * 100:.0f}% voiced)"
            )
        out.extend(self.warnings)
        return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_bass(
    contour: Any,  # np.ndarray (frames, 264) or None
    note_events: list[NoteEvent],
    *,
    bass_low_midi: int = DEFAULT_BASS_LOW_MIDI,
    bass_high_midi: int = DEFAULT_BASS_HIGH_MIDI,
    voicing_floor: float = DEFAULT_BASS_VOICING_FLOOR,
    transition_weight: float = DEFAULT_BASS_TRANSITION_WEIGHT,
    max_transition_bins: int = DEFAULT_BASS_MAX_TRANSITION_BINS,
    voiced_enter_cost: float = DEFAULT_BASS_VOICED_ENTER_COST,
    unvoiced_enter_cost: float = DEFAULT_BASS_UNVOICED_ENTER_COST,
    match_tolerance_semitones: float = DEFAULT_BASS_MATCH_TOL_SEMITONES,
    match_fraction: float = DEFAULT_BASS_MATCH_FRACTION,
    frame_rate_hz: float | None = None,
) -> tuple[list[NoteEvent], list[NoteEvent], BassExtractionStats]:
    """Run Phase 3 bass extraction over a list of candidate note events.

    ``note_events`` should be whatever the preceding melody extractor
    left un-tagged (i.e. ``chord_events`` from ``extract_melody``). The
    function runs the Viterbi F0 tracer over the low-register slice of
    the contour matrix and tags each input note as ``bass`` iff its
    pitch + time window agrees with the traced path.

    Returns ``(bass_events, remaining_events, stats)``. Remaining
    events are the ones that didn't match the bass path — in the
    transcribe wiring they become the CHORDS bucket.

    When numpy is unavailable, the contour matrix is missing /
    malformed, or the tracer raises, we return
    ``([], note_events, stats)`` with ``stats.skipped = True`` so the
    caller can fall back cleanly.
    """
    # Late import to avoid circular dep with melody_extraction's own
    # FRAME_RATE_HZ re-export (it's the same value, we just don't want
    # bass_extraction to pin itself to melody_extraction at import time).
    from backend.services.melody_extraction import FRAME_RATE_HZ  # noqa: PLC0415

    if frame_rate_hz is None:
        frame_rate_hz = FRAME_RATE_HZ

    stats = BassExtractionStats(input_note_count=len(note_events))

    if contour is None:
        stats.skipped = True
        return [], list(note_events), stats

    try:
        import numpy as np  # noqa: PLC0415
    except ImportError:
        stats.skipped = True
        return [], list(note_events), stats

    contour_arr = np.asarray(contour)
    if contour_arr.ndim != 2 or contour_arr.shape[1] != N_CONTOUR_BINS:
        stats.warnings.append(
            f"contour has unexpected shape {contour_arr.shape}; skipping"
        )
        stats.skipped = True
        return [], list(note_events), stats
    if contour_arr.shape[0] < 2:
        stats.skipped = True
        return [], list(note_events), stats

    low_bin = midi_to_bin(bass_low_midi)
    high_bin = midi_to_bin(bass_high_midi)
    if low_bin >= high_bin:
        stats.warnings.append(
            f"invalid bass band [{bass_low_midi}, {bass_high_midi}]"
        )
        stats.skipped = True
        return [], list(note_events), stats

    try:
        path = _trace_f0_contour(
            contour_arr,
            low_bin=low_bin,
            high_bin=high_bin,
            voicing_floor=voicing_floor,
            transition_weight=transition_weight,
            max_transition_bins=max_transition_bins,
            voiced_enter_cost=voiced_enter_cost,
            unvoiced_enter_cost=unvoiced_enter_cost,
        )
    except Exception as exc:  # noqa: BLE001 — tracer must never sink transcribe
        log.warning("bass Viterbi tracer failed: %s", exc)
        stats.warnings.append(f"viterbi failed: {exc}")
        stats.skipped = True
        return [], list(note_events), stats

    voiced_frac = float((path >= 0).mean()) if len(path) else 0.0
    stats.voiced_frame_fraction = voiced_frac

    bass, remaining = _split_notes_by_path_agreement(
        note_events,
        path,
        low_midi=bass_low_midi,
        high_midi=bass_high_midi,
        frame_rate_hz=frame_rate_hz,
        match_tolerance_semitones=match_tolerance_semitones,
        match_fraction=match_fraction,
    )

    stats.bass_note_count = len(bass)
    stats.remaining_note_count = len(remaining)
    log.debug(
        "bass extraction: %d notes → %d bass / %d remaining (%.0f%% voiced)",
        len(note_events),
        len(bass),
        len(remaining),
        voiced_frac * 100.0,
    )
    return bass, remaining, stats
