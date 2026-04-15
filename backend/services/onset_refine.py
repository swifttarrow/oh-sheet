"""Post-cleanup onset sharpening via spectral flux.

Basic Pitch's frame resolution is ~23 ms (hop_length=512, sr=22050), and
the cleanup passes in :mod:`transcription_cleanup` may shift note
boundaries further.  This module refines note onsets by aligning them to
nearby peaks in a higher-resolution spectral onset-strength function
computed by ``librosa.onset.onset_strength`` with ``hop_length=256``
(~11.6 ms resolution — 2x better than Basic Pitch's internal grid).

The refinement is a pure seconds-domain operation that runs **after**
cleanup and **before** the note events are converted to :class:`Note`
contract objects, so it slots cleanly between Phase 1 post-processing and
the arrange stage's seconds-to-beats conversion.

``librosa`` and ``scipy`` are imported lazily — they ship with the
``basic-pitch`` extra but are not required by the dev install, so CI
tests that mock the imports work without either package.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from backend.services.transcription_cleanup import NoteEvent

log = logging.getLogger(__name__)


@dataclass
class OnsetRefineStats:
    """Telemetry for one onset-refinement pass."""
    total_notes: int = 0
    refined_count: int = 0
    mean_shift_sec: float = 0.0
    max_shift_sec: float = 0.0
    skipped: bool = False
    warnings: list[str] = field(default_factory=list)

    def as_warnings(self) -> list[str]:
        """One-line human summary entries for the QualitySignal."""
        out: list[str] = []
        if self.skipped:
            out.extend(self.warnings)
            return out
        if self.refined_count:
            out.append(
                f"onset-refine: shifted {self.refined_count}/{self.total_notes} "
                f"onsets (mean={self.mean_shift_sec * 1000:.1f}ms, "
                f"max={self.max_shift_sec * 1000:.1f}ms)"
            )
        out.extend(self.warnings)
        return out


def refine_onsets(
    events: list[NoteEvent],
    audio_path: Path,
    *,
    sr: int = 22050,
    hop_length: int = 256,
    max_shift_sec: float = 0.05,
    preloaded_audio: tuple | None = None,
) -> tuple[list[NoteEvent], OnsetRefineStats]:
    """Sharpen note onsets by snapping them to spectral onset-strength peaks.

    For each note event, the nearest peak in the onset-strength function
    within ``±max_shift_sec`` of the current onset is found.  If a peak
    exists, the onset is moved to the peak time; otherwise it stays put.
    The onset is never moved past ``end - 0.01`` to guarantee a minimum
    10 ms note duration.

    Parameters
    ----------
    events:
        Cleaned note events from :func:`cleanup_note_events` or
        :func:`cleanup_for_role`.
    audio_path:
        Path to the audio file (original mix or a Demucs stem).
    sr:
        Sample rate for loading audio (default 22050, matching Basic Pitch).
    hop_length:
        Hop length for onset strength computation.  256 gives ~11.6 ms
        resolution at 22050 Hz — twice as fine as Basic Pitch's 512.
    max_shift_sec:
        Maximum allowed onset adjustment in seconds.
    preloaded_audio:
        Optional ``(y, sr)`` tuple of pre-loaded mono audio. When provided,
        skips ``librosa.load`` and uses the given waveform directly. The
        ``sr`` element overrides the ``sr`` parameter.

    Returns
    -------
    tuple[list[NoteEvent], OnsetRefineStats]
        The (possibly adjusted) events and a stats summary.
    """
    stats = OnsetRefineStats(total_notes=len(events))

    if not events:
        return [], stats

    if not audio_path.exists():
        stats.skipped = True
        stats.warnings.append(
            f"onset-refine: skipped — audio file not found: {audio_path}"
        )
        return list(events), stats

    # Lazy imports — librosa and scipy are optional deps.
    try:
        import librosa  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        from scipy.signal import find_peaks  # noqa: PLC0415
    except ImportError as exc:
        stats.skipped = True
        stats.warnings.append(f"onset-refine: skipped — missing dep: {exc}")
        return list(events), stats

    # Load audio and compute onset strength — once for all notes.
    try:
        if preloaded_audio is not None:
            y, actual_sr = preloaded_audio
        else:
            y, actual_sr = librosa.load(str(audio_path), sr=sr, mono=True)
        if len(y) == 0:
            stats.skipped = True
            stats.warnings.append("onset-refine: skipped — empty audio")
            return list(events), stats

        odf = librosa.onset.onset_strength(y=y, sr=actual_sr, hop_length=hop_length)
        odf_times = librosa.frames_to_time(
            np.arange(len(odf)), sr=actual_sr, hop_length=hop_length,
        )
    except Exception as exc:  # noqa: BLE001 — never let onset refine sink transcribe
        log.warning("onset-refine: audio load / ODF computation failed: %s", exc)
        stats.skipped = True
        stats.warnings.append(f"onset-refine: skipped — ODF failed: {exc}")
        return list(events), stats

    # Find peaks in the onset strength function.
    try:
        median_odf = float(np.median(odf))
        peak_indices, _ = find_peaks(odf, height=median_odf)
        if len(peak_indices) == 0:
            stats.skipped = True
            stats.warnings.append("onset-refine: skipped — no ODF peaks found")
            return list(events), stats
        peak_times = odf_times[peak_indices]
    except Exception as exc:  # noqa: BLE001
        log.warning("onset-refine: peak detection failed: %s", exc)
        stats.skipped = True
        stats.warnings.append(f"onset-refine: skipped — peak detection failed: {exc}")
        return list(events), stats

    # For efficient nearest-peak lookup, use a sorted array + searchsorted.
    peak_times_sorted = np.sort(peak_times)

    refined: list[NoteEvent] = []
    shifts: list[float] = []

    for ev in events:
        start, end, pitch, amp, bends = ev

        # Find the nearest peak to this onset.
        idx = int(np.searchsorted(peak_times_sorted, start))
        best_peak: float | None = None
        best_dist = max_shift_sec + 1.0  # sentinel > max_shift

        # Check the candidate at idx and idx-1 (the two nearest).
        for candidate_idx in (idx - 1, idx):
            if 0 <= candidate_idx < len(peak_times_sorted):
                pt = float(peak_times_sorted[candidate_idx])
                dist = abs(pt - start)
                if dist <= max_shift_sec and dist < best_dist:
                    best_dist = dist
                    best_peak = pt

        if best_peak is not None:
            # Clamp: onset must not move past end - 0.01.
            new_start = min(best_peak, end - 0.01)
            shift = abs(new_start - start)
            if shift > 1e-6:  # only count meaningful shifts
                shifts.append(shift)
                refined.append((new_start, end, pitch, amp, bends))
            else:
                refined.append(ev)
        else:
            refined.append(ev)

    stats.refined_count = len(shifts)
    if shifts:
        stats.mean_shift_sec = sum(shifts) / len(shifts)
        stats.max_shift_sec = max(shifts)

    log.debug(
        "onset-refine: %d/%d onsets shifted (mean=%.1fms, max=%.1fms)",
        stats.refined_count,
        stats.total_notes,
        stats.mean_shift_sec * 1000,
        stats.max_shift_sec * 1000,
    )
    return refined, stats
