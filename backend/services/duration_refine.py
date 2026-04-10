"""Per-note pitch-specific energy envelope refinement for note durations.

Computes a CQT spectrogram once for the whole audio file, then for each note
extracts the energy curve at the note's specific pitch bin.  The offset is
trimmed to the point where pitch-specific energy drops below a configurable
fraction of its within-note peak, plus a small tail allowance.

This complements any global-RMS gating (e.g. a coarse Pass 5 in a cleanup
pipeline) by being *pitch-aware*: a loud bass note won't keep an upper-register
note artificially long.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from backend.services.transcription_cleanup import NoteEvent

log = logging.getLogger(__name__)


@dataclass
class DurationRefineStats:
    total_notes: int = 0
    refined_count: int = 0
    mean_trim_sec: float = 0.0
    max_trim_sec: float = 0.0

    def as_warnings(self) -> list[str]:
        out: list[str] = []
        if self.refined_count:
            out.append(
                f"duration_refine: trimmed {self.refined_count}/{self.total_notes} notes "
                f"(mean={self.mean_trim_sec:.4f}s, max={self.max_trim_sec:.4f}s)"
            )
        return out


_CQT_FMIN_MIDI = 24  # CQT default fmin = C1 = MIDI 24


def refine_durations(
    events: list[NoteEvent],
    audio_path: Path,
    *,
    sr: int = 22050,
    hop_length: int = 256,
    floor_ratio: float = 0.15,
    tail_sec: float = 0.03,
    min_duration_sec: float = 0.03,
) -> tuple[list[NoteEvent], DurationRefineStats]:
    """Refine note offsets using per-pitch CQT energy envelopes.

    Returns (refined_events, stats).
    """
    stats = DurationRefineStats(total_notes=len(events))

    if not events:
        return list(events), stats

    try:
        import librosa  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError:
        log.warning("librosa/numpy not installed — skipping duration refinement")
        return list(events), stats

    try:
        y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    except Exception:
        log.warning("Could not load audio at %s — skipping duration refinement", audio_path, exc_info=True)
        return list(events), stats

    n_bins = 84  # 7 octaves, 12 bins/octave — covers MIDI 24..107
    try:
        C = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=n_bins, bins_per_octave=12))
    except Exception:
        log.warning("CQT computation failed — skipping duration refinement", exc_info=True)
        return list(events), stats

    frame_duration = hop_length / sr

    refined: list[NoteEvent] = []
    trims: list[float] = []

    for ev in events:
        start, end, pitch, amp, bends = ev
        cqt_bin = pitch - _CQT_FMIN_MIDI

        # Pitch outside CQT range — keep as-is
        if cqt_bin < 0 or cqt_bin >= n_bins:
            refined.append(ev)
            continue

        start_frame = max(0, min(int(start / frame_duration), C.shape[1] - 1))
        end_frame = max(start_frame + 1, min(int(end / frame_duration), C.shape[1]))

        energy = C[cqt_bin, start_frame:end_frame]

        if energy.size == 0:
            refined.append(ev)
            continue

        peak = float(np.max(energy))
        if peak <= 0:
            refined.append(ev)
            continue

        threshold = floor_ratio * peak

        # Walk backwards from end to find where energy is still above threshold
        decay_idx: int | None = None
        for idx in range(len(energy) - 1, -1, -1):
            if energy[idx] >= threshold:
                decay_idx = idx
                break

        if decay_idx is None:
            refined.append(ev)
            continue

        new_offset = start + (start_frame + decay_idx) * frame_duration + tail_sec

        # Enforce minimum duration
        if new_offset - start < min_duration_sec:
            new_offset = start + min_duration_sec

        # Only shorten, never extend
        if new_offset >= end:
            refined.append(ev)
            continue

        trim_amount = end - new_offset
        trims.append(trim_amount)
        refined.append((start, new_offset, pitch, amp, bends))

    stats.refined_count = len(trims)
    if trims:
        stats.mean_trim_sec = sum(trims) / len(trims)
        stats.max_trim_sec = max(trims)

    log.info(
        "duration_refine: refined %d / %d notes  mean_trim=%.4fs  max_trim=%.4fs",
        stats.refined_count, stats.total_notes, stats.mean_trim_sec, stats.max_trim_sec,
    )

    return refined, stats
