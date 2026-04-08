"""Derive a piecewise tempo map from audio beat positions.

``TempoMapEntry`` segments use the contract in ``backend.contracts``:
``sec_to_beat`` walks anchors in time order and extrapolates with each
segment's ``bpm``. We build one segment per inter-beat interval so local
tempo matches the waveform (see :func:`build_tempo_map_from_beat_times`).
"""
from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path

from backend.contracts import TempoMapEntry

log = logging.getLogger(__name__)


def build_tempo_map_from_beat_times(
    beat_times_sec: Sequence[float],
    *,
    duration_sec: float,
    fallback_bpm: float = 120.0,
) -> list[TempoMapEntry]:
    """Turn monotonic beat instants (seconds) into a valid ``tempo_map``.

    Each segment [t_i, t_{i+1}) uses bpm = 60 / (t_{i+1} - t_i). A final
    anchor at the last beat repeats the last inter-beat tempo so times
    past the last detection still convert sensibly.
    """
    beats = [float(t) for t in beat_times_sec if t >= 0.0]
    beats.sort()
    if len(beats) < 2:
        bpm = max(fallback_bpm, 30.0)
        return [TempoMapEntry(time_sec=0.0, beat=0.0, bpm=bpm)]

    entries: list[TempoMapEntry] = []
    for i in range(len(beats) - 1):
        t0, t1 = beats[i], beats[i + 1]
        dt = max(t1 - t0, 1e-3)
        bpm = min(max(60.0 / dt, 30.0), 480.0)
        entries.append(TempoMapEntry(time_sec=t0, beat=float(i), bpm=bpm))

    t_last = beats[-1]
    dt_prev = max(beats[-1] - beats[-2], 1e-3)
    bpm_tail = min(max(60.0 / dt_prev, 30.0), 480.0)
    last_idx = float(len(beats) - 1)
    entries.append(TempoMapEntry(time_sec=t_last, beat=last_idx, bpm=bpm_tail))

    if duration_sec > 0 and t_last > duration_sec + 1.0:
        log.debug("Last beat after declared duration; trimming not applied")

    return entries


def tempo_map_from_audio_path(path: Path, *, sr: int = 22_050) -> list[TempoMapEntry] | None:
    """Run beat tracking on a WAV (or librosa-readable) file.

    Returns ``None`` if ``librosa`` is missing or beat tracking fails, so
    callers can fall back to MT3 / stub tempo.
    """
    try:
        import librosa  # noqa: PLC0415 — optional; ships with mt3 extra
        import numpy as np  # noqa: PLC0415
    except ImportError:
        log.debug("librosa/numpy unavailable; skipping audio beat map")
        return None

    try:
        y, file_sr = librosa.load(str(path), sr=sr, mono=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("librosa.load failed for %s: %s", path, exc)
        return None

    duration = float(len(y) / file_sr) if file_sr else 0.0
    if duration < 0.5:
        return None

    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=file_sr, hop_length=512)
    except Exception as exc:  # noqa: BLE001
        log.warning("beat_track failed for %s: %s", path, exc)
        return None

    beat_times = librosa.frames_to_time(beat_frames, sr=file_sr, hop_length=512)
    beat_list = [float(t) for t in np.atleast_1d(beat_times).tolist()]
    if not beat_list:
        return None

    tempo_scalar = float(np.atleast_1d(tempo).ravel()[0])
    if not math.isfinite(tempo_scalar) or not (30.0 <= tempo_scalar <= 480.0):
        tempo_scalar = 120.0

    return build_tempo_map_from_beat_times(
        beat_list,
        duration_sec=duration,
        fallback_bpm=tempo_scalar,
    )
