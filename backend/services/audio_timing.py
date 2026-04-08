"""Derive a piecewise tempo map from audio beat positions.

Basic Pitch is a polyphonic pitch tracker and does not estimate tempo, so
the transcribe stage otherwise collapses to a single global BPM anchor at
``t=0`` (either ``pretty_midi.estimate_tempo`` or a 120 BPM default). That
mismatches the real pulse of most recordings, and because
``ArrangeService`` runs ``sec_to_beat`` in beat space, small tempo errors
compound into noticeable quantization drift.

This module runs ``librosa.beat.beat_track`` on the waveform and converts
the detected beat instants into a list of ``TempoMapEntry`` — one anchor
per beat, with each segment's BPM = ``60 / Δt`` between adjacent beats.
That gives ``sec_to_beat`` a piecewise-linear map that follows the
waveform pulse.

Everything here is best-effort: if ``librosa`` is missing, the audio is
too short, or beat tracking fails, callers get ``None`` and are expected
to fall back to the single-anchor map already built from the model.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path

from backend.contracts import TempoMapEntry

log = logging.getLogger(__name__)

# Clamp inter-beat bpm to a sane band so a single missed / doubled beat
# can't produce a pathological segment that blows up sec_to_beat downstream.
_MIN_BPM = 30.0
_MAX_BPM = 300.0
_MIN_DT_SEC = 1e-3
_MIN_DURATION_SEC = 0.5


def build_tempo_map_from_beat_times(
    beat_times_sec: Sequence[float],
    *,
    fallback_bpm: float = 120.0,
) -> list[TempoMapEntry]:
    """Turn monotonic beat instants (seconds) into a valid ``tempo_map``.

    Each segment ``[t_i, t_{i+1})`` uses ``bpm = 60 / (t_{i+1} - t_i)``, and
    the final anchor at ``t_last`` repeats the previous inter-beat tempo so
    ``sec_to_beat`` still converts times past the last detection sensibly.

    If there are fewer than two beats (nothing to measure Δt from), we
    emit a single anchor at ``t=0`` using ``fallback_bpm`` — the same
    shape the caller would've built without beat tracking. Notes that
    occur before the first detected beat extrapolate backwards via
    ``sec_to_beat`` using the first segment's BPM (may produce small
    negative beat values, which arrange tolerates).
    """
    beats = sorted(float(t) for t in beat_times_sec if t >= 0.0)
    if len(beats) < 2:
        bpm = min(max(fallback_bpm, _MIN_BPM), _MAX_BPM)
        return [TempoMapEntry(time_sec=0.0, beat=0.0, bpm=bpm)]

    entries: list[TempoMapEntry] = []
    for i in range(len(beats) - 1):
        t0, t1 = beats[i], beats[i + 1]
        dt = max(t1 - t0, _MIN_DT_SEC)
        bpm = min(max(60.0 / dt, _MIN_BPM), _MAX_BPM)
        entries.append(TempoMapEntry(time_sec=t0, beat=float(i), bpm=bpm))

    # Tail anchor: repeat the final inter-beat tempo so the map keeps
    # extending past the last detected beat.
    dt_tail = max(beats[-1] - beats[-2], _MIN_DT_SEC)
    bpm_tail = min(max(60.0 / dt_tail, _MIN_BPM), _MAX_BPM)
    entries.append(
        TempoMapEntry(
            time_sec=beats[-1],
            beat=float(len(beats) - 1),
            bpm=bpm_tail,
        )
    )
    return entries


def tempo_map_from_audio_path(
    path: Path,
    *,
    sr: int = 22_050,
) -> list[TempoMapEntry] | None:
    """Run beat tracking on an audio file and return a piecewise tempo map.

    Returns ``None`` on any failure (librosa missing, audio unreadable,
    beat tracking crash, audio too short, no beats detected) so callers
    can fall back to the model-derived single-tempo map.
    """
    try:
        import librosa  # noqa: PLC0415 — optional, ships with basic-pitch extra
        import numpy as np  # noqa: PLC0415
    except ImportError:
        log.debug("librosa/numpy unavailable; skipping audio beat tracking")
        return None

    try:
        y, file_sr = librosa.load(str(path), sr=sr, mono=True)
    except Exception as exc:  # noqa: BLE001 — bad bytes shouldn't crash the worker
        log.warning("librosa.load failed for %s: %s", path, exc)
        return None

    duration = float(len(y) / file_sr) if file_sr else 0.0
    if duration < _MIN_DURATION_SEC:
        log.debug("audio too short for beat tracking (%.2fs)", duration)
        return None

    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=file_sr, hop_length=512)
    except Exception as exc:  # noqa: BLE001
        log.warning("librosa.beat.beat_track failed for %s: %s", path, exc)
        return None

    beat_times = librosa.frames_to_time(beat_frames, sr=file_sr, hop_length=512)
    beat_list = [float(t) for t in np.atleast_1d(beat_times).tolist()]
    if not beat_list:
        log.debug("beat tracker returned no beats for %s", path)
        return None

    tempo_scalar = float(np.atleast_1d(tempo).ravel()[0])
    if not math.isfinite(tempo_scalar) or not (_MIN_BPM <= tempo_scalar <= _MAX_BPM):
        tempo_scalar = 120.0

    return build_tempo_map_from_beat_times(beat_list, fallback_bpm=tempo_scalar)
