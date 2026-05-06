"""Derive a piecewise tempo map and downbeats from audio beat positions.

Basic Pitch is a polyphonic pitch tracker and does not estimate tempo, so
the transcribe stage otherwise collapses to a single global BPM anchor at
``t=0`` (either ``pretty_midi.estimate_tempo`` or a 120 BPM default). That
mismatches the real pulse of most recordings, and because
``ArrangeService`` runs ``sec_to_beat`` in beat space, small tempo errors
compound into noticeable quantization drift.

This module supports two beat-tracking backends:

* **Beat This!** (default) — CPJKU's MIT-licensed transformer beat tracker.
  Significantly more accurate than madmom on pop / hip-hop / K-pop with
  busy hi-hat patterns, and emits downbeats as a first-class output.
* **librosa** — ``librosa.beat.beat_track``, the legacy fallback. Beats
  only; downbeats are returned as an empty list.

The backend is selected via ``Settings.beat_tracker`` (``"beat_this"``,
``"librosa"``, or ``"auto"``).  ``"auto"`` tries Beat This! first, then
falls back to librosa.

Beat instants are converted into a list of ``TempoMapEntry`` — one anchor
per beat, with each segment's BPM = ``60 / Δt`` between adjacent beats.
That gives ``sec_to_beat`` a piecewise-linear map that follows the
waveform pulse.

Downbeat instants (where available) flow onto ``HarmonicAnalysis.downbeats``
in seconds and from there onto ``ScoreMetadata.downbeats``, ultimately
emitted as MIDI Cue Point text events by ``midi_render`` so the engraver
can lock bar lines to the perceived pulse.

Everything here is best-effort: if the chosen backend is missing, the
audio is too short, or beat tracking fails, callers get ``None`` and are
expected to fall back to the single-anchor map already built from the
model.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from backend.contracts import TempoMapEntry

log = logging.getLogger(__name__)

# Clamp inter-beat bpm to a sane band so a single missed / doubled beat
# can't produce a pathological segment that blows up sec_to_beat downstream.
_MIN_BPM = 30.0
_MAX_BPM = 300.0
_MIN_DT_SEC = 1e-3
_MIN_DURATION_SEC = 0.5


@dataclass
class BeatTrackResult:
    """Beats + downbeats from an audio beat tracker.

    ``beats`` is the full beat grid in seconds; ``downbeats`` is the
    subset that fall on bar starts. ``downbeats`` is empty when the
    backend doesn't expose downbeat info (librosa) or the track is
    too short / ambiguous to disambiguate the meter.
    """
    beats: list[float] = field(default_factory=list)
    downbeats: list[float] = field(default_factory=list)


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


# ── Backend: Beat This! (CPJKU MIT-licensed transformer) ───────────────


def _beat_this_track(y, sr: int) -> BeatTrackResult | None:
    """Run Beat This! on a waveform.

    Returns ``BeatTrackResult`` with sorted beats + downbeats in seconds,
    or ``None`` if the package is unavailable or inference fails.
    """
    try:
        # ``Audio2Beats`` takes a numpy array directly — preferred when
        # we already have the waveform in memory (transcribe pipeline).
        from beat_this.inference import Audio2Beats  # noqa: PLC0415
    except ImportError:
        log.debug("beat_this unavailable; skipping Beat This! backend")
        return None

    try:
        import numpy as np  # noqa: PLC0415

        # Beat This! ships its own resampling; pass the waveform as-is.
        # ``dbn=False`` keeps inference inside the trained transformer
        # (no madmom DBN post-processor) — cleaner license posture.
        # ``device="cpu"`` is the safe default for Cloud Run; callers
        # can override via env var if a GPU is wired up.
        a2b = Audio2Beats(checkpoint_path="final0", device="cpu", dbn=False)
        # The model expects float32 mono; coerce so callers don't have to.
        wav = np.ascontiguousarray(np.atleast_1d(y).astype(np.float32, copy=False))
        if wav.ndim > 1:
            wav = wav.mean(axis=0).astype(np.float32, copy=False)
        beats, downbeats = a2b(wav, sr)
    except Exception as exc:  # noqa: BLE001 — never let beat tracking sink transcribe
        log.warning("Beat This! beat tracking failed: %s", exc)
        return None

    try:
        import numpy as np  # noqa: PLC0415
        beat_list = sorted(float(t) for t in np.atleast_1d(beats).tolist() if t >= 0.0)
        down_list = sorted(float(t) for t in np.atleast_1d(downbeats).tolist() if t >= 0.0)
    except Exception as exc:  # noqa: BLE001
        log.warning("Beat This! returned non-numeric output: %s", exc)
        return None

    if not beat_list:
        log.debug("Beat This! returned no beats")
        return None

    return BeatTrackResult(beats=beat_list, downbeats=down_list)


# ── Backend: librosa (legacy fallback) ─────────────────────────────────


def _librosa_track(y, sr: int) -> BeatTrackResult | None:
    """Run ``librosa.beat.beat_track`` on a waveform.

    librosa exposes only beats, not downbeats — the returned
    ``BeatTrackResult`` always has ``downbeats=[]``.
    """
    try:
        import librosa  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError:
        log.debug("librosa/numpy unavailable; cannot use librosa beat tracker")
        return None

    try:
        _tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)
        beat_list = [float(t) for t in np.atleast_1d(beat_times).tolist()]
        if not beat_list:
            log.debug("librosa beat tracker returned no beats")
            return None
        return BeatTrackResult(beats=sorted(beat_list), downbeats=[])
    except Exception as exc:  # noqa: BLE001
        log.warning("librosa.beat.beat_track failed: %s", exc)
        return None


# ── Public API ──────────────────────────────────────────────────────────


def beats_from_audio_path(
    path: Path,
    *,
    sr: int = 22_050,
    preloaded_audio: tuple | None = None,
) -> BeatTrackResult | None:
    """Run beat tracking and return beats + downbeats in seconds.

    Backend selected by ``Settings.beat_tracker``:

    * ``"beat_this"`` (default) — Beat This! transformer; falls back to
      librosa if the package isn't installed.
    * ``"librosa"`` — librosa only (downbeats always empty).
    * ``"auto"`` — try Beat This! first, then librosa.

    When ``preloaded_audio`` is a ``(y, sr)`` tuple the ``librosa.load``
    call is skipped, reusing a waveform already in memory.

    Returns ``None`` when no backend produced beats (callers fall back
    to the model-derived single-tempo map).
    """
    from backend.config import settings  # noqa: PLC0415 — avoid circular import

    try:
        import librosa as _librosa  # noqa: PLC0415
    except ImportError:
        log.debug("librosa unavailable; skipping audio beat tracking")
        return None

    if preloaded_audio is not None:
        y, file_sr = preloaded_audio
    else:
        try:
            y, file_sr = _librosa.load(str(path), sr=sr, mono=True)
        except Exception as exc:  # noqa: BLE001 — bad bytes shouldn't crash the worker
            log.warning("librosa.load failed for %s: %s", path, exc)
            return None

    duration = float(len(y) / file_sr) if file_sr else 0.0
    if duration < _MIN_DURATION_SEC:
        log.debug("audio too short for beat tracking (%.2fs)", duration)
        return None

    backend = settings.beat_tracker.lower()
    result: BeatTrackResult | None = None

    if backend == "beat_this":
        result = _beat_this_track(y, file_sr)
        if result is None:
            log.info("Beat This! unavailable or failed; falling back to librosa")
            result = _librosa_track(y, file_sr)
    elif backend == "librosa":
        result = _librosa_track(y, file_sr)
    elif backend == "auto":
        result = _beat_this_track(y, file_sr)
        if result is None:
            result = _librosa_track(y, file_sr)
    else:
        log.warning("unknown beat_tracker setting %r; using librosa", backend)
        result = _librosa_track(y, file_sr)

    if result is None or not result.beats:
        log.debug("beat tracker returned no beats for %s", path)
        return None

    return result


def tempo_map_from_audio_path(
    path: Path,
    *,
    sr: int = 22_050,
    preloaded_audio: tuple | None = None,
) -> list[TempoMapEntry] | None:
    """Run beat tracking on an audio file and return a piecewise tempo map.

    Thin wrapper over :func:`beats_from_audio_path` that drops the
    downbeats. Use the underlying function directly when you also need
    bar-line positions (Phase 2+: midi_render emits them as Cue Points).
    """
    result = beats_from_audio_path(path, sr=sr, preloaded_audio=preloaded_audio)
    if result is None:
        return None

    fallback_bpm = _estimate_fallback_bpm(path, preloaded_audio=preloaded_audio)
    return build_tempo_map_from_beat_times(result.beats, fallback_bpm=fallback_bpm)


def tempo_map_and_downbeats_from_audio_path(
    path: Path,
    *,
    sr: int = 22_050,
    preloaded_audio: tuple | None = None,
) -> tuple[list[TempoMapEntry], list[float]] | None:
    """Return ``(tempo_map, downbeats_sec)`` from a single beat-tracker run.

    Preferred over calling :func:`tempo_map_from_audio_path` plus a
    separate downbeat lookup — Beat This! inference is the slow step,
    so running it once and reusing the result is ~2× faster.
    """
    result = beats_from_audio_path(path, sr=sr, preloaded_audio=preloaded_audio)
    if result is None:
        return None

    fallback_bpm = _estimate_fallback_bpm(path, preloaded_audio=preloaded_audio)
    tempo_map = build_tempo_map_from_beat_times(result.beats, fallback_bpm=fallback_bpm)
    return tempo_map, result.downbeats


def _estimate_fallback_bpm(
    path: Path,
    *,
    preloaded_audio: tuple | None = None,
) -> float:
    """Best-effort scalar BPM via librosa, used for the single-anchor fallback."""
    try:
        import librosa  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError:
        return 120.0

    if preloaded_audio is not None:
        y, file_sr = preloaded_audio
    else:
        try:
            y, file_sr = librosa.load(str(path), sr=22_050, mono=True)
        except Exception:  # noqa: BLE001
            return 120.0

    try:
        tempo_arr = librosa.beat.beat_track(y=y, sr=file_sr, hop_length=512)[0]
        tempo_scalar = float(np.atleast_1d(tempo_arr).ravel()[0])
        if not math.isfinite(tempo_scalar) or not (_MIN_BPM <= tempo_scalar <= _MAX_BPM):
            return 120.0
        return tempo_scalar
    except Exception:  # noqa: BLE001
        return 120.0
