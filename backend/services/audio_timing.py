"""Derive a piecewise tempo map from audio beat positions.

Basic Pitch is a polyphonic pitch tracker and does not estimate tempo, so
the transcribe stage otherwise collapses to a single global BPM anchor at
``t=0`` (either ``pretty_midi.estimate_tempo`` or a 120 BPM default). That
mismatches the real pulse of most recordings, and because
``ArrangeService`` runs ``sec_to_beat`` in beat space, small tempo errors
compound into noticeable quantization drift.

This module supports two beat-tracking backends:

* **madmom** (default) — ``DBNBeatProcessor`` + ``RNNBeatProcessor``.
  Significantly more robust than librosa for variable-tempo music.
* **librosa** — ``librosa.beat.beat_track``, the legacy fallback.

The backend is selected via ``Settings.beat_tracker`` (``"madmom"``,
``"librosa"``, or ``"auto"``).  ``"auto"`` tries madmom first, then
falls back to librosa.

Beat instants are converted into a list of ``TempoMapEntry`` — one anchor
per beat, with each segment's BPM = ``60 / Δt`` between adjacent beats.
That gives ``sec_to_beat`` a piecewise-linear map that follows the
waveform pulse.

Everything here is best-effort: if the chosen backend is missing, the
audio is too short, or beat tracking fails, callers get ``None`` and are
expected to fall back to the single-anchor map already built from the
model.
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

# madmom's DBNBeatProcessor expects audio at 44100 Hz by default.
_MADMOM_SR = 44_100


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


# ── Backend: madmom (DBNBeatProcessor) ──────────────────────────────────


def _madmom_beat_track(y, sr: int) -> list[float] | None:
    """Run madmom RNNBeatProcessor + DBNBeatProcessor on a waveform.

    Returns sorted list of beat times in seconds, or ``None`` on failure.
    """
    try:
        from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor  # noqa: PLC0415
    except ImportError:
        log.debug("madmom unavailable; cannot use DBNBeatProcessor")
        return None

    try:
        import numpy as np  # noqa: PLC0415

        # Resample to 44100 Hz if needed — DBNBeatProcessor's default.
        if sr != _MADMOM_SR:
            try:
                import resampy  # noqa: PLC0415
                y = resampy.resample(y, sr, _MADMOM_SR)
            except ImportError:
                import librosa  # noqa: PLC0415
                y = librosa.resample(y, orig_sr=sr, target_sr=_MADMOM_SR)

        # RNNBeatProcessor expects a file path or Signal object.
        import tempfile  # noqa: PLC0415

        import soundfile as sf  # noqa: PLC0415

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            sf.write(tmp.name, y, _MADMOM_SR)
            tmp.close()
            rnn_processor = RNNBeatProcessor()
            activations = rnn_processor(tmp.name)
        finally:
            Path(tmp.name).unlink(missing_ok=True)

        dbn_processor = DBNBeatTrackingProcessor(fps=100)
        beat_times = dbn_processor(activations)

        beat_list = sorted(float(t) for t in np.atleast_1d(beat_times).tolist())
        if not beat_list:
            log.debug("madmom returned no beats")
            return None

        return beat_list

    except Exception as exc:  # noqa: BLE001
        log.warning("madmom beat tracking failed: %s", exc)
        return None


# ── Backend: librosa ────────────────────────────────────────────────────


def _librosa_beat_track(y, sr: int) -> list[float] | None:
    """Run librosa.beat.beat_track on a waveform.

    Returns sorted list of beat times in seconds, or ``None`` on failure.
    """
    try:
        import librosa  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError:
        log.debug("librosa/numpy unavailable; cannot use librosa beat tracker")
        return None

    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)
        beat_list = [float(t) for t in np.atleast_1d(beat_times).tolist()]
        if not beat_list:
            log.debug("librosa beat tracker returned no beats")
            return None
        return beat_list
    except Exception as exc:  # noqa: BLE001
        log.warning("librosa.beat.beat_track failed: %s", exc)
        return None


# ── Public API ──────────────────────────────────────────────────────────


def tempo_map_from_audio_path(
    path: Path,
    *,
    sr: int = 22_050,
    preloaded_audio: tuple | None = None,
) -> list[TempoMapEntry] | None:
    """Run beat tracking on an audio file and return a piecewise tempo map.

    The beat-tracking backend is selected by ``Settings.beat_tracker``:

    * ``"madmom"`` (default) — use madmom, fall back to librosa if unavailable.
    * ``"librosa"`` — use librosa only (legacy behaviour).
    * ``"auto"`` — try madmom first, then librosa.

    When ``preloaded_audio`` is a ``(y, sr)`` tuple the ``librosa.load``
    call is skipped, reusing a waveform already in memory.

    Returns ``None`` on any failure so callers can fall back to the
    model-derived single-tempo map.
    """
    from backend.config import settings  # noqa: PLC0415 — avoid circular import

    try:
        import librosa as _librosa  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError:
        log.debug("librosa/numpy unavailable; skipping audio beat tracking")
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
    beat_list: list[float] | None = None

    if backend == "madmom":
        beat_list = _madmom_beat_track(y, file_sr)
        if beat_list is None:
            log.info("madmom unavailable or failed; falling back to librosa")
            beat_list = _librosa_beat_track(y, file_sr)
    elif backend == "librosa":
        beat_list = _librosa_beat_track(y, file_sr)
    elif backend == "auto":
        beat_list = _madmom_beat_track(y, file_sr)
        if beat_list is None:
            beat_list = _librosa_beat_track(y, file_sr)
    else:
        log.warning("unknown beat_tracker setting %r; using librosa", backend)
        beat_list = _librosa_beat_track(y, file_sr)

    if not beat_list:
        log.debug("beat tracker returned no beats for %s", path)
        return None

    # Estimate a fallback BPM for build_tempo_map_from_beat_times.
    try:
        tempo_arr = _librosa.beat.beat_track(y=y, sr=file_sr, hop_length=512)[0]
        tempo_scalar = float(np.atleast_1d(tempo_arr).ravel()[0])
        if not math.isfinite(tempo_scalar) or not (_MIN_BPM <= tempo_scalar <= _MAX_BPM):
            tempo_scalar = 120.0
    except Exception:  # noqa: BLE001
        tempo_scalar = 120.0

    return build_tempo_map_from_beat_times(beat_list, fallback_bpm=tempo_scalar)
