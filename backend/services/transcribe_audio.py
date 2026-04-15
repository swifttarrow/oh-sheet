"""Audio I/O and analysis helpers for the transcription stage.

Utilities that bridge between the filesystem / audio libraries and the
pipeline orchestration logic. Extracted from ``transcribe.py`` to keep
the orchestrator focused on pipeline sequencing.
"""
from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

from backend.config import settings
from backend.services.key_estimation import (
    KeyEstimationStats,
    MeterEstimationStats,
    analyze_audio,
)
from backend.services.transcription_cleanup import AmplitudeEnvelope

log = logging.getLogger(__name__)


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


def _compute_amplitude_envelope(
    audio_path: Path,
    *,
    hop_length: int = 441,  # ~10 ms windows at 44100 Hz
    preloaded_audio: tuple | None = None,
) -> AmplitudeEnvelope | None:
    """Compute an RMS amplitude envelope from an audio file.

    Returns a list of ``(time_sec, rms_value)`` tuples sampled in ~10 ms
    windows, suitable for passing to the energy gating cleanup pass.

    When ``preloaded_audio`` is a ``(y, sr)`` tuple the ``librosa.load``
    call is skipped.  The hop_length may need adjusting when sr differs
    from the native rate.

    Returns ``None`` on any failure (missing librosa, unreadable file)
    so callers can gracefully fall back to the heuristic cleanup path.
    """
    try:
        import librosa  # noqa: PLC0415 — ships with the basic-pitch extra

        if preloaded_audio is not None:
            y, sr = preloaded_audio
        else:
            y, sr = librosa.load(str(audio_path), sr=None, mono=True)
        if len(y) == 0:
            return None

        # librosa.feature.rms returns shape (1, n_frames)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        times = librosa.frames_to_time(
            list(range(len(rms))), sr=sr, hop_length=hop_length,
        )
        return [(float(t), float(r)) for t, r in zip(times, rms)]
    except Exception as exc:  # noqa: BLE001 — best-effort
        log.debug("amplitude envelope computation failed for %s: %s", audio_path, exc)
        return None


def _maybe_analyze_key_and_meter(
    audio_path: Path,
    *,
    preloaded_audio: tuple | None = None,
) -> tuple[str, tuple[int, int], KeyEstimationStats | None, MeterEstimationStats | None]:
    """Run key + meter estimation, honouring the config feature flags.

    Either estimator can be disabled independently via
    ``OHSHEET_KEY_DETECTION_ENABLED`` /
    ``OHSHEET_METER_DETECTION_ENABLED``, and any runtime failure
    inside :func:`~backend.services.key_estimation.analyze_audio`
    already falls back to the hardcoded ``"C:major"`` / ``(4, 4)``
    defaults, so callers can use the returned values unconditionally.

    When a flag is off, the corresponding stats object is ``None``
    — the assembler in :func:`_pretty_midi_to_transcription_result`
    skips the ``as_warnings()`` extend for None stats so disabled
    estimators leave no trace in ``QualitySignal.warnings``.

    A top-level try/except catches any unexpected crash from the
    key_estimation module so a bug there can never sink the
    transcribe stage — mirrors the defensive pattern used around
    :func:`~backend.services.chord_recognition.recognize_chords`.
    """
    if (
        not settings.key_detection_enabled
        and not settings.meter_detection_enabled
    ):
        return "C:major", (4, 4), None, None

    try:
        key_label, time_signature, raw_key_stats, raw_meter_stats = analyze_audio(
            audio_path,
            key_min_confidence=settings.key_min_confidence,
            meter_confidence_margin=settings.meter_confidence_margin,
            meter_min_beats=settings.meter_min_beats,
            preloaded_audio=preloaded_audio,
        )
    except Exception as exc:  # noqa: BLE001 — never let analysis sink transcribe
        log.warning("key/meter analysis raised on %s: %s", audio_path, exc)
        return "C:major", (4, 4), None, None

    # Honour the individual feature flags by discarding one side of
    # the analysis when it's disabled. analyze_audio always runs both
    # estimators (single librosa.load) so the wall-clock cost is paid
    # once regardless — flipping an individual flag off is an escape
    # hatch, not an optimization.
    key_stats: KeyEstimationStats | None = raw_key_stats
    meter_stats: MeterEstimationStats | None = raw_meter_stats
    if not settings.key_detection_enabled:
        key_label = "C:major"
        key_stats = None
    if not settings.meter_detection_enabled:
        time_signature = (4, 4)
        meter_stats = None

    return key_label, time_signature, key_stats, meter_stats
