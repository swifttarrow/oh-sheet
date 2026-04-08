"""Audio pre-processing stage — HPSS + loudness normalization.

Runs before Basic Pitch inference to hand the pitch tracker a cleaner,
more consistent waveform:

1. **HPSS harmonic extraction.** ``librosa.effects.harmonic`` strips
   percussive transients (drums, cymbal hiss, plosive consonants) that
   the tracker otherwise emits as ghost onsets. This is the same trick
   :mod:`backend.services.chord_recognition` uses on the waveform
   before chroma, but with a gentler margin — we still want piano
   attacks to survive.

2. **RMS loudness normalization with peak ceiling.** Scales the
   waveform to a target RMS level, clamped so the peak never exceeds
   ``peak_ceiling_dbfs``. Basic Pitch's ``onset_threshold`` and
   ``frame_threshold`` are amplitude-sensitive, so un-normalized input
   makes the thresholds effectively level-dependent: a quiet YouTube
   rip and a mastered WAV hit them very differently. After
   normalization the thresholds become properties of the *signal*, not
   the *gain stage*.

Both passes are independently feature-flagged via :mod:`backend.config`
and degrade gracefully — any failure (missing librosa, unreadable
audio, HPSS crash, ...) returns the original ``audio_path`` with
``stats.skipped = True`` and a descriptive warning, so the caller
carries on with un-preprocessed audio.

The output is a WAV (32-bit float) written to a tempfile at the
original sample rate. Basic Pitch will do its own internal resampling
to 22050 Hz; preserving the native SR here avoids a double-resample.
The caller owns the returned path and must unlink it when done (only
when it differs from the input — see ``_run_basic_pitch_sync``).

Note that downstream consumers that need *percussive* information —
the librosa beat tracker in :mod:`backend.services.audio_timing` and
:mod:`backend.services.chord_recognition` — must continue reading the
*original* audio path, not the preprocessed one. HPSS strips the
transients they lock onto.
"""
from __future__ import annotations

import logging
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults — mirrored in backend/config.py so config and tests agree.
# ---------------------------------------------------------------------------

# librosa.effects.harmonic default is 1.0 (soft-mask, no bias). We keep
# the default gentle so piano attacks survive; users with drum-heavy
# material can crank up via OHSHEET_AUDIO_PREPROCESS_HPSS_MARGIN.
DEFAULT_HPSS_MARGIN = 1.0

# -20 dBFS RMS ≈ typical broadcast/speech normalization target.
# Combined with a -1 dBFS peak ceiling this leaves ~19 dB of crest
# factor headroom, which is enough for piano/rock/pop program material.
DEFAULT_TARGET_RMS_DBFS = -20.0
DEFAULT_PEAK_CEILING_DBFS = -1.0

# Below this duration we bail out — HPSS STFT windows and meaningful
# RMS both need a few hundred ms of audio to be stable.
DEFAULT_MIN_DURATION_SEC = 0.25


@dataclass
class PreprocessStats:
    """Per-run summary of what the preprocessor did.

    Structured like the other service stats objects (``CleanupStats``,
    ``ChordRecognitionStats``) so the transcribe wiring can thread it
    through to :class:`~backend.contracts.QualitySignal.warnings` the
    same way.
    """
    skipped: bool = False
    hpss_applied: bool = False
    normalize_applied: bool = False
    input_rms_dbfs: float | None = None
    output_rms_dbfs: float | None = None
    input_peak_dbfs: float | None = None
    output_peak_dbfs: float | None = None
    warnings: list[str] = field(default_factory=list)

    def as_warnings(self) -> list[str]:
        """One-line human summary entries for the QualitySignal."""
        if self.skipped:
            if self.warnings:
                return [f"audio preprocess skipped: {w}" for w in self.warnings]
            return ["audio preprocess skipped"]
        out: list[str] = []
        applied: list[str] = []
        if self.hpss_applied:
            applied.append("hpss")
        if (
            self.normalize_applied
            and self.input_rms_dbfs is not None
            and self.output_rms_dbfs is not None
            and math.isfinite(self.input_rms_dbfs)
            and math.isfinite(self.output_rms_dbfs)
        ):
            applied.append(
                f"rms {self.input_rms_dbfs:.1f}→{self.output_rms_dbfs:.1f} dBFS"
            )
        elif self.normalize_applied:
            applied.append("rms normalize")
        if applied:
            out.append("audio preprocess: " + ", ".join(applied))
        out.extend(self.warnings)
        return out


# ---------------------------------------------------------------------------
# Math helpers — numpy-only, late import.
# ---------------------------------------------------------------------------

def _rms_dbfs(y: Any) -> float:
    """Compute RMS of a waveform in dBFS. Silent input → ``-inf``."""
    import numpy as np  # noqa: PLC0415

    rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
    if rms <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(rms)


def _peak_dbfs(y: Any) -> float:
    """Compute peak level of a waveform in dBFS. Silent input → ``-inf``."""
    import numpy as np  # noqa: PLC0415

    peak = float(np.max(np.abs(y)))
    if peak <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(peak)


# ---------------------------------------------------------------------------
# In-memory waveform entry point — unit-testable core
# ---------------------------------------------------------------------------

def preprocess_waveform(
    y: Any,
    sr: int,
    *,
    hpss_enabled: bool = True,
    hpss_margin: float = DEFAULT_HPSS_MARGIN,
    normalize_enabled: bool = True,
    target_rms_dbfs: float = DEFAULT_TARGET_RMS_DBFS,
    peak_ceiling_dbfs: float = DEFAULT_PEAK_CEILING_DBFS,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
) -> tuple[Any, PreprocessStats]:
    """Run the preprocessing pipeline on a loaded waveform.

    Factored out from :func:`preprocess_audio_file` so unit tests can
    drive it with synthetic numpy signals without touching disk. Returns
    ``(processed_y, stats)``. On any failure the original waveform is
    returned with ``stats.skipped = True``.
    """
    stats = PreprocessStats()

    if not hpss_enabled and not normalize_enabled:
        stats.skipped = True
        return y, stats

    try:
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:
        log.debug("numpy unavailable for preprocess: %s", exc)
        stats.skipped = True
        stats.warnings.append("numpy unavailable")
        return y, stats

    if y is None:
        stats.skipped = True
        stats.warnings.append("empty audio")
        return y, stats

    y = np.asarray(y, dtype=np.float32)
    if y.size == 0 or sr <= 0:
        stats.skipped = True
        stats.warnings.append("empty audio")
        return y, stats

    duration = float(y.size / sr)
    if duration < min_duration_sec:
        stats.skipped = True
        stats.warnings.append(f"audio too short ({duration:.2f}s)")
        return y, stats

    stats.input_rms_dbfs = _rms_dbfs(y)
    stats.input_peak_dbfs = _peak_dbfs(y)

    # HPSS — keep only the harmonic component. Any failure falls
    # through and we continue with the un-HPSS'd signal.
    if hpss_enabled:
        try:
            import librosa  # noqa: PLC0415

            y_h = librosa.effects.harmonic(y, margin=hpss_margin)
            y = np.asarray(y_h, dtype=np.float32)
            stats.hpss_applied = True
        except ImportError:
            stats.warnings.append("hpss skipped: librosa unavailable")
        except Exception as exc:  # noqa: BLE001 — never let preprocess sink the job
            log.warning("HPSS failed: %s", exc)
            stats.warnings.append(f"hpss failed: {exc}")

    # RMS normalization with a peak ceiling. The peak-ceiling check
    # runs *after* the RMS gain is applied, so we can reduce the final
    # gain to honor the ceiling without ever having to clip. Silent
    # input (RMS ≈ 0) skips this pass — there's nothing to scale.
    if normalize_enabled:
        try:
            rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
            if rms > 1e-6:
                target_rms = 10.0 ** (target_rms_dbfs / 20.0)
                gain = target_rms / rms
                projected_peak = float(np.max(np.abs(y))) * gain
                ceiling = 10.0 ** (peak_ceiling_dbfs / 20.0)
                if projected_peak > ceiling:
                    gain *= ceiling / projected_peak
                y = (y * gain).astype(np.float32)
                stats.normalize_applied = True
            else:
                stats.warnings.append("normalize skipped: silent input")
        except Exception as exc:  # noqa: BLE001
            log.warning("normalize failed: %s", exc)
            stats.warnings.append(f"normalize failed: {exc}")

    stats.output_rms_dbfs = _rms_dbfs(y)
    stats.output_peak_dbfs = _peak_dbfs(y)

    # If neither pass actually ran, flag skipped so the file-path
    # wrapper doesn't bother writing a redundant temp file.
    if not stats.hpss_applied and not stats.normalize_applied:
        stats.skipped = True

    return y, stats


# ---------------------------------------------------------------------------
# File-path entry point — what transcribe.py calls
# ---------------------------------------------------------------------------

def preprocess_audio_file(
    audio_path: Path,
    *,
    hpss_enabled: bool = True,
    hpss_margin: float = DEFAULT_HPSS_MARGIN,
    normalize_enabled: bool = True,
    target_rms_dbfs: float = DEFAULT_TARGET_RMS_DBFS,
    peak_ceiling_dbfs: float = DEFAULT_PEAK_CEILING_DBFS,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
) -> tuple[Path, PreprocessStats]:
    """Pre-process an audio file for Basic Pitch inference.

    Returns ``(processed_path, stats)``. When preprocessing ran,
    ``processed_path`` points at a newly-written tempfile (32-bit float
    WAV at the native sample rate) and the caller is responsible for
    unlinking it. When preprocessing was skipped or failed,
    ``processed_path == audio_path`` and ``stats.skipped`` is True —
    the caller must **not** unlink in that case.

    Compare the returned path against the input to decide whether a
    cleanup is needed::

        processed, stats = preprocess_audio_file(audio_path)
        try:
            run_inference(processed)
        finally:
            if processed != audio_path:
                processed.unlink(missing_ok=True)
    """
    stats = PreprocessStats()

    if not hpss_enabled and not normalize_enabled:
        stats.skipped = True
        return audio_path, stats

    try:
        import librosa  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        import soundfile  # noqa: PLC0415
    except ImportError as exc:
        log.debug("audio preprocess deps unavailable: %s", exc)
        stats.skipped = True
        stats.warnings.append(f"missing dep: {exc.name}")
        return audio_path, stats

    # Load at native SR, force mono — Basic Pitch will resample to
    # 22050 Hz internally regardless, so we skip an extra resample here.
    try:
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    except Exception as exc:  # noqa: BLE001 — bad audio shouldn't crash the worker
        log.warning("preprocess load failed for %s: %s", audio_path, exc)
        stats.skipped = True
        stats.warnings.append(f"load failed: {exc}")
        return audio_path, stats

    y_proc, stats = preprocess_waveform(
        y, int(sr),
        hpss_enabled=hpss_enabled,
        hpss_margin=hpss_margin,
        normalize_enabled=normalize_enabled,
        target_rms_dbfs=target_rms_dbfs,
        peak_ceiling_dbfs=peak_ceiling_dbfs,
        min_duration_sec=min_duration_sec,
    )

    if stats.skipped:
        return audio_path, stats

    # Write to a tempfile. 32-bit float WAV is lossless and reads fast.
    tmp_path: Path | None = None
    try:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".wav", prefix="ohsheet-preprocess-", delete=False,
        )
        tmp.close()
        tmp_path = Path(tmp.name)
        soundfile.write(str(tmp_path), np.asarray(y_proc), int(sr), subtype="FLOAT")
        return tmp_path, stats
    except Exception as exc:  # noqa: BLE001
        log.warning("preprocess write failed for %s: %s", audio_path, exc)
        stats.skipped = True
        stats.warnings.append(f"write failed: {exc}")
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass
        return audio_path, stats
