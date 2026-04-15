"""Estimate ``HarmonicAnalysis.key`` and ``time_signature`` from audio.

Basic Pitch does not estimate key or meter, so the transcribe stage
previously hardcoded ``"C:major"`` and ``(4, 4)`` into every
:class:`~backend.contracts.HarmonicAnalysis`. The engrave stage then
renders those literally — every piece ends up in C major 4/4
regardless of its actual tonality, which is the most visible quality
symptom in the output PDF.

This module fills the gap with two cheap waveform-based detectors:

* :func:`estimate_key_from_waveform` — Krumhansl-Schmuckler against a
  time-averaged ``chroma_cqt`` vector. Returns a ``"Root:major"`` /
  ``"Root:minor"`` label in the format ``engrave.py`` already parses
  via ``score.metadata.key.split(":")``.
* :func:`estimate_meter_from_waveform` — onset-strength-at-beat-time
  periodicity analysis. Picks between 3/4 and 4/4 (denominator always
  4 for v1) by folding the beat-strength vector modulo each hypothesis
  and scoring the "first-beat prominence" pattern.

Both entry points have three layers so they can be exercised at
whichever granularity the caller needs:

1. Pure-numpy cores (``*_from_chroma``, ``_score_meter_hypothesis``) —
   deterministic, hermetic, unit-testable with synthetic inputs.
2. Waveform wrappers (``*_from_waveform``) — handle HPSS / beat
   tracking / chroma extraction.
3. File-path entry (:func:`analyze_audio`) — single librosa.load that
   feeds both estimators, so the transcribe wiring only pays the
   audio-decode cost once per job.

Everything mirrors the graceful-degradation contract of
:mod:`backend.services.chord_recognition`: any failure (missing
librosa, unreadable audio, short clip, degenerate chroma) returns
the hardcoded defaults with ``stats.skipped=True`` so the caller can
carry on without losing the job.

The module is intentionally v1 scope:
  * Key: major / minor only — no modal detection (Dorian, Mixolydian,
    etc.) since ``music21.key.Key`` in the engrave stage only accepts
    standard major / minor labels anyway.
  * Meter: 3/4 vs 4/4 only. 6/8 usually beat-tracks as 2 dotted-
    quarter pulses which this heuristic would misclassify as 2/4, so
    we don't try. The denominator is always 4.

.. _Krumhansl-Schmuckler: https://en.wikipedia.org/wiki/Krumhansl%E2%80%93Schmuckler_key-finding_algorithm
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Krumhansl-Schmuckler tonal profiles
# ---------------------------------------------------------------------------

# Canonical K-S major / minor key profiles from Krumhansl & Kessler
# (1982, "Tracing the dynamic changes in perceived tonal organization
# in a spatial representation of musical keys"). These are probe-tone
# correlation ratings that encode the statistical "feel" of the pitch
# class distribution in a given key. Index 0 is the tonic, index 7 is
# the dominant — rotate by ``root`` to get the profile for any key.
_KS_MAJOR: tuple[float, ...] = (
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
    2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
)
_KS_MINOR: tuple[float, ...] = (
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
    2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
)

_PITCH_NAMES: tuple[str, ...] = (
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
)

# Default librosa parameters for the file-path entry points. Match
# ``chord_recognition`` so the two modules can share a single load on
# the same waveform if a future caller wants to (today we just do two
# separate loads — see ``analyze_audio`` for the single-load path).
DEFAULT_SAMPLE_RATE = 22_050
DEFAULT_HOP_LENGTH = 512
DEFAULT_BINS_PER_OCTAVE = 36
DEFAULT_HPSS_MARGIN = 3.0

# Confidence floor for claiming a key label. The Pearson correlation
# between the (zero-meaned, L2-normalized) song-level chroma vector
# and each KS profile lives in [-1, 1]. Real tonal music typically
# scores in the 0.55–0.90 range; below 0.45 is usually non-tonal,
# percussion-heavy, or too short. We fall back to the hardcoded
# C:major default when we can't clear the floor, so the result is
# never worse than the old behaviour.
DEFAULT_KEY_MIN_CONFIDENCE = 0.55

# Meter detection: test 3-beat-bar and 4-beat-bar hypotheses only.
# Larger windows (5, 7) are rare enough that a false positive would
# hurt more than the occasional missed 5/4; smaller windows (2) are
# indistinguishable from 4 under this heuristic.
_METER_HYPOTHESES: tuple[int, ...] = (3, 4)

# Meter detection needs enough beats to form a statistically meaningful
# periodicity estimate. 8 beats = 2 bars of 4/4 minimum; anything below
# that gives false-positive 3/4 labels on short audio clips.
DEFAULT_METER_MIN_BEATS = 8

# Tie-break margin: 4/4 wins unless 3/4 scores more than this much
# higher. Skews the default toward the far-more-common meter in
# pop/rock/jazz so borderline cases don't flap the output.
DEFAULT_METER_CONFIDENCE_MARGIN = 0.05


@dataclass
class KeyEstimationStats:
    """Telemetry for one ``estimate_key*`` call.

    Attached to ``QualitySignal.warnings`` via :meth:`as_warnings` —
    mirrors the pattern used by every other estimator module so the
    transcribe assembly site only has to do
    ``warnings.extend(stats.as_warnings())``.
    """
    skipped: bool = False
    key_label: str = ""
    confidence: float = 0.0
    # Second-best label + score. Useful for debugging relative-key
    # confusions (C:major and A:minor share a chroma profile — only
    # the KS probe weights break the tie, and the tie is usually close).
    runner_up_label: str = ""
    runner_up_confidence: float = 0.0
    # Chord-based cross-validation telemetry.
    chord_validated: bool = False    # True when refine_key_with_chords ran
    chord_flipped: bool = False      # True when the key was changed by chord validation
    chord_diatonic_fraction: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def as_warnings(self) -> list[str]:
        out: list[str] = []
        if self.skipped:
            out.append(
                "key detection skipped"
                + (f" ({self.warnings[0]})" if self.warnings else "")
            )
            return out
        if self.key_label:
            out.append(
                f"key: {self.key_label} (conf={self.confidence:.2f}, "
                f"runner-up {self.runner_up_label} "
                f"{self.runner_up_confidence:.2f})"
            )
        if self.chord_validated:
            flip_tag = " [FLIPPED]" if self.chord_flipped else ""
            out.append(
                f"chord cross-validation: diatonic_frac="
                f"{self.chord_diatonic_fraction:.2f}{flip_tag}"
            )
        out.extend(self.warnings)
        return out


@dataclass
class MeterEstimationStats:
    """Telemetry for one ``estimate_meter*`` call."""
    skipped: bool = False
    time_signature: tuple[int, int] = (4, 4)
    confidence: float = 0.0
    n_beats: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_warnings(self) -> list[str]:
        out: list[str] = []
        if self.skipped:
            out.append(
                "meter detection skipped"
                + (f" ({self.warnings[0]})" if self.warnings else "")
            )
            return out
        num, den = self.time_signature
        out.append(
            f"time_signature: {num}/{den} (conf={self.confidence:.2f}, "
            f"{self.n_beats} beats analyzed)"
        )
        out.extend(self.warnings)
        return out


# ---------------------------------------------------------------------------
# Krumhansl-Schmuckler — pure-numpy core
# ---------------------------------------------------------------------------

def _build_key_profiles() -> tuple[Any, list[str]]:
    """Build all 24 KS profiles as a single ``(24, 12)`` matrix.

    Rows 0–11 are the 12 major keys, rows 12–23 are the 12 minor keys,
    each rotated so index 0 aligns with its tonic pitch class. Labels
    use the ``"Root:major"`` / ``"Root:minor"`` format
    ``engrave.py`` already parses via ``score.metadata.key.split(":")``.

    Profiles are zero-meaned and L2-normalized so the dot product in
    :func:`estimate_key_from_chroma` is a Pearson correlation rather
    than a biased raw template match. This matters: without the zero-
    mean, the major profile's overall higher weight dominates every
    match and biases the result toward major regardless of content.
    """
    import numpy as np  # noqa: PLC0415

    profiles = np.zeros((24, 12), dtype=np.float32)
    labels: list[str] = []

    for root in range(12):
        # ``np.roll(x, k)`` shifts right by k: after rolling by ``root``,
        # position ``root`` holds the original index 0 (the tonic weight).
        profiles[root] = np.roll(
            np.asarray(_KS_MAJOR, dtype=np.float32), root,
        )
        labels.append(f"{_PITCH_NAMES[root]}:major")

    for root in range(12):
        idx = 12 + root
        profiles[idx] = np.roll(
            np.asarray(_KS_MINOR, dtype=np.float32), root,
        )
        labels.append(f"{_PITCH_NAMES[root]}:minor")

    profiles = profiles - profiles.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    profiles = profiles / np.clip(norms, 1e-9, None)
    return profiles, labels


def estimate_key_from_chroma(
    chroma: Any,  # np.ndarray shape (12, n_frames)
    *,
    min_confidence: float = DEFAULT_KEY_MIN_CONFIDENCE,
) -> tuple[str, KeyEstimationStats]:
    """Pick the best-fitting KS key label for a precomputed chroma matrix.

    ``chroma`` is a librosa-shape chroma matrix ``(12, n_frames)``. Any
    source — ``chroma_cqt``, ``chroma_stft``, ``chroma_cens`` — works
    as long as the row order is ``[C, C#, D, ..., B]``. The matrix is
    time-averaged to a single 12-vector pitch class profile before the
    Pearson match.

    Returns ``("C:major", stats_with_skipped=True)`` on any failure or
    when the best correlation falls below ``min_confidence``, so the
    caller can still populate ``HarmonicAnalysis.key`` with a sensible
    default. ``stats.skipped=True`` means we fell back.
    """
    stats = KeyEstimationStats()

    try:
        import numpy as np  # noqa: PLC0415
    except ImportError:
        stats.skipped = True
        stats.warnings.append("numpy unavailable")
        return "C:major", stats

    arr = np.asarray(chroma, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != 12 or arr.shape[1] == 0:
        stats.skipped = True
        stats.warnings.append(f"invalid chroma shape: {tuple(arr.shape)}")
        return "C:major", stats

    avg = arr.mean(axis=1).astype(np.float32)
    if not np.all(np.isfinite(avg)) or float(avg.sum()) <= 0.0:
        stats.skipped = True
        stats.warnings.append("degenerate chroma (all zero / non-finite)")
        return "C:major", stats

    # Zero-mean + L2-normalize so ``profiles @ avg`` is a Pearson
    # correlation in [-1, 1]. The profile side of this normalization
    # already happens in ``_build_key_profiles``.
    avg = avg - avg.mean()
    norm = float(np.linalg.norm(avg))
    if norm < 1e-9:
        stats.skipped = True
        stats.warnings.append("chroma has no tonal variance")
        return "C:major", stats
    avg = avg / norm

    profiles, labels = _build_key_profiles()
    scores = profiles @ avg  # (24,) in [-1, 1]

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    # Runner-up for the warning — the KS algorithm's most common error
    # is swapping a major key for its relative minor (same chroma,
    # different tonic weighting), so logging the second place lets
    # future debugging spot those flaps immediately.
    masked = scores.copy()
    masked[best_idx] = -np.inf
    runner_up_idx = int(np.argmax(masked))
    runner_up_score = float(scores[runner_up_idx])

    stats.key_label = labels[best_idx]
    stats.confidence = round(best_score, 3)
    stats.runner_up_label = labels[runner_up_idx]
    stats.runner_up_confidence = round(runner_up_score, 3)

    if best_score < min_confidence:
        stats.skipped = True
        stats.warnings.append(
            f"key confidence {best_score:.2f} below floor "
            f"{min_confidence:.2f}; falling back to C:major "
            f"(best guess was {labels[best_idx]})"
        )
        return "C:major", stats

    return labels[best_idx], stats


def estimate_key_from_waveform(
    y: Any,  # np.ndarray shape (samples,)
    sr: int,
    *,
    min_confidence: float = DEFAULT_KEY_MIN_CONFIDENCE,
    hpss_margin: float = DEFAULT_HPSS_MARGIN,
    hop_length: int = DEFAULT_HOP_LENGTH,
    bins_per_octave: int = DEFAULT_BINS_PER_OCTAVE,
) -> tuple[str, KeyEstimationStats]:
    """Estimate key from a loaded mono waveform.

    HPSS-harmonics the signal so percussive transients don't smear
    the chroma vector, computes ``chroma_cqt``, then delegates to
    :func:`estimate_key_from_chroma`. Mirrors the chord-recognition
    preprocessing so both modules see consistent input.
    """
    stats = KeyEstimationStats()

    try:
        import librosa  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError:
        stats.skipped = True
        stats.warnings.append("librosa unavailable")
        return "C:major", stats

    if y is None or len(y) == 0 or sr <= 0:
        stats.skipped = True
        stats.warnings.append("empty waveform")
        return "C:major", stats

    duration = float(len(y) / sr)
    if duration < 1.0:
        stats.skipped = True
        stats.warnings.append(
            f"audio too short for key detection ({duration:.2f}s)"
        )
        return "C:major", stats

    y_np = np.asarray(y)
    try:
        y_h = librosa.effects.harmonic(y_np, margin=hpss_margin)
    except Exception as exc:  # noqa: BLE001 — best-effort preprocessing
        log.warning("HPSS failed for key detection: %s", exc)
        y_h = y_np

    try:
        chroma = librosa.feature.chroma_cqt(
            y=y_h, sr=sr,
            hop_length=hop_length,
            bins_per_octave=bins_per_octave,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("chroma_cqt failed for key detection: %s", exc)
        stats.skipped = True
        stats.warnings.append(f"chroma_cqt failed: {exc}")
        return "C:major", stats

    return estimate_key_from_chroma(chroma, min_confidence=min_confidence)


# ---------------------------------------------------------------------------
# Meter detection — beat-strength periodicity
# ---------------------------------------------------------------------------

def _score_meter_hypothesis(
    beat_strengths: Any,
    k: int,
) -> tuple[float, int]:
    """Score how well ``beat_strengths`` fits a k-beat-per-bar hypothesis.

    For each candidate phase ``p`` in ``[0, k)``, fold the strengths
    modulo ``k`` and compute the ratio of the mean strength at phase
    ``p`` to the mean strength across all other phases. The highest
    ratio across phases wins — that phase is the downbeat position.

    A ratio > 1 means the chosen phase has above-average beat
    strength, i.e. the bar genuinely starts on that beat. Returns
    ``(best_ratio, best_phase)``. ``best_ratio = 0.0`` when the
    input is too short for a meaningful fold.

    Kept as a pure function over the 1D strengths array so meter
    tests can hand-build synthetic stream without touching librosa.
    """
    import numpy as np  # noqa: PLC0415

    arr = np.asarray(beat_strengths, dtype=np.float32)
    n = int(arr.shape[0])
    if n < k * 2:
        return 0.0, 0

    best_ratio = 0.0
    best_phase = 0
    for phase in range(k):
        on_beat = arr[phase::k]
        # Build the off-beat mask by zeroing every k-th sample starting
        # at ``phase``. This is more accurate than ``arr[phase+1::k]``
        # for k > 2 because it actually includes every non-phase index.
        mask = np.ones(n, dtype=bool)
        mask[phase::k] = False
        off_beat = arr[mask]
        if on_beat.size == 0 or off_beat.size == 0:
            continue
        on_mean = float(on_beat.mean())
        off_mean = float(off_beat.mean())
        if off_mean <= 1e-9:
            # Degenerate off-beat mean: all mass concentrated at the
            # chosen phase, which is a strong signal. Use ``on_mean``
            # directly so we don't divide by zero.
            ratio = on_mean
        else:
            ratio = on_mean / off_mean
        if ratio > best_ratio:
            best_ratio = ratio
            best_phase = phase
    return best_ratio, best_phase


def estimate_meter_from_beat_strengths(
    beat_strengths: Any,
    *,
    confidence_margin: float = DEFAULT_METER_CONFIDENCE_MARGIN,
    min_beats: int = DEFAULT_METER_MIN_BEATS,
) -> tuple[tuple[int, int], MeterEstimationStats]:
    """Pick 3/4 vs 4/4 from a precomputed beat-strength vector.

    ``beat_strengths`` is one scalar per detected beat — typically
    ``onset_env[beat_frames]`` from librosa, but any monotonic
    per-beat strength signal will do. The function is deterministic
    and numpy-only, so meter tests can hand-build synthetic vectors
    (e.g. ``[1, 0.3, 0.5, 0.3] * 8`` for a clean 4/4).

    Returns ``((4, 4), stats_with_skipped=True)`` when confidence is
    below the tie-break margin or the input is too short — 4/4 is
    the far more common meter in pop/rock/jazz, so the default
    costs us nothing on ambiguous material.
    """
    stats = MeterEstimationStats()

    try:
        import numpy as np  # noqa: PLC0415  — used by _score_meter_hypothesis
    except ImportError:
        stats.skipped = True
        stats.warnings.append("numpy unavailable")
        return (4, 4), stats

    arr = np.asarray(beat_strengths, dtype=np.float32)
    stats.n_beats = int(arr.shape[0])

    if arr.ndim != 1 or arr.shape[0] < min_beats:
        stats.skipped = True
        stats.warnings.append(
            f"only {arr.shape[0]} beats, need {min_beats} for meter"
        )
        return (4, 4), stats

    if not np.all(np.isfinite(arr)):
        stats.skipped = True
        stats.warnings.append("beat strengths contain non-finite values")
        return (4, 4), stats

    scores: dict[int, float] = {}
    for k in _METER_HYPOTHESES:
        ratio, _phase = _score_meter_hypothesis(arr, k)
        scores[k] = ratio

    if not scores or max(scores.values()) <= 0.0:
        stats.skipped = True
        stats.warnings.append("all meter hypotheses scored zero")
        return (4, 4), stats

    best_k = max(scores, key=lambda k: scores[k])
    best_score = scores[best_k]

    # Tie-break toward 4/4 — if it's within ``confidence_margin`` of
    # the nominal winner, prefer 4. The margin is a raw ratio delta
    # (e.g. 0.05 = 5% stronger on-beat mean required to flip).
    if (
        best_k != 4
        and 4 in scores
        and best_score - scores[4] <= confidence_margin
    ):
        best_k = 4
        best_score = scores[4]

    # Confidence is the "how much stronger was the downbeat vs the
    # off-beats" lift, normalized into [0, 1] so stats consumers can
    # interpret it uniformly alongside the key confidence. A lift of
    # 2.0 (downbeats twice as strong as off-beats) saturates to 1.0.
    confidence = min(max(best_score - 1.0, 0.0), 1.0)

    stats.time_signature = (best_k, 4)
    stats.confidence = round(confidence, 3)
    return stats.time_signature, stats


def estimate_meter_from_waveform(
    y: Any,
    sr: int,
    *,
    hop_length: int = DEFAULT_HOP_LENGTH,
    confidence_margin: float = DEFAULT_METER_CONFIDENCE_MARGIN,
    min_beats: int = DEFAULT_METER_MIN_BEATS,
) -> tuple[tuple[int, int], MeterEstimationStats]:
    """Estimate meter from a loaded waveform.

    Pipeline
    --------
    1. ``librosa.onset.onset_strength`` → per-frame novelty envelope.
    2. ``librosa.beat.beat_track`` on the same envelope → beat frames.
    3. Sample the envelope at beat frames → one scalar per beat.
    4. Delegate to :func:`estimate_meter_from_beat_strengths`.

    Returns ``((4, 4), stats_with_skipped=True)`` on any librosa
    failure so the caller falls back to the old default.
    """
    stats = MeterEstimationStats()

    try:
        import librosa  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError:
        stats.skipped = True
        stats.warnings.append("librosa unavailable")
        return (4, 4), stats

    if y is None or len(y) == 0 or sr <= 0:
        stats.skipped = True
        stats.warnings.append("empty waveform")
        return (4, 4), stats

    y_np = np.asarray(y)

    try:
        onset_env = librosa.onset.onset_strength(
            y=y_np, sr=sr, hop_length=hop_length,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("onset_strength failed for meter detection: %s", exc)
        stats.skipped = True
        stats.warnings.append(f"onset_strength failed: {exc}")
        return (4, 4), stats

    try:
        _tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=hop_length,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("beat_track failed for meter detection: %s", exc)
        stats.skipped = True
        stats.warnings.append(f"beat_track failed: {exc}")
        return (4, 4), stats

    beat_frames = np.atleast_1d(beat_frames).astype(int)
    if beat_frames.size == 0 or onset_env.shape[0] == 0:
        stats.skipped = True
        stats.warnings.append("no beats detected")
        return (4, 4), stats

    # beat_track occasionally emits an index at ``len(onset_env)`` for
    # the trailing beat, which would out-of-bounds on a direct lookup.
    safe_frames = np.clip(beat_frames, 0, onset_env.shape[0] - 1)
    beat_strengths = onset_env[safe_frames]

    return estimate_meter_from_beat_strengths(
        beat_strengths,
        confidence_margin=confidence_margin,
        min_beats=min_beats,
    )


# ---------------------------------------------------------------------------
# Combined file-path entry — single librosa.load for both estimators
# ---------------------------------------------------------------------------

def analyze_audio(
    audio_path: Path,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    hop_length: int = DEFAULT_HOP_LENGTH,
    bins_per_octave: int = DEFAULT_BINS_PER_OCTAVE,
    hpss_margin: float = DEFAULT_HPSS_MARGIN,
    key_min_confidence: float = DEFAULT_KEY_MIN_CONFIDENCE,
    meter_confidence_margin: float = DEFAULT_METER_CONFIDENCE_MARGIN,
    meter_min_beats: int = DEFAULT_METER_MIN_BEATS,
    preloaded_audio: tuple | None = None,
) -> tuple[str, tuple[int, int], KeyEstimationStats, MeterEstimationStats]:
    """Run both key and meter estimators on ``audio_path`` with one load.

    The transcribe wiring calls this exactly once per job. Loading
    audio via librosa costs ~100 ms for a 30 s file at 22 kHz, and
    both estimators want the same mono downmix, so the pragmatic
    move is to load once and fan out.

    When ``preloaded_audio`` is a ``(y, sr)`` tuple the ``librosa.load``
    call is skipped, reusing a waveform already in memory.

    Returns the full ``(key_label, time_signature, key_stats,
    meter_stats)`` tuple so the caller can attach both stats to the
    ``QualitySignal.warnings`` list without a second dispatch.

    Any load failure returns ``("C:major", (4, 4), skipped, skipped)``
    so the caller falls back to the old hardcoded defaults without
    losing the job. Each estimator is wrapped in its own try so a
    failure in one doesn't poison the other.
    """
    key_stats = KeyEstimationStats()
    meter_stats = MeterEstimationStats()

    try:
        import librosa  # noqa: PLC0415
    except ImportError:
        key_stats.skipped = True
        key_stats.warnings.append("librosa unavailable")
        meter_stats.skipped = True
        meter_stats.warnings.append("librosa unavailable")
        return "C:major", (4, 4), key_stats, meter_stats

    if preloaded_audio is not None:
        y, file_sr = preloaded_audio
    else:
        try:
            y, file_sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "librosa.load failed for key/meter analysis on %s: %s",
                audio_path, exc,
            )
            key_stats.skipped = True
            key_stats.warnings.append(f"librosa.load failed: {exc}")
            meter_stats.skipped = True
            meter_stats.warnings.append(f"librosa.load failed: {exc}")
            return "C:major", (4, 4), key_stats, meter_stats

    try:
        key_label, key_stats = estimate_key_from_waveform(
            y, int(file_sr),
            min_confidence=key_min_confidence,
            hpss_margin=hpss_margin,
            hop_length=hop_length,
            bins_per_octave=bins_per_octave,
        )
    except Exception as exc:  # noqa: BLE001 — never let key detection sink transcribe
        log.warning("key estimation raised on %s: %s", audio_path, exc)
        key_label = "C:major"
        key_stats = KeyEstimationStats(skipped=True)
        key_stats.warnings.append(f"key estimation failed: {exc}")

    try:
        time_sig, meter_stats = estimate_meter_from_waveform(
            y, int(file_sr),
            hop_length=hop_length,
            confidence_margin=meter_confidence_margin,
            min_beats=meter_min_beats,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("meter estimation raised on %s: %s", audio_path, exc)
        time_sig = (4, 4)
        meter_stats = MeterEstimationStats(skipped=True)
        meter_stats.warnings.append(f"meter estimation failed: {exc}")

    return key_label, time_sig, key_stats, meter_stats


# ---------------------------------------------------------------------------
# Chord-based key cross-validation
# ---------------------------------------------------------------------------

# Diatonic chord qualities for major and (natural) minor scales.
# Each entry is (interval_from_tonic_in_semitones, quality_string).
# The vii° is approximated as "min" because the chord recognizer only
# emits "maj" / "min" triads — dim templates don't exist in the
# current v1 chord_recognition module.

_MAJOR_DIATONIC_INTERVALS: list[tuple[int, str]] = [
    (0, "maj"),   # I
    (2, "min"),   # ii
    (4, "min"),   # iii
    (5, "maj"),   # IV
    (7, "maj"),   # V
    (9, "min"),   # vi
    (11, "min"),  # vii° (approximate as min)
]

_MINOR_DIATONIC_INTERVALS: list[tuple[int, str]] = [
    (0, "min"),   # i
    (2, "min"),   # ii° (approximate as min)
    (3, "maj"),   # III
    (5, "min"),   # iv
    (7, "min"),   # v (natural minor; harmonic minor would be maj)
    (8, "maj"),   # VI
    (10, "maj"),  # VII
]

# Map pitch class names (as used in chord labels) to pitch class index.
_NAME_TO_PC: dict[str, int] = {
    name: i for i, name in enumerate(_PITCH_NAMES)
}


def _diatonic_chords_for_key(key_label: str) -> set[tuple[int, str]]:
    """Return set of ``(root_pc, quality)`` tuples for chords diatonic to *key_label*.

    *key_label* uses the ``"Root:major"`` / ``"Root:minor"`` format
    emitted by :func:`estimate_key_from_chroma`.  The returned set
    uses pitch class integers (0–11, C=0) and quality strings matching
    the chord recognition module's output (``"maj"`` / ``"min"``).
    """
    parts = key_label.split(":")
    if len(parts) != 2:
        return set()

    root_name, quality = parts
    root_pc = _NAME_TO_PC.get(root_name)
    if root_pc is None:
        return set()

    intervals = (
        _MAJOR_DIATONIC_INTERVALS
        if quality == "major"
        else _MINOR_DIATONIC_INTERVALS
    )
    return {
        ((root_pc + semitone) % 12, q)
        for semitone, q in intervals
    }


def _chord_label_to_pc_quality(label: str) -> tuple[int, str] | None:
    """Parse a Harte chord label like ``"C:maj7"`` into ``(pc, quality)``.

    Strips extended quality suffixes (``"7"``, ``"9"``, …) so
    ``"C:maj7"`` matches the ``"maj"`` diatonic entry. Returns
    ``None`` for unparseable / no-chord labels.
    """
    parts = label.split(":")
    if len(parts) != 2:
        return None
    root_name, raw_quality = parts
    root_pc = _NAME_TO_PC.get(root_name)
    if root_pc is None:
        return None
    # Normalize quality: strip trailing digits (7, 9, 11, 13) and
    # common suffixes so "maj7" → "maj", "min7" → "min".
    quality = raw_quality.rstrip("0123456789")
    if not quality:
        return None
    return root_pc, quality


def _is_relative_major_minor(key_a: str, key_b: str) -> bool:
    """Return ``True`` if *key_a* and *key_b* are relative major/minor pairs.

    E.g. ``("C:major", "A:minor")`` → True, ``("G:major", "E:minor")``
    → True.  The relative minor of a major key is 9 semitones above
    (equivalently 3 below).
    """
    parts_a = key_a.split(":")
    parts_b = key_b.split(":")
    if len(parts_a) != 2 or len(parts_b) != 2:
        return False
    root_a = _NAME_TO_PC.get(parts_a[0])
    root_b = _NAME_TO_PC.get(parts_b[0])
    if root_a is None or root_b is None:
        return False
    q_a, q_b = parts_a[1], parts_b[1]
    if q_a == "major" and q_b == "minor":
        return (root_a + 9) % 12 == root_b
    if q_a == "minor" and q_b == "major":
        return (root_b + 9) % 12 == root_a
    return False


def _diatonic_fraction(
    chords: list[Any],
    diatonic_set: set[tuple[int, str]],
) -> float:
    """Fraction of total chord duration whose root+quality is in *diatonic_set*.

    Chords whose label cannot be parsed are ignored (neither counted
    in the numerator nor the denominator).
    """
    total_dur = 0.0
    diatonic_dur = 0.0
    for ch in chords:
        parsed = _chord_label_to_pc_quality(ch.label)
        if parsed is None:
            continue
        dur = ch.duration_sec
        total_dur += dur
        if parsed in diatonic_set:
            diatonic_dur += dur
    if total_dur <= 0.0:
        return 0.0
    return diatonic_dur / total_dur


def _tonic_pc_for_key(key_label: str) -> int | None:
    """Extract the tonic pitch class from a key label like ``"C:major"``."""
    parts = key_label.split(":")
    if len(parts) != 2:
        return None
    return _NAME_TO_PC.get(parts[0])


# Default thresholds for chord-based cross-validation.
DEFAULT_KEY_CHORD_DIATONIC_THRESHOLD = 0.6
DEFAULT_KEY_CHORD_FLIP_MARGIN = 0.15


def refine_key_with_chords(
    ks_key: str,
    ks_confidence: float,
    runner_up_key: str,
    runner_up_confidence: float,
    chord_labels: list,
    *,
    diatonic_threshold: float = DEFAULT_KEY_CHORD_DIATONIC_THRESHOLD,
    flip_margin: float = DEFAULT_KEY_CHORD_FLIP_MARGIN,
) -> tuple[str, KeyEstimationStats]:
    """Refine the KS key estimate by cross-validating against detected chords.

    The Krumhansl-Schmuckler algorithm's biggest weakness is relative
    major/minor confusion (e.g. Am vs C) because the two keys share an
    identical pitch class set — only the KS probe weights break the
    tie, and the tie is usually close.  This function checks what
    fraction of the *detected chord durations* fall within the diatonic
    chord set for each candidate key and optionally flips to the
    runner-up when the chords provide stronger evidence.

    **Decision logic:**

    1. If ``ks_key``'s diatonic fraction >= ``diatonic_threshold``, keep
       it (already well-supported by the chords).
    2. If the runner-up's diatonic fraction is higher *and* the KS
       confidence gap is within ``flip_margin``, flip to the runner-up.
    3. For relative major/minor pairs (Am <-> C), the diatonic sets are
       identical so fractions cannot break the tie.  Instead, check
       whether the first and last high-confidence chords match the
       runner-up's tonic more than the KS key's tonic — if so, flip.
    4. Otherwise keep the KS key.

    Returns ``(refined_key_label, updated_stats)`` — the stats object
    records whether chord validation ran and whether it flipped the key.
    Any exception returns the original KS key unchanged.
    """
    stats = KeyEstimationStats(
        key_label=ks_key,
        confidence=ks_confidence,
        runner_up_label=runner_up_key,
        runner_up_confidence=runner_up_confidence,
    )

    try:
        return _refine_key_with_chords_inner(
            ks_key, ks_confidence,
            runner_up_key, runner_up_confidence,
            chord_labels,
            stats,
            diatonic_threshold=diatonic_threshold,
            flip_margin=flip_margin,
        )
    except Exception as exc:  # noqa: BLE001 — graceful degradation
        log.warning("refine_key_with_chords failed: %s", exc)
        stats.warnings.append(f"chord cross-validation failed: {exc}")
        return ks_key, stats


def _refine_key_with_chords_inner(
    ks_key: str,
    ks_confidence: float,
    runner_up_key: str,
    runner_up_confidence: float,
    chord_labels: list,
    stats: KeyEstimationStats,
    *,
    diatonic_threshold: float,
    flip_margin: float,
) -> tuple[str, KeyEstimationStats]:
    """Core logic for :func:`refine_key_with_chords` (no exception wrapper)."""

    stats.chord_validated = True

    if not chord_labels:
        return ks_key, stats

    # Build diatonic sets for both candidates.
    ks_diatonic = _diatonic_chords_for_key(ks_key)
    ru_diatonic = _diatonic_chords_for_key(runner_up_key)

    ks_frac = _diatonic_fraction(chord_labels, ks_diatonic)
    ru_frac = _diatonic_fraction(chord_labels, ru_diatonic)
    stats.chord_diatonic_fraction = round(ks_frac, 3)

    confidence_gap = ks_confidence - runner_up_confidence

    # Check if the two candidates are a relative major/minor pair.
    is_relative = _is_relative_major_minor(ks_key, runner_up_key)

    if is_relative:
        # Relative major/minor share the same diatonic set, so the
        # diatonic fraction cannot differentiate them. Use a secondary
        # signal: do the first and last high-confidence chords land on
        # the runner-up's tonic more than the KS key's tonic?
        ks_tonic = _tonic_pc_for_key(ks_key)
        ru_tonic = _tonic_pc_for_key(runner_up_key)
        if ks_tonic is not None and ru_tonic is not None:
            strong = [
                ch for ch in chord_labels
                if ch.confidence > 0.6
            ]
            if strong:
                # Look at first and last strong chords.
                boundary_chords = [strong[0]]
                if len(strong) > 1:
                    boundary_chords.append(strong[-1])
                ks_matches = sum(
                    1 for ch in boundary_chords
                    if ch.root == ks_tonic
                )
                ru_matches = sum(
                    1 for ch in boundary_chords
                    if ch.root == ru_tonic
                )
                if ru_matches > ks_matches and confidence_gap < flip_margin:
                    stats.chord_flipped = True
                    stats.key_label = runner_up_key
                    return runner_up_key, stats
        return ks_key, stats

    # Non-relative keys: compare diatonic fractions.
    if ks_frac >= diatonic_threshold:
        # KS key is well-supported by the chords — keep it.
        return ks_key, stats

    if ru_frac > ks_frac and confidence_gap < flip_margin:
        stats.chord_flipped = True
        stats.key_label = runner_up_key
        return runner_up_key, stats

    return ks_key, stats
