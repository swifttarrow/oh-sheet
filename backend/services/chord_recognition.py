"""Phase 3 post-processing — chroma-based chord recognition.

Basic Pitch hands us a clean polyphonic pitch stream but no harmonic
labels, so ``HarmonicAnalysis.chords`` stays empty through Phases 1
and 2. This module fills the gap by running a standard chroma + template
matching pass over the source waveform:

1. HPSS-harmonic the audio (``librosa.effects.harmonic``) so percussive
   transients don't smear the chroma vector.
2. ``chroma_cqt`` with ``bins_per_octave=36`` for finer pitch precision
   than the default STFT chroma.
3. Beat tracking (the same ``librosa.beat.beat_track`` call the tempo
   map uses) + ``librosa.util.sync`` to aggregate chroma across beats
   — chord changes naturally align to beat boundaries in most music.
4. Dot-product match against 60 normalized templates (12 major + 12
   minor triads + 12 major 7th + 12 minor 7th + 12 dominant 7th) and
   score each beat span against all templates.
5. Optional HMM Viterbi smoothing over the per-beat template scores,
   using a key-aware transition matrix so diatonic progressions are
   preferred over chromatic jumps and one-beat flickers are suppressed.
6. Collapse consecutive identical labels into ``RealtimeChordEvent``
   spans and emit them into the existing ``HarmonicAnalysis.chords``
   contract field.

Everything is late-imported so the module can be imported (and
unit-tested with ``pytest.importorskip``) on machines without librosa.
The transcribe wiring treats chord recognition as best-effort: on any
failure we fall back to an empty label list.

  * No inversion tracking — the root we return is the template root,
    not necessarily the lowest note.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.contracts import RealtimeChordEvent

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults — iteration will want to tune these against real audio fixtures.
# ---------------------------------------------------------------------------

DEFAULT_CHORD_MIN_SCORE = 0.55
DEFAULT_CHORD_HPSS_MARGIN = 3.0
DEFAULT_CHORD_SAMPLE_RATE = 22_050
DEFAULT_CHORD_HOP_LENGTH = 512
DEFAULT_CHORD_BINS_PER_OCTAVE = 36

# 7th chord templates and HMM smoothing.
DEFAULT_CHORD_SEVENTH_TEMPLATES_ENABLED = True
DEFAULT_CHORD_HMM_ENABLED = True
DEFAULT_CHORD_HMM_SELF_TRANSITION = 0.8
DEFAULT_CHORD_HMM_TEMPERATURE = 1.0

# Pitch class index → Harte note name (major roots; minor roots follow
# the same 12 names with a ``:min`` suffix).
_PITCH_NAMES: tuple[str, ...] = (
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
)


@dataclass
class ChordRecognitionStats:
    """Per-run summary of what the chord recognizer decided."""
    detected_count: int = 0        # number of labeled spans (excluding "N")
    no_chord_count: int = 0        # number of spans below the score threshold
    unique_labels: int = 0
    warnings: list[str] = field(default_factory=list)
    skipped: bool = False
    hmm_smoothed: bool = False

    def as_warnings(self) -> list[str]:
        if self.skipped:
            return ["chord recognition skipped (librosa or audio unavailable)"]
        out: list[str] = []
        if self.detected_count:
            out.append(
                f"chords: {self.detected_count} spans "
                f"({self.unique_labels} unique labels)"
            )
        elif self.no_chord_count:
            out.append("chords: no spans above score threshold")
        out.extend(self.warnings)
        return out


# ---------------------------------------------------------------------------
# Template construction — triads + optional 7th chords
# ---------------------------------------------------------------------------

def _build_chord_templates(
    *,
    seventh_enabled: bool = DEFAULT_CHORD_SEVENTH_TEMPLATES_ENABLED,
) -> tuple[Any, list[str], list[int]]:
    """Return (templates (N, 12), labels, roots).

    When ``seventh_enabled`` is True, N = 60 (24 triads + 36 sevenths).
    When False, N = 24 (12 major + 12 minor triads only).

    Templates weight root and fifth at 1.0, the third at 0.85, and (for
    7th chords) the seventh at 0.7 — the weaker seventh weight avoids
    false positives on plain triads whose chroma happens to bleed into
    the seventh bin. Each template is L2-normalized so the match score
    is a cosine.
    """
    import numpy as np  # noqa: PLC0415

    n_templates = 60 if seventh_enabled else 24
    templates = np.zeros((n_templates, 12), dtype=np.float32)
    labels: list[str] = []
    roots: list[int] = []

    # --- Major triads: root, +4, +7 ---
    for root in range(12):
        templates[root, root % 12] = 1.0
        templates[root, (root + 4) % 12] = 0.85
        templates[root, (root + 7) % 12] = 1.0
        labels.append(f"{_PITCH_NAMES[root]}:maj")
        roots.append(root)

    # --- Minor triads: root, +3, +7 ---
    for root in range(12):
        idx = 12 + root
        templates[idx, root % 12] = 1.0
        templates[idx, (root + 3) % 12] = 0.85
        templates[idx, (root + 7) % 12] = 1.0
        labels.append(f"{_PITCH_NAMES[root]}:min")
        roots.append(root)

    if seventh_enabled:
        # --- Major 7th: root, +4, +7, +11 ---
        for root in range(12):
            idx = 24 + root
            templates[idx, root % 12] = 1.0
            templates[idx, (root + 4) % 12] = 0.85
            templates[idx, (root + 7) % 12] = 1.0
            templates[idx, (root + 11) % 12] = 0.7
            labels.append(f"{_PITCH_NAMES[root]}:maj7")
            roots.append(root)

        # --- Minor 7th: root, +3, +7, +10 ---
        for root in range(12):
            idx = 36 + root
            templates[idx, root % 12] = 1.0
            templates[idx, (root + 3) % 12] = 0.85
            templates[idx, (root + 7) % 12] = 1.0
            templates[idx, (root + 10) % 12] = 0.7
            labels.append(f"{_PITCH_NAMES[root]}:min7")
            roots.append(root)

        # --- Dominant 7th: root, +4, +7, +10 ---
        for root in range(12):
            idx = 48 + root
            templates[idx, root % 12] = 1.0
            templates[idx, (root + 4) % 12] = 0.85
            templates[idx, (root + 7) % 12] = 1.0
            templates[idx, (root + 10) % 12] = 0.7
            labels.append(f"{_PITCH_NAMES[root]}:7")
            roots.append(root)

    norms = np.linalg.norm(templates, axis=1, keepdims=True)
    templates = templates / np.clip(norms, 1e-9, None)
    return templates, labels, roots


# Keep the old name available for backwards compatibility.
_build_triad_templates = _build_chord_templates


# ---------------------------------------------------------------------------
# Diatonic scale helpers (for HMM transition matrix)
# ---------------------------------------------------------------------------

# Major scale diatonic chords: I=maj, ii=min, iii=min, IV=maj, V=maj,
# vi=min, vii=dim (approximated as min for template matching purposes).
_MAJOR_DIATONIC_INTERVALS: tuple[tuple[int, str], ...] = (
    (0, "maj"),   # I
    (2, "min"),   # ii
    (4, "min"),   # iii
    (5, "maj"),   # IV
    (7, "maj"),   # V
    (9, "min"),   # vi
    (11, "min"),  # vii (dim ≈ min)
)

# Natural minor diatonic chords: i=min, ii=dim(≈min), III=maj, iv=min,
# v=min, VI=maj, VII=maj.
_MINOR_DIATONIC_INTERVALS: tuple[tuple[int, str], ...] = (
    (0, "min"),   # i
    (2, "min"),   # ii (dim ≈ min)
    (3, "maj"),   # III
    (5, "min"),   # iv
    (7, "min"),   # v
    (8, "maj"),   # VI
    (10, "maj"),  # VII
)


def _diatonic_labels_for_key(key_label: str) -> set[str]:
    """Return the set of Harte-notation labels that are diatonic to the key.

    Accepts ``"X:major"`` or ``"X:minor"`` where X is a pitch name
    from ``_PITCH_NAMES``. Unknown formats fall back to an empty set
    (the HMM will use uniform non-self transitions).

    The returned set includes both triad and 7th-chord variants of each
    diatonic degree so the HMM can assign medium transition probability
    to any diatonic chord regardless of extension.
    """
    parts = key_label.split(":")
    if len(parts) != 2:
        return set()

    key_name, mode = parts[0], parts[1].lower()
    if key_name not in _PITCH_NAMES:
        return set()

    key_root = _PITCH_NAMES.index(key_name)

    if mode == "major":
        intervals = _MAJOR_DIATONIC_INTERVALS
    elif mode == "minor":
        intervals = _MINOR_DIATONIC_INTERVALS
    else:
        return set()

    result: set[str] = set()
    for semitone_offset, quality in intervals:
        pitch_class = (key_root + semitone_offset) % 12
        name = _PITCH_NAMES[pitch_class]
        # Add the triad label.
        result.add(f"{name}:{quality}")
        # Add the 7th extensions of the same root and quality.
        if quality == "maj":
            result.add(f"{name}:maj7")
            result.add(f"{name}:7")    # dominant 7th shares major root
        elif quality == "min":
            result.add(f"{name}:min7")
    return result


# ---------------------------------------------------------------------------
# HMM Viterbi smoothing
# ---------------------------------------------------------------------------

def _smooth_chords_hmm(
    scores: Any,                      # np.ndarray (n_templates, n_spans)
    labels: list[str],
    roots: list[int],
    *,
    key_label: str = "C:major",
    self_transition: float = DEFAULT_CHORD_HMM_SELF_TRANSITION,
    temperature: float = DEFAULT_CHORD_HMM_TEMPERATURE,
) -> list[int]:
    """Return the best label index per span via Viterbi decoding.

    Parameters
    ----------
    scores : ndarray (n_templates, n_spans)
        Raw template match scores (cosine similarities).
    labels : list[str]
        Template labels corresponding to rows of *scores*.
    roots : list[int]
        Pitch-class root for each template.
    key_label : str
        Detected key in ``"X:major"`` / ``"X:minor"`` format. Used to
        build a key-aware transition matrix.
    self_transition : float
        Probability of staying on the same chord across beats.
    temperature : float
        Temperature applied to emission scores before the Viterbi pass.
        Values < 1 sharpen the distribution (more confident picks);
        values > 1 soften it (more influence from transitions).

    Returns
    -------
    list[int]
        Index into *labels* for each span (length = n_spans).
    """
    import numpy as np  # noqa: PLC0415

    n_states, n_spans = scores.shape
    if n_spans == 0:
        return []
    if n_states == 0:
        return [0] * n_spans

    # --- Transition matrix (n_states x n_states) in log-space ---
    diatonic = _diatonic_labels_for_key(key_label)

    trans = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(n_states):
        # Self-transition.
        trans[i, i] = self_transition

        # Distribute remaining probability among other states.
        remaining = 1.0 - self_transition
        if diatonic:
            # Count how many *other* states are diatonic vs not.
            n_diatonic = 0
            n_non_diatonic = 0
            for j in range(n_states):
                if j == i:
                    continue
                if labels[j] in diatonic:
                    n_diatonic += 1
                else:
                    n_non_diatonic += 1

            # Diatonic share: 0.15 of remaining, non-diatonic: 0.05.
            diatonic_share = 0.15
            non_diatonic_share = 0.05
            total_share = (
                diatonic_share * n_diatonic
                + non_diatonic_share * n_non_diatonic
            )
            if total_share == 0:
                # All non-self states are in one category — uniform fallback.
                total_share = 1.0
                diatonic_share = 1.0 / max(n_diatonic + n_non_diatonic, 1)
                non_diatonic_share = diatonic_share
            for j in range(n_states):
                if j == i:
                    continue
                if labels[j] in diatonic:
                    trans[i, j] = remaining * diatonic_share / total_share
                else:
                    trans[i, j] = remaining * non_diatonic_share / total_share
        else:
            # No key info — uniform non-self transitions.
            n_other = n_states - 1
            if n_other > 0:
                for j in range(n_states):
                    if j != i:
                        trans[i, j] = remaining / n_other

    # Ensure rows sum to 1 (numerical safety).
    row_sums = trans.sum(axis=1, keepdims=True)
    trans = trans / np.clip(row_sums, 1e-12, None)

    log_trans = np.log(np.clip(trans, 1e-12, None))

    # --- Emission scores in log-space ---
    # Apply temperature: emission = score^(1/temperature), then log.
    emission = scores.astype(np.float64)
    emission = np.clip(emission, 1e-12, None)
    if temperature > 0 and temperature != 1.0:
        emission = emission ** (1.0 / temperature)
    log_emission = np.log(np.clip(emission, 1e-12, None))

    # --- Viterbi forward pass ---
    # viterbi[s, t] = max log-prob of the best path ending in state s at
    # time t.
    viterbi = np.full((n_states, n_spans), -np.inf, dtype=np.float64)
    backptr = np.zeros((n_states, n_spans), dtype=np.intp)

    # Uniform initial prior (log 1/n_states).
    log_prior = np.log(1.0 / n_states)
    viterbi[:, 0] = log_prior + log_emission[:, 0]

    for t in range(1, n_spans):
        for s in range(n_states):
            # Best previous state for arriving at s at time t.
            candidates = viterbi[:, t - 1] + log_trans[:, s]
            best_prev = int(np.argmax(candidates))
            viterbi[s, t] = candidates[best_prev] + log_emission[s, t]
            backptr[s, t] = best_prev

    # --- Backtrace ---
    path: list[int] = [0] * n_spans
    path[-1] = int(np.argmax(viterbi[:, -1]))
    for t in range(n_spans - 2, -1, -1):
        path[t] = int(backptr[path[t + 1], t + 1])

    return path


# ---------------------------------------------------------------------------
# Core recognizer — operates on a preloaded waveform for testability
# ---------------------------------------------------------------------------

def recognize_chords_from_waveform(
    y: Any,                               # np.ndarray (samples,)
    sr: int,
    *,
    min_score: float = DEFAULT_CHORD_MIN_SCORE,
    hpss_margin: float = DEFAULT_CHORD_HPSS_MARGIN,
    hop_length: int = DEFAULT_CHORD_HOP_LENGTH,
    bins_per_octave: int = DEFAULT_CHORD_BINS_PER_OCTAVE,
    seventh_enabled: bool = DEFAULT_CHORD_SEVENTH_TEMPLATES_ENABLED,
    hmm_enabled: bool = DEFAULT_CHORD_HMM_ENABLED,
    hmm_self_transition: float = DEFAULT_CHORD_HMM_SELF_TRANSITION,
    hmm_temperature: float = DEFAULT_CHORD_HMM_TEMPERATURE,
    key_label: str = "C:major",
) -> tuple[list[RealtimeChordEvent], ChordRecognitionStats]:
    """Recognize chords directly from a loaded mono waveform.

    This is the unit-testable core — feed it a synthetic signal and it
    runs the same chroma / template pipeline as the file-path entry
    point without touching disk. Returns the list of labeled spans and
    a stats summary.

    Chord labels use Harte notation (``"C:maj"`` / ``"A:min"`` /
    ``"C:maj7"`` / ``"C:min7"`` / ``"C:7"``) so they drop straight
    into ``RealtimeChordEvent.label``. The ``root`` field holds the
    pitch class index (0-11, C=0).

    When ``hmm_enabled`` is True, a Viterbi pass smooths the per-beat
    chord labels using a key-aware transition matrix. On HMM failure
    the function falls back to raw argmax gracefully.
    """
    stats = ChordRecognitionStats()

    try:
        import librosa  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError:
        stats.skipped = True
        return [], stats

    if y is None or len(y) == 0:
        stats.skipped = True
        return [], stats

    # Duration guard: ≥ 0.25 s is roughly the minimum chroma_cqt needs
    # before it starts complaining about short input.
    duration = float(len(y) / sr) if sr else 0.0
    if duration < 0.25:
        stats.skipped = True
        return [], stats

    # HPSS the signal *for chroma only*. Beat tracking happens on the
    # raw waveform below — HPSS strips the percussive transients that
    # librosa.beat.beat_track actually locks onto, so feeding HPSS'd
    # audio into the beat tracker returns zero beats on material where
    # the raw tracker finds a clean pulse.
    try:
        y_h = librosa.effects.harmonic(y, margin=hpss_margin)
    except Exception as exc:  # noqa: BLE001
        log.warning("HPSS failed for chord recognition: %s", exc)
        stats.warnings.append(f"hpss failed: {exc}")
        y_h = y

    try:
        chroma = librosa.feature.chroma_cqt(
            y=y_h, sr=sr,
            hop_length=hop_length,
            bins_per_octave=bins_per_octave,
        )
    except Exception as exc:  # noqa: BLE001 — short / degenerate audio
        log.warning("chroma_cqt failed: %s", exc)
        stats.warnings.append(f"chroma_cqt failed: {exc}")
        stats.skipped = True
        return [], stats

    if chroma.size == 0 or chroma.shape[1] < 2:
        stats.skipped = True
        return [], stats

    # Beat-sync the chroma so each column corresponds to one beat span.
    # When beat tracking fails or finds <2 beats, fall back to a single
    # global span across the whole clip.
    try:
        _tempo, beat_frames = librosa.beat.beat_track(
            y=y, sr=sr, hop_length=hop_length,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("beat_track failed during chord recognition: %s", exc)
        beat_frames = np.array([], dtype=int)

    beat_frames = np.atleast_1d(beat_frames).astype(int)
    if beat_frames.size >= 2:
        # librosa.util.sync wants a Python sequence, not ndarray.
        beat_list: list[int] = [int(b) for b in beat_frames.tolist()]
        synced = librosa.util.sync(chroma, beat_list, aggregate=np.mean)
        span_times = librosa.frames_to_time(
            beat_frames, sr=sr, hop_length=hop_length,
        )
        # Append a trailing time for the last span = end of audio.
        span_times = np.concatenate([span_times, [duration]])
    else:
        synced = np.mean(chroma, axis=1, keepdims=True)
        span_times = np.array([0.0, duration], dtype=float)

    # synced shape is (12, N_spans). span_times has length N_spans + 1
    # when we have beats (one entry per span boundary including the
    # trailing end time), so make sure the two agree on N.
    n_spans = synced.shape[1]
    if span_times.size < n_spans + 1:
        # Degenerate: pad the trailing time.
        span_times = np.concatenate([
            span_times,
            np.full(n_spans + 1 - span_times.size, duration),
        ])
    elif span_times.size > n_spans + 1:
        span_times = span_times[: n_spans + 1]

    # Normalize chroma columns and score against templates.
    templates, labels, roots = _build_chord_templates(seventh_enabled=seventh_enabled)
    col_norms = np.linalg.norm(synced, axis=0, keepdims=True)
    synced_n = synced / np.clip(col_norms, 1e-9, None)
    scores = templates @ synced_n  # (n_templates, N_spans)

    # Pick best label per span — either HMM-smoothed or raw argmax.
    hmm_applied = False
    if hmm_enabled:
        try:
            best_indices = _smooth_chords_hmm(
                scores, labels, roots,
                key_label=key_label,
                self_transition=hmm_self_transition,
                temperature=hmm_temperature,
            )
            best_idx = np.array(best_indices, dtype=int)
            hmm_applied = True
        except Exception as exc:  # noqa: BLE001 — HMM failure falls back to argmax
            log.warning("HMM smoothing failed, falling back to argmax: %s", exc)
            stats.warnings.append(f"hmm smoothing failed: {exc}")
            best_idx = np.argmax(scores, axis=0)
    else:
        best_idx = np.argmax(scores, axis=0)

    best_score = scores[best_idx, np.arange(n_spans)]
    stats.hmm_smoothed = hmm_applied

    # Build raw per-span labels (or "N" for below-threshold spans).
    raw_labels: list[tuple[float, float, str, int, float]] = []
    for i in range(n_spans):
        start = float(span_times[i])
        end = float(span_times[i + 1])
        if end <= start or not math.isfinite(end - start):
            continue
        score_i = float(best_score[i])
        if score_i < min_score:
            raw_labels.append((start, end, "N", -1, score_i))
        else:
            idx = int(best_idx[i])
            raw_labels.append((start, end, labels[idx], roots[idx], score_i))

    # Collapse consecutive identical labels.
    collapsed: list[tuple[float, float, str, int, float]] = []
    for entry in raw_labels:
        if collapsed and collapsed[-1][2] == entry[2] and collapsed[-1][3] == entry[3]:
            prev = collapsed[-1]
            collapsed[-1] = (
                prev[0],
                entry[1],
                prev[2],
                prev[3],
                max(prev[4], entry[4]),
            )
        else:
            collapsed.append(entry)

    # Emit contract events for labeled spans only (skip "N").
    out: list[RealtimeChordEvent] = []
    seen_labels: set[str] = set()
    no_chord_count = 0
    for start, end, label, root, score in collapsed:
        if label == "N":
            no_chord_count += 1
            continue
        out.append(
            RealtimeChordEvent(
                time_sec=start,
                duration_sec=end - start,
                label=label,
                root=root,
                confidence=round(min(max(score, 0.0), 1.0), 3),
            )
        )
        seen_labels.add(label)

    stats.detected_count = len(out)
    stats.no_chord_count = no_chord_count
    stats.unique_labels = len(seen_labels)
    log.debug(
        "chord recognition: %d spans (%d unique, %d below threshold)",
        stats.detected_count,
        stats.unique_labels,
        stats.no_chord_count,
    )
    return out, stats


# ---------------------------------------------------------------------------
# File-path entry point — what the transcribe wiring calls
# ---------------------------------------------------------------------------

def recognize_chords(
    audio_path: Path,
    *,
    min_score: float = DEFAULT_CHORD_MIN_SCORE,
    hpss_margin: float = DEFAULT_CHORD_HPSS_MARGIN,
    sample_rate: int = DEFAULT_CHORD_SAMPLE_RATE,
    hop_length: int = DEFAULT_CHORD_HOP_LENGTH,
    bins_per_octave: int = DEFAULT_CHORD_BINS_PER_OCTAVE,
    seventh_enabled: bool = DEFAULT_CHORD_SEVENTH_TEMPLATES_ENABLED,
    hmm_enabled: bool = DEFAULT_CHORD_HMM_ENABLED,
    hmm_self_transition: float = DEFAULT_CHORD_HMM_SELF_TRANSITION,
    hmm_temperature: float = DEFAULT_CHORD_HMM_TEMPERATURE,
    key_label: str = "C:major",
) -> tuple[list[RealtimeChordEvent], ChordRecognitionStats]:
    """Load ``audio_path`` and return a chord label stream.

    Mirrors the graceful-degradation contract of the other Phase 3
    extractors: any failure (missing librosa, unreadable audio, short
    audio, chroma failure, ...) returns an empty label list with
    ``stats.skipped = True`` so the caller can carry on with no chord
    annotations.

    Parameters ``key_label``, ``hmm_enabled``, ``hmm_self_transition``,
    and ``hmm_temperature`` control the optional HMM Viterbi smoothing
    pass. ``seventh_enabled`` gates whether 7th chord templates are
    included alongside the 24 standard triads.
    """
    stats = ChordRecognitionStats()

    try:
        import librosa  # noqa: PLC0415
    except ImportError:
        stats.skipped = True
        return [], stats

    try:
        y, file_sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    except Exception as exc:  # noqa: BLE001 — bad audio shouldn't crash worker
        log.warning("librosa.load failed for chord recognition: %s", exc)
        stats.warnings.append(f"librosa.load failed: {exc}")
        stats.skipped = True
        return [], stats

    return recognize_chords_from_waveform(
        y, int(file_sr),
        min_score=min_score,
        hpss_margin=hpss_margin,
        hop_length=hop_length,
        bins_per_octave=bins_per_octave,
        seventh_enabled=seventh_enabled,
        hmm_enabled=hmm_enabled,
        hmm_self_transition=hmm_self_transition,
        hmm_temperature=hmm_temperature,
        key_label=key_label,
    )
