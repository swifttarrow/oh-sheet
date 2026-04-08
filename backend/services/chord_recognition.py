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
4. Dot-product match against 24 normalized templates (12 major + 12
   minor triads) and pick the best label per beat span.
5. Collapse consecutive identical labels into ``RealtimeChordEvent``
   spans and emit them into the existing ``HarmonicAnalysis.chords``
   contract field.

Everything is late-imported so the module can be imported (and
unit-tested with ``pytest.importorskip``) on machines without librosa.
The transcribe wiring treats chord recognition as best-effort: on any
failure we fall back to an empty label list.

This is deliberately v1 scope:
  * Triads only — no 7ths, sus, dim. Ambiguous harmonies fall through
    to whichever triad the chroma looks most like, which is usually
    the right triad anyway.
  * No key-aware smoothing — a purely local template match. A future
    pass could add an HMM over a key-conditioned template set.
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
# Template construction — 24 triads
# ---------------------------------------------------------------------------

def _build_triad_templates() -> tuple[Any, list[str], list[int]]:
    """Return (templates (24, 12), labels, roots).

    Templates weight root and fifth at 1.0 and the third at 0.85 — a
    compromise that nudges ambiguous vectors toward the right quality.
    Each template is L2-normalized so the match score is a cosine.
    """
    import numpy as np  # noqa: PLC0415

    templates = np.zeros((24, 12), dtype=np.float32)
    labels: list[str] = []
    roots: list[int] = []

    for root in range(12):
        # Major triad: root, +4 semitones, +7 semitones.
        templates[root, root % 12] = 1.0
        templates[root, (root + 4) % 12] = 0.85
        templates[root, (root + 7) % 12] = 1.0
        labels.append(f"{_PITCH_NAMES[root]}:maj")
        roots.append(root)

    for root in range(12):
        # Minor triad: root, +3 semitones, +7 semitones.
        idx = 12 + root
        templates[idx, root % 12] = 1.0
        templates[idx, (root + 3) % 12] = 0.85
        templates[idx, (root + 7) % 12] = 1.0
        labels.append(f"{_PITCH_NAMES[root]}:min")
        roots.append(root)

    norms = np.linalg.norm(templates, axis=1, keepdims=True)
    templates = templates / np.clip(norms, 1e-9, None)
    return templates, labels, roots


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
) -> tuple[list[RealtimeChordEvent], ChordRecognitionStats]:
    """Recognize chords directly from a loaded mono waveform.

    This is the unit-testable core — feed it a synthetic signal and it
    runs the same chroma / template pipeline as the file-path entry
    point without touching disk. Returns the list of labeled spans and
    a stats summary.

    Chord labels use Harte notation (``"C:maj"`` / ``"A:min"``) so they
    drop straight into ``RealtimeChordEvent.label``. The ``root`` field
    holds the pitch class index (0–11, C=0).
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
    templates, labels, roots = _build_triad_templates()
    col_norms = np.linalg.norm(synced, axis=0, keepdims=True)
    synced_n = synced / np.clip(col_norms, 1e-9, None)
    scores = templates @ synced_n  # (24, N_spans)
    best_idx = np.argmax(scores, axis=0)
    best_score = scores[best_idx, np.arange(n_spans)]

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
) -> tuple[list[RealtimeChordEvent], ChordRecognitionStats]:
    """Load ``audio_path`` and return a chord label stream.

    Mirrors the graceful-degradation contract of the other Phase 3
    extractors: any failure (missing librosa, unreadable audio, short
    audio, chroma failure, …) returns an empty label list with
    ``stats.skipped = True`` so the caller can carry on with no chord
    annotations.
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
    )
