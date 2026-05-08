"""Tier 2 structural-fidelity metrics for the Phase 7 eval ladder.

Tier 2 measures whether the transcription gets the **musical bones** right:
key, tempo, beat grid, chord progression, and section boundaries. Most
metrics here are reference-free in our sense — the "reference" is an
audio-side analysis pass on the *original input audio* (the same key /
beat / chord pipeline the transcribe stage already runs in production),
and the "estimate" is the same pass applied to the FluidSynth re-synth
of the engraved MIDI. Any divergence between the two attribution chains
is attributable to losses in arrange / humanize / engrave.

See ``docs/research/transcription-improvement-strategy.md`` Part III §2.2
for the metric definitions and citations.

The five metrics:

* :func:`key_score` — ``mir_eval.key.weighted_score`` (perfect=1, parallel /
  relative / dominant errors weighted partial). Compares
  :func:`backend.services.key_estimation.estimate_key_from_waveform` on
  input audio vs. resynth audio.
* :func:`tempo_score` — ``mir_eval.tempo.detection`` P-score, ±4% strict.
  Computes the median inter-beat tempo from each side's beat tracker.
* :func:`beat_score` — ``mir_eval.beat.f_measure`` at ±70 ms.
* :func:`chord_score` — ``mir_eval.chord.evaluate`` MIREX mode (already
  the Tier 2 RF metric at :func:`eval.tier_rf.chord_rf_score`; re-exported
  here so the harness has one Tier 2 entry point).
* :func:`section_score` — ``mir_eval.segment.detection`` HR3F at ±3 s, when
  both sides emit non-empty section lists. Returns 0 with a note when one
  side has no sections (the most common case — Oh Sheet's transcribe stage
  doesn't emit them by default).

Heavy deps (``librosa``, ``mir_eval``) are imported lazily inside each
metric so this module is cheap to import and easy to unit-test in
isolation. Each function tolerates missing librosa / short audio /
empty event lists by returning a 0.0 score plus an entry in the
returned ``notes`` list — same graceful-degradation contract as
:mod:`eval.tier_rf`.

The :func:`compute_tier2` entry point runs all five metrics for one
``(input_audio_path, engraved_midi_bytes)`` pair, mirroring the
``compute_tier_rf`` orchestration shape so a future ``scripts/eval.py``
can iterate all tiers uniformly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval.tier_rf import (
    CHROMA_SR,
    FLUIDSYNTH_SR,
    chord_rf_score,
    fluidsynth_resynth,
)

log = logging.getLogger(__name__)


# Beat F1 tolerance from strategy doc §2.2 — 70 ms is the de-facto MIREX
# default for beat tracking (Davies 2014).
DEFAULT_BEAT_TOLERANCE_SEC = 0.070

# Section-boundary HR3F tolerance — strategy doc Part III §2.2 (SALAMI).
DEFAULT_SECTION_TOLERANCE_SEC = 3.0

# Tempo P-score relative tolerance (4%) — mir_eval default.
DEFAULT_TEMPO_TOLERANCE = 0.04


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class Tier2Result:
    """Per-song Tier 2 scores produced by :func:`compute_tier2`.

    Each field is in ``[0, 1]`` with 1 meaning "input and resynth fully
    agree on this structural attribute". The composite ``mean_score``
    is the simple unweighted mean of the four core metrics
    (key / tempo / beat / chord); sections are excluded from the
    composite because Oh Sheet's transcribe stage doesn't emit them
    yet (sections is reported alongside but not folded into the mean).
    """

    key_score: float
    tempo_score: float
    beat_score: float
    chord_score: float
    section_score: float
    n_input_beats: int
    n_resynth_beats: int
    n_input_chords: int
    n_resynth_chords: int
    n_input_sections: int
    n_resynth_sections: int
    notes: list[str] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        """Unweighted mean of key / tempo / beat / chord."""
        return (
            self.key_score
            + self.tempo_score
            + self.beat_score
            + self.chord_score
        ) / 4.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "key_score": round(self.key_score, 4),
            "tempo_score": round(self.tempo_score, 4),
            "beat_score": round(self.beat_score, 4),
            "chord_score": round(self.chord_score, 4),
            "section_score": round(self.section_score, 4),
            "mean_score": round(self.mean_score, 4),
            "n_input_beats": self.n_input_beats,
            "n_resynth_beats": self.n_resynth_beats,
            "n_input_chords": self.n_input_chords,
            "n_resynth_chords": self.n_resynth_chords,
            "n_input_sections": self.n_input_sections,
            "n_resynth_sections": self.n_resynth_sections,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Key — mir_eval.key.weighted_score on audio-side estimates
# ---------------------------------------------------------------------------

def key_score(
    input_audio: tuple[Any, int],
    resynth_audio: tuple[Any, int],
) -> tuple[float, list[str]]:
    """``mir_eval.key.weighted_score`` between key estimates on each side.

    Both sides run through
    :func:`backend.services.key_estimation.estimate_key_from_waveform`
    so the recognizer-internal bias is symmetric. Score is in ``[0, 1]``
    with the standard MIREX partial credits for parallel (0.2) /
    relative (0.3) / fifth (0.5) errors.
    """
    notes: list[str] = []
    try:
        import mir_eval  # noqa: PLC0415

        from backend.services.key_estimation import (  # noqa: PLC0415
            estimate_key_from_waveform,
        )
    except ImportError as exc:
        notes.append(f"key_score: import failed: {exc}")
        return 0.0, notes

    in_y, in_sr = input_audio
    rs_y, rs_sr = resynth_audio
    try:
        in_key, _ = estimate_key_from_waveform(in_y, in_sr)
        rs_key, _ = estimate_key_from_waveform(rs_y, rs_sr)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"key_score: estimation failed: {exc}")
        return 0.0, notes

    in_label = _harte_to_mir_eval_key(in_key)
    rs_label = _harte_to_mir_eval_key(rs_key)
    if not in_label or not rs_label:
        notes.append(f"key_score: unparseable label in={in_key!r} rs={rs_key!r}")
        return 0.0, notes

    try:
        score = float(mir_eval.key.weighted_score(in_label, rs_label))
    except Exception as exc:  # noqa: BLE001
        notes.append(f"key_score: mir_eval failed: {exc}")
        return 0.0, notes
    return max(0.0, min(1.0, score)), notes


def _harte_to_mir_eval_key(label: str) -> str | None:
    """Convert ``"C:major"`` (ours) → ``"C major"`` (mir_eval expects space).

    ``backend/services/key_estimation.py`` emits Harte-style ``Root:mode``
    strings; ``mir_eval.key`` accepts ``"Root mode"`` with a space. Returns
    None for unparseable input so callers can record a graceful failure.
    """
    if not label or ":" not in label:
        return None
    root, mode = label.split(":", 1)
    if mode not in ("major", "minor"):
        return None
    return f"{root} {mode}"


# ---------------------------------------------------------------------------
# Tempo — median inter-beat tempo + mir_eval.tempo.detection
# ---------------------------------------------------------------------------

def tempo_score(
    input_beats: list[float],
    resynth_beats: list[float],
    *,
    tolerance: float = DEFAULT_TEMPO_TOLERANCE,
) -> tuple[float, list[str]]:
    """Tempo P-score between two beat sequences.

    Computes the median inter-beat interval on each side, converts to
    BPM, and runs ``mir_eval.tempo.detection`` with a single-tempo pair
    plus a 1.0 strength weight. The default tolerance is 4% relative —
    half-tempo / double-tempo errors will score 0 (octave-relaxed
    reporting is left to nightly/§4 because the gate metric should be
    strict).
    """
    notes: list[str] = []
    try:
        import mir_eval  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:
        notes.append(f"tempo_score: import failed: {exc}")
        return 0.0, notes

    in_bpm = _bpm_from_beats(input_beats)
    rs_bpm = _bpm_from_beats(resynth_beats)
    if in_bpm is None or rs_bpm is None:
        notes.append(
            f"tempo_score: insufficient beats (in={len(input_beats)}, "
            f"rs={len(resynth_beats)})"
        )
        return 0.0, notes

    # mir_eval.tempo.detection wants two reference tempi (e.g. one for
    # half, one for double) with a relative strength. We treat each side
    # as a single-tempo estimate, weight 1.0.
    ref_tempi = np.array([in_bpm, in_bpm * 2.0])
    ref_weight = 1.0  # all weight on first tempo
    est_tempi = np.array([rs_bpm, rs_bpm * 2.0])
    try:
        p_score, _, _ = mir_eval.tempo.detection(
            ref_tempi, ref_weight, est_tempi, tol=tolerance,
        )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"tempo_score: mir_eval failed: {exc}")
        return 0.0, notes
    return max(0.0, min(1.0, float(p_score))), notes


def _bpm_from_beats(beats: list[float]) -> float | None:
    """Median inter-beat tempo in BPM, or None for <2 beats."""
    if len(beats) < 2:
        return None
    import statistics as _s  # noqa: PLC0415

    deltas = [beats[i + 1] - beats[i] for i in range(len(beats) - 1)]
    deltas = [d for d in deltas if d > 1e-3]
    if not deltas:
        return None
    median_dt = _s.median(deltas)
    return 60.0 / median_dt if median_dt > 0 else None


# ---------------------------------------------------------------------------
# Beat — mir_eval.beat.f_measure
# ---------------------------------------------------------------------------

def beat_score(
    input_beats: list[float],
    resynth_beats: list[float],
    *,
    tolerance_sec: float = DEFAULT_BEAT_TOLERANCE_SEC,
) -> tuple[float, list[str]]:
    """``mir_eval.beat.f_measure`` between two beat sequences."""
    notes: list[str] = []
    try:
        import mir_eval  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:
        notes.append(f"beat_score: import failed: {exc}")
        return 0.0, notes

    if len(input_beats) < 2 or len(resynth_beats) < 2:
        notes.append(
            f"beat_score: insufficient beats (in={len(input_beats)}, "
            f"rs={len(resynth_beats)})"
        )
        return 0.0, notes

    ref = np.array(input_beats, dtype=float)
    est = np.array(resynth_beats, dtype=float)
    try:
        f = float(mir_eval.beat.f_measure(ref, est, f_measure_threshold=tolerance_sec))
    except Exception as exc:  # noqa: BLE001
        notes.append(f"beat_score: mir_eval failed: {exc}")
        return 0.0, notes
    return max(0.0, min(1.0, f)), notes


# ---------------------------------------------------------------------------
# Chord — re-export tier_rf.chord_rf_score under the Tier 2 name
# ---------------------------------------------------------------------------

def chord_score(
    input_audio: tuple[Any, int],
    resynth_audio: tuple[Any, int],
    *,
    key_label: str = "C:major",
) -> tuple[float, int, int, list[str]]:
    """``mir_eval.chord.evaluate`` MIREX score, audio-anchored both sides.

    Identical to :func:`eval.tier_rf.chord_rf_score`. Re-exported here so
    Tier 2's public surface is self-contained — the harness reaches for
    ``tier2_structural.chord_score`` without a cross-tier import.
    """
    return chord_rf_score(input_audio, resynth_audio, key_label=key_label)


# ---------------------------------------------------------------------------
# Section — mir_eval.segment.detection HR3F
# ---------------------------------------------------------------------------

def section_score(
    input_section_boundaries_sec: list[float],
    resynth_section_boundaries_sec: list[float],
    *,
    tolerance_sec: float = DEFAULT_SECTION_TOLERANCE_SEC,
) -> tuple[float, list[str]]:
    """Section-boundary HR3F (F-measure at ±3 s).

    Both inputs are sorted lists of section-start instants in seconds.
    Returns 0.0 when either side has fewer than 2 boundaries — section
    detection on a single boundary is degenerate, and Oh Sheet's
    transcribe stage emits empty sections by default
    (``backend/services/transcribe_result.py``); the metric reports the
    miss honestly via a note.
    """
    notes: list[str] = []
    try:
        import mir_eval  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:
        notes.append(f"section_score: import failed: {exc}")
        return 0.0, notes

    n_in = len(input_section_boundaries_sec)
    n_rs = len(resynth_section_boundaries_sec)
    if n_in < 2 or n_rs < 2:
        notes.append(
            f"section_score: insufficient sections (in={n_in}, rs={n_rs})"
        )
        return 0.0, notes

    ref = np.array(sorted(input_section_boundaries_sec), dtype=float)
    est = np.array(sorted(resynth_section_boundaries_sec), dtype=float)
    try:
        f, _, _ = mir_eval.segment.detection(
            _boundaries_to_intervals(ref),
            _boundaries_to_intervals(est),
            window=tolerance_sec,
        )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"section_score: mir_eval failed: {exc}")
        return 0.0, notes
    return max(0.0, min(1.0, float(f))), notes


def _boundaries_to_intervals(boundaries: Any) -> Any:
    """Convert a 1-D boundary array to mir_eval segment-interval shape ``(N-1, 2)``."""
    import numpy as np  # noqa: PLC0415

    arr = np.asarray(boundaries, dtype=float)
    return np.column_stack((arr[:-1], arr[1:]))


# ---------------------------------------------------------------------------
# Top-level entrypoint
# ---------------------------------------------------------------------------

def compute_tier2(
    input_audio_path: Path,
    engraved_midi_bytes: bytes,
    *,
    key_label: str = "C:major",
    input_sections_sec: list[float] | None = None,
    resynth_sections_sec: list[float] | None = None,
    fluidsynth_bin: str | None = None,
    soundfont_path: Path | None = None,
) -> Tier2Result:
    """Run all five Tier 2 metrics for one (audio, MIDI) pair.

    Loads the input audio at :data:`eval.tier_rf.CHROMA_SR`, FluidSynth-
    renders the engraved MIDI, then computes:

    * key (audio-side estimator on both sides)
    * tempo (median inter-beat from each side's beat tracker)
    * beat F1 at ±70 ms
    * chord (mir_eval.chord MIREX, re-exported from tier_rf)
    * section (only if both ``*_sections_sec`` lists provided and ≥2 each)

    Sections default to empty — Oh Sheet's transcribe stage doesn't emit
    them yet — and the metric records a graceful 0.0 + note. Pass real
    section lists when running against a structural reference (Phase 3+
    pop_eval_v1) or against the future RefineService output.
    """
    import librosa  # noqa: PLC0415

    in_y, in_sr = librosa.load(str(input_audio_path), sr=CHROMA_SR, mono=True)
    rs_y, rs_sr = fluidsynth_resynth(
        engraved_midi_bytes,
        sample_rate=FLUIDSYNTH_SR,
        soundfont_path=soundfont_path,
        fluidsynth_bin=fluidsynth_bin,
    )
    if rs_sr != CHROMA_SR:
        rs_y = librosa.resample(rs_y, orig_sr=rs_sr, target_sr=CHROMA_SR)
        rs_sr = CHROMA_SR

    # Beat tracking on both sides via librosa — keeps this module
    # importable without Beat This! and avoids running the full
    # ``tempo_map_and_downbeats_from_audio_path`` pipeline (which writes
    # to disk and pulls in the settings module).
    in_beats = _track_beats(in_y, in_sr)
    rs_beats = _track_beats(rs_y, rs_sr)

    notes: list[str] = []

    k_score, k_notes = key_score((in_y, in_sr), (rs_y, rs_sr))
    notes.extend(k_notes)

    t_score, t_notes = tempo_score(in_beats, rs_beats)
    notes.extend(t_notes)

    b_score, b_notes = beat_score(in_beats, rs_beats)
    notes.extend(b_notes)

    c_score, n_in_chords, n_rs_chords, c_notes = chord_score(
        (in_y, in_sr), (rs_y, rs_sr), key_label=key_label,
    )
    notes.extend(c_notes)

    in_secs = list(input_sections_sec or [])
    rs_secs = list(resynth_sections_sec or [])
    s_score, s_notes = section_score(in_secs, rs_secs)
    notes.extend(s_notes)

    return Tier2Result(
        key_score=k_score,
        tempo_score=t_score,
        beat_score=b_score,
        chord_score=c_score,
        section_score=s_score,
        n_input_beats=len(in_beats),
        n_resynth_beats=len(rs_beats),
        n_input_chords=n_in_chords,
        n_resynth_chords=n_rs_chords,
        n_input_sections=len(in_secs),
        n_resynth_sections=len(rs_secs),
        notes=notes,
    )


def _track_beats(y: Any, sr: int) -> list[float]:
    """Beat-track via librosa; returns sorted beat instants in seconds."""
    try:
        import librosa  # noqa: PLC0415
    except ImportError:
        return []
    try:
        _tempo, frames = librosa.beat.beat_track(y=y, sr=sr)
        times = librosa.frames_to_time(frames, sr=sr)
    except Exception as exc:  # noqa: BLE001
        log.debug("librosa.beat.beat_track failed: %s", exc)
        return []
    return [float(t) for t in times]
