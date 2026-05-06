"""Reference-free metrics for the Phase 0 mini-eval.

Three metrics, all per-song, all in ``[0, 1]``, all computed without paired
ground-truth MIDI/score. See
``docs/research/transcription-improvement-implementation-plan.md`` §Phase 0
for the goal and ``transcription-improvement-strategy.md`` Part III §2 for
the metric definitions.

* :func:`chord_rf_score` — Tier 2 RF: ``mir_eval.chord`` MIREX score between
  chord recognition on the input audio and chord recognition on the
  FluidSynth re-synth of the engraved MIDI.
* :func:`playability_rf_score` — Tier 3 RF: fraction of per-hand chord
  groupings with span ≤14 semitones AND ≤5 notes per hand. Computed from
  the post-arrange ``PianoScore``.
* :func:`chroma_rf_score` — Tier 4 RF: mean per-beat ``chroma_cqt`` cosine
  between the input audio and the FluidSynth re-synth of the engraved MIDI.
  Beats are tracked once on the input and shared between both chromas so
  timestamps line up.

The :func:`compute_tier_rf` top-level entry point runs all three and
returns a :class:`TierRfResult`. The CLI (``scripts/eval_mini.py``)
calls :func:`compute_tier_rf` once per manifest song.

Heavy deps (``librosa``, ``mir_eval``, ``soundfile``, ``pretty_midi``) are
imported lazily inside the metric functions so this module is cheap to
import and easy to unit-test in isolation. The metric implementations
also degrade gracefully — missing librosa or short / silent audio
produces a 0.0 score plus an entry in ``TierRfResult.notes`` rather than
raising.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shared.contracts import PianoScore, ScoreNote

log = logging.getLogger(__name__)

# Sample rate used for chord recognition + chroma. Matches
# ``backend.services.chord_recognition.DEFAULT_CHORD_SAMPLE_RATE`` so the
# input-audio pass is byte-identical to a production transcribe-stage call.
CHROMA_SR = 22_050

# Sample rate FluidSynth renders at by default. ``scripts/eval_transcription.py``
# uses 44_100 here for the same reason: the bundled ``TimGM6mb.sf2`` is
# tuned to 44.1 kHz and downsampling for chord-recognition is cheap.
FLUIDSYNTH_SR = 44_100

# Chroma + beat-tracking hop length. Matches the value
# ``chord_recognition.recognize_chords_from_waveform`` uses internally so the
# beat-frame indices we compute here align 1:1 with how the recognizer
# would have bucketed them.
CHROMA_HOP_LENGTH = 512

# Tier 3 RF playability constraints.
PLAYABILITY_MAX_SPAN_SEMITONES = 14
PLAYABILITY_MAX_NOTES_PER_HAND = 5

# Two notes share a chord grouping when their ``onset_beat`` values are
# within this tolerance. Quantization in arrange already snaps onsets to a
# 1/16th-note grid, so a tolerance well below that is sufficient.
_CHORD_GROUP_BEAT_EPS = 1e-6


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TierRfResult:
    """Per-song reference-free scores produced by :func:`compute_tier_rf`."""

    chord_rf: float                 # Tier 2 RF — mir_eval.chord MIREX score
    playability_rf: float           # Tier 3 RF — playability fraction
    chroma_rf: float                # Tier 4 RF — mean per-beat chroma cosine
    n_chord_segments_input: int     # diagnostic: chords detected on input
    n_chord_segments_resynth: int   # diagnostic: chords detected on resynth
    n_playable_chords: int          # diagnostic: numerator of playability_rf
    n_total_chords: int             # diagnostic: denominator of playability_rf
    n_beats: int                    # diagnostic: beats used for chroma bucketing
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "chord_rf": round(self.chord_rf, 4),
            "playability_rf": round(self.playability_rf, 4),
            "chroma_rf": round(self.chroma_rf, 4),
            "n_chord_segments_input": self.n_chord_segments_input,
            "n_chord_segments_resynth": self.n_chord_segments_resynth,
            "n_playable_chords": self.n_playable_chords,
            "n_total_chords": self.n_total_chords,
            "n_beats": self.n_beats,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# FluidSynth re-synth — extracted from scripts/eval_transcription.py
# ---------------------------------------------------------------------------

def fluidsynth_resynth(
    midi_bytes: bytes,
    *,
    sample_rate: int = FLUIDSYNTH_SR,
    soundfont_path: Path | None = None,
    fluidsynth_bin: str | None = None,
) -> tuple[Any, int]:
    """Render MIDI bytes to a ``(mono float32 numpy array, sample_rate)`` tuple.

    Mirrors the ``_synthesize`` helper in ``scripts/eval_transcription.py``:
    we shell out to the ``fluidsynth`` CLI rather than binding pyFluidSynth
    so the harness works on any environment with a fluidsynth install
    (``brew install fluid-synth`` on macOS, ``apt install fluidsynth`` on
    Debian/Ubuntu) and zero Python wheel churn.

    The default soundfont is the ``TimGM6mb.sf2`` bundled inside the
    ``pretty_midi`` wheel — present in every environment that runs the
    transcribe stage, so callers don't need to provision a soundfont
    separately.
    """
    import numpy as np  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415

    if fluidsynth_bin is None:
        fluidsynth_bin = shutil.which("fluidsynth")
        if not fluidsynth_bin:
            raise RuntimeError(
                "fluidsynth binary not found on PATH. Install with "
                "`brew install fluid-synth` (macOS) or `apt install fluidsynth` "
                "(Debian/Ubuntu)."
            )
    if soundfont_path is None:
        import pretty_midi  # noqa: PLC0415
        soundfont_path = Path(pretty_midi.__file__).parent / "TimGM6mb.sf2"
        if not soundfont_path.is_file():
            raise RuntimeError(
                f"Bundled TimGM6mb.sf2 not found at {soundfont_path}; "
                "upgrade pretty_midi or pass an explicit soundfont_path."
            )

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        midi_path = td_path / "in.mid"
        wav_path = td_path / "out.wav"
        midi_path.write_bytes(midi_bytes)
        proc = subprocess.run(
            [
                fluidsynth_bin,
                "-ni",
                "-g", "1.0",
                "-r", str(sample_rate),
                "-F", str(wav_path),
                str(soundfont_path),
                str(midi_path),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"fluidsynth failed (rc={proc.returncode}): "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )
        if not wav_path.is_file():
            raise RuntimeError(
                "fluidsynth produced no output file — likely a bad MIDI payload."
            )
        data, sr = sf.read(str(wav_path))
        if data.ndim > 1:
            # FluidSynth writes stereo by default; flatten to mono so
            # chord_recognition + chroma operate on a single channel.
            data = data.mean(axis=1)
        return np.asarray(data, dtype=np.float32), int(sr)


# ---------------------------------------------------------------------------
# Tier 2 RF — chord-progression accuracy via mir_eval.chord
# ---------------------------------------------------------------------------

def chord_rf_score(
    input_audio: tuple[Any, int],
    resynth_audio: tuple[Any, int],
    *,
    key_label: str = "C:major",
) -> tuple[float, int, int, list[str]]:
    """MIREX score between chord recognition on input vs. resynth audio.

    Both sides run through the production
    :func:`backend.services.chord_recognition.recognize_chords_from_waveform`
    with the same ``key_label`` prior so the HMM transition matrix is
    symmetric — any drop is attributable to harmonic divergence between
    the input and the engraved MIDI, not to recognizer-internal bias.

    Returns ``(mirex_score, n_input_segments, n_resynth_segments, notes)``.
    On any failure (missing librosa, empty chord list, mir_eval exception)
    returns ``(0.0, n_in, n_rs, [reason])`` rather than raising — Phase 0's
    acceptance criterion is "no exceptions, all metrics in [0,1]".
    """
    notes: list[str] = []

    try:
        import mir_eval  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        from backend.services.chord_recognition import (  # noqa: PLC0415
            recognize_chords_from_waveform,
        )
    except ImportError as exc:
        notes.append(f"chord_rf: import failed: {exc}")
        return 0.0, 0, 0, notes

    in_y, in_sr = input_audio
    rs_y, rs_sr = resynth_audio

    in_chords, _ = recognize_chords_from_waveform(in_y, in_sr, key_label=key_label)
    rs_chords, _ = recognize_chords_from_waveform(rs_y, rs_sr, key_label=key_label)

    if not in_chords or not rs_chords:
        notes.append(
            f"chord_rf: empty chord list (in={len(in_chords)}, rs={len(rs_chords)})"
        )
        return 0.0, len(in_chords), len(rs_chords), notes

    ref_intervals, ref_labels = _events_to_intervals_and_labels(in_chords)
    est_intervals, est_labels = _events_to_intervals_and_labels(rs_chords)

    # mir_eval.chord.evaluate auto-aligns intervals via merge_labeled_intervals
    # internally, so the two segment lists don't need to share boundaries.
    # The 'mirex' key in the returned dict is the duration-weighted MIREX
    # score (1 if pred and ref share ≥3 chord tones, else 0).
    try:
        scores = mir_eval.chord.evaluate(
            ref_intervals, ref_labels, est_intervals, est_labels,
        )
    except Exception as exc:  # noqa: BLE001 — mir_eval can raise on edge cases
        notes.append(f"chord_rf: mir_eval failed: {exc}")
        return 0.0, len(in_chords), len(rs_chords), notes

    raw = scores.get("mirex", 0.0)
    try:
        mirex = float(np.asarray(raw).item() if hasattr(raw, "item") else raw)
    except (TypeError, ValueError):
        mirex = 0.0
    # Clamp to [0, 1] — mir_eval is well-behaved but defensively clip
    # anyway so downstream JSON consumers can rely on the bound.
    mirex = max(0.0, min(1.0, mirex))
    return mirex, len(in_chords), len(rs_chords), notes


def _events_to_intervals_and_labels(
    events: list,  # list[RealtimeChordEvent]
) -> tuple[Any, list[str]]:
    """Convert RealtimeChordEvent list to ``(intervals_array, labels_list)``.

    mir_eval expects intervals as an ``(N, 2)`` float ndarray of
    ``[start, end]`` pairs in seconds, and labels as a parallel list of
    Harte-notation strings. Our chord recognizer emits exactly that
    shape, so the conversion is direct.
    """
    import numpy as np  # noqa: PLC0415

    intervals = np.array(
        [[e.time_sec, e.time_sec + e.duration_sec] for e in events],
        dtype=float,
    )
    labels = [e.label for e in events]
    return intervals, labels


# ---------------------------------------------------------------------------
# Tier 3 RF — playability fraction
# ---------------------------------------------------------------------------

def playability_rf_score(
    score: PianoScore,
    *,
    max_span_semitones: int = PLAYABILITY_MAX_SPAN_SEMITONES,
    max_notes_per_hand: int = PLAYABILITY_MAX_NOTES_PER_HAND,
) -> tuple[float, int, int]:
    """Fraction of per-hand chord groupings within span+density limits.

    A "chord grouping" is the set of notes in one hand that share an
    ``onset_beat`` (within ``_CHORD_GROUP_BEAT_EPS``). A grouping passes
    the playability gate when ``len(group) <= max_notes_per_hand`` AND
    ``max(pitch) - min(pitch) <= max_span_semitones``.

    Hands are scored independently so a song with a sparse RH and a busy
    LH gets the right diagnostic signal — a "hand-impossible" event in
    one hand isn't laundered by an empty grouping in the other. Total =
    RH groupings + LH groupings; numerator = playable groupings across
    both hands.

    Returns ``(fraction, n_playable, n_total)``. A score with zero notes
    in both hands returns ``(0.0, 0, 0)``: the score is meaningless,
    and we let the caller/aggregate filter it out via
    ``n_total == 0`` rather than divide by zero.
    """
    rh_groups = _chord_groups(score.right_hand)
    lh_groups = _chord_groups(score.left_hand)
    n_total = len(rh_groups) + len(lh_groups)
    if n_total == 0:
        return 0.0, 0, 0

    n_playable = sum(
        1
        for group in (*rh_groups, *lh_groups)
        if _is_playable(group, max_span_semitones, max_notes_per_hand)
    )
    return n_playable / n_total, n_playable, n_total


def _chord_groups(notes: list[ScoreNote]) -> list[list[ScoreNote]]:
    """Bucket notes by ``onset_beat`` (within tolerance)."""
    if not notes:
        return []
    sorted_notes = sorted(notes, key=lambda n: n.onset_beat)
    groups: list[list[ScoreNote]] = [[sorted_notes[0]]]
    for n in sorted_notes[1:]:
        if abs(n.onset_beat - groups[-1][0].onset_beat) <= _CHORD_GROUP_BEAT_EPS:
            groups[-1].append(n)
        else:
            groups.append([n])
    return groups


def _is_playable(
    group: list[ScoreNote],
    max_span_semitones: int,
    max_notes_per_hand: int,
) -> bool:
    if len(group) > max_notes_per_hand:
        return False
    pitches = [n.pitch for n in group]
    return max(pitches) - min(pitches) <= max_span_semitones


# ---------------------------------------------------------------------------
# Tier 4 RF — chroma cosine, beat-bucketed
# ---------------------------------------------------------------------------

def chroma_rf_score(
    input_audio: tuple[Any, int],
    resynth_audio: tuple[Any, int],
) -> tuple[float, int, list[str]]:
    """Mean per-beat chroma cosine between input and resynth audio.

    1. Beat-track the input audio (librosa.beat.beat_track on the raw
       waveform — same call ``chord_recognition`` uses internally).
    2. Compute ``chroma_cqt`` on a HPSS-harmonic version of each audio
       (suppresses percussive transients that smear the chroma vector).
    3. Convert the input's beat frames to seconds, then to frame indices
       in each audio's chroma matrix; sync both chromas at the same
       beat boundaries via ``librosa.util.sync``.
    4. For each beat span, compute the cosine similarity of the two
       12-dim chroma vectors. Average across spans.

    Sharing the *input's* beat times across both chromas keeps the
    per-beat comparison time-aligned even when the resynth audio's
    duration differs slightly from the input.

    Returns ``(mean_cosine, n_beats_used, notes)``. Falls back to a
    single global comparison when beat tracking finds <2 beats.
    """
    notes: list[str] = []

    try:
        import librosa  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:
        notes.append(f"chroma_rf: import failed: {exc}")
        return 0.0, 0, notes

    in_y, in_sr = input_audio
    rs_y, rs_sr = resynth_audio

    if in_sr != CHROMA_SR:
        in_y = librosa.resample(in_y, orig_sr=in_sr, target_sr=CHROMA_SR)
        in_sr = CHROMA_SR
    if rs_sr != CHROMA_SR:
        rs_y = librosa.resample(rs_y, orig_sr=rs_sr, target_sr=CHROMA_SR)
        rs_sr = CHROMA_SR

    if len(in_y) == 0 or len(rs_y) == 0:
        notes.append("chroma_rf: empty audio")
        return 0.0, 0, notes

    chroma_in = _chroma_cqt(in_y, in_sr)
    chroma_rs = _chroma_cqt(rs_y, rs_sr)
    if chroma_in.size == 0 or chroma_rs.size == 0 or chroma_in.shape[1] < 2:
        notes.append("chroma_rf: chroma too short")
        return 0.0, 0, notes

    try:
        _tempo, beat_frames_in = librosa.beat.beat_track(
            y=in_y, sr=in_sr, hop_length=CHROMA_HOP_LENGTH,
        )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"chroma_rf: beat_track failed: {exc}")
        beat_frames_in = np.array([], dtype=int)

    beat_frames_in = np.atleast_1d(beat_frames_in).astype(int)

    if beat_frames_in.size < 2:
        # Fallback: compare a single global chroma vector. Honest answer
        # for short / unbeat-trackable clips; the n_beats count of 1
        # signals to the aggregator that the cosine is global, not
        # per-beat.
        v_in = chroma_in.mean(axis=1)
        v_rs = chroma_rs.mean(axis=1)
        notes.append("chroma_rf: <2 beats detected; global chroma cosine")
        return _cosine(v_in, v_rs), 1, notes

    # Project the input's beat times into the resynth's chroma frame space.
    # Both audios are at CHROMA_SR after the resample above so the frame
    # math is the same on each side, but the chroma shapes can differ when
    # the resynth's duration is slightly off (FluidSynth pads / clips).
    beats_sec = librosa.frames_to_time(
        beat_frames_in, sr=in_sr, hop_length=CHROMA_HOP_LENGTH,
    )
    beat_frames_rs = librosa.time_to_frames(
        beats_sec, sr=rs_sr, hop_length=CHROMA_HOP_LENGTH,
    )
    # Clip to valid frame indices on each side (a beat past the end of
    # one chroma simply doesn't contribute to that side's sync vector).
    beat_frames_in = np.atleast_1d(beat_frames_in).astype(int)
    beat_frames_rs = np.atleast_1d(beat_frames_rs).astype(int)
    beat_frames_in = beat_frames_in[
        (beat_frames_in >= 0) & (beat_frames_in < chroma_in.shape[1])
    ]
    beat_frames_rs = beat_frames_rs[
        (beat_frames_rs >= 0) & (beat_frames_rs < chroma_rs.shape[1])
    ]

    if beat_frames_in.size < 2 or beat_frames_rs.size < 2:
        v_in = chroma_in.mean(axis=1)
        v_rs = chroma_rs.mean(axis=1)
        notes.append("chroma_rf: insufficient in-range beats; global cosine")
        return _cosine(v_in, v_rs), 1, notes

    sync_in = librosa.util.sync(
        chroma_in, beat_frames_in.tolist(), aggregate=np.mean,
    )
    sync_rs = librosa.util.sync(
        chroma_rs, beat_frames_rs.tolist(), aggregate=np.mean,
    )
    n_spans = int(min(sync_in.shape[1], sync_rs.shape[1]))
    if n_spans <= 0:
        notes.append("chroma_rf: no overlapping beat spans")
        return 0.0, 0, notes

    cos_per_beat = [
        _cosine(sync_in[:, i], sync_rs[:, i]) for i in range(n_spans)
    ]
    return float(np.mean(cos_per_beat)), n_spans, notes


def _chroma_cqt(y: Any, sr: int) -> Any:
    """HPSS-harmonic + ``chroma_cqt`` mirroring chord_recognition's setup.

    Same hop / bins-per-octave so the per-frame semantics match what the
    chord recognizer would have computed on the same waveform.
    """
    import librosa  # noqa: PLC0415

    try:
        y_h = librosa.effects.harmonic(y, margin=3.0)
    except Exception:  # noqa: BLE001
        y_h = y
    return librosa.feature.chroma_cqt(
        y=y_h,
        sr=sr,
        hop_length=CHROMA_HOP_LENGTH,
        bins_per_octave=36,
    )


def _cosine(a: Any, b: Any) -> float:
    """Cosine similarity, clamped to ``[0, 1]``.

    Chroma vectors are non-negative so the geometric range of
    ``dot / (||a|| ||b||)`` is naturally ``[0, 1]``; clamp for floating
    point safety.
    """
    import numpy as np  # noqa: PLC0415

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    cos = float(np.dot(a, b) / (na * nb))
    return max(0.0, min(1.0, cos))


# ---------------------------------------------------------------------------
# Top-level entrypoint
# ---------------------------------------------------------------------------

def compute_tier_rf(
    input_audio_path: Path,
    score: PianoScore,
    engraved_midi_bytes: bytes,
    *,
    key_label: str = "C:major",
    fluidsynth_bin: str | None = None,
    soundfont_path: Path | None = None,
) -> TierRfResult:
    """Run all three RF metrics for one (input audio, PianoScore, MIDI) tuple.

    Loads the input audio at :data:`CHROMA_SR`, FluidSynth-renders the
    engraved MIDI, then computes :func:`chord_rf_score`,
    :func:`playability_rf_score`, and :func:`chroma_rf_score` against
    a single in-memory copy of each audio.

    ``key_label`` is forwarded to the chord recognizer's HMM prior on
    both sides — pass ``HarmonicAnalysis.key`` from the transcribe stage
    so the recognizer's diatonic transition matrix is the same on input
    and resynth.
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

    chord_score, n_in_seg, n_rs_seg, chord_notes = chord_rf_score(
        (in_y, in_sr), (rs_y, rs_sr), key_label=key_label,
    )
    play_score, n_playable, n_total = playability_rf_score(score)
    chroma_score, n_beats, chroma_notes = chroma_rf_score(
        (in_y, in_sr), (rs_y, rs_sr),
    )

    return TierRfResult(
        chord_rf=chord_score,
        playability_rf=play_score,
        chroma_rf=chroma_score,
        n_chord_segments_input=n_in_seg,
        n_chord_segments_resynth=n_rs_seg,
        n_playable_chords=n_playable,
        n_total_chords=n_total,
        n_beats=n_beats,
        notes=[*chord_notes, *chroma_notes],
    )
