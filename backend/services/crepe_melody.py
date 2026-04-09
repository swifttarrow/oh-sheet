"""CREPE-based monophonic melody extractor for the Demucs vocals stem.

Basic Pitch is a polyphonic CNN and runs fine on vocals, but it's not
optimized for monophonic singing — it over-emits short ghost notes on
legato phrases, under-tracks vibrato, and its onset detector is tuned
for percussive attacks rather than consonant-driven vocal onsets. The
baseline eval (``scripts/eval_transcription.py``) shows a per-role
``melody`` F1 of ~0.10, vs ~0.34 for ``chords`` — the vocal pass is
by far the weakest link in the stems pipeline.

CREPE (Kim et al. 2018, *CREPE: A Convolutional Representation for
Pitch Estimation*) is a mono F0 estimator trained on singing and
speech. It's SOTA for this exact task and — via `torchcrepe`_ — runs
on the ``torch`` install we already have for Demucs, with the model
weights shipped inside the wheel so there's nothing to fetch at
runtime.

Pipeline
--------
1. Load the vocals stem as mono 16 kHz (CREPE's training SR)
2. ``torchcrepe.predict`` → per-frame ``(pitch_hz, periodicity)`` at
   100 Hz (10 ms hop)
3. Median-filter the pitch track to kill vibrato wobble + spikes
4. ``torchcrepe.threshold.At`` to zero out low-confidence frames
5. Segment the result into discrete :data:`NoteEvent` tuples by
   walking contiguous runs of same-integer-MIDI voiced frames,
   merging same-pitch runs across short unvoiced bridges (legato),
   and dropping sub-``min_duration`` passing ornaments

The output is the same ``NoteEvent`` tuple format Basic Pitch emits,
so it drops straight into the existing cleanup + role-assignment
wiring in :mod:`backend.services.transcribe`.

Feature-flagged via ``settings.crepe_vocal_melody_enabled`` —
**disabled by default** after the first A/B came in net-neutral on
the 25-file clean_midi baseline (see ``backend.config.Settings`` for
the full numbers). The module is kept in-tree so the next tuning
pass (higher voicing threshold, longer smoothing, or the ``tiny``
model variant) can flip it back on via env without reimplementing
the pipeline from scratch. Graceful fallback to Basic Pitch on the
vocals stem applies when torchcrepe is missing or any runtime step
raises, so flipping the flag on is always safe.

.. _torchcrepe: https://github.com/maxrmorrison/torchcrepe
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.services._torch_utils import pick_device
from backend.services.transcription_cleanup import NoteEvent

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults — mirrored in backend/config.py so config and tests agree.
# ---------------------------------------------------------------------------

# CREPE model size. ``full`` is 22 MB and ~3x slower than ``tiny`` (2 MB),
# but more accurate on held notes and edge-of-range pitches. Vocals are
# the weakest link in our eval, so we pay the accuracy tax by default.
DEFAULT_MODEL = "full"

# CREPE is trained at 16 kHz. Anything else gets resampled with
# torchaudio before inference.
DEFAULT_SAMPLE_RATE = 16000

# 10 ms hop at 16 kHz → 100 Hz frame rate. torchcrepe's default is
# also 160 samples; we mirror it so our segmentation constants (in
# seconds) map cleanly to integer frame counts.
DEFAULT_HOP_LENGTH_SAMPLES = 160

# Pitch search range. C2 (~65 Hz) is well below the lowest bass-baritone
# fundamental; C#6 (~1100 Hz) comfortably covers soprano belts. Widening
# the range further mostly invites octave errors on breathy onsets.
DEFAULT_FMIN_HZ = 65.0
DEFAULT_FMAX_HZ = 1100.0

# Voicing gate. CREPE periodicity is noisy on consonants + breaths —
# 0.5 rejects most of those while keeping confident sung frames. Tuned
# against the eval harness; see ``scripts/eval_transcription.py``.
DEFAULT_VOICING_THRESHOLD = 0.5

# 5-frame (50 ms) median filter is long enough to smooth normal vibrato
# (4-6 Hz, ±50 cents) into a stable semitone-quantized track without
# destroying fast melismatic runs.
DEFAULT_MEDIAN_FILTER_FRAMES = 5

# Drop passing ornaments / CREPE artifacts shorter than this. 60 ms is
# below a fast 16th-note at 240 BPM, so we don't clip real content.
DEFAULT_MIN_NOTE_DURATION_SEC = 0.06

# Bridge short unvoiced gaps between two same-pitch runs. Captures
# legato singing where the periodicity dips momentarily on a consonant
# transition mid-word but the pitch doesn't actually move.
DEFAULT_MERGE_GAP_SEC = 0.06

# Per-note amplitude → velocity proxy. CREPE periodicity is a "how sure
# are we this is voiced" signal, not a loudness signal, but it correlates
# well enough with confident attacks that we can feed it into the same
# ``velocity = round(127 * amp)`` formula Basic Pitch uses. Clamped into
# a middle band so CREPE-derived notes don't render as either barely-
# audible ghosts or max-velocity hammers in the arranged output.
DEFAULT_AMP_MIN = 0.25
DEFAULT_AMP_MAX = 0.85


@dataclass
class CrepeMelodyStats:
    """Telemetry for one ``extract_vocal_melody_crepe`` call.

    Attached to the ``QualitySignal.warnings`` list via
    :meth:`as_warnings` — mirrors the pattern used by every other
    extractor module so the transcribe assembly site only has to do
    ``warnings.extend(stats.as_warnings())``.
    """
    skipped: bool = False
    model: str = ""
    device: str = ""
    n_frames: int = 0
    n_voiced_frames: int = 0
    n_notes: int = 0
    wall_sec: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def as_warnings(self) -> list[str]:
        out: list[str] = []
        if not self.skipped and self.n_notes:
            voiced_pct = (
                100.0 * self.n_voiced_frames / self.n_frames
                if self.n_frames else 0.0
            )
            out.append(
                f"crepe-melody: {self.n_notes} notes from "
                f"{self.n_voiced_frames}/{self.n_frames} voiced frames "
                f"({voiced_pct:.0f}%, {self.model}, {self.device}, "
                f"{self.wall_sec:.1f}s)"
            )
        out.extend(self.warnings)
        return out


def _load_mono_16k(audio_path: Path) -> tuple[Any, int]:
    """Load an audio file as ``(1, n_samples)`` mono tensor at 16 kHz.

    CREPE is trained at 16 kHz and expects mono; feeding it anything
    else either silently resamples inside torchcrepe (wasting time)
    or produces distorted periodicity estimates. We handle both up
    front with torchaudio so the CREPE call sees exactly what its
    weights were trained on.
    """
    import torchaudio  # noqa: PLC0415

    wav, sr = torchaudio.load(str(audio_path))
    # Downmix to mono if needed. Demucs vocals stems are stereo.
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != DEFAULT_SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, DEFAULT_SAMPLE_RATE)
    return wav, DEFAULT_SAMPLE_RATE


def _f0_to_notes(
    pitch_hz: list[float],
    periodicity: list[float],
    frame_rate: float,
    *,
    min_note_duration_sec: float,
    merge_gap_sec: float,
    amp_min: float,
    amp_max: float,
) -> list[NoteEvent]:
    """Segment a frame-level F0 stream into discrete ``NoteEvent`` tuples.

    Algorithm
    ---------
    1. Convert each voiced frame's Hz to rounded integer MIDI. Unvoiced
       frames (Hz == 0 after the voicing threshold) become a sentinel
       ``-1`` so they naturally break runs.
    2. Walk the frames; a note is a maximal run of same-integer-MIDI
       voiced frames. Per-note amplitude is the mean periodicity in
       that run, clipped to ``[amp_min, amp_max]``.
    3. Merge adjacent notes that share the same pitch and are
       separated by at most ``merge_gap_sec`` (legato bridges across
       momentary consonant-driven voicing dips).
    4. Drop notes shorter than ``min_note_duration_sec`` (passing
       ornaments, CREPE attack artifacts).

    Pure Python — no numpy — so this is easy to unit-test with
    hand-built synthetic frame streams. Performance is fine for our
    clip sizes (100 Hz × 30 s = 3000 frames per file).
    """
    n_frames = len(pitch_hz)
    if n_frames == 0 or frame_rate <= 0:
        return []
    frame_sec = 1.0 / frame_rate

    # Step 1 — Hz → rounded int MIDI, sentinel -1 for unvoiced. The
    # caller is expected to pre-scrub the input so unvoiced frames
    # arrive as 0, but we guard against NaN/inf too so this function
    # stays safe to call with a raw torchcrepe output (which uses
    # ``torchcrepe.UNVOICED == nan`` as its sentinel).
    midi_per_frame: list[int] = []
    for hz in pitch_hz:
        if hz <= 0 or math.isnan(hz) or math.isinf(hz):
            midi_per_frame.append(-1)
        else:
            midi_per_frame.append(int(round(69 + 12 * math.log2(hz / 440.0))))

    # Step 2 — walk contiguous same-pitch voiced runs.
    notes: list[NoteEvent] = []
    i = 0
    while i < n_frames:
        p = midi_per_frame[i]
        if p < 0:
            i += 1
            continue
        start_frame = i
        amps: list[float] = [periodicity[i]]
        i += 1
        while i < n_frames and midi_per_frame[i] == p:
            amps.append(periodicity[i])
            i += 1
        end_frame = i
        raw_amp = sum(amps) / len(amps) if amps else 0.0
        clamped = max(amp_min, min(amp_max, raw_amp))
        notes.append((
            start_frame * frame_sec,
            end_frame * frame_sec,
            p,
            float(clamped),
            [],  # pitch bends — CREPE gives us micro-pitch data but
                 # we collapse to int MIDI, so there's nothing to emit
        ))

    # Step 3 — merge short same-pitch gaps.
    merged: list[NoteEvent] = []
    for note in notes:
        if merged:
            prev = merged[-1]
            if prev[2] == note[2] and (note[0] - prev[1]) <= merge_gap_sec:
                prev_dur = prev[1] - prev[0]
                new_dur = note[1] - note[0]
                # Duration-weighted amplitude so a long stable note
                # isn't pulled around by a tiny tail merge.
                merged_amp = (
                    (prev[3] * prev_dur + note[3] * new_dur)
                    / (prev_dur + new_dur)
                )
                merged[-1] = (prev[0], note[1], prev[2], float(merged_amp), [])
                continue
        merged.append(note)

    # Step 4 — drop runs shorter than the minimum musical duration.
    return [n for n in merged if (n[1] - n[0]) >= min_note_duration_sec]


def extract_vocal_melody_crepe(
    vocals_path: Path,
    *,
    model: str = DEFAULT_MODEL,
    hop_length_samples: int = DEFAULT_HOP_LENGTH_SAMPLES,
    fmin_hz: float = DEFAULT_FMIN_HZ,
    fmax_hz: float = DEFAULT_FMAX_HZ,
    voicing_threshold: float = DEFAULT_VOICING_THRESHOLD,
    median_filter_frames: int = DEFAULT_MEDIAN_FILTER_FRAMES,
    min_note_duration_sec: float = DEFAULT_MIN_NOTE_DURATION_SEC,
    merge_gap_sec: float = DEFAULT_MERGE_GAP_SEC,
    amp_min: float = DEFAULT_AMP_MIN,
    amp_max: float = DEFAULT_AMP_MAX,
    device: str | None = None,
) -> tuple[list[NoteEvent], CrepeMelodyStats]:
    """Run CREPE on ``vocals_path`` and return cleaned-up ``NoteEvent`` list.

    Returns ``([], stats_with_skipped=True)`` on any failure — missing
    ``torchcrepe``, missing file, load error, predict crash — so the
    caller in :func:`backend.services.transcribe._run_with_stems` can
    fall back to the Basic Pitch vocals pass without losing notes.

    The returned events are in the same tuple format Basic Pitch emits
    (``(start, end, midi_pitch, amplitude, bends)``), so the caller
    assigns them directly to ``events_by_role[InstrumentRole.MELODY]``
    without any further conversion.
    """
    stats = CrepeMelodyStats(model=model)

    # Late-imported so the module is importable on machines without
    # torchcrepe (mirrors the pattern in stem_separation.py). A missing
    # dep is a "skip" — the caller falls back to Basic Pitch.
    try:
        import torchcrepe  # noqa: PLC0415
    except ImportError as exc:
        log.debug("torchcrepe unavailable: %s", exc)
        stats.skipped = True
        stats.warnings.append(f"crepe-melody: missing dep ({exc.name})")
        return [], stats

    if not vocals_path.is_file():
        stats.skipped = True
        stats.warnings.append(f"crepe-melody: vocals file missing: {vocals_path}")
        return [], stats

    t0 = time.perf_counter()
    try:
        wav, sr = _load_mono_16k(vocals_path)
    except Exception as exc:  # noqa: BLE001 — bad bytes / decode failure
        log.warning("crepe-melody: load failed for %s: %s", vocals_path, exc)
        stats.skipped = True
        stats.warnings.append(f"crepe-melody: load failed: {exc}")
        return [], stats

    device_str = pick_device(device)
    stats.device = device_str

    try:
        pitch, periodicity = torchcrepe.predict(
            wav,
            sr,
            hop_length_samples,
            fmin_hz,
            fmax_hz,
            model,
            batch_size=2048,
            device=device_str,
            return_periodicity=True,
        )
    except Exception as exc:  # noqa: BLE001 — inference errors shouldn't sink the job
        log.warning("crepe-melody: torchcrepe.predict failed: %s", exc)
        stats.skipped = True
        stats.warnings.append(f"crepe-melody: predict failed: {exc}")
        return [], stats

    # Smooth the pitch track, then gate on periodicity. Filter order
    # matters: smoothing first means vibrato wobble doesn't push
    # individual frames below the threshold; gating first would mean
    # the smoother sees NaNs (torchcrepe's unvoiced sentinel) where
    # it should see real pitch data.
    if median_filter_frames >= 3:
        pitch = torchcrepe.filter.median(pitch, median_filter_frames)
    pitch = torchcrepe.threshold.At(voicing_threshold)(pitch, periodicity)

    # torchcrepe uses ``torchcrepe.UNVOICED == nan`` as the sentinel
    # for rejected frames. Our segmenter wants plain zeros for the
    # "unvoiced" lane, so we scrub NaN/inf out on the way down to
    # Python. numpy.nan_to_num is the cheapest path: one vectorised
    # pass, no Python-level loop, safe on empty arrays.
    import numpy as np  # noqa: PLC0415

    pitch_np = pitch.squeeze(0).detach().cpu().numpy()
    periodicity_np = periodicity.squeeze(0).detach().cpu().numpy()
    stats.n_frames = int(pitch_np.shape[0])
    stats.n_voiced_frames = int((pitch_np > 0).sum())
    pitch_np = np.nan_to_num(pitch_np, nan=0.0, posinf=0.0, neginf=0.0)
    periodicity_np = np.nan_to_num(periodicity_np, nan=0.0, posinf=0.0, neginf=0.0)

    frame_rate = float(sr) / float(hop_length_samples)
    notes = _f0_to_notes(
        pitch_hz=pitch_np.tolist(),
        periodicity=periodicity_np.tolist(),
        frame_rate=frame_rate,
        min_note_duration_sec=min_note_duration_sec,
        merge_gap_sec=merge_gap_sec,
        amp_min=amp_min,
        amp_max=amp_max,
    )

    stats.n_notes = len(notes)
    stats.wall_sec = time.perf_counter() - t0
    return notes, stats
