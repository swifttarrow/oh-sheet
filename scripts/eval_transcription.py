"""Transcription eval harness: score the pipeline against clean_midi.

The :mod:`clean_midi` dataset (Lakh MIDI Clean) is a MIDI-only corpus of
~17k songs organized by artist. A 25-file reproducible subset of that
corpus lives at ``eval/fixtures/clean_midi/`` and is tracked in-repo
— running this script with no arguments scores the pipeline against
that subset and produces an output that should match
``eval-baseline.json`` byte-for-byte (modulo transcription-logic
changes). For broader sweeps over the full 17k-song corpus, fetch
the canonical tarball from
``http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz`` (CC BY
4.0) and run with ``--dataset clean_midi --sample N``. The top-level
``clean_midi/`` is gitignored so the full corpus never leaks into
the repo.

To turn it into a ground-truth set for our audio-in / notes-out
pipeline, we:

  1. Sample a subset of .mid files
  2. Truncate each to a sanity-sized window (default 30 s) so a full
     eval run finishes in a few minutes, not a few hours
  3. Synthesize the truncated MIDI to WAV via ``fluidsynth`` + the
     ``TimGM6mb.sf2`` soundfont bundled inside the ``pretty_midi``
     wheel (so there's no soundfont to install)
  4. Run :func:`backend.services.transcribe._run_basic_pitch_sync` on
     the synthesized audio — same entry point ``bench_preprocess.py``
     uses, so everything the production pipeline does (preprocess,
     Demucs stems, cleanup, Viterbi split, chord recog) is exercised
  5. Score the predicted notes against the original MIDI with
     :func:`mir_eval.transcription.precision_recall_f1_overlap`, both
     no-offset (the permissive headline) and with 20% offset tolerance
     (the strict headline)

The synthesized WAVs are cached keyed on ``(midi_path, max_duration,
soundfont)`` so re-runs skip straight to inference. This makes the
harness cheap to iterate on while tuning thresholds or swapping
transcription backends.

There is no "is-this-piano" filter here: the pipeline is a general
audio-to-notes transcriber, so we score it on whatever clean_midi
throws at us — full bands, a-cappella arrangements, orchestral
transcriptions, the works. That's the right input distribution for a
service that accepts arbitrary audio. Per-track (melody/bass/chords)
breakdowns are reported separately at the end so the role-split
extractors can be evaluated in isolation.

Usage::

    # Quick smoke test: 3 random files, 20 s each
    python scripts/eval_transcription.py --sample 3 --max-duration 20

    # Fuller run with JSON output for CI diffs
    python scripts/eval_transcription.py --sample 50 --seed 7 \\
        --out eval-results.json

    # Specific files (no sampling) — useful for regression repro
    python scripts/eval_transcription.py \\
        "clean_midi/The Corrs/Dreams.mid" \\
        "clean_midi/Elton John/Tiny Dancer.mid"

Any file that fails to parse, synthesize, or transcribe is logged and
skipped so one bad fixture doesn't sink the whole run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATASET = REPO_ROOT / "eval" / "fixtures" / "clean_midi"
DEFAULT_CACHE = REPO_ROOT / ".cache" / "eval_transcription"
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_MAX_DURATION = 30.0
DEFAULT_ONSET_TOL = 0.05   # ±50 ms — mir_eval default
DEFAULT_PITCH_TOL = 50.0   # ±50 cents (quarter tone) — mir_eval default
DEFAULT_OFFSET_RATIO = 0.2


# --------------------------------------------------------------------------- #
# Dependency probes
# --------------------------------------------------------------------------- #

def _find_fluidsynth() -> str:
    """Locate the ``fluidsynth`` CLI, erroring out with a friendly message.

    We shell out instead of binding pyFluidSynth because the binary is a
    single brew/apt install away and avoids the native-extension wheel
    churn that the Python binding drags in on macOS arm64.
    """
    path = shutil.which("fluidsynth")
    if not path:
        raise RuntimeError(
            "fluidsynth binary not found on PATH. Install with "
            "`brew install fluid-synth` (macOS) or `apt install fluidsynth` "
            "(Debian/Ubuntu)."
        )
    return path


def _default_soundfont() -> Path:
    """Return the TimGM6mb.sf2 bundled inside the ``pretty_midi`` wheel.

    Every environment that can run the transcription pipeline already
    has pretty_midi installed (it's a Basic Pitch dep), so piggy-backing
    on the wheel means the harness works out of the box with zero
    soundfont setup. TimGM6mb is CC-licensed and small (~6 MB) so it's
    also safe to ship in CI.
    """
    import pretty_midi  # noqa: PLC0415
    sf2 = Path(pretty_midi.__file__).parent / "TimGM6mb.sf2"
    if not sf2.is_file():
        raise RuntimeError(
            f"Expected bundled soundfont at {sf2}, but it's missing. "
            "Upgrade pretty_midi or pass --soundfont."
        )
    return sf2


# --------------------------------------------------------------------------- #
# Dataset discovery + sampling
# --------------------------------------------------------------------------- #

def _discover_midi_paths(dataset_root: Path) -> list[Path]:
    """Walk ``dataset_root`` and return every .mid below it (sorted).

    Sorting makes sampling reproducible — ``random.Random(seed).sample``
    over a sorted list gives the same subset across machines. The
    expected dataset is clean_midi, which has ~17k files and takes well
    under a second to enumerate on SSD.
    """
    if not dataset_root.is_dir():
        raise RuntimeError(f"Dataset root not found: {dataset_root}")
    return sorted(dataset_root.rglob("*.mid"))


def _select_paths(
    dataset_root: Path,
    explicit: list[Path],
    sample: int,
    seed: int,
) -> list[Path]:
    """Pick the files to evaluate.

    Precedence: explicit paths beat sampling. If neither is specified,
    we default to a small random sample so ``--help`` → run gives a
    fast, meaningful result without a pile of flags.
    """
    if explicit:
        # Resolve against the repo root so users can pass relative
        # paths like "clean_midi/Artist/Song.mid" from anywhere.
        resolved: list[Path] = []
        for p in explicit:
            candidate = p if p.is_absolute() else (REPO_ROOT / p)
            if not candidate.is_file():
                print(f"! skipping missing file: {p}")
                continue
            resolved.append(candidate)
        return resolved

    all_paths = _discover_midi_paths(dataset_root)
    if not all_paths:
        raise RuntimeError(f"No .mid files under {dataset_root}")
    rng = random.Random(seed)
    return rng.sample(all_paths, min(sample, len(all_paths)))


# --------------------------------------------------------------------------- #
# Synthesis + ground truth
# --------------------------------------------------------------------------- #

def _cache_key(midi_path: Path, max_duration: float, soundfont: Path) -> str:
    """Stable hash for the (midi, duration, soundfont) tuple.

    Changing any of the three invalidates the cached WAV. Using the
    resolved path means two different symlinks to the same file share
    the same cache entry — a minor win that also makes the key
    insensitive to where the user cd's from when invoking the script.
    """
    payload = f"{midi_path.resolve()}|{max_duration}|{soundfont.resolve()}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _truncate_midi(midi_path: Path, max_duration: float | None) -> Any:
    """Load ``midi_path`` and drop every event past ``max_duration``.

    Returns a ``pretty_midi.PrettyMIDI`` object with the truncated note
    lists so the caller can both (a) write a shortened .mid for
    fluidsynth and (b) read notes back for ground-truth scoring. We
    also trim the tail of any note that straddles the cutoff so the
    synthesized WAV and the scoring arrays agree on where audio ends.

    Subtlety: notes aren't the only per-instrument events pretty_midi
    tracks. ``control_changes``, ``pitch_bends``, and ``pedal_events``
    also carry timestamps, and fluidsynth will happily render past
    the last note if any of those linger in the 30-second-plus range
    (``pm.get_end_time()`` returns the max of all of them). Leaving
    them unpruned produced 200-second WAVs from 30-second sampling
    calls during the first baseline run. We truncate every timestamped
    event class pretty_midi exposes so fluidsynth stops exactly where
    we intended.

    ``max_duration=None`` is a no-op — returns the full MIDI unchanged.
    """
    import pretty_midi  # noqa: PLC0415
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    if max_duration is None or max_duration <= 0:
        return pm
    for inst in pm.instruments:
        kept_notes = []
        for n in inst.notes:
            if n.start >= max_duration:
                continue
            if n.end > max_duration:
                n.end = max_duration
            kept_notes.append(n)
        inst.notes = kept_notes
        inst.control_changes = [
            cc for cc in inst.control_changes if cc.time < max_duration
        ]
        inst.pitch_bends = [
            pb for pb in inst.pitch_bends if pb.time < max_duration
        ]
    # Track-level meta events also count toward ``pm.get_end_time()``,
    # and fluidsynth honors the end-of-track marker when deciding how
    # much silence to render after the last note. Lyric/karaoke files
    # (e.g. ``Celine Dion/Power of Love``) stash 300+ ``text_events``
    # at the real playback times, which kept synthesis going for the
    # full song despite every note ending at 30s. Scrubbing these
    # caps the MIDI at ``max_duration`` regardless of source format.
    pm.lyrics = [lyr for lyr in pm.lyrics if lyr.time < max_duration]
    pm.text_events = [te for te in pm.text_events if te.time < max_duration]
    pm.time_signature_changes = [
        ts for ts in pm.time_signature_changes if ts.time < max_duration
    ]
    pm.key_signature_changes = [
        ks for ks in pm.key_signature_changes if ks.time < max_duration
    ]
    return pm


def _synthesize(
    pm: Any,
    soundfont: Path,
    out_wav: Path,
    sample_rate: int,
    fluidsynth_bin: str,
    max_duration: float | None,
) -> None:
    """Synthesize a PrettyMIDI to WAV via ``fluidsynth``.

    We write the PrettyMIDI to a temp .mid and hand it to the binary
    instead of piping, because fluidsynth's CLI reads the file by path.
    ``-ni`` runs non-interactively (no shell), ``-g 1.0`` sets master
    gain to unity so the output isn't clipped or whispered, and
    ``-F out.wav`` is the standard fast-render flag.

    If ``max_duration`` is set, we hard-clip the rendered WAV to that
    duration as a belt-and-suspenders safety net. ``_truncate_midi``
    handles most files cleanly, but pretty_midi doesn't expose tempo
    changes as a mutable list (they live in ``_tick_scales`` at the
    MIDI tick level), so karaoke / multi-tempo arrangements with
    tempo events past the cutoff still produce MIDIs whose
    ``pm.get_end_time()`` walks to the original track length.
    fluidsynth then renders silence all the way to that marker. The
    post-hoc clip catches these files without reaching into
    pretty_midi internals.

    On failure, stderr from fluidsynth is surfaced in the raised
    exception — it's usually self-explanatory (missing soundfont, bad
    MIDI, etc.).

    The rendered WAV is staged next to ``out_wav`` as
    ``<stem>.partial.wav`` and only promoted to ``out_wav`` via
    :func:`os.replace` after both fluidsynth and the post-hoc clip
    succeed. If the process is killed mid-render (SIGKILL, OOM, ^C)
    the partial file stays on disk but ``out_wav`` does not, so the
    next run re-synthesizes rather than silently reusing a truncated
    cache entry. The staging name keeps a ``.wav`` suffix on purpose:
    both fluidsynth's ``-F`` writer and libsndfile (via
    ``soundfile``) pick the file format from the extension, and
    anything other than a recognized audio suffix causes fluidsynth
    to emit headerless raw PCM and ``sf.read`` to raise "No format
    specified and unable to get format from file extension".
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    staging_wav = out_wav.with_name(f"{out_wav.stem}.partial{out_wav.suffix}")
    staging_wav.unlink(missing_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        pm.write(str(tmp_path))
        proc = subprocess.run(
            [
                fluidsynth_bin,
                "-ni",
                "-g", "1.0",
                "-r", str(sample_rate),
                "-F", str(staging_wav),
                str(soundfont),
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"fluidsynth failed (rc={proc.returncode}): "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )

        # Hard-clip the rendered WAV to max_duration if needed.
        # soundfile round-trip is cheap (~10 ms for a 30 s clip) and
        # guarantees the cached audio matches the ground-truth window
        # exactly. Stays on the staging path so a failure here
        # doesn't leave a bad file at ``out_wav``.
        if max_duration is not None and max_duration > 0:
            import soundfile as sf  # noqa: PLC0415

            data, sr = sf.read(str(staging_wav))
            max_samples = int(max_duration * sr)
            if len(data) > max_samples:
                sf.write(str(staging_wav), data[:max_samples], sr)

        os.replace(staging_wav, out_wav)
    except BaseException:
        staging_wav.unlink(missing_ok=True)
        raise
    finally:
        tmp_path.unlink(missing_ok=True)


def _ensure_synthesized(
    midi_path: Path,
    max_duration: float | None,
    soundfont: Path,
    cache_dir: Path,
    fluidsynth_bin: str,
    sample_rate: int,
) -> tuple[Path, Any]:
    """Synthesize ``midi_path`` to WAV (or reuse cache) and return the PM.

    Returns ``(wav_path, truncated_pm)``. The truncated PrettyMIDI is
    always built fresh — it's the ground-truth source of notes, and we
    want the in-memory representation to match the WAV exactly even on
    cache hits.
    """
    import pretty_midi  # noqa: PLC0415
    pm = _truncate_midi(midi_path, max_duration)
    key = _cache_key(midi_path, max_duration or 0.0, soundfont)
    out_wav = cache_dir / f"{key}.wav"
    if not out_wav.is_file():
        _synthesize(
            pm, soundfont, out_wav, sample_rate, fluidsynth_bin, max_duration,
        )
    assert isinstance(pm, pretty_midi.PrettyMIDI)
    return out_wav, pm


# --------------------------------------------------------------------------- #
# Note extraction + metrics
# --------------------------------------------------------------------------- #

def _ground_truth_notes(pm: Any) -> tuple[Any, Any]:
    """Flatten a ``pretty_midi.PrettyMIDI`` to (intervals, pitches_hz).

    Drum tracks are excluded — Basic Pitch is a pitched-note
    transcriber, so scoring it on drum hits would only add noise. We
    return pitches in Hz because that's the unit ``mir_eval``'s
    transcription metrics consume (cents-based tolerance under the
    hood).
    """
    import numpy as np  # noqa: PLC0415
    import pretty_midi  # noqa: PLC0415

    intervals: list[list[float]] = []
    pitches: list[float] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if n.end <= n.start:
                continue
            intervals.append([float(n.start), float(n.end)])
            pitches.append(float(pretty_midi.note_number_to_hz(n.pitch)))
    return np.array(intervals, dtype=float), np.array(pitches, dtype=float)


def _predicted_notes(result: Any, *, role_filter: str | None = None) -> tuple[Any, Any]:
    """Flatten a ``TranscriptionResult`` to (intervals, pitches_hz).

    If ``role_filter`` is set (e.g. ``"melody"``), only notes from
    tracks whose instrument role matches are returned. Used for the
    per-role breakdown pass at the end of a run.
    """
    import numpy as np  # noqa: PLC0415
    import pretty_midi  # noqa: PLC0415

    intervals: list[list[float]] = []
    pitches: list[float] = []
    for track in result.midi_tracks:
        if role_filter and track.instrument.value != role_filter:
            continue
        for n in track.notes:
            if n.offset_sec <= n.onset_sec:
                continue
            intervals.append([float(n.onset_sec), float(n.offset_sec)])
            pitches.append(float(pretty_midi.note_number_to_hz(n.pitch)))
    return np.array(intervals, dtype=float), np.array(pitches, dtype=float)


@dataclass
class ScoreRow:
    """Per-file scoring record. Serialized straight to JSON in ``--out``."""
    path: str
    ref_notes: int
    est_notes: int
    # No-offset (permissive): correct iff onset ±50 ms and pitch ±50 cents.
    p_no_offset: float
    r_no_offset: float
    f1_no_offset: float
    # With 20% offset tolerance (strict): the above plus offset within
    # 20% of the reference note's duration.
    p_with_offset: float
    r_with_offset: float
    f1_with_offset: float
    # Pipeline self-reported confidence — useful to correlate against F1
    # when debugging calibration issues.
    confidence: float
    wall_sec: float
    error: str | None = None


def _score(
    ref_intervals: Any,
    ref_pitches: Any,
    est_intervals: Any,
    est_pitches: Any,
) -> dict[str, float]:
    """Compute no-offset and with-offset P/R/F1 for one file.

    mir_eval raises on empty inputs, so we short-circuit to a zero row
    when either side has no notes. That's the semantically correct
    answer (precision is undefined, recall is 0 if ref is non-empty,
    and F1 degenerates to 0) and keeps the aggregate means honest.
    """
    import mir_eval  # noqa: PLC0415

    if len(ref_intervals) == 0 or len(est_intervals) == 0:
        return {
            "p_no_offset": 0.0,
            "r_no_offset": 0.0,
            "f1_no_offset": 0.0,
            "p_with_offset": 0.0,
            "r_with_offset": 0.0,
            "f1_with_offset": 0.0,
        }

    p_no, r_no, f_no, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches, est_intervals, est_pitches,
        onset_tolerance=DEFAULT_ONSET_TOL,
        pitch_tolerance=DEFAULT_PITCH_TOL,
        offset_ratio=None,
    )
    p_w, r_w, f_w, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches, est_intervals, est_pitches,
        onset_tolerance=DEFAULT_ONSET_TOL,
        pitch_tolerance=DEFAULT_PITCH_TOL,
        offset_ratio=DEFAULT_OFFSET_RATIO,
    )
    return {
        "p_no_offset": float(p_no),
        "r_no_offset": float(r_no),
        "f1_no_offset": float(f_no),
        "p_with_offset": float(p_w),
        "r_with_offset": float(r_w),
        "f1_with_offset": float(f_w),
    }


# --------------------------------------------------------------------------- #
# Pipeline invocation
# --------------------------------------------------------------------------- #

def _run_transcribe(audio_path: Path) -> tuple[Any, float]:
    """Run the same entry point ``bench_preprocess.py`` uses.

    Imported lazily because pulling the transcribe module in at script
    import time eagerly loads Basic Pitch + Demucs weights, which adds
    several seconds to a ``--help`` invocation.
    """
    import backend.services.transcribe as T  # noqa: PLC0415
    t0 = time.perf_counter()
    result, _ = T._run_basic_pitch_sync(audio_path)
    return result, time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def _fmt_path(midi_path: Path) -> str:
    """Pretty-print a MIDI path as ``Artist/Song.mid`` when possible.

    Falls back to the full resolved path for anything outside the
    clean_midi tree so the table stays readable regardless of source.
    """
    try:
        rel = midi_path.relative_to(DEFAULT_DATASET)
        return str(rel)
    except ValueError:
        return str(midi_path)


def _print_header(path_width: int) -> None:
    cols = (
        f"  {'#':>3}  "
        f"{'file':<{path_width}}  "
        f"{'ref':>5}  {'est':>5}  "
        f"{'P':>5}  {'R':>5}  {'F1':>5}  "
        f"{'F1off':>6}  "
        f"{'conf':>5}  "
        f"{'sec':>6}"
    )
    print(cols)
    print("  " + "-" * (len(cols) - 2))


def _print_row(idx: int, row: ScoreRow, path_width: int) -> None:
    if row.error:
        print(
            f"  {idx:>3}  {row.path[:path_width]:<{path_width}}  "
            f"{'!':>5}  {'!':>5}  error: {row.error}"
        )
        return
    print(
        f"  {idx:>3}  {row.path[:path_width]:<{path_width}}  "
        f"{row.ref_notes:>5}  {row.est_notes:>5}  "
        f"{row.p_no_offset:>5.2f}  {row.r_no_offset:>5.2f}  {row.f1_no_offset:>5.2f}  "
        f"{row.f1_with_offset:>6.2f}  "
        f"{row.confidence:>5.2f}  "
        f"{row.wall_sec:>6.1f}"
    )


def _aggregate(rows: list[ScoreRow]) -> dict[str, float]:
    """Compute mean / median F1 over non-errored rows.

    We average P/R/F1 per file (equal weight per song) rather than
    micro-averaging over notes. This matches the convention used in
    music-transcription papers and gives a small-corpus eval a
    bounded-variance headline number.
    """
    ok = [r for r in rows if r.error is None]
    if not ok:
        return {}
    def _m(attr: str) -> float:
        return statistics.fmean(getattr(r, attr) for r in ok)
    def _med(attr: str) -> float:
        return statistics.median(getattr(r, attr) for r in ok)
    return {
        "n_files_scored": len(ok),
        "n_files_errored": len(rows) - len(ok),
        "mean_p_no_offset": _m("p_no_offset"),
        "mean_r_no_offset": _m("r_no_offset"),
        "mean_f1_no_offset": _m("f1_no_offset"),
        "median_f1_no_offset": _med("f1_no_offset"),
        "mean_f1_with_offset": _m("f1_with_offset"),
        "median_f1_with_offset": _med("f1_with_offset"),
        "mean_confidence": _m("confidence"),
        "mean_wall_sec": _m("wall_sec"),
    }


def _print_aggregate(agg: dict[str, float]) -> None:
    if not agg:
        print("\nNo files scored.")
        return
    print("\n=== Aggregate ===")
    print(f"  scored:              {int(agg['n_files_scored'])}")
    if agg["n_files_errored"]:
        print(f"  errored:             {int(agg['n_files_errored'])}")
    print(f"  mean P (no-offset):   {agg['mean_p_no_offset']:.3f}")
    print(f"  mean R (no-offset):   {agg['mean_r_no_offset']:.3f}")
    print(f"  mean F1 (no-offset):  {agg['mean_f1_no_offset']:.3f}")
    print(f"  median F1 (no-off):   {agg['median_f1_no_offset']:.3f}")
    print(f"  mean F1 (w/ offset):  {agg['mean_f1_with_offset']:.3f}")
    print(f"  median F1 (w/ off):   {agg['median_f1_with_offset']:.3f}")
    print(f"  mean confidence:      {agg['mean_confidence']:.3f}")
    print(f"  mean wall sec/file:   {agg['mean_wall_sec']:.1f}")


def _role_breakdown(
    rows_with_results: list[tuple[ScoreRow, Any, Any, Any]],
) -> dict[str, dict[str, float]]:
    """Compute per-role F1 across all scored files.

    Re-runs ``_score`` on the subset of predicted notes whose
    ``MidiTrack.instrument`` matches each role. The reference side is
    the *full* ground truth — we don't try to split the reference into
    roles because clean_midi's program numbers don't cleanly map to
    our MELODY/BASS/CHORDS taxonomy. The resulting numbers are lower
    bounds on the per-role extractor quality (the melody extractor
    can only ever hit notes that belong to the true melody, but the
    reference also contains non-melody notes), but deltas between
    runs remain meaningful for A/B tuning.

    Files where the extractor produced zero notes for a given role
    contribute F1 = 0.0 to ``mean_f1_no_offset`` (it's the only
    honest answer: the reference is non-empty, so recall — and thus
    F1 — is zero). Excluding them biased the mean upward whenever a
    tuning change activated the extractor on more files, which
    inverted the A/B signal the harness is meant to produce. The
    subset mean is still reported as ``mean_f1_no_offset_when_active``
    alongside ``n_files_with_role`` for diagnosis.
    """
    roles = ("melody", "bass", "chords", "piano")
    out: dict[str, dict[str, float]] = {}
    for role in roles:
        all_f1s: list[float] = []
        active_f1s: list[float] = []
        for _row, ref_i, ref_p, result in rows_with_results:
            est_i, est_p = _predicted_notes(result, role_filter=role)
            if len(est_i) == 0:
                all_f1s.append(0.0)
                continue
            scores = _score(ref_i, ref_p, est_i, est_p)
            all_f1s.append(scores["f1_no_offset"])
            active_f1s.append(scores["f1_no_offset"])
        if all_f1s:
            out[role] = {
                "mean_f1_no_offset": statistics.fmean(all_f1s),
                "mean_f1_no_offset_when_active": (
                    statistics.fmean(active_f1s) if active_f1s else 0.0
                ),
                "n_files_scored": len(all_f1s),
                "n_files_with_role": len(active_f1s),
            }
    return out


def _print_role_breakdown(role_scores: dict[str, dict[str, float]]) -> None:
    if not role_scores:
        return
    print("\n=== Per-role F1 (predicted role vs full ground truth) ===")
    print("  Note: reference is unsplit, so these are lower bounds.")
    print("  Deltas between runs are what matter for A/B tuning.")
    print("  mean F1 counts files with zero predictions as 0.0;")
    print("  'active' restricts to files where the role fired.")
    for role, scores in role_scores.items():
        print(
            f"  {role:<8}  mean F1 {scores['mean_f1_no_offset']:.3f}  "
            f"(active {scores['mean_f1_no_offset_when_active']:.3f}, "
            f"{int(scores['n_files_with_role'])}/"
            f"{int(scores['n_files_scored'])} files)"
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

@dataclass
class RunConfig:
    dataset_root: Path
    cache_dir: Path
    soundfont: Path
    max_duration: float | None
    sample_rate: int
    paths: list[Path] = field(default_factory=list)


def _eval_one(
    midi_path: Path,
    cfg: RunConfig,
    fluidsynth_bin: str,
) -> tuple[ScoreRow, Any, Any, Any]:
    """Synthesize → transcribe → score a single MIDI file.

    Returns the score row plus the (ref_intervals, ref_pitches,
    result) triple so the per-role breakdown pass can re-score
    without re-running the pipeline.
    """
    display = _fmt_path(midi_path)
    try:
        wav_path, pm = _ensure_synthesized(
            midi_path,
            cfg.max_duration,
            cfg.soundfont,
            cfg.cache_dir,
            fluidsynth_bin,
            cfg.sample_rate,
        )
    except Exception as exc:  # noqa: BLE001 — one bad file must not sink the run
        return ScoreRow(
            path=display, ref_notes=0, est_notes=0,
            p_no_offset=0.0, r_no_offset=0.0, f1_no_offset=0.0,
            p_with_offset=0.0, r_with_offset=0.0, f1_with_offset=0.0,
            confidence=0.0, wall_sec=0.0,
            error=f"synthesis: {exc}",
        ), None, None, None

    try:
        ref_i, ref_p = _ground_truth_notes(pm)
    except Exception as exc:  # noqa: BLE001
        return ScoreRow(
            path=display, ref_notes=0, est_notes=0,
            p_no_offset=0.0, r_no_offset=0.0, f1_no_offset=0.0,
            p_with_offset=0.0, r_with_offset=0.0, f1_with_offset=0.0,
            confidence=0.0, wall_sec=0.0,
            error=f"ground_truth: {exc}",
        ), None, None, None

    try:
        result, wall = _run_transcribe(wav_path)
    except Exception as exc:  # noqa: BLE001
        return ScoreRow(
            path=display, ref_notes=int(len(ref_i)), est_notes=0,
            p_no_offset=0.0, r_no_offset=0.0, f1_no_offset=0.0,
            p_with_offset=0.0, r_with_offset=0.0, f1_with_offset=0.0,
            confidence=0.0, wall_sec=0.0,
            error=f"transcribe: {exc}",
        ), None, None, None

    est_i, est_p = _predicted_notes(result)
    scores = _score(ref_i, ref_p, est_i, est_p)
    return ScoreRow(
        path=display,
        ref_notes=int(len(ref_i)),
        est_notes=int(len(est_i)),
        p_no_offset=scores["p_no_offset"],
        r_no_offset=scores["r_no_offset"],
        f1_no_offset=scores["f1_no_offset"],
        p_with_offset=scores["p_with_offset"],
        r_with_offset=scores["r_with_offset"],
        f1_with_offset=scores["f1_with_offset"],
        confidence=float(result.quality.overall_confidence),
        wall_sec=wall,
    ), ref_i, ref_p, result


def _write_json(
    out_path: Path,
    rows: list[ScoreRow],
    agg: dict[str, float],
    role_scores: dict[str, dict[str, float]],
    cfg: RunConfig,
) -> None:
    """Dump the run as JSON for CI diffing / plotting.

    The schema is flat and tool-friendly on purpose: ``meta`` /
    ``files`` / ``aggregate`` / ``per_role`` — so you can jq
    ``.aggregate.mean_f1_no_offset`` in a CI gate without having to
    know the dataclass layout.
    """
    # Sort files by path so regenerating the baseline is order-stable —
    # ``random.sample`` returns a permutation that depends on both the
    # seed and the full input list, but reviewers care about the scoring
    # output, not the schedule the harness happened to run in.
    sorted_rows = sorted(rows, key=lambda r: r.path)
    payload = {
        "meta": {
            "max_duration_sec": cfg.max_duration,
            "sample_rate": cfg.sample_rate,
            "n_files": len(cfg.paths),
        },
        "files": [asdict(r) for r in sorted_rows],
        "aggregate": agg,
        "per_role": role_scores,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "explicit_paths",
        nargs="*",
        type=Path,
        help="Explicit .mid files to eval (overrides --sample).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Dataset root (default: {DEFAULT_DATASET.relative_to(REPO_ROOT)}).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=25,
        help=(
            "Number of files to randomly sample from the dataset "
            "(default: 25, which scores every file in the tracked "
            "``eval/fixtures/clean_midi/`` subset)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for sampling (default: 42 for reproducibility).",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=DEFAULT_MAX_DURATION,
        help=(
            f"Clip each MIDI to N seconds before synthesis "
            f"(default: {DEFAULT_MAX_DURATION}; pass 0 for full songs)."
        ),
    )
    parser.add_argument(
        "--soundfont",
        type=Path,
        default=None,
        help=(
            "Path to a .sf2 soundfont. Defaults to the TimGM6mb.sf2 "
            "bundled inside the pretty_midi wheel."
        ),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE,
        help=f"WAV cache directory (default: {DEFAULT_CACHE.relative_to(REPO_ROOT)}).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Synthesis sample rate (default: {DEFAULT_SAMPLE_RATE}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path for the full run report.",
    )
    args = parser.parse_args()

    try:
        fluidsynth_bin = _find_fluidsynth()
    except RuntimeError as exc:
        print(f"! {exc}", file=sys.stderr)
        return 1

    soundfont = args.soundfont or _default_soundfont()
    if not soundfont.is_file():
        print(f"! soundfont not found: {soundfont}", file=sys.stderr)
        return 1

    max_duration: float | None = args.max_duration if args.max_duration > 0 else None

    try:
        selected = _select_paths(
            args.dataset, args.explicit_paths, args.sample, args.seed,
        )
    except RuntimeError as exc:
        print(f"! {exc}", file=sys.stderr)
        return 1
    if not selected:
        print("! no files selected", file=sys.stderr)
        return 1

    cfg = RunConfig(
        dataset_root=args.dataset,
        cache_dir=args.cache,
        soundfont=soundfont,
        max_duration=max_duration,
        sample_rate=args.sample_rate,
        paths=selected,
    )

    print("=== Transcription eval ===")
    print(f"  dataset:     {cfg.dataset_root}")
    print(f"  files:       {len(selected)}")
    print(f"  max dur:     {max_duration if max_duration else 'full'}")
    print(f"  soundfont:   {cfg.soundfont.name}")
    print(f"  cache:       {cfg.cache_dir}")
    print(f"  fluidsynth:  {fluidsynth_bin}")
    print()

    path_width = min(50, max(len(_fmt_path(p)) for p in selected))
    _print_header(path_width)

    rows: list[ScoreRow] = []
    scored_with_results: list[tuple[ScoreRow, Any, Any, Any]] = []
    for i, midi_path in enumerate(selected, start=1):
        row, ref_i, ref_p, result = _eval_one(midi_path, cfg, fluidsynth_bin)
        rows.append(row)
        _print_row(i, row, path_width)
        if result is not None:
            scored_with_results.append((row, ref_i, ref_p, result))

    agg = _aggregate(rows)
    _print_aggregate(agg)

    role_scores = _role_breakdown(scored_with_results)
    _print_role_breakdown(role_scores)

    if args.out:
        _write_json(args.out, rows, agg, role_scores, cfg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
