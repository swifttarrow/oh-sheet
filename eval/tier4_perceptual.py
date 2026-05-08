"""Tier 4 perceptual / re-synthesis metrics for the Phase 7 eval ladder.

Tier 4 measures the **end-to-end perceptual fidelity** of the
transcription pipeline by re-synthesizing the engraved MIDI and
comparing it back to the original input audio. Every Tier 4 metric is
reference-free in the strategy doc's sense — the "reference" is the
input audio itself, not a paired piano cover. This is the metric class
that gives us production-monitoring quality measurement on user
uploads without a ground-truth MIDI.

See ``docs/research/transcription-improvement-strategy.md`` Part III
§2.4 for the metric definitions and §8.2 for the composite weighting
used by the production-Q score.

The four metrics:

* :func:`chroma_cosine_score` — beat-bucketed ``chroma_cqt`` cosine
  between input audio and FluidSynth resynth. Re-exported from
  :mod:`eval.tier_rf` so Tier 4's public surface is self-contained.
* :func:`round_trip_f1_score` — ``mir_eval.transcription.f_measure``
  on the (transcribe(input_audio), transcribe(resynth(engraved_midi)))
  pair. Self-consistency F1: a perfect arrange/engrave round-trip
  scores 1.0; drops here implicate the symbolic pipeline rather than
  the transcriber.
* :func:`clap_cosine_score` — LAION-CLAP music checkpoint cosine
  between input audio and resynth. **Heavy dep, not in CI by default.**
  Skips gracefully (returns ``None`` + a note) when ``laion_clap`` is
  missing.
* :func:`mert_cosine_score` — ``m-a-p/MERT-v1-330M`` embedding cosine.
  **Heavy dep, not in CI by default.** Skips gracefully when
  ``transformers`` / weights aren't available.

The :func:`compute_tier4` entry point runs all four for one
``(input_audio, engraved_midi_bytes, transcribe_callable)`` triple
and returns a :class:`Tier4Result`. The ``transcribe_callable`` is
the round-trip's hook into Basic Pitch (or any future replacement) —
the test harness passes in :func:`backend.services.transcribe._run_basic_pitch_sync`,
unit tests pass in a pure-Python stub.

The composite ``Tier4Result.composite`` follows strategy doc §8.2's
``tier4 = (s_clap + s_chroma + s_rtf1) / 3`` shape, with one
clarification: the composite drops missing terms (``None``) and
re-averages over what ran. A CI run without CLAP/MERT installed
reports an honest 2-of-3 average, not an artificially zero composite.
"""
from __future__ import annotations

import io
import logging
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval.tier_rf import (
    CHROMA_SR,
    FLUIDSYNTH_SR,
    chroma_rf_score,
    fluidsynth_resynth,
)

log = logging.getLogger(__name__)

# Round-trip note-event tolerances. Defaults from
# ``mir_eval.transcription.f_measure``: 50ms onset, no offset
# constraint when ``offset_ratio=None``. Strategy doc §2.4 cites these
# as the de-facto MIREX values; we expose them as constants so future
# tier-1 work can share them.
ROUND_TRIP_ONSET_TOLERANCE_SEC = 0.050
ROUND_TRIP_PITCH_TOLERANCE_CENTS = 50.0
ROUND_TRIP_OFFSET_RATIO = 0.2  # tolerance is 20% of note duration

# CLAP-music checkpoint per strategy doc §2.4 — the music-domain
# fine-tune outperforms the general LAION-CLAP for music-style audio.
# Setting via env var ``OHSHEET_CLAP_CKPT`` overrides the default so
# CI can pin a vendored copy without code changes.
CLAP_DEFAULT_CKPT = "music_audioset_epoch_15_esc_90.14.pt"

# MERT model id per strategy doc §2.4. v1-330M is the largest checkpoint
# that runs in <1s per 30-second clip on a single CPU; the 95M variant
# is the cheaper alternative for nightly. Configurable via env var
# ``OHSHEET_MERT_MODEL``.
MERT_DEFAULT_MODEL = "m-a-p/MERT-v1-330M"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class Tier4Result:
    """Per-song Tier 4 metrics produced by :func:`compute_tier4`.

    All headline fields are in ``[0, 1]``. Heavy-dep metrics
    (``clap_cosine``, ``mert_cosine``) default to ``None`` when their
    Python deps aren't installed; the composite drops missing terms
    and re-averages so a deps-light CI run reports an honest
    chord+round-trip composite without an artificial zero.
    """

    chroma_cosine: float | None
    round_trip_f1_no_offset: float | None
    round_trip_f1_with_offset: float | None
    clap_cosine: float | None
    mert_cosine: float | None
    n_chroma_beats: int
    n_notes_input_transcription: int
    n_notes_resynth_transcription: int
    notes: list[str] = field(default_factory=list)

    @property
    def composite(self) -> float:
        """Strategy doc §8.2: ``(clap + chroma + round_trip_f1) / 3``.

        Drops ``None`` entries and re-averages over the present terms.
        Returns 0.0 when no Tier 4 metric ran (e.g., FluidSynth missing).
        """
        parts: list[float] = []
        if self.chroma_cosine is not None:
            parts.append(self.chroma_cosine)
        if self.round_trip_f1_no_offset is not None:
            parts.append(self.round_trip_f1_no_offset)
        if self.clap_cosine is not None:
            parts.append(self.clap_cosine)
        if not parts:
            return 0.0
        return sum(parts) / len(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "chroma_cosine": _round_or_none(self.chroma_cosine),
            "round_trip_f1_no_offset": _round_or_none(self.round_trip_f1_no_offset),
            "round_trip_f1_with_offset": _round_or_none(self.round_trip_f1_with_offset),
            "clap_cosine": _round_or_none(self.clap_cosine),
            "mert_cosine": _round_or_none(self.mert_cosine),
            "composite": round(self.composite, 4),
            "n_chroma_beats": self.n_chroma_beats,
            "n_notes_input_transcription": self.n_notes_input_transcription,
            "n_notes_resynth_transcription": self.n_notes_resynth_transcription,
            "notes": list(self.notes),
        }


def _round_or_none(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


# ---------------------------------------------------------------------------
# Chroma cosine — re-export from tier_rf
# ---------------------------------------------------------------------------

def chroma_cosine_score(
    input_audio: tuple[Any, int],
    resynth_audio: tuple[Any, int],
) -> tuple[float, int, list[str]]:
    """Beat-bucketed chroma_cqt cosine between input and resynth audio.

    Identical to :func:`eval.tier_rf.chroma_rf_score`. Re-exported
    here so Tier 4's public surface is self-contained — the harness
    reaches for ``tier4_perceptual.chroma_cosine_score`` without a
    cross-tier import. Returns ``(score, n_beats, notes)``.
    """
    return chroma_rf_score(input_audio, resynth_audio)


# ---------------------------------------------------------------------------
# Round-trip self-consistency F1
# ---------------------------------------------------------------------------

def round_trip_f1_score(
    input_audio_path: Path,
    engraved_midi_bytes: bytes,
    transcribe_callable: Callable[[Path], bytes],
    *,
    onset_tolerance_sec: float = ROUND_TRIP_ONSET_TOLERANCE_SEC,
    pitch_tolerance_cents: float = ROUND_TRIP_PITCH_TOLERANCE_CENTS,
    offset_ratio: float | None = ROUND_TRIP_OFFSET_RATIO,
    fluidsynth_bin: str | None = None,
    soundfont_path: Path | None = None,
) -> tuple[float | None, float | None, int, int, list[str]]:
    """Self-consistency F1 between input transcription and resynth transcription.

    Concrete pipeline:

    1. Run ``transcribe_callable(input_audio_path)`` → ``midi1_bytes``.
       This is "what the transcriber thinks the song is".
    2. FluidSynth-render ``engraved_midi_bytes`` to a temp WAV.
    3. Run ``transcribe_callable(resynth_wav_path)`` → ``midi2_bytes``.
       This is "what the transcriber thinks the engraved score is".
    4. ``mir_eval.transcription.f_measure(notes1, notes2)`` —
       no-offset and with-offset variants.

    Drops here implicate the **symbolic pipeline** (arrange / humanize /
    engrave) rather than the transcriber, which sees both audios
    through the same model. Strategy doc §4.4: this is the diagnostic
    that quantifies arrange/engrave loss song-by-song without needing
    a reference cover.

    The ``transcribe_callable`` indirection keeps this metric free of
    a hard dependency on Basic Pitch — the harness wires up the real
    transcribe function; tests can pass a pure-Python stub. Returns
    ``(f1_no_offset, f1_with_offset, n_notes_input, n_notes_resynth, notes)``.
    A failure on any side returns ``(None, None, n_in, n_rs, [reason])``
    rather than raising — Phase 0's "no exceptions" contract.
    """
    notes: list[str] = []

    try:
        import mir_eval  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        import pretty_midi  # noqa: PLC0415
    except ImportError as exc:
        notes.append(f"round_trip_f1: import failed: {exc}")
        return None, None, 0, 0, notes

    try:
        midi1_bytes = transcribe_callable(input_audio_path)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"round_trip_f1: transcribe(input) failed: {exc}")
        return None, None, 0, 0, notes

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        resynth_wav = td_path / "resynth.wav"
        try:
            audio, sr = fluidsynth_resynth(
                engraved_midi_bytes,
                sample_rate=FLUIDSYNTH_SR,
                soundfont_path=soundfont_path,
                fluidsynth_bin=fluidsynth_bin,
            )
            import soundfile as sf  # noqa: PLC0415
            sf.write(str(resynth_wav), audio, sr)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"round_trip_f1: fluidsynth failed: {exc}")
            return None, None, 0, 0, notes

        try:
            midi2_bytes = transcribe_callable(resynth_wav)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"round_trip_f1: transcribe(resynth) failed: {exc}")
            return None, None, 0, 0, notes

    try:
        notes1 = _midi_to_note_events(midi1_bytes, pretty_midi)
        notes2 = _midi_to_note_events(midi2_bytes, pretty_midi)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"round_trip_f1: parse failed: {exc}")
        return None, None, 0, 0, notes

    n_in = len(notes1["onsets"])
    n_rs = len(notes2["onsets"])
    if n_in == 0 or n_rs == 0:
        notes.append(
            f"round_trip_f1: empty note set (n_input={n_in}, n_resynth={n_rs})"
        )
        return None, None, n_in, n_rs, notes

    ref_intervals = np.column_stack((notes1["onsets"], notes1["offsets"]))
    est_intervals = np.column_stack((notes2["onsets"], notes2["offsets"]))
    ref_pitches = np.array(notes1["pitches"], dtype=float)
    est_pitches = np.array(notes2["pitches"], dtype=float)

    try:
        p_no, r_no, f_no, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, ref_pitches, est_intervals, est_pitches,
            onset_tolerance=onset_tolerance_sec,
            pitch_tolerance=pitch_tolerance_cents,
            offset_ratio=None,
        )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"round_trip_f1: mir_eval no-offset failed: {exc}")
        f_no = None

    try:
        p_w, r_w, f_w, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, ref_pitches, est_intervals, est_pitches,
            onset_tolerance=onset_tolerance_sec,
            pitch_tolerance=pitch_tolerance_cents,
            offset_ratio=offset_ratio,
        )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"round_trip_f1: mir_eval with-offset failed: {exc}")
        f_w = None

    return (
        _clamp_optional(f_no),
        _clamp_optional(f_w),
        n_in,
        n_rs,
        notes,
    )


def _midi_to_note_events(midi_bytes: bytes, pretty_midi_mod: Any) -> dict[str, list[float]]:
    """Flatten a MIDI's notes to onsets / offsets / pitches lists.

    Pitches are in Hz (mir_eval's expected unit). Tracks are merged —
    we don't keep per-instrument structure because mir_eval F1
    treats all notes equally.
    """
    pm = pretty_midi_mod.PrettyMIDI(io.BytesIO(midi_bytes))
    onsets: list[float] = []
    offsets: list[float] = []
    pitches_hz: list[float] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            onsets.append(float(n.start))
            offsets.append(float(n.end))
            pitches_hz.append(float(440.0 * (2.0 ** ((n.pitch - 69) / 12.0))))
    return {"onsets": onsets, "offsets": offsets, "pitches": pitches_hz}


def _clamp_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


# ---------------------------------------------------------------------------
# CLAP-music cosine — heavy dep, optional
# ---------------------------------------------------------------------------

def clap_cosine_score(
    input_audio_path: Path,
    resynth_audio: tuple[Any, int],
    *,
    checkpoint: str | None = None,
) -> tuple[float | None, list[str]]:
    """LAION-CLAP cosine between input audio and resynth audio.

    Returns ``(cosine_in_unit_range, notes)`` where the cosine is
    rescaled from CLAP's native ``[-1, 1]`` to ``[0, 1]`` per the
    composite weighting in §8.2: ``s_clap = (clap_cosine + 1) / 2``.

    Skips gracefully when ``laion_clap`` isn't installed: returns
    ``(None, ["clap_cosine: laion_clap not installed"])``. CI runs
    without the heavy dep see the metric as "not run", and the
    composite filters it out — no artificial zero.
    """
    notes: list[str] = []

    try:
        import laion_clap  # type: ignore[import-not-found]  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        import soundfile as sf  # noqa: PLC0415
    except ImportError as exc:
        notes.append(f"clap_cosine: deps missing: {exc}")
        return None, notes

    if checkpoint is None:
        import os  # noqa: PLC0415
        checkpoint = os.environ.get("OHSHEET_CLAP_CKPT", CLAP_DEFAULT_CKPT)

    rs_audio, rs_sr = resynth_audio

    with tempfile.TemporaryDirectory() as td:
        rs_path = Path(td) / "resynth.wav"
        try:
            sf.write(str(rs_path), rs_audio, rs_sr)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"clap_cosine: write resynth failed: {exc}")
            return None, notes

        try:
            model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
            model.load_ckpt(checkpoint)
            embeds = model.get_audio_embedding_from_filelist(
                x=[str(input_audio_path), str(rs_path)],
                use_tensor=False,
            )
        except Exception as exc:  # noqa: BLE001
            notes.append(f"clap_cosine: model failed: {exc}")
            return None, notes

    a = np.asarray(embeds[0], dtype=float)
    b = np.asarray(embeds[1], dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        notes.append("clap_cosine: zero-norm embedding")
        return None, notes
    cos = float(np.dot(a, b) / (na * nb))
    # Rescale [-1, 1] → [0, 1] per strategy doc §8.2.
    rescaled = max(0.0, min(1.0, (cos + 1.0) / 2.0))
    return rescaled, notes


# ---------------------------------------------------------------------------
# MERT cosine — heavy dep, optional
# ---------------------------------------------------------------------------

def mert_cosine_score(
    input_audio_path: Path,
    resynth_audio: tuple[Any, int],
    *,
    model_id: str | None = None,
) -> tuple[float | None, list[str]]:
    """``m-a-p/MERT-v1-330M`` embedding cosine between input and resynth.

    Mean-pools the per-frame hidden states from MERT's last layer
    on each audio, then takes the cosine of the two pooled vectors.
    Returns the cosine clamped to ``[0, 1]`` (MERT embeddings are
    already non-negative-correlated for similar music inputs).

    Skips gracefully when ``transformers``/``torch`` aren't installed.
    """
    notes: list[str] = []

    try:
        import numpy as np  # noqa: PLC0415
        import soundfile as sf  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from transformers import AutoModel, Wav2Vec2FeatureExtractor  # noqa: PLC0415
    except ImportError as exc:
        notes.append(f"mert_cosine: deps missing: {exc}")
        return None, notes

    if model_id is None:
        import os  # noqa: PLC0415
        model_id = os.environ.get("OHSHEET_MERT_MODEL", MERT_DEFAULT_MODEL)

    try:
        in_audio, in_sr = sf.read(str(input_audio_path))
        if in_audio.ndim > 1:
            in_audio = in_audio.mean(axis=1)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"mert_cosine: read input failed: {exc}")
        return None, notes

    rs_audio, rs_sr = resynth_audio

    try:
        feat = Wav2Vec2FeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        model.eval()
    except Exception as exc:  # noqa: BLE001
        notes.append(f"mert_cosine: model load failed: {exc}")
        return None, notes

    target_sr = int(getattr(feat, "sampling_rate", 24_000))

    try:
        in_resampled = _resample_to(in_audio, in_sr, target_sr)
        rs_resampled = _resample_to(rs_audio, rs_sr, target_sr)
        with torch.no_grad():
            in_embed = _mert_pooled_embed(model, feat, in_resampled, target_sr)
            rs_embed = _mert_pooled_embed(model, feat, rs_resampled, target_sr)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"mert_cosine: inference failed: {exc}")
        return None, notes

    a = np.asarray(in_embed, dtype=float)
    b = np.asarray(rs_embed, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        notes.append("mert_cosine: zero-norm embedding")
        return None, notes
    cos = float(np.dot(a, b) / (na * nb))
    return max(0.0, min(1.0, cos)), notes


def _resample_to(audio: Any, src_sr: int, tgt_sr: int) -> Any:
    if src_sr == tgt_sr:
        return audio
    import librosa  # noqa: PLC0415
    return librosa.resample(audio, orig_sr=src_sr, target_sr=tgt_sr)


def _mert_pooled_embed(model: Any, feat: Any, audio: Any, sr: int) -> Any:
    import torch  # noqa: PLC0415

    inputs = feat(audio, sampling_rate=sr, return_tensors="pt")
    outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.last_hidden_state[0].mean(dim=0).cpu().numpy()
    del outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return hidden


# ---------------------------------------------------------------------------
# Top-level entrypoint
# ---------------------------------------------------------------------------

def compute_tier4(
    input_audio_path: Path,
    engraved_midi_bytes: bytes,
    transcribe_callable: Callable[[Path], bytes] | None = None,
    *,
    enable_clap: bool = False,
    enable_mert: bool = False,
    fluidsynth_bin: str | None = None,
    soundfont_path: Path | None = None,
) -> Tier4Result:
    """Run all four Tier 4 metrics for one ``(audio, MIDI, transcribe)`` triple.

    Loads the input audio at :data:`eval.tier_rf.CHROMA_SR`,
    FluidSynth-renders the engraved MIDI, then computes:

    * chroma cosine (always)
    * round-trip self-consistency F1 (when ``transcribe_callable`` is provided)
    * CLAP-music cosine (when ``enable_clap`` AND ``laion_clap`` is importable)
    * MERT cosine (when ``enable_mert`` AND ``transformers``/``torch`` available)

    The CI/per-PR ``eval ci`` subcommand passes ``enable_clap=False``
    and ``enable_mert=False`` so the gate stays under 2 minutes;
    nightly flips both on per strategy doc §5.2.

    Returns a :class:`Tier4Result`. Per-metric failures (missing
    librosa, fluidsynth, transcribe error) populate ``notes`` rather
    than raising — Phase 0's contract.
    """
    import librosa  # noqa: PLC0415

    notes: list[str] = []

    try:
        in_y, in_sr = librosa.load(str(input_audio_path), sr=CHROMA_SR, mono=True)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"compute_tier4: load input audio failed: {exc}")
        return Tier4Result(
            chroma_cosine=None,
            round_trip_f1_no_offset=None,
            round_trip_f1_with_offset=None,
            clap_cosine=None,
            mert_cosine=None,
            n_chroma_beats=0,
            n_notes_input_transcription=0,
            n_notes_resynth_transcription=0,
            notes=notes,
        )

    try:
        rs_y, rs_sr = fluidsynth_resynth(
            engraved_midi_bytes,
            sample_rate=FLUIDSYNTH_SR,
            soundfont_path=soundfont_path,
            fluidsynth_bin=fluidsynth_bin,
        )
        if rs_sr != CHROMA_SR:
            rs_y_chroma = librosa.resample(rs_y, orig_sr=rs_sr, target_sr=CHROMA_SR)
            rs_sr_chroma = CHROMA_SR
        else:
            rs_y_chroma = rs_y
            rs_sr_chroma = rs_sr
    except Exception as exc:  # noqa: BLE001
        notes.append(f"compute_tier4: fluidsynth resynth failed: {exc}")
        rs_y = rs_y_chroma = None
        rs_sr = rs_sr_chroma = 0

    if rs_y_chroma is not None:
        chroma_score, n_beats, chroma_notes = chroma_cosine_score(
            (in_y, in_sr), (rs_y_chroma, rs_sr_chroma),
        )
        notes.extend(chroma_notes)
    else:
        chroma_score, n_beats = None, 0

    if transcribe_callable is not None:
        rt_no, rt_w, n_in_notes, n_rs_notes, rt_notes = round_trip_f1_score(
            input_audio_path, engraved_midi_bytes, transcribe_callable,
            fluidsynth_bin=fluidsynth_bin, soundfont_path=soundfont_path,
        )
        notes.extend(rt_notes)
    else:
        notes.append("round_trip_f1: skipped (no transcribe_callable provided)")
        rt_no, rt_w, n_in_notes, n_rs_notes = None, None, 0, 0

    clap_score: float | None = None
    if enable_clap and rs_y is not None:
        clap_score, clap_notes = clap_cosine_score(
            input_audio_path, (rs_y, rs_sr),
        )
        notes.extend(clap_notes)

    mert_score: float | None = None
    if enable_mert and rs_y is not None:
        mert_score, mert_notes = mert_cosine_score(
            input_audio_path, (rs_y, rs_sr),
        )
        notes.extend(mert_notes)

    return Tier4Result(
        chroma_cosine=chroma_score,
        round_trip_f1_no_offset=rt_no,
        round_trip_f1_with_offset=rt_w,
        clap_cosine=clap_score,
        mert_cosine=mert_score,
        n_chroma_beats=n_beats,
        n_notes_input_transcription=n_in_notes,
        n_notes_resynth_transcription=n_rs_notes,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Diagnostic helper: confirm fluidsynth is present (cheaper than running it)
# ---------------------------------------------------------------------------

def fluidsynth_available() -> bool:
    """Return True iff the fluidsynth CLI is on PATH.

    Used by the harness to skip Tier 4 cleanly on environments without
    fluidsynth (e.g., minimal CI containers) rather than catching the
    runtime ``RuntimeError`` from :func:`fluidsynth_resynth`.
    """
    return shutil.which("fluidsynth") is not None


def fluidsynth_version() -> str | None:
    """Return the fluidsynth ``--version`` string, or ``None`` if unavailable.

    Diagnostic for the run manifest — the harness records the binary
    version alongside per-song scores so an unexpected resynth jump
    can be traced to a binary upgrade.
    """
    binary = shutil.which("fluidsynth")
    if binary is None:
        return None
    try:
        out = subprocess.check_output([binary, "--version"], text=True)
    except Exception:  # noqa: BLE001
        return None
    return out.strip().splitlines()[0] if out else None
