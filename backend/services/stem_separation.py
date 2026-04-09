"""Source-separation stage — Demucs stem split.

Runs `demucs.apply.apply_model`_ on the source waveform to split it
into the four ``htdemucs`` sources — ``{drums, bass, other, vocals}``
— each written to its own WAV tempfile. The transcribe wiring then
feeds each stem to a dedicated downstream stage:

  * ``vocals.wav`` → Basic Pitch → ``MELODY`` note events
  * ``bass.wav``   → Basic Pitch → ``BASS`` note events
  * ``other.wav``  → Basic Pitch → ``CHORDS`` note events
  * ``other.wav``  → ``chord_recognition`` (chroma + triad templates)
  * ``drums.wav``  → ``tempo_map_from_audio_path`` (beat tracking)

With Demucs enabled the transcribe stage skips the Viterbi melody /
bass split entirely: those heuristics exist to compensate for the
fact that Basic Pitch is a single-stream polyphonic tracker run over
a *mixed* waveform, and a proper source-separation front-end renders
them redundant.

Everything here is feature-flagged via
:attr:`backend.config.Settings.demucs_enabled` — **on by default**,
because the stems path produces cleaner role splits than the
single-mix Viterbi heuristics whenever ``demucs`` is installed.
Every failure mode degrades gracefully: missing ``demucs`` / ``torch``,
unreadable audio, model load crash, apply_model OOM, tempfile write
failure all return ``(None, stats)`` with ``stats.skipped = True`` and
a descriptive warning, so the caller falls back to the original
single-mix Basic Pitch path without losing notes. Commercial
deployments that can't accept the CC BY-NC ``htdemucs`` weights
should set ``OHSHEET_DEMUCS_ENABLED=0`` to force the fallback.

Cost
----
Demucs is a heavy model. ``htdemucs`` runs at roughly 0.2–0.5x
real-time on CPU, 2–3x real-time on Apple MPS, and 5–10x real-time on
CUDA. The pretrained weights are ~80 MB and are fetched on first use
via ``demucs.pretrained.get_model``; once loaded, the model is cached
at module scope (``_MODEL_CACHE``) so subsequent jobs pay only the
inference cost.

License
-------
Demucs itself is MIT, but the pretrained ``htdemucs`` weights are
distributed under CC BY-NC 4.0 (non-commercial). Commercial
deployments must either train their own weights, pick a different
pretrained bag with a commercial license, or swap in another
separator — the rest of the transcribe wiring only cares about the
``SeparatedStems`` paths, not which model produced them.

.. _demucs.apply.apply_model: https://github.com/facebookresearch/demucs
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.services._torch_utils import pick_device

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults — mirrored in backend/config.py so config and tests agree.
# ---------------------------------------------------------------------------

# Default 4-stem pretrained bag. htdemucs is the latest Hybrid
# Transformer Demucs model and is what ``demucs.separate`` itself
# picks when ``--name`` is omitted.
DEFAULT_MODEL_NAME = "htdemucs"

# None → use the model's own segment default. Lower values trade
# accuracy for memory; the htdemucs default (~7.8s) is a good balance.
DEFAULT_SEGMENT_SEC: float | None = None

# ``shifts=1`` matches ``demucs.separate`` defaults. Increasing to 5-10
# improves SDR by ~0.2 points but multiplies inference time by ``shifts``.
DEFAULT_SHIFTS = 1

# Overlap between split chunks when ``split=True``. 0.25 is the
# upstream default and gives a good accuracy / speed trade-off.
DEFAULT_OVERLAP = 0.25

# Splitting breaks the input into ~8s chunks so large audio files
# don't OOM. Disable only for very short clips where the extra
# overlap cost dominates.
DEFAULT_SPLIT = True

# Below this duration we bail — Demucs segment inference needs at
# least a few hundred ms of audio to do anything meaningful.
DEFAULT_MIN_DURATION_SEC = 0.5

# Stem names used by the 4-source htdemucs family. Order matches
# ``model.sources`` exactly — these are the names we route on in
# ``_tensor_to_stems``. Custom-trained bags may ship with different
# source lists; we detect missing names via the per-stem ``is None``
# fallback in the consumers rather than failing here.
STEM_VOCALS = "vocals"
STEM_BASS = "bass"
STEM_DRUMS = "drums"
STEM_OTHER = "other"

# Process-scoped cache for loaded models. Keyed on model name so a
# test that swaps in a different pretrained bag gets its own slot.
_MODEL_CACHE: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Data carriers
# ---------------------------------------------------------------------------

@dataclass
class SeparatedStems:
    """Paths to the four stem tempfiles written by :func:`separate_stems`.

    Any subset of these may be ``None`` if the selected Demucs bag
    doesn't emit that source (e.g. a 2-stem vocal/accompaniment model).
    Callers should treat ``None`` as "stem unavailable, fall back to
    the original audio" — that's why each slot is independently
    optional rather than baked into a tuple.

    The stems share a parent ``_tempdir`` so :meth:`cleanup` can
    ``rmtree`` them in one call; the transcribe wiring should always
    invoke ``cleanup()`` in a ``finally`` around the downstream
    inference passes so large WAV blobs don't pile up on disk.
    """
    vocals: Path | None = None
    bass: Path | None = None
    drums: Path | None = None
    other: Path | None = None
    _tempdir: Path | None = None

    def cleanup(self) -> None:
        """Remove the tempdir that holds the stem WAVs. Idempotent."""
        if self._tempdir is not None:
            try:
                shutil.rmtree(self._tempdir, ignore_errors=True)
            except Exception as exc:  # noqa: BLE001 — best-effort cleanup
                log.warning(
                    "failed to clean up demucs tempdir %s: %s",
                    self._tempdir, exc,
                )
            self._tempdir = None
            self.vocals = None
            self.bass = None
            self.drums = None
            self.other = None


@dataclass
class StemSeparationStats:
    """Per-run summary of what the separator did.

    Structured like the other service stats objects (``PreprocessStats``,
    ``ChordRecognitionStats``) so the transcribe wiring can thread it
    through :class:`~backend.contracts.QualitySignal.warnings` the
    same way.
    """
    skipped: bool = False
    model_name: str = ""
    device: str = ""
    wall_time_sec: float = 0.0
    stems_written: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_warnings(self) -> list[str]:
        """One-line human summary entries for the QualitySignal."""
        if self.skipped:
            if self.warnings:
                return [f"stem separation skipped: {w}" for w in self.warnings]
            return ["stem separation skipped"]
        out: list[str] = []
        if self.stems_written:
            stems = ", ".join(self.stems_written)
            out.append(
                f"stem separation: {self.model_name} on {self.device} "
                f"({stems}) in {self.wall_time_sec:.1f}s"
            )
        out.extend(self.warnings)
        return out


# ---------------------------------------------------------------------------
# Model + device helpers
# ---------------------------------------------------------------------------

def _load_model(name: str) -> Any:
    """Lazy-load and cache a pretrained Demucs model. Process-scoped."""
    cached = _MODEL_CACHE.get(name)
    if cached is not None:
        return cached

    from demucs.pretrained import get_model  # noqa: PLC0415

    log.info("Loading Demucs model %r", name)
    model = get_model(name)
    model.eval()
    _MODEL_CACHE[name] = model
    return model


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def separate_stems(
    audio_path: Path,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    segment_sec: float | None = DEFAULT_SEGMENT_SEC,
    shifts: int = DEFAULT_SHIFTS,
    overlap: float = DEFAULT_OVERLAP,
    split: bool = DEFAULT_SPLIT,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
) -> tuple[SeparatedStems | None, StemSeparationStats]:
    """Run Demucs on ``audio_path`` and write each stem to a tempfile.

    Returns ``(stems, stats)``. On any failure returns ``(None, stats)``
    with ``stats.skipped = True`` and a reason in ``stats.warnings``, so
    the caller can fall back cleanly to the original (stemless)
    pipeline.

    The caller owns the returned :class:`SeparatedStems` object and
    **must** call :meth:`SeparatedStems.cleanup` when done — typically
    in a ``try``/``finally`` around the downstream inference passes.
    Large Demucs outputs (multi-minute audio × 4 stems) can pile up
    fast if cleanup is skipped.
    """
    stats = StemSeparationStats(model_name=model_name)

    # Deps — late-imported so the module is importable (and
    # unit-testable with ``pytest.importorskip``) on machines without
    # demucs / torch.
    try:
        import torch  # noqa: PLC0415
        from demucs.apply import apply_model  # noqa: PLC0415
        from demucs.audio import AudioFile, save_audio  # noqa: PLC0415
    except ImportError as exc:
        log.debug("demucs deps unavailable: %s", exc)
        stats.skipped = True
        stats.warnings.append(f"missing dep: {exc.name}")
        return None, stats

    if not audio_path.is_file():
        stats.skipped = True
        stats.warnings.append(f"audio file missing: {audio_path}")
        return None, stats

    try:
        model = _load_model(model_name)
    except Exception as exc:  # noqa: BLE001 — network/disk/import at load
        log.warning("Failed to load Demucs model %r: %s", model_name, exc)
        stats.skipped = True
        stats.warnings.append(f"model load failed: {exc}")
        return None, stats

    device_str = pick_device(device)
    stats.device = device_str

    # Load at the model's native samplerate / channel count so apply
    # sees exactly what the weights were trained on. AudioFile shells
    # out to ffmpeg — we already require ffmpeg for audio ingest
    # upstream, so there's no torchaudio fallback dance here.
    try:
        wav = AudioFile(audio_path).read(
            streams=0,
            samplerate=model.samplerate,
            channels=model.audio_channels,
        )
    except Exception as exc:  # noqa: BLE001 — bad bytes / ffmpeg missing
        log.warning("Demucs load failed for %s: %s", audio_path, exc)
        stats.skipped = True
        stats.warnings.append(f"load failed: {exc}")
        return None, stats

    if wav is None or wav.numel() == 0:
        stats.skipped = True
        stats.warnings.append("empty audio")
        return None, stats

    duration = float(wav.shape[-1] / model.samplerate) if model.samplerate else 0.0
    if duration < min_duration_sec:
        stats.skipped = True
        stats.warnings.append(f"audio too short ({duration:.2f}s)")
        return None, stats

    # Zero-mean, unit-variance inference, undone after apply. This is
    # what ``demucs.separate`` does — the model was trained on
    # normalized inputs and we don't want to drift from that.
    ref = wav.mean(0)
    ref_mean = ref.mean()
    ref_std = ref.std()
    if float(ref_std) < 1e-8:
        stats.skipped = True
        stats.warnings.append("silent input")
        return None, stats
    wav_norm = (wav - ref_mean) / ref_std

    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            sources = apply_model(
                model,
                wav_norm[None],  # add batch dim → (1, C, T)
                device=device_str,
                shifts=shifts,
                split=split,
                overlap=overlap,
                progress=False,
                num_workers=0,
                segment=segment_sec,
            )
        sources = sources[0]  # drop batch dim → (S, C, T)
        sources = sources * ref_std + ref_mean
    except Exception as exc:  # noqa: BLE001 — inference boundary
        log.warning("Demucs apply_model failed for %s: %s", audio_path, exc)
        stats.skipped = True
        stats.warnings.append(f"apply failed: {exc}")
        return None, stats
    stats.wall_time_sec = time.perf_counter() - t0

    # Write each stem to its own WAV inside a dedicated tempdir so
    # SeparatedStems.cleanup can rmtree the whole thing in one call.
    tempdir = Path(tempfile.mkdtemp(prefix="ohsheet-demucs-"))
    stems = SeparatedStems(_tempdir=tempdir)

    try:
        for source, name in zip(sources, model.sources):
            out_path = tempdir / f"{name}.wav"
            save_audio(
                source,
                str(out_path),
                samplerate=model.samplerate,
                clip="rescale",
                bits_per_sample=16,
                as_float=False,
            )
            if name == STEM_VOCALS:
                stems.vocals = out_path
            elif name == STEM_BASS:
                stems.bass = out_path
            elif name == STEM_DRUMS:
                stems.drums = out_path
            elif name == STEM_OTHER:
                stems.other = out_path
            else:
                # Unknown stem from a custom bag — still write it so
                # the tempdir cleanup removes it, but don't attach it
                # to a SeparatedStems slot. The consumer routing is
                # name-based and will just treat that stem as absent.
                log.debug("ignoring unknown demucs stem %r", name)
            stats.stems_written.append(name)
    except Exception as exc:  # noqa: BLE001 — write boundary
        log.warning("Demucs stem write failed: %s", exc)
        stems.cleanup()
        stats.skipped = True
        stats.warnings.append(f"write failed: {exc}")
        return None, stats

    log.debug(
        "demucs separated %s into %d stems on %s in %.2fs",
        audio_path, len(stats.stems_written), device_str, stats.wall_time_sec,
    )
    return stems, stats
