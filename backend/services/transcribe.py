"""Transcription stage — pretrained MT3 baseline.

Wraps the vendored MR-MT3 inference helpers under
``backend.vendor.mr_mt3`` into the async pipeline. The model architecture
and the pretrained checkpoint both live in-tree
(``backend/vendor/mr_mt3/``); the checkpoint is tracked via git-lfs.

This is a baseline — MT3 takes the full mix and emits GM-program-tagged
notes, so the contract no longer carries any stem URIs. Source separation
is intentionally out of scope.

If torch / the checkpoint / the audio file aren't available the service
falls back to a small stub TranscriptionResult so the rest of the pipeline
(and the unit tests, which run without torch installed) can still be
exercised end-to-end.
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from backend.config import settings
from backend.contracts import (
    SCHEMA_VERSION,
    HarmonicAnalysis,
    InputBundle,
    InstrumentRole,
    MidiTrack,
    Note,
    QualitySignal,
    TempoMapEntry,
    TranscriptionResult,
)
from backend.services.audio_timing import tempo_map_from_audio_path

log = logging.getLogger(__name__)


# Cached MT3 model — building it costs ~5s and a few hundred MB of RAM, so we
# load it once per process. Held as Any to avoid importing torch at module
# import time (heavy dep, optional for the stub fallback).
_MT3_MODEL: Any = None
_MT3_DEVICE: Any = None


def _gm_to_role(program: int, is_drum: bool) -> InstrumentRole:
    """GM program → InstrumentRole. Mirrors transcribe_mrmt3._gm_to_role."""
    if is_drum:
        return InstrumentRole.OTHER
    if program < 8:
        return InstrumentRole.PIANO
    if 24 <= program <= 31:
        return InstrumentRole.CHORDS
    if 32 <= program <= 39:
        return InstrumentRole.BASS
    if 40 <= program <= 51:
        return InstrumentRole.CHORDS
    if 56 <= program <= 71:
        return InstrumentRole.CHORDS
    if 72 <= program <= 79:
        return InstrumentRole.MELODY
    return InstrumentRole.OTHER


def _track_confidence(notes: list[Any], role: InstrumentRole) -> float:
    """Heuristic per-track confidence (mirrors transcribe_mrmt3)."""
    if not notes:
        return 0.1
    durations = [n.end_time - n.start_time for n in notes]
    median_dur = sorted(durations)[len(durations) // 2]
    short_frac = sum(1 for d in durations if d < 0.05) / len(durations)
    dur_score = max(0.0, 1.0 - short_frac * 2)
    span = max(notes[-1].end_time - notes[0].start_time, 1.0)
    density = len(notes) / span
    if density < 15:
        density_score = min(1.0, density / 2.0)
    else:
        density_score = max(0.3, 1.0 - (density - 15) / 30)
    role_bonus = 0.15 if role in (
        InstrumentRole.PIANO, InstrumentRole.BASS, InstrumentRole.MELODY
    ) else 0.0
    conf = (
        0.3 * dur_score
        + 0.3 * density_score
        + 0.2 * min(median_dur / 0.3, 1.0)
        + 0.2
        + role_bonus
    )
    return round(min(max(conf, 0.1), 1.0), 2)


def _ns_to_transcription_result(
    ns: Any,
    default_bpm: float = 120.0,
    *,
    tempo_map_override: list[TempoMapEntry] | None = None,
) -> TranscriptionResult:
    """Convert a note_seq.NoteSequence to our pydantic TranscriptionResult.

    Self-contained — does *not* import temp1/contracts. Harmonic key/chords
    stay placeholder until a dedicated analysis stage exists.

    If ``tempo_map_override`` is set (e.g. from waveform beat tracking), it
    replaces the single-tempo map derived from the NoteSequence so arrange's
    ``sec_to_beat`` aligns quantization to real beats.
    """
    # Group notes by (program, is_drum)
    groups: dict[tuple[int, bool], list[Any]] = {}
    for note in ns.notes:
        key = (int(note.program), bool(note.is_drum))
        groups.setdefault(key, []).append(note)

    midi_tracks: list[MidiTrack] = []
    for (program, is_drum), notes in groups.items():
        if not notes:
            continue
        role = _gm_to_role(program, is_drum)
        contract_notes = [
            Note(
                pitch=int(n.pitch),
                onset_sec=float(n.start_time),
                offset_sec=float(n.end_time),
                velocity=int(n.velocity),
            )
            for n in notes
        ]
        midi_tracks.append(MidiTrack(
            notes=contract_notes,
            instrument=role,
            program=None if is_drum else program,
            confidence=_track_confidence(notes, role),
        ))

    if tempo_map_override:
        tempo_map = tempo_map_override
    else:
        bpm = default_bpm
        if ns.tempos:
            bpm = float(ns.tempos[0].qpm)
        tempo_map = [TempoMapEntry(time_sec=0.0, beat=0.0, bpm=bpm)]

    analysis = HarmonicAnalysis(
        key="C:major",
        time_signature=(4, 4),
        tempo_map=tempo_map,
        chords=[],
        sections=[],
    )

    total_notes = sum(len(t.notes) for t in midi_tracks)
    warnings: list[str] = ["Pretrained MT3 baseline (no source separation)"]
    if tempo_map_override:
        warnings.append("tempo_map from audio beat tracking (librosa)")
    if total_notes < 20:
        warnings.append(f"Low note count ({total_notes}) — possible quality issue")
    avg_conf = (
        sum(t.confidence for t in midi_tracks) / len(midi_tracks)
        if midi_tracks else 0.3
    )
    quality = QualitySignal(
        overall_confidence=round(avg_conf, 2),
        warnings=warnings,
    )

    return TranscriptionResult(
        schema_version=SCHEMA_VERSION,
        midi_tracks=midi_tracks,
        analysis=analysis,
        quality=quality,
    )


def _stub_result(reason: str) -> TranscriptionResult:
    """Tiny shape-correct fallback so downstream stages still run."""
    return TranscriptionResult(
        schema_version=SCHEMA_VERSION,
        midi_tracks=[
            MidiTrack(
                notes=[
                    Note(pitch=60, onset_sec=0.0, offset_sec=0.5, velocity=80),
                    Note(pitch=64, onset_sec=0.5, offset_sec=1.0, velocity=80),
                    Note(pitch=67, onset_sec=1.0, offset_sec=1.5, velocity=80),
                    Note(pitch=72, onset_sec=1.5, offset_sec=2.0, velocity=80),
                ],
                instrument=InstrumentRole.MELODY,
                program=None,
                confidence=0.7,
            ),
        ],
        analysis=HarmonicAnalysis(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
            chords=[],
            sections=[],
        ),
        quality=QualitySignal(
            overall_confidence=0.3,
            warnings=[f"MT3 fallback stub: {reason}"],
        ),
    )


def _load_mt3_model() -> tuple[Any, Any]:
    """Lazy-load the MT3 model on first use. Cached for the process lifetime."""
    global _MT3_MODEL, _MT3_DEVICE
    if _MT3_MODEL is not None:
        return _MT3_MODEL, _MT3_DEVICE

    ckpt = settings.mt3_checkpoint_path
    if ckpt is None:
        raise RuntimeError("OHSHEET_MT3_CHECKPOINT_PATH not configured")
    if not Path(ckpt).is_file():
        raise FileNotFoundError(f"MT3 checkpoint missing: {ckpt}")

    import torch  # noqa: PLC0415 — heavy/optional dep

    from backend.vendor.mr_mt3.inference import load_model  # noqa: PLC0415

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    log.info("Loading pretrained MT3 from %s on %s", ckpt, device)
    model, _config = load_model(str(ckpt), device)
    _MT3_MODEL = model
    _MT3_DEVICE = device
    return model, device


def _audio_path_from_uri(uri: str) -> Path:
    """Resolve a Remote*File URI to a real path on disk.

    Today we only handle file:// URIs (LocalBlobStore). When the S3 store
    lands, this should download to a temp file instead.
    """
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"TranscribeService can only read file:// URIs, got {uri!r}")
    return Path(parsed.path)


def _run_mt3_sync(audio_path: Path) -> TranscriptionResult:
    """Synchronous MT3 inference. Run inside asyncio.to_thread."""
    model, device = _load_mt3_model()
    from backend.vendor.mr_mt3.inference import (  # noqa: PLC0415
        rescale_velocity_to_rms,
    )
    from backend.vendor.mr_mt3.inference import (
        transcribe as mt3_transcribe,
    )

    ns = mt3_transcribe(
        str(audio_path),
        model,
        device,
        batch_size=settings.mt3_batch_size,
    )
    # MT3 is trained with num_velocity_bins=1 → all velocities = 127. Reshape
    # them via onset RMS so dynamics survive into the score.
    ns = rescale_velocity_to_rms(ns, str(audio_path))
    audio_tempo_map = tempo_map_from_audio_path(audio_path)
    return _ns_to_transcription_result(ns, tempo_map_override=audio_tempo_map)


class TranscribeService:
    name = "transcribe"

    async def run(self, payload: InputBundle) -> TranscriptionResult:
        if payload.audio is None:
            return _stub_result("no audio in InputBundle")

        # Resolve the audio URI to a local path. For file:// URIs we can read
        # directly; otherwise we'd need to stage to a temp file via the blob
        # store. The blob store import is local to keep the stub path light.
        try:
            audio_path = _audio_path_from_uri(payload.audio.uri)
        except ValueError as exc:
            # Non-file URI: stage via the blob store into a temp file. The
            # blob store is imported lazily here because backend.api.deps
            # imports this module — pulling it in at module load would create
            # a cycle.
            from backend.api.deps import get_blob_store  # noqa: PLC0415
            try:
                blob = get_blob_store()
                data = blob.get_bytes(payload.audio.uri)
            except Exception as fetch_exc:  # noqa: BLE001
                log.warning("Could not fetch audio for MT3: %s", fetch_exc)
                return _stub_result(f"audio fetch failed: {fetch_exc}")
            suffix = f".{payload.audio.format}"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp.write(data)
            tmp.close()
            audio_path = Path(tmp.name)
            log.debug("Staged %s → %s for MT3 (%s)", payload.audio.uri, audio_path, exc)

        if not audio_path.is_file():
            return _stub_result(f"audio file missing: {audio_path}")

        try:
            return await asyncio.to_thread(_run_mt3_sync, audio_path)
        except FileNotFoundError as exc:
            log.warning("MT3 checkpoint unavailable, using stub: %s", exc)
            return _stub_result(str(exc))
        except ImportError as exc:
            log.warning("MT3 deps unavailable (%s) — using stub", exc)
            return _stub_result(f"missing dependency: {exc}")
        except Exception as exc:  # noqa: BLE001 — boundary; we don't want one bad audio file to crash the worker
            log.exception("MT3 inference failed for %s", audio_path)
            return _stub_result(f"inference failed: {exc}")
