"""Transcription stage — Basic Pitch baseline.

Wraps Spotify's `basic-pitch`_ polyphonic pitch-tracker into the async
pipeline. Basic Pitch is a lightweight CNN that consumes arbitrary mixed
audio and emits polyphonic note events with per-note amplitudes. It
produces a single un-instrumented pitch stream — we collapse the whole
prediction into one ``PIANO`` track, which is the right shape for a
piano-reduction pipeline anyway.

Backend selection is left to basic-pitch's auto-pick order via
``ICASSP_2022_MODEL_PATH``: on Darwin this resolves to the CoreML
model (fastest on Apple Silicon), on Linux CI it falls through to
ONNX/TFLite. The model is cached at module scope so the runtime only
loads once per process.

If basic-pitch isn't installed, or inference fails on a specific audio
file, the service falls back to a tiny stub ``TranscriptionResult`` so
the rest of the pipeline can still be exercised end-to-end.

.. _basic-pitch: https://github.com/spotify/basic-pitch

Implementation is split across focused sub-modules:

* :mod:`transcribe_audio` — audio I/O and analysis helpers
* :mod:`transcribe_inference` — Basic Pitch model + single-pass wrapper
* :mod:`transcribe_midi` — MIDI artifact construction for blob storage
* :mod:`transcribe_result` — ``TranscriptionResult`` assembly + stub
* :mod:`transcribe_pipeline_single` — single-mix pipeline (no Demucs)
* :mod:`transcribe_pipeline_stems` — Demucs-driven per-stem pipeline
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

from backend.config import settings
from backend.contracts import InputBundle, TranscriptionResult
from backend.services.stem_separation import separate_stems
from backend.services.transcribe_audio import _audio_path_from_uri
from backend.services.transcribe_inference import _BasicPitchPass  # noqa: F401 — re-export for tests
from backend.services.transcribe_midi import _rebuild_blob_midi  # noqa: F401 — re-export for tests
from backend.services.transcribe_pipeline_single import _run_without_stems  # noqa: F401 — re-export for tests
from backend.services.transcribe_pipeline_stems import _run_with_stems  # noqa: F401 — re-export for tests
from backend.services.transcribe_result import _stub_result  # noqa: F401 — re-export for tests
from backend.storage.base import BlobStore

log = logging.getLogger(__name__)


def _run_basic_pitch_sync(audio_path: Path) -> tuple[TranscriptionResult, bytes | None]:
    """Synchronous Basic Pitch inference. Run inside asyncio.to_thread.

    Returns both the parsed ``TranscriptionResult`` and the raw MIDI bytes
    (if serialization succeeded) so the async caller can persist the MIDI
    to blob storage without blocking on disk I/O in the worker thread.

    Dispatches between two pipelines:

      * **Demucs path** (``settings.demucs_enabled``): split the source
        into 4 stems, run Basic Pitch once per stem (vocals / bass /
        other), and route drums + other to the beat-track and chord
        recognizers. See :func:`_run_with_stems`.
      * **Single-mix path** (default): one Basic Pitch pass on the
        whole mix + Viterbi melody/bass split + chord recognition on
        the original waveform. See :func:`_run_without_stems`.

    Any failure in the Demucs path (separation crash, all-stems-empty,
    missing torch) transparently falls back to the single-mix path so
    flipping ``demucs_enabled`` is always safe — the worst case is that
    the user pays the Demucs load cost for nothing.
    """
    # Import the pipeline functions via their modules so monkeypatching
    # in tests works correctly (patches the module attribute, not a
    # local binding).
    from backend.services import transcribe_pipeline_single as _single_mod  # noqa: PLC0415
    from backend.services import transcribe_pipeline_stems as _stems_mod  # noqa: PLC0415

    stems = None
    stem_stats = None

    if settings.demucs_enabled:
        stems, stem_stats = separate_stems(
            audio_path,
            model_name=settings.demucs_model,
            device=settings.demucs_device,
            segment_sec=settings.demucs_segment_sec,
            shifts=settings.demucs_shifts,
            overlap=settings.demucs_overlap,
            split=settings.demucs_split,
        )

    try:
        if stems is not None and stem_stats is not None:
            return _stems_mod._run_with_stems(audio_path, stems, stem_stats)
        return _single_mod._run_without_stems(audio_path, stem_stats)
    finally:
        if stems is not None:
            stems.cleanup()


class TranscribeService:
    name = "transcribe"

    def __init__(self, blob_store: BlobStore | None = None) -> None:
        # Optional so the service can still be constructed in bare unit tests
        # that don't exercise the persistence path. In production (via
        # backend.api.deps.get_runner) it's always injected.
        self.blob_store = blob_store

    async def run(
        self,
        payload: InputBundle,
        *,
        job_id: str | None = None,
    ) -> TranscriptionResult:
        log.info(
            "transcribe: start job_id=%s audio_uri=%s",
            job_id or "—",
            payload.audio.uri if payload.audio else None,
        )
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
                log.warning("Could not fetch audio for Basic Pitch: %s", fetch_exc)
                return _stub_result(f"audio fetch failed: {fetch_exc}")
            suffix = f".{payload.audio.format}"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp.write(data)
            tmp.close()
            audio_path = Path(tmp.name)
            log.debug(
                "Staged %s → %s for Basic Pitch (%s)",
                payload.audio.uri, audio_path, exc,
            )

        if not audio_path.is_file():
            return _stub_result(f"audio file missing: {audio_path}")

        try:
            result, midi_bytes = await asyncio.to_thread(
                _run_basic_pitch_sync, audio_path,
            )
        except ImportError as exc:
            log.warning("Basic Pitch deps unavailable (%s) — using stub", exc)
            return _stub_result(f"missing dependency: {exc}")
        except Exception as exc:  # noqa: BLE001 — boundary; we don't want one bad audio file to crash the worker
            log.exception("Basic Pitch inference failed for %s", audio_path)
            return _stub_result(f"inference failed: {exc}")

        # Persist the raw transcription MIDI to blob storage so it's
        # retrievable alongside the engraved output. Best-effort: a storage
        # failure shouldn't sink the job, since the downstream pipeline only
        # needs the parsed notes in ``result``.
        if midi_bytes and self.blob_store is not None and job_id is not None:
            try:
                uri = self.blob_store.put_bytes(
                    f"jobs/{job_id}/transcription/basic-pitch.mid",
                    midi_bytes,
                )
                result = result.model_copy(update={"transcription_midi_uri": uri})
            except Exception as exc:  # noqa: BLE001 — best-effort persistence
                log.warning("Failed to persist transcription MIDI for %s: %s", job_id, exc)

        n_notes = sum(len(t.notes) for t in result.midi_tracks)
        log.info(
            "transcribe: done job_id=%s tracks=%d notes=%d transcription_midi=%s",
            job_id or "—",
            len(result.midi_tracks),
            n_notes,
            "yes" if result.transcription_midi_uri else "no",
        )
        return result
