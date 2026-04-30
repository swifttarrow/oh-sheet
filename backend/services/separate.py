"""Source-separation pipeline stage — runs HTDemucs and persists stems.

Phase 5 promotes Demucs from an inline call inside ``transcribe.py``
to a first-class pipeline stage between ingest and transcribe. The
service:

  1. Resolves the ``InputBundle.audio.uri`` to a local audio file
     (staging from blob storage when the URI doesn't point at the
     local filesystem already).
  2. Runs :func:`backend.services.stem_separation.separate_stems`
     with the configured htdemucs settings.
  3. Persists each stem WAV to blob storage at a **content-addressed
     cache key** (``cache/separate/{sha256}/{model}/{stem}.wav``) so a
     re-run with the same input audio + model bypasses the expensive
     Demucs inference and just returns the already-cached URIs.
  4. Returns an updated ``InputBundle`` whose ``audio_stems`` field
     maps each stem name to its blob URI.

The transcribe stage (``backend/services/transcribe.py``) consumes
``audio_stems`` directly when populated, fetching the WAVs into a
local tempdir and dispatching to the existing per-stem pipeline.

Failure modes — missing Demucs deps, model load crash, apply OOM,
silent-input audio — return the bundle **unchanged** with an empty
``audio_stems`` dict, so the pipeline degrades gracefully into
transcribe's legacy inline path. Operators who require the separator
to run can monitor for the warning logs and decide whether to fail
the job at the orchestrator level.
"""
from __future__ import annotations

import hashlib
import logging
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from backend.config import settings
from backend.contracts import InputBundle
from backend.services.stem_separation import (
    SeparatedStems,
    separate_stems,
)
from backend.storage.base import BlobStore

log = logging.getLogger(__name__)


def _audio_bytes_for_bundle(
    bundle: InputBundle, *, blob_store: BlobStore | None,
) -> tuple[bytes, Path | None]:
    """Resolve the bundle audio to raw bytes plus a local Path when available.

    Returns ``(audio_bytes, local_path)``. When the audio URI is a local
    ``file://`` URI we return its path directly so the caller can hand it
    straight to Demucs. Otherwise we stage via the blob store and return
    ``(bytes, None)`` — the caller is responsible for writing to a temp
    file before invoking Demucs.
    """
    if bundle.audio is None:
        raise ValueError("InputBundle has no audio to separate")

    parsed = urlparse(bundle.audio.uri)
    if parsed.scheme == "file":
        path = Path(parsed.path)
        if not path.is_file():
            raise FileNotFoundError(f"audio file missing: {path}")
        return path.read_bytes(), path

    if blob_store is None:
        raise ValueError(
            f"non-file audio URI {bundle.audio.uri!r} requires a BlobStore"
        )
    return blob_store.get_bytes(bundle.audio.uri), None


def _cache_key(audio_bytes: bytes, model_name: str) -> str:
    """Deterministic content-addressed cache prefix for a (bytes, model) pair.

    ``model_name`` is folded in so swapping bags (htdemucs → htdemucs_ft,
    or eventually a commercially-licensed weight set) lands in a separate
    cache namespace. The 16-char hex prefix is short enough to keep blob
    paths readable while still giving ~64 bits of collision resistance.
    """
    digest = hashlib.sha256(audio_bytes).hexdigest()
    return f"cache/separate/{digest}/{model_name}"


def _write_stem(
    blob_store: BlobStore, stem_path: Path, blob_key: str,
) -> str:
    """Persist a stem WAV to blob storage and return its URI."""
    return blob_store.put_bytes(blob_key, stem_path.read_bytes())


def _stem_paths(stems: SeparatedStems) -> dict[str, Path]:
    """Map stem names to file paths, dropping unset slots."""
    return {
        name: getattr(stems, name)
        for name in ("vocals", "drums", "bass", "other")
        if getattr(stems, name) is not None
    }


class SeparateService:
    """Synchronous wrapper around :func:`separate_stems` + blob persistence."""

    name = "separate"

    def __init__(self, blob_store: BlobStore | None = None) -> None:
        self.blob_store = blob_store

    def run(self, bundle: InputBundle) -> InputBundle:
        """Run Demucs on ``bundle.audio`` and return an enriched bundle.

        On any failure path (missing audio, missing Demucs deps, all
        stems empty, blob persistence error) the original bundle is
        returned unchanged with ``audio_stems = {}`` — callers detect
        the empty dict and route to transcribe's legacy inline path.
        """
        if not settings.demucs_enabled:
            log.info("separate: OHSHEET_DEMUCS_ENABLED=0, returning bundle unchanged")
            return bundle

        if bundle.audio is None:
            log.info("separate: no audio in bundle, skipping")
            return bundle

        if self.blob_store is None:
            log.warning(
                "separate: blob_store not configured, returning bundle unchanged"
            )
            return bundle

        try:
            audio_bytes, local_path = _audio_bytes_for_bundle(
                bundle, blob_store=self.blob_store,
            )
        except (ValueError, FileNotFoundError) as exc:
            log.warning("separate: audio resolution failed (%s)", exc)
            return bundle

        cache_prefix = _cache_key(audio_bytes, settings.demucs_model)

        # Cache hit fast-path: if every expected stem already lives in
        # the blob store under the content-addressed key, return their
        # URIs without ever loading Demucs. This is the difference
        # between a cold-start re-run (~30 s of inference) and a
        # warm-start re-run (~10 ms of blob existence checks).
        cached_uris = self._lookup_cache(cache_prefix)
        if cached_uris:
            log.info(
                "separate: cache hit for prefix=%s stems=%s",
                cache_prefix, list(cached_uris),
            )
            return bundle.model_copy(update={"audio_stems": cached_uris})

        # Stage non-file URIs to a tempfile so Demucs / ffmpeg can read.
        staged_tmp: Path | None = None
        if local_path is None:
            suffix = f".{bundle.audio.format}"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp.write(audio_bytes)
            tmp.close()
            local_path = Path(tmp.name)
            staged_tmp = local_path

        try:
            stems, stats = separate_stems(
                local_path,
                model_name=settings.demucs_model,
                device=settings.demucs_device,
                segment_sec=settings.demucs_segment_sec,
                shifts=settings.demucs_shifts,
                overlap=settings.demucs_overlap,
                split=settings.demucs_split,
            )
        finally:
            if staged_tmp is not None:
                try:
                    staged_tmp.unlink(missing_ok=True)
                except OSError:
                    log.debug("failed to clean up staged audio %s", staged_tmp)

        if stems is None or stats.skipped:
            warnings = "; ".join(stats.warnings) or "skipped"
            log.warning("separate: stems unavailable (%s)", warnings)
            return bundle

        # Persist + cleanup. Cleanup is unconditional in the finally
        # — Demucs writes ~80 MB per stem on a 3-min song, so we MUST
        # rmtree the tempdir even when blob persistence raises.
        try:
            stem_uris: dict[str, str] = {}
            for name, path in _stem_paths(stems).items():
                blob_key = f"{cache_prefix}/{name}.wav"
                stem_uris[name] = _write_stem(self.blob_store, path, blob_key)
        except Exception as exc:  # noqa: BLE001 — blob layer boundary
            log.warning(
                "separate: blob persistence failed (%s); returning bundle unchanged",
                exc,
            )
            stems.cleanup()
            return bundle
        finally:
            stems.cleanup()

        if not stem_uris:
            log.warning("separate: no stems written; returning bundle unchanged")
            return bundle

        log.info(
            "separate: wrote %d stems to %s in %.2fs",
            len(stem_uris), cache_prefix, stats.wall_time_sec,
        )
        return bundle.model_copy(update={"audio_stems": stem_uris})

    def _lookup_cache(self, cache_prefix: str) -> dict[str, str]:
        """Return cached stem URIs if every htdemucs source is present.

        The htdemucs bag emits four sources (vocals/drums/bass/other);
        we treat the cache as valid iff all four are written. A partial
        cache from a crashed run misses the gate and gets re-separated.
        """
        if self.blob_store is None:
            return {}
        cached: dict[str, str] = {}
        for name in ("vocals", "drums", "bass", "other"):
            blob_key = f"{cache_prefix}/{name}.wav"
            try:
                # ``put_bytes`` returns a URI shaped the same way for a
                # given key, so we can derive the URI without an explicit
                # exists() check on the public BlobStore protocol — a 0-byte
                # write would be visible at read time anyway. Use the local
                # ``exists`` helper when the store offers one.
                exists = getattr(self.blob_store, "exists", None)
                if exists is not None:
                    # Probe via the LocalBlobStore-style key→path mapping.
                    candidate_path = getattr(self.blob_store, "_path_for_key", None)
                    if candidate_path is None:
                        return {}
                    uri = candidate_path(blob_key).as_uri()
                    if not exists(uri):
                        return {}
                    cached[name] = uri
                else:
                    return {}
            except Exception:  # noqa: BLE001 — any probe failure invalidates the cache
                return {}
        return cached
