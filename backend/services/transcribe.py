"""Transcription stage — multi-backend dispatch with cover/faithful split.

Dispatch priority (see :func:`_run_basic_pitch_sync`):

1. **AMT-APC cover mode** — when the user opted into cover output
   (``variant == "pop_cover"`` or ``user_hint == "cover"``); MIT-licensed
   pianistic-cover transcriber from misya11p/amt-apc.
2. **Kong piano-stem AMT** — when pre-separated stems are present and
   the input is piano-dominant (Phase 6); emits sustain pedal events.
3. **Pre-separated stems → Basic Pitch** — Phase 5 fast-path with
   Demucs stems already on disk.
4. **Pop2Piano** — deprecated, retained for backwards compatibility.
   See ``POP2PIANO_DEPRECATED.md`` — AMT-APC supersedes it.
5. **Inline Demucs + Basic Pitch** — split source into 4 stems and
   run Basic Pitch per stem.
6. **Single-mix Basic Pitch** — final fallback, one BP pass on the mix.

Each lower-priority path is a fallback for any failure in the higher
priority paths above it, so flipping individual feature flags is always
safe.

Basic Pitch backend selection is left to basic-pitch's auto-pick order
via ``ICASSP_2022_MODEL_PATH``: on Darwin this resolves to the CoreML
model (fastest on Apple Silicon), on Linux CI it falls through to
ONNX/TFLite.

If no transcriber is available, the service raises
:class:`TranscriptionFailure` so the failure surfaces to the user
instead of shipping a fake melody.

.. _basic-pitch: https://github.com/spotify/basic-pitch

Implementation is split across focused sub-modules:

* :mod:`transcribe_audio` — audio I/O and analysis helpers
* :mod:`transcribe_inference` — Basic Pitch model + single-pass wrapper
* :mod:`transcribe_midi` — MIDI artifact construction for blob storage
* :mod:`transcribe_result` — ``TranscriptionResult`` assembly + stub
* :mod:`transcribe_pipeline_amt_apc` — AMT-APC cover-mode pipeline (Phase 8)
* :mod:`transcribe_pipeline_kong` — Kong piano-stem AMT (Phase 6)
* :mod:`transcribe_pipeline_pop2piano` — Pop2Piano pipeline (deprecated)
* :mod:`transcribe_pipeline_single` — single-mix pipeline (no Demucs)
* :mod:`transcribe_pipeline_stems` — Demucs-driven per-stem pipeline
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from backend.config import settings
from backend.contracts import InputBundle, RealtimePedalEvent, TranscriptionResult
from backend.services.stem_separation import (
    SeparatedStems,
    StemSeparationStats,
    separate_stems,
)
from backend.services.transcribe_audio import _audio_path_from_uri
from backend.services.transcribe_inference import _BasicPitchPass  # noqa: F401 — re-export for tests
from backend.services.transcribe_midi import _rebuild_blob_midi  # noqa: F401 — re-export for tests
from backend.services.transcribe_pipeline_amt_apc import (  # noqa: F401 — re-export for tests
    _run_with_amt_apc,
    should_route_to_amt_apc,
)
from backend.services.transcribe_pipeline_kong import (  # noqa: F401 — re-export for tests
    _run_with_kong,
    should_route_to_kong,
)
from backend.services.transcribe_pipeline_pop2piano import _run_with_pop2piano  # noqa: F401 — re-export for tests
from backend.services.transcribe_pipeline_single import _run_without_stems  # noqa: F401 — re-export for tests
from backend.services.transcribe_pipeline_stems import _run_with_stems  # noqa: F401 — re-export for tests
from backend.services.transcribe_result import (  # noqa: F401 — re-export for tests
    TranscriptionFailure,
    _stub_result,
)
from backend.storage.base import BlobStore

log = logging.getLogger(__name__)

# Mirrors the ``PipelineVariant`` Literal in shared.contracts. Kept as a
# plain set here so a contract addition (new variant) only needs to update
# the Literal — the dispatcher then logs a warning when an unrecognised
# value flows in via the wire-format ``variant_hint`` field.
_KNOWN_VARIANTS: frozenset[str] = frozenset({
    "full",
    "audio_upload",
    "midi_upload",
    "sheet_only",
    "pop_cover",
})


def _run_basic_pitch_sync(
    audio_path: Path,
    *,
    pre_separated: SeparatedStems | None = None,
    user_hint: str | None = None,
    variant: str | None = None,
) -> tuple[TranscriptionResult, bytes | None, list[RealtimePedalEvent]]:
    """Synchronous transcription inference. Run inside asyncio.to_thread.

    Returns the parsed ``TranscriptionResult``, the raw MIDI bytes (if
    serialization succeeded), and the list of seconds-domain pedal
    events emitted by the active transcriber (empty for transcribers
    that don't model pedal). The async caller persists the MIDI to
    blob storage and merges the pedal events onto the result.

    Dispatches between five pipelines in priority order:

      * **AMT-APC cover path** (``settings.amt_apc_enabled``, Phase 8):
        when the user opted into cover mode (``variant == "pop_cover"``
        OR ``user_hint == "cover"``), run AMT-APC for a pianistic cover
        with idiomatic accompaniment. Always tries this first when
        cover mode is requested; falls through to the faithful paths
        on failure so the user still gets a transcription.
      * **Kong path** (``settings.kong_enabled``, Phase 6): when
        pre-separated stems are available and the routing heuristic
        picks Kong (vocal energy < threshold OR user_hint == "piano"),
        run Kong's piano AMT on the summed ``bass + other`` stem.
        Emits sustain-pedal events that flow through to the engraver.
      * **Pre-separated stems path** (Phase 5 fast-path): when stems
        exist but Kong didn't claim them, dispatch to the legacy
        Basic Pitch stems pipeline via :func:`_run_with_stems`.
      * **Pop2Piano path** (``settings.pop2piano_enabled``, deprecated):
        run the ``sweetcocoa/pop2piano`` transformer on the full audio.
        Quarantined behind a default-off flag — AMT-APC supersedes
        this path for cover-style transcription (license-clean).
      * **Demucs path** (inline): split the source into 4 stems and
        run Basic Pitch once per stem when ``settings.demucs_enabled``.
      * **Single-mix path** (fallback): one Basic Pitch pass on the
        whole mix + Viterbi melody/bass split + chord recognition on
        the original waveform.

    Any failure in a higher-priority path transparently falls back to
    the next one, so flipping ``amt_apc_enabled`` / ``kong_enabled`` /
    ``pop2piano_enabled`` / ``demucs_enabled`` is always safe.
    """
    # Import the pipeline functions via their modules so monkeypatching
    # in tests works correctly (patches the module attribute, not a
    # local binding).
    from backend.services import transcribe_pipeline_amt_apc as _amt_mod  # noqa: PLC0415
    from backend.services import transcribe_pipeline_kong as _kong_mod  # noqa: PLC0415
    from backend.services import transcribe_pipeline_pop2piano as _p2p_mod  # noqa: PLC0415
    from backend.services import transcribe_pipeline_single as _single_mod  # noqa: PLC0415
    from backend.services import transcribe_pipeline_stems as _stems_mod  # noqa: PLC0415

    # --- AMT-APC cover mode (Phase 8) ---
    # Runs first when the user explicitly asked for cover output. Uses
    # the Demucs instrumental stem when available, falling back to the
    # mix. Failure falls through to the faithful paths so a cover-mode
    # job still produces *something* if AMT-APC is unavailable.
    if _amt_mod.should_route_to_amt_apc(user_hint=user_hint, variant=variant):
        amt_stem_stats: StemSeparationStats | None = None
        if pre_separated is not None:
            amt_stem_stats = StemSeparationStats(
                model_name=settings.demucs_model,
                device="pre_separated",
                stems_written=[
                    name for name in ("vocals", "drums", "bass", "other")
                    if getattr(pre_separated, name) is not None
                ],
            )
        try:
            result, midi_bytes = _amt_mod._run_with_amt_apc(
                audio_path, pre_separated, amt_stem_stats,
            )
            return result, midi_bytes, []
        except ImportError as exc:
            log.warning(
                "AMT-APC deps unavailable (%s) — falling back to faithful path",
                exc,
            )
        except Exception as exc:  # noqa: BLE001 — never let AMT-APC sink the job
            log.warning(
                "AMT-APC inference failed (%s) — falling back to faithful path",
                exc,
            )

    # --- Pre-separated stems (Phase 5 fast-path) ---
    # If the separate worker already ran, every stem WAV is staged
    # locally and we skip Pop2Piano (which expects the full mix) plus
    # the inline Demucs invocation. We still own ``pre_separated``
    # cleanup — the caller wraps us in a try/finally that calls
    # ``cleanup()`` on the SeparatedStems instance regardless of
    # which branch ran here.
    if pre_separated is not None:
        pre_stats = StemSeparationStats(
            model_name=settings.demucs_model,
            device="pre_separated",
            stems_written=[
                name for name in ("vocals", "drums", "bass", "other")
                if getattr(pre_separated, name) is not None
            ],
        )
        # Kong (Phase 6): piano-stem AMT with sustain pedal events.
        # Gated on stems being available + vocal energy heuristic.
        # Any failure (missing dep, model load crash, empty notes)
        # falls through to the Basic Pitch stems pipeline below.
        if _kong_mod.should_route_to_kong(pre_separated, user_hint=user_hint):
            try:
                return _kong_mod._run_with_kong(
                    audio_path, pre_separated, pre_stats,
                )
            except ImportError as exc:
                log.warning(
                    "Kong deps unavailable (%s) — falling back to BP stems",
                    exc,
                )
            except Exception as exc:  # noqa: BLE001 — Kong must not sink the job
                log.warning(
                    "Kong inference failed (%s) — falling back to BP stems",
                    exc,
                )
        result, midi_bytes = _stems_mod._run_with_stems(
            audio_path, pre_separated, pre_stats,
        )
        return result, midi_bytes, []

    # --- Pop2Piano path (deprecated, retained for backwards-compat) ---
    # See backend/services/POP2PIANO_DEPRECATED.md — AMT-APC supersedes
    # this for cover-mode transcription. The runtime warning fires once
    # per process invocation so operators notice in production logs.
    if settings.pop2piano_enabled:
        import warnings  # noqa: PLC0415
        warnings.warn(
            "OHSHEET_POP2PIANO_ENABLED=True invokes the deprecated Pop2Piano "
            "transcriber. See backend/services/POP2PIANO_DEPRECATED.md — "
            "use OHSHEET_AMT_APC_ENABLED=True with the pop_cover variant instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            result, midi_bytes = _p2p_mod._run_with_pop2piano(audio_path)
            return result, midi_bytes, []
        except ImportError as exc:
            log.warning(
                "Pop2Piano deps unavailable (%s) — falling back to Demucs+BP",
                exc,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Pop2Piano inference failed (%s) — falling back to Demucs+BP",
                exc,
            )

    # --- Demucs + Basic Pitch path (legacy inline) ---
    stems: SeparatedStems | None = None
    stem_stats: StemSeparationStats | None = None

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
            result, midi_bytes = _stems_mod._run_with_stems(
                audio_path, stems, stem_stats,
            )
            return result, midi_bytes, []
        result, midi_bytes = _single_mod._run_without_stems(
            audio_path, stem_stats,
        )
        return result, midi_bytes, []
    finally:
        if stems is not None:
            stems.cleanup()


def _stage_pre_separated_stems(
    audio_stems: dict[str, str],
    blob_store: BlobStore | None,
) -> SeparatedStems | None:
    """Materialize blob-stored stems to a tempdir for Demucs-style consumption.

    Builds a ``SeparatedStems`` carrier so the existing per-stem pipeline
    in :func:`_run_with_stems` can read each WAV directly. The caller MUST
    invoke ``cleanup()`` on the returned instance (typically in a
    ``finally``) to remove the tempdir.

    Returns ``None`` when the stems can't be staged — missing blob store,
    unsupported URI scheme, or read failures — so the caller falls back
    to the inline path. Partial stems (some present, some missing) still
    return the carrier with the missing slots set to ``None``: the
    per-stem pipeline already tolerates absent slots.
    """
    if not audio_stems:
        return None
    tempdir = Path(tempfile.mkdtemp(prefix="ohsheet-stems-"))
    stems = SeparatedStems(_tempdir=tempdir)
    found_any = False
    for name in ("vocals", "drums", "bass", "other"):
        uri = audio_stems.get(name)
        if not uri:
            continue
        parsed = urlparse(uri)
        try:
            if parsed.scheme == "file":
                src = Path(parsed.path)
                if not src.is_file():
                    log.warning("pre-separated stem missing on disk: %s", src)
                    continue
                dst = tempdir / f"{name}.wav"
                shutil.copyfile(src, dst)
            else:
                if blob_store is None:
                    log.warning(
                        "non-file stem URI %r requires a BlobStore; skipping",
                        uri,
                    )
                    continue
                data = blob_store.get_bytes(uri)
                dst = tempdir / f"{name}.wav"
                dst.write_bytes(data)
        except Exception as exc:  # noqa: BLE001 — staging boundary
            log.warning("failed to stage stem %s from %s: %s", name, uri, exc)
            continue
        setattr(stems, name, dst)
        found_any = True

    if not found_any:
        stems.cleanup()
        return None
    return stems


def _apply_score_hpt(result: TranscriptionResult) -> TranscriptionResult:
    """Run Phase 9A velocity refinement on a transcription result.

    Pulled out of ``TranscribeService.run`` so the call site stays a
    single line and the helper is independently testable. Any failure
    falls through with the original result + a warning so the job
    never sinks because of a velocity-refinement bug.
    """
    try:
        from backend.services.score_hpt import (  # noqa: PLC0415
            ScoreHPTConfig,
            refine_velocities,
        )
    except ImportError as exc:  # pragma: no cover — module is stdlib-only
        log.warning("score_hpt unavailable (%s); skipping refinement", exc)
        return result

    cfg = ScoreHPTConfig(
        blend_alpha=settings.score_hpt_blend_alpha,
        downbeat_boost=settings.score_hpt_downbeat_boost,
        beat_boost=settings.score_hpt_beat_boost,
        offbeat_attenuation=settings.score_hpt_offbeat_attenuation,
        register_curve_strength=settings.score_hpt_register_curve_strength,
        density_compensation=settings.score_hpt_density_compensation,
        min_velocity=settings.score_hpt_min_velocity,
        max_velocity=settings.score_hpt_max_velocity,
    )
    try:
        refined_tracks, stats = refine_velocities(
            result.midi_tracks,
            result.analysis.tempo_map,
            downbeats_sec=result.analysis.downbeats,
            config=cfg,
        )
    except Exception as exc:  # noqa: BLE001 — never sink the job
        log.warning("score_hpt: velocity refinement failed (%s)", exc)
        return result

    log.info(
        "score_hpt: refined %d/%d velocities (mean_abs_delta=%.1f, "
        "max_abs_delta=%d, skipped=%s)",
        stats.n_changed,
        stats.n_notes,
        stats.mean_abs_delta,
        stats.max_abs_delta,
        stats.skipped,
    )
    new_quality = result.quality.model_copy(
        update={"warnings": result.quality.warnings + stats.as_warnings()}
    )
    return result.model_copy(
        update={"midi_tracks": refined_tracks, "quality": new_quality},
    )


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
        variant: str | None = None,
    ) -> TranscriptionResult:
        log.info(
            "transcribe: start job_id=%s variant=%s audio_uri=%s pre_separated_stems=%d",
            job_id or "—",
            variant or "—",
            payload.audio.uri if payload.audio else None,
            len(payload.audio_stems),
        )
        # Surface API-layer typos: ``variant_hint`` is wire-string-typed
        # (str | None) so a misspelt value silently falls through to the
        # default (faithful) routing path. Log loudly so the bad value is
        # discoverable in production logs without changing behaviour.
        if variant is not None and variant not in _KNOWN_VARIANTS:
            log.warning(
                "transcribe: unknown variant_hint=%r — falling through to default routing",
                variant,
            )
        if payload.audio is None:
            raise TranscriptionFailure("no audio in InputBundle")

        # Resolve the audio URI to a local path. For file:// URIs we can read
        # directly; otherwise we'd need to stage to a temp file via the blob
        # store. The blob store import is local to keep the stub path light.
        staged_tmp: Path | None = None
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
                log.warning("Could not fetch audio for transcription: %s", fetch_exc)
                raise TranscriptionFailure(
                    f"audio fetch failed: {fetch_exc}"
                ) from fetch_exc
            suffix = f".{payload.audio.format}"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp.write(data)
            tmp.close()
            audio_path = Path(tmp.name)
            staged_tmp = audio_path
            log.debug(
                "Staged %s → %s for transcription (%s)",
                payload.audio.uri, audio_path, exc,
            )

        # When the separate worker already produced stems, materialize
        # them locally so :func:`_run_with_stems` can read each WAV
        # directly. The local tempdir is owned by ``pre_separated`` and
        # cleaned up in the outer finally, regardless of whether the
        # stems-path or fallback branch ran.
        pre_separated = _stage_pre_separated_stems(
            payload.audio_stems, self.blob_store,
        )

        try:
            if not audio_path.is_file():
                raise TranscriptionFailure(f"audio file missing: {audio_path}")

            result, midi_bytes, pedal_events = await asyncio.to_thread(
                _run_basic_pitch_sync, audio_path,
                pre_separated=pre_separated,
                user_hint=None,
                variant=variant,
            )
        except TranscriptionFailure:
            raise
        except ImportError as exc:
            log.warning("Transcription deps unavailable: %s", exc)
            raise TranscriptionFailure(f"missing dependency: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 — convert all errors to a uniform failure
            log.exception("Transcription inference failed for %s", audio_path)
            raise TranscriptionFailure(f"inference failed: {exc}") from exc
        finally:
            if pre_separated is not None:
                pre_separated.cleanup()
            if staged_tmp is not None:
                try:
                    staged_tmp.unlink(missing_ok=True)
                except OSError:
                    log.debug("failed to clean up staged temp file %s", staged_tmp)

        # Persist the raw transcription MIDI to blob storage so it's
        # retrievable alongside the engraved output. Best-effort: a storage
        # failure shouldn't sink the job, since the downstream pipeline only
        # needs the parsed notes in ``result``.
        if midi_bytes and self.blob_store is not None and job_id is not None:
            try:
                uri = self.blob_store.put_bytes(
                    f"jobs/{job_id}/transcription/transcription.mid",
                    midi_bytes,
                )
                result = result.model_copy(update={"transcription_midi_uri": uri})
            except Exception as exc:  # noqa: BLE001 — best-effort persistence
                log.warning("Failed to persist transcription MIDI for %s: %s", job_id, exc)

        # Carry pre-separated stem URIs through onto the
        # TranscriptionResult so artifacts/UI surfaces and downstream
        # stages can see what the transcription was based on. This is
        # the contract field added in Phase 5 (contracts §TranscriptionResult).
        if payload.audio_stems:
            result = result.model_copy(update={"audio_stems": dict(payload.audio_stems)})

        # Phase 6: surface Kong's sustain-pedal events on the result so
        # arrange can convert them to beat-domain ExpressionMap events.
        # Empty list when the active transcriber doesn't model pedal —
        # downstream stages already default to the existing humanize
        # heuristic in that case.
        if pedal_events:
            result = result.model_copy(update={"pedal_events": list(pedal_events)})

        # Phase 9A: Score-HPT-style velocity refinement. Re-estimates
        # per-note velocities from metric position / register / onset
        # density before arrange's percentile-band remap runs. Stand-in
        # for Foscarin et al.'s ~1M-param HPT model — flagged off by
        # default; flip ``OHSHEET_SCORE_HPT_ENABLED=1`` to opt in.
        if settings.score_hpt_enabled:
            result = _apply_score_hpt(result)

        n_notes = sum(len(t.notes) for t in result.midi_tracks)
        log.info(
            "transcribe: done job_id=%s tracks=%d notes=%d transcription_midi=%s "
            "audio_stems=%d pedal_events=%d",
            job_id or "—",
            len(result.midi_tracks),
            n_notes,
            "yes" if result.transcription_midi_uri else "no",
            len(result.audio_stems),
            len(result.pedal_events),
        )
        return result
