"""PipelineRunner — dispatches pipeline stages as Celery tasks.

The runner owns the execution plan (which stages run in what order)
and uses the claim-check pattern: serialize each stage's input to
blob storage, dispatch a Celery task with the payload URI, wait for
the result URI, and deserialize the output for the next stage.
"""
from __future__ import annotations

import asyncio
import io
import logging
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

from celery import Celery

from backend.config import settings
from backend.contracts import (
    SCHEMA_VERSION,
    EngravedOutput,
    EngravedScoreData,
    HarmonicAnalysis,
    InputBundle,
    InstrumentRole,
    MidiTrack,
    Note,
    PipelineConfig,
    QualitySignal,
    TempoMapEntry,
    TranscriptionResult,
)
from backend.jobs.events import JobEvent
from backend.services.pretty_midi_tracks import (
    harmonic_analysis_from_pretty_midi,
    midi_tracks_from_pretty_midi,
)
from backend.storage.base import BlobStore

log = logging.getLogger(__name__)

EventCallback = Callable[[JobEvent], None]

# Maps execution plan step names to Celery task names.
STEP_TO_TASK: dict[str, str] = {
    "ingest": "ingest.run",
    "transcribe": "transcribe.run",
    "arrange": "arrange.run",
    "condense": "condense.run",
    "transform": "transform.run",
    "humanize": "humanize.run",
    "refine": "refine.run",
    "engrave": "engrave.run",
}


def _stub_transcription(reason: str) -> TranscriptionResult:
    return TranscriptionResult(
        schema_version=SCHEMA_VERSION,
        midi_tracks=[
            MidiTrack(
                notes=[
                    Note(pitch=60, onset_sec=0.0, offset_sec=0.5, velocity=80),
                    Note(pitch=62, onset_sec=0.5, offset_sec=1.0, velocity=80),
                ],
                instrument=InstrumentRole.PIANO,
                program=0,
                confidence=0.9,
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
            overall_confidence=0.5,
            warnings=[f"midi-to-transcription stub: {reason}"],
        ),
    )


def _bundle_to_transcription(
    bundle: InputBundle,
    *,
    blob_store: BlobStore | None = None,
    job_id: str | None = None,
) -> TranscriptionResult:
    """Build a TranscriptionResult from a midi_upload bundle.

    Real path: parse the MIDI file via pretty_midi, recover the tempo map,
    fold each instrument into a MidiTrack, infer key/time-signature.
    When ``blob_store`` and ``job_id`` are provided, the upload bytes are
    copied to ``jobs/{job_id}/transcription/upload.mid`` and
    ``transcription_midi_uri`` is set (claim-check for arrange / HF).
    Fallback: a small shape-correct stub so downstream stages still run.
    """
    if bundle.midi is None:
        return _stub_transcription("no midi in bundle")

    parsed = urlparse(bundle.midi.uri)
    if parsed.scheme != "file":
        return _stub_transcription(f"unsupported midi URI scheme: {parsed.scheme!r}")
    midi_path = Path(parsed.path)
    if not midi_path.is_file():
        return _stub_transcription(f"midi file missing: {midi_path}")

    try:
        import pretty_midi  # noqa: PLC0415 — optional dep
    except ImportError:
        return _stub_transcription("pretty_midi not installed")

    midi_bytes = midi_path.read_bytes()
    try:
        pm = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    except Exception as exc:  # noqa: BLE001 — bad MIDI bytes shouldn't crash the worker
        return _stub_transcription(f"pretty_midi parse failed: {exc}")

    midi_tracks = midi_tracks_from_pretty_midi(pm)
    if not midi_tracks:
        return _stub_transcription("midi file contained no notes")

    analysis = harmonic_analysis_from_pretty_midi(pm)

    total_notes = sum(len(t.notes) for t in midi_tracks)
    log.info(
        "Parsed MIDI %s: %d tracks, %d notes",
        midi_path.name, len(midi_tracks), total_notes,
    )

    result = TranscriptionResult(
        schema_version=SCHEMA_VERSION,
        midi_tracks=midi_tracks,
        analysis=analysis,
        quality=QualitySignal(
            overall_confidence=0.95,
            warnings=["MIDI input — no harmonic analysis"],
        ),
    )
    if blob_store is not None and job_id is not None:
        try:
            upload_uri = blob_store.put_bytes(
                f"jobs/{job_id}/transcription/upload.mid",
                midi_bytes,
            )
            result = result.model_copy(update={"transcription_midi_uri": upload_uri})
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Could not persist upload MIDI for job_id=%s (HF/arrange claim-check): %s",
                job_id,
                exc,
            )
    return result


class PipelineRunner:
    def __init__(
        self,
        blob_store: BlobStore,
        celery_app: Celery,
    ) -> None:
        self.blob_store = blob_store
        self.celery_app = celery_app

    def _serialize_stage_input(
        self,
        job_id: str,
        step: str,
        payload: dict,
    ) -> str:
        """Write stage input to blob store, return URI."""
        return self.blob_store.put_json(
            f"jobs/{job_id}/{step}/input.json",
            payload,
        )

    async def _dispatch_task(
        self,
        task_name: str,
        job_id: str,
        payload_uri: str,
        timeout: int,
    ) -> str:
        """Send Celery task and wait for result URI without blocking event loop.

        Uses ``apply_async`` via the local task registry when the task is
        registered (monolith workers + test stubs), and falls back to
        ``send_task`` for tasks that only exist on remote workers.
        ``send_task`` does NOT honour ``task_always_eager``, so the test
        conftest registers decomposer/assembler stubs locally.

        Everything runs inside ``to_thread`` so eager-mode tasks (which call
        ``asyncio.run`` internally) execute without an event-loop conflict.
        """
        def _run() -> str:
            if task_name in self.celery_app.tasks:
                result = self.celery_app.tasks[task_name].apply_async(
                    args=[job_id, payload_uri],
                )
            else:
                if self.celery_app.conf.task_always_eager:
                    raise RuntimeError(
                        f"Task {task_name!r} is not registered on the Celery app "
                        f"but task_always_eager is True. Register a stub in "
                        f"conftest.py to avoid hitting a real broker in tests."
                    )
                result = self.celery_app.send_task(
                    task_name, args=[job_id, payload_uri],
                )
            return result.get(timeout=timeout)

        return await asyncio.to_thread(_run)

    async def run(
        self,
        *,
        job_id: str,
        bundle: InputBundle,
        config: PipelineConfig,
        on_event: EventCallback | None = None,
    ) -> EngravedOutput:
        plan = config.get_execution_plan()
        n = len(plan)

        def emit(stage: str, event_type, **kw) -> None:
            if on_event is None:
                return
            on_event(JobEvent(job_id=job_id, type=event_type, stage=stage, **kw))

        title = bundle.metadata.title or "Untitled"
        composer = bundle.metadata.artist or "Unknown"

        # Current state as we walk the pipeline -- always a dict for JSON serialization.
        current_payload: dict = bundle.model_dump(mode="json")
        txr_dict: dict | None = None
        score_dict: dict | None = None
        perf_dict: dict | None = None
        result_dict: dict | None = None

        log.info(
            "pipeline start job_id=%s variant=%s plan=%s skip_humanizer=%s source=%s "
            "has_audio=%s has_midi=%s",
            job_id,
            config.variant,
            plan,
            config.skip_humanizer,
            bundle.metadata.source,
            bundle.audio is not None,
            bundle.midi is not None,
        )

        for i, step in enumerate(plan):
            emit(step, "stage_started", progress=i / n)
            log.info(
                "pipeline stage begin job_id=%s stage=%s index=%d/%d",
                job_id, step, i + 1, n,
            )
            t0 = time.perf_counter()
            task_name = STEP_TO_TASK[step]

            try:
                if step == "ingest":
                    payload_uri = self._serialize_stage_input(job_id, step, current_payload)
                    output_uri = await self._dispatch_task(task_name, job_id, payload_uri, config.stage_timeout_sec)
                    current_payload = self.blob_store.get_json(output_uri)

                    # Update the display title/artist from the resolved
                    # ingest metadata. The user submitted a YouTube URL as
                    # the "title" — ingest probed the video and resolved
                    # the real song name + artist. We update title/composer
                    # so the result screen shows the song name, not the URL.
                    ingest_meta = current_payload.get("metadata", {})
                    resolved_title = ingest_meta.get("title")
                    resolved_artist = ingest_meta.get("artist")
                    # Keep the original URL for linking back to the source
                    original_url = title if title.startswith("http") else None
                    if resolved_title and resolved_title != title:
                        title = resolved_title
                    if resolved_artist and resolved_artist != composer:
                        composer = resolved_artist
                    # Update the bundle so the API returns the resolved
                    # title instead of the raw YouTube URL.
                    bundle = bundle.model_copy(update={
                        "metadata": bundle.metadata.model_copy(update={
                            "title": title,
                            "artist": composer if composer != "Unknown" else bundle.metadata.artist,
                            "source_url": original_url,
                        }),
                    })

                    # For title_lookup jobs (YouTube URL / song title search),
                    # delegate entirely to TuneChat when enabled. TuneChat
                    # uses tcalgo + MuseScore which produces much cleaner
                    # scores than Basic Pitch + music21. Skip the remaining
                    # Oh Sheet pipeline stages and return TuneChat's result.
                    if settings.tunechat_enabled:
                        audio_data = current_payload.get("audio")
                        is_title_job = bundle.metadata.source == "title_lookup"

                        if is_title_job and audio_data and audio_data.get("uri"):
                            # TuneChat-only path: send audio, await result,
                            # return minimal EngravedOutput with TuneChat fields.
                            try:
                                from backend.services.tunechat_client import transcribe_via_tunechat
                                audio_bytes = self.blob_store.get_bytes(audio_data["uri"])
                                emit("ingest", "stage_completed", progress=0.25)
                                emit("transcribe", "stage_started", progress=0.25)
                                log.info("tunechat-only: sending audio for job_id=%s", job_id)
                                tc_result = await transcribe_via_tunechat(
                                    audio_bytes, "audio.wav",
                                    title=title, artist=composer,
                                )
                                if tc_result is not None:
                                    emit("transcribe", "stage_completed", progress=0.75)
                                    emit("engrave", "stage_started", progress=0.75)
                                    emit("engrave", "stage_completed", progress=1.0)
                                    log.info(
                                        "tunechat-only: success job_id=%s tc_job_id=%s",
                                        job_id, tc_result.job_id,
                                    )
                                    return EngravedOutput(
                                        metadata=EngravedScoreData(
                                            includes_dynamics=False,
                                            includes_pedal_marks=False,
                                            includes_fingering=False,
                                            includes_chord_symbols=False,
                                            title=title,
                                            composer=composer,
                                        ),
                                        pdf_uri="",
                                        musicxml_uri="",
                                        humanized_midi_uri="",
                                        tunechat_job_id=tc_result.job_id,
                                        tunechat_preview_image_url=tc_result.preview_image_url,
                                    )
                                else:
                                    log.warning("tunechat-only: returned None, falling back to Oh Sheet pipeline")
                            # Silent-failure contract: any error (network,
                            # SDK, blob read) drops us to the Oh Sheet
                            # pipeline below. Never crash the job just
                            # because the optional TuneChat path failed.
                            except Exception as exc:  # noqa: BLE001
                                log.warning("tunechat-only: failed (%s), falling back to Oh Sheet pipeline", exc)

                        # Audio/MIDI uploads use Oh Sheet's own pipeline
                        # (Basic Pitch + music21). TuneChat is not fired for
                        # these routes — only for title_lookup jobs above.

                elif step == "transcribe":
                    payload_uri = self._serialize_stage_input(job_id, step, current_payload)
                    output_uri = await self._dispatch_task(task_name, job_id, payload_uri, config.stage_timeout_sec)
                    txr_dict = self.blob_store.get_json(output_uri)

                elif step in ("arrange", "condense"):
                    if txr_dict is None:
                        bundle_obj = InputBundle.model_validate(current_payload)
                        log.info(
                            "pipeline job_id=%s %s: using MIDI→TranscriptionResult passthrough",
                            job_id, step,
                        )
                        txr_obj = _bundle_to_transcription(
                            bundle_obj,
                            blob_store=self.blob_store,
                            job_id=job_id,
                        )
                        txr_dict = txr_obj.model_dump(mode="json")
                    payload_uri = self._serialize_stage_input(job_id, step, txr_dict)
                    output_uri = await self._dispatch_task(task_name, job_id, payload_uri, config.stage_timeout_sec)
                    score_dict = self.blob_store.get_json(output_uri)

                elif step == "transform":
                    if score_dict is None:
                        raise RuntimeError(
                            "transform stage requires a PianoScore — none was produced"
                        )
                    payload_uri = self._serialize_stage_input(job_id, step, score_dict)
                    output_uri = await self._dispatch_task(task_name, job_id, payload_uri, config.stage_timeout_sec)
                    score_dict = self.blob_store.get_json(output_uri)

                elif step == "humanize":
                    if score_dict is None:
                        raise RuntimeError("humanize stage requires a PianoScore — none was produced")
                    payload_uri = self._serialize_stage_input(job_id, step, score_dict)
                    output_uri = await self._dispatch_task(task_name, job_id, payload_uri, config.stage_timeout_sec)
                    perf_dict = self.blob_store.get_json(output_uri)

                elif step == "refine":
                    if perf_dict is not None:
                        refine_envelope = {
                            "payload": perf_dict,
                            "payload_type": "HumanizedPerformance",
                            "title_hint": bundle.metadata.title,
                            "artist_hint": bundle.metadata.artist,
                            "filename_hint": bundle.metadata.source_filename,
                        }
                    elif score_dict is not None:
                        refine_envelope = {
                            "payload": score_dict,
                            "payload_type": "PianoScore",
                            "title_hint": bundle.metadata.title,
                            "artist_hint": bundle.metadata.artist,
                            "filename_hint": bundle.metadata.source_filename,
                        }
                    else:
                        raise RuntimeError(
                            "refine stage requires a score or performance — none was produced"
                        )
                    payload_uri = self._serialize_stage_input(job_id, step, refine_envelope)
                    output_uri = await self._dispatch_task(task_name, job_id, payload_uri, config.stage_timeout_sec)
                    refined = self.blob_store.get_json(output_uri)
                    if refined["payload_type"] == "HumanizedPerformance":
                        perf_dict = refined["payload"]
                    else:
                        score_dict = refined["payload"]

                elif step == "engrave":
                    # Title/composer precedence: refined ScoreMetadata > InputMetadata > defaults.
                    refined_md: dict | None = None
                    if perf_dict is not None:
                        refined_md = perf_dict.get("score", {}).get("metadata") if isinstance(perf_dict, dict) else None
                    elif score_dict is not None:
                        refined_md = score_dict.get("metadata") if isinstance(score_dict, dict) else None
                    resolved_title = (refined_md or {}).get("title") or title
                    resolved_composer = (refined_md or {}).get("composer") or composer

                    # OHSHEET_ENGRAVER_INFERENCE toggle: when on, route audio /
                    # midi uploads through the oh-sheet-ml-pipeline engraver
                    # service instead of the local music21 engrave stage.
                    # title_lookup jobs (TuneChat, cover_search) keep the
                    # existing path regardless.
                    use_ml_engraver = (
                        settings.engraver_inference
                        and bundle.metadata.source in ("audio_upload", "midi_upload")
                    )

                    if use_ml_engraver:
                        from backend.contracts import (  # noqa: PLC0415
                            ExpressionMap,
                            ExpressiveNote,
                            HumanizedPerformance,
                            PianoScore,
                        )
                        from backend.services.engrave import _render_midi_bytes  # noqa: PLC0415
                        from backend.services.ml_engraver_client import (  # noqa: PLC0415
                            engrave_midi_via_ml_service,
                        )

                        if perf_dict is not None:
                            perf_obj = HumanizedPerformance.model_validate(perf_dict)
                        elif score_dict is not None:
                            score_obj = PianoScore.model_validate(score_dict)
                            expressive_notes = [
                                ExpressiveNote(
                                    score_note_id=n.id,
                                    pitch=n.pitch,
                                    onset_beat=n.onset_beat,
                                    duration_beat=n.duration_beat,
                                    velocity=n.velocity,
                                    hand=hand_name,  # type: ignore[arg-type]
                                    voice=n.voice,
                                    timing_offset_ms=0.0,
                                    velocity_offset=0,
                                )
                                for hand_name, notes in (
                                    ("rh", score_obj.right_hand),
                                    ("lh", score_obj.left_hand),
                                )
                                for n in notes
                            ]
                            perf_obj = HumanizedPerformance(
                                schema_version=SCHEMA_VERSION,
                                expressive_notes=expressive_notes,
                                expression=ExpressionMap(),
                                score=score_obj,
                                quality=QualitySignal(
                                    overall_confidence=0.5,
                                    warnings=["engrave-from-score"],
                                ),
                            )
                        else:
                            raise RuntimeError(
                                "engrave stage requires a score or performance — none was produced"
                            )

                        # _render_midi_bytes is synchronous (pretty_midi I/O);
                        # keep the event loop free.
                        midi_bytes = await asyncio.to_thread(_render_midi_bytes, perf_obj)
                        musicxml_bytes = await engrave_midi_via_ml_service(midi_bytes)

                        prefix = f"jobs/{job_id}/output"
                        musicxml_uri = self.blob_store.put_bytes(
                            f"{prefix}/score.musicxml", musicxml_bytes,
                        )
                        midi_uri = self.blob_store.put_bytes(
                            f"{prefix}/humanized.mid", midi_bytes,
                        )

                        result_dict = EngravedOutput(
                            schema_version=SCHEMA_VERSION,
                            metadata=EngravedScoreData(
                                includes_dynamics=False,
                                includes_pedal_marks=False,
                                includes_fingering=False,
                                includes_chord_symbols=False,
                                title=resolved_title,
                                composer=resolved_composer,
                            ),
                            pdf_uri="",
                            musicxml_uri=musicxml_uri,
                            humanized_midi_uri=midi_uri,
                            audio_preview_uri=None,
                        ).model_dump(mode="json")
                        log.info(
                            "pipeline engrave via ML service job_id=%s source=%s "
                            "musicxml_bytes=%d midi_bytes=%d",
                            job_id,
                            bundle.metadata.source,
                            len(musicxml_bytes),
                            len(midi_bytes),
                        )
                    else:
                        if perf_dict is not None:
                            engrave_envelope = {
                                "payload": perf_dict,
                                "payload_type": "HumanizedPerformance",
                                "job_id": job_id,
                                "title": resolved_title,
                                "composer": resolved_composer,
                            }
                        elif score_dict is not None:
                            engrave_envelope = {
                                "payload": score_dict,
                                "payload_type": "PianoScore",
                                "job_id": job_id,
                                "title": resolved_title,
                                "composer": resolved_composer,
                            }
                        else:
                            raise RuntimeError("engrave stage requires a score or performance — none was produced")
                        payload_uri = self._serialize_stage_input(job_id, step, engrave_envelope)
                        output_uri = await self._dispatch_task(task_name, job_id, payload_uri, config.stage_timeout_sec)
                        result_dict = self.blob_store.get_json(output_uri)

                else:
                    raise RuntimeError(f"unknown stage in execution plan: {step!r}")
            except Exception:
                log.exception(
                    "pipeline stage failed job_id=%s stage=%s index=%d/%d",
                    job_id, step, i + 1, n,
                )
                raise

            elapsed_ms = (time.perf_counter() - t0) * 1000
            log.info(
                "pipeline stage done job_id=%s stage=%s duration_ms=%.0f",
                job_id, step, elapsed_ms,
            )
            emit(step, "stage_completed", progress=(i + 1) / n)

        if result_dict is None:
            raise RuntimeError("pipeline finished without producing an EngravedOutput")

        result = EngravedOutput.model_validate(result_dict)
        # Carry the raw transcription MIDI URI (if any) onto the final
        # output so the artifacts route can serve it without needing a
        # separate handle on TranscriptionResult.
        if txr_dict is not None and txr_dict.get("transcription_midi_uri"):
            result = result.model_copy(
                update={"transcription_midi_uri": txr_dict["transcription_midi_uri"]},
            )
        log.info("pipeline finished job_id=%s", job_id)
        return result
