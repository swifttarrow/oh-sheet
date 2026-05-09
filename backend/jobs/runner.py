"""PipelineRunner — dispatches pipeline stages as Celery tasks.

The runner owns the execution plan (which stages run in what order)
and uses the claim-check pattern: serialize each stage's input to
blob storage, dispatch a Celery task with the payload URI, wait for
the result URI, and deserialize the output for the next stage.
"""
from __future__ import annotations

import asyncio
import hashlib
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
    PianoScore,
    PipelineConfig,
    QualitySignal,
    ScoreMetadata,
    ScoreNote,
    TempoMapEntry,
    TranscriptionResult,
    sec_to_beat,
)
from backend.jobs.events import JobEvent
from backend.services.pretty_midi_tracks import (
    harmonic_analysis_from_pretty_midi,
    midi_tracks_from_pretty_midi,
)
from backend.storage.base import BlobStore

log = logging.getLogger(__name__)

EventCallback = Callable[[JobEvent], None]

# Maps execution plan step names to Celery task names. The engrave stage
# is NOT in this map — it's handled inline via the ml_engraver HTTP client
# rather than dispatched as a Celery task.
STEP_TO_TASK: dict[str, str] = {
    "ingest": "ingest.run",
    "separate": "separate.run",
    "transcribe": "transcribe.run",
    "interpret": "interpret.run",
    "arrange": "arrange.run",
    "condense": "condense.run",
    "transform": "transform.run",
    "humanize": "humanize.run",
    "refine": "refine.run",
}


def _compute_audio_hash(bundle: InputBundle, blob_store: BlobStore) -> str:
    """SHA-256 of the user's audio bytes, hex-encoded.

    Used as the ``user_audio_hash`` column in
    ``eval_production_quality_scores`` per strategy doc §6.1 — the
    hash is GDPR-clean (the audio itself never lands in Postgres) and
    stable across re-runs so the dashboard can de-dup repeat uploads
    of the same source.

    Returns ``"unknown"`` when the bundle has no audio (sheet-only
    paths) or when the blob fetch fails — telemetry still records
    the row with ``"unknown"`` so the per-job table stays complete.
    """
    if bundle.audio is None:
        return "unknown"
    try:
        audio_bytes = blob_store.get_bytes(bundle.audio.uri)
    except Exception:  # noqa: BLE001
        return "unknown"
    return hashlib.sha256(audio_bytes).hexdigest()


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


_DEFAULT_STAFF_SPLIT = 60  # middle C — also the engrave-side default.


def _cover_score_from_transcription(
    txr: TranscriptionResult,
    *,
    difficulty: str = "intermediate",
) -> PianoScore:
    """Synthesize a :class:`PianoScore` from cover-mode transcription output.

    The Phase 8 ``pop_cover`` variant skips ``arrange`` because AMT-APC
    has already done the arrangement work — chord voicings, accompaniment
    patterns, hand-friendly note placement. We don't want the rules-based
    arranger to second-guess any of that. But the engraver expects a
    :class:`PianoScore` (right_hand / left_hand split), so we still need
    to attribute notes to staves and convert from seconds to beats.

    The split is the simplest possible: pitch < 60 → LH, pitch ≥ 60 → RH.
    Cover-mode models tend to produce overlapping registers across hands
    in dense chordal sections, but the staff cut at middle C is a
    reasonable default and matches the engraver's ``staff_split_hint``.
    Voice assignment is also flat (voice=1 for every note) since the
    cover model didn't emit voice info — the engraver will lay this out
    as single-voice notation per staff.

    Chord symbols, sections, key, time signature, tempo map, and
    downbeats are all carried from :attr:`TranscriptionResult.analysis`
    so the rendered PDF still benefits from the audio-side analysis
    (Beat This! downbeats, librosa key estimation, chord recognition)
    even though arrange was skipped.
    """
    analysis = txr.analysis
    tempo_map = list(analysis.tempo_map) or [
        TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0),
    ]

    rh_idx = 0
    lh_idx = 0
    right_hand: list[ScoreNote] = []
    left_hand: list[ScoreNote] = []
    for track in txr.midi_tracks:
        for note in track.notes:
            onset_beat = sec_to_beat(note.onset_sec, tempo_map)
            offset_beat = sec_to_beat(note.offset_sec, tempo_map)
            duration_beat = max(offset_beat - onset_beat, 0.0)
            if note.pitch >= _DEFAULT_STAFF_SPLIT:
                right_hand.append(
                    ScoreNote(
                        id=f"rh-{rh_idx:04d}",
                        pitch=note.pitch,
                        onset_beat=onset_beat,
                        duration_beat=duration_beat,
                        velocity=note.velocity,
                        voice=1,
                    )
                )
                rh_idx += 1
            else:
                left_hand.append(
                    ScoreNote(
                        id=f"lh-{lh_idx:04d}",
                        pitch=note.pitch,
                        onset_beat=onset_beat,
                        duration_beat=duration_beat,
                        velocity=note.velocity,
                        voice=1,
                    )
                )
                lh_idx += 1

    right_hand.sort(key=lambda n: (n.onset_beat, n.pitch))
    left_hand.sort(key=lambda n: (n.onset_beat, n.pitch))

    # Reuse arrange's chord/section helpers so the cover path emits the
    # same beat-domain structures arrange would have produced. Imported
    # lazily to avoid pulling music21/numpy into the runner module on
    # the import path of variants that don't exercise cover mode.
    from backend.services.arrange import (  # noqa: PLC0415
        _chord_to_score_chord,
        _section_to_score_section,
    )

    chord_symbols = [_chord_to_score_chord(c, tempo_map) for c in analysis.chords]
    sections = [_section_to_score_section(s, tempo_map) for s in analysis.sections]

    return PianoScore(
        schema_version=SCHEMA_VERSION,
        right_hand=right_hand,
        left_hand=left_hand,
        metadata=ScoreMetadata(
            key=analysis.key,
            time_signature=analysis.time_signature,
            tempo_map=tempo_map,
            difficulty=difficulty,  # type: ignore[arg-type]
            sections=sections,
            chord_symbols=chord_symbols,
            downbeats=list(analysis.downbeats),
            pedal_events=[],  # AMT-APC does not emit pedal
            staff_split_hint=_DEFAULT_STAFF_SPLIT,
        ),
    )


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
            # engrave is handled inline (ML HTTP client) and has no
            # STEP_TO_TASK entry; other stages still dispatch via Celery.
            task_name = STEP_TO_TASK.get(step, "")

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
                                log.info("tunechat-only: sending audio for job_id=%s", job_id)
                                tc_result = await transcribe_via_tunechat(
                                    audio_bytes, "audio.wav",
                                    title=title, artist=composer,
                                )
                                if tc_result is not None:
                                    # Synthetic stage events only fire on the
                                    # success path. On TuneChat failure / None
                                    # result we fall back to the regular Oh
                                    # Sheet pipeline below; the outer loop's
                                    # own emit() calls become canonical.
                                    # Previously these fired before the await,
                                    # leaving subscribers with a phantom
                                    # "transcribe stage_started" + duplicate
                                    # "ingest stage_completed" on every fallback.
                                    emit("ingest", "stage_completed", progress=0.25)
                                    emit("transcribe", "stage_started", progress=0.25)
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
                                        pdf_uri=None,
                                        musicxml_uri="",
                                        humanized_midi_uri="",
                                        tunechat_job_id=tc_result.job_id,
                                        tunechat_preview_image_url=tc_result.preview_image_url,
                                        # Capture the hosted artifact URLs so
                                        # /v1/artifacts/{job}/{kind} can proxy
                                        # downloads (see artifacts.py). Without
                                        # these the download chips 404 on every
                                        # TuneChat-fast-path job.
                                        tunechat_midi_url=tc_result.midi_url,
                                        tunechat_musicxml_url=tc_result.musicxml_url,
                                        tunechat_pdf_url=tc_result.pdf_url,
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

                elif step == "separate":
                    # Phase 5: source-separation runs between ingest and
                    # transcribe. The worker enriches the bundle with
                    # ``audio_stems`` URIs (or returns it unchanged on
                    # failure — graceful degradation into transcribe's
                    # legacy inline path). The next stage reads
                    # ``current_payload`` directly.
                    payload_uri = self._serialize_stage_input(job_id, step, current_payload)
                    output_uri = await self._dispatch_task(task_name, job_id, payload_uri, config.stage_timeout_sec)
                    current_payload = self.blob_store.get_json(output_uri)
                    n_stems = len(current_payload.get("audio_stems", {}))
                    log.info(
                        "pipeline separate done job_id=%s stems=%d",
                        job_id, n_stems,
                    )

                elif step == "transcribe":
                    payload_uri = self._serialize_stage_input(job_id, step, current_payload)
                    output_uri = await self._dispatch_task(task_name, job_id, payload_uri, config.stage_timeout_sec)
                    txr_dict = self.blob_store.get_json(output_uri)

                elif step == "interpret":
                    # For midi_upload jobs, transcribe was skipped so
                    # txr_dict is None. Run the same MIDI→TranscriptionResult
                    # passthrough that arrange uses so interpret always
                    # receives a valid TranscriptionResult.
                    if txr_dict is None:
                        bundle_obj = InputBundle.model_validate(current_payload)
                        log.info(
                            "pipeline job_id=%s interpret: using MIDI→TranscriptionResult passthrough",
                            job_id,
                        )
                        txr_obj = _bundle_to_transcription(
                            bundle_obj,
                            blob_store=self.blob_store,
                            job_id=job_id,
                        )
                        txr_dict = txr_obj.model_dump(mode="json")
                    interpret_envelope = {
                        "txr": txr_dict,
                        "prompt": bundle.metadata.arrangement_prompt or "",
                        "title_hint": bundle.metadata.title,
                        "artist_hint": bundle.metadata.artist,
                    }
                    payload_uri = self._serialize_stage_input(job_id, step, interpret_envelope)
                    output_uri = await self._dispatch_task(task_name, job_id, payload_uri, config.stage_timeout_sec)
                    enriched = self.blob_store.get_json(output_uri)
                    txr_dict = enriched.get("txr")
                    if txr_dict is None:
                        raise RuntimeError(
                            "interpret worker output missing required 'txr' key"
                        )

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
                    # title_lookup jobs normally resolve via TuneChat and
                    # return before reaching this stage. If TuneChat failed
                    # or returned None, the pipeline falls through to Oh
                    # Sheet's own stages. Let the local engrave run — a
                    # lower-quality result is better than a user-facing crash.
                    if bundle.metadata.source == "title_lookup":
                        log.warning(
                            "title_lookup job reached engrave without TuneChat result — "
                            "proceeding with local pipeline for job_id=%s",
                            job_id,
                        )

                    # Title/composer precedence: refined ScoreMetadata > InputMetadata > defaults.
                    refined_md: dict | None = None
                    if perf_dict is not None:
                        refined_md = perf_dict.get("score", {}).get("metadata") if isinstance(perf_dict, dict) else None
                    elif score_dict is not None:
                        refined_md = score_dict.get("metadata") if isinstance(score_dict, dict) else None
                    resolved_title = (refined_md or {}).get("title") or title
                    resolved_composer = (refined_md or {}).get("composer") or composer

                    from backend.contracts import (  # noqa: PLC0415
                        ExpressionMap,
                        ExpressiveNote,
                        HumanizedPerformance,
                        PianoScore,
                    )
                    from backend.services import engrave_local as engrave_local_module  # noqa: PLC0415
                    from backend.services.engrave_local import (  # noqa: PLC0415
                        EngraveLocalError,
                    )
                    from backend.services.midi_render import render_midi  # noqa: PLC0415
                    from backend.services.ml_engraver_client import (  # noqa: PLC0415
                        engrave_midi_via_ml_service,
                    )

                    if perf_dict is not None:
                        perf_obj = HumanizedPerformance.model_validate(perf_dict)
                    elif score_dict is None and config.variant == "pop_cover" and txr_dict is not None:
                        # Phase 8: cover mode skips arrange/humanize. Convert
                        # the AMT-APC TranscriptionResult into a minimal
                        # PianoScore (middle-C hand split) so the engraver
                        # can render directly. The cover model has already
                        # made arrangement decisions — we just need to
                        # split the stream onto two staves and convert
                        # seconds → beats for the engraver.
                        txr_obj = TranscriptionResult.model_validate(txr_dict)
                        score_obj = _cover_score_from_transcription(txr_obj)
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
                        synthesized_expression = ExpressionMap()
                        perf_obj = HumanizedPerformance(
                            schema_version=SCHEMA_VERSION,
                            expressive_notes=expressive_notes,
                            expression=synthesized_expression,
                            score=score_obj,
                            quality=QualitySignal(
                                overall_confidence=0.5,
                                warnings=["engrave-from-cover-transcription"],
                            ),
                        )
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
                        # Phase 6: when humanize is skipped (sheet_only),
                        # carry transcribed pedal events from the score
                        # metadata into the synthesized ExpressionMap so
                        # engrave_local renders ``Ped. ___ *`` brackets
                        # even on the sheet_only path.
                        synthesized_expression = ExpressionMap(
                            pedal_events=list(score_obj.metadata.pedal_events),
                        )
                        perf_obj = HumanizedPerformance(
                            schema_version=SCHEMA_VERSION,
                            expressive_notes=expressive_notes,
                            expression=synthesized_expression,
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

                    # render_midi is synchronous (pretty_midi + mido I/O);
                    # keep the event loop free. The MIDI bytes are persisted
                    # as the ``humanized_midi_uri`` artifact regardless of
                    # which engrave backend produces the score.
                    rendered = await asyncio.to_thread(render_midi, perf_obj)
                    midi_bytes = rendered.midi_bytes
                    emitted = rendered.features

                    # ── Backend dispatch ─────────────────────────────────
                    # ``local``       — music21 → MusicXML + LilyPond → PDF.
                    #                   Reads the structured score directly,
                    #                   so chord symbols / dynamics / pedal /
                    #                   per-note voice all survive into
                    #                   MusicXML — and the includes_* flags
                    #                   come from the actual emitted content.
                    # ``remote_http`` — POST MIDI bytes to the ML service.
                    #                   No PDF; flags read off the MIDI render.
                    #
                    # Local backend falls through to remote on
                    # :class:`EngraveLocalError` so a missing LilyPond
                    # install (dev machines without the apt package) still
                    # produces a MusicXML artifact via the remote service.
                    pdf_bytes: bytes | None = None
                    local_features = None
                    engrave_route = "remote_http"
                    if settings.engrave_backend == "local":
                        try:
                            local_result = await asyncio.to_thread(
                                engrave_local_module.engrave_score_locally,
                                perf_obj.score,
                                perf_obj.expression,
                                title=resolved_title,
                                composer=resolved_composer,
                                render_pdf=True,
                            )
                            musicxml_bytes = local_result.musicxml_bytes
                            pdf_bytes = local_result.pdf_bytes
                            local_features = local_result.features
                            engrave_route = "local"
                        except EngraveLocalError as exc:
                            log.warning(
                                "local engrave failed (%s) — falling through to remote HTTP "
                                "for job_id=%s",
                                exc, job_id,
                            )
                            musicxml_bytes = await engrave_midi_via_ml_service(midi_bytes)
                            engrave_route = "remote_http_fallback"
                    else:
                        musicxml_bytes = await engrave_midi_via_ml_service(midi_bytes)

                    prefix = f"jobs/{job_id}/output"
                    musicxml_uri = self.blob_store.put_bytes(
                        f"{prefix}/score.musicxml", musicxml_bytes,
                    )
                    midi_uri = self.blob_store.put_bytes(
                        f"{prefix}/humanized.mid", midi_bytes,
                    )
                    pdf_uri: str | None = None
                    if pdf_bytes:
                        pdf_uri = self.blob_store.put_bytes(
                            f"{prefix}/score.pdf", pdf_bytes,
                        )

                    # ``includes_*`` flags reflect what was actually emitted
                    # into the score artifact: when local engrave succeeds,
                    # they come from music21's emission counters; otherwise
                    # they fall back to the MIDI render's feature summary
                    # (which is what the remote engrave consumes).
                    # ``includes_fingering`` stays False until a fingering
                    # generator ships; nothing in the current pipeline
                    # produces that data.
                    if local_features is not None:
                        includes_dynamics = local_features.dynamic_count > 0
                        includes_pedal_marks = local_features.pedal_event_count > 0
                        includes_chord_symbols = local_features.chord_symbol_count > 0
                    else:
                        includes_dynamics = emitted.dynamics
                        includes_pedal_marks = emitted.pedal_marks
                        includes_chord_symbols = emitted.chord_symbols

                    # Phase 7: composite-Q telemetry. Computes the Tier 3
                    # + Tier 2-lite quality report from the engraved score
                    # in-process (cheap), persists to Postgres when
                    # ``OHSHEET_EVAL_TELEMETRY_DSN`` is configured (no-op
                    # otherwise), and attaches the report to the output
                    # contract so the API surface and result screen can
                    # show "high quality / decent / lower confidence".
                    evaluation_report_payload: dict | None = None
                    try:
                        from backend.eval.telemetry import (  # noqa: PLC0415
                            emit_production_quality,
                        )
                        prod_report = emit_production_quality(
                            score=perf_obj.score,
                            perf=perf_obj,
                            job_id=job_id,
                            user_audio_hash=_compute_audio_hash(bundle, self.blob_store),
                            engrave_route=engrave_route,
                            title=resolved_title,
                        )
                        if prod_report is not None:
                            evaluation_report_payload = prod_report.as_evaluation_report()
                    except Exception:  # noqa: BLE001 — telemetry MUST NOT fail the job
                        log.exception(
                            "pipeline engrave: composite-Q telemetry failed "
                            "(job_id=%s) — continuing without evaluation_report",
                            job_id,
                        )

                    result_dict = EngravedOutput(
                        schema_version=SCHEMA_VERSION,
                        metadata=EngravedScoreData(
                            includes_dynamics=includes_dynamics,
                            includes_pedal_marks=includes_pedal_marks,
                            includes_fingering=False,
                            includes_chord_symbols=includes_chord_symbols,
                            title=resolved_title,
                            composer=resolved_composer,
                        ),
                        pdf_uri=pdf_uri,
                        musicxml_uri=musicxml_uri,
                        humanized_midi_uri=midi_uri,
                        audio_preview_uri=None,
                        evaluation_report=evaluation_report_payload,
                    ).model_dump(mode="json")
                    log.info(
                        "pipeline engrave route=%s job_id=%s source=%s "
                        "musicxml_bytes=%d midi_bytes=%d pdf_bytes=%d "
                        "key_sig=%s tempo_changes=%d chord_markers=%d "
                        "downbeat_cues=%d pedal_events=%d dynamics=%s",
                        engrave_route,
                        job_id,
                        bundle.metadata.source,
                        len(musicxml_bytes),
                        len(midi_bytes),
                        len(pdf_bytes) if pdf_bytes else 0,
                        emitted.key_signature,
                        emitted.tempo_change_count,
                        emitted.chord_marker_count,
                        emitted.downbeat_cue_count,
                        emitted.pedal_event_count,
                        emitted.dynamics,
                    )

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
