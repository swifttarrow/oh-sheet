"""PipelineRunner — dispatches pipeline stages as Celery tasks.

The runner owns the execution plan (which stages run in what order)
and uses the claim-check pattern: serialize each stage's input to
blob storage, dispatch a Celery task with the payload URI, wait for
the result URI, and deserialize the output for the next stage.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

from celery import Celery

from backend.contracts import (
    SCHEMA_VERSION,
    EngravedOutput,
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
from backend.storage.base import BlobStore

log = logging.getLogger(__name__)

EventCallback = Callable[[JobEvent], None]

# Maps execution plan step names to Celery task names.
STEP_TO_TASK: dict[str, str] = {
    "ingest": "ingest.run",
    "transcribe": "transcribe.run",
    "arrange": "arrange.run",
    "humanize": "humanize.run",
    "engrave": "engrave.run",
}


def _gm_program_to_role(program: int, is_drum: bool) -> InstrumentRole:
    if is_drum:
        return InstrumentRole.OTHER
    if program < 8:
        return InstrumentRole.PIANO
    if 32 <= program <= 39:
        return InstrumentRole.BASS
    if 72 <= program <= 79:
        return InstrumentRole.MELODY
    return InstrumentRole.CHORDS


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


def _bundle_to_transcription(bundle: InputBundle) -> TranscriptionResult:
    """Build a TranscriptionResult from a midi_upload bundle.

    Real path: parse the MIDI file via pretty_midi, recover the tempo map,
    fold each instrument into a MidiTrack, infer key/time-signature.
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

    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as exc:  # noqa: BLE001 — bad MIDI bytes shouldn't crash the worker
        return _stub_transcription(f"pretty_midi parse failed: {exc}")

    midi_tracks: list[MidiTrack] = []
    for instrument in pm.instruments:
        notes = [
            Note(
                pitch=int(n.pitch),
                onset_sec=float(n.start),
                offset_sec=float(max(n.end, n.start + 0.01)),
                velocity=int(max(1, min(127, n.velocity))),
            )
            for n in instrument.notes
        ]
        if not notes:
            continue
        midi_tracks.append(MidiTrack(
            notes=notes,
            instrument=_gm_program_to_role(int(instrument.program), bool(instrument.is_drum)),
            program=None if instrument.is_drum else int(instrument.program),
            confidence=0.9,
        ))

    if not midi_tracks:
        return _stub_transcription("midi file contained no notes")

    # Tempo map: pretty_midi returns parallel arrays of (time, qpm).
    tempo_times, tempo_bpms = pm.get_tempo_changes()
    tempo_map: list[TempoMapEntry] = []
    if len(tempo_times) > 0:
        beat_cursor = 0.0
        prev_time = 0.0
        prev_bpm = float(tempo_bpms[0])
        for t, bpm in zip(tempo_times, tempo_bpms):
            t = float(t)
            bpm = float(bpm)
            beat_cursor += (t - prev_time) * (prev_bpm / 60.0)
            tempo_map.append(TempoMapEntry(time_sec=t, beat=beat_cursor, bpm=bpm))
            prev_time = t
            prev_bpm = bpm
    if not tempo_map:
        tempo_map = [TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)]

    time_signature: tuple[int, int] = (4, 4)
    if pm.time_signature_changes:
        first = pm.time_signature_changes[0]
        time_signature = (int(first.numerator), int(first.denominator))

    key = "C:major"
    if pm.key_signature_changes:
        try:
            key_number = int(pm.key_signature_changes[0].key_number)
            key = pretty_midi.key_number_to_key_name(key_number).replace(" ", ":")
        except Exception:  # noqa: BLE001
            pass

    total_notes = sum(len(t.notes) for t in midi_tracks)
    log.info(
        "Parsed MIDI %s: %d tracks, %d notes",
        midi_path.name, len(midi_tracks), total_notes,
    )

    return TranscriptionResult(
        schema_version=SCHEMA_VERSION,
        midi_tracks=midi_tracks,
        analysis=HarmonicAnalysis(
            key=key,
            time_signature=time_signature,
            tempo_map=tempo_map,
            chords=[],
            sections=[],
        ),
        quality=QualitySignal(
            overall_confidence=0.95,
            warnings=["MIDI input — no harmonic analysis"],
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
            task_name = STEP_TO_TASK[step]

            try:
                if step == "ingest":
                    payload_uri = self._serialize_stage_input(job_id, step, current_payload)
                    output_uri = await self._dispatch_task(task_name, job_id, payload_uri, config.stage_timeout_sec)
                    current_payload = self.blob_store.get_json(output_uri)

                elif step == "transcribe":
                    payload_uri = self._serialize_stage_input(job_id, step, current_payload)
                    output_uri = await self._dispatch_task(task_name, job_id, payload_uri, config.stage_timeout_sec)
                    txr_dict = self.blob_store.get_json(output_uri)

                elif step == "arrange":
                    if txr_dict is None:
                        # midi_upload variant: build transcription from bundle
                        bundle_obj = InputBundle.model_validate(current_payload)
                        log.info(
                            "pipeline job_id=%s arrange: using MIDI→TranscriptionResult passthrough",
                            job_id,
                        )
                        txr_obj = _bundle_to_transcription(bundle_obj)
                        txr_dict = txr_obj.model_dump(mode="json")
                    payload_uri = self._serialize_stage_input(job_id, step, txr_dict)
                    output_uri = await self._dispatch_task(task_name, job_id, payload_uri, config.stage_timeout_sec)
                    score_dict = self.blob_store.get_json(output_uri)

                elif step == "humanize":
                    if score_dict is None:
                        raise RuntimeError("humanize stage requires a PianoScore — none was produced")
                    payload_uri = self._serialize_stage_input(job_id, step, score_dict)
                    output_uri = await self._dispatch_task(task_name, job_id, payload_uri, config.stage_timeout_sec)
                    perf_dict = self.blob_store.get_json(output_uri)

                elif step == "engrave":
                    if perf_dict is not None:
                        engrave_envelope = {
                            "payload": perf_dict,
                            "payload_type": "HumanizedPerformance",
                            "job_id": job_id,
                            "title": title,
                            "composer": composer,
                        }
                    elif score_dict is not None:
                        engrave_envelope = {
                            "payload": score_dict,
                            "payload_type": "PianoScore",
                            "job_id": job_id,
                            "title": title,
                            "composer": composer,
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
