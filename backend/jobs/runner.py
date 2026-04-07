"""PipelineRunner — walks the execution plan and invokes services in order.

Mirrors temp1/orchestrator.py:run_pipeline but works on Pydantic contracts
only — no file paths, no MT4 specifics. Each stage emits stage_started /
stage_completed events through the supplied callback so the JobManager can
forward them to WebSocket subscribers.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

from backend.contracts import (
    SCHEMA_VERSION,
    EngravedOutput,
    HarmonicAnalysis,
    HumanizedPerformance,
    InputBundle,
    InstrumentRole,
    MidiTrack,
    Note,
    PianoScore,
    PipelineConfig,
    QualitySignal,
    TempoMapEntry,
    TranscriptionResult,
)
from backend.jobs.events import JobEvent
from backend.services.arrange import ArrangeService
from backend.services.engrave import EngraveService
from backend.services.humanize import HumanizeService
from backend.services.ingest import IngestService
from backend.services.transcribe import TranscribeService

log = logging.getLogger(__name__)

EventCallback = Callable[[JobEvent], None]


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
        ingest: IngestService,
        transcribe: TranscribeService,
        arrange: ArrangeService,
        humanize: HumanizeService,
        engrave: EngraveService,
    ) -> None:
        self.ingest = ingest
        self.transcribe = transcribe
        self.arrange = arrange
        self.humanize = humanize
        self.engrave = engrave

    async def run(
        self,
        *,
        job_id: str,
        bundle: InputBundle,
        config: PipelineConfig,
        on_event: Optional[EventCallback] = None,
    ) -> EngravedOutput:
        plan = config.get_execution_plan()
        n = len(plan)

        def emit(stage: str, event_type, **kw) -> None:
            if on_event is None:
                return
            on_event(JobEvent(job_id=job_id, type=event_type, stage=stage, **kw))

        txr: Optional[TranscriptionResult] = None
        score: Optional[PianoScore] = None
        perf: Optional[HumanizedPerformance] = None
        result: Optional[EngravedOutput] = None

        title = bundle.metadata.title or "Untitled"
        composer = bundle.metadata.artist or "Unknown"

        for i, step in enumerate(plan):
            emit(step, "stage_started", progress=i / n)

            if step == "ingest":
                bundle = await self.ingest.run(bundle)

            elif step == "transcribe":
                txr = await self.transcribe.run(bundle)

            elif step == "arrange":
                if txr is None and bundle.midi is not None:
                    # midi_upload variant skips transcription; build a passthrough.
                    txr = _bundle_to_transcription(bundle)
                if txr is None:
                    raise RuntimeError(
                        "arrange stage requires a TranscriptionResult — none was produced"
                    )
                score = await self.arrange.run(txr)

            elif step == "humanize":
                if score is None:
                    raise RuntimeError(
                        "humanize stage requires a PianoScore — none was produced"
                    )
                perf = await self.humanize.run(score)

            elif step == "engrave":
                target: HumanizedPerformance | PianoScore | None = perf or score
                if target is None:
                    raise RuntimeError(
                        "engrave stage requires a score or performance — none was produced"
                    )
                result = await self.engrave.run(
                    target, job_id=job_id, title=title, composer=composer,
                )

            else:
                raise RuntimeError(f"unknown stage in execution plan: {step!r}")

            emit(step, "stage_completed", progress=(i + 1) / n)

        if result is None:
            raise RuntimeError("pipeline finished without producing an EngravedOutput")
        return result
