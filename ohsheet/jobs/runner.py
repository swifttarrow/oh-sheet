"""PipelineRunner — walks the execution plan and invokes services in order.

Mirrors temp1/orchestrator.py:run_pipeline but works on Pydantic contracts
only — no file paths, no MT4 specifics. Each stage emits stage_started /
stage_completed events through the supplied callback so the JobManager can
forward them to WebSocket subscribers.
"""
from __future__ import annotations

from typing import Callable, Optional

from ohsheet.contracts import (
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
    SeparatedStems,
    TempoMapEntry,
    TranscriptionResult,
)
from ohsheet.jobs.events import JobEvent
from ohsheet.services.arrange import ArrangeService
from ohsheet.services.engrave import EngraveService
from ohsheet.services.humanize import HumanizeService
from ohsheet.services.ingest import IngestService
from ohsheet.services.transcribe import TranscribeService

EventCallback = Callable[[JobEvent], None]


def _bundle_to_transcription(bundle: InputBundle) -> TranscriptionResult:
    """Synthesize a TranscriptionResult from a midi_upload bundle.

    Mirrors temp1/orchestrator.py:_midi_to_transcription. The real version
    parses the MIDI file pointed to by bundle.midi.uri and recovers tempo,
    instruments, and notes via pretty_midi. The stub returns a shape-correct
    placeholder so the downstream stages can be exercised end-to-end.
    """
    return TranscriptionResult(
        schema_version=SCHEMA_VERSION,
        stems=SeparatedStems(),
        midi_tracks=[
            MidiTrack(
                notes=[
                    Note(pitch=60, onset_sec=0.0, offset_sec=0.5, velocity=80),
                    Note(pitch=62, onset_sec=0.5, offset_sec=1.0, velocity=80),
                ],
                instrument=InstrumentRole.PIANO,
                source_stem="midi_upload",
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
            overall_confidence=0.9,
            warnings=["stub midi-to-transcription — replace with pretty_midi parse"],
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
