"""Pydantic models for the Song-to-Humanized-Piano-Sheet-Music pipeline.

Mirrors ``api-contracts-v2.md`` (Schema Version ``3.0.0``). Field names and
semantics match the spec exactly so JSON payloads can round-trip between this
service, orchestrators (Temporal/Step Functions), and the existing local
``temp1/contracts.py`` dataclasses.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = "3.1.0"


# ---------------------------------------------------------------------------
# §1  Orchestration envelopes
# ---------------------------------------------------------------------------

class OrchestratorCommand(BaseModel):
    schema_version: str
    job_id: str
    step_id: str
    payload_uri: str               # URI to the input contract JSON in blob storage
    timeout_sec: int


class WorkerResponse(BaseModel):
    schema_version: str
    job_id: str
    status: Literal["success", "recoverable_error", "fatal_error"]
    output_uri: str | None = None
    logs: str | None = None


# ---------------------------------------------------------------------------
# §2  Shared primitives
# ---------------------------------------------------------------------------

class RemoteAudioFile(BaseModel):
    uri: str
    format: Literal["mp3", "wav", "flac", "m4a"]
    sample_rate: int
    duration_sec: float
    channels: int
    content_hash: str | None = None


class RemoteMidiFile(BaseModel):
    uri: str
    ticks_per_beat: int
    content_hash: str | None = None


class SectionLabel(str, Enum):
    INTRO = "intro"
    VERSE = "verse"
    PRE_CHORUS = "pre_chorus"
    CHORUS = "chorus"
    BRIDGE = "bridge"
    INTERLUDE = "interlude"
    OUTRO = "outro"
    SOLO = "solo"
    OTHER = "other"


class InstrumentRole(str, Enum):
    MELODY = "melody"
    BASS = "bass"
    CHORDS = "chords"
    PIANO = "piano"
    OTHER = "other"


class QualitySignal(BaseModel):
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


class TempoMapEntry(BaseModel):
    """Single anchor in the seconds↔beats tempo map.

    A constant-tempo song has one entry; variable-tempo songs have one entry
    per change point. Workers crossing the seconds/beats boundary MUST use
    the tempo map and MUST NOT assume constant tempo.
    """
    time_sec: float
    beat: float
    bpm: float


def sec_to_beat(time_sec: float, tempo_map: list[TempoMapEntry]) -> float:
    """Convert real-time seconds to a beat position via the tempo map.

    Walks the map and uses the last anchor at or before ``time_sec`` —
    constant-tempo songs collapse to the trivial ``beat = sec * bpm/60``.
    """
    if not tempo_map:
        raise ValueError("tempo_map is empty")
    entry = tempo_map[0]
    for e in tempo_map:
        if e.time_sec <= time_sec:
            entry = e
        else:
            break
    return entry.beat + (time_sec - entry.time_sec) * (entry.bpm / 60.0)


def beat_to_sec(beat: float, tempo_map: list[TempoMapEntry]) -> float:
    """Convert a beat position to real-time seconds via the tempo map."""
    if not tempo_map:
        raise ValueError("tempo_map is empty")
    entry = tempo_map[0]
    for e in tempo_map:
        if e.beat <= beat:
            entry = e
        else:
            break
    return entry.time_sec + (beat - entry.beat) * (60.0 / entry.bpm)


# ---------------------------------------------------------------------------
# Contract 1 — INPUT INGESTION
# ---------------------------------------------------------------------------

class InputMetadata(BaseModel):
    title: str | None = None
    artist: str | None = None
    source: Literal["title_lookup", "audio_upload", "midi_upload"]
    # When True, the ingest stage will attempt to find a clean piano
    # cover of the song (via yt-dlp search + scoring) and swap the
    # user's URL for the cover's URL before transcription. See
    # backend/services/cover_search.py for the matching logic.
    # Defaults to False so existing callers and fixtures keep working.
    prefer_clean_source: bool = False


class InputBundle(BaseModel):
    schema_version: str = SCHEMA_VERSION
    audio: RemoteAudioFile | None = None
    midi: RemoteMidiFile | None = None
    metadata: InputMetadata


# ---------------------------------------------------------------------------
# Contract 2 — TRANSCRIBE
# ---------------------------------------------------------------------------

class Note(BaseModel):
    pitch: int = Field(..., ge=0, le=127)
    onset_sec: float
    offset_sec: float
    velocity: int = Field(..., ge=0, le=127)


class MidiTrack(BaseModel):
    notes: list[Note]
    instrument: InstrumentRole
    program: int | None = Field(default=None, ge=0, le=127)  # GM program emitted by the model
    confidence: float = Field(..., ge=0.0, le=1.0)


class RealtimeChordEvent(BaseModel):
    time_sec: float
    duration_sec: float
    label: str                                 # Harte notation, e.g. "C:maj7"
    root: int
    confidence: float = Field(..., ge=0.0, le=1.0)


class Section(BaseModel):
    start_sec: float
    end_sec: float
    label: SectionLabel


class HarmonicAnalysis(BaseModel):
    key: str                                   # e.g. "C:major"
    time_signature: tuple[int, int]
    tempo_map: list[TempoMapEntry]
    chords: list[RealtimeChordEvent] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)


class TranscriptionResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    midi_tracks: list[MidiTrack]
    analysis: HarmonicAnalysis
    quality: QualitySignal
    transcription_midi_uri: str | None = None


# ---------------------------------------------------------------------------
# Contract 3 — PIANO ARRANGEMENT
# ---------------------------------------------------------------------------

Difficulty = Literal["beginner", "intermediate", "advanced"]


class ScoreNote(BaseModel):
    id: str                                    # e.g. "rh-0042"
    pitch: int = Field(..., ge=0, le=127)
    onset_beat: float
    duration_beat: float
    velocity: int = Field(..., ge=0, le=127)
    voice: int


class ScoreChordEvent(BaseModel):
    beat: float
    duration_beat: float
    label: str                                 # Harte notation
    root: int
    # Propagated from the upstream ``RealtimeChordEvent`` so engrave can
    # gate chord-symbol rendering on transcriber confidence. Defaults to
    # 1.0 for legacy/test inputs that don't carry a confidence score.
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ScoreSection(BaseModel):
    start_beat: float
    end_beat: float
    label: SectionLabel
    phrase_boundaries: list[float] = Field(default_factory=list)


class ScoreMetadata(BaseModel):
    key: str
    time_signature: tuple[int, int]
    tempo_map: list[TempoMapEntry]
    difficulty: Difficulty
    sections: list[ScoreSection] = Field(default_factory=list)
    chord_symbols: list[ScoreChordEvent] = Field(default_factory=list)


class PianoScore(BaseModel):
    schema_version: str = SCHEMA_VERSION
    right_hand: list[ScoreNote]
    left_hand: list[ScoreNote]
    metadata: ScoreMetadata


# ---------------------------------------------------------------------------
# Contract 4 — HUMANIZE PERFORMANCE
# ---------------------------------------------------------------------------

class ExpressiveNote(BaseModel):
    score_note_id: str
    pitch: int = Field(..., ge=0, le=127)
    onset_beat: float
    duration_beat: float
    velocity: int = Field(..., ge=0, le=127)
    hand: Literal["rh", "lh"]
    voice: int
    # Onset-only nudge: engrave applies this to the attack time and leaves the
    # release on the metronomic grid, so a positive value shortens the note
    # and a negative value lengthens it. Do not interpret as a whole-note shift.
    timing_offset_ms: float = Field(..., ge=-50.0, le=50.0)
    velocity_offset: int = Field(..., ge=-30, le=30)


class DynamicMarking(BaseModel):
    beat: float
    type: Literal["pp", "p", "mp", "mf", "f", "ff", "crescendo", "decrescendo"]
    span_beats: float | None = None
    target: str | None = None


class Articulation(BaseModel):
    beat: float
    hand: Literal["rh", "lh"]
    score_note_id: str
    type: Literal["tenuto", "staccato", "legato", "accent", "fermata"]


class PedalEvent(BaseModel):
    onset_beat: float
    offset_beat: float
    type: Literal["sustain", "sostenuto", "una_corda"]


class TempoChange(BaseModel):
    beat: float
    type: Literal["accel", "rit", "a_tempo", "fermata"]
    target_bpm: float | None = None


class ExpressionMap(BaseModel):
    dynamics: list[DynamicMarking] = Field(default_factory=list)
    articulations: list[Articulation] = Field(default_factory=list)
    pedal_events: list[PedalEvent] = Field(default_factory=list)
    tempo_changes: list[TempoChange] = Field(default_factory=list)


class HumanizedPerformance(BaseModel):
    schema_version: str = SCHEMA_VERSION
    expressive_notes: list[ExpressiveNote]
    expression: ExpressionMap
    score: PianoScore
    quality: QualitySignal


# ---------------------------------------------------------------------------
# Contract 4b — REFINE (opt-in LLM stage between humanize and engrave)
# ---------------------------------------------------------------------------

RefineRationale = Literal[
    "harmony_correction",
    "ghost_note_removal",
    "octave_correction",
    "voice_leading",
    "duplicate_removal",
    "out_of_range",
    "timing_cleanup",
    "velocity_cleanup",
    "other",
]


class RefineEditOp(BaseModel):
    """A single modify-or-delete edit emitted by the refine LLM.

    The LLM has modify+delete authority only; note addition is explicitly
    forbidden (see REQUIREMENTS.md "Out of Scope" — LLM adding notes). The
    ``rationale`` enum is closed per CTR-01; new categories require a schema
    bump. Phase 2 may tighten after prompt-quality data is in.
    """

    op: Literal["modify", "delete"]
    target_note_id: str
    rationale: RefineRationale
    # Modify-only fields — all optional; validator enforces at-least-one for op="modify".
    # For op="delete", target_note_id alone is sufficient; these stay None.
    pitch: int | None = Field(default=None, ge=0, le=127)
    velocity: int | None = Field(default=None, ge=0, le=127)          # absolute replacement
    velocity_offset: int | None = Field(default=None, ge=-30, le=30)  # additive (mirrors ExpressiveNote)
    timing_offset_ms: float | None = Field(default=None, ge=-50.0, le=50.0)
    duration_beat: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _modify_requires_at_least_one_field(self) -> "RefineEditOp":
        if self.op == "modify" and all(
            v is None
            for v in (
                self.pitch,
                self.velocity,
                self.velocity_offset,
                self.timing_offset_ms,
                self.duration_beat,
            )
        ):
            raise ValueError(
                "RefineEditOp with op='modify' must provide at least one of: "
                "pitch, velocity, velocity_offset, timing_offset_ms, duration_beat"
            )
        return self


class RefineCitation(BaseModel):
    """Web-search grounding citation for an LLM refine decision.

    ``confidence`` is [0.0, 1.0] per Claude's Discretion (mirrors QualitySignal.overall_confidence,
    Note.confidence, ScoreChordEvent.confidence convention).
    """

    url: str
    snippet: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class RefinedPerformance(BaseModel):
    """Wrapper around a HumanizedPerformance post-LLM refinement (D-05, D-06).

    Nested composition — NOT subclass (refined is not a humanized-in-place;
    edits are a separate log). Engrave unwraps via ``refined.refined_performance``
    per D-07. ``source_performance_digest`` is the SHA-256 (hexdigest, 64 chars)
    of the pre-edit HumanizedPerformance — drift detection between refine
    input and output.
    """

    schema_version: str = SCHEMA_VERSION  # "3.1.0" post-bump
    refined_performance: HumanizedPerformance  # POST-edit — what engrave renders
    edits: list[RefineEditOp]
    citations: list[RefineCitation]
    model: str
    source_performance_digest: str  # hashlib.sha256(canonical_json).hexdigest() — 64-char lowercase hex


# ---------------------------------------------------------------------------
# Contract 5 — ENGRAVE → OUTPUT
# ---------------------------------------------------------------------------

class EngravedScoreData(BaseModel):
    includes_dynamics: bool
    includes_pedal_marks: bool
    includes_fingering: bool
    includes_chord_symbols: bool
    title: str
    composer: str


class EngravedOutput(BaseModel):
    schema_version: str = SCHEMA_VERSION
    metadata: EngravedScoreData
    pdf_uri: str
    musicxml_uri: str
    humanized_midi_uri: str
    audio_preview_uri: str | None = None
    transcription_midi_uri: str | None = None


# ---------------------------------------------------------------------------
# Pipeline routing
# ---------------------------------------------------------------------------

PipelineVariant = Literal["full", "audio_upload", "midi_upload", "sheet_only"]

# How seconds-domain transcription becomes a beat-domain PianoScore.
# ``arrange`` — hand assignment, dedup, quantization (default).
# ``condense_transform`` — merge all tracks into one piano stream (condense) then
# transform (passthrough for now).
ScorePipelineMode = Literal["arrange", "condense_transform"]


class PipelineConfig(BaseModel):
    variant: PipelineVariant
    skip_humanizer: bool = False
    stage_timeout_sec: int = 600
    score_pipeline: ScorePipelineMode = "arrange"
    # Phase 1 (CFG-01): opt-in LLM refine stage. When True,
    # get_execution_plan inserts "refine" after "humanize" (or after
    # "arrange" / "transform" for sheet_only, which has no humanize).
    # The kill switch (OHSHEET_REFINE_KILL_SWITCH) is NOT checked here —
    # it is enforced at the HTTP boundary in create_job(), which coerces
    # enable_refine to False before constructing PipelineConfig. This keeps
    # get_execution_plan pure: same PipelineConfig produces same plan, always.
    enable_refine: bool = False

    def get_execution_plan(self) -> list[str]:
        """Return the list of stages to invoke in order, per the variant."""
        routing: dict[str, list[str]] = {
            "full":         ["ingest", "transcribe", "arrange", "humanize", "engrave"],
            "audio_upload": ["ingest", "transcribe", "arrange", "humanize", "engrave"],
            "midi_upload":  ["ingest", "arrange", "humanize", "engrave"],
            "sheet_only":   ["ingest", "transcribe", "arrange", "engrave"],
        }
        plan = list(routing[self.variant])
        if self.skip_humanizer and "humanize" in plan:
            plan.remove("humanize")
        if self.score_pipeline == "condense_transform":
            try:
                idx = plan.index("arrange")
            except ValueError:
                pass
            else:
                plan[idx : idx + 1] = ["condense", "transform"]
        # Phase 1 (CFG-01): refine insertion after humanize; for sheet_only
        # (no humanize) refine goes after the last score-producing stage
        # (transform if condense_transform is active, else arrange).
        if self.enable_refine:
            if "humanize" in plan:
                plan.insert(plan.index("humanize") + 1, "refine")
            else:
                for anchor in ("transform", "arrange"):
                    if anchor in plan:
                        plan.insert(plan.index(anchor) + 1, "refine")
                        break
        return plan
