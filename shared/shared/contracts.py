"""Pydantic models for the Song-to-Humanized-Piano-Sheet-Music pipeline.

Mirrors ``api-contracts-v2.md`` (Schema Version ``3.0.0``). Field names and
semantics match the spec exactly so JSON payloads can round-trip between this
service, orchestrators (Temporal/Step Functions), and the existing local
``temp1/contracts.py`` dataclasses.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "3.2.0"


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
    source_filename: str | None = None


class RemoteMidiFile(BaseModel):
    uri: str
    ticks_per_beat: int
    content_hash: str | None = None
    source_filename: str | None = None


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
    source_filename: str | None = None
    # When True, the ingest stage will attempt to find a clean piano
    # cover of the song (via yt-dlp search + scoring) and swap the
    # user's URL for the cover's URL before transcription. See
    # backend/services/cover_search.py for the matching logic.
    # Defaults to False so existing callers and fixtures keep working.
    prefer_clean_source: bool = False
    # Original YouTube URL the user submitted (preserved after
    # title is replaced with the resolved song name by ingest).
    source_url: str | None = None
    # Phase 8: routing hint that survives the InputBundle → transcribe
    # serialization boundary. Set by the API route from the chosen
    # :class:`PipelineVariant` (e.g. ``"pop_cover"`` when the user
    # selected the "Piano cover" toggle on upload). The transcribe
    # worker reads this to pick AMT-APC over Kong/BP — this is
    # plumbing-only and does not affect any other stage. ``str`` (not
    # the PipelineVariant Literal) so a typo in the API layer doesn't
    # blow up serialization on legacy stored bundles; the dispatcher
    # ignores unknown values.
    variant_hint: str | None = None
    # Free-form user prompt for the interpret stage (e.g. "make it
    # beginner-friendly, sparse left hand"). Carried through to the
    # interpret stage worker which converts it to structured
    # ArrangementHints. Not a SecretStr — user-visible in their own job.
    arrangement_prompt: str | None = None


class InputBundle(BaseModel):
    schema_version: str = SCHEMA_VERSION
    audio: RemoteAudioFile | None = None
    midi: RemoteMidiFile | None = None
    metadata: InputMetadata
    # Per-stem blob URIs populated by the source-separation stage
    # (Phase 5, ``backend/workers/separate.py``). Keys are the htdemucs
    # source names — ``vocals``, ``drums``, ``bass``, ``other`` — and
    # values are ``file://`` URIs into the configured blob store. When
    # populated, the transcribe stage consumes these directly instead
    # of running Demucs inline. Empty (default) means separation hasn't
    # run yet OR the separator stage was disabled in PipelineConfig.
    audio_stems: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Contract 2 — TRANSCRIBE
# ---------------------------------------------------------------------------

class PitchBendPoint(BaseModel):
    """One sample on a note's pitch-bend trajectory.

    Self-documenting alternative to a ``tuple[float, float]`` payload:
    the field names survive every cross-language serialization (Dart,
    JSON Schema generators, downstream consumers) so callers don't have
    to remember which slot holds time vs. cents.
    """
    time_sec: float
    cents: float


class Note(BaseModel):
    pitch: int = Field(..., ge=0, le=127)
    onset_sec: float
    offset_sec: float
    velocity: int = Field(..., ge=0, le=127)
    # Pitch-bend trajectory. Populated when Basic Pitch is run with
    # ``multiple_pitch_bends=True`` so vibrato / portamento can survive
    # the contract boundary; an empty list means "no bend information
    # available" (not "no bend present").
    pitch_bend_cents: list[PitchBendPoint] = Field(default_factory=list)


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


class RealtimePedalEvent(BaseModel):
    """Seconds-domain sustain/sostenuto/una_corda pedal event from transcription.

    Emitted by piano-AMT models that track the pedal channel (e.g. ByteDance
    Kong's piano-transcription-inference). ``cc`` is the MIDI control number
    (64=sustain, 66=sostenuto, 67=una_corda). The arrange stage converts
    this to the beat-domain :class:`PedalEvent` for the ExpressionMap so
    the engraver can render ``Ped. ___ *`` brackets at the right place.
    """
    cc: int = Field(..., ge=0, le=127)
    onset_sec: float
    offset_sec: float
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


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
    # Downbeat instants in seconds (one per bar). Populated by the audio
    # beat tracker (Beat This!) when running the audio path; empty for
    # MIDI uploads or when the tracker found nothing. midi_render emits
    # these as Cue Point text events so the engraver can lock bar lines
    # to the perceived pulse instead of inferring them from quantization.
    downbeats: list[float] = Field(default_factory=list)


class TranscriptionResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    midi_tracks: list[MidiTrack]
    analysis: HarmonicAnalysis
    quality: QualitySignal
    transcription_midi_uri: str | None = None
    # Carried through from the InputBundle so downstream stages and the
    # API surface can see which stems backed the transcription. Empty
    # when the separator was off or no stems made it into the pipeline.
    audio_stems: dict[str, str] = Field(default_factory=dict)
    # Phase 6: per-event sustain/sostenuto/una_corda pedal lifeline,
    # populated when the transcriber emits pedal data (currently Kong's
    # piano-transcription-inference). Empty when the active transcriber
    # doesn't model pedal (Basic Pitch, Pop2Piano). Arrange converts
    # these to the beat-domain ExpressionMap.pedal_events.
    pedal_events: list[RealtimePedalEvent] = Field(default_factory=list)
    # Structured arrangement guidance produced by the interpret stage.
    # None when the interpret stage was skipped or returned no hints.
    arrangement_hints: ArrangementHints | None = None


# ---------------------------------------------------------------------------
# Contract 3 — PIANO ARRANGEMENT
# ---------------------------------------------------------------------------

Difficulty = Literal["beginner", "intermediate", "advanced"]
Density = Literal["sparse", "moderate", "dense"]
HandBalance = Literal["lh_lead", "balanced", "rh_lead"]


class ArrangementHints(BaseModel):
    """Structured arrangement guidance produced by the interpret stage.

    All fields are optional — a partially-populated hints object is valid.
    Downstream stages (arrange, humanize) consume whichever fields they
    support and ignore the rest.
    """
    difficulty: Difficulty | None = None
    density: Density | None = None
    style_tags: list[str] = Field(default_factory=list)  # e.g. ["jazz", "ballad"]
    dynamic_emphasis: Literal["soft", "neutral", "bold"] | None = None
    tempo_bias: float = Field(default=0.0, ge=-0.25, le=0.25)  # fractional nudge
    hand_balance: HandBalance | None = None
    # Short LLM rationale for the chosen hints. User-visible: serialized
    # into the job output blob and reachable by any client that
    # deserializes the full ``TranscriptionResult`` (not stripped at the
    # API boundary). Treat as user-facing copy, not internal telemetry.
    notes: str | None = None


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


class Repeat(BaseModel):
    """Repeat bracket in the score.

    ``simple`` — a plain ``|: ... :|`` repeat with no volta brackets.
    ``with_endings`` — a repeated section with 1st/2nd-ending brackets.
    Populated by the refine stage; consumed by engrave.
    """
    start_beat: float
    end_beat: float
    kind: Literal["simple", "with_endings"]


class ScoreSection(BaseModel):
    start_beat: float
    end_beat: float
    label: SectionLabel
    phrase_boundaries: list[float] = Field(default_factory=list)
    # Free-form label set by refine stage. Falls back to ``label`` when
    # absent. Engrave renders whichever is present.
    custom_label: str | None = None


class ScoreMetadata(BaseModel):
    key: str
    time_signature: tuple[int, int]
    tempo_map: list[TempoMapEntry]
    difficulty: Difficulty
    sections: list[ScoreSection] = Field(default_factory=list)
    chord_symbols: list[ScoreChordEvent] = Field(default_factory=list)
    # Populated by the refine stage. All optional so upstream producers
    # that don't know about refine can still build valid ScoreMetadata.
    title: str | None = None
    composer: str | None = None
    arranger: str | None = None
    tempo_marking: str | None = None        # e.g., "Andante"
    # MIDI pitch where the left/right hand split — engrave's default is ~60.
    staff_split_hint: int | None = Field(default=None, ge=0, le=127)
    repeats: list[Repeat] = Field(default_factory=list)
    # Downbeat instants (seconds) carried through from HarmonicAnalysis so
    # render_midi_bytes can emit them as MIDI Cue Points. Lives on
    # ScoreMetadata (not HumanizedPerformance) so condense / arrange /
    # humanize all preserve it without touching the expression contract.
    downbeats: list[float] = Field(default_factory=list)
    # Phase 6: beat-domain pedal events from the transcriber (Kong) carried
    # through arrange so the engrave stage can render ``Ped. ___ *`` even
    # when humanize is skipped (sheet_only variant) or when humanize's
    # heuristic generator should defer to the real transcribed pedal
    # data. Empty when the active transcriber doesn't emit pedal.
    # ``PedalEvent`` is a forward reference because the class is declared
    # in the next section; ``model_rebuild`` runs at the bottom of the file
    # to resolve it (the module's ``from __future__ import annotations``
    # already turns all annotations into strings, so no quoting needed).
    pedal_events: list[PedalEvent] = Field(default_factory=list)
    # Structured arrangement guidance from the interpret stage. Copied
    # from TranscriptionResult.arrangement_hints by ArrangeService so
    # hints survive the txr → PianoScore boundary.
    arrangement_hints: ArrangementHints | None = None


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
# Contract 5 — ENGRAVE → OUTPUT
# ---------------------------------------------------------------------------

class EngravedScoreData(BaseModel):
    includes_dynamics: bool
    includes_pedal_marks: bool
    includes_fingering: bool
    includes_chord_symbols: bool
    title: str
    composer: str


class EvaluationReport(BaseModel):
    """Phase 7 — composite-Q + per-tier breakdown for one production job.

    Populated by ``backend.eval.telemetry.compute_production_quality_report``
    inside the engrave stage. ``composite_q ∈ [0, 1]`` is the strategy
    doc §8.2 score (production-flavored: Tier 3 + Tier 2-lite, no
    Tier 4 because resynth is too slow inline). ``q_version`` lets the
    Grafana dashboard distinguish v0.1 readings from a future
    Tier-5-calibrated v1.0.

    All sub-metric fields are nullable — an empty score (variant where
    arrange / humanize was skipped) populates only the fields that
    actually ran. The ``contributing_terms`` list records which terms
    fed into ``composite_q`` so dashboards can flag degraded readings.
    """

    composite_q: float = Field(..., ge=0.0, le=1.0)
    q_version: str
    contributing_terms: list[str] = Field(default_factory=list)

    tier3_playability_fraction: float | None = None
    tier3_voice_leading_smoothness: float | None = None
    tier3_polyphony_in_target_range: float | None = None
    tier3_sight_readability: float | None = None
    tier3_engraving_warning_count: int | None = None
    tier3_composite: float | None = None

    tier2_has_key: bool | None = None
    tier2_chord_symbol_count: int | None = None


class EngravedOutput(BaseModel):
    schema_version: str = SCHEMA_VERSION
    metadata: EngravedScoreData
    # Nullable: the ML engraver returns MusicXML only, so pdf_uri is
    # ``None`` for the audio/midi-upload path. Legacy consumers that
    # expected an always-present string should now check for ``None``.
    # No SCHEMA_VERSION bump — this branch hasn't shipped and no
    # persisted EngravedOutput blobs depend on the old ``str`` shape.
    pdf_uri: str | None = None
    musicxml_uri: str
    humanized_midi_uri: str
    audio_preview_uri: str | None = None
    transcription_midi_uri: str | None = None
    # Phase 7 telemetry attaches a per-job composite-Q report here.
    # ``None`` when the engrave stage couldn't compute the report
    # (e.g. arrange-stage failure that still produced a stub score).
    evaluation_report: EvaluationReport | None = None

    # TuneChat integration — populated when tunechat_enabled=True and
    # TuneChat responded successfully. job_id powers the "Open in
    # TuneChat" deep link. preview_image_url is a first-page PNG of
    # TuneChat's rendered score for display in Oh Sheet's result screen.
    tunechat_job_id: str | None = None
    tunechat_preview_image_url: str | None = None
    # Artifact URLs hosted on TuneChat's server. When Oh Sheet's
    # pipeline skipped local engraving (TuneChat-only path), these
    # are the only place the rendered artifacts exist. The artifacts
    # endpoint (backend/api/routes/artifacts.py) proxies downloads
    # through these when ``pdf_uri`` / ``musicxml_uri`` / ``humanized_midi_uri``
    # are empty, so /v1/artifacts/{job}/{kind} keeps working
    # regardless of which engraver produced the files.
    tunechat_midi_url: str | None = None
    tunechat_musicxml_url: str | None = None
    tunechat_pdf_url: str | None = None


# ---------------------------------------------------------------------------
# Pipeline routing
# ---------------------------------------------------------------------------

PipelineVariant = Literal[
    "full",
    "audio_upload",
    "midi_upload",
    "sheet_only",
    # Phase 8: cover mode. Audio → Demucs separation → AMT-APC piano cover →
    # local engraver, skipping arrange/humanize/refine entirely. AMT-APC
    # already emits a two-staff piano stream with hand assignment baked in,
    # so the rules-based arranger would only fight it. The result is a
    # pianistic cover (idiomatic accompaniment patterns) instead of a
    # faithful transcription — surfaced as a UI toggle.
    "pop_cover",
]

# How seconds-domain transcription becomes a beat-domain PianoScore.
# ``arrange`` — hand assignment, dedup, quantization (default).
# ``condense_only`` — merge all tracks into one piano stream (condense) then
# transform (passthrough for now).
ScorePipelineMode = Literal["arrange", "condense_only"]

# Source-separation backend. ``htdemucs`` runs the HTDemucs Hybrid Transformer
# Demucs model as a dedicated pipeline stage between ingest and transcribe;
# ``off`` skips the stage entirely (transcribe falls back to its inline path).
SeparatorMode = Literal["htdemucs", "off"]


class PipelineConfig(BaseModel):
    variant: PipelineVariant
    skip_humanizer: bool = False
    enable_refine: bool = True
    enable_interpret: bool = False
    stage_timeout_sec: int = 600
    score_pipeline: ScorePipelineMode = "arrange"
    # Phase 5: pre-transcribe source separation. Default ``htdemucs``
    # routes the audio through ``backend/workers/separate.py`` so the
    # transcribe stage operates on per-stem WAVs instead of the full
    # mix. Set to ``off`` to disable the dedicated stage and fall back
    # to transcribe's legacy inline path (used for backwards compat
    # with deployments that don't run the separate worker).
    separator: SeparatorMode = "htdemucs"

    def get_execution_plan(self) -> list[str]:
        """Return the list of stages to invoke in order, per the variant."""
        routing: dict[str, list[str]] = {
            "full":         ["ingest", "transcribe", "arrange", "humanize", "engrave"],
            "audio_upload": ["ingest", "transcribe", "arrange", "humanize", "engrave"],
            "midi_upload":  ["ingest", "arrange", "humanize", "engrave"],
            "sheet_only":   ["ingest", "transcribe", "arrange", "engrave"],
            # pop_cover: AMT-APC emits an arrangement-ready piano stream,
            # so we deliberately skip arrange/humanize. Refine is also
            # skipped (it dispatches on PianoScore / HumanizedPerformance,
            # neither of which exists in cover mode — only the raw
            # TranscriptionResult does). Separate is inserted below.
            "pop_cover":    ["ingest", "transcribe", "engrave"],
        }
        plan = list(routing[self.variant])
        if self.skip_humanizer and "humanize" in plan:
            plan.remove("humanize")
        if self.score_pipeline == "condense_only":
            try:
                idx = plan.index("arrange")
            except ValueError:
                pass
            else:
                # Replace arrange with just condense (transform is a
                # no-op stub, so skip it to save pipeline time).
                plan[idx : idx + 1] = ["condense"]
        # Insert the separate stage immediately before transcribe when
        # enabled. Variants without a transcribe step (midi_upload) get
        # nothing inserted — Demucs has nothing to separate from a MIDI
        # file.
        if self.separator != "off" and "transcribe" in plan:
            plan.insert(plan.index("transcribe"), "separate")
        # Refine annotates score metadata; it requires a PianoScore or
        # HumanizedPerformance, so cover mode (which has neither) skips
        # the stage even when ``enable_refine`` is set.
        if (
            self.enable_refine
            and "engrave" in plan
            and self.variant != "pop_cover"
        ):
            plan.insert(plan.index("engrave"), "refine")
        # Interpret converts a free-form user prompt to structured
        # ArrangementHints before the arrange stage. Skipped for
        # pop_cover (AMT-APC has already committed to a style).
        if self.enable_interpret and "arrange" in plan and self.variant != "pop_cover":
            plan.insert(plan.index("arrange"), "interpret")
        return plan


# Resolve forward references now that all classes are defined.
# ``ScoreMetadata.pedal_events`` and ``ScoreMetadata.arrangement_hints``,
# ``TranscriptionResult.arrangement_hints`` — referenced types defined later
# in the module.
ScoreMetadata.model_rebuild()
PianoScore.model_rebuild()
TranscriptionResult.model_rebuild()
