# LLM Engrave Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `refine` pipeline stage that uses Anthropic Claude (with `web_search`) to produce human-readable score metadata (title, composer, key, tempo marking, sections, repeats) that `engrave` renders — without touching note data.

**Architecture:** New Celery worker + service pair (`backend/workers/refine.py`, `backend/services/refine.py`) that slots between `humanize` and `engrave` (or between `arrange` and `engrave` for `sheet_only`). LLM produces annotations only; deterministic code merges them into `ScoreMetadata` on the same `PianoScore` / `HumanizedPerformance` envelope. Gracefully passes input through on any LLM failure.

**Tech Stack:** Python 3.10+, FastAPI, Celery, Pydantic v2, Anthropic Python SDK, pytest, asyncio.

**Spec:** [`docs/superpowers/specs/2026-04-13-llm-engrave-refinement-design.md`](../specs/2026-04-13-llm-engrave-refinement-design.md)

---

## File Structure

**Create:**
- `backend/workers/refine.py` — Celery task shell (~30 lines).
- `backend/services/refine.py` — LLM orchestration, retry/budget/fallback, merge, caching (~250 lines).
- `backend/services/refine_prompt.py` — pure prompt/tool-schema builders (~150 lines).
- `tests/test_refine_prompt.py` — prompt module unit tests.
- `tests/test_refine_service.py` — service unit tests with fake Anthropic client.
- `tests/fixtures/refine/canned_claude_response.json` — recorded LLM response for integration test.
- `eval/fixtures/refine_golden/README.md` — golden-set authoring guide.
- `eval/fixtures/refine_golden/clair_de_lune/input_score.json` — seed fixture.
- `eval/fixtures/refine_golden/clair_de_lune/ground_truth.json` — seed fixture.
- `scripts/eval_refine.py` — live golden-set evaluation harness.

**Modify:**
- `shared/shared/contracts.py` — add `Repeat`, extend `ScoreSection` / `ScoreMetadata`, bump `SCHEMA_VERSION`, extend `PipelineConfig`.
- `backend/contracts.py` — re-export `Repeat`.
- `backend/config.py` — add refine settings.
- `backend/workers/celery_app.py` — route `refine.run`.
- `backend/jobs/runner.py` — handle `"refine"` step + resolve engrave title/composer from refined metadata.
- `tests/conftest.py` — import refine worker, add `disable_real_refine_llm` autouse fixture.
- `tests/test_worker_tasks.py` — add `TestRefineTask`.
- `tests/test_pipeline_config.py` — update existing tests, add new ones.
- `tests/test_jobs.py` — add engrave title/composer resolution test.
- `pyproject.toml` — add `anthropic` dep, `[eval]` extras.
- `Makefile` — add `eval-refine` target.

---

## Task 1: Extend contracts with refinement fields

**Files:**
- Modify: `shared/shared/contracts.py:15` (SCHEMA_VERSION), `:173` (ScoreSection), `:229` (ScoreMetadata), `:341` (PipelineConfig)
- Modify: `backend/contracts.py` (add Repeat to re-exports)
- Test: `tests/test_contracts_refine.py` (new)

### Step 1: Write failing tests

- [ ] Create `tests/test_contracts_refine.py`:

```python
"""Schema-level tests for the refinement contract additions (SCHEMA_VERSION 3.1.0)."""
from __future__ import annotations

from shared.contracts import (
    SCHEMA_VERSION,
    PianoScore,
    Repeat,
    ScoreMetadata,
    ScoreNote,
    ScoreSection,
    SectionLabel,
    TempoMapEntry,
)


def test_schema_version_bumped_to_3_1_0() -> None:
    assert SCHEMA_VERSION == "3.1.0"


def test_repeat_model_requires_beat_range_and_kind() -> None:
    r = Repeat(start_beat=0.0, end_beat=16.0, kind="simple")
    assert r.kind == "simple"
    r2 = Repeat(start_beat=32.0, end_beat=48.0, kind="with_endings")
    assert r2.kind == "with_endings"


def test_score_section_allows_custom_label() -> None:
    s = ScoreSection(
        start_beat=0.0,
        end_beat=16.0,
        label=SectionLabel.OTHER,
        custom_label="Bridge → Solo",
    )
    assert s.custom_label == "Bridge → Solo"


def test_score_section_custom_label_defaults_to_none() -> None:
    s = ScoreSection(start_beat=0.0, end_beat=16.0, label=SectionLabel.INTRO)
    assert s.custom_label is None


def test_score_metadata_accepts_refinement_fields() -> None:
    md = ScoreMetadata(
        key="Db:major",
        time_signature=(4, 4),
        tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=66.0)],
        difficulty="intermediate",
        title="Clair de Lune",
        composer="Claude Debussy",
        arranger=None,
        tempo_marking="Andante",
        staff_split_hint=60,
        repeats=[Repeat(start_beat=0.0, end_beat=16.0, kind="simple")],
    )
    assert md.title == "Clair de Lune"
    assert md.composer == "Claude Debussy"
    assert md.tempo_marking == "Andante"
    assert md.staff_split_hint == 60
    assert len(md.repeats) == 1


def test_score_metadata_refinement_fields_all_optional() -> None:
    md = ScoreMetadata(
        key="C:major",
        time_signature=(4, 4),
        tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
        difficulty="intermediate",
    )
    assert md.title is None
    assert md.composer is None
    assert md.arranger is None
    assert md.tempo_marking is None
    assert md.staff_split_hint is None
    assert md.repeats == []


def test_backwards_compatible_with_v3_0_0_payloads() -> None:
    """A ScoreMetadata JSON with no refinement fields still validates."""
    payload = {
        "key": "G:major",
        "time_signature": [3, 4],
        "tempo_map": [{"time_sec": 0.0, "beat": 0.0, "bpm": 90.0}],
        "difficulty": "beginner",
        "sections": [],
        "chord_symbols": [],
    }
    md = ScoreMetadata.model_validate(payload)
    assert md.title is None
    assert md.repeats == []


def test_backend_reexports_repeat() -> None:
    from backend.contracts import Repeat as BackendRepeat
    assert BackendRepeat is Repeat
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_contracts_refine.py -v`
Expected: FAIL with `ImportError: cannot import name 'Repeat'` and assertions on new fields.

- [ ] **Step 3: Add `Repeat` model, extend `ScoreSection`, extend `ScoreMetadata`, bump `SCHEMA_VERSION`**

In `shared/shared/contracts.py`, change line 15:

```python
SCHEMA_VERSION = "3.1.0"
```

Add new `Repeat` class directly above `ScoreSection` (around line 222):

```python
class Repeat(BaseModel):
    """Repeat bracket in the score.

    ``simple`` — a plain ``|: ... :|`` repeat with no volta brackets.
    ``with_endings`` — a repeated section with 1st/2nd-ending brackets.
    Populated by the refine stage; consumed by engrave.
    """
    start_beat: float
    end_beat: float
    kind: Literal["simple", "with_endings"]
```

Extend `ScoreSection` (add `custom_label`):

```python
class ScoreSection(BaseModel):
    start_beat: float
    end_beat: float
    label: SectionLabel
    phrase_boundaries: list[float] = Field(default_factory=list)
    # Free-form label set by refine stage. Falls back to ``label`` when
    # absent. Engrave renders whichever is present.
    custom_label: str | None = None
```

Extend `ScoreMetadata` (add refinement fields at the end):

```python
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
    staff_split_hint: int | None = None     # MIDI pitch; engrave default ~60
    repeats: list[Repeat] = Field(default_factory=list)
```

In `backend/contracts.py`, add `Repeat` to the import block (keeping alphabetical order — insert after `QualitySignal`):

```python
from shared.contracts import (
    SCHEMA_VERSION,
    Articulation,
    Difficulty,
    DynamicMarking,
    EngravedOutput,
    EngravedScoreData,
    ExpressionMap,
    ExpressiveNote,
    HarmonicAnalysis,
    HumanizedPerformance,
    InputBundle,
    InputMetadata,
    InstrumentRole,
    MidiTrack,
    Note,
    OrchestratorCommand,
    PedalEvent,
    PianoScore,
    PipelineConfig,
    PipelineVariant,
    QualitySignal,
    RealtimeChordEvent,
    RemoteAudioFile,
    RemoteMidiFile,
    Repeat,
    ScoreChordEvent,
    ScoreMetadata,
    ScoreNote,
    ScorePipelineMode,
    ScoreSection,
    Section,
    SectionLabel,
    TempoChange,
    TempoMapEntry,
    TranscriptionResult,
    WorkerResponse,
    beat_to_sec,
    sec_to_beat,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_contracts_refine.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Run full suite to confirm schema bump is backward-compatible**

Run: `pytest tests/ -x --tb=short`
Expected: All tests pass. If any test asserts `SCHEMA_VERSION == "3.0.0"`, update it to `"3.1.0"`.

- [ ] **Step 6: Commit**

```bash
git add shared/shared/contracts.py backend/contracts.py tests/test_contracts_refine.py
git commit -m "feat(contracts): add refine-stage fields + bump SCHEMA_VERSION to 3.1.0"
```

---

## Task 2: Add `enable_refine` flag and wire `refine` into execution plan

**Files:**
- Modify: `shared/shared/contracts.py:341` (PipelineConfig)
- Modify: `tests/test_pipeline_config.py` (existing tests + new tests)

### Step 1: Write failing tests

- [ ] Replace the contents of `tests/test_pipeline_config.py` with:

```python
from __future__ import annotations

from backend.contracts import PipelineConfig


def test_default_plan_includes_refine_before_engrave() -> None:
    """enable_refine defaults to True — refine slots immediately before engrave."""
    cfg = PipelineConfig(variant="audio_upload")
    assert cfg.get_execution_plan() == [
        "ingest",
        "transcribe",
        "arrange",
        "humanize",
        "refine",
        "engrave",
    ]


def test_enable_refine_false_omits_stage() -> None:
    cfg = PipelineConfig(variant="audio_upload", enable_refine=False)
    assert cfg.get_execution_plan() == [
        "ingest",
        "transcribe",
        "arrange",
        "humanize",
        "engrave",
    ]


def test_sheet_only_refine_runs_after_arrange() -> None:
    cfg = PipelineConfig(variant="sheet_only", enable_refine=True)
    assert cfg.get_execution_plan() == [
        "ingest",
        "transcribe",
        "arrange",
        "refine",
        "engrave",
    ]


def test_midi_upload_plan_includes_refine() -> None:
    cfg = PipelineConfig(variant="midi_upload", enable_refine=True)
    assert cfg.get_execution_plan() == [
        "ingest",
        "arrange",
        "humanize",
        "refine",
        "engrave",
    ]


def test_condense_transform_plan_places_refine_before_engrave() -> None:
    cfg = PipelineConfig(
        variant="midi_upload",
        score_pipeline="condense_transform",
    )
    assert cfg.get_execution_plan() == [
        "ingest",
        "condense",
        "transform",
        "humanize",
        "refine",
        "engrave",
    ]


def test_condense_transform_with_skip_humanizer_includes_refine() -> None:
    cfg = PipelineConfig(
        variant="sheet_only",
        score_pipeline="condense_transform",
        skip_humanizer=True,
    )
    assert cfg.get_execution_plan() == [
        "ingest",
        "transcribe",
        "condense",
        "transform",
        "refine",
        "engrave",
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline_config.py -v`
Expected: FAIL — `enable_refine` is not a `PipelineConfig` field; plans don't contain `"refine"`.

- [ ] **Step 3: Add `enable_refine` and insert `"refine"` into the plan**

In `shared/shared/contracts.py`, modify the `PipelineConfig` class and `get_execution_plan`:

```python
class PipelineConfig(BaseModel):
    variant: PipelineVariant
    skip_humanizer: bool = False
    enable_refine: bool = True
    stage_timeout_sec: int = 600
    score_pipeline: ScorePipelineMode = "arrange"

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
        if self.enable_refine and "engrave" in plan:
            # Refine always runs immediately before engrave, regardless of
            # which upstream stages are present. Insert after all prior
            # substitutions so we don't re-anchor on a stage that was
            # already replaced.
            plan.insert(plan.index("engrave"), "refine")
        return plan
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline_config.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add shared/shared/contracts.py tests/test_pipeline_config.py
git commit -m "feat(pipeline): add enable_refine flag, insert refine before engrave"
```

---

## Task 3: Add refine settings to config and declare Anthropic dependency

**Files:**
- Modify: `backend/config.py:40` (near other top-level settings)
- Modify: `pyproject.toml` (dependencies)
- Test: `tests/test_refine_config.py` (new)

### Step 1: Write failing test

- [ ] Create `tests/test_refine_config.py`:

```python
"""Sanity checks on the refine-stage settings surface."""
from __future__ import annotations

from backend.config import Settings


def test_refine_defaults() -> None:
    s = Settings()
    assert s.refine_enabled is True
    assert s.refine_model == "claude-sonnet-4-6"
    assert s.refine_max_searches == 5
    assert s.refine_budget_sec == 300
    assert s.refine_call_timeout_sec == 120
    assert s.anthropic_api_key is None


def test_refine_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("OHSHEET_REFINE_ENABLED", "false")
    monkeypatch.setenv("OHSHEET_REFINE_MODEL", "claude-haiku-4-5-20251001")
    monkeypatch.setenv("OHSHEET_REFINE_MAX_SEARCHES", "3")
    monkeypatch.setenv("OHSHEET_REFINE_BUDGET_SEC", "600")
    monkeypatch.setenv("OHSHEET_REFINE_CALL_TIMEOUT_SEC", "90")
    monkeypatch.setenv("OHSHEET_ANTHROPIC_API_KEY", "sk-test-key")
    s = Settings()
    assert s.refine_enabled is False
    assert s.refine_model == "claude-haiku-4-5-20251001"
    assert s.refine_max_searches == 3
    assert s.refine_budget_sec == 600
    assert s.refine_call_timeout_sec == 90
    assert s.anthropic_api_key == "sk-test-key"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_refine_config.py -v`
Expected: FAIL — attributes don't exist on `Settings`.

- [ ] **Step 3: Add the settings**

In `backend/config.py`, add a new section (after `log_level` around line 40) or at the end of the `Settings` class body:

```python
    # ---- Refine stage (LLM-driven score annotation) -------------------------
    # The refine stage uses Anthropic Claude + the built-in web_search tool to
    # produce human-readable score metadata (title, composer, key, tempo
    # marking, section structure, repeats). See
    # backend/services/refine.py.
    refine_enabled: bool = True
    refine_model: str = "claude-sonnet-4-6"
    refine_max_searches: int = 5              # web_search cap per refinement
    refine_budget_sec: int = 300              # overall wall-time budget
    refine_call_timeout_sec: int = 120        # per-API-call timeout
    anthropic_api_key: str | None = None      # required when refine_enabled
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_refine_config.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Add `anthropic` dependency**

In `pyproject.toml`, add to the `dependencies` list (keep alphabetical where possible):

```toml
dependencies = [
    "ohsheet-shared @ file:shared",
    "anthropic>=0.40",
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "pydantic>=2.5",
    "pydantic-settings>=2.1",
    "python-multipart>=0.0.9",
    "websockets>=12.0",
    "yt-dlp>=2024.1",
    "celery[redis]>=5.3",
    "pretty_midi>=0.2.10",
    "music21>=9.1",
]
```

- [ ] **Step 6: Reinstall backend and run the refine config test**

Run: `make install-backend && pytest tests/test_refine_config.py -v`
Expected: PASS. If `anthropic` fails to install, check Python version (requires 3.9+).

- [ ] **Step 7: Wire `settings.refine_enabled` into `PipelineConfig` at job creation**

The `OHSHEET_REFINE_ENABLED` env var controls whether new jobs include the refine stage. The job-creation route is the only site where `PipelineConfig` is constructed for a live job.

In `backend/api/routes/jobs.py`, find the `config = PipelineConfig(...)` block (around line 151) and add `enable_refine=settings.refine_enabled`:

```python
    config = PipelineConfig(
        variant=variant,
        skip_humanizer=body.skip_humanizer,
        enable_refine=settings.refine_enabled,
        score_pipeline=settings.score_pipeline,
    )
```

- [ ] **Step 8: Commit**

```bash
git add backend/config.py pyproject.toml backend/api/routes/jobs.py tests/test_refine_config.py
git commit -m "feat(config): refine-stage settings + anthropic dep"
```

---

## Task 4: Build the prompt module (pure, no network)

**Files:**
- Create: `backend/services/refine_prompt.py`
- Test: `tests/test_refine_prompt.py` (new)

### Step 1: Write failing tests

- [ ] Create `tests/test_refine_prompt.py`:

```python
"""Unit tests for the refine prompt module — pure functions, no network."""
from __future__ import annotations

from shared.contracts import (
    PianoScore,
    ScoreChordEvent,
    ScoreMetadata,
    ScoreNote,
    SectionLabel,
    TempoMapEntry,
)

from backend.services.refine_prompt import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_chord_sketch,
    build_user_prompt,
    format_chord_sketch,
    submit_refinements_tool_schema,
    web_search_tool_schema,
)


def test_prompt_version_is_a_stable_string() -> None:
    assert isinstance(PROMPT_VERSION, str)
    assert PROMPT_VERSION != ""


def test_system_prompt_mentions_web_search_and_submit() -> None:
    assert "web_search" in SYSTEM_PROMPT
    assert "submit_refinements" in SYSTEM_PROMPT


def test_build_chord_sketch_buckets_by_measure() -> None:
    chords = [
        ScoreChordEvent(beat=0.0, duration_beat=2.0, label="C:maj", root=0),
        ScoreChordEvent(beat=2.0, duration_beat=2.0, label="F:maj", root=5),
        ScoreChordEvent(beat=4.0, duration_beat=4.0, label="G:maj", root=7),
        ScoreChordEvent(beat=8.0, duration_beat=4.0, label="C:maj", root=0),
    ]
    sketch = build_chord_sketch(chords, time_signature=(4, 4))
    assert sketch == [
        (1, ["C:maj", "F:maj"]),
        (2, ["G:maj"]),
        (3, ["C:maj"]),
    ]


def test_build_chord_sketch_handles_empty() -> None:
    assert build_chord_sketch([], time_signature=(4, 4)) == []


def test_build_chord_sketch_handles_three_four() -> None:
    chords = [
        ScoreChordEvent(beat=0.0, duration_beat=3.0, label="D:min", root=2),
        ScoreChordEvent(beat=3.0, duration_beat=3.0, label="A:maj", root=9),
    ]
    sketch = build_chord_sketch(chords, time_signature=(3, 4))
    assert sketch == [(1, ["D:min"]), (2, ["A:maj"])]


def test_format_chord_sketch_renders_bar_lines() -> None:
    sketch = [(1, ["C:maj", "F:maj"]), (2, ["G:maj"])]
    out = format_chord_sketch(sketch)
    assert "bar 1: C:maj | F:maj" in out
    assert "bar 2: G:maj" in out


def test_format_chord_sketch_handles_empty() -> None:
    assert "(no chord analysis available)" in format_chord_sketch([])


def _score_fixture() -> PianoScore:
    return PianoScore(
        right_hand=[
            ScoreNote(id="rh-1", pitch=72, onset_beat=0.0, duration_beat=1.0, velocity=80, voice=1),
            ScoreNote(id="rh-2", pitch=76, onset_beat=4.0, duration_beat=1.0, velocity=80, voice=1),
        ],
        left_hand=[
            ScoreNote(id="lh-1", pitch=48, onset_beat=0.0, duration_beat=2.0, velocity=70, voice=1),
        ],
        metadata=ScoreMetadata(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=100.0)],
            difficulty="intermediate",
            chord_symbols=[
                ScoreChordEvent(beat=0.0, duration_beat=4.0, label="C:maj", root=0),
            ],
        ),
    )


def test_build_user_prompt_includes_detected_fields_and_hint() -> None:
    score = _score_fixture()
    prompt = build_user_prompt(
        title_hint="Test Song",
        artist_hint="Test Artist",
        score=score,
    )
    assert "Test Song" in prompt
    assert "Test Artist" in prompt
    assert "C:major" in prompt
    assert "4/4" in prompt
    assert "100" in prompt
    assert "48-76" in prompt or "48" in prompt  # pitch range
    assert "C:maj" in prompt  # chord sketch


def test_build_user_prompt_handles_missing_hints() -> None:
    score = _score_fixture()
    prompt = build_user_prompt(title_hint=None, artist_hint=None, score=score)
    assert "None" in prompt or "null" in prompt or "unknown" in prompt.lower()


def test_submit_refinements_tool_schema_shape() -> None:
    schema = submit_refinements_tool_schema()
    assert schema["name"] == "submit_refinements"
    props = schema["input_schema"]["properties"]
    for field in (
        "title",
        "composer",
        "arranger",
        "key_signature",
        "time_signature",
        "tempo_bpm",
        "tempo_marking",
        "staff_split_hint",
        "sections",
        "repeats",
    ):
        assert field in props, f"missing field {field!r} in submit_refinements schema"
    # sections items use SectionLabel enum values
    section_label_enum = props["sections"]["items"]["properties"]["label"]["enum"]
    assert "verse" in section_label_enum
    assert "chorus" in section_label_enum
    # repeats items constrain kind
    assert props["repeats"]["items"]["properties"]["kind"]["enum"] == ["simple", "with_endings"]


def test_web_search_tool_schema_has_max_uses() -> None:
    schema = web_search_tool_schema(5)
    assert schema["type"] == "web_search_20250305"
    assert schema["name"] == "web_search"
    assert schema["max_uses"] == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_refine_prompt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.services.refine_prompt'`.

- [ ] **Step 3: Create the prompt module**

Create `backend/services/refine_prompt.py`:

```python
"""Prompt + tool-schema builders for the refine stage. Pure, no network.

The refine stage sends Claude a compact digest of the PianoScore + the
user's title hint, plus the ``web_search`` server tool and a
``submit_refinements`` client tool. The model is expected to research
the song, then emit a single ``submit_refinements`` tool call with the
metadata it was able to justify.
"""
from __future__ import annotations

from typing import Any

from shared.contracts import (
    PianoScore,
    ScoreChordEvent,
    SectionLabel,
)

# Bumping this invalidates the refine cache. See backend/services/refine.py.
PROMPT_VERSION = "v1"

SYSTEM_PROMPT = (
    "You are a music editor refining an automatically-generated piano "
    "transcription. Use the web_search tool to confirm the song's identity, "
    "canonical key signature, form, section structure, and tempo marking. "
    "Then call submit_refinements exactly once with your conclusions.\n\n"
    "Rules:\n"
    " * Only submit values you can justify from your research. Omit any "
    "field you are not confident about — omitted fields fall back to the "
    "automatic detection.\n"
    " * Do NOT invent note data. You are only editing metadata.\n"
    " * Prefer canonical published key signatures over the detected key "
    "when they disagree.\n"
    " * Section boundaries should align with the chord sketch when possible.\n"
)


def build_chord_sketch(
    chord_symbols: list[ScoreChordEvent],
    time_signature: tuple[int, int],
) -> list[tuple[int, list[str]]]:
    """Bucket chord events by 1-based measure number.

    Uses the time-signature numerator as beats-per-measure, which matches
    how the rest of the pipeline counts beats (quarter-note-based when
    ``denominator == 4``). Returns measures in ascending order, each
    paired with its in-order chord labels.
    """
    beats_per_measure = time_signature[0]
    if beats_per_measure <= 0 or not chord_symbols:
        return []
    by_measure: dict[int, list[str]] = {}
    for ev in chord_symbols:
        measure = int(ev.beat // beats_per_measure) + 1
        by_measure.setdefault(measure, []).append(ev.label)
    return sorted(by_measure.items())


def format_chord_sketch(sketch: list[tuple[int, list[str]]]) -> str:
    if not sketch:
        return "  (no chord analysis available)"
    lines = []
    for measure, chords in sketch:
        lines.append(f"  bar {measure}: {' | '.join(chords)}")
    return "\n".join(lines)


def build_user_prompt(
    *,
    title_hint: str | None,
    artist_hint: str | None,
    score: PianoScore,
) -> str:
    """Assemble the user-facing refinement prompt from a PianoScore digest."""
    md = score.metadata
    beats_per_measure = md.time_signature[0] or 4

    last_beat = 0.0
    for n in score.right_hand:
        end = n.onset_beat + n.duration_beat
        if end > last_beat:
            last_beat = end
    for n in score.left_hand:
        end = n.onset_beat + n.duration_beat
        if end > last_beat:
            last_beat = end
    measures = int(last_beat / beats_per_measure) + 1

    pitches = [n.pitch for n in score.right_hand] + [n.pitch for n in score.left_hand]
    low = min(pitches) if pitches else 60
    high = max(pitches) if pitches else 60

    tempo_bpm = md.tempo_map[0].bpm if md.tempo_map else 120.0

    sketch = build_chord_sketch(md.chord_symbols, md.time_signature)

    return (
        "User-provided hint:\n"
        f"  title={title_hint!r}, artist={artist_hint!r}\n"
        "\n"
        "Detected from transcription:\n"
        f"  key = {md.key}\n"
        f"  time_signature = {md.time_signature[0]}/{md.time_signature[1]}\n"
        f"  tempo_bpm = {tempo_bpm:g}\n"
        f"  duration_measures = {measures}\n"
        f"  pitch_range = MIDI {low}-{high}\n"
        "\n"
        "Per-bar chord sketch (Harte notation):\n"
        f"{format_chord_sketch(sketch)}\n"
    )


def submit_refinements_tool_schema() -> dict[str, Any]:
    """JSON-schema tool definition for the terminal ``submit_refinements`` call.

    Field names mirror the new ScoreMetadata refinement fields (see
    shared/shared/contracts.py). All fields are optional — the model is
    instructed to omit values it cannot justify.
    """
    section_label_values = [e.value for e in SectionLabel]
    return {
        "name": "submit_refinements",
        "description": (
            "Submit refined metadata for the piano score. Call exactly once. "
            "Omit fields you are not confident about."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "composer": {"type": "string"},
                "arranger": {"type": "string"},
                "key_signature": {
                    "type": "string",
                    "description": "Harte-style key like 'Db:major' or 'A:minor'.",
                },
                "time_signature": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "[numerator, denominator] — e.g., [4,4] or [3,4].",
                },
                "tempo_bpm": {"type": "number"},
                "tempo_marking": {
                    "type": "string",
                    "description": "Italian marking like 'Andante' or 'Allegro con brio'.",
                },
                "staff_split_hint": {
                    "type": "integer",
                    "description": "MIDI pitch where left/right hand split (typically 60).",
                },
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "start_beat": {"type": "number"},
                            "end_beat": {"type": "number"},
                            "label": {"type": "string", "enum": section_label_values},
                            "custom_label": {"type": "string"},
                        },
                        "required": ["start_beat", "end_beat", "label"],
                    },
                },
                "repeats": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "start_beat": {"type": "number"},
                            "end_beat": {"type": "number"},
                            "kind": {"type": "string", "enum": ["simple", "with_endings"]},
                        },
                        "required": ["start_beat", "end_beat", "kind"],
                    },
                },
            },
        },
    }


def web_search_tool_schema(max_uses: int) -> dict[str, Any]:
    """Anthropic server-side web_search tool — returns inline search results."""
    return {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": max_uses,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_refine_prompt.py -v`
Expected: PASS (12 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/services/refine_prompt.py tests/test_refine_prompt.py
git commit -m "feat(refine): prompt + tool-schema builders (pure module)"
```

---

## Task 5: Build `RefineService` with fake Anthropic client

**Files:**
- Create: `backend/services/refine.py`
- Test: `tests/test_refine_service.py` (new)

### Step 1: Write failing tests

- [ ] Create `tests/test_refine_service.py`:

```python
"""Unit tests for RefineService — uses a fake Anthropic client.

The fake client mimics AsyncAnthropic.messages.create() just enough to
drive the service through its success, cache, and fallback paths without
touching the network.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest
from shared.contracts import (
    HumanizedPerformance,
    PianoScore,
    QualitySignal,
    ScoreMetadata,
    ScoreNote,
    TempoMapEntry,
)
from shared.storage.local import LocalBlobStore

from backend.config import settings
from backend.services.refine import RefineService


def _score() -> PianoScore:
    return PianoScore(
        right_hand=[
            ScoreNote(id="rh-1", pitch=72, onset_beat=0.0, duration_beat=1.0, velocity=80, voice=1),
        ],
        left_hand=[
            ScoreNote(id="lh-1", pitch=48, onset_beat=0.0, duration_beat=2.0, velocity=70, voice=1),
        ],
        metadata=ScoreMetadata(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=100.0)],
            difficulty="intermediate",
        ),
    )


def _humanized(score: PianoScore) -> HumanizedPerformance:
    from shared.contracts import ExpressionMap
    return HumanizedPerformance(
        expressive_notes=[],
        expression=ExpressionMap(),
        score=score,
        quality=QualitySignal(overall_confidence=0.9, warnings=[]),
    )


class _FakeToolUseBlock:
    def __init__(self, name: str, input_: dict[str, Any]) -> None:
        self.type = "tool_use"
        self.name = name
        self.input = input_


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, content: list[Any]) -> None:
        self.content = content


class _FakeMessages:
    def __init__(self, outputs: list[Any]) -> None:
        self._outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        out = self._outputs.pop(0)
        if isinstance(out, BaseException):
            raise out
        if callable(out):
            return await out() if asyncio.iscoroutinefunction(out) else out()
        return out


class _FakeAnthropic:
    def __init__(self, outputs: list[Any]) -> None:
        self.messages = _FakeMessages(outputs)


@pytest.fixture
def blob(tmp_path):
    return LocalBlobStore(tmp_path / "blob")


@pytest.mark.asyncio
async def test_success_merges_refinements_into_score(blob):
    refinements = {
        "title": "Test Song",
        "composer": "Test Composer",
        "tempo_marking": "Andante",
        "key_signature": "Db:major",
        "time_signature": [4, 4],
        "tempo_bpm": 66,
        "staff_split_hint": 60,
        "sections": [
            {"start_beat": 0.0, "end_beat": 16.0, "label": "intro", "custom_label": "Opening"},
        ],
        "repeats": [
            {"start_beat": 0.0, "end_beat": 16.0, "kind": "simple"},
        ],
    }
    fake = _FakeAnthropic([
        _FakeResponse([_FakeToolUseBlock("submit_refinements", refinements)]),
    ])
    svc = RefineService(blob_store=blob, client=fake)
    result = await svc.run(_score(), title_hint="test", artist_hint=None)
    assert isinstance(result, PianoScore)
    md = result.metadata
    assert md.title == "Test Song"
    assert md.composer == "Test Composer"
    assert md.tempo_marking == "Andante"
    assert md.key == "Db:major"
    assert md.time_signature == (4, 4)
    assert md.tempo_map[0].bpm == pytest.approx(66.0)
    assert md.staff_split_hint == 60
    assert len(md.sections) == 1
    assert md.sections[0].custom_label == "Opening"
    assert len(md.repeats) == 1


@pytest.mark.asyncio
async def test_success_preserves_notes(blob):
    fake = _FakeAnthropic([
        _FakeResponse([_FakeToolUseBlock("submit_refinements", {"title": "X"})]),
    ])
    svc = RefineService(blob_store=blob, client=fake)
    original = _score()
    result = await svc.run(original, title_hint=None, artist_hint=None)
    assert [n.pitch for n in result.right_hand] == [n.pitch for n in original.right_hand]
    assert [n.pitch for n in result.left_hand] == [n.pitch for n in original.left_hand]


@pytest.mark.asyncio
async def test_humanized_performance_roundtrip(blob):
    fake = _FakeAnthropic([
        _FakeResponse([_FakeToolUseBlock("submit_refinements", {"title": "X", "composer": "Y"})]),
    ])
    svc = RefineService(blob_store=blob, client=fake)
    perf = _humanized(_score())
    result = await svc.run(perf, title_hint=None, artist_hint=None)
    assert isinstance(result, HumanizedPerformance)
    assert result.score.metadata.title == "X"
    assert result.score.metadata.composer == "Y"


@pytest.mark.asyncio
async def test_llm_failure_passes_input_through_with_warning(blob):
    fake = _FakeAnthropic([
        RuntimeError("network melted"),
        RuntimeError("network melted"),
        RuntimeError("network melted"),
    ])
    svc = RefineService(blob_store=blob, client=fake)
    perf = _humanized(_score())
    result = await svc.run(perf, title_hint=None, artist_hint=None)
    assert isinstance(result, HumanizedPerformance)
    # Notes unchanged
    assert result.score.metadata.title is None
    # Warning attached
    assert any("refine" in w for w in result.quality.warnings)


@pytest.mark.asyncio
async def test_missing_submit_refinements_falls_back(blob):
    fake = _FakeAnthropic([
        _FakeResponse([_FakeTextBlock("I could not find the song.")]),
    ])
    svc = RefineService(blob_store=blob, client=fake)
    perf = _humanized(_score())
    result = await svc.run(perf, title_hint=None, artist_hint=None)
    assert result.score.metadata.title is None
    assert any("refine" in w for w in result.quality.warnings)


@pytest.mark.asyncio
async def test_cache_hit_skips_llm(blob, monkeypatch):
    # First call: write to cache.
    fake1 = _FakeAnthropic([
        _FakeResponse([_FakeToolUseBlock("submit_refinements", {"title": "Cached"})]),
    ])
    svc1 = RefineService(blob_store=blob, client=fake1)
    result1 = await svc1.run(_score(), title_hint=None, artist_hint=None)
    assert result1.metadata.title == "Cached"
    assert len(fake1.messages.calls) == 1

    # Second call with a client whose create() would fail — should not be called.
    fake2 = _FakeAnthropic([RuntimeError("should not be called")])
    svc2 = RefineService(blob_store=blob, client=fake2)
    result2 = await svc2.run(_score(), title_hint=None, artist_hint=None)
    assert result2.metadata.title == "Cached"
    assert len(fake2.messages.calls) == 0


@pytest.mark.asyncio
async def test_budget_exceeded_falls_back(blob, monkeypatch):
    async def _slow() -> Any:
        await asyncio.sleep(10)
        return _FakeResponse([_FakeToolUseBlock("submit_refinements", {"title": "late"})])

    fake = _FakeAnthropic([_slow])
    svc = RefineService(blob_store=blob, client=fake)
    monkeypatch.setattr(settings, "refine_budget_sec", 0.05)
    perf = _humanized(_score())
    result = await svc.run(perf, title_hint=None, artist_hint=None)
    assert result.score.metadata.title is None
    assert any("budget" in w.lower() or "refine" in w for w in result.quality.warnings)


@pytest.mark.asyncio
async def test_invalid_section_label_falls_through_to_other(blob):
    refinements = {
        "sections": [
            {"start_beat": 0.0, "end_beat": 4.0, "label": "not_a_real_label", "custom_label": "Weird"},
            {"start_beat": 4.0, "end_beat": 8.0, "label": "verse"},
        ],
    }
    fake = _FakeAnthropic([
        _FakeResponse([_FakeToolUseBlock("submit_refinements", refinements)]),
    ])
    svc = RefineService(blob_store=blob, client=fake)
    result = await svc.run(_score(), title_hint=None, artist_hint=None)
    assert len(result.metadata.sections) == 2
    assert result.metadata.sections[0].label.value == "other"
    assert result.metadata.sections[1].label.value == "verse"


@pytest.mark.asyncio
async def test_invalid_repeat_kind_is_dropped(blob):
    refinements = {
        "repeats": [
            {"start_beat": 0.0, "end_beat": 8.0, "kind": "da_capo"},   # unsupported
            {"start_beat": 8.0, "end_beat": 16.0, "kind": "simple"},
        ],
    }
    fake = _FakeAnthropic([
        _FakeResponse([_FakeToolUseBlock("submit_refinements", refinements)]),
    ])
    svc = RefineService(blob_store=blob, client=fake)
    result = await svc.run(_score(), title_hint=None, artist_hint=None)
    assert [r.kind for r in result.metadata.repeats] == ["simple"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_refine_service.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.services.refine'`.

- [ ] **Step 3: Implement the service**

Create `backend/services/refine.py`:

```python
"""Refine stage — LLM-driven score metadata annotation.

Consumes a ``PianoScore`` or ``HumanizedPerformance`` envelope, asks
Claude to research the song and return metadata annotations (title,
composer, key, tempo marking, sections, repeats), and merges those
annotations into ``ScoreMetadata`` before handing the envelope
downstream to engrave.

Never raises on LLM failure: on any error / timeout / invalid response
the input is returned unchanged with a warning appended to
``quality.warnings`` (when the envelope is a HumanizedPerformance —
PianoScore has no warnings field).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from shared.contracts import (
    HumanizedPerformance,
    PianoScore,
    QualitySignal,
    Repeat,
    ScoreSection,
    SectionLabel,
)

from backend.config import settings
from backend.services.refine_prompt import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_user_prompt,
    submit_refinements_tool_schema,
    web_search_tool_schema,
)

log = logging.getLogger(__name__)


_VALID_SECTION_LABELS = {e.value for e in SectionLabel}
_VALID_REPEAT_KINDS = {"simple", "with_endings"}


class RefineService:
    name = "refine"

    def __init__(
        self,
        *,
        blob_store: Any | None = None,
        client: Any | None = None,
    ) -> None:
        self.blob_store = blob_store
        self._client = client

    # ---- public entrypoint -------------------------------------------------

    async def run(
        self,
        payload: HumanizedPerformance | PianoScore,
        *,
        title_hint: str | None = None,
        artist_hint: str | None = None,
    ) -> HumanizedPerformance | PianoScore:
        log.info(
            "refine: start title_hint=%r artist_hint=%r humanized=%s",
            title_hint, artist_hint, isinstance(payload, HumanizedPerformance),
        )

        cache_key = self._cache_key(payload)
        cached = self._cache_get(cache_key)
        if cached is not None:
            log.info("refine: cache hit key=%s", cache_key[:12])
            return self._merge(payload, cached)

        score = payload.score if isinstance(payload, HumanizedPerformance) else payload
        try:
            refinements = await asyncio.wait_for(
                self._call_llm(score, title_hint, artist_hint),
                timeout=settings.refine_budget_sec,
            )
        except asyncio.TimeoutError:
            log.warning(
                "refine: budget exceeded (%ss), passing through",
                settings.refine_budget_sec,
            )
            return self._with_warning(
                payload, "refine: LLM budget exceeded, passing through",
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("refine: LLM call failed (%s), passing through", exc)
            return self._with_warning(
                payload,
                f"refine: LLM unavailable ({type(exc).__name__}), passing through",
            )

        if not refinements:
            log.warning("refine: LLM returned no submit_refinements call, passing through")
            return self._with_warning(
                payload, "refine: LLM produced no refinements, passing through",
            )

        self._cache_put(cache_key, refinements)
        merged = self._merge(payload, refinements)
        log.info("refine: done applied_fields=%d", len(refinements))
        return merged

    # ---- LLM plumbing ------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from anthropic import AsyncAnthropic  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(f"anthropic SDK not installed: {exc}") from exc
        api_key = settings.anthropic_api_key
        if not api_key:
            raise RuntimeError("OHSHEET_ANTHROPIC_API_KEY not set")
        self._client = AsyncAnthropic(
            api_key=api_key,
            timeout=settings.refine_call_timeout_sec,
        )
        return self._client

    async def _call_llm(
        self,
        score: PianoScore,
        title_hint: str | None,
        artist_hint: str | None,
    ) -> dict[str, Any] | None:
        user_prompt = build_user_prompt(
            title_hint=title_hint,
            artist_hint=artist_hint,
            score=score,
        )
        tools = [
            web_search_tool_schema(settings.refine_max_searches),
            submit_refinements_tool_schema(),
        ]
        client = self._get_client()

        # 3 attempts total — backoffs before attempts 2 and 3 only.
        backoffs = [0.0, 1.0, 4.0]
        last_exc: BaseException | None = None
        for attempt, delay in enumerate(backoffs):
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                response = await client.messages.create(
                    model=settings.refine_model,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=tools,
                    messages=[{"role": "user", "content": user_prompt}],
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if not _is_transient(exc) or attempt == len(backoffs) - 1:
                    raise
                log.warning(
                    "refine: attempt %d failed (%s), retrying",
                    attempt + 1, exc,
                )
                continue

            for block in response.content:
                if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "submit_refinements":
                    raw = block.input
                    return dict(raw) if isinstance(raw, dict) else json.loads(str(raw))
            return None

        if last_exc is not None:
            raise last_exc
        return None

    # ---- caching -----------------------------------------------------------

    def _cache_key(self, payload: HumanizedPerformance | PianoScore) -> str:
        canon = json.dumps(
            payload.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        h = hashlib.sha256()
        h.update(canon.encode("utf-8"))
        h.update(PROMPT_VERSION.encode("utf-8"))
        h.update(settings.refine_model.encode("utf-8"))
        return h.hexdigest()

    def _cache_uri(self, key: str) -> str | None:
        """Build the cache URI for a LocalBlobStore-backed store only.

        Returns ``None`` for other store types (treated as cache-disabled).
        """
        root = getattr(self.blob_store, "root", None)
        if root is None:
            return None
        return (Path(root) / f"refine-cache/{key}.json").as_uri()

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        if self.blob_store is None:
            return None
        uri = self._cache_uri(key)
        if uri is None or not self.blob_store.exists(uri):
            return None
        try:
            return self.blob_store.get_json(uri)
        except Exception as exc:  # noqa: BLE001
            log.warning("refine: cache read failed (%s), ignoring", exc)
            return None

    def _cache_put(self, key: str, data: dict[str, Any]) -> None:
        if self.blob_store is None:
            return
        try:
            self.blob_store.put_json(f"refine-cache/{key}.json", data)
        except Exception as exc:  # noqa: BLE001
            log.warning("refine: cache write failed (%s), continuing", exc)

    # ---- merge + fallback --------------------------------------------------

    def _with_warning(
        self,
        payload: HumanizedPerformance | PianoScore,
        warning: str,
    ) -> HumanizedPerformance | PianoScore:
        if isinstance(payload, HumanizedPerformance):
            new_quality = QualitySignal(
                overall_confidence=payload.quality.overall_confidence,
                warnings=[*payload.quality.warnings, warning],
            )
            return payload.model_copy(update={"quality": new_quality})
        # PianoScore has no warnings field — just return unchanged.
        return payload

    def _merge(
        self,
        payload: HumanizedPerformance | PianoScore,
        refinements: dict[str, Any],
    ) -> HumanizedPerformance | PianoScore:
        score = payload.score if isinstance(payload, HumanizedPerformance) else payload
        md = score.metadata
        update: dict[str, Any] = {}

        if "title" in refinements:
            update["title"] = str(refinements["title"])[:200]
        if "composer" in refinements:
            update["composer"] = str(refinements["composer"])[:200]
        if "arranger" in refinements:
            update["arranger"] = str(refinements["arranger"])[:200]
        if "tempo_marking" in refinements:
            update["tempo_marking"] = str(refinements["tempo_marking"])[:100]
        if "staff_split_hint" in refinements:
            try:
                v = int(refinements["staff_split_hint"])
            except (TypeError, ValueError):
                v = None
            if v is not None and 0 <= v <= 127:
                update["staff_split_hint"] = v
        if "key_signature" in refinements and isinstance(refinements["key_signature"], str):
            update["key"] = refinements["key_signature"]
        if "time_signature" in refinements:
            ts = refinements["time_signature"]
            if isinstance(ts, (list, tuple)) and len(ts) == 2:
                try:
                    update["time_signature"] = (int(ts[0]), int(ts[1]))
                except (TypeError, ValueError):
                    pass
        if "tempo_bpm" in refinements and md.tempo_map:
            try:
                new_bpm = float(refinements["tempo_bpm"])
            except (TypeError, ValueError):
                new_bpm = None
            if new_bpm is not None and new_bpm > 0:
                first = md.tempo_map[0].model_copy(update={"bpm": new_bpm})
                update["tempo_map"] = [first, *md.tempo_map[1:]]
        if "sections" in refinements:
            parsed = _parse_sections(refinements["sections"])
            if parsed:
                update["sections"] = parsed
        if "repeats" in refinements:
            parsed_r = _parse_repeats(refinements["repeats"])
            update["repeats"] = parsed_r

        new_md = md.model_copy(update=update)
        new_score = score.model_copy(update={"metadata": new_md})
        if isinstance(payload, HumanizedPerformance):
            return payload.model_copy(update={"score": new_score})
        return new_score


def _parse_sections(items: Any) -> list[ScoreSection]:
    if not isinstance(items, list):
        return []
    out: list[ScoreSection] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            start = float(it["start_beat"])
            end = float(it["end_beat"])
        except (KeyError, TypeError, ValueError):
            continue
        label_raw = str(it.get("label", "other")).lower()
        label = SectionLabel(label_raw) if label_raw in _VALID_SECTION_LABELS else SectionLabel.OTHER
        custom = it.get("custom_label")
        custom_str = str(custom)[:100] if isinstance(custom, str) and custom else None
        out.append(ScoreSection(
            start_beat=start,
            end_beat=end,
            label=label,
            custom_label=custom_str,
        ))
    return out


def _parse_repeats(items: Any) -> list[Repeat]:
    if not isinstance(items, list):
        return []
    out: list[Repeat] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        kind = str(it.get("kind", ""))
        if kind not in _VALID_REPEAT_KINDS:
            continue
        try:
            start = float(it["start_beat"])
            end = float(it["end_beat"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append(Repeat(start_beat=start, end_beat=end, kind=kind))  # type: ignore[arg-type]
    return out


def _is_transient(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if "timeout" in msg or "overloaded" in msg or "rate limit" in msg:
        return True
    status = getattr(exc, "status_code", None)
    try:
        return status is not None and 500 <= int(status) < 600
    except (TypeError, ValueError):
        return False
```

- [ ] **Step 4: Install pytest-asyncio if not already present**

Run: `pip show pytest-asyncio | head -2`

If not installed, the `[dev]` extra already pins it — run `make install-backend`.

- [ ] **Step 5: Verify pytest-asyncio mode**

Check that `pyproject.toml` or `pytest.ini` configures `asyncio_mode = "auto"` OR the tests use `@pytest.mark.asyncio` explicitly (they do in this plan).

```bash
grep -A1 "asyncio" pyproject.toml | head -10
```

If `asyncio_mode` is not set, add `[tool.pytest.ini_options]` with `asyncio_mode = "auto"` — or keep the explicit markers already used in the tests.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_refine_service.py -v`
Expected: PASS (9 tests).

- [ ] **Step 7: Commit**

```bash
git add backend/services/refine.py tests/test_refine_service.py
git commit -m "feat(refine): RefineService with retry/budget/fallback + cache"
```

---

## Task 6: Add the Celery worker task

**Files:**
- Create: `backend/workers/refine.py`
- Modify: `backend/workers/celery_app.py:19` (task_routes)
- Modify: `tests/conftest.py:11` (import refine worker) + add `disable_real_refine_llm` autouse fixture
- Modify: `tests/test_worker_tasks.py` (add TestRefineTask)

### Step 1: Write failing test

- [ ] Append `TestRefineTask` to `tests/test_worker_tasks.py`:

```python
class TestRefineTask:
    def test_passes_through_on_llm_failure(self, blob, monkeypatch):
        """The worker wraps the input envelope and calls RefineService,
        which returns the input unchanged when the LLM is unavailable
        (the default for tests — no API key)."""
        from shared.contracts import (
            PianoScore,
            ScoreMetadata,
            ScoreNote,
        )

        from backend.workers.refine import run as refine_run

        score = PianoScore(
            right_hand=[
                ScoreNote(id="rh-1", pitch=60, onset_beat=0.0, duration_beat=1.0, velocity=80, voice=1),
            ],
            left_hand=[
                ScoreNote(id="lh-1", pitch=48, onset_beat=0.0, duration_beat=1.0, velocity=70, voice=1),
            ],
            metadata=ScoreMetadata(
                key="C:major",
                time_signature=(4, 4),
                tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
                difficulty="intermediate",
            ),
        )
        payload_uri = blob.put_json(
            "jobs/test-job/refine/input.json",
            {
                "payload": score.model_dump(mode="json"),
                "payload_type": "PianoScore",
                "title_hint": "test",
                "artist_hint": None,
            },
        )
        output_uri = refine_run("test-job", payload_uri)
        result = blob.get_json(output_uri)
        assert result["payload_type"] == "PianoScore"
        assert result["payload"]["metadata"]["key"] == "C:major"

    def test_humanized_performance_envelope_roundtrip(self, blob):
        from shared.contracts import (
            ExpressionMap,
            HumanizedPerformance,
            PianoScore,
            QualitySignal,
            ScoreMetadata,
            ScoreNote,
        )

        from backend.workers.refine import run as refine_run

        score = PianoScore(
            right_hand=[
                ScoreNote(id="rh-1", pitch=60, onset_beat=0.0, duration_beat=1.0, velocity=80, voice=1),
            ],
            left_hand=[
                ScoreNote(id="lh-1", pitch=48, onset_beat=0.0, duration_beat=1.0, velocity=70, voice=1),
            ],
            metadata=ScoreMetadata(
                key="C:major",
                time_signature=(4, 4),
                tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
                difficulty="intermediate",
            ),
        )
        perf = HumanizedPerformance(
            expressive_notes=[],
            expression=ExpressionMap(),
            score=score,
            quality=QualitySignal(overall_confidence=0.9, warnings=[]),
        )
        payload_uri = blob.put_json(
            "jobs/test-job/refine/input.json",
            {
                "payload": perf.model_dump(mode="json"),
                "payload_type": "HumanizedPerformance",
                "title_hint": None,
                "artist_hint": None,
            },
        )
        output_uri = refine_run("test-job", payload_uri)
        result = blob.get_json(output_uri)
        assert result["payload_type"] == "HumanizedPerformance"
        assert "expressive_notes" in result["payload"]
        assert "score" in result["payload"]
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `pytest tests/test_worker_tasks.py::TestRefineTask -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.workers.refine'`.

- [ ] **Step 3: Create the worker**

Create `backend/workers/refine.py`:

```python
"""Celery task for the refine pipeline stage.

Like engrave, refine accepts a discriminated envelope so the worker
can hydrate the right Pydantic model:

    {
        "payload_type": "PianoScore" | "HumanizedPerformance",
        "payload": <model JSON>,
        "title_hint": str | None,
        "artist_hint": str | None,
    }

The output envelope mirrors the input shape, so the runner can unwrap
it and feed the refined score straight into engrave.
"""
import asyncio

from shared.contracts import HumanizedPerformance, PianoScore
from shared.storage.local import LocalBlobStore

from backend.config import settings
from backend.services.refine import RefineService
from backend.workers.celery_app import celery_app


@celery_app.task(name="refine.run")
def run(job_id: str, payload_uri: str) -> str:
    blob = LocalBlobStore(settings.blob_root)
    raw = blob.get_json(payload_uri)

    payload_type = raw["payload_type"]
    payload_data = raw["payload"]
    title_hint = raw.get("title_hint")
    artist_hint = raw.get("artist_hint")

    if payload_type == "HumanizedPerformance":
        payload = HumanizedPerformance.model_validate(payload_data)
    elif payload_type == "PianoScore":
        payload = PianoScore.model_validate(payload_data)
    else:
        raise ValueError(
            f"Unknown payload_type: {payload_type!r}. "
            "Expected 'HumanizedPerformance' or 'PianoScore'."
        )

    service = RefineService(blob_store=blob)
    # asyncio.run() is safe with Celery's default prefork pool; breaks with gevent/eventlet.
    result = asyncio.run(
        service.run(payload, title_hint=title_hint, artist_hint=artist_hint),
    )

    out = {
        "payload_type": payload_type,
        "payload": result.model_dump(mode="json"),
    }
    output_uri = blob.put_json(
        f"jobs/{job_id}/refine/output.json",
        out,
    )
    return output_uri
```

- [ ] **Step 4: Route the Celery task**

In `backend/workers/celery_app.py`, add a route for `refine.run` to the `task_routes` dict (place it after `humanize.run`):

```python
    task_routes={
        "ingest.run": {"queue": "ingest"},
        "transcribe.run": {"queue": "transcribe"},
        "arrange.run": {"queue": "arrange"},
        "condense.run": {"queue": "arrange"},
        "transform.run": {"queue": "arrange"},
        "humanize.run": {"queue": "humanize"},
        "refine.run": {"queue": "refine"},
        "engrave.run": {"queue": "engrave"},
    },
```

- [ ] **Step 5: Register the worker in the test harness + guarantee no test hits the real API**

In `tests/conftest.py`:

1. Add `import backend.workers.refine  # noqa: F401` to the block of worker imports (keep alphabetical after `humanize`):

```python
import backend.workers.arrange  # noqa: F401
import backend.workers.condense  # noqa: F401
import backend.workers.engrave  # noqa: F401
import backend.workers.humanize  # noqa: F401

# Import monolith worker modules so their tasks are registered on the celery_app.
import backend.workers.ingest  # noqa: F401
import backend.workers.refine  # noqa: F401
import backend.workers.transcribe  # noqa: F401
import backend.workers.transform  # noqa: F401
```

2. Add an autouse fixture that nulls out the Anthropic API key so no test can reach the real API. With the key absent, `RefineService._get_client()` raises `RuntimeError("OHSHEET_ANTHROPIC_API_KEY not set")`, which is caught by `RefineService.run()` and turned into the pass-through + warning path. Tests that want to exercise the real merge logic pass `client=` explicitly to `RefineService` — that path never calls `_get_client()`.

```python
@pytest.fixture(autouse=True)
def disable_real_refine_llm(monkeypatch):
    """Null out the Anthropic API key for every test.

    No test may hit the real API. The service's fallback path (no key →
    raise → caught → pass-through with warning) is exactly what bare
    pipeline tests need. Tests that want to exercise the merge logic
    construct ``RefineService(..., client=fake)`` directly — that path
    bypasses the key check entirely.
    """
    monkeypatch.setattr(settings, "anthropic_api_key", None)
```

- [ ] **Step 6: Run the worker tests**

Run: `pytest tests/test_worker_tasks.py::TestRefineTask -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Commit**

```bash
git add backend/workers/refine.py backend/workers/celery_app.py tests/conftest.py tests/test_worker_tasks.py
git commit -m "feat(refine): Celery worker + test harness wiring"
```

---

## Task 7: Wire `refine` into the PipelineRunner

**Files:**
- Modify: `backend/jobs/runner.py:40` (STEP_TO_TASK) + around `:353` (refine+engrave steps)
- Test: `tests/test_refine_runner.py` (new)

### Step 1: Write failing test

- [ ] Create `tests/test_refine_runner.py`:

```python
"""Integration test: PipelineRunner dispatches refine and passes refined envelope to engrave."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from shared.contracts import (
    InputBundle,
    InputMetadata,
    PipelineConfig,
)
from shared.storage.local import LocalBlobStore

from backend.config import settings
from backend.jobs.runner import PipelineRunner
from backend.workers.celery_app import celery_app


@pytest.mark.asyncio
async def test_runner_invokes_refine_before_engrave():
    """With enable_refine=True, the runner dispatches refine before engrave
    and the refined envelope is what engrave receives."""
    stages_dispatched: list[str] = []

    original_dispatch = PipelineRunner._dispatch_task

    async def _spy_dispatch(self, task_name, job_id, payload_uri, timeout):
        stages_dispatched.append(task_name)
        return await original_dispatch(self, task_name, job_id, payload_uri, timeout)

    with patch.object(PipelineRunner, "_dispatch_task", _spy_dispatch):
        blob = LocalBlobStore(settings.blob_root)
        runner = PipelineRunner(blob_store=blob, celery_app=celery_app)
        bundle = InputBundle(
            metadata=InputMetadata(title="test", source="audio_upload"),
        )
        config = PipelineConfig(variant="audio_upload", enable_refine=True)
        result = await runner.run(job_id="t-refine", bundle=bundle, config=config)

    assert "refine.run" in stages_dispatched
    # Ordering: refine dispatched strictly before engrave.
    assert stages_dispatched.index("refine.run") < stages_dispatched.index("engrave.run")
    assert result.pdf_uri  # engrave still produced an output


@pytest.mark.asyncio
async def test_runner_skips_refine_when_disabled():
    """enable_refine=False omits refine from dispatched tasks."""
    stages_dispatched: list[str] = []

    original_dispatch = PipelineRunner._dispatch_task

    async def _spy_dispatch(self, task_name, job_id, payload_uri, timeout):
        stages_dispatched.append(task_name)
        return await original_dispatch(self, task_name, job_id, payload_uri, timeout)

    with patch.object(PipelineRunner, "_dispatch_task", _spy_dispatch):
        blob = LocalBlobStore(settings.blob_root)
        runner = PipelineRunner(blob_store=blob, celery_app=celery_app)
        bundle = InputBundle(
            metadata=InputMetadata(title="test", source="audio_upload"),
        )
        config = PipelineConfig(variant="audio_upload", enable_refine=False)
        await runner.run(job_id="t-no-refine", bundle=bundle, config=config)

    assert "refine.run" not in stages_dispatched
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_refine_runner.py -v`
Expected: FAIL — runner does not recognize `"refine"` step (`RuntimeError: unknown stage in execution plan: 'refine'`).

- [ ] **Step 3: Wire the runner**

In `backend/jobs/runner.py`, modify `STEP_TO_TASK` to include refine (around line 40):

```python
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
```

In the stage dispatch block inside `PipelineRunner.run`, add a branch for `"refine"` immediately before the `"engrave"` branch (around line 353):

```python
                elif step == "refine":
                    if perf_dict is not None:
                        refine_envelope = {
                            "payload": perf_dict,
                            "payload_type": "HumanizedPerformance",
                            "title_hint": bundle.metadata.title,
                            "artist_hint": bundle.metadata.artist,
                        }
                    elif score_dict is not None:
                        refine_envelope = {
                            "payload": score_dict,
                            "payload_type": "PianoScore",
                            "title_hint": bundle.metadata.title,
                            "artist_hint": bundle.metadata.artist,
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_refine_runner.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run full pipeline-ish tests to verify nothing else regressed**

Run: `pytest tests/test_jobs.py tests/test_stages.py tests/test_worker_tasks.py -v`
Expected: PASS. If any test asserts a specific execution plan, update it to include `"refine"` (or construct the config with `enable_refine=False`).

- [ ] **Step 6: Commit**

```bash
git add backend/jobs/runner.py tests/test_refine_runner.py
git commit -m "feat(refine): wire refine stage into PipelineRunner"
```

---

## Task 8: Engrave reads title/composer from refined ScoreMetadata

**Files:**
- Modify: `backend/jobs/runner.py` (engrave step, around line 353)
- Test: `tests/test_refine_engrave_metadata.py` (new)

### Step 1: Write failing test

- [ ] Create `tests/test_refine_engrave_metadata.py`:

```python
"""Title/composer precedence: refined ScoreMetadata > InputMetadata > defaults."""
from __future__ import annotations

import pytest
from shared.contracts import (
    InputBundle,
    InputMetadata,
    PipelineConfig,
)
from shared.storage.local import LocalBlobStore

from backend.config import settings
from backend.jobs.runner import PipelineRunner
from backend.services import refine as refine_module
from backend.workers.celery_app import celery_app


@pytest.mark.asyncio
async def test_engrave_prefers_refined_title_over_bundle(monkeypatch):
    """When refine populates ScoreMetadata.title, engrave uses it even if
    InputMetadata.title was supplied by the user."""
    async def _canned_refine(self, payload, *, title_hint=None, artist_hint=None):
        # Merge in known refined values.
        return self._merge(payload, {
            "title": "Canonical Title",
            "composer": "Canonical Composer",
        })

    monkeypatch.setattr(refine_module.RefineService, "run", _canned_refine)

    blob = LocalBlobStore(settings.blob_root)
    runner = PipelineRunner(blob_store=blob, celery_app=celery_app)
    bundle = InputBundle(
        metadata=InputMetadata(
            title="user-supplied typo",
            artist="user-supplied artist",
            source="audio_upload",
        ),
    )
    config = PipelineConfig(variant="audio_upload", enable_refine=True)
    result = await runner.run(job_id="t-meta-1", bundle=bundle, config=config)

    assert result.metadata.title == "Canonical Title"
    assert result.metadata.composer == "Canonical Composer"


@pytest.mark.asyncio
async def test_engrave_falls_back_to_bundle_when_refine_empty():
    """With refine disabled, engrave uses InputMetadata.title/artist."""
    blob = LocalBlobStore(settings.blob_root)
    runner = PipelineRunner(blob_store=blob, celery_app=celery_app)
    bundle = InputBundle(
        metadata=InputMetadata(
            title="My User Title",
            artist="My User Artist",
            source="audio_upload",
        ),
    )
    config = PipelineConfig(variant="audio_upload", enable_refine=False)
    result = await runner.run(job_id="t-meta-2", bundle=bundle, config=config)

    assert result.metadata.title == "My User Title"
    assert result.metadata.composer == "My User Artist"


@pytest.mark.asyncio
async def test_engrave_defaults_when_nothing_provided():
    blob = LocalBlobStore(settings.blob_root)
    runner = PipelineRunner(blob_store=blob, celery_app=celery_app)
    bundle = InputBundle(
        metadata=InputMetadata(title=None, artist=None, source="audio_upload"),
    )
    config = PipelineConfig(variant="audio_upload", enable_refine=False)
    result = await runner.run(job_id="t-meta-3", bundle=bundle, config=config)

    assert result.metadata.title == "Untitled"
    assert result.metadata.composer == "Unknown"
```

- [ ] **Step 2: Run tests to verify the first one fails**

Run: `pytest tests/test_refine_engrave_metadata.py -v`
Expected: `test_engrave_prefers_refined_title_over_bundle` FAILs because engrave currently uses the bundle title exclusively.

- [ ] **Step 3: Resolve title/composer from refined metadata inside the engrave step**

In `backend/jobs/runner.py`, update the `"engrave"` branch to prefer refined values. Replace the existing block:

```python
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
```

…with:

```python
                elif step == "engrave":
                    # Title/composer precedence: refined ScoreMetadata > InputMetadata > defaults.
                    refined_md: dict | None = None
                    if perf_dict is not None:
                        refined_md = perf_dict.get("score", {}).get("metadata") if isinstance(perf_dict, dict) else None
                    elif score_dict is not None:
                        refined_md = score_dict.get("metadata") if isinstance(score_dict, dict) else None
                    resolved_title = (refined_md or {}).get("title") or title
                    resolved_composer = (refined_md or {}).get("composer") or composer

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_refine_engrave_metadata.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/jobs/runner.py tests/test_refine_engrave_metadata.py
git commit -m "feat(refine): engrave prefers refined title/composer over bundle metadata"
```

---

## Task 9: Integration test with canned LLM response

**Files:**
- Create: `tests/fixtures/refine/canned_claude_response.json`
- Create: `tests/test_refine_integration.py`

### Step 1: Add a canned LLM response fixture

- [ ] Create `tests/fixtures/refine/canned_claude_response.json`:

```json
{
  "content": [
    {
      "type": "tool_use",
      "name": "submit_refinements",
      "input": {
        "title": "Clair de Lune",
        "composer": "Claude Debussy",
        "arranger": null,
        "key_signature": "Db:major",
        "time_signature": [9, 8],
        "tempo_bpm": 66,
        "tempo_marking": "Andante très expressif",
        "staff_split_hint": 60,
        "sections": [
          {"start_beat": 0.0, "end_beat": 32.0, "label": "intro", "custom_label": "A"},
          {"start_beat": 32.0, "end_beat": 96.0, "label": "verse", "custom_label": "B"},
          {"start_beat": 96.0, "end_beat": 128.0, "label": "outro", "custom_label": "A'"}
        ],
        "repeats": []
      }
    }
  ]
}
```

### Step 2: Write the integration test

- [ ] Create `tests/test_refine_integration.py`:

```python
"""End-to-end integration test: canned LLM response → RefineService → merged envelope → engrave input."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from shared.contracts import (
    HumanizedPerformance,
    ExpressionMap,
    PianoScore,
    QualitySignal,
    ScoreMetadata,
    ScoreNote,
    TempoMapEntry,
)
from shared.storage.local import LocalBlobStore

from backend.config import settings
from backend.services.refine import RefineService


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "refine" / "canned_claude_response.json"


class _CannedToolUse:
    def __init__(self, name: str, input_: dict) -> None:
        self.type = "tool_use"
        self.name = name
        self.input = input_


class _CannedResponse:
    def __init__(self, content: list) -> None:
        self.content = content


class _CannedMessages:
    def __init__(self, raw: dict) -> None:
        self._raw = raw

    async def create(self, **_kwargs):
        blocks = [
            _CannedToolUse(b["name"], b["input"])
            for b in self._raw["content"]
            if b["type"] == "tool_use"
        ]
        return _CannedResponse(blocks)


class _CannedClient:
    def __init__(self, raw: dict) -> None:
        self.messages = _CannedMessages(raw)


@pytest.mark.asyncio
async def test_canned_clair_de_lune_merges_end_to_end():
    raw = json.loads(FIXTURE_PATH.read_text())
    client = _CannedClient(raw)
    blob = LocalBlobStore(settings.blob_root)

    score = PianoScore(
        right_hand=[
            ScoreNote(id="rh-1", pitch=73, onset_beat=0.0, duration_beat=2.0, velocity=70, voice=1),
        ],
        left_hand=[
            ScoreNote(id="lh-1", pitch=37, onset_beat=0.0, duration_beat=4.0, velocity=60, voice=1),
        ],
        metadata=ScoreMetadata(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
            difficulty="intermediate",
        ),
    )
    perf = HumanizedPerformance(
        expressive_notes=[],
        expression=ExpressionMap(),
        score=score,
        quality=QualitySignal(overall_confidence=0.9, warnings=[]),
    )

    svc = RefineService(blob_store=blob, client=client)
    result = await svc.run(perf, title_hint="claire de lune", artist_hint=None)

    assert isinstance(result, HumanizedPerformance)
    md = result.score.metadata
    assert md.title == "Clair de Lune"
    assert md.composer == "Claude Debussy"
    assert md.key == "Db:major"
    assert md.time_signature == (9, 8)
    assert md.tempo_map[0].bpm == pytest.approx(66.0)
    assert md.tempo_marking == "Andante très expressif"
    assert md.staff_split_hint == 60
    assert len(md.sections) == 3
    assert md.sections[0].custom_label == "A"
```

- [ ] **Step 3: Run test to verify it passes**

Run: `pytest tests/test_refine_integration.py -v`
Expected: PASS (1 test).

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/refine/canned_claude_response.json tests/test_refine_integration.py
git commit -m "test(refine): end-to-end integration with canned LLM response"
```

---

## Task 10: Live golden-set eval harness

**Files:**
- Create: `eval/fixtures/refine_golden/README.md`
- Create: `eval/fixtures/refine_golden/clair_de_lune/input_score.json`
- Create: `eval/fixtures/refine_golden/clair_de_lune/ground_truth.json`
- Create: `scripts/eval_refine.py`
- Modify: `pyproject.toml` (add anthropic to `[eval]` extras)
- Modify: `Makefile` (add `eval-refine` target)

### Step 1: Create the fixture directory with a seed song

- [ ] Create `eval/fixtures/refine_golden/README.md`:

```markdown
# Refine Golden Set

A small set of canonical songs used by `scripts/eval_refine.py` to score
the refine stage against human-curated ground truth.

## Fixture layout

Each song gets a directory with two files:

    eval/fixtures/refine_golden/<slug>/
        input_score.json     # PianoScore JSON — the refine stage's input
        ground_truth.json    # expected title/composer/key/time/tempo/sections/repeats

## Authoring a new fixture

1. Pick a song whose title / composer / key / form are unambiguous and
   findable by a competent researcher (Wikipedia, IMSLP, sheet-music
   databases). Prefer classical, standard jazz, and well-documented
   pop/game OST over obscure bootlegs.
2. Generate a plausible `PianoScore` via the pipeline or by hand — it
   should have the approximate measure count, detected (wrong-ish) key,
   detected time signature, and per-beat chord symbols of a real
   transcription of the song. The notes themselves don't need to match
   the recording precisely; refine only sees the digest.
3. Fill in `ground_truth.json` with:
   - `title` (string)
   - `composer` (string)
   - `key_signature` (string, Harte notation)
   - `time_signature` ([numerator, denominator])
   - `tempo_bpm` (number)
   - `sections` (list of `{start_beat, end_beat, label}`)
   - `repeats` (list of `{start_beat, end_beat, kind}`)
4. Run `make eval-refine` — the new fixture's metrics will be added
   to `refine-baseline.json`.

Target: 10–15 fixtures spanning genres. Seed fixture: Clair de Lune.
```

- [ ] Create `eval/fixtures/refine_golden/clair_de_lune/input_score.json` (a small shape-correct PianoScore; the harness doesn't need dense notes since the digest is summary-level):

```json
{
  "schema_version": "3.1.0",
  "right_hand": [
    {"id": "rh-1", "pitch": 73, "onset_beat": 0.0, "duration_beat": 2.0, "velocity": 60, "voice": 1},
    {"id": "rh-2", "pitch": 75, "onset_beat": 2.0, "duration_beat": 2.0, "velocity": 60, "voice": 1},
    {"id": "rh-3", "pitch": 77, "onset_beat": 4.0, "duration_beat": 2.0, "velocity": 60, "voice": 1}
  ],
  "left_hand": [
    {"id": "lh-1", "pitch": 37, "onset_beat": 0.0, "duration_beat": 4.0, "velocity": 55, "voice": 1},
    {"id": "lh-2", "pitch": 44, "onset_beat": 4.0, "duration_beat": 4.0, "velocity": 55, "voice": 1}
  ],
  "metadata": {
    "key": "C:major",
    "time_signature": [4, 4],
    "tempo_map": [{"time_sec": 0.0, "beat": 0.0, "bpm": 72.0}],
    "difficulty": "intermediate",
    "sections": [],
    "chord_symbols": [
      {"beat": 0.0, "duration_beat": 4.0, "label": "Db:maj", "root": 1, "confidence": 0.8},
      {"beat": 4.0, "duration_beat": 4.0, "label": "Ab:7", "root": 8, "confidence": 0.75}
    ]
  }
}
```

- [ ] Create `eval/fixtures/refine_golden/clair_de_lune/ground_truth.json`:

```json
{
  "title": "Clair de Lune",
  "composer": "Claude Debussy",
  "key_signature": "Db:major",
  "time_signature": [9, 8],
  "tempo_bpm": 66,
  "tempo_marking": "Andante très expressif",
  "sections": [
    {"start_beat": 0.0, "end_beat": 24.0, "label": "intro"},
    {"start_beat": 24.0, "end_beat": 72.0, "label": "verse"},
    {"start_beat": 72.0, "end_beat": 96.0, "label": "outro"}
  ],
  "repeats": [],
  "title_hint": "claire de lune",
  "artist_hint": null
}
```

### Step 2: Write the eval harness

- [ ] Create `scripts/eval_refine.py`:

```python
"""Refine eval harness — scores the live refine service against golden ground truth.

Walks every directory under ``eval/fixtures/refine_golden/``, runs the
real ``RefineService`` against each ``input_score.json`` (with the
fixture's ``title_hint`` / ``artist_hint``), and scores the output
against ``ground_truth.json``. Writes per-song and aggregate metrics
to ``refine-baseline.json``.

Requires ``OHSHEET_ANTHROPIC_API_KEY`` to be set. Costs real money.
Excluded from the default test suite and CI — run manually via
``make eval-refine``.

Metrics
-------
* title_exact_match            — case/whitespace-insensitive string equality
* composer_exact_match         — same, for composer
* key_match                    — true if tonic + mode match (Db:major == C#:major)
* time_signature_exact         — tuple equality
* tempo_within_5bpm            — |predicted - ground| <= 5
* section_label_f1             — F1 on section *labels* by greedy overlap match
* repeat_f1                    — F1 on (start_beat, end_beat) pairs (rounded to 0.1 beat)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from shared.contracts import PianoScore  # noqa: E402
from shared.storage.local import LocalBlobStore  # noqa: E402

from backend.services.refine import RefineService  # noqa: E402


# Enharmonic tonic equivalence: normalize before comparison.
_ENHARMONIC = {
    "C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb",
    "Db": "Db", "Eb": "Eb", "Gb": "Gb", "Ab": "Ab", "Bb": "Bb",
    "C": "C", "D": "D", "E": "E", "F": "F", "G": "G", "A": "A", "B": "B",
}


def _norm_key(key: str) -> str:
    if ":" not in key:
        return key.strip().lower()
    tonic, mode = key.split(":", 1)
    tonic = tonic.strip()
    return f"{_ENHARMONIC.get(tonic, tonic)}:{mode.strip().lower()}"


def _norm_str(s: str | None) -> str:
    if s is None:
        return ""
    return " ".join(s.split()).lower()


@dataclass
class FixtureResult:
    slug: str
    title_match: bool
    composer_match: bool
    key_match: bool
    time_sig_match: bool
    tempo_within_5bpm: bool
    section_f1: float
    repeat_f1: float
    predicted_title: str | None
    predicted_composer: str | None


def _score_sections(pred: list[dict[str, Any]], gt: list[dict[str, Any]]) -> float:
    """Simple F1 on section *labels* with overlap matching.

    For each ground-truth section, find any predicted section whose
    [start, end] overlaps at all AND whose label matches; that's a TP.
    """
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    tp = 0
    for g in gt:
        for p in pred:
            overlap = max(0.0, min(p["end_beat"], g["end_beat"]) - max(p["start_beat"], g["start_beat"]))
            if overlap > 0 and p["label"] == g["label"]:
                tp += 1
                break
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gt) if gt else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _score_repeats(pred: list[dict[str, Any]], gt: list[dict[str, Any]]) -> float:
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    def _key(r: dict[str, Any]) -> tuple[float, float, str]:
        return (round(r["start_beat"], 1), round(r["end_beat"], 1), r.get("kind", "simple"))
    pred_set = {_key(r) for r in pred}
    gt_set = {_key(r) for r in gt}
    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt_set) if gt_set else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


async def _eval_fixture(svc: RefineService, fixture_dir: Path) -> FixtureResult:
    input_path = fixture_dir / "input_score.json"
    gt_path = fixture_dir / "ground_truth.json"
    score = PianoScore.model_validate(json.loads(input_path.read_text()))
    gt = json.loads(gt_path.read_text())

    refined = await svc.run(
        score,
        title_hint=gt.get("title_hint"),
        artist_hint=gt.get("artist_hint"),
    )

    md = refined.metadata if hasattr(refined, "metadata") else refined.score.metadata
    pred_sections = [
        {"start_beat": s.start_beat, "end_beat": s.end_beat, "label": s.label.value}
        for s in md.sections
    ]
    gt_sections = gt.get("sections", [])
    pred_repeats = [
        {"start_beat": r.start_beat, "end_beat": r.end_beat, "kind": r.kind}
        for r in md.repeats
    ]
    gt_repeats = gt.get("repeats", [])

    ts_gt = tuple(gt["time_signature"])
    tempo_gt = float(gt["tempo_bpm"])
    tempo_pred = md.tempo_map[0].bpm if md.tempo_map else 0.0

    return FixtureResult(
        slug=fixture_dir.name,
        title_match=_norm_str(md.title) == _norm_str(gt.get("title")),
        composer_match=_norm_str(md.composer) == _norm_str(gt.get("composer")),
        key_match=_norm_key(md.key) == _norm_key(gt["key_signature"]),
        time_sig_match=md.time_signature == ts_gt,
        tempo_within_5bpm=abs(tempo_pred - tempo_gt) <= 5,
        section_f1=_score_sections(pred_sections, gt_sections),
        repeat_f1=_score_repeats(pred_repeats, gt_repeats),
        predicted_title=md.title,
        predicted_composer=md.composer,
    )


async def _main(out_path: Path, fixtures_root: Path) -> int:
    if not os.environ.get("OHSHEET_ANTHROPIC_API_KEY"):
        print("ERROR: OHSHEET_ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 2
    if not fixtures_root.is_dir():
        print(f"ERROR: fixtures root not found: {fixtures_root}", file=sys.stderr)
        return 2

    blob_root = REPO_ROOT / "blob"
    blob = LocalBlobStore(blob_root)
    svc = RefineService(blob_store=blob)

    fixture_dirs = sorted(
        d for d in fixtures_root.iterdir()
        if d.is_dir() and (d / "input_score.json").is_file() and (d / "ground_truth.json").is_file()
    )
    if not fixture_dirs:
        print(f"ERROR: no fixtures found under {fixtures_root}", file=sys.stderr)
        return 2

    results = []
    for fd in fixture_dirs:
        print(f"scoring {fd.name} ...")
        res = await _eval_fixture(svc, fd)
        results.append(res)
        print(
            f"  title={res.title_match} composer={res.composer_match} "
            f"key={res.key_match} ts={res.time_sig_match} "
            f"tempo<=5bpm={res.tempo_within_5bpm} "
            f"sec_f1={res.section_f1:.2f} rep_f1={res.repeat_f1:.2f}"
        )

    n = len(results)
    aggregate = {
        "count": n,
        "title_exact_match_pct": sum(r.title_match for r in results) / n * 100,
        "composer_exact_match_pct": sum(r.composer_match for r in results) / n * 100,
        "key_match_pct": sum(r.key_match for r in results) / n * 100,
        "time_signature_exact_pct": sum(r.time_sig_match for r in results) / n * 100,
        "tempo_within_5bpm_pct": sum(r.tempo_within_5bpm for r in results) / n * 100,
        "section_label_f1_avg": sum(r.section_f1 for r in results) / n,
        "repeat_f1_avg": sum(r.repeat_f1 for r in results) / n,
    }
    report = {
        "aggregate": aggregate,
        "per_song": [
            {
                "slug": r.slug,
                "title_match": r.title_match,
                "composer_match": r.composer_match,
                "key_match": r.key_match,
                "time_signature_exact": r.time_sig_match,
                "tempo_within_5bpm": r.tempo_within_5bpm,
                "section_label_f1": r.section_f1,
                "repeat_f1": r.repeat_f1,
                "predicted_title": r.predicted_title,
                "predicted_composer": r.predicted_composer,
            }
            for r in results
        ],
    }
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out_path}")
    print(json.dumps(aggregate, indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "refine-baseline.json",
        help="Output JSON path (default: refine-baseline.json)",
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=REPO_ROOT / "eval" / "fixtures" / "refine_golden",
        help="Fixtures root (default: eval/fixtures/refine_golden/)",
    )
    args = parser.parse_args()
    rc = asyncio.run(_main(args.out, args.fixtures))
    sys.exit(rc)


if __name__ == "__main__":
    main()
```

### Step 3: Update pyproject and Makefile

- [ ] In `pyproject.toml`, extend the `[eval]` extra with `anthropic`:

```toml
eval = [
    "anthropic>=0.40",
    "mir_eval>=0.6.0",
]
```

(`anthropic` is already in `dependencies` from Task 3, but listing it here keeps `make install-eval` self-contained.)

- [ ] In `Makefile`, extend the help block and add the new target. Insert after the existing `eval:` target (around line 124):

```makefile
# Live refine eval. Runs the RefineService against
# eval/fixtures/refine_golden/ using the real Anthropic API.
# Costs real money; excluded from CI. Writes refine-baseline.json.
# Requires OHSHEET_ANTHROPIC_API_KEY.
eval-refine:
	@test -n "$$OHSHEET_ANTHROPIC_API_KEY" || (echo "OHSHEET_ANTHROPIC_API_KEY not set" && exit 2)
	python scripts/eval_refine.py --out refine-baseline.json
```

Also update the `help` block (around line 38) to mention the new target:

```makefile
	@echo "  make eval-refine        score RefineService against the refine_golden set (requires OHSHEET_ANTHROPIC_API_KEY)"
```

And update the `.PHONY` line (around line 19) to include `eval-refine`:

```makefile
.PHONY: help install install-backend install-basic-pitch install-demucs install-eval install-frontend backend frontend test test-backend test-e2e eval eval-refine lint typecheck clean require-flutter require-port-free
```

### Step 4: Smoke-test the harness without hitting the API

- [ ] Run the harness with the API key unset — it should exit cleanly with a clear error:

```bash
unset OHSHEET_ANTHROPIC_API_KEY
python scripts/eval_refine.py --out /tmp/refine-smoke.json
```

Expected stdout: `ERROR: OHSHEET_ANTHROPIC_API_KEY is not set.` Exit code 2.

- [ ] Optional: if you have an API key, run one real evaluation (costs a few cents):

```bash
export OHSHEET_ANTHROPIC_API_KEY=sk-...
make eval-refine
cat refine-baseline.json | python -m json.tool | head -30
```

Expected: aggregate block with non-zero percentages for at least title/composer on the Clair de Lune fixture.

### Step 5: Commit

- [ ] Commit the harness and fixtures:

```bash
git add eval/fixtures/refine_golden/ scripts/eval_refine.py pyproject.toml Makefile
git commit -m "feat(eval): live golden-set harness for refine stage"
```

---

## Self-Review Checklist

After implementing all tasks, run the following checks before merging:

- [ ] `make lint` — ruff + flutter analyze clean.
- [ ] `make typecheck` — mypy clean.
- [ ] `make test` — full suite green (live refine excluded by default: the `disable_real_refine_llm` autouse fixture nulls the API key, so `RefineService.run()` routes through its pass-through path on every test that doesn't explicitly inject a client).
- [ ] `pytest tests/test_contracts_refine.py tests/test_refine_prompt.py tests/test_refine_service.py tests/test_refine_runner.py tests/test_refine_engrave_metadata.py tests/test_refine_integration.py -v` — all refine-specific tests green.
- [ ] `unset OHSHEET_ANTHROPIC_API_KEY && python scripts/eval_refine.py` — exits with code 2 and a clear error.
- [ ] With API key: `make eval-refine` produces `refine-baseline.json` with non-zero Clair de Lune scores.
- [ ] `grep -rn 'SCHEMA_VERSION == "3.0.0"' backend/ shared/ tests/` — no lingering references to the old schema version.
- [ ] `grep -rn 'TODO\|TBD\|FIXME' backend/services/refine.py backend/services/refine_prompt.py backend/workers/refine.py` — no placeholders in new code.
- [ ] Frontend progress screen shows `refine` between `humanize` and `engrave` during a live job (manual QA via `make backend` + `make frontend`).

---

## Notes for implementers

1. **Anthropic SDK version.** The plan targets `anthropic>=0.40`, which exposes `AsyncAnthropic.messages.create(..., tools=[...])` and the `web_search_20250305` server tool. If an older version is resolved, the web_search tool will be rejected server-side. Pin accordingly.

2. **Retry semantics.** The service retries on transient-looking errors (`timeout`, `overloaded`, HTTP 5xx). Authentication / schema / 4xx errors are surfaced immediately via the fallback path — no point retrying those.

3. **PianoScore has no `quality` field.** For `sheet_only` runs, refine-stage fallback warnings have nowhere to land. This is by design — the pipeline still succeeds, and the logs capture the failure reason. If we later need user-visible warnings on `sheet_only`, add a `quality: QualitySignal` field to `PianoScore` in a follow-up schema bump.

4. **Cache is LocalBlobStore-only for v1.** The service's `_cache_uri` introspects `blob_store.root`. When the project grows an S3 store, add a parallel cache strategy (e.g., a small Redis index keyed by hash) without changing the service's public API.

5. **Golden-set fixtures are a seed, not a full set.** The plan ships with one fixture (Clair de Lune) so the harness runs end-to-end. Adding the remaining 10–15 is follow-up work — the `README.md` in `eval/fixtures/refine_golden/` describes how.
