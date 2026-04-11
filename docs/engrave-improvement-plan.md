# Engrave Stage — Improvement Plan

Synthesis of three parallel research passes on `backend/services/engrave.py`:
renderer infrastructure & output fidelity, musical notation quality, and
evaluation strategy. Scoped to measurable, incremental improvements that can
each land as a single PR.

## TL;DR

The engrave stage today is functional but leaks value on every axis:

- **Two latent correctness issues** — `timing_offset_ms` is applied to both
  onset and offset at `engrave.py:87-88` (duration-preserving shift — probably
  not what humanize intends), and `EngravedScoreData.includes_dynamics/pedal`
  at `engrave.py:515` claim features that are never actually rendered.
- **Data is being dropped on the floor.** `humanize.py:101-114` generates
  dynamics; engrave never reads `perf.expression.dynamics`. Pedal events
  reach MIDI but not MusicXML. Chord symbols are disabled entirely.
- **The OSMD sanitizer regex (`engrave.py:242-296`) is a workaround masquerading
  as infrastructure.** It collapses voices 1+2 (destroying piano stems-up/
  stems-down) and rescales divisions after the fact. Both problems are better
  fixed at the music21 export boundary.
- **Test coverage for engrave is ~zero.** No fixture-based golden tests,
  no schema validation, no MIDI round-trip. Every change is a shot in the
  dark.
- **Production emits stub PDFs.** The Cloud Run image has no MuseScore or
  LilyPond, so every job returns the 60-byte `%PDF-1.4` stub.

The right order is **build the evaluation harness first**, then fix the bugs
it surfaces, then ship the incremental notation/renderer improvements with
regression coverage on every PR.

---

## Phase 0 — Evaluation harness (land before any behavior changes)

Rationale: almost every improvement below is risky without a way to A/B old
vs. new output. Building the harness first means subsequent PRs are
diffable and reviewable.

### 0.1 Score fixtures — `tests/fixtures/scores/`
Commit ten small hand-authored `PianoScore` / `HumanizedPerformance` fixtures
as Pydantic-built JSON so they survive contract changes via re-validation:

1. `single_note.json` — RH C4 only (trivial baseline)
2. `c_major_scale.json` — RH scale + LH whole notes (beaming, single voice)
3. `two_hand_chordal.json` — RH triads + LH octaves (`<staff>` separation)
4. `bach_invention_excerpt.json` — 8 bars of two-voice counterpoint, RH
   only — **the voice-handling fixture**
5. `jazz_voicings.json` — chromatic bass + 7th-chord shells (accidentals)
6. `seven_eight.json` — 7/8 irregular grouping (time-sig propagation,
   `quarterLengthDivisors=(4,3)` edge cases)
7. `tempo_change.json` — multi-segment `tempo_map`
8. `humanized_with_offsets.json` — c_major_scale with
   `timing_offset_ms ∈ [-30,30]` and a sustain pedal event — **the timing-bug
   fixture**
9. `empty_left_hand.json` — RH only (catches no-note backup logic)
10. `overlapping_same_pitch.json` — two RH C4 notes that overlap (exercises
    `engrave.py:94-103` overlap resolver)

Promote the `_piano_score()` / `_humanized_performance()` builders from
`tests/test_stages.py` into a shared `tests/fixtures/_builders.py` +
`load_score_fixture(name)` helper.

### 0.2 Test layers, in priority order

| Layer | Catches | Runs |
|---|---|---|
| **L1 — MIDI round-trip** | `timing_offset_ms` bug, dropped notes, wrong pitches | Every PR |
| **L2 — Notation quality lints** (lxml xpath) | `<voice>` out of range, divisions > 480, out-of-piano-range pitches, note-count mismatch | Every PR |
| **L3 — MusicXML 4.0 XSD validation** | Structural schema violations | Every PR |
| **L4 — MusicXML golden diffs** | Regressions in music21 output after normalization | Every PR |
| **L5 — Verovio render check** | MusicXML that validates but won't render; missing measure attributes | Every PR |
| **L6 — OSMD/Playwright headless** | `parentVoiceEntry` class crashes in actual consumer | **Nightly only** |
| **L7 — MuseScore/LilyPond PDF** | Real engraver feedback | **Manual / nightly**, not PR-blocking |

Layers L1–L5 are all cheap (sub-second each) and add <10s to CI. Dev deps
to add to `pyproject.toml`: `lxml` (already transitive), `verovio` (~10 MB
pure-Python wheel, no system deps). Vendor `musicxml-4.0.xsd` into
`tests/fixtures/`.

**Skip** MuseScore/LilyPond in PR CI (500 MB+ images, nondeterministic
output, X-server hassles). Reserve for optional nightly on a beefier runner.

### 0.3 Golden-file normalization
Before diffing MusicXML goldens, strip time-varying tags: `<encoding-date>`,
`<software>`, `<encoder>`, `<creator type="software">`, `<supports>`. Use
`lxml.etree.canonicalize`. Ship a `pytest --update-goldens` flag from day
one so music21 version bumps are a reviewable diff, not a mystery churn.
Pin music21 exactly in `pyproject.toml`.

### 0.4 A/B regression harness — `scripts/engrave_diff.py`
Takes two git refs, runs all fixtures through both, prints a table of
MusicXML normalized-byte-diff sizes and Verovio note counts per fixture.
Not a test — an investigative CLI for PR authors.

**Expected first finding:** the failing `humanized_with_offsets` MIDI
round-trip test surfaces the `timing_offset_ms` behavior so we can make an
informed decision (is it a bug, or is it the intended "whole-note shift"
model?).

---

## Phase 1 — Bug fixes and truth-telling (small PRs, low risk)

### 1.1 Truthful `EngravedScoreData` flags
`engrave.py:512-526` currently sets `includes_dynamics=is_humanized` and
`includes_pedal_marks=is_humanized` unconditionally. Change to reflect
actual rendered content:

```python
includes_dynamics=bool(perf and perf.expression.dynamics),
includes_pedal_marks=bool(perf and perf.expression.pedal_events),
```

**Effort: S. Risk: none.** Land independently before 1.3/1.4 because those
will flip these flags to actually-True.

### 1.2 `timing_offset_ms` semantics decision — `engrave.py:87-88`
The shift is applied to *both* onset and offset, so duration is preserved
(it's a pure translation, not a stretch). Whether that's correct depends on
humanize's intent — does `timing_offset_ms` mean "nudge the whole note" or
"nudge the onset only"? Check `backend/services/humanize.py` and the schema
docstring on `ExpressiveNote`. Two possible fixes:

- **If onset-only:** apply the offset only to `onset` and let `offset`
  absorb a duration change, OR introduce a separate release-offset field.
- **If whole-note shift:** current code is correct; document it and close
  the issue.

The L1 MIDI round-trip test (fixture 8) will fail either way; resolving it
forces the decision to be made explicit.

**Effort: S (after decision). Risk: low — test will pin behavior.**

### 1.3 Render dynamics from `perf.expression.dynamics`
Currently zero lines reference `perf.expression.dynamics` in engrave, even
though humanize populates them at `humanize.py:101-114`. In the RH part
loop (`engrave.py:186-227`), after `part.append(clef)`, walk
`perf.expression.dynamics`:

- `pp/p/mp/mf/f/ff` → `music21.dynamics.Dynamic(type)` inserted at `dyn.beat`
- `crescendo/decrescendo` → `music21.dynamics.Crescendo()` / `Diminuendo()`
  spanners using `span_beats` for the end anchor

Attach to RH only so markings sit between staves.

**Effort: S. Risk: low.** Flip flag from 1.1 after this lands.

### 1.4 Render pedal markings in MusicXML
`engrave.py:111-124` writes CC64 to MIDI only. Add pedal marks to the LH
part via `music21.expressions.TextExpression("Ped.")` at
`pedal.onset_beat` and `"*"` at `pedal.offset_beat` (falling back from
`music21.expressions.PedalMark`, whose support varies across music21
versions).

**Effort: S. Risk: low — text directions always work.**

### 1.5 Emit sostenuto (CC66) and una_corda (CC67)
`engrave.py:111-124` hard-codes `pedal.type == "sustain"`. The
`PedalEvent.type` literal in `shared/shared/contracts.py` already supports
`sostenuto` and `una_corda`. Add two more branches mapping to CC66/CC67.

**Effort: S. Risk: none.**

### 1.6 Handle fermata articulation
`engrave.py:203-208` handles staccato/accent/tenuto but drops `fermata`
from the `Articulation.type` literal. Add:
`music21.expressions.Fermata()` appended to the note.

Skip `legato` for now — slurs need spanner endpoints that humanize doesn't
yet emit.

**Effort: S. Risk: none.**

---

## Phase 2 — Renderer infrastructure (medium PRs)

### 2.1 Install LilyPond in the production image — **biggest user-visible win**
`Dockerfile` currently installs only ffmpeg on `python:3.12-slim`, so
`_render_pdf_bytes` always returns the 60-byte stub in production. Options
ranked:

| Option | Size add | Quality | Complexity |
|---|---|---|---|
| **LilyPond via apt** | ~250 MB | Excellent engraving, but `musicxml2ly` is lossy (loses chord symbols, some articulations) | S — 3 lines in Dockerfile |
| MuseScore 4 headless | ~600–900 MB | Highest fidelity | M — Xvfb + Qt deps, 2–5s cold start |
| Verovio → SVG → cairosvg → PDF | ~10 MB | Modern engraving, MusicXML 4.0 native | M — new code, page-break logic |
| Server-side OSMD via Playwright | ~300 MB | Visual parity with web viewer | L — Chromium, 1–3s/page, fragile |

**Recommendation: LilyPond first.** Smallest delta from current state,
ships real PDFs tomorrow. Verovio is the more attractive long-term play
(smallest image, modern output) but is M-sized new code; schedule as a
follow-up if LilyPond's `musicxml2ly` loses notation we care about.

**Effort: S. Risk: image size budget review.**

### 2.2 Hard-require music21; delete `_minimal_musicxml`
The `_minimal_musicxml` fallback at `engrave.py:299-358` is "not bar-aware"
per its own docstring and would produce unrenderable MusicXML if it ever
ran in prod. The bare `except Exception` at `engrave.py:237` masks music21
errors and would silently fall through to this path. Music21 is already
required in practice — promote it in `pyproject.toml`, delete lines
299–358 and the import-error branches at 156–160, and let exceptions
propagate to the job manager so failures are visible.

**Effort: S. Risk: verify no dev environment actually runs without
music21. Install size is ~30 MB.**

### 2.3 PianoStaff grouping — one part with `<staves>2</staves>`
Currently each hand is a separate `<part>` named "Right Hand" / "Left
Hand". Correct piano notation is one `<part>` with two staves and a brace.
In music21: two `PartStaff` objects wrapped in
`music21.layout.StaffGroup(..., symbol='brace', barTogether=True)`. See
how `_minimal_musicxml` already structures this at `engrave.py:344-346`.

**Effort: S (~10 lines around `engrave.py:186-227`). Impact: visible
improvement — real grand staff instead of two stacked instruments.**

### 2.4 Set `divisionsPerQuarter` upfront; retire the regex sanitizer's
divisions branch
The regex sanitizer (`engrave.py:242-296`) exists because music21 picks
`divisions=10080` and OSMD chokes. Fix it at the source by calling
`music21.musicxml.m21ToXml.ScoreExporter(score).parse()` with an explicit
`divisionsPerQuarter=4`, or by invoking `makeNotation(inPlace=True)` at
the score level with `quarterLengthDivisors` set. This makes durations
correct *by construction* instead of by integer-rounded ratios (which
silently truncates anything finer than a 16th).

Keep the voice-collapse regex for now; remove it in 3.1.

**Effort: M. Risk: scoring quirks for tuplets. Mitigation: gate behind a
feature flag and A/B against fixtures 2, 4, 6 using the harness from
0.4.**

### 2.5 Multi-tempo MIDI export
`engrave.py:69` reads only `tempo_map[0].bpm`. The schema supports
multi-segment tempo maps. `pretty_midi` doesn't expose mid-piece tempo
writes cleanly; drop to `mido.MidiFile` with `MetaMessage('set_tempo', ...)`
events. This currently has no user-visible impact because upstream only
emits constant tempo — ship only after arrange/humanize start producing
real tempo maps.

**Effort: S–M. Risk: low. Defer until upstream actually varies tempo.**

---

## Phase 3 — Notation quality (medium-to-large PRs)

### 3.1 Preserve voices 1 and 2 per staff
The biggest visual-readability improvement in this plan. Piano notation is
*defined* by two voices per staff (stems up = melody, stems down =
accompaniment). The regex at `engrave.py:294` collapses everything to
voice 1, destroying music21's voice output.

Two-PR sequence:
1. Cap `MAX_VOICES_RH` / `MAX_VOICES_LH` at 2 in `arrange.py:40-41`
   (arrange currently allows 4/3 via greedy first-fit in
   `arrange.py:194-210`).
2. Change the OSMD sanitizer to clamp voices ≥3 to 2 instead of
   collapsing everything to 1; set `n.voice = sn.voice` before insertion
   in `engrave.py:209` so music21 honors the assignment.

OSMD reliably handles 2 voices per staff — only 3+ are buggy. Validate
with fixture 4 (`bach_invention_excerpt`) before merging.

**Effort: M. Risk: M — if there really are 2-voice OSMD bugs, nightly
Playwright test (L6) will catch them before release.**

### 3.2 Chord-symbol cleanup + gated re-enable
Currently disabled at `engrave.py:211-220` because transcription produces
noisy labels like "G5", "E5" (pitch names, not chord qualities). Filter
design:

- Reject any label matching `^[A-G][#b]?\d+$` (those are pitch-and-octave,
  not chord qualities)
- Require `music21.harmony.ChordSymbol(label)` parses cleanly AND
  `len(chord.pitches) >= 3`
- Gate by `RealtimeChordEvent.confidence` — but note
  `arrange.py:378-389` drops confidence when converting to
  `ScoreChordEvent`. **One-line contract change:** add
  `confidence: float` to `ScoreChordEvent` in
  `shared/shared/contracts.py:205-209` and propagate.

Wrap each `ChordSymbol` construction in try/except; the parser is
finicky.

**Effort: M. Risk: M — contract change affects serialization.**

### 3.3 Key signature verification via Krumhansl-Schmuckler
`engrave.py:171-172` trusts `score.metadata.key` blindly. If transcription
says `"C:major"` for a piece that's actually F# minor, every accidental
explodes. music21 ships `analysis.discrete.KrumhanslSchmuckler` — a
~5-line check. If analyzer confidence is high and disagrees with metadata,
override and log.

**Effort: M. Risk: low (override is gated on high confidence).**

### 3.4 Quantization grid improvements
`engrave.py:225` uses `quarterLengthDivisors=(4, 3)` (16ths + triplets).
Two mutually-exclusive directions:

- **Expand divisor tuple tempo-adaptively:** add 8 (32nds) and 6
  (sextuplets) at `bpm < 80`; drop to `(4, 3)` only above ~140. Simple
  but noisy transcriptions fragment into 32nds.
- **Better:** remove the re-quantize entirely. arrange already quantizes
  via `_estimate_best_grid` (`arrange.py:67-97`); engrave only
  re-quantizes because the OSMD sanitizer collapses divisions to 4. Once
  2.4 lands, engrave can trust arrange's grid.

**Effort: S–M depending on direction. Blocked on 2.4.**

### 3.5 System breaks at section boundaries
Insert `music21.layout.SystemLayout(isNew=True)` at `ScoreSection`
boundaries so each verse/chorus starts on a new system. Polish item —
ship only if users request it.

**Effort: S. Impact: low. Skip until requested.**

---

## Phase 4 — Long-horizon / spikes

### 4.1 Verovio frontend swap (OSMD → Verovio)
The entire `_sanitize_musicxml_for_osmd` function (`engrave.py:242-296`)
exists because OSMD's VexFlow backend is limited. Verovio supports
MusicXML 4.0, unlimited voices, and complex tuplets. Bundle size is ~3–5
MB (vs OSMD's ~1.5 MB), but the sanitizer disappears and phase 3 becomes
trivial. **Scope:** Flutter web viewer rewrite in
`frontend/lib/widgets/sheet_music_viewer_web.dart`.

**Effort: L. Do this as a dedicated spike after phases 0–2 land.**

### 4.2 Fingering via `pianoplayer`
`pianoplayer` (pip) consumes MusicXML, runs hand-physics search, writes
fingerings back. Integration point: after `s.write("musicxml", ...)` at
`engrave.py:232`, shell out and re-read. Adds numpy + scipy dependencies
(~50 MB) and a few-second runtime hit. Worth it only if users want
practice-grade scores; `EngravedScoreData.includes_fingering` is
currently hardcoded `False`.

**Effort: ML–L. Schedule after user demand.**

### 4.3 Voice separation with pitch-continuity bias
arrange's voice allocator (`arrange.py:194-210`) is overlap-avoidance,
not musical voice separation. Phase 1: bias voice choice toward the voice
whose last note is closest in pitch. Phase 2: investigate Temperley's
Streamer or music21's `makeVoices`. **Belongs in arrange, not engrave.**

---

## Recommended PR sequence

1. **PR-1 (Phase 0.1–0.2):** Score fixtures + L1 MIDI round-trip + L2
   notation lints. Lands with one known-failing test (`humanized_with_offsets`).
2. **PR-2 (Phase 0.3–0.4):** L3 XSD + L4 goldens + L5 Verovio + A/B CLI.
3. **PR-3 (Phase 1.1):** Truthful `EngravedScoreData` flags.
4. **PR-4 (Phase 1.2):** Resolve `timing_offset_ms` semantics; turn the
   failing test green.
5. **PR-5 (Phase 1.3–1.6):** Dynamics + pedal marks + CC66/CC67 + fermata.
   One combined "stop dropping data" PR.
6. **PR-6 (Phase 2.1):** LilyPond in Dockerfile. Real PDFs in prod.
7. **PR-7 (Phase 2.2):** Delete `_minimal_musicxml`; hard-require music21.
8. **PR-8 (Phase 2.3):** PianoStaff grouping / grand staff.
9. **PR-9 (Phase 2.4):** `divisionsPerQuarter` upfront; retire divisions
   regex branch.
10. **PR-10 (Phase 3.1):** Cap arrange voices at 2 + preserve 2 voices in
    engrave.
11. **PR-11 (Phase 3.2):** Chord symbol cleanup + contract tweak + re-enable.
12. **PR-12 (Phase 3.3):** Key-signature verification.
13. **PR-13 (Phase 3.4):** Quantization grid rework (only after PR-9).

PRs 1–2 unlock everything else. PRs 3–6 are all S-sized and land in the
first week. PRs 7–13 are each reviewable against the harness.

---

## Key file:line references

- `backend/services/engrave.py:87-88` — `timing_offset_ms` applied to both
  onset and offset
- `backend/services/engrave.py:111-124` — sustain-only pedal filter,
  MIDI-only output
- `backend/services/engrave.py:162-227` — music21 score build (no dynamics,
  no pedal marks)
- `backend/services/engrave.py:211-220` — disabled chord symbols
- `backend/services/engrave.py:225` — `quarterLengthDivisors=(4, 3)`
- `backend/services/engrave.py:237` — bare `except Exception`, masks
  music21 errors
- `backend/services/engrave.py:242-296` — OSMD regex sanitizer
- `backend/services/engrave.py:294` — voice-collapse-to-1
- `backend/services/engrave.py:299-358` — `_minimal_musicxml` (not
  bar-aware, latent footgun)
- `backend/services/engrave.py:378-421` — PDF rendering (stubs in prod)
- `backend/services/engrave.py:512-526` — `EngravedScoreData` flags that
  lie
- `backend/services/humanize.py:101-114` — dynamics generated then dropped
- `backend/services/arrange.py:40-41` — `MAX_VOICES_RH=4`,
  `MAX_VOICES_LH=3`
- `backend/services/arrange.py:194-210` — greedy voice allocator
- `backend/services/arrange.py:378-389` — chord confidence dropped on
  conversion
- `shared/shared/contracts.py` — `ScoreChordEvent` lacks `confidence`;
  `PedalEvent.type` already supports sostenuto/una_corda;
  `ExpressionMap.dynamics` already supports crescendo/decrescendo
- `Dockerfile` — no MuseScore/LilyPond in production image
- `tests/test_stages.py` — existing `_piano_score()` /
  `_humanized_performance()` builders, promote to fixture loader
- `tests/conftest.py` — reuse `isolated_blob_root` fixture

---

## What this plan does *not* do

- No speculative refactors of the engrave service structure. The file is
  ~500 lines and single-purpose; it doesn't need to be split.
- No runtime performance work. engrave is not on any hot path.
- No new file format outputs (ABC, MEI, etc.) — users want PDF + MIDI.
- No upstream pipeline changes except the minimal ones called out
  (voice caps in arrange, `confidence` on `ScoreChordEvent`).

## Open questions for follow-up

1. Does `timing_offset_ms` mean "nudge the whole note" or "nudge the
   onset only"? Requires a read of `humanize.py` and a product-intent
   call.
2. Is music21 actually an optional dep or already required? Check
   `pyproject.toml` before PR-7.
3. What's the image size budget for Cloud Run? LilyPond adds ~250 MB —
   need a deploy-side yes/no before PR-6.
4. Is the frontend team open to a Verovio spike, or is OSMD a hard
   constraint? Determines whether Phase 4.1 is even worth scoping.
