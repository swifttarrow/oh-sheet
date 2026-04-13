# Feature Research

**Domain:** LLM-augmented music notation refinement (post-transcription cleanup stage in a piano sheet music pipeline)
**Researched:** 2026-04-13
**Confidence:** MEDIUM

## Ecosystem Context

The "refine a generated PianoScore with an LLM" pattern has no single dominant OSS reference. The closest neighbours are:

- **End-to-end audio-to-sheet products** (AnthemScore, ScoreCloud, Klangio, Melody Scanner, PlayScore 2, SCANSCORE). All treat notation cleanup as a manual post-step in their own editor, not an automated LLM stage. None expose a programmatic "clean up this MusicXML" API.
- **LLM-native music generation research** (ChatMusician, NotaGen, MIDI-LLM, YNote/HNote). Focus on generation from prompts, not refinement of existing scores. Most operate in ABC rather than MusicXML because it's LLM-tokenizer-friendly.
- **Generic LLM-on-structured-content patterns** (Cursor's two-model Apply pattern, the "edit trick", Anthropic Structured Outputs, Langfuse observability, the Eleuther lm-evaluation-harness). These are where the real playbook for this milestone comes from — the music-notation-specific literature is sparse, so we borrow from LLM application engineering wholesale.

The honest read: this milestone is a **music-specific application of general LLM pipeline hygiene**, not a music ML problem with prior art. Most table-stakes features below come from the "how to safely run an LLM inside a deterministic pipeline" literature, and are anchored to the concrete transcription artifacts catalogued in CONCERNS.md.

## Feature Landscape

### Table Stakes (Users Expect These)

Non-negotiable. Missing any of these and the refine stage is worse than skipping it.

| Feature | Why Expected | Complexity | Artifact / Gotcha Anchor |
|---------|--------------|------------|--------------------------|
| **Structured output contract (Pydantic / JSON schema)** | Refine must return a `PianoScore`-shaped object the `engrave` stage can consume. LLMs without schema enforcement produce unparseable output ~1-5% of the time; structured outputs (Claude Sonnet 4.5+, Nov 2025) guarantee schema compliance in one shot. | LOW | Contract reuse: `shared/shared/contracts.py` defines `PianoScore`; engrave expects that shape. Without schema enforcement, every malformed response is a job failure. |
| **Web-search grounding of song identity** | Generated transcription has default-fallback key `"C:major"` and meter `(4,4)` (CONCERNS.md: "Time-signature and key detection are hardcoded defaults", `backend/services/ingest.py`, `backend/jobs/runner.py:158-169`). LLM with web search can look up the real key/meter from lyrics/tab/fan sites. Without grounding, the LLM just re-guesses from the same flawed evidence. | MEDIUM | Anthropic web search tool is native to the API; uses Brave Search; citations included. $10/1K searches + standard token costs. |
| **Modify + delete authority only (no note addition)** | Prevents LLM from "filling in" perceived gaps, which would defeat the whole point of grounding-in-evidence. CONCERNS.md flags ghost notes from Basic Pitch — we want the LLM to *remove* those, not imagine new ones. Linear issue explicitly calls this out. | MEDIUM | Implementation: system prompt constraint + output validator that diffs refined vs. original and rejects any note whose (pitch, onset) was not in the original. Validator is cheap; prompt alone is not enough (LLMs ignore constraints under pressure). |
| **Skip-on-failure with `refine_skipped` event** | LLM stage adds a new external dependency (Anthropic API). Rate limits, 429s, 503s, timeouts happen. PROJECT.md Active line 39 requires: *PipelineRunner skips refine and proceeds to engrave on LLM failure — job succeeds with unrefined output.* Without this, Anthropic outages turn into product outages. | LOW | Anthropic docs recommend retry-after → reset-header → jittered exponential backoff + circuit breaker. Read timeout ≥120s for outputs >2K tokens. Our failure mode is "log and skip", which sidesteps most of this — we just need bounded retry + circuit breaker. |
| **Opt-in per-job toggle (`enable_refine: bool`, default false)** | PROJECT.md constraint: keeps default latency unchanged, avoids cost/behavior surprise. Without this, every existing user's job suddenly gets 5-30s slower and pays LLM cost. Also essential for A/B harness (same codepath, different flag). | LOW | Plumbing: `JobCreateRequest` → `PipelineConfig` → `PipelineRunner` execution plan selection. Frontend adds a checkbox on upload screen. No new pipeline variant string. |
| **Output persisted as separate blob artifact** | PROJECT.md Active line 38 requires `jobs/{job_id}/refine/output.json`. Debuggability: when a refine output produces weird engraved notation, we need the raw LLM output + prompt to diagnose. CONCERNS.md: lack of stage-level artifact visibility ("Pipeline stage failures are not isolated") is already a known pain point; refine shouldn't compound it. | LOW | Matches existing claim-check URI pattern (`jobs/{job_id}/{stage}/output.json`). BlobStore already supports put_json. |
| **Input validation before LLM call (schema + size)** | LLM sees whatever we send. If `HumanizedPerformance` has corrupted data (CONCERNS.md: "stage input is deserialized from JSON without type checks on mutation paths"), we waste tokens on bad input and get bad output back. Pydantic re-validation at stage entry catches this cheaply. | LOW | Pydantic v2 `model_validate()` on read-from-blob is ~1ms. Run before prompt construction. Reject with actionable error if shape is broken. |
| **Output validation against `PianoScore` schema + music21 parse** | Two-layer validation: (a) Pydantic deserialization catches structural errors; (b) try `music21.converter.parse()` or equivalent to catch semantically-invalid scores (negative durations, pitches outside MIDI range, voices >2 per hand). Existing concern: "voice-cap silently drops notes >2 per hand" (CONCERNS.md, `backend/services/arrange.py:47-48`) — if refine re-introduces >2 voices, engrave downstream silently drops them. Validate at refine exit. | MEDIUM | Research: ChatMusician and GPT-4 achieve >90% ABC parseability; both music21 and direct Pydantic model validation are fast. |
| **Environment-driven secret handling (`OHSHEET_ANTHROPIC_API_KEY`)** | Follows existing `backend/config.py` convention. Without this, fails open (key in code, fails silently at runtime, or worse, logged). PROJECT.md Constraints line 74: "never logged; same handling as other secrets". | LOW | Pydantic Settings already supports this. `fail-fast at startup if enable_refine flag is on globally but key missing` protects against misconfigured deploys. |
| **Stage lifecycle events (`stage_started / progress / completed / skipped`)** | PROJECT.md Active line 40: WebSocket JobEvent stream must surface refine stage events. Existing `JobManager` fan-out already supports this; new stage needs to emit the same events as ingest/transcribe/etc. Without events, frontend shows a 30-second freeze with no feedback. | LOW | Add `refine` to the stage-name → label map in `frontend/lib/`. Emit from Celery task wrapper in `backend/workers/refine.py`. |
| **Bounded retry with jittered exponential backoff** | Anthropic production-reliability guidance: respect `Retry-After` → reset headers → jittered exponential backoff. Without it, one rate-limit burst cascades into N failing jobs, all retrying in lockstep (thundering herd). Full jitter breaks synchronization. | LOW | Celery task `autoretry_for=(ApiStatusError,)`, `retry_backoff=True`, `retry_backoff_max=60`, `retry_jitter=True`, `max_retries=3`. Then skip-on-failure kicks in. |
| **Observability: prompt, completion, token counts, latency logged** | LLM calls are opaque by default. When a refine produces a surprising output, we need the exact prompt+completion+tool-calls to diagnose. Langfuse is the OSS industry default (MIT, native Anthropic cost tracking). CONCERNS.md already flags observability gaps ("Queue-full warnings are silent", generic stage-failed events). | MEDIUM | v1 can be logging-only (Python `logging` with structured JSON), but we should design the interface so a Langfuse client can be plugged in later without refactoring. |

### Differentiators (Competitive Advantage)

Features where this can be *the* reference design for "LLM stage inside a deterministic pipeline". Aligned with Core Value ("accuracy matters more than speed/fidelity/breadth").

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **A/B harness with LilyPond-output diff + regression flagging** | PROJECT.md Active line 44 requires this. Competitors like AnthemScore/ScoreCloud have no programmatic eval — they rely on user manual editing as the "validation". Our harness takes N reference songs, runs with/without refine, diffs the LilyPond source (stable, deterministic, text), flags regressions. Eleuther-style evaluation harness for a music pipeline. | MEDIUM-HIGH | Design: red/green per-song table; hard-floor regressions (refine broke something) vs. drift alerts (refine degraded something). Reuse existing `GAU-59` e2e test fixtures as seed corpus. Deferred: scored rubric (PROJECT.md Out of Scope). |
| **Modify+delete-only authority enforced via output validator** | Most LLM-music-generation projects operate in "free generation" mode (ChatMusician, NotaGen, MIDI-LLM). Explicitly constraining the LLM to the transcription's evidence — and *verifying* post-hoc with a diff validator that rejects note additions — is genuinely novel as a design stance. Prevents the #1 failure mode ("LLM hallucinated a chorus") that kills trust. | MEDIUM | Implementation: a pure-Python validator that diffs `set(refined.notes)` vs `set(original.notes)` on a `(pitch, onset_bin)` key and rejects any additions. Cheap, deterministic, 100% coverage. Pair with a prompt constraint so the LLM rarely trips it. |
| **Inspectable refine artifact + replay** | PROJECT.md Active line 38: refine output persisted to `jobs/{job_id}/refine/output.json`. Differentiator because most LLM-in-pipeline products don't expose intermediate state — you either trust the output or you don't. We expose it; future debugging / fine-tuning / golden-set curation all benefit. Aligns with existing claim-check pattern. | LOW | Also enables "re-engrave from refined output without re-running LLM" future workflow. |
| **Opt-in UX end-to-end plumbed (UI → API → runner)** | Competitors either (a) default-on for every user (ScoreCloud, Klangio AI cleanup) or (b) purely manual (AnthemScore editor). Our "default-off, one checkbox away, per-job" model respects user agency, controls cost, and makes the feature testable. PROJECT.md Key Decisions row: "avoid surprise cost / surprise behavior changes for existing users". | MEDIUM | Depends on JobCreateRequest plumbing. Coordinate Python + Dart + WebSocket event names in one milestone. |
| **Variant-aware stage placement (after humanize *or* after arrange)** | PROJECT.md Active line 36: "runs between `humanize` and `engrave` (and after `arrange` for the `sheet_only` variant)". Nontrivial — most LLM integrations are placed in a single pipeline slot. Respecting the four existing pipeline variants keeps the sheet_only path useful (no humanization to refine, but arrangement artifacts still need cleanup). | LOW-MEDIUM | Implementation: extend `PipelineRunner.STEP_TO_TASK` and the four per-variant execution plans. Straightforward but requires touching every variant. |
| **Music21-parse validation gate before passing to engrave** | Catches semantically-invalid LLM output *before* LilyPond subprocess call (which is slow, fails cryptically, and can hang — CONCERNS.md: "LilyPond PDF rendering is subprocess-based and single-threaded"). Cheap safety net. | LOW | `music21.converter.parseData(xml_string, format='musicxml')` — exceptions on parse = invalid output = skip refine, use unrefined. |

### Anti-Features (Commonly Requested, Often Problematic)

Explicitly NOT building in this milestone. Each is already in PROJECT.md "Out of Scope" for a reason.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **LLM response caching by song-hash** | Obvious cost-saver for repeat submissions. | (a) Requires a stable hash over the refine input; the input contains float-valued humanized timings that are seed-dependent, so naïve hashing misses cache. (b) Adds a new infrastructure dependency (cache store) that's unused until traffic justifies it. (c) PROJECT.md Out of Scope explicitly defers until cost justifies. | Revisit after A/B harness shows cost curve; cache on (song_title, artist, variant) only when we have repeat submissions. |
| **Streaming partial refine results** | UX nicety — show "refined section 1 of 4…" progress. | Refine is a whole-score correctness pass; partial output is semantically meaningless (half a key signature, half a voice assignment). PROJECT.md Out of Scope: "refine returns the whole refined score atomically". Streaming would also fight against structured outputs (which are inherently atomic). | Use existing `stage_progress` events with stage-level granularity only. Let the job succeed atomically. |
| **Local / self-hosted LLM (Ollama, vLLM)** | Cost, privacy, offline operation. | Requires a multi-provider abstraction (SDK swap, structured-output portability, web-search-tool portability). Ollama quality on music reasoning is currently poor (ChatMusician-quality models are not mainstream in local inference). PROJECT.md Out of Scope. | Anthropic SDK only for v1. If multi-provider becomes necessary later, the service-interface boundary is the right seam. |
| **Multi-language song metadata handling** | Users submit CJK, Spanish, Japanese songs. | Web-search grounding degrades sharply outside English (fan sites, tab archives, lyrics sites skew English-dominant). Prompt engineering for non-English is a separate research effort. PROJECT.md Out of Scope: "English-language songs first". | Detect language from `InputBundle.metadata.title`; skip refine with `refine_skipped_reason=unsupported_language` event if non-English. Revisit as a future milestone. |
| **Scored eval rubric + golden set** | "Proper" eval hygiene. Defensible quality claims. | Premature optimization: we don't have a stable refine output to score yet. Building a rubric before the refine stage exists is cart-before-horse. PROJECT.md Out of Scope. | A/B harness (diff-based regression detection) is the v1 tool. Golden-set rubric becomes a v2 milestone once refine output stabilizes. |
| **LLM adding notes to the score** | Obvious "fill in what the transcription missed" appeal. | Defeats grounding: the LLM has no acoustic evidence for notes that weren't transcribed; it would invent notes to fit a chord progression. Breaks user trust the first time a ghost chorus appears. PROJECT.md Key Decision: "modify + delete authority only". | Output validator rejects any `(pitch, onset_bin)` tuple not present in original. If users need "invention," that's a composition tool, not a refinement tool. |
| **Default-on rollout** | Make every user benefit immediately. | Unproven quality + unmeasured cost = high risk. PROJECT.md Out of Scope: "opt-in until quality is proven on the A/B harness". | Gate default-on behind A/B harness passing a regression floor (e.g., zero hard regressions on seed corpus). Separate milestone. |
| **Cross-cutting codebase audit fixes** | CONCERNS.md catalogs many issues (persistence, auth, rate limiting, path traversal, cover search fragility). | Each is its own milestone. Bundling them here dilutes focus and makes the refine stage untestable in isolation. PROJECT.md Out of Scope explicitly scopes this milestone to refine only. | Tracked as separate future milestones; this milestone stays focused on the refine stage. |

## Feature Dependencies

```
enable_refine toggle (UI)
    └──requires──> JobCreateRequest.enable_refine plumbing
                       └──requires──> PipelineConfig.enable_refine plumbing
                                          └──requires──> PipelineRunner execution-plan branching
                                                             └──requires──> STEP_TO_TASK["refine"] registration

OHSHEET_ANTHROPIC_API_KEY (config)
    └──requires──> backend/config.py Settings field
                       └──requires──> startup validation (fail-fast if feature enabled without key)
                                          └──enables──> RefineService construction

Structured output contract
    └──requires──> PianoScore re-export or RefinedPianoScore subtype
                       └──must-be-accepted-by──> EngraveService (contract compatibility)

Web-search grounding
    └──requires──> Anthropic tool_use API with web_search tool
                       └──enhances──> key/meter correction (fixes default-fallback bug)
                                          └──validates-against──> output validator

Modify+delete validator
    └──requires──> Original (humanized) score read alongside refined output
                       └──rejects──> Any note addition
                                          └──on-reject──> refine_skipped fallback

Skip-on-failure
    └──requires──> refine_skipped event schema
                       └──requires──> PipelineRunner fallthrough to engrave on error
                                          └──requires──> engrave accepting HumanizedPerformance (already does)

A/B harness
    └──requires──> opt-in toggle (reuses enable_refine flag)
    └──requires──> deterministic LilyPond output (already exists; seed fixed)
    └──requires──> seed corpus (reuse GAU-59 e2e fixtures)

WebSocket refine events
    └──requires──> frontend stage-name → label map update
    └──requires──> refine_skipped event wired into progress screen UI

Inspectable refine artifact
    └──requires──> BlobStore put_json at jobs/{job_id}/refine/output.json
                       └──enables──> future replay / re-engrave workflow

Observability
    └──enhances──> All above; optional at v1, design seam now
```

### Dependency Notes

- **enable_refine toggle requires JobCreateRequest plumbing:** end-to-end change touching Dart frontend, FastAPI route, Pydantic model, PipelineConfig, and PipelineRunner. All one milestone.
- **Output validator depends on reading the original alongside the refined score:** the Celery task must pull both `humanize/output.json` and the new `refine/output.json` inputs to perform the diff. Matches the claim-check pattern.
- **Skip-on-failure depends on engrave accepting HumanizedPerformance:** engrave already does (it's the current default flow). Refine is a strict superset of the no-refine path.
- **A/B harness depends on deterministic base pipeline:** `humanize_seed=42` is actually useful here (CONCERNS.md flags it as a bug for user-perceived variation, but for A/B testing it's a feature — same seed with/without refine = comparable outputs).
- **Web-search grounding enhances but doesn't block key/meter correction:** if web search fails, the LLM can still attempt correction from transcription evidence alone (just with less confidence). Design for graceful degradation within the refine stage itself.

## MVP Definition

### Launch With (v1) — This Milestone

Minimum viable refine stage that's actually shippable.

- [ ] `refine` Celery stage registered in `STEP_TO_TASK` and per-variant execution plans (after humanize for full/audio/midi variants, after arrange for sheet_only)
- [ ] `RefineService` invoking Anthropic Claude with web-search tool, structured output against `PianoScore` schema
- [ ] System prompt enforces modify+delete-only authority; output validator diffs and rejects additions
- [ ] Music21 parse gate validates refined output before passing to engrave
- [ ] `enable_refine: bool` toggle plumbed end-to-end (Dart → JobCreateRequest → PipelineConfig → PipelineRunner)
- [ ] `OHSHEET_ANTHROPIC_API_KEY` loaded via pydantic-settings; startup validation when toggle is globally enabled
- [ ] `refine_skipped` JobEvent + `stage_started / progress / completed` events emitted
- [ ] Refined output persisted to `jobs/{job_id}/refine/output.json`
- [ ] Bounded retry (max 3, jittered exponential backoff), circuit-breaker-on-repeat-fail, skip-on-failure
- [ ] A/B harness script running N songs with/without refine, diffing LilyPond output, flagging regressions

### Add After Validation (v1.x) — Future Milestones

Features to add once refine stage is proven and stable.

- [ ] LLM response caching by `(song_title, artist, variant_hash)` — trigger: per-song cost exceeds a threshold, or repeat-submit traffic justifies.
- [ ] Langfuse (or OpenTelemetry) observability integration — trigger: need to debug a recurring quality regression across multiple jobs.
- [ ] Scored eval rubric + golden set — trigger: A/B harness shows refine is net-positive and we need defensible regression floors.
- [ ] Default-on rollout — trigger: golden set passes zero-regression floor for two consecutive milestones.
- [ ] Multi-provider LLM abstraction (OpenAI fallback) — trigger: Anthropic availability drops below product SLA.

### Future Consideration (v2+)

Deferred until product-market fit.

- [ ] Self-hosted / local LLM (Ollama, vLLM) — deferred until local music-reasoning models match API-hosted quality.
- [ ] Multi-language support — deferred until English quality is proven and web-search grounding heuristics can be language-adapted.
- [ ] LLM note addition (guarded by stronger grounding evidence) — deferred indefinitely; requires a completely different trust model.
- [ ] Streaming partial refine — deferred indefinitely; conflicts with structured output and atomic correctness semantics.

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Structured output contract (Pydantic schema enforcement) | HIGH | LOW | P1 |
| Modify+delete-only validator | HIGH | MEDIUM | P1 |
| Web-search grounding | HIGH | MEDIUM | P1 |
| Opt-in toggle plumbed end-to-end | HIGH | MEDIUM | P1 |
| Skip-on-failure + `refine_skipped` event | HIGH | LOW | P1 |
| Refined output persisted as blob artifact | MEDIUM | LOW | P1 |
| Bounded retry + circuit breaker | MEDIUM | LOW | P1 |
| Stage lifecycle events (WebSocket) | MEDIUM | LOW | P1 |
| Music21-parse validation gate | MEDIUM | LOW | P1 |
| Env-driven secret handling + startup validation | HIGH | LOW | P1 |
| A/B harness with LilyPond diff | HIGH | MEDIUM-HIGH | P1 |
| Variant-aware stage placement | MEDIUM | LOW-MEDIUM | P1 |
| Observability (structured logging, Langfuse-ready interface) | MEDIUM | MEDIUM | P2 |
| LLM response caching | LOW (at current traffic) | MEDIUM | P3 |
| Scored eval rubric + golden set | MEDIUM | HIGH | P3 |
| Default-on rollout | HIGH (once proven) | LOW | P3 |
| Multi-provider abstraction | LOW (Anthropic is reliable) | HIGH | P3 |

**Priority key:**
- P1: Must ship in this milestone (GAU-105)
- P2: Should have, add when possible within milestone if cheap; otherwise next
- P3: Explicitly deferred; tracked in "Add After Validation" or "Future Consideration"

Note: Every P1 item maps to either a PROJECT.md Active requirement or a CONCERNS.md artifact this stage is meant to address.

## Competitor / Prior-Art Analysis

| Feature | AnthemScore | ScoreCloud | Klangio | ChatMusician / NotaGen | Our Approach |
|---------|-------------|------------|---------|------------------------|--------------|
| Grounded song identity | None (signal only) | None (signal only) | None (instrument-model only) | None (pure generation, no grounding) | Anthropic web-search tool for key/meter/arrangement lookup |
| Note-addition prevention | N/A (manual editor) | N/A (manual editor) | N/A | No (generation is free) | Output validator rejects additions |
| Output format | MIDI + notation (internal) | Notation (internal) | MIDI + MusicXML | ABC (text) | MusicXML via existing `PianoScore` contract |
| Opt-in UX | Always on (desktop app) | Always on (web/mobile) | Always on (cloud) | N/A | Per-job toggle, default off |
| Failure semantics | Crash / garbage output | Crash / garbage output | Crash / garbage output | N/A | Skip-on-failure; job still produces PDF |
| Eval / regression detection | Manual QA | Manual QA | Manual QA | MusicTheoryBench (offline eval only) | A/B harness with LilyPond diff |
| Schema enforcement | N/A (proprietary) | N/A (proprietary) | N/A | ABC parser (music21 compatible) | Pydantic + music21 parse gate |
| Observability | None (desktop) | Minimal | Minimal | N/A | Structured logging; Langfuse-ready interface |

**Takeaway:** The end-to-end transcription products all treat cleanup as a manual UI task — there's no competitor doing automated LLM refinement in a pipeline. The academic LLM-music work focuses on generation, not refinement. Our design is assembling known-good LLM-application-engineering patterns into a music-specific stage; differentiation comes from the *discipline* (opt-in, grounded, validator-enforced, skip-safe, A/B-measurable), not from the individual features.

## Gotchas Carried Forward to Implementation

Specific things the implementation phase will need to design around (not all features, but bears on feature design):

1. **Schema re-export vs. subclass:** `PianoScore` vs. `RefinedPianoScore` is still TBD per PROJECT.md Constraints line 71. Decide in plan phase — but FEATURES.md assumes output is `PianoScore`-shape to preserve engrave contract.
2. **Two voices per hand remains a hard engrave constraint** (CONCERNS.md: `MAX_VOICES_RH = 2`). The refine prompt must know this; the validator must enforce it; otherwise refine "improves" the score and engrave silently drops notes.
3. **Humanization seed=42 is hardcoded** (CONCERNS.md). For A/B harness this is a feature (deterministic comparisons). For user-facing perceived variation, it's a bug — but out of scope this milestone.
4. **Cover-search already produces fallback behavior silently** (CONCERNS.md: cover-search exceptions swallowed). Refine's `refine_skipped` event needs to be *visibly different* in the WebSocket stream, so users can tell refine-was-asked-for-but-skipped apart from refine-was-never-requested.
5. **In-memory job state is process-scoped** (CONCERNS.md). If the orchestrator restarts mid-refine, the job is lost regardless. Not refine's problem to solve, but affects how we think about refine latency bounds (longer stage = bigger window for lost jobs). Out of scope.

## Sources

### LLM music generation and notation research
- [YNote: A Novel Music Notation for Fine-Tuning LLMs in Music Generation (arxiv 2502.10467)](https://arxiv.org/abs/2502.10467) — MEDIUM confidence; recent (2025), simplified notation for LLM training
- [ChatMusician (arxiv 2402.16153)](https://arxiv.org/html/2402.16153v1) — MEDIUM; fine-tuned LLaMA2 on ABC notation, >90% ABC parse success
- [NotaGen: Advancing Musicality in Symbolic Music Generation (arxiv 2502.18008)](https://arxiv.org/html/2502.18008v5) — MEDIUM; CLaMP-DPO for refinement, but generation-focused
- [HNote: Hexadecimal Encoding for Fine-Tuning LLMs (arxiv 2509.25694)](https://arxiv.org/abs/2509.25694v2) — LOW; very recent, not widely cited yet
- [Factual and Musical Evaluation Metrics for Music Language Models (arxiv 2511.05550)](https://arxiv.org/html/2511.05550) — MEDIUM; grounding/evaluation discussion

### Competitor products
- [AnthemScore by Lunaverus](https://www.lunaverus.com/) — HIGH; official product page
- [ScoreCloud](https://scorecloud.com/learn/best-music-transcription-software/) — HIGH; official comparison
- [Klangio Transcription Studio](https://klang.io/) — HIGH; official product; AI "music grammar" LM as final pipeline step
- [AnthemScore vs ScoreCloud SaaS comparison](https://www.saashub.com/compare-scorecloud-vs-anthemscore) — MEDIUM

### Audio transcription and ghost notes
- [Spotify basic-pitch GitHub](https://github.com/spotify/basic-pitch) — HIGH; official repo, documents lightweight-model tradeoffs
- [Beat-Based Rhythm Quantization of MIDI Performances (arxiv 2508.19262)](https://arxiv.org/html/2508.19262) — MEDIUM; transformer-based quantization
- [Well-Tempered Spelling: Key-Invariant Pitch Spelling Algorithm (ISMIR 2004)](https://archives.ismir.net/ismir2004/paper/000208.pdf) — HIGH; foundational algorithm, still cited

### Anthropic / LLM infrastructure
- [Anthropic web search tool docs](https://docs.claude.com/en/docs/agents-and-tools/tool-use/web-search-tool) — HIGH; official
- [Anthropic structured outputs docs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) — HIGH; official, public beta Nov 2025
- [Anthropic rate limits docs](https://platform.claude.com/docs/en/api/rate-limits) — HIGH; official
- [Claude API 429 handling guide (sitepoint)](https://www.sitepoint.com/claude-api-429-error-handling-python/) — MEDIUM; production patterns
- [Langfuse Observability overview](https://langfuse.com/docs/observability/overview) — HIGH; official docs, native Anthropic cost tracking
- [Eleuther lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — HIGH; widely-used framework

### LLM pipeline patterns
- [The Edit Trick: Efficient LLM Annotation of Documents (Medium, Waleed Kadous)](https://waleedk.medium.com/the-edit-trick-efficient-llm-annotation-of-documents-d078429faf37) — MEDIUM; refine-instead-of-regenerate pattern
- [Code Surgery: How AI Assistants Make Precise Edits to Your Files](https://fabianhertwig.com/blog/coding-assistants-file-edits/) — MEDIUM; Cursor's two-model approach

### Audio fingerprinting (researched for completeness; not in v1)
- [Shazam Inside: Five-Second Fingerprint (TDS)](https://towardsdatascience.com/the-five-second-fingerprint-inside-shazams-instant-song-id/) — MEDIUM; background for future "real song match" feature
- [AudD Music Recognition API](https://audd.io/) — HIGH; documented API, alternative to Shazam (no public API)

---
*Feature research for: LLM-augmented music notation refinement (GAU-105)*
*Researched: 2026-04-13*
