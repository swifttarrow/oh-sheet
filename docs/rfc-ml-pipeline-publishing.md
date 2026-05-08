# RFC: Publishing `oh-sheet-ml-pipeline`

**Status:** Draft for discussion. Tracks issue [#107](https://github.com/Oh-Sheet-Team/oh-sheet/issues/107). Not a decision.

## 1. Current architecture (post-#103, #106)

The engrave stage has two backends, selected by `OHSHEET_ENGRAVE_BACKEND`:

| Backend | What it is | Output |
| --- | --- | --- |
| `local` (default) | In-process: `music21` → MusicXML, then `LilyPond` → PDF | MusicXML + PDF |
| `remote_http` | POST MIDI bytes to the `oh-sheet-ml-pipeline` HTTP service at `OHSHEET_ENGRAVER_SERVICE_URL` | MusicXML only |

`backend/jobs/runner.py` records the route as one of `local`, `remote_http`, or `remote_http_fallback` (the last when `local` raised `EngraveLocalError` and we fell through). The remote service still has no public repo, no published image, and no docs for running it yourself — it remains a hosted, proprietary Oh Sheet component.

**Implication for self-hosting:** Self-hosters can run the full pipeline today on the `local` backend without `oh-sheet-ml-pipeline`. What they cannot get is the **output quality** of the ML engraver. That, not raw functionality, is the gap this RFC is about.

The `local` backend reads the structured `(PianoScore, ExpressionMap)` directly, so chord symbols, dynamics, pedal marks, key/time signatures, and per-note voicing survive into the score. The `remote_http` backend re-derives those from MIDI bytes — the output looks more polished in practice but loses the structural information passed by the rest of the pipeline.

## 2. The four options from #107

| # | Option | Effort | Ongoing cost | Supportability | User impact | Blocking risks |
| --- | --- | --- | --- | --- | --- | --- |
| A | Open-source repo + publish image | High (audit, scrub history, write docs, set up CI for releases) | Medium-high (community PRs, issue triage, security disclosures) | Hardest — public source raises expectation of versioned releases | Highest — full parity, full transparency, third-party contributions possible | Weight licensing (Q1), training data provenance, security review (Q3), maintenance commitment (Q4) |
| B | Image-only (closed source, freely runnable) | Medium (build pipeline, threat model, docs for running) | Medium (image refreshes, vuln patching, maintaining a registry) | Easier than A — no source-level expectations, but image bugs still our problem | High — closes the parity gap for self-hosters | Weight licensing (Q1), security review (Q3) |
| C | Document a plugin protocol (HTTP contract + reference impl) | Low-medium (write OpenAPI spec, document, possibly publish a thin reference engraver) | Low (own only the contract; third parties own implementations) | Easiest — we maintain a spec, not a service | Medium — unlocks third-party engravers but no out-of-the-box parity option ships | Plugin protocol (Q5) needs discussion; risk of fragmentation |
| D | Status quo + clearer docs | Already done in #106 | None new | Trivial — we already do this | Low — self-hosting works on `local`, but parity gap remains | None — but the parity gap stays unsolved |

Options are not mutually exclusive. C is compatible with any of A/B/D (publishing a contract doesn't preclude also publishing the service). A implies B. D is the floor — we are already there.

## 3. The five open questions

### Q1. Model weights & licensing

**What we'd need to know:** Provenance of every dataset used to train the engraver model. Whether any are under licenses that prohibit redistribution of derived weights (e.g. non-commercial datasets, scraped corpora with disputed rights). Whether we can re-train or fine-tune on a clean corpus if the answer is bad.

**First step:** Audit the training pipeline in the `oh-sheet-ml-pipeline` repo (private). Catalogue every dataset and its license. Surface findings here.

### Q2. Hosting cost (for any public demo endpoint)

**What we'd need to know:** Per-request cost of the current hosted endpoint (GPU vs CPU inference, p50/p95 latency, expected QPS if we publish). Rate-limit and abuse-protection mechanisms we'd attach to a public demo.

**First step:** Pull a week of telemetry from the current hosted endpoint. Decide whether a free public demo is sustainable, or whether a self-host-or-pay model is cleaner.

### Q3. Security review

**What we'd need to know:** Threat model for the engraver: it accepts arbitrary MIDI input and produces MusicXML. Risks include MIDI parser bugs (libsmf, mido, etc.), resource exhaustion via crafted inputs, and any subprocess shell-out. A pre-publication threat model is a precondition for B and A.

**First step:** Half-day threat-modeling session against `oh-sheet-ml-pipeline`'s request handler. Output: list of input-validation gaps and mitigations.

### Q4. Maintenance commitment

**What we'd need to know:** Honest answer to "if we publish this, how often will we update the image, how fast will we patch CVEs, will we accept community PRs?" The answer determines which option is credible. "Best-effort, no SLA" is fine — "we don't know" is not.

**First step:** Maintainer (one person — me) write a one-paragraph commitment statement. If it can't be written honestly, that rules out A and probably B.

### Q5. Plugin protocol

**What we'd need to know:** What's the minimum HTTP contract a third-party engraver would need to implement? `POST /engrave` with MIDI in / MusicXML out is the obvious starting point, but: do we want to pass structured `PianoScore` instead of MIDI (preserves dynamics/pedal/voices, see §1)? Versioned protocol? Capability negotiation?

**First step (landed):** Extracted the current `oh-sheet-ml-pipeline` endpoint into an OpenAPI spec, frozen as v0.1, published under [`docs/engraver-protocol/`](./engraver-protocol/). The structured-input evolution remains open as a v0.2+ question.

## 4. Recommendation (starting point for discussion, not a decision)

Sequence: **C → consider B**.

1. **Land C immediately.** ✅ **Done** — see [`docs/engraver-protocol/`](./engraver-protocol/) for the v0.1 OpenAPI spec and conformance notes. Documents the existing extension point, preserves optionality for everything else, lets a motivated self-hoster build their own engraver today.

2. **Answer Q1 + Q3 + Q4 before committing to B or A.** Specifically:
   - Q1 (training-data audit): if any dataset blocks redistribution, A and B are blocked until re-train.
   - Q3 (threat model): a precondition for any public runnable image.
   - Q4 (maintenance commitment): the honest answer determines whether A is even responsible.

3. **Default to B (image-only) over A (open source) once Q1/Q3/Q4 clear.** Reasoning: B closes the parity gap with the smallest ongoing cost we can credibly support. A adds source-level expectations (community PRs, public issue triage, supply-chain audit) that we may not be able to honor. If Q4 says "yes, we can commit to that," reconsider A.

4. **Do not pick D (status quo) as the long-term answer**, even though we are there now. The parity gap is real for self-hosters and is the substance of #105. D is the floor while we work toward B.

## 5. What this RFC is not

- **Not a decision.** Any of A/B/C/D could be the right call once Q1–Q5 are answered. The §4 recommendation is a starting point so the discussion has something concrete to push back on.
- **Not a license claim.** Q1 is open. The RFC does not claim weights are or aren't redistributable.
- **Not a security claim.** Q3 is open. The RFC does not claim the engraver is or isn't safe to publish.
- **Not a commitment.** Maintainer time is finite; whatever we publish is what we can credibly support.

Comments welcome on this doc, on #107, or both.
