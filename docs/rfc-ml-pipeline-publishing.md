# RFC: Publishing `oh-sheet-ml-pipeline`

**Status:** Draft — discussion document for issue [#107](https://github.com/Oh-Sheet-Team/oh-sheet/issues/107)
**Author:** rfc-author (engraver-docs-fix team)
**Date:** 2026-05-08
**Related:** #104 (env var docs), #105 (self-hosting gap), #106 (local fallback), #107 (this RFC)

---

## 1. Context

### What the engrave stage does today

Oh Sheet's pipeline ends with an **engrave** step that converts a MIDI performance into MusicXML (which downstream tools render to a PDF). In the current `main`, the engrave step is implemented as a single HTTP call:

- Code: `backend/services/ml_engraver_client.py` — `engrave_midi_via_ml_service()`
- Caller: `backend/jobs/runner.py` line 504 (inline in the orchestrator, not a Celery worker)
- Contract: `POST {OHSHEET_ENGRAVER_SERVICE_URL}/engrave`
  - Request: raw MIDI bytes (`Content-Type: application/octet-stream`)
  - Response: raw MusicXML bytes (200 OK)
  - Error model: 4xx = client error (no retry), 5xx / timeout / transport = retry up to 3× with exponential backoff (0.5s base)
  - Stub guard: any 200 response under 500 bytes is rejected as a placeholder
  - Per-call timeout: `OHSHEET_ENGRAVER_SERVICE_TIMEOUT_SEC` (default 60s)

### Deployment shape

Per `docker-compose.prod.yml`:

- Orchestrator container reads `OHSHEET_ENGRAVER_SERVICE_URL` (declared `${VAR:?...}` so compose refuses to start with it unset)
- The legacy `worker-engrave` Celery worker has been retired — engrave is inline in the orchestrator
- The default in `backend/config.py` is `http://localhost:8080`, which is only meaningful in dev

### The gap (issue #105 / #107)

The `oh-sheet-ml-pipeline` service that satisfies this contract is **not publicly available** — no source repo, no published image, no docs. As a result, anyone trying to self-host Oh Sheet hits a hard failure at the engrave stage. Issue #107 tracks the decision of whether and how to publish it.

### Verification note (please confirm)

Issue #107's introduction states *"We've shipped an opt-in `music21`-based local fallback (`OHSHEET_ENGRAVER_LOCAL_FALLBACK=1`, see #106)."* In the current `main` branch I could not find:

- An `OHSHEET_ENGRAVE_BACKEND` or `OHSHEET_ENGRAVER_LOCAL_FALLBACK` setting in `backend/config.py`
- An `engrave_local.py` module in `backend/services/`
- Any local-fallback branch in `backend/jobs/runner.py` — line 504 calls `engrave_midi_via_ml_service` unconditionally
- Any `OHSHEET_ENGRAVE*` entries in `.env.example`

The merged PR for #105 (per its body) "dropped" a music21 fallback because #103 was supposed to have shipped a better one — but neither appears in `main` today. **This RFC therefore treats the current state as "no local fallback" and asks the maintainer to confirm before any publication decision is finalized.** If a fallback does exist on a branch I missed, the user-impact rows below should be re-rated.

---

## 2. The four candidate options

The four outcomes named in #107, restated in our own words:

| # | Option | One-line description |
|---|--------|----------------------|
| A | Open-source repo + image | Public GitHub repo with build instructions and a published Docker image (with weights) |
| B | Image-only | Pre-built Docker image freely runnable, source remains private |
| C | Plugin protocol | Document the HTTP contract; ship a reference open-source impl (e.g. music21-only, no ML) but invite third-party engravers |
| D | Status quo | Keep the service private; document explicitly that "remote engraver" is the only supported engrave path |

These are not mutually exclusive — C is in particular composable with A, B, or D.

---

## 3. Decision matrix

Each cell is rated **Low / Medium / High** with a short rationale. Effort and cost are *higher = worse*; supportability and user-impact-positive are *higher = better*; blocking risks are listed not rated.

### 3.1 One-time effort (to ship)

| Option | Rating | Notes |
|--------|--------|-------|
| A. Repo + image | **High** | Cleanup pass on private code, license review, CI for image build, repo hygiene (CONTRIBUTING, CODE_OF_CONDUCT, security policy), public issue triage setup. |
| B. Image only | **Medium** | Image build CI + a public registry namespace. No public source review, but still need a basic threat model since users run it. |
| C. Plugin protocol | **Medium–High** | Spec writing (OpenAPI), reference implementation, conformance test suite. Lower if reference impl is just music21 wrapping. Higher if we also want to ship our service under the protocol. |
| D. Status quo | **Low** | Edit README + deployment.md to clearly say "engrave requires a hosted service that is not currently public; self-hosting is not supported end-to-end." |

### 3.2 Ongoing cost / maintenance

| Option | Rating | Notes |
|--------|--------|-------|
| A. Repo + image | **High** | Issues, PRs, security disclosures, weight/image releases tied to model retrains, semver for the HTTP contract. |
| B. Image only | **Medium** | Image release cadence, CVE patching of the base image, but no PR review burden. |
| C. Plugin protocol | **Medium** | Spec versioning + reference impl maintenance. Compatibility breakage is the main risk — every contract change is a community-coordination event. |
| D. Status quo | **Low** | None beyond what we already do. |

### 3.3 Supportability (can the maintainer credibly back this?)

| Option | Rating | Notes |
|--------|--------|-------|
| A. Repo + image | **Low–Medium** | Public expectations are high; "best-effort" is a hard sell once the repo is public. Needs a written support policy day one. |
| B. Image only | **Medium** | Easier to scope: "we publish images, we patch on a cadence, no source contributions accepted." |
| C. Plugin protocol | **High** | Maintainer only owns the contract + reference impl. Quality of any third-party engraver is its author's problem. |
| D. Status quo | **High** | Nothing changes. |

### 3.4 User impact — closes the self-hosting gap?

| Option | Rating | Notes |
|--------|--------|-------|
| A. Repo + image | **High** | Full parity self-hosting, including the ability to fork and modify. |
| B. Image only | **High** | Full parity self-hosting; no source access but most self-hosters don't need it. GPU requirements (if any) become the next bottleneck. |
| C. Plugin protocol | **Medium** | Solves the "I can self-host" question. Does not solve the "I can self-host with parity output quality" question unless someone publishes a parity-quality plugin (which, today, only the maintainer could). |
| D. Status quo | **Low** | Honest about the gap, but does not close it. Without a local fallback (see §1), self-hosters have no working engrave path at all. |

### 3.5 Blocking risks

| Option | Risks |
|--------|-------|
| A. Repo + image | Model-weight licensing (open question Q1). Threat model for code+image (Q3). Security disclosure inbox needed. |
| B. Image only | Model-weight redistribution (Q1) — even without source, distributing weights triggers licensing review. Threat model for the running container (Q3). |
| C. Plugin protocol | Risk of fragmentation: multiple incompatible plugins, version-skew bugs reported as Oh Sheet bugs. Reference impl quality sets the floor — too weak and the protocol looks like a fig leaf. |
| D. Status quo | Continued community frustration (#105-class issues will keep arriving). Reputational risk of an "open source" project that can't be self-hosted. |

---

## 4. Open questions from #107 — what we'd need to answer them

For each question, what would unblock a decision and a concrete first step.

### Q1. Model weights & licensing

**To decide we need:**
- Provenance audit of training data (what corpora, under what terms)
- Whether weights can be redistributed (commercial vs. research-only, attribution requirements, share-alike clauses)
- Whether weight licensing differs from code licensing — e.g. CC-BY-NC for weights with MIT for code

**First step:** Internal training-data inventory. Until that's done, this RFC cannot recommend A or B with confidence. **Flagged as an open question, not an assertion.**

### Q2. Hosting cost (if we run a public demo)

**To decide we need:**
- Per-engrave compute cost (GPU seconds × hourly rate, or CPU time if not GPU)
- Expected volume from public traffic — hard to estimate without launching
- Abuse model: is MIDI input cheap to validate? Can a malicious payload force expensive inference?

**First step:** Measure p50/p95 engrave time and resource use under current production load (we already have prod data). Multiply by a plausible 10× public-traffic factor. If the answer is small, B can include a hosted demo; if not, B is "image only, BYO compute."

### Q3. Security review

**To decide we need:**
- Threat model for the engrave HTTP endpoint:
  - **Input validation:** untrusted MIDI parsing — is the MIDI parser hardened? Memory limits on track count / event count?
  - **Resource exhaustion:** per-request CPU/GPU caps; concurrent-request limits
  - **Output validation:** can a crafted input cause MusicXML output that exploits downstream LilyPond/music21? (Oh Sheet itself currently consumes the MusicXML directly, so this matters even before publication.)
  - **Supply chain:** for B/A, what's the base image, what CVEs does it carry, who patches?
- Disclosure policy (private email vs. GitHub Security Advisories)

**First step:** Half-day threat-modeling session covering the four items above. Output is a one-page SECURITY.md skeleton.

### Q4. Maintenance commitment

**To decide we need:**
- Maintainer time budget — hours/week the project can sustain on public engraver issues
- Support tiers we're willing to publish: "best-effort", "monthly image releases", "semver'd HTTP contract", "none of the above"
- Backport policy for security fixes

**First step:** Maintainer answers two questions in writing: *"How many hours/month am I willing to spend on this?"* and *"What happens to a P1 bug filed against the published image at 11pm on a Friday?"* The answers gate A vs. B vs. C.

### Q5. Plugin protocol — would a contract be more useful than a service?

**To decide we need:**
- Survey: would actual self-hosters in #105 / #107 use a plugin slot, or do they want our exact engraver?
- A draft contract (the one we already implement) — does it generalize to non-ML engravers (rules-based, pure music21, third-party SaaS)?
- Reference implementation choice: pure music21? a thin shim over an external service? both?

**First step:** Draft an OpenAPI spec for `POST /engrave` (we have all the inputs already — see §1) and circulate on #107 for feedback. **This step is cheap, useful in every option, and unblocks A/B/D as well.**

---

## 5. Recommended option (starting point for discussion)

> This is a recommendation for *what to discuss first*, not a final decision. The maintainer (and #107 commenters) should push back freely.

**Recommendation: a sequenced combination of D + C, with B as the explicit goal.**

Concretely, in order:

1. **Immediate (this week):** Land the issue-#105 docs (task #7) so users know the engrave path is remote-only and self-hosting is currently not supported end-to-end. Honest framing reduces pressure while bigger questions are answered. *(This corresponds to Option D as a near-term posture, not a final destination.)*
2. **Short term (next 1–2 sprints):** Publish the `POST /engrave` HTTP contract as an OpenAPI spec in this repo (Option C step 1). This is cheap, helps every other option, and lets motivated contributors plug in *something* (even a non-ML music21 engraver) without blocking on licensing review. If a community-built music21 plugin lands, the self-hosting gap is partially closed without any decision on weight licensing.
3. **Medium term (gated on Q1 + Q3):** Once the training-data audit and threat model are done, default-target Option B (image-only). B closes the user-impact gap without committing to a public-source maintenance burden, and it is the option where the answers to Q1 and Q3 are most likely to land in our favor.
4. **Re-evaluate Option A only if** there's clear contributor demand and Q4 (maintenance commitment) admits an honest "yes."

### Why this ordering

- **D first** is not a destination, it's truth-in-advertising while we work on a real fix. The current ambiguity is itself harmful.
- **C second** has the best cost/benefit: low effort, high optionality, useful regardless of what we eventually pick for the service.
- **B as the eventual default** because it closes the gap that motivated #105 and #107 with the smallest ongoing cost we can credibly support.
- **A deferred** because public source brings the highest maintenance cost and the most licensing exposure (Q1 applies to weights either way, but A also exposes training/inference code).

### Why *not* to lock this in yet

- Q1 (model-weight licensing) could rule out B. If weights cannot be redistributed, B and A are both off the table and the answer is "C with a non-ML reference impl, plus D."
- Q4 (maintenance commitment) could rule out A. If the maintainer's honest answer is "<2 hours/week," then A's upkeep cost is unacceptable and B is the ceiling.
- A reasonable counter-argument: skip C and go straight to B once Q1/Q3 clear. Worth debating on the issue.

---

## 6. What this RFC is not

- Not a final decision. The maintainer owns option selection.
- Not a claim about training-data licensing. Q1 is genuinely open.
- Not a commitment to any timeline. Step 1 is the only thing already in flight (task #7).
- Not a substitute for the threat model in Q3 — sketching what we'd review is not the same as doing the review.

---

## 7. Asks from commenters on #107

If you're reading this from the linked issue, the most useful inputs you can leave are:

1. Your self-hosting use case and which option (A/B/C/D) actually unblocks it
2. Experience publishing model weights — pitfalls, license templates that worked
3. Opinions on the OpenAPI-spec-first step (§5 step 2): useful, or premature?
4. Whether you'd build/ship a plugin if the protocol were public
