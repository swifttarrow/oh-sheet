---
name: log-architectural-decisions
description: Use when settling an architectural or design choice that has real tradeoffs (multiple viable options, rejected alternatives, or non-obvious downsides) - appends a structured entry to docs/dev-journal.md so future readers see context, options, and rationale
---

# Log Architectural Decisions (Dev Journal)

## Overview

**Core principle:** If a decision has tradeoffs, it belongs in `docs/dev-journal.md`. The journal is append-only context for humans: what we chose, what we rejected, and why.

**Do not use this skill** for trivial choices (lint rule picks, obvious library upgrades with no architectural impact). Use it when someone could reasonably disagree, or when revisiting the code in six months without notes would waste time.

## When to Log

Log after the decision is made (or when documenting a decision the team already took):

- Multiple approaches were considered and one was selected.
- The chosen approach has meaningful cons or operational cost.
- Cross-cutting concerns: data flow, deployment topology, persistence, concurrency, security boundaries, public API shape.

## File and Placement

- **Path:** `docs/dev-journal.md` (repository root).
- **Order:** New entries go **after** the `# Dev Journal` title and **before** would break chronology — default is **append the new `##` section at the end of the file** (after the latest dated entry). If the user asks to group with an existing topic, insert directly under that topic only when they say so.
- **Do not rewrite or delete** prior entries. Fix typos only if the user explicitly asks.

## Section Template (Match Existing Journal)

Mirror the structure and tone of the current journal. Each decision is one top-level dated section.

```markdown
## YYYY-MM-DD: <Short title — decision name>

### TL;DR

**YYYY-MM-DD:** <Exactly one sentence stating what was chosen and the main reason or contrast vs alternatives. No second sentence.>

### Context

<Problem, constraints, and goal. What the system does today vs what you are changing. Enough detail that a new engineer understands why this came up.>

### Approaches Considered

**Approach 1: <Name>**
- <What it is in plain language.>
- *Pros:* <Short bullets.>
- *Cons:* <Short bullets.>

**Approach 2: <Name> (CHOSEN)**
- <What it is.>
- *Pros:*
- *Cons:*

**Approach 3: <Name>**
- ...

<Include every serious alternative that was discussed or should have been. Minimum two approaches besides the chosen one when such options exist; if only two options exist total, that is fine. Mark the selected option with `(CHOSEN)` in the heading.>

### Why Approach N

1. <Numbered rationale tied to goals and tradeoffs — not repetition of pros lists.>
2. ...

### Key Decisions

- **<Concrete bullet>** <Optional short clarification.>
- ...
```

## Hard Requirements

1. **TL;DR** is **one sentence only** after `**YYYY-MM-DD:**` (same date as the section heading). It must name the choice and hint at *why* or *versus what* in that single sentence.
2. **Approaches Considered** must document **alternatives**, each with *Pros* and *Cons* (use italics for those labels to match the journal). The chosen approach must be clearly marked `(CHOSEN)`.
3. **Why Approach N** must explain the selection with numbered points referencing tradeoffs, not marketing language.
4. **Key Decisions** lists follow-on commitments (policies, naming, migration notes) implied by the choice.
5. Use **today’s date** in the heading and TL;DR when the user does not specify one (authoritative calendar from user/session context).

## Workflow

1. Read `docs/dev-journal.md` to confirm latest entry and consistent formatting.
2. Draft the section in the template above.
3. Append to the file (or insert per user instruction).
4. Do not duplicate an existing entry for the same decision on the same day; if updating is needed, ask whether to amend the existing section or add a follow-up dated note.

## Example (Abbreviated)

See the Celery refactor entry already in `docs/dev-journal.md` — match its heading style, TL;DR prefix with bold date, approach blocks, and Key Decisions bullets.
