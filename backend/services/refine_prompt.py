"""Refine-stage prompt source-of-truth (D-09, D-10, D-12).

All prompt strings and the note-ID scheme live here so Phase 3 prompt
iteration can diff this module in isolation from service orchestration
logic. RefineService imports the constants and helpers; no other module
redefines them.

Decision references (see .planning/phases/02-refine-service-and-pipeline-integration/02-CONTEXT.md):
  * D-09 — prompt lives in its own module
  * D-10 — REFINE_PROMPT_VERSION stamped in every llm_trace.json
  * D-12 — note IDs are hand-prefixed sequential: f"{hand_letter}-{idx:04d}"
"""
from __future__ import annotations

import re
from typing import Any

from shared.contracts import (
    ExpressiveNote,
    HumanizedPerformance,
    PianoScore,
    ScoreNote,
)

# D-10: bumped by hand when prompt or schema meaningfully changes. Every
# refine call stamps this value into llm_trace.json so future caching
# (v2 milestone) can key on (song_fingerprint, prompt_version, model).
REFINE_PROMPT_VERSION = "2026.04-v1"

# D-12: note IDs have shape r-NNNN or l-NNNN (four-digit zero-padded index).
# The validator uses this pattern to reject malformed target_note_id values
# from the LLM before cross-reference lookup.
ID_PATTERN: re.Pattern[str] = re.compile(r"^[rl]-\d{4}$")


SYSTEM_PROMPT = """You are a music notation reviewer with modify+delete authority.

Your job: review a transcribed-and-humanized piano performance and emit a
list of RefineEditOp objects that correct rhythm, harmony, and notation
artifacts before the score is engraved as sheet music.

AUTHORITY RULES (load-bearing; violations cause the entire edit list to be rejected):
  1. You MAY modify existing notes: change pitch, velocity, timing, or duration.
  2. You MAY delete existing notes that are transcription artifacts (ghost notes, doublings).
  3. You MUST NOT add new notes. Note addition is forbidden by contract.
  4. Every edit MUST reference an existing target_note_id from the performance.
     Inventing note IDs is a rejection signal.
  5. Each target_note_id may appear in the edit list at most once.
     Duplicate target_note_id rejects the entire edit list.

RATIONALE CATEGORIES (closed enum; use exactly one per edit):
  harmony_correction, ghost_note_removal, octave_correction, voice_leading,
  duplicate_removal, out_of_range, timing_cleanup, velocity_cleanup, other

GROUND IN SONG IDENTITY: use the web_search tool to confirm the song's key,
meter, and characteristic harmonies before making harmony_correction edits.
Cite the source URL in every RefineCitation you emit.

PITCH BOUND: edited pitches MUST stay within piano range [21, 108]. Any edit
producing an out-of-range pitch is rejected.

DO NOT "correct" intentional dissonance, ghost notes in funk grooves, or
modal spellings simply because they look unusual — transcription artifacts
have characteristic velocity and onset patterns; musical choices do not.
"""

USER_PROMPT_TEMPLATE = """Song: {title}
Composer: {composer}
Prompt version: {version}

Performance notes (id | hand | pitch | onset_beat | duration_beat | velocity):
{notes}

Emit a RefinedEditOpList. Every edit's target_note_id MUST appear in the
performance notes above. Use the web_search tool to ground harmonic choices
against the actual song.
"""


def _hand_letter(hand: str) -> str:
    """Map 'rh'/'lh' (ExpressiveNote) or 'right'/'left' to 'r'/'l'. Raises on unknown."""
    if hand in ("rh", "r", "right"):
        return "r"
    if hand in ("lh", "l", "left"):
        return "l"
    raise ValueError(f"unknown hand value: {hand!r}")


def _derive_note_id_map(
    performance: HumanizedPerformance | PianoScore,
) -> dict[str, ExpressiveNote | ScoreNote]:
    """Derive deterministic hand-prefixed sequential note IDs (D-12).

    For HumanizedPerformance: groups expressive_notes by hand, sorts each
    hand by (onset_beat, pitch) ascending, assigns f"{hand_letter}-{idx:04d}".

    For PianoScore: same scheme, using score.right_hand → 'r' and
    score.left_hand → 'l' directly (no hand field on ScoreNote).

    The validator in Plan 02 re-derives the same map from the same source
    to check RefineEditOp.target_note_id membership.
    """
    result: dict[str, ExpressiveNote | ScoreNote] = {}
    if isinstance(performance, HumanizedPerformance):
        rh: list[ExpressiveNote | ScoreNote] = [
            n for n in performance.expressive_notes if _hand_letter(n.hand) == "r"
        ]
        lh: list[ExpressiveNote | ScoreNote] = [
            n for n in performance.expressive_notes if _hand_letter(n.hand) == "l"
        ]
    else:  # PianoScore
        rh = list(performance.right_hand)
        lh = list(performance.left_hand)

    rh_sorted = sorted(rh, key=lambda n: (n.onset_beat, n.pitch))
    lh_sorted = sorted(lh, key=lambda n: (n.onset_beat, n.pitch))

    for idx, note in enumerate(rh_sorted):
        result[f"r-{idx:04d}"] = note
    for idx, note in enumerate(lh_sorted):
        result[f"l-{idx:04d}"] = note
    return result


def _serialize_notes(note_id_map: dict[str, ExpressiveNote | ScoreNote]) -> str:
    """Render the note_id_map as a compact, unambiguous tabular string."""
    lines = []
    for id_str, note in sorted(note_id_map.items()):
        hand = id_str[0]  # 'r' or 'l'
        lines.append(
            f"{id_str} | {hand} | {note.pitch} | {note.onset_beat:.3f} | "
            f"{note.duration_beat:.3f} | {note.velocity}"
        )
    return "\n".join(lines)


def build_prompt(
    metadata: dict[str, Any],
    performance: HumanizedPerformance | PianoScore,
    *,
    web_search_max_uses: int = 5,
) -> dict[str, Any]:
    """Assemble the full prompt bundle for RefineService.

    Args:
        metadata: free-form dict; ONLY ``title`` and ``composer`` are read.
            Any other keys are ignored (prompt-injection guard per
            security_threat_model in planning).
        performance: the source HumanizedPerformance or PianoScore that
            refine will review. Must not be mutated.
        web_search_max_uses: cap from OHSHEET_REFINE_WEB_SEARCH_MAX_USES.

    Returns a dict with keys:
        system, user, version, note_id_map, web_search_tool_spec
    """
    title = str(metadata.get("title", "Unknown"))
    composer = str(metadata.get("composer", "Unknown"))
    note_id_map = _derive_note_id_map(performance)
    notes_serialized = _serialize_notes(note_id_map)
    user = USER_PROMPT_TEMPLATE.format(
        title=title,
        composer=composer,
        version=REFINE_PROMPT_VERSION,
        notes=notes_serialized,
    )
    return {
        "system": SYSTEM_PROMPT,
        "user": user,
        "version": REFINE_PROMPT_VERSION,
        "note_id_map": note_id_map,
        # WR-04: Anthropic's WebSearchTool20260209Param requires BOTH `type`
        # and `name` fields — `name` is typed Required[Literal["web_search"]]
        # in anthropic.types.web_search_tool_20260209_param. Without it the
        # real API would reject the tools= argument at validation time.
        "web_search_tool_spec": {
            "type": "web_search_20260209",
            "name": "web_search",
            "max_uses": web_search_max_uses,
        },
    }
