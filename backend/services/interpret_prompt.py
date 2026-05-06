"""Prompt + tool-schema builders for the interpret stage. Pure, no network.

The interpret stage sends Claude a brief musical summary of the transcription
result plus the user's free-form arrangement prompt. Claude must respond with
a single ``submit_arrangement_hints`` tool call containing structured hints.
No prose output is permitted.
"""
from __future__ import annotations

from typing import Any

PROMPT_VERSION = "interpret-v1"

SYSTEM_PROMPT = (
    "You are a music arrangement advisor. The user has submitted a free-form "
    "request describing how they want their piano sheet music to sound or feel. "
    "Your job is to interpret that request and emit structured arrangement hints "
    "by calling submit_arrangement_hints exactly once.\n\n"
    "Rules:\n"
    " * You MUST call submit_arrangement_hints — never return prose.\n"
    " * Only set fields you can justify from the user's prompt. Omit fields "
    "the prompt does not address.\n"
    " * difficulty: beginner = simple melody + basic chords, intermediate = "
    "standard piano arrangement, advanced = dense textures / complex voicings.\n"
    " * density: sparse = few simultaneous notes, moderate = normal, "
    "dense = full-texture chording.\n"
    " * tempo_bias: fractional adjustment in [-0.25, 0.25]. Positive = push "
    "(slightly faster feel), negative = lay back (looser feel). Use only "
    "when the user explicitly asks for tempo feel changes.\n"
    " * Do NOT invent musical content. You are only setting arrangement "
    "parameters based on the user's stated preferences.\n"
)


def build_user_prompt(
    prompt: str,
    txr_summary: dict[str, Any],
    title_hint: str | None,
    artist_hint: str | None,
) -> str:
    """Assemble the user-facing prompt from the arrangement request + txr summary."""
    key = txr_summary.get("key", "unknown")
    time_sig = txr_summary.get("time_signature", "unknown")
    tempo = txr_summary.get("tempo_bpm", "unknown")
    duration = txr_summary.get("duration_sec", "unknown")
    chord_count = txr_summary.get("chord_count", 0)
    sections = txr_summary.get("section_labels", [])

    if isinstance(time_sig, (list, tuple)) and len(time_sig) == 2:
        time_sig_str = f"{time_sig[0]}/{time_sig[1]}"
    else:
        time_sig_str = str(time_sig)

    if isinstance(tempo, (int, float)):
        tempo_str = f"{tempo:g} BPM"
    else:
        tempo_str = str(tempo)

    if isinstance(duration, (int, float)):
        duration_str = f"{duration:.1f}s"
    else:
        duration_str = str(duration)

    section_str = ", ".join(str(s) for s in sections) if sections else "none detected"

    return (
        "User arrangement request:\n"
        f"  {prompt!r}\n"
        "\n"
        "Song context:\n"
        f"  title={title_hint!r}, artist={artist_hint!r}\n"
        f"  key = {key}\n"
        f"  time_signature = {time_sig_str}\n"
        f"  tempo = {tempo_str}\n"
        f"  duration = {duration_str}\n"
        f"  chord_count = {chord_count}\n"
        f"  sections = {section_str}\n"
        "\n"
        "Call submit_arrangement_hints with the structured hints that best "
        "match the user's request. Omit fields the request does not address.\n"
    )


def submit_arrangement_hints_tool_schema() -> dict[str, Any]:
    """Anthropic tool schema mirroring the ArrangementHints Pydantic model.

    All fields are optional — Claude is instructed to omit fields the user's
    prompt does not address.
    """
    return {
        "name": "submit_arrangement_hints",
        "description": (
            "Submit structured arrangement hints derived from the user's prompt. "
            "Call exactly once. Omit fields the prompt does not address."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "difficulty": {
                    "type": "string",
                    "enum": ["beginner", "intermediate", "advanced"],
                    "description": "Target difficulty level for the piano arrangement.",
                },
                "density": {
                    "type": "string",
                    "enum": ["sparse", "moderate", "dense"],
                    "description": "Note density / texture of the arrangement.",
                },
                "style_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Style descriptors (e.g. ['jazz', 'ballad', 'classical']).",
                },
                "dynamic_emphasis": {
                    "type": "string",
                    "enum": ["soft", "neutral", "bold"],
                    "description": "Overall dynamic character of the arrangement.",
                },
                "tempo_bias": {
                    "type": "number",
                    "minimum": -0.25,
                    "maximum": 0.25,
                    "description": (
                        "Fractional tempo-feel adjustment. Positive = push (tighter), "
                        "negative = lay back (looser). 0 means no change."
                    ),
                },
                "hand_balance": {
                    "type": "string",
                    "enum": ["lh_lead", "balanced", "rh_lead"],
                    "description": "Which hand carries the primary musical interest.",
                },
                "notes": {
                    "type": "string",
                    "description": (
                        "Short rationale explaining your hint choices. "
                        "For telemetry only — not shown to the user."
                    ),
                },
            },
        },
    }
