"""Transform stage - optional Lyria-backed reference arrangement hook.

The transform stage still returns a ``PianoScore`` so the pipeline contract
stays stable. When the Lyria feature flag is enabled, we call Google's
``lyria-3-pro-preview`` model with a prompt tailored to solo piano arranging
and persist any returned audio/text as sidecar artifacts. Because Lyria emits
audio/text rather than symbolic notes, the incoming score remains the stage
output for now.
"""
from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass
from typing import Any

from backend.config import settings
from backend.contracts import PianoScore, ScoreNote, beat_to_sec
from backend.storage.base import BlobStore

log = logging.getLogger(__name__)

_MAX_PROMPT_CHORDS = 12
_MAX_PROMPT_SECTIONS = 8


@dataclass(slots=True)
class _LyriaOutput:
    text_parts: list[str]
    audio_data: bytes | None
    audio_mime_type: str | None


def _import_google_genai() -> Any:
    from google import genai  # noqa: PLC0415

    return genai


def _midi_to_pitch_name(pitch: int) -> str:
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    octave = (pitch // 12) - 1
    return f"{names[pitch % 12]}{octave}"


def _note_range(notes: list[ScoreNote]) -> str:
    if not notes:
        return "none"
    pitches = [n.pitch for n in notes]
    return f"{_midi_to_pitch_name(min(pitches))}-{_midi_to_pitch_name(max(pitches))}"


def _score_duration_sec(score: PianoScore) -> int | None:
    notes = [*score.right_hand, *score.left_hand]
    if not notes or not score.metadata.tempo_map:
        return None
    end_beat = max(n.onset_beat + n.duration_beat for n in notes)
    return max(1, round(beat_to_sec(end_beat, score.metadata.tempo_map)))


def _build_section_hint(score: PianoScore) -> str:
    sections = score.metadata.sections[:_MAX_PROMPT_SECTIONS]
    if not sections:
        return (
            "Shape it like a complete piano piece with intro, main statement, "
            "contrasting middle section, and closing cadence."
        )

    parts: list[str] = []
    for section in sections:
        span = max(1, round(section.end_beat - section.start_beat))
        label = str(section.label.value).replace("_", " ")
        parts.append(f"[{label.title()}] about {span} beats")
    return "Suggested form: " + ", ".join(parts) + "."


def _build_chord_hint(score: PianoScore) -> str:
    chords = score.metadata.chord_symbols[:_MAX_PROMPT_CHORDS]
    if not chords:
        return "Keep the harmony grounded in the original tonal center and cadence naturally."
    chord_labels = ", ".join(ch.label for ch in chords)
    more = "..." if len(score.metadata.chord_symbols) > len(chords) else ""
    return f"Use these harmonic landmarks where helpful: {chord_labels}{more}."


def _build_lyria_prompt(score: PianoScore) -> str:
    tempo_map = score.metadata.tempo_map
    approx_bpm = round(tempo_map[0].bpm) if tempo_map else 120
    duration_sec = _score_duration_sec(score)
    duration_hint = (
        f"Aim for roughly {duration_sec} seconds of music."
        if duration_sec is not None
        else "Keep the arrangement compact and self-contained."
    )

    return "\n".join(
        [
            "Create a solo acoustic piano arrangement from this source sketch.",
            "Target player: intermediate pianist.",
            "Instrumentation: solo piano only. No vocals, drums, synths, pads, or extra instruments.",
            "Write idiomatic piano texture with a clear right-hand melody and supportive left-hand accompaniment.",
            "Favor broken chords, simple inner voices, octave bass, and playable two-hand voicing.",
            "Avoid virtuosic runs, huge leaps, dense cluster chords, "
            "repeated-note tremolos, or stretches wider than an octave per hand.",
            "Keep the rhythm readable and the harmony clear enough to engrave as approachable sheet music.",
            duration_hint,
            f"Key: {score.metadata.key}.",
            f"Time signature: {score.metadata.time_signature[0]}/{score.metadata.time_signature[1]}.",
            f"Approximate tempo: {approx_bpm} BPM.",
            f"Source right-hand range: {_note_range(score.right_hand)} ({len(score.right_hand)} notes).",
            f"Source left-hand range: {_note_range(score.left_hand)} ({len(score.left_hand)} notes).",
            _build_section_hint(score),
            _build_chord_hint(score),
        ]
    )


def _normalize_audio_bytes(data: Any) -> bytes | None:
    if data is None:
        return None
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, str):
        try:
            return base64.b64decode(data)
        except Exception:  # noqa: BLE001
            return data.encode("utf-8")
    return None


def _response_parts(response: Any) -> list[Any]:
    parts = getattr(response, "parts", None)
    if parts is not None:
        return list(parts)

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return []

    content = getattr(candidates[0], "content", None)
    candidate_parts = getattr(content, "parts", None)
    return list(candidate_parts or [])


def _run_lyria_sync(prompt: str) -> _LyriaOutput:
    genai = _import_google_genai()
    client = (
        genai.Client(api_key=settings.transform_lyria_api_key)
        if settings.transform_lyria_api_key
        else genai.Client()
    )

    config = None
    types_mod = getattr(genai, "types", None)
    if types_mod is not None and hasattr(types_mod, "GenerateContentConfig"):
        config = types_mod.GenerateContentConfig(
            response_modalities=["AUDIO", "TEXT"],
        )

    kwargs = {
        "model": settings.transform_lyria_model,
        "contents": prompt,
    }
    if config is not None:
        kwargs["config"] = config

    response = client.models.generate_content(**kwargs)

    text_parts: list[str] = []
    audio_data: bytes | None = None
    audio_mime_type: str | None = None
    for part in _response_parts(response):
        text = getattr(part, "text", None)
        if text is not None:
            text_parts.append(text)
            continue

        inline = getattr(part, "inline_data", None)
        if inline is None:
            continue

        audio_data = _normalize_audio_bytes(getattr(inline, "data", None))
        audio_mime_type = getattr(inline, "mime_type", None)

    return _LyriaOutput(
        text_parts=text_parts,
        audio_data=audio_data,
        audio_mime_type=audio_mime_type,
    )


def _audio_extension(audio_mime_type: str | None) -> str:
    if audio_mime_type == "audio/wav":
        return ".wav"
    return ".mp3"


class TransformService:
    name = "transform"

    def __init__(self, blob_store: BlobStore | None = None) -> None:
        self.blob_store = blob_store

    def _persist_lyria_output(
        self,
        *,
        job_id: str,
        prompt: str,
        output: _LyriaOutput,
    ) -> None:
        if self.blob_store is None:
            return

        payload = {
            "model": settings.transform_lyria_model,
            "prompt": prompt,
            "text_parts": output.text_parts,
            "audio_mime_type": output.audio_mime_type,
        }
        meta_uri = self.blob_store.put_json(
            f"jobs/{job_id}/transform/lyria-response.json",
            payload,
        )

        audio_uri = None
        if output.audio_data:
            audio_uri = self.blob_store.put_bytes(
                f"jobs/{job_id}/transform/lyria-arrangement{_audio_extension(output.audio_mime_type)}",
                output.audio_data,
            )

        log.info(
            "transform: persisted Lyria sidecar job_id=%s meta_uri=%s audio_uri=%s",
            job_id,
            meta_uri,
            audio_uri,
        )

    async def run(
        self,
        score: PianoScore,
        *,
        job_id: str | None = None,
    ) -> PianoScore:
        job_label = job_id or "-"
        rh_notes = len(score.right_hand)
        lh_notes = len(score.left_hand)

        if not settings.transform_lyria_enabled:
            log.info(
                "transform: Lyria skipped job_id=%s reason=disabled model=%s rh_notes=%d lh_notes=%d",
                job_label,
                settings.transform_lyria_model,
                rh_notes,
                lh_notes,
            )
            return score
        if not score.right_hand and not score.left_hand:
            log.info(
                "transform: Lyria skipped job_id=%s reason=empty_score model=%s",
                job_label,
                settings.transform_lyria_model,
            )
            return score

        prompt = _build_lyria_prompt(score)
        log.info(
            "transform: invoking Lyria job_id=%s model=%s rh_notes=%d lh_notes=%d key=%s time_signature=%s "
            "prompt_chars=%d explicit_api_key=%s",
            job_label,
            settings.transform_lyria_model,
            rh_notes,
            lh_notes,
            score.metadata.key,
            f"{score.metadata.time_signature[0]}/{score.metadata.time_signature[1]}",
            len(prompt),
            bool(settings.transform_lyria_api_key),
        )
        try:
            output = await asyncio.to_thread(_run_lyria_sync, prompt)
        except ImportError as exc:
            log.warning("transform: google-genai unavailable, skipping Lyria (%s)", exc)
            return score
        except Exception as exc:  # noqa: BLE001 - optional model hook
            log.warning("transform: Lyria request failed, keeping original score (%s)", exc)
            return score

        audio_size = len(output.audio_data) if output.audio_data is not None else 0
        log.info(
            "transform: Lyria completed job_id=%s model=%s text_parts=%d has_audio=%s audio_mime_type=%s "
            "audio_bytes=%d",
            job_label,
            settings.transform_lyria_model,
            len(output.text_parts),
            output.audio_data is not None,
            output.audio_mime_type,
            audio_size,
        )

        if job_id is not None:
            try:
                self._persist_lyria_output(job_id=job_id, prompt=prompt, output=output)
            except Exception as exc:  # noqa: BLE001 - best-effort artifact write
                log.warning("transform: failed to persist Lyria sidecar for %s (%s)", job_id, exc)
        else:
            log.info("transform: Lyria output not persisted because job_id was not provided")
        return score
