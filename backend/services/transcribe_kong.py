"""ByteDance Kong piano-transcription wrapper (Phase 6).

Wraps ``piano_transcription_inference.PianoTranscription`` — the
high-resolution piano AMT from Kong et al. 2021 — and converts its
output into the same internal representations the rest of the
transcribe pipeline already speaks:

  * ``NoteEvent`` tuples per piano note (onset/offset/pitch/velocity)
  * ``RealtimePedalEvent`` instances per sustained pedal segment

The PyPI package is wrapped in a try/except at the call site so a
missing ``piano_transcription_inference`` install raises ``ImportError``
which the dispatcher in ``transcribe.py`` converts into a fallback to
the Basic Pitch stems path. Failures inside Kong's inference (weight
fetch error, CUDA OOM) bubble the same way.

Routing policy (enforced upstream in ``transcribe.py``):

  * Kong is run on a piano-ish *summed stem* (``bass + other`` from
    Demucs), never on raw input audio. Edwards et al. 2024 documents
    Kong's MAESTRO-overfit behavior: −19.2 F1 on pitch-shift, −10.1
    on reverb. Demucs gives us a stem clean enough to play to Kong's
    strengths.
  * The dispatcher gates on vocal-energy / user_hint to decide
    between Kong and the legacy Basic Pitch stems path.

License: ``piano_transcription_inference`` is MIT. Pre-trained
weights ship under MIT via the upstream's Zenodo release.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.config import settings
from backend.contracts import RealtimePedalEvent
from backend.services.transcription_cleanup import NoteEvent

log = logging.getLogger(__name__)

# Cached transcriptor — model load is ~2–4 s and ~700 MB of RAM, so
# we hold one instance per process behind a double-checked lock,
# matching the pop2piano + basic_pitch caching pattern.
_KONG_MODEL: Any = None
_KONG_LOCK = threading.Lock()


@dataclass
class KongStats:
    """Diagnostics emitted by a single Kong inference run."""
    model_id: str = "piano_transcription_inference"
    note_count: int = 0
    pedal_count: int = 0
    audio_duration_sec: float = 0.0
    skipped: bool = False
    warnings: list[str] = field(default_factory=list)

    def as_warnings(self) -> list[str]:
        out = list(self.warnings)
        if self.note_count == 0 and not self.skipped:
            out.append("kong: model produced zero notes")
        return out


def _select_device(override: str | None) -> str:
    """Pick a torch device, mirroring the pop2piano logic."""
    if override:
        return override
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_kong() -> Any:
    """Lazy-load the Kong piano transcriptor. Cached for process lifetime."""
    global _KONG_MODEL
    if _KONG_MODEL is not None:
        return _KONG_MODEL

    with _KONG_LOCK:
        if _KONG_MODEL is not None:
            return _KONG_MODEL

        try:
            from piano_transcription_inference import (  # noqa: PLC0415
                PianoTranscription,
            )
        except ImportError as exc:
            raise ImportError(
                "piano_transcription_inference is not installed; "
                "install the [kong] extra to enable Kong pedal AMT"
            ) from exc

        device = _select_device(settings.kong_device)
        log.info(
            "Loading Kong piano transcriptor on device=%s checkpoint=%s",
            device, settings.kong_checkpoint_path or "<bundled>",
        )
        kwargs: dict[str, Any] = {"device": device}
        if settings.kong_checkpoint_path:
            kwargs["checkpoint_path"] = settings.kong_checkpoint_path
        _KONG_MODEL = PianoTranscription(**kwargs)
        return _KONG_MODEL


def _load_audio_for_kong(audio_path: Path) -> tuple[Any, float]:
    """Load audio at Kong's expected sample rate.

    ``piano_transcription_inference`` expects mono 16 kHz float32. The
    upstream ``load_audio`` helper handles ffmpeg-decodable formats —
    we delegate to it so format-fallback (mp3 / wav / flac / m4a)
    behaves the same way as the upstream's CLI.
    """
    try:
        from piano_transcription_inference import (  # noqa: PLC0415
            sample_rate as KONG_SR,
        )
        from piano_transcription_inference.utilities import (  # noqa: PLC0415
            load_audio,
        )
    except ImportError as exc:
        raise ImportError(
            "piano_transcription_inference utilities unavailable"
        ) from exc

    audio, _ = load_audio(str(audio_path), sr=KONG_SR, mono=True)
    duration_sec = float(len(audio)) / float(KONG_SR)
    return audio, duration_sec


def _midi_events_to_note_events(midi_events: list[dict]) -> list[NoteEvent]:
    """Convert Kong's note dicts into the pipeline's ``NoteEvent`` tuples.

    Each Kong note dict has ``onset_time``, ``offset_time``,
    ``midi_note``, and ``velocity`` (0–127). The internal pipeline
    stores velocity as an *amplitude* in [0, 1] and uses the same
    encoding as Basic Pitch's ``int(round(127 * amp))`` round-trip.
    """
    events: list[NoteEvent] = []
    for n in midi_events:
        amplitude = float(n.get("velocity", 0)) / 127.0
        events.append((
            float(n["onset_time"]),
            float(n["offset_time"]),
            int(n["midi_note"]),
            float(max(0.0, min(1.0, amplitude))),
            [],  # Kong does not emit pitch bends
        ))
    events.sort(key=lambda e: (e[0], e[2]))
    return events


def _pedal_events_from_kong(
    pedal_events: list[dict],
    *,
    cc: int = 64,
    min_confidence: float | None = None,
) -> list[RealtimePedalEvent]:
    """Convert Kong's pedal dicts into ``RealtimePedalEvent`` instances.

    Kong's pedal output is a list of ``{onset_time, offset_time}``
    segments — the model regresses sustain-pedal on/off intervals
    rather than emitting per-frame CC values. We label them as CC64
    (sustain) since that's the only pedal Kong tracks; future
    refinements (sostenuto/una_corda) would land here as new ``cc``
    values. Confidence floors via ``min_confidence`` drop low-quality
    segments so the engraver doesn't render visually noisy
    ``Ped. ___ *`` brackets.
    """
    threshold = (
        min_confidence if min_confidence is not None
        else settings.kong_pedal_min_confidence
    )
    out: list[RealtimePedalEvent] = []
    for ev in pedal_events:
        # Kong's segments don't always carry a confidence; default to
        # 1.0 when missing so the legacy "always emit" behavior is
        # preserved. When a model upgrade starts emitting confidence
        # the floor will start gating real values.
        confidence = float(ev.get("confidence", 1.0))
        if confidence < threshold:
            continue
        on = float(ev["onset_time"])
        off = float(ev["offset_time"])
        if off <= on:
            continue
        out.append(RealtimePedalEvent(
            cc=cc,
            onset_sec=on,
            offset_sec=off,
            confidence=max(0.0, min(1.0, confidence)),
        ))
    out.sort(key=lambda p: (p.onset_sec, p.offset_sec))
    return out


def run_kong(
    audio_path: Path,
) -> tuple[list[NoteEvent], list[RealtimePedalEvent], KongStats]:
    """Run Kong piano transcription on the given audio path.

    Returns ``(note_events, pedal_events, stats)``. Raises
    ``ImportError`` when the optional dep isn't installed; callers in
    ``transcribe.py`` catch this and fall back to the Basic Pitch
    stems path so a deploy without the [kong] extra still produces
    transcriptions (just without sustain pedal markings).
    """
    stats = KongStats()
    try:
        model = _load_kong()
    except ImportError:
        stats.skipped = True
        stats.warnings.append("kong: piano_transcription_inference not installed")
        raise

    audio, duration = _load_audio_for_kong(audio_path)
    stats.audio_duration_sec = duration

    # ``transcribe`` writes a MIDI file as a side-effect AND returns
    # a structured dict — we pass a temp path to satisfy the API but
    # the dict is the source of truth for our pipeline.
    import tempfile  # noqa: PLC0415

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=True) as tmp:
        result = model.transcribe(audio, tmp.name)

    midi_events = result.get("est_note_events") or []
    pedal_events = (
        result.get("est_pedal_events")
        or result.get("pedal_events")
        or []
    )

    note_events = _midi_events_to_note_events(midi_events)
    realtime_pedals = _pedal_events_from_kong(pedal_events)

    stats.note_count = len(note_events)
    stats.pedal_count = len(realtime_pedals)
    if not note_events:
        stats.warnings.append("kong: zero notes returned by model")

    log.info(
        "kong: notes=%d pedals=%d duration=%.2fs",
        stats.note_count, stats.pedal_count, duration,
    )
    return note_events, realtime_pedals, stats
