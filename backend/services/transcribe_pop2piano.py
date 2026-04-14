"""Pop2Piano transcription — audio → piano MIDI in one shot.

Replaces the Demucs + Basic Pitch pipeline with a single transformer
pass via ``sweetcocoa/pop2piano``. The model is a seq2seq transformer
trained on pop music → piano covers and produces a ``pretty_midi``
object directly from the input waveform.

The pretty_midi output is converted to ``NoteEvent`` tuples and returned
as a single ``PIANO`` track so the caller can feed it through the same
post-processing pipeline (melody/bass extraction, onset/duration
refinement, key/chord/tempo estimation) as the single-mix Basic Pitch
path.

If Pop2Piano's dependencies are not installed or inference fails for any
reason, the caller should fall back to the old Demucs+BP path — this
module raises on failure rather than returning stubs.

Dependencies (``[pop2piano]`` extra in pyproject.toml):
  torch, transformers, librosa, resampy, scipy, pretty_midi,
  essentia==2.1b6.dev1389
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.config import settings
from backend.services.transcription_cleanup import NoteEvent

log = logging.getLogger(__name__)

# Cached model + processor — loading costs several seconds and ~1 GB
# of memory, so we load once per process with double-checked locking
# (same pattern as Basic Pitch in transcribe_inference.py).
_P2P_MODEL: Any = None
_P2P_PROCESSOR: Any = None
_P2P_LOCK = threading.Lock()


@dataclass
class Pop2PianoStats:
    """Diagnostics emitted by a single Pop2Piano inference run."""
    model_id: str = ""
    note_count: int = 0
    audio_duration_sec: float = 0.0
    skipped: bool = False
    warnings: list[str] = field(default_factory=list)

    def as_warnings(self) -> list[str]:
        out = list(self.warnings)
        if self.note_count == 0 and not self.skipped:
            out.append("pop2piano: model produced zero notes")
        return out


def _load_pop2piano() -> tuple[Any, Any]:
    """Lazy-load the Pop2Piano model + processor. Cached for process lifetime."""
    global _P2P_MODEL, _P2P_PROCESSOR
    if _P2P_MODEL is not None and _P2P_PROCESSOR is not None:
        return _P2P_MODEL, _P2P_PROCESSOR

    with _P2P_LOCK:
        if _P2P_MODEL is not None and _P2P_PROCESSOR is not None:
            return _P2P_MODEL, _P2P_PROCESSOR

        import torch  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            Pop2PianoForConditionalGeneration,
            Pop2PianoProcessor,
        )

        repo_id = settings.pop2piano_model
        log.info("Loading Pop2Piano model from %s", repo_id)

        model = Pop2PianoForConditionalGeneration.from_pretrained(repo_id)
        processor = Pop2PianoProcessor.from_pretrained(repo_id)

        device_name = settings.pop2piano_device
        if device_name is None:
            if torch.cuda.is_available():
                device_name = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_name = "mps"
            else:
                device_name = "cpu"

        model = model.to(device_name)
        log.info("Pop2Piano loaded on device=%s", device_name)

        _P2P_MODEL = model
        _P2P_PROCESSOR = processor
        return _P2P_MODEL, _P2P_PROCESSOR


def _pretty_midi_to_note_events(pm: Any) -> list[NoteEvent]:
    """Convert a pretty_midi.PrettyMIDI to our internal NoteEvent list.

    Each note becomes ``(start_sec, end_sec, pitch, amplitude, [])``
    where amplitude is ``velocity / 127`` (inverse of the encoding in
    ``_event_to_note`` in transcribe_result.py).
    """
    events: list[NoteEvent] = []
    for instrument in pm.instruments:
        for note in instrument.notes:
            amplitude = note.velocity / 127.0
            events.append((
                float(note.start),
                float(note.end),
                int(note.pitch),
                float(amplitude),
                [],  # no pitch bends
            ))
    events.sort(key=lambda e: (e[0], e[2]))
    return events


def run_pop2piano(audio_path: Path) -> tuple[list[NoteEvent], Any, Pop2PianoStats]:
    """Run Pop2Piano inference on a single audio file.

    Returns:
        events: NoteEvent list suitable for the post-processing pipeline.
        pm: The raw pretty_midi.PrettyMIDI from Pop2Piano (for blob MIDI).
        stats: Diagnostics for QualitySignal warnings.

    Raises on any failure (missing deps, model load, inference) — the
    caller is expected to catch and fall back to the Demucs+BP path.
    """
    import librosa  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    model, processor = _load_pop2piano()
    sr = settings.pop2piano_sample_rate

    log.info("Pop2Piano: loading audio %s at sr=%d", audio_path, sr)
    audio, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    audio_duration_sec = len(audio) / sr

    inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    log.info("Pop2Piano: generating (audio=%.1fs)", audio_duration_sec)
    with torch.no_grad():
        model_output = model.generate(
            input_features=inputs["input_features"],
            composer=settings.pop2piano_composer,
        )

    decoded = processor.batch_decode(
        token_ids=model_output,
        feature_extractor_output=inputs,
    )["pretty_midi_objects"][0]

    events = _pretty_midi_to_note_events(decoded)

    stats = Pop2PianoStats(
        model_id=settings.pop2piano_model,
        note_count=len(events),
        audio_duration_sec=audio_duration_sec,
    )
    log.info(
        "Pop2Piano: done — %d notes from %.1fs audio",
        len(events), audio_duration_sec,
    )
    return events, decoded, stats
