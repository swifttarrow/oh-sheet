"""AMT-APC piano-cover transcription wrapper (Phase 8).

Wraps `misya11p/amt-apc <https://github.com/misya11p/amt-apc>`_ — an
MIT-licensed hFT-Transformer descendant trained on Billboard-pop /
piano-cover pairs. Unlike Kong (faithful transcription) and Basic Pitch
(generic polyphonic AMT), AMT-APC produces a *pianistic cover*:
idiomatic LH accompaniment patterns, melody re-voicings, and
arrangement decisions a human cover pianist would make. The output is
a single piano stream (no melody/bass split) suitable for direct
hand-off to the engraver, skipping the rules-based arranger.

Routing policy (enforced upstream in ``transcribe.py``):

  * Run on the *instrumental* stem (``bass + other`` from Demucs) when
    available, falling back to the full mix when separation hasn't run.
    AMT-APC was trained on full-mix audio so raw input is acceptable;
    routing through the instrumental stem still helps because vocals
    bias the model toward melody-only output.
  * Bound to the ``pop_cover`` :class:`PipelineVariant`. The dispatcher
    invokes AMT-APC when the user picks "Piano cover" in the UI
    (``user_hint == "cover"``) OR ``settings.amt_apc_enabled`` is True.

License:
    AMT-APC is MIT-licensed end-to-end (model code, weights, training
    pipeline). It supersedes Pop2Piano for cover-style transcription —
    Pop2Piano's upstream repo has no LICENSE file, which the strategy
    doc §G3 flags as a legal blocker. The ``[amt_apc]`` extra in
    pyproject.toml pins a vetted upstream commit so weight provenance
    stays auditable.
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

# Cached transcriptor — model load is multi-second + ~600 MB of RAM,
# so we hold one instance per process behind a double-checked lock,
# matching the kong / pop2piano caching pattern.
_AMT_APC_MODEL: Any = None
_AMT_APC_LOCK = threading.Lock()


@dataclass
class AmtApcStats:
    """Diagnostics emitted by a single AMT-APC inference run."""
    model_id: str = "amt_apc"
    style: str = ""
    note_count: int = 0
    audio_duration_sec: float = 0.0
    skipped: bool = False
    warnings: list[str] = field(default_factory=list)

    def as_warnings(self) -> list[str]:
        out = list(self.warnings)
        if self.note_count == 0 and not self.skipped:
            out.append("amt_apc: model produced zero notes")
        return out


def _select_device(override: str | None) -> str:
    """Pick a torch device, mirroring the kong / pop2piano logic."""
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


def _load_amt_apc() -> Any:
    """Lazy-load the AMT-APC piano-cover transcriptor. Cached for process lifetime."""
    global _AMT_APC_MODEL
    if _AMT_APC_MODEL is not None:
        return _AMT_APC_MODEL

    with _AMT_APC_LOCK:
        if _AMT_APC_MODEL is not None:
            return _AMT_APC_MODEL

        try:
            # The upstream package's public surface is ``amt_apc.Transcriber``
            # (exposed via __init__.py) — load weights eagerly so the first
            # job pays the model-load cost upfront instead of on-demand.
            from amt_apc import Transcriber  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "amt_apc is not installed; install the [amt_apc] extra "
                "to enable AMT-APC piano-cover transcription"
            ) from exc

        device = _select_device(settings.amt_apc_device)
        log.info(
            "Loading AMT-APC transcriptor on device=%s style=%s checkpoint=%s",
            device, settings.amt_apc_style,
            settings.amt_apc_checkpoint_path or "<bundled>",
        )
        kwargs: dict[str, Any] = {"device": device}
        if settings.amt_apc_checkpoint_path:
            kwargs["checkpoint_path"] = settings.amt_apc_checkpoint_path
        _AMT_APC_MODEL = Transcriber(**kwargs)
        return _AMT_APC_MODEL


def _pretty_midi_to_note_events(pm: Any) -> list[NoteEvent]:
    """Convert a pretty_midi.PrettyMIDI to our internal NoteEvent list.

    Mirrors :func:`transcribe_pop2piano._pretty_midi_to_note_events`:
    each note becomes ``(start_sec, end_sec, pitch, amplitude, [])``
    where amplitude is ``velocity / 127``. AMT-APC does not emit pitch
    bends — the cover-mode output is quantized piano notation.
    """
    events: list[NoteEvent] = []
    for instrument in pm.instruments:
        for note in instrument.notes:
            amplitude = float(note.velocity) / 127.0
            events.append((
                float(note.start),
                float(note.end),
                int(note.pitch),
                float(max(0.0, min(1.0, amplitude))),
                [],
            ))
    events.sort(key=lambda e: (e[0], e[2]))
    return events


def run_amt_apc(audio_path: Path) -> tuple[list[NoteEvent], Any, AmtApcStats]:
    """Run AMT-APC piano-cover transcription on the given audio path.

    Returns ``(note_events, pretty_midi, stats)``. Raises ``ImportError``
    when the optional dep isn't installed; callers in ``transcribe.py``
    catch this and fall back to the Kong / Basic Pitch path so a deploy
    without the [amt_apc] extra still produces transcriptions (just
    without cover-mode rearrangement).

    The returned ``pretty_midi`` is the raw model output; the caller
    typically serializes it as the ``transcription_midi_uri`` artifact
    while the ``note_events`` flow into the contract's MidiTracks.
    """
    stats = AmtApcStats(style=settings.amt_apc_style)
    try:
        model = _load_amt_apc()
    except ImportError:
        stats.skipped = True
        stats.warnings.append("amt_apc: package not installed")
        raise

    import librosa  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    sr = settings.amt_apc_sample_rate
    log.info("amt_apc: loading audio %s at sr=%d", audio_path, sr)
    audio, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    duration_sec = float(len(audio)) / float(sr)
    stats.audio_duration_sec = duration_sec

    log.info(
        "amt_apc: generating cover (style=%s audio=%.1fs)",
        settings.amt_apc_style, duration_sec,
    )
    # The upstream Transcriber exposes ``transcribe(audio, sr=..., style=...)``
    # → pretty_midi.PrettyMIDI. We pass ``style`` so the model emits
    # arrangement decisions appropriate for the requested register.
    pm = model.transcribe(audio, sr=sr, style=settings.amt_apc_style)

    note_events = _pretty_midi_to_note_events(pm)
    stats.note_count = len(note_events)
    if not note_events:
        stats.warnings.append("amt_apc: zero notes returned by model")

    log.info(
        "amt_apc: done — %d notes from %.1fs audio (style=%s)",
        stats.note_count, duration_sec, settings.amt_apc_style,
    )
    return note_events, pm, stats
