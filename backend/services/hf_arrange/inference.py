"""HF inference: MIDI bytes in → MIDI bytes out (identity stub by default)."""
from __future__ import annotations

import logging
from typing import Literal

log = logging.getLogger(__name__)

ArrangeHfInferenceMode = Literal["identity"]


def run_hf_midi_inference(
    midi_in: bytes,
    inference_mode: ArrangeHfInferenceMode,
) -> bytes:
    """Run configured MIDI→MIDI model.

    ``inference_mode``:
    - ``identity`` — return input unchanged (tests + integration without weights).
    """
    if inference_mode == "identity":
        return midi_in
    raise ValueError(f"Unknown arrange_hf_inference_mode: {inference_mode!r}")
