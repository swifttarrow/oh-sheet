"""Shared torch helpers for services that call into PyTorch.

Kept deliberately minimal — one function. Extracted so
:mod:`stem_separation` and :mod:`crepe_melody` can share the same
device-selection logic instead of maintaining two copies of the
four-line function.
"""
from __future__ import annotations


def pick_device(preferred: str | None = None) -> str:
    """Pick a torch device string. Auto-selects cuda → mps → cpu.

    An explicit ``preferred`` value is trusted verbatim — no
    availability check — so tests can force ``"cpu"`` even on a CUDA
    box without paying the probe cost.
    """
    if preferred:
        return preferred

    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
