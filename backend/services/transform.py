"""Transform stage — post-condense refinement (stub / passthrough).

Planned home for voicing, register, or style transforms on a ``PianoScore``.
For now this stage returns the input unchanged so the pipeline shape is stable.
"""
from __future__ import annotations

from backend.contracts import PianoScore


class TransformService:
    name = "transform"

    async def run(self, score: PianoScore) -> PianoScore:
        return score
