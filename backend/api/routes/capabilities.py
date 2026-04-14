"""GET /v1/capabilities - feature-availability probe for the frontend.

D-22: The frontend calls this at upload-screen load time so it can render
the refine checkbox in the appropriate state:
    refine_available=true  -> enabled checkbox + tooltip
    refine_available=false -> disabled checkbox + helper text "AI refinement
                              not configured on this server"

Deliberately minimal response shape. Avoid disclosing: the key itself, its
length, the model name, the prompt version, or any build-time metadata
that adds attacker surface without earning its keep in the frontend.

No auth: the project has no auth layer yet; this endpoint is readable by
anyone who can reach the API, which is the same posture as every other
route today.
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from backend.config import settings

router = APIRouter()


class CapabilitiesResponse(BaseModel):
    """Single-field response. Frontend's Capabilities Dart model
    (frontend/lib/api/models.dart, Plan 03) mirrors this exactly.
    """

    refine_available: bool


@router.get("/capabilities", response_model=CapabilitiesResponse)
def get_capabilities() -> CapabilitiesResponse:
    """D-22: True iff the backend has an Anthropic API key configured.

    Note: does NOT check OHSHEET_REFINE_KILL_SWITCH. The switch is a
    server-side emergency brake - when it's on, refine submissions are
    silently coerced to unrefined at job-creation time (Phase 1, CFG-06).
    Exposing the switch state via capabilities would let a client distinguish
    'temporarily killed' from 'not configured', which is operator-level
    information the frontend does not need. D-33 keeps the kill switch out
    of product surface area.
    """
    return CapabilitiesResponse(
        refine_available=settings.anthropic_api_key is not None,
    )
