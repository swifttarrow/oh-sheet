from __future__ import annotations

import os

from fastapi import APIRouter

from backend.contracts import SCHEMA_VERSION

router = APIRouter()

_COMMIT_SHA = os.environ.get("COMMIT_SHA", "unknown")


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "schema_version": SCHEMA_VERSION, "commit": _COMMIT_SHA}
