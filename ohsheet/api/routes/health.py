from __future__ import annotations

from fastapi import APIRouter

from ohsheet.contracts import SCHEMA_VERSION

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "schema_version": SCHEMA_VERSION}
