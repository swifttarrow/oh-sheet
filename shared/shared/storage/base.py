"""BlobStore protocol — implementations live alongside this file.

The pipeline contracts use the Claim-Check pattern: heavy media files
(audio, MIDI, MusicXML, PDF) live in blob storage, and only URIs cross the
wire. The store is responsible for materializing those URIs.
"""
from __future__ import annotations

from typing import Any, Protocol


class BlobStore(Protocol):
    def put_bytes(self, key: str, data: bytes) -> str:
        """Write raw bytes under ``key``; return a URI usable by ``get_bytes``."""
        ...

    def get_bytes(self, uri: str) -> bytes:
        """Read raw bytes from a URI previously returned by ``put_bytes``."""
        ...

    def put_json(self, key: str, payload: dict[str, Any]) -> str:
        """Convenience: serialize ``payload`` as JSON and write it as bytes."""
        ...

    def get_json(self, uri: str) -> dict[str, Any]:
        ...
