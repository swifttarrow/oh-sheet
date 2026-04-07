"""Filesystem-backed BlobStore.

Writes blobs under a configured root directory and returns ``file://`` URIs.
Used for local dev/tests; an ``S3BlobStore`` will live alongside this when we
need to deploy.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class LocalBlobStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- key/path helpers ------------------------------------------------

    def _path_for_key(self, key: str) -> Path:
        normalized = key.lstrip("/")
        if ".." in Path(normalized).parts:
            raise ValueError(f"Refusing key with parent traversal: {key!r}")
        return self.root / normalized

    def _path_from_uri(self, uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            raise ValueError(f"LocalBlobStore only handles file:// URIs, got {uri!r}")
        path = Path(parsed.path).resolve()
        if path != self.root and self.root not in path.parents:
            raise ValueError(f"URI path {path} escapes blob root {self.root}")
        return path

    # ---- bytes -----------------------------------------------------------

    def put_bytes(self, key: str, data: bytes) -> str:
        path = self._path_for_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path.as_uri()

    def get_bytes(self, uri: str) -> bytes:
        return self._path_from_uri(uri).read_bytes()

    # ---- json convenience ------------------------------------------------

    def put_json(self, key: str, payload: dict[str, Any]) -> str:
        return self.put_bytes(key, json.dumps(payload, indent=2).encode("utf-8"))

    def get_json(self, uri: str) -> dict[str, Any]:
        return json.loads(self.get_bytes(uri).decode("utf-8"))

    # ---- misc ------------------------------------------------------------

    @staticmethod
    def hash_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()
