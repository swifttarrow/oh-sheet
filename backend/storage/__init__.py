"""Blob storage abstraction (Claim-Check pattern)."""

from backend.storage.base import BlobStore
from backend.storage.local import LocalBlobStore

__all__ = ["BlobStore", "LocalBlobStore"]
