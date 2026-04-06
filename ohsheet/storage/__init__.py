"""Blob storage abstraction (Claim-Check pattern)."""

from ohsheet.storage.base import BlobStore
from ohsheet.storage.local import LocalBlobStore

__all__ = ["BlobStore", "LocalBlobStore"]
