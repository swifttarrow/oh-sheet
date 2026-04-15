"""Backend `LocalBlobStore`: shared implementation plus API-only helpers.

Docker Compose bind-mounts ``./backend`` into the dev container but not
``shared``, so the orchestrator can run newer route code against an older
``ohsheet-shared`` wheel. Subclassing here keeps ``exists()`` available for
job integrity checks regardless of wheel age; when the wheel includes
``exists``, we delegate to it.
"""
from __future__ import annotations

from shared.storage.local import LocalBlobStore as _SharedLocalBlobStore


class LocalBlobStore(_SharedLocalBlobStore):
    def exists(self, uri: str) -> bool:
        try:
            return super().exists(uri)
        except AttributeError:
            try:
                path = self._path_from_uri(uri)
            except ValueError:
                return False
            return path.is_file()
