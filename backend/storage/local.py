"""Backend ``LocalBlobStore`` — defensive subclass of the shared impl.

Kept as a subclass (not a pure re-export) so backend code can rely on
``exists()`` even if a deployment bind-mounts ``./backend`` against an
older ``ohsheet-shared`` wheel that predates the method. When the wheel
already ships ``exists``, we delegate to it; otherwise we fall back to
the same filesystem check.

Historical context: dev ``docker-compose.yml`` now bind-mounts both
``./backend`` and ``./shared`` so drift is unlikely, and prod bakes the
wheel into the image. The shim still costs nothing and removes a class
of footguns if a future deployment mounts only ``./backend``.
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
