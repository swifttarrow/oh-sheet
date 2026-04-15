"""Re-export from shared package.

Kept as a shim so backend code can ``from backend.storage.local import
LocalBlobStore`` without reaching into ``shared`` directly. This keeps
the import graph one-directional: backend → shared, never shared → backend.
"""
from shared.storage.local import *  # noqa: F401, F403
from shared.storage.local import LocalBlobStore as LocalBlobStore  # noqa: F401
