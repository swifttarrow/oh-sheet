"""Redis-backed cache: YouTube video ID → completed job summary.

Why this exists
---------------
A user submits the same YouTube link twice. The first submit runs the
full pipeline: yt-dlp download → upload to TuneChat → wait for
TuneChat's transcribe + engrave (~3-5 min wall clock). The second
submit should not re-pay that cost — the artifacts already exist on
TuneChat's server.

The cache stores enough metadata to skip ingestion entirely on a hit:
the original ``job_id`` and the TuneChat artifact URLs. The caller
(``routes/jobs.py``) translates a cache hit into an immediate
``JobSummary`` response without dispatching to the runner.

Pattern: ``Cache as durable read-side``. The JobManager remains the
in-flight tracker; this cache is the "completed-jobs lookup" — a
form of denormalized read store. See ADR-002 in the runbook for the
"option B" decision (denormalized vs. JobManager-as-source-of-truth).

Fail-open contract
------------------
Redis going down MUST NOT fail a user request. ``get`` returns None
on any error (so the route falls through to a normal cache miss);
``set`` is a no-op (logged, but never raises). The cache is an
optimization, not a correctness boundary.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Protocol

log = logging.getLogger(__name__)

# Namespace prefix — keep cache keys clear of Celery's broker/result
# keys which share the same Redis db. Discoverable via
# ``redis-cli --scan --pattern 'ohsheet:cache:youtube:*'``.
_KEY_PREFIX = "ohsheet:cache:youtube:"


class _RedisLike(Protocol):
    """Structural type — anything with these two methods works.

    Lets tests pass a 15-line FakeRedis without depending on fakeredis-py.
    Production passes the real redis.Redis client wired up in deps.py.
    """

    def setex(self, key: str, ttl_sec: int, value: str | bytes) -> Any: ...
    def get(self, key: str) -> bytes | None: ...


class YouTubeJobCache:
    """video_id → completed job summary, with TTL and fail-open IO."""

    def __init__(self, redis_client: _RedisLike, ttl_seconds: int) -> None:
        self._redis = redis_client
        self._ttl_seconds = ttl_seconds

    def get(self, video_id: str) -> dict[str, Any] | None:
        """Return the cached entry for ``video_id`` or None.

        Returns None on cache miss, expired entry, or any Redis error.
        Callers should treat None as "run the pipeline normally."
        """
        key = _KEY_PREFIX + video_id
        try:
            raw = self._redis.get(key)
        except Exception as exc:  # noqa: BLE001 — fail open is the point
            log.warning("youtube_cache get failed: %s", exc)
            return None

        if raw is None:
            return None

        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            # Corrupt entry — log and treat as miss. Don't delete it
            # here (would need a separate code path); next ``set`` will
            # overwrite, and the TTL will eventually evict.
            log.warning(
                "youtube_cache corrupt entry video_id=%s err=%s", video_id, exc
            )
            return None

    def set(self, video_id: str, entry: dict[str, Any]) -> None:
        """Cache ``entry`` under ``video_id`` with the configured TTL.

        Never raises. A Redis write failure is logged and swallowed —
        the job already succeeded; missing the cache write costs us
        one wasted re-run on next duplicate submit, not correctness.
        """
        key = _KEY_PREFIX + video_id
        try:
            payload = json.dumps(entry)
        except (TypeError, ValueError) as exc:
            # Programmer error — log loudly. We don't want callers
            # silently feeding us non-serializable objects.
            log.error(
                "youtube_cache refusing non-JSON entry video_id=%s err=%s",
                video_id,
                exc,
            )
            return

        try:
            self._redis.setex(key, self._ttl_seconds, payload)
        except Exception as exc:  # noqa: BLE001 — fail open is the point
            log.warning("youtube_cache set failed: %s", exc)
