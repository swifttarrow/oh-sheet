"""Tests for the YouTube job cache.

The cache lets a re-submitted YouTube URL short-circuit straight to a
prior successful job's artifacts, skipping the full ingest →
TuneChat round trip (~3-5 minutes saved per duplicate submit).

These tests use a tiny in-memory ``FakeRedis`` rather than fakeredis-py
because the surface area we need is small (get/set/setex) and adding
a test dep for ~15 lines of stub isn't worth it.
"""
from __future__ import annotations

import time

import pytest

from backend.jobs.youtube_cache import YouTubeJobCache


class FakeRedis:
    """Minimal stand-in for redis.Redis covering only the methods the
    cache uses. ``setex`` stores (value, expiry_epoch); ``get`` returns
    None once expired so TTL tests can use a fake clock without
    background timers.
    """

    def __init__(self, now=time.time):
        self._store: dict[str, tuple[bytes, float | None]] = {}
        self._now = now

    def setex(self, key: str, ttl_sec: int, value: str | bytes) -> None:
        if isinstance(value, str):
            value = value.encode("utf-8")
        self._store[key] = (value, self._now() + ttl_sec)

    def get(self, key: str) -> bytes | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expiry = entry
        if expiry is not None and self._now() >= expiry:
            return None
        return value


class FakeRedisExploding:
    """Redis client that raises on every operation — used to verify
    the cache fails OPEN (returns None on get, no-op on set) rather
    than propagating infrastructure errors into request handling.
    """

    def setex(self, key, ttl_sec, value):
        raise ConnectionError("redis is down")

    def get(self, key):
        raise ConnectionError("redis is down")


@pytest.fixture
def cache():
    return YouTubeJobCache(redis_client=FakeRedis(), ttl_seconds=86400)


def test_get_miss_returns_none(cache):
    assert cache.get("dQw4w9WgXcQ") is None


def test_set_then_get_roundtrip(cache):
    entry = {
        "job_id": "j-abc",
        "completed_at": "2026-05-17T12:00:00Z",
        "tunechat_pdf_url": "https://tunechat.example/p/x.pdf",
        "tunechat_musicxml_url": "https://tunechat.example/p/x.xml",
        "tunechat_midi_url": "https://tunechat.example/p/x.mid",
    }
    cache.set("dQw4w9WgXcQ", entry)
    assert cache.get("dQw4w9WgXcQ") == entry


def test_different_video_ids_do_not_collide(cache):
    cache.set("aaaaaaaaaaa", {"job_id": "j-a"})
    cache.set("bbbbbbbbbbb", {"job_id": "j-b"})
    assert cache.get("aaaaaaaaaaa") == {"job_id": "j-a"}
    assert cache.get("bbbbbbbbbbb") == {"job_id": "j-b"}


def test_key_is_namespaced_under_ohsheet_prefix():
    """Internal-shape test — important because the same Redis instance
    is shared with Celery's broker/result backend. Cache keys must NOT
    collide with Celery's namespace.
    """
    fake = FakeRedis()
    cache = YouTubeJobCache(redis_client=fake, ttl_seconds=60)
    cache.set("dQw4w9WgXcQ", {"job_id": "j-abc"})
    keys = list(fake._store.keys())
    assert len(keys) == 1
    assert keys[0].startswith("ohsheet:cache:youtube:")


def test_ttl_expires_entries():
    """Move the fake clock past the TTL and verify the entry is gone."""
    clock = {"t": 1000.0}

    def now():
        return clock["t"]

    cache = YouTubeJobCache(
        redis_client=FakeRedis(now=now),
        ttl_seconds=60,
    )
    cache.set("dQw4w9WgXcQ", {"job_id": "j-abc"})
    assert cache.get("dQw4w9WgXcQ") == {"job_id": "j-abc"}

    clock["t"] += 61  # advance past TTL
    assert cache.get("dQw4w9WgXcQ") is None


def test_get_fails_open_on_redis_error():
    """Redis being down must NEVER fail a job submission. ``get`` must
    return None so the route falls through to the normal pipeline.
    """
    cache = YouTubeJobCache(redis_client=FakeRedisExploding(), ttl_seconds=60)
    assert cache.get("dQw4w9WgXcQ") is None  # no exception


def test_set_fails_open_on_redis_error():
    """Same for writes — a runner finishing a job must not crash if
    the cache write fails. The job already succeeded; the cache miss
    on next submit is acceptable.
    """
    cache = YouTubeJobCache(redis_client=FakeRedisExploding(), ttl_seconds=60)
    cache.set("dQw4w9WgXcQ", {"job_id": "j-abc"})  # no exception
