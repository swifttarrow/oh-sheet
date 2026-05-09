"""Unit tests for backend.services._ytdlp_utils.apply_ytdlp_cookies.

The function has four observable branches:
  1. settings.ytdlp_cookies_path is None (default) → no-op
  2. path points at a missing file → no-op (graceful)
  3. path points at an empty file → no-op (deploy.yml writes empty
     file when secret unset; this is the "safe no-op" contract)
  4. path points at a non-empty file → sets ydl_opts["cookiefile"]

A fifth (error) branch — OSError from stat() — is covered by
monkeypatching Path.stat to raise.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.services._ytdlp_utils import apply_ytdlp_cookies


def _patch_path(monkeypatch: pytest.MonkeyPatch, path: str | None) -> None:
    """Override settings.ytdlp_cookies_path for the duration of a test.

    Uses the already-imported settings module so the change is visible
    to apply_ytdlp_cookies' module-level import.
    """
    from backend.config import settings
    monkeypatch.setattr(settings, "ytdlp_cookies_path", path)


def test_no_op_when_path_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_path(monkeypatch, None)
    opts: dict = {}
    apply_ytdlp_cookies(opts)
    assert "cookiefile" not in opts, "unset path should be a no-op"


def test_no_op_when_path_empty_string(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_path(monkeypatch, "")
    opts: dict = {}
    apply_ytdlp_cookies(opts)
    assert "cookiefile" not in opts, "empty-string path should be a no-op"


def test_no_op_when_file_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    missing = tmp_path / "does-not-exist.txt"
    _patch_path(monkeypatch, str(missing))
    opts: dict = {}
    apply_ytdlp_cookies(opts)
    assert "cookiefile" not in opts, "missing file should be a no-op"


def test_no_op_when_file_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # The "safe no-op" contract — deploy.yml always writes this file,
    # with empty contents when the OHSHEET_YTDLP_COOKIES secret is unset.
    # An empty file must not be passed to yt-dlp.
    empty = tmp_path / "empty.txt"
    empty.touch()
    assert empty.stat().st_size == 0
    _patch_path(monkeypatch, str(empty))

    opts: dict = {}
    apply_ytdlp_cookies(opts)
    assert "cookiefile" not in opts, "empty file should be a no-op"


def test_sets_cookiefile_to_writable_copy_not_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # The source file is bind-mounted :ro in prod, so we copy it to
    # a writable tmp path and point yt-dlp at the copy — NOT the
    # source directly. This test enforces that contract: cookiefile
    # must differ from the source path, and its contents must match.
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\nfoo\tbar\n")
    _patch_path(monkeypatch, str(cookies))

    opts: dict = {}
    apply_ytdlp_cookies(opts)

    assert "cookiefile" in opts, "non-empty file should activate cookies"
    copy_path = Path(opts["cookiefile"])
    assert copy_path != cookies, (
        "cookiefile must point at a writable copy, not the :ro source"
    )
    assert copy_path.read_text() == cookies.read_text(), (
        "copy must mirror source contents"
    )


def test_preserves_existing_opts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # apply_ytdlp_cookies mutates the dict in place — must not drop
    # other keys that callers already set.
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("x")
    _patch_path(monkeypatch, str(cookies))

    opts: dict = {"quiet": True, "socket_timeout": 15}
    apply_ytdlp_cookies(opts)
    assert opts["quiet"] is True
    assert opts["socket_timeout"] == 15
    # cookiefile now points at a tmp copy, not the source itself.
    assert "cookiefile" in opts
    assert Path(opts["cookiefile"]).read_text() == "x"


def test_no_op_when_copy_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # If /tmp is full or unwritable, apply_ytdlp_cookies must not
    # crash the job — it logs and runs yt-dlp anonymously.
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("x")
    _patch_path(monkeypatch, str(cookies))

    import shutil as _shutil

    from backend.services import _ytdlp_utils

    def _fail(*args: object, **kwargs: object) -> None:
        raise OSError("simulated copyfile failure")

    monkeypatch.setattr(_ytdlp_utils.shutil, "copyfile", _fail)
    # Also patch the shutil alias used at call-site module scope, in
    # case the re-export resolves differently.
    monkeypatch.setattr(_shutil, "copyfile", _fail)

    opts: dict = {}
    apply_ytdlp_cookies(opts)
    assert "cookiefile" not in opts


def test_stat_raising_oserror_is_graceful(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # Simulates a race / permission-denied on .stat(). Contract: the
    # function must not crash; yt-dlp proceeds without cookies.
    path = tmp_path / "blocked.txt"
    path.write_text("x")
    _patch_path(monkeypatch, str(path))

    def _raise(self, *args, **kwargs):  # noqa: ARG001 — shimming Path.stat
        # Accept *args / **kwargs so Python-internal callers (like pytest's
        # tmp_path teardown via follow_symlinks=True) don't hit TypeError
        # while our monkeypatch of the Path class is still in scope.
        raise OSError("simulated stat() failure")

    monkeypatch.setattr(Path, "stat", _raise)

    opts: dict = {}
    apply_ytdlp_cookies(opts)  # must not raise
    assert "cookiefile" not in opts


def test_concurrent_calls_get_distinct_tmp_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Regression: two calls in the same process used to share one
    pid-scoped tmp path, so concurrent yt-dlp downloads in one Celery
    worker would clobber each other's rotated session tokens. Now each
    call must get its own tmp file.
    """
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\nfoo\tbar\n")
    _patch_path(monkeypatch, str(cookies))

    opts_a: dict = {}
    apply_ytdlp_cookies(opts_a)
    opts_b: dict = {}
    apply_ytdlp_cookies(opts_b)

    assert "cookiefile" in opts_a and "cookiefile" in opts_b
    assert opts_a["cookiefile"] != opts_b["cookiefile"], (
        "concurrent calls must not share a tmp path"
    )
    # Both copies still mirror the source.
    assert Path(opts_a["cookiefile"]).read_text() == cookies.read_text()
    assert Path(opts_b["cookiefile"]).read_text() == cookies.read_text()
