"""Smoke test for the LilyPond / MuseScore PDF render path.

Gates on whether a real PDF renderer is on ``$PATH``. When neither
``lilypond`` nor a ``musescore*`` binary is installed, this is skipped
so it doesn't break dev machines without the engraver tooling. The
production Docker image installs ``lilypond`` via apt (see
``Dockerfile``), so CI runs inside that image will execute this path.
"""
from __future__ import annotations

import shutil

import pytest

from backend.services.engrave import _STUB_PDF, _engrave_sync, _render_pdf_bytes
from tests.fixtures import load_score_fixture


def _has_real_renderer() -> bool:
    if shutil.which("musicxml2ly") and shutil.which("lilypond"):
        return True
    return any(
        shutil.which(b) for b in ("musescore4", "musescore3", "mscore", "MuseScore4")
    )


pytestmark = pytest.mark.skipif(
    not _has_real_renderer(),
    reason="no lilypond/MuseScore on PATH — production image installs lilypond",
)


def test_engrave_emits_real_pdf_not_stub() -> None:
    """A fixture routed through ``_engrave_sync`` produces a real PDF.

    Uses ``c_major_scale`` as the smoke case: it's small, uses both
    hands, and has been rendering cleanly through music21 since PR-1.
    """
    score = load_score_fixture("c_major_scale")
    pdf_bytes, musicxml_bytes, _midi_bytes, _ly_bytes, _chord_count = _engrave_sync(
        score, title="Smoke Test", composer="pytest",
    )

    assert musicxml_bytes.startswith(b"<?xml"), "musicxml should be real xml"
    assert pdf_bytes != _STUB_PDF, (
        "a real renderer is on PATH but _render_pdf_bytes fell through to the stub"
    )
    assert pdf_bytes.startswith(b"%PDF-"), "output is not a PDF"
    assert len(pdf_bytes) > 1024, (
        f"PDF suspiciously small ({len(pdf_bytes)} bytes) — likely a render failure"
    )


def test_render_pdf_bytes_direct_call() -> None:
    """``_render_pdf_bytes`` on plausible MusicXML returns a real PDF.

    Keeps the pdf path under test even if future refactors move the
    high-level ``_engrave_sync`` plumbing around.
    """
    score = load_score_fixture("single_note")
    _, musicxml_bytes, _, _, _ = _engrave_sync(score, title="", composer="")

    pdf_bytes, _ly_bytes = _render_pdf_bytes(musicxml_bytes)
    assert pdf_bytes != _STUB_PDF
    assert pdf_bytes.startswith(b"%PDF-")
