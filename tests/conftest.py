from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import backend.workers.arrange  # noqa: F401
import backend.workers.condense  # noqa: F401
import backend.workers.humanize  # noqa: F401

# Import monolith worker modules so their tasks are registered on the celery_app.
import backend.workers.ingest  # noqa: F401
import backend.workers.refine  # noqa: F401
import backend.workers.separate  # noqa: F401
import backend.workers.transcribe  # noqa: F401
import backend.workers.transform  # noqa: F401
from backend.api import deps
from backend.config import settings
from backend.main import create_app
from backend.services import engrave_local as engrave_local_module
from backend.services import ml_engraver_client as ml_engraver_module
from backend.services import separate as separate_module
from backend.services import transcribe as transcribe_module
from backend.workers.celery_app import celery_app as _celery_app

_FAKE_ML_MUSICXML = (
    b'<?xml version="1.0" encoding="UTF-8" standalone="no"?>'
    b'<score-partwise version="3.1"><part id="P1"/></score-partwise>'
)

# Real-but-tiny MusicXML the local engrave stub returns. Padded above the
# 200-byte sanity floor that ``score_to_musicxml`` enforces so anything
# downstream that re-validates the payload still treats it as well-formed.
_FAKE_LOCAL_MUSICXML = (
    b'<?xml version="1.0" encoding="UTF-8" standalone="no"?>'
    b'<score-partwise version="4.0"><part-list>'
    b'<score-part id="P1"><part-name>Piano</part-name></score-part>'
    b'</part-list><part id="P1"><measure number="1">'
    b'<note><pitch><step>C</step><octave>4</octave></pitch>'
    b'<duration>4</duration><type>whole</type></note>'
    b'</measure></part></score-partwise>'
)
assert len(_FAKE_LOCAL_MUSICXML) > 200  # guard the engrave_local size floor


@pytest.fixture(autouse=True)
def isolated_blob_root(tmp_path: Path, monkeypatch):
    """Each test gets a fresh blob root and fresh DI singletons."""
    blob = tmp_path / "blob"
    blob.mkdir()
    monkeypatch.setattr(settings, "blob_root", blob)

    deps.get_blob_store.cache_clear()
    deps.get_runner.cache_clear()
    deps.get_job_manager.cache_clear()
    yield
    deps.get_blob_store.cache_clear()
    deps.get_runner.cache_clear()
    deps.get_job_manager.cache_clear()


def _build_fake_transcription_result():
    """Construct a minimal-but-valid TranscriptionResult for tests.

    Replaces the previous reliance on ``_stub_result`` (which now raises
    ``TranscriptionFailure``). Tests that need a successful transcription
    handoff still get a shape-correct payload; tests exercising real
    behavior opt out via ``@pytest.mark.real_transcribe``.
    """
    from backend.contracts import (
        SCHEMA_VERSION,
        HarmonicAnalysis,
        InstrumentRole,
        MidiTrack,
        Note,
        QualitySignal,
        TempoMapEntry,
        TranscriptionResult,
    )
    return TranscriptionResult(
        schema_version=SCHEMA_VERSION,
        midi_tracks=[
            MidiTrack(
                notes=[
                    Note(pitch=60, onset_sec=0.0, offset_sec=0.5, velocity=80),
                    Note(pitch=64, onset_sec=0.5, offset_sec=1.0, velocity=80),
                    Note(pitch=67, onset_sec=1.0, offset_sec=1.5, velocity=80),
                    Note(pitch=72, onset_sec=1.5, offset_sec=2.0, velocity=80),
                ],
                instrument=InstrumentRole.MELODY,
                program=None,
                confidence=0.7,
            ),
        ],
        analysis=HarmonicAnalysis(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
            chords=[],
            sections=[],
        ),
        quality=QualitySignal(
            overall_confidence=0.5,
            warnings=["test-fake transcription"],
        ),
    )


@pytest.fixture
def skip_real_transcription(monkeypatch):
    """Opt-in: replace ``TranscribeService.run`` with a stub-returning fake.

    Most tests exercise pipeline orchestration with fake audio bytes that
    real Basic Pitch can't decode — they should request this fixture (or
    rely on the default autouse layer below) to keep their pipelines
    moving. Tests that exercise real Basic Pitch inference mark themselves
    with ``@pytest.mark.real_transcribe`` to opt out of the default stub
    layer; they do not request this fixture.
    """
    async def _fake_run(self, payload, *, job_id=None, variant=None):
        # ``variant`` was added in Phase 8 so the dispatcher can pick the
        # AMT-APC cover path; the test stub ignores it (the fake result
        # is identical for faithful and cover modes — runner-level cover
        # routing is verified separately in test_celery_dispatch.py).
        del variant  # noqa: F841 — accepted for signature parity, not used
        stub = _build_fake_transcription_result()
        if self.blob_store is not None and job_id is not None:
            fake_midi = b"MThd\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00"
            uri = self.blob_store.put_bytes(
                f"jobs/{job_id}/transcription/basic-pitch.mid", fake_midi,
            )
            stub = stub.model_copy(update={"transcription_midi_uri": uri})
        return stub

    monkeypatch.setattr(transcribe_module.TranscribeService, "run", _fake_run)


@pytest.fixture(autouse=True)
def _default_skip_real_transcription(request):
    """Apply ``skip_real_transcription`` by default unless the test opts out.

    Tests marked ``@pytest.mark.real_transcribe`` (e.g. the Basic Pitch
    smoke test) skip this layer so they exercise the real inference path.
    """
    if "real_transcribe" in request.keywords:
        yield
        return
    request.getfixturevalue("skip_real_transcription")
    yield


@pytest.fixture
def skip_real_separation(monkeypatch):
    """Replace ``SeparateService.run`` with a no-op pass-through.

    Phase 5 inserts the ``separate`` stage between ingest and transcribe
    in the default execution plan. Real Demucs inference is too heavy
    for the unit suite (and the fake audio bytes most tests pass would
    fail to decode anyway), so the default behavior is to return the
    bundle unchanged with ``audio_stems = {}`` — transcribe then falls
    back to its inline path (which the ``skip_real_transcription``
    fixture also stubs out).
    """
    def _fake_run(self, payload):
        return payload

    monkeypatch.setattr(separate_module.SeparateService, "run", _fake_run)


@pytest.fixture(autouse=True)
def _default_skip_real_separation(request):
    """Apply ``skip_real_separation`` by default unless the test opts out.

    Tests marked ``@pytest.mark.real_separate`` exercise the real
    SeparateService logic (cache lookups, blob persistence, error
    fall-through). They typically still monkeypatch the underlying
    ``separate_stems`` so they don't pull in Demucs proper.
    """
    if "real_separate" in request.keywords:
        yield
        return
    request.getfixturevalue("skip_real_separation")
    yield


@pytest.fixture(autouse=True)
def celery_eager_mode():
    """Run Celery tasks in-process for all tests."""
    _celery_app.conf.task_always_eager = True
    _celery_app.conf.task_eager_propagates = True
    yield
    _celery_app.conf.task_always_eager = False
    _celery_app.conf.task_eager_propagates = False


@pytest.fixture(autouse=True)
def stub_ml_engraver(monkeypatch):
    """Replace the ML engraver HTTP client with a deterministic fake.

    The engrave stage routes through this client whenever the local
    backend is disabled or falls through; without this stub every
    pipeline test that hits remote engrave would try to reach a real ML
    service. Tests that want to exercise error paths override this
    within their own monkeypatch scope.
    """
    async def fake_engrave(midi_bytes: bytes) -> bytes:
        return _FAKE_ML_MUSICXML

    monkeypatch.setattr(
        ml_engraver_module, "engrave_midi_via_ml_service", fake_engrave,
    )


@pytest.fixture
def stub_engrave_local(monkeypatch):
    """Replace ``engrave_score_locally`` with a deterministic fake.

    Phase 4 flips the default ``engrave_backend`` to ``"local"``, so
    every pipeline test now exercises ``engrave_score_locally``
    instead of the remote HTTP path. Without a stub each test would
    pay music21's ~3 s import + emit cost and depend on a working
    LilyPond binary — both of which are out of scope for unit tests.

    The stub returns a tiny but well-formed MusicXML payload plus a
    feature summary that mirrors what the real emitter would report
    for a minimal score: no chord symbols, no dynamics, no pedal. Tests
    that need to verify the runner reads non-zero counts off the
    ``LocalEngraveResult.features`` override this within their own
    monkeypatch scope.
    """
    def fake_engrave_locally(score, expression=None, *, title=None, composer=None,
                             render_pdf=True, render_svg=False):
        features = engrave_local_module.EngravedFeatures(
            title=title or (score.metadata.title if score.metadata else "") or "",
            composer=composer or (score.metadata.composer if score.metadata else "") or "",
            has_key_signature=False,
            has_tempo_marking=False,
            chord_symbol_count=0,
            dynamic_count=0,
            pedal_event_count=0,
            articulation_count=0,
            voice_count=1,
            note_count=len(score.right_hand) + len(score.left_hand),
        )
        return engrave_local_module.LocalEngraveResult(
            musicxml_bytes=_FAKE_LOCAL_MUSICXML,
            pdf_bytes=None,
            svg_bytes=None,
            features=features,
        )

    monkeypatch.setattr(
        engrave_local_module, "engrave_score_locally", fake_engrave_locally,
    )


@pytest.fixture(autouse=True)
def _default_stub_engrave_local(request):
    """Apply ``stub_engrave_local`` by default unless the test opts out.

    Tests marked ``@pytest.mark.real_engrave`` (``test_engrave_local.py``
    and any future engrave smoke tests) skip this layer so they exercise
    the real ``engrave_score_locally`` wrapper end-to-end.
    """
    if "real_engrave" in request.keywords:
        yield
        return
    request.getfixturevalue("stub_engrave_local")
    yield


@pytest.fixture(autouse=True)
def disable_real_refine_llm(monkeypatch):
    """Null out the Anthropic API key for every test.

    No test may hit the real API. The service's fallback path (no key →
    raise → caught → pass-through with warning) is exactly what bare
    pipeline tests need. Tests that want to exercise the merge logic
    construct ``RefineService(..., client=fake)`` directly — that path
    bypasses the key check entirely.
    """
    monkeypatch.setattr(settings, "anthropic_api_key", None)


@pytest.fixture
def client():
    """TestClient inside a `with` block so the lifespan and ASGI portal stay alive
    for the whole test. Without this, background asyncio tasks created during
    a request never get a chance to progress between sync calls."""
    app = create_app()
    with TestClient(app) as c:
        yield c
