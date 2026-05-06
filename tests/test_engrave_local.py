"""Tests for the Phase 4 local engrave stack.

Covers three layers, each independently:
  * ``score_to_musicxml`` — music21 emitter; verifies feature counts,
    well-formed XML, key signature / dynamics / pedal / chord-symbol /
    articulation / voice plumbing.
  * ``musicxml_to_pdf`` / ``musicxml_to_svg`` — subprocess + verovio
    renderers; gated on the presence of the binary / Python module so
    dev machines without LilyPond or verovio still pass the suite.
  * ``PipelineRunner`` integration — verifies the runner dispatches to
    ``engrave_score_locally`` by default and falls through to the
    remote HTTP service when the local stack raises.

Tests marked ``@pytest.mark.real_engrave`` opt out of the conftest's
``stub_engrave_local`` autouse layer so they exercise music21 directly.
Pipeline-runner integration tests stay on the stub but pin the engrave
backend explicitly and override the stub in-place when they need to
verify a specific failure / fallback shape.
"""
from __future__ import annotations

import shutil

import pytest
from shared.contracts import (
    InputBundle,
    InputMetadata,
    PipelineConfig,
    RemoteAudioFile,
)
from shared.storage.local import LocalBlobStore

from backend.config import settings
from backend.jobs.runner import PipelineRunner
from backend.services import engrave_local as engrave_local_module
from backend.services import ml_engraver_client
from backend.services.engrave_local import (
    EngravedFeatures,
    EngraveLocalError,
    LocalEngraveResult,
    musicxml_to_pdf,
    musicxml_to_svg,
    score_to_musicxml,
)
from backend.workers.celery_app import celery_app
from tests.fixtures import load_score_fixture

# ─────────────────────────────────────────────────────────────────────────
# score_to_musicxml — music21 emitter (uses real music21)
# ─────────────────────────────────────────────────────────────────────────
# The conftest's ``stub_engrave_local`` autouse fixture only swaps the
# ``engrave_score_locally`` wrapper — ``score_to_musicxml``,
# ``musicxml_to_pdf``, and ``musicxml_to_svg`` are untouched, so direct
# calls below run against the real implementations. Tests that exercise
# the wrapper itself opt out of the stub via ``@pytest.mark.real_engrave``.


def test_emits_well_formed_partwise_score_for_minimal_piano():
    """A trivial single-note score round-trips through music21 cleanly."""
    score = load_score_fixture("single_note")

    xml_bytes, features = score_to_musicxml(score, title="Test Piece", composer="Test Composer")

    assert xml_bytes.startswith(b"<?xml")
    assert b"<score-partwise" in xml_bytes
    # The ~200-byte sanity floor is the silent-failure trip-wire — make
    # sure even the smallest valid input produces well past it.
    assert len(xml_bytes) > 500
    assert features.title == "Test Piece"
    assert features.composer == "Test Composer"
    assert features.note_count == 1


def test_emits_key_signature_when_metadata_key_parses():
    """``metadata.key`` propagates into the MusicXML key signature element."""
    score = load_score_fixture("mislabeled_key")  # metadata.key == "C:major"

    _xml, features = score_to_musicxml(score)

    assert features.has_key_signature is True


def test_skips_key_signature_silently_on_unparseable_key():
    """A garbage key string drops the key signature rather than emit a wrong one."""
    score = load_score_fixture("single_note").model_copy(deep=True)
    score = score.model_copy(update={
        "metadata": score.metadata.model_copy(update={"key": "garbage:weird"}),
    })

    _xml, features = score_to_musicxml(score)

    assert features.has_key_signature is False


def test_emits_chord_symbols_from_metadata():
    """Each ``ScoreChordEvent`` parseable as a ChordSymbol increments the counter."""
    score = load_score_fixture("chord_symbols")

    xml, features = score_to_musicxml(score)

    # Three of the seven chord_symbols fixture entries are parseable
    # (C:maj7, Dm7, F:maj7); the rest fail Harte parsing or music21's
    # ChordSymbol constructor and are skipped silently. The exact count
    # is implementation-detail of ``_harte_to_figure`` + music21's
    # parser, but at minimum the well-formed colon-form labels must
    # all survive — so we check ≥3 instead of pinning to 3.
    assert features.chord_symbol_count >= 3
    # Chord symbols travel as <harmony> elements in MusicXML.
    assert b"<harmony" in xml


def test_emits_tempo_marking_for_simple_tempo_map():
    score = load_score_fixture("c_major_scale")

    _xml, features = score_to_musicxml(score)

    assert features.has_tempo_marking is True


def test_emits_dynamics_pedal_and_articulations_from_humanized_performance():
    """The humanized fixture exercises every expression-map branch."""
    perf = load_score_fixture("humanized_with_expression")

    xml, features = score_to_musicxml(perf.score, perf.expression)

    # Two dynamics (p + crescendo), three pedal events (sustain +
    # sostenuto + una_corda), one articulation (fermata).
    assert features.dynamic_count >= 2
    assert features.pedal_event_count >= 1  # PedalMark may collapse overlaps
    assert features.articulation_count >= 1
    # Visible dynamic markings render as <direction><direction-type><dynamics> in MusicXML.
    assert b"<dynamics" in xml or b"<words" in xml


def test_voice_count_includes_distinct_voices_across_both_staves():
    """A two-voice RH fixture should report voice_count ≥ 2."""
    score = load_score_fixture("bach_invention_excerpt")

    _xml, features = score_to_musicxml(score)

    assert features.voice_count >= 2


def test_handles_empty_left_hand_without_dropping_the_staff():
    """RH-only fixtures must still produce a two-staff piano score —
    a missing LH staff would silently degrade the engraved output."""
    score = load_score_fixture("empty_left_hand")

    xml, features = score_to_musicxml(score)

    assert features.note_count == 4  # the four RH notes
    # Both parts (rh + lh) appear in the partwise score.
    assert xml.count(b"<part ") >= 2


def test_size_floor_guard_rejects_blank_payloads(monkeypatch):
    """If music21 ever emits a near-empty payload, the engrave must
    fail loudly rather than silently surface a blank score."""
    from music21.musicxml.m21ToXml import GeneralObjectExporter

    def _tiny_export(self):
        return b"<x/>"

    monkeypatch.setattr(GeneralObjectExporter, "parse", _tiny_export)

    score = load_score_fixture("single_note")
    with pytest.raises(EngraveLocalError, match="suspiciously small"):
        score_to_musicxml(score)


def test_well_formed_check_rejects_malformed_root(monkeypatch):
    """Anything other than score-partwise/score-timewise is a hard fail."""
    from music21.musicxml.m21ToXml import GeneralObjectExporter

    def _bad_root(self):
        # Padded above the 200-byte floor so it's the root check that fires,
        # not the size guard.
        return b"<?xml version='1.0'?><not-a-score>" + (b"x" * 300) + b"</not-a-score>"

    monkeypatch.setattr(GeneralObjectExporter, "parse", _bad_root)

    score = load_score_fixture("single_note")
    with pytest.raises(EngraveLocalError, match="unexpected XML root"):
        score_to_musicxml(score)


# ─────────────────────────────────────────────────────────────────────────
# musicxml_to_pdf — LilyPond subprocess
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not (shutil.which("lilypond") and shutil.which("musicxml2ly")),
    reason="LilyPond toolchain not installed locally",
)
def test_musicxml_to_pdf_round_trips_a_real_score():
    """End-to-end: PianoScore → MusicXML → PDF via real LilyPond."""
    score = load_score_fixture("c_major_scale")
    xml, _ = score_to_musicxml(score, title="LilyPond Smoke", composer="Tester")

    pdf_bytes = musicxml_to_pdf(xml)

    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000  # real PDFs have headers, fonts, etc.


def test_musicxml_to_pdf_raises_when_lilypond_missing(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    with pytest.raises(EngraveLocalError, match="lilypond is not installed"):
        musicxml_to_pdf(b"<score-partwise/>")


# ─────────────────────────────────────────────────────────────────────────
# musicxml_to_svg — Verovio
# ─────────────────────────────────────────────────────────────────────────


def test_musicxml_to_svg_round_trips_a_real_score():
    """End-to-end SVG render via verovio (skipped if not installed)."""
    pytest.importorskip("verovio")

    score = load_score_fixture("c_major_scale")
    xml, _ = score_to_musicxml(score, title="Verovio Smoke")

    svg = musicxml_to_svg(xml)

    assert svg.lstrip().startswith((b"<?xml", b"<svg"))
    assert b"<svg" in svg


def test_musicxml_to_svg_raises_when_verovio_missing(monkeypatch):
    """Importing verovio under the engrave_local namespace fails clean."""
    import builtins

    real_import = builtins.__import__

    def _no_verovio(name, *args, **kwargs):
        if name == "verovio":
            raise ImportError("simulated missing verovio")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_verovio)

    with pytest.raises(EngraveLocalError, match="verovio is not installed"):
        musicxml_to_svg(b"<score-partwise/>")


# ─────────────────────────────────────────────────────────────────────────
# engrave_score_locally — convenience wrapper
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.real_engrave
def test_engrave_score_locally_pdf_failure_degrades_gracefully(monkeypatch):
    """A broken PDF renderer must not lose the MusicXML payload — the
    runner relies on this to keep producing artifacts when LilyPond
    is missing on the host."""
    def _broken_pdf(_xml):
        raise EngraveLocalError("pretend lilypond just exploded")

    monkeypatch.setattr(engrave_local_module, "musicxml_to_pdf", _broken_pdf)

    score = load_score_fixture("single_note")
    result = engrave_local_module.engrave_score_locally(
        score, render_pdf=True, render_svg=False, title="Graceful", composer="Tester",
    )

    assert result.musicxml_bytes  # MusicXML survived
    assert result.pdf_bytes is None  # PDF render failure was swallowed
    assert result.features.title == "Graceful"


@pytest.mark.real_engrave
def test_engrave_score_locally_returns_complete_bundle():
    """The convenience wrapper round-trips a real PianoScore into a
    populated ``LocalEngraveResult`` (PDF best-effort, MusicXML required)."""
    score = load_score_fixture("c_major_scale")

    result = engrave_local_module.engrave_score_locally(
        score, render_pdf=False, render_svg=False, title="Bundle", composer="Tester",
    )

    assert result.musicxml_bytes.startswith(b"<?xml")
    assert result.features.note_count >= 8  # eight RH + two LH notes
    assert result.features.title == "Bundle"
    assert result.features.composer == "Tester"


# ─────────────────────────────────────────────────────────────────────────
# PipelineRunner integration
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def runner():
    return PipelineRunner(
        blob_store=LocalBlobStore(settings.blob_root),
        celery_app=celery_app,
    )


@pytest.mark.asyncio
async def test_runner_dispatches_to_local_engrave_by_default(runner, monkeypatch):
    """With the default ``engrave_backend = "local"``, the runner stamps
    the local stub's MusicXML onto the result and the remote HTTP path
    is never invoked."""
    monkeypatch.setattr(settings, "engrave_backend", "local")

    remote_calls = 0

    async def _spy_remote(_midi):
        nonlocal remote_calls
        remote_calls += 1
        return b"<should-not-be-used/>"

    monkeypatch.setattr(
        ml_engraver_client, "engrave_midi_via_ml_service", _spy_remote,
    )

    bundle = InputBundle(
        audio=RemoteAudioFile(
            uri="file:///fake/audio.wav",
            format="wav",
            sample_rate=44100,
            duration_sec=10.0,
            channels=1,
        ),
        metadata=InputMetadata(title="Local Default", artist="Tester", source="audio_upload"),
    )
    config = PipelineConfig(variant="audio_upload", enable_refine=False)

    result = await runner.run(
        job_id="local-default-001",
        bundle=bundle,
        config=config,
    )

    assert result.musicxml_uri  # local stub MusicXML was persisted
    assert remote_calls == 0    # remote service was not consulted


@pytest.mark.asyncio
async def test_runner_falls_back_to_remote_when_local_engrave_raises(
    runner, monkeypatch,
):
    """An ``EngraveLocalError`` from the local stack must trigger the
    remote HTTP fallback so the job still produces a MusicXML artifact."""
    monkeypatch.setattr(settings, "engrave_backend", "local")

    def _raise_local(*_args, **_kwargs):
        raise EngraveLocalError("simulated music21 failure")

    monkeypatch.setattr(
        engrave_local_module, "engrave_score_locally", _raise_local,
    )

    remote_calls = 0
    fallback_xml = (
        b'<?xml version="1.0" encoding="UTF-8"?>'
        b'<score-partwise version="3.1"><part-list>'
        b'<score-part id="P1"><part-name>Piano</part-name></score-part>'
        b'</part-list><part id="P1"><measure number="1"/></part></score-partwise>'
    ) + (b" " * 600)  # padded above the stub-ceiling

    async def _spy_remote(_midi):
        nonlocal remote_calls
        remote_calls += 1
        return fallback_xml

    monkeypatch.setattr(
        ml_engraver_client, "engrave_midi_via_ml_service", _spy_remote,
    )

    bundle = InputBundle(
        audio=RemoteAudioFile(
            uri="file:///fake/audio.wav",
            format="wav",
            sample_rate=44100,
            duration_sec=10.0,
            channels=1,
        ),
        metadata=InputMetadata(title="Fallback Test", artist="Tester", source="audio_upload"),
    )
    config = PipelineConfig(variant="audio_upload", enable_refine=False)

    result = await runner.run(
        job_id="local-fallback-001",
        bundle=bundle,
        config=config,
    )

    assert result.musicxml_uri
    assert remote_calls == 1  # remote path picked up the slack


@pytest.mark.asyncio
async def test_runner_persists_pdf_when_local_engrave_returns_pdf_bytes(
    runner, monkeypatch,
):
    """When the local stack returns PDF bytes, ``pdf_uri`` must be set
    on the result. Verifies the runner's PDF persistence + URI propagation."""
    monkeypatch.setattr(settings, "engrave_backend", "local")

    fake_pdf = b"%PDF-1.4\n" + (b"x" * 1024)
    fake_xml = (
        b'<?xml version="1.0"?>'
        b'<score-partwise version="4.0"><part-list>'
        b'<score-part id="P1"><part-name>Piano</part-name></score-part>'
        b'</part-list><part id="P1"><measure number="1"/></part></score-partwise>'
    ) + (b" " * 600)

    def _local_with_pdf(score, expression=None, *, title=None, composer=None,
                       render_pdf=True, render_svg=False):
        return LocalEngraveResult(
            musicxml_bytes=fake_xml,
            pdf_bytes=fake_pdf,
            svg_bytes=None,
            features=EngravedFeatures(
                title=title or "",
                composer=composer or "",
                note_count=1,
            ),
        )

    monkeypatch.setattr(
        engrave_local_module, "engrave_score_locally", _local_with_pdf,
    )

    bundle = InputBundle(
        audio=RemoteAudioFile(
            uri="file:///fake/audio.wav",
            format="wav",
            sample_rate=44100,
            duration_sec=10.0,
            channels=1,
        ),
        metadata=InputMetadata(title="PDF Test", artist="Tester", source="audio_upload"),
    )
    config = PipelineConfig(variant="audio_upload", enable_refine=False)

    result = await runner.run(
        job_id="local-pdf-001",
        bundle=bundle,
        config=config,
    )

    assert result.pdf_uri  # pdf_uri populated from the bytes


@pytest.mark.asyncio
async def test_runner_derives_includes_flags_from_local_features(runner, monkeypatch):
    """``EngravedScoreData.includes_*`` should reflect the LOCAL feature
    counts (not the MIDI-render features) when local engrave succeeds."""
    monkeypatch.setattr(settings, "engrave_backend", "local")

    fake_xml = b"<score-partwise/>" + (b" " * 600)

    def _local_with_features(score, expression=None, *, title=None, composer=None,
                             render_pdf=True, render_svg=False):
        return LocalEngraveResult(
            musicxml_bytes=fake_xml,
            pdf_bytes=None,
            svg_bytes=None,
            features=EngravedFeatures(
                title=title or "",
                composer=composer or "",
                chord_symbol_count=4,
                dynamic_count=2,
                pedal_event_count=1,
                note_count=10,
            ),
        )

    monkeypatch.setattr(
        engrave_local_module, "engrave_score_locally", _local_with_features,
    )

    bundle = InputBundle(
        audio=RemoteAudioFile(
            uri="file:///fake/audio.wav",
            format="wav",
            sample_rate=44100,
            duration_sec=10.0,
            channels=1,
        ),
        metadata=InputMetadata(title="Flags Test", artist="Tester", source="audio_upload"),
    )
    config = PipelineConfig(variant="audio_upload", enable_refine=False)

    result = await runner.run(
        job_id="local-flags-001",
        bundle=bundle,
        config=config,
    )

    assert result.metadata.includes_chord_symbols is True
    assert result.metadata.includes_dynamics is True
    assert result.metadata.includes_pedal_marks is True
    assert result.metadata.includes_fingering is False  # no fingering generator yet
