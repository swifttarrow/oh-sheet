"""Tests for the Phase-5 source-separation pipeline stage.

The separate worker is a thin Celery wrapper around
:class:`backend.services.separate.SeparateService`. These tests
monkeypatch the heavy ``separate_stems`` call so we don't pull in
Demucs / Torch from the unit suite, and exercise:

  * Happy path — bundle with audio in, blob URIs returned, stems
    persisted under the content-addressed cache key.
  * Cache hit — running the service twice for the same audio returns
    the cached URIs without re-invoking ``separate_stems``.
  * Graceful fallback — ``separate_stems`` returning ``(None, stats)``
    leaves the bundle unchanged with ``audio_stems = {}``.
  * Bundle without audio is a no-op pass-through.
  * Pipeline integration — the runner inserts the separate step ahead
    of transcribe when ``separator != "off"``.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from shared.storage.local import LocalBlobStore

from backend.contracts import (
    InputBundle,
    InputMetadata,
    PipelineConfig,
    RemoteAudioFile,
)
from backend.services import separate as separate_module
from backend.services.separate import SeparateService
from backend.services.stem_separation import (
    SeparatedStems,
    StemSeparationStats,
)

# Every test in this module exercises the real SeparateService (with
# Demucs itself monkeypatched out) — opt out of the autouse stub.
pytestmark = pytest.mark.real_separate


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_blob_store(tmp_path: Path) -> LocalBlobStore:
    root = tmp_path / "blob"
    root.mkdir(exist_ok=True)
    return LocalBlobStore(root)


def _make_bundle_with_audio(
    blob: LocalBlobStore, audio_bytes: bytes,
) -> InputBundle:
    audio_uri = blob.put_bytes("uploads/audio.wav", audio_bytes)
    return InputBundle(
        audio=RemoteAudioFile(
            uri=audio_uri,
            format="wav",
            sample_rate=44100,
            duration_sec=10.0,
            channels=2,
        ),
        midi=None,
        metadata=InputMetadata(
            source="audio_upload",
            title="t",
            artist="a",
        ),
    )


def _write_fake_stems(tmp_path: Path) -> SeparatedStems:
    """Build a SeparatedStems pointing at four placeholder WAVs."""
    stem_dir = tmp_path / "demucs_out"
    stem_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in ("vocals", "drums", "bass", "other"):
        p = stem_dir / f"{name}.wav"
        p.write_bytes(f"fake-{name}".encode())
        paths[name] = p
    return SeparatedStems(
        vocals=paths["vocals"],
        drums=paths["drums"],
        bass=paths["bass"],
        other=paths["other"],
        _tempdir=stem_dir,
    )


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------

def test_separate_service_writes_stems_and_returns_uris(monkeypatch, tmp_path):
    """Stems get persisted under the content-addressed cache key."""
    blob = _make_blob_store(tmp_path)
    bundle = _make_bundle_with_audio(blob, b"audio-bytes-original")

    fake_stems = _write_fake_stems(tmp_path)
    fake_stats = StemSeparationStats(
        model_name="htdemucs",
        device="cpu",
        wall_time_sec=1.5,
        stems_written=["vocals", "drums", "bass", "other"],
    )

    call_count = {"n": 0}

    def fake_separate_stems(audio_path, **_kwargs):
        call_count["n"] += 1
        return fake_stems, fake_stats

    monkeypatch.setattr(separate_module, "separate_stems", fake_separate_stems)

    svc = SeparateService(blob_store=blob)
    result = svc.run(bundle)

    assert call_count["n"] == 1
    # Four URIs, one per htdemucs source, all reachable from the blob store.
    assert set(result.audio_stems) == {"vocals", "drums", "bass", "other"}
    for stem_name, uri in result.audio_stems.items():
        assert blob.exists(uri)
        assert blob.get_bytes(uri) == f"fake-{stem_name}".encode()

    # Cache prefix is content-addressed: bytes through sha256, model name folded in.
    import hashlib
    expected_digest = hashlib.sha256(b"audio-bytes-original").hexdigest()
    expected_prefix = f"cache/separate/{expected_digest}/htdemucs"
    for uri in result.audio_stems.values():
        assert expected_prefix in uri


# ---------------------------------------------------------------------------
# cache hit
# ---------------------------------------------------------------------------

def test_separate_service_cache_hit_skips_inference(monkeypatch, tmp_path):
    """Re-running on the same audio bytes finds the cache and skips Demucs.

    The expensive ``separate_stems`` call is monkeypatched to bump a
    counter; the second run must NOT increment it. URIs returned by
    both runs match exactly so downstream stages are deterministic.
    """
    blob = _make_blob_store(tmp_path)
    bundle = _make_bundle_with_audio(blob, b"same-bytes-twice")

    fake_stems_factory = lambda: _write_fake_stems(  # noqa: E731 — single-use lambda
        tmp_path / f"demucs_call_{call_count['n']}",
    )

    call_count = {"n": 0}

    def fake_separate_stems(audio_path, **_kwargs):
        call_count["n"] += 1
        # Each call gets a fresh tempdir so the cleanup() path is safe.
        return fake_stems_factory(), StemSeparationStats(
            model_name="htdemucs",
            stems_written=["vocals", "drums", "bass", "other"],
        )

    monkeypatch.setattr(separate_module, "separate_stems", fake_separate_stems)

    svc = SeparateService(blob_store=blob)
    first = svc.run(bundle)
    assert call_count["n"] == 1

    # Re-build the bundle (same audio URI, same bytes) and run again.
    second = svc.run(bundle)
    assert call_count["n"] == 1, "cache hit should skip a second Demucs call"
    assert first.audio_stems == second.audio_stems


# ---------------------------------------------------------------------------
# graceful fallbacks
# ---------------------------------------------------------------------------

def test_separate_service_returns_empty_stems_when_demucs_skips(
    monkeypatch, tmp_path,
):
    """``separate_stems`` returning ``(None, stats)`` leaves bundle unchanged."""
    blob = _make_blob_store(tmp_path)
    bundle = _make_bundle_with_audio(blob, b"audio")

    fake_stats = StemSeparationStats(
        skipped=True,
        warnings=["missing dep: demucs"],
    )
    monkeypatch.setattr(
        separate_module,
        "separate_stems",
        lambda *_a, **_kw: (None, fake_stats),
    )

    svc = SeparateService(blob_store=blob)
    result = svc.run(bundle)

    assert result.audio_stems == {}
    # Bundle is otherwise unchanged.
    assert result.audio is not None
    assert result.audio.uri == bundle.audio.uri


def test_separate_service_no_audio_is_passthrough(tmp_path):
    """Bundles without audio are returned unchanged."""
    blob = _make_blob_store(tmp_path)
    bundle = InputBundle(
        audio=None,
        midi=None,
        metadata=InputMetadata(source="midi_upload", title="t", artist="a"),
    )

    svc = SeparateService(blob_store=blob)
    result = svc.run(bundle)
    assert result.audio_stems == {}
    assert result.audio is None


def test_separate_service_no_blob_store_is_passthrough(tmp_path):
    """Without a blob store wired in, the service can't persist; pass through."""
    blob = _make_blob_store(tmp_path)
    bundle = _make_bundle_with_audio(blob, b"audio")

    svc = SeparateService(blob_store=None)
    result = svc.run(bundle)
    assert result.audio_stems == {}


# ---------------------------------------------------------------------------
# pipeline routing
# ---------------------------------------------------------------------------

def test_pipeline_config_inserts_separate_before_transcribe():
    """Default ``separator='htdemucs'`` puts ``separate`` ahead of transcribe."""
    config = PipelineConfig(variant="audio_upload")
    plan = config.get_execution_plan()
    assert "separate" in plan
    assert plan.index("separate") < plan.index("transcribe")


def test_pipeline_config_separator_off_omits_stage():
    """``separator='off'`` skips the dedicated stage entirely."""
    config = PipelineConfig(variant="audio_upload", separator="off")
    plan = config.get_execution_plan()
    assert "separate" not in plan
    assert "transcribe" in plan


def test_pipeline_config_midi_upload_never_runs_separate():
    """Variants without transcribe (midi_upload) get no separate stage."""
    config = PipelineConfig(variant="midi_upload", separator="htdemucs")
    plan = config.get_execution_plan()
    assert "separate" not in plan


# ---------------------------------------------------------------------------
# Celery task wiring
# ---------------------------------------------------------------------------

def test_separate_celery_task_round_trips_bundle(monkeypatch, tmp_path):
    """The Celery wrapper round-trips a bundle through blob storage."""
    from backend.config import settings
    monkeypatch.setattr(settings, "blob_root", tmp_path / "blob")
    blob = LocalBlobStore(settings.blob_root)
    bundle = _make_bundle_with_audio(blob, b"audio-celery")

    fake_stems = _write_fake_stems(tmp_path)
    monkeypatch.setattr(
        separate_module, "separate_stems",
        lambda *_a, **_kw: (
            fake_stems,
            StemSeparationStats(
                model_name="htdemucs",
                stems_written=["vocals", "drums", "bass", "other"],
            ),
        ),
    )

    payload_uri = blob.put_json(
        "jobs/job-1/separate/input.json",
        bundle.model_dump(mode="json"),
    )

    from backend.workers import separate as separate_worker
    output_uri = separate_worker.run("job-1", payload_uri)
    raw = blob.get_json(output_uri)
    out = InputBundle.model_validate(raw)
    assert set(out.audio_stems) == {"vocals", "drums", "bass", "other"}


# ---------------------------------------------------------------------------
# transcribe consumption of pre-separated stems
# ---------------------------------------------------------------------------

def test_transcribe_stages_pre_separated_stems_to_tempdir(monkeypatch, tmp_path):
    """``_stage_pre_separated_stems`` materializes blob URIs into a tempdir."""
    blob = _make_blob_store(tmp_path)
    stem_uris = {}
    for name in ("vocals", "drums", "bass", "other"):
        stem_uris[name] = blob.put_bytes(
            f"cache/separate/abc/htdemucs/{name}.wav",
            f"stem-{name}".encode(),
        )

    from backend.services import transcribe as transcribe_module
    stems = transcribe_module._stage_pre_separated_stems(stem_uris, blob)
    assert stems is not None
    try:
        assert stems.vocals is not None and stems.vocals.read_bytes() == b"stem-vocals"
        assert stems.drums is not None and stems.drums.read_bytes() == b"stem-drums"
        assert stems.bass is not None and stems.bass.read_bytes() == b"stem-bass"
        assert stems.other is not None and stems.other.read_bytes() == b"stem-other"
    finally:
        stems.cleanup()


def test_transcribe_pre_separated_path_skips_inline_demucs(monkeypatch, tmp_path):
    """When ``audio_stems`` is populated, ``_run_basic_pitch_sync`` routes to stems.

    The inline ``separate_stems`` call must NOT fire — we verify by
    monkeypatching it to bump a counter that should stay at zero.
    """
    from backend.contracts import (
        SCHEMA_VERSION,
        HarmonicAnalysis,
        QualitySignal,
        TempoMapEntry,
        TranscriptionResult,
    )
    from backend.services import transcribe as transcribe_module

    inline_calls = {"n": 0}

    def fake_inline_separate(*_a, **_kw):
        inline_calls["n"] += 1
        return None, StemSeparationStats(skipped=True)

    monkeypatch.setattr(transcribe_module, "separate_stems", fake_inline_separate)

    # Stub the stems pipeline so we can verify routing without exercising
    # the full per-stem orchestrator.
    captured = {"called": False}

    def fake_run_with_stems(audio_path, stems, stem_stats):
        captured["called"] = True
        captured["stem_stats"] = stem_stats
        return (
            TranscriptionResult(
                schema_version=SCHEMA_VERSION,
                midi_tracks=[],
                analysis=HarmonicAnalysis(
                    key="C:major",
                    time_signature=(4, 4),
                    tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
                    chords=[],
                    sections=[],
                ),
                quality=QualitySignal(overall_confidence=0.5, warnings=[]),
            ),
            None,
        )

    from backend.services import transcribe_pipeline_stems as stems_mod
    monkeypatch.setattr(stems_mod, "_run_with_stems", fake_run_with_stems)

    pre = _write_fake_stems(tmp_path)
    audio = tmp_path / "mix.wav"
    audio.write_bytes(b"\x00")
    try:
        transcribe_module._run_basic_pitch_sync(audio, pre_separated=pre)
    finally:
        pre.cleanup()

    assert inline_calls["n"] == 0, "pre_separated path must not call inline separate"
    assert captured["called"] is True
    # The synthetic stats stamp ``device='pre_separated'`` so log inspection
    # can tell the two routes apart.
    assert captured["stem_stats"].device == "pre_separated"
    assert set(captured["stem_stats"].stems_written) == {
        "vocals", "drums", "bass", "other",
    }
