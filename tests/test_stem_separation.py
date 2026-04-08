"""Unit tests for the Demucs stem-separation stage.

The real Demucs inference path is too expensive to exercise in the
unit suite (htdemucs weights are ~80 MB and inference on a 30-second
clip takes multiple seconds even on the fastest CPU), so the
``separate_stems`` entry point is driven by monkeypatching the
``demucs.apply.apply_model`` call — same pattern the cleanup /
preprocess tests use to keep CI fast and deterministic.

The bits we *do* exercise for real:
  * :class:`StemSeparationStats.as_warnings` formatting on every
    skipped / applied branch.
  * :class:`SeparatedStems.cleanup` idempotency + tempdir removal.
  * :func:`separate_stems` graceful degradation when the demucs
    import fails (simulated by hiding the module from sys.modules).
  * :func:`separate_stems` end-to-end via a monkeypatched model +
    apply_model stub that returns a fake (S, C, T) tensor — verifies
    the tempdir contract and the stem-routing.
  * Config defaults wire up to the stem_separation module defaults.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.services.stem_separation import (
    _MODEL_CACHE,
    DEFAULT_MODEL_NAME,
    DEFAULT_OVERLAP,
    DEFAULT_SHIFTS,
    SeparatedStems,
    StemSeparationStats,
    _pick_device,
    separate_stems,
)

# ---------------------------------------------------------------------------
# StemSeparationStats.as_warnings
# ---------------------------------------------------------------------------

def test_as_warnings_skipped_with_reason():
    stats = StemSeparationStats(
        skipped=True,
        warnings=["missing dep: demucs"],
    )
    msgs = stats.as_warnings()
    assert any("stem separation skipped" in m for m in msgs)
    assert any("demucs" in m for m in msgs)


def test_as_warnings_skipped_no_reason():
    stats = StemSeparationStats(skipped=True)
    assert stats.as_warnings() == ["stem separation skipped"]


def test_as_warnings_active_includes_model_device_stems_time():
    stats = StemSeparationStats(
        model_name="htdemucs",
        device="cpu",
        wall_time_sec=12.3,
        stems_written=["drums", "bass", "other", "vocals"],
    )
    msgs = stats.as_warnings()
    joined = " ".join(msgs)
    assert "htdemucs" in joined
    assert "cpu" in joined
    assert "drums" in joined and "vocals" in joined
    assert "12.3" in joined


def test_as_warnings_active_with_extra_warnings():
    stats = StemSeparationStats(
        model_name="htdemucs",
        device="mps",
        wall_time_sec=5.0,
        stems_written=["vocals", "bass"],
        warnings=["skipped 2 silent chunks"],
    )
    msgs = stats.as_warnings()
    assert any("silent chunks" in m for m in msgs)
    assert any("htdemucs" in m for m in msgs)


# ---------------------------------------------------------------------------
# SeparatedStems.cleanup
# ---------------------------------------------------------------------------

def test_cleanup_removes_tempdir_and_is_idempotent(tmp_path: Path):
    # Set up a fake tempdir populated with stem WAV placeholders.
    tempdir = tmp_path / "demucs-fake"
    tempdir.mkdir()
    for name in ("vocals", "bass", "drums", "other"):
        (tempdir / f"{name}.wav").write_bytes(b"RIFF....WAVEfake")

    stems = SeparatedStems(
        vocals=tempdir / "vocals.wav",
        bass=tempdir / "bass.wav",
        drums=tempdir / "drums.wav",
        other=tempdir / "other.wav",
        _tempdir=tempdir,
    )

    assert tempdir.exists()
    stems.cleanup()
    assert not tempdir.exists()
    # All slots are nilled so accidental reuse is obvious.
    assert stems.vocals is None
    assert stems.bass is None
    assert stems.drums is None
    assert stems.other is None
    # Idempotent — second call is a no-op.
    stems.cleanup()


def test_cleanup_handles_missing_tempdir(tmp_path: Path):
    ghost = tmp_path / "does-not-exist"
    stems = SeparatedStems(_tempdir=ghost)
    # Must not raise even though the tempdir was never created.
    stems.cleanup()
    assert stems._tempdir is None


def test_cleanup_noop_when_tempdir_never_set():
    stems = SeparatedStems()
    stems.cleanup()
    assert stems._tempdir is None


# ---------------------------------------------------------------------------
# _pick_device
# ---------------------------------------------------------------------------

def test_pick_device_honors_explicit_preference():
    # Explicit "cpu" short-circuits all probing — safe on any host.
    assert _pick_device("cpu") == "cpu"
    assert _pick_device("cuda") == "cuda"
    assert _pick_device("mps") == "mps"


def test_pick_device_falls_back_to_cpu_when_torch_missing(monkeypatch):
    # Hide torch so the import fails inside _pick_device.
    monkeypatch.setitem(sys.modules, "torch", None)
    assert _pick_device(None) == "cpu"


# ---------------------------------------------------------------------------
# separate_stems — graceful degradation
# ---------------------------------------------------------------------------

def test_separate_stems_missing_file(tmp_path: Path):
    ghost = tmp_path / "not-there.wav"
    stems, stats = separate_stems(ghost)
    assert stems is None
    assert stats.skipped
    assert any("missing" in w for w in stats.warnings)


def test_separate_stems_graceful_on_missing_demucs(tmp_path: Path, monkeypatch):
    """When demucs can't be imported we skip cleanly."""
    # Build a placeholder audio file — separate_stems checks is_file()
    # *before* touching deps, but the graceful-skip path runs during
    # the late import.
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF0000WAVEfake")

    # Hide one of the demucs modules so the late import raises.
    monkeypatch.setitem(sys.modules, "demucs.apply", None)

    stems, stats = separate_stems(audio)
    assert stems is None
    assert stats.skipped
    assert any("missing dep" in w for w in stats.warnings)


# ---------------------------------------------------------------------------
# separate_stems — happy path via monkeypatched demucs internals
# ---------------------------------------------------------------------------

class _FakeAudioFile:
    """Stand-in for ``demucs.audio.AudioFile`` — just yields a tensor."""
    def __init__(self, path):
        self._path = path

    def read(self, streams: int = 0, samplerate: int = 44_100, channels: int = 2):
        import torch
        # 2-channel, 1-second fake waveform at the requested samplerate.
        return torch.randn(channels, samplerate)


def _install_fake_demucs(monkeypatch, stem_names):
    """Install a fake demucs.{apply, audio, pretrained} trio.

    ``apply_model`` returns a tensor shaped ``(1, S, C, T)`` with one
    source per ``stem_names`` entry, so the caller sees the normal
    Demucs contract. ``save_audio`` writes a one-byte placeholder to
    each stem path so the cleanup path has something to remove.
    """
    torch = pytest.importorskip("torch")

    class FakeModel:
        samplerate = 44_100
        audio_channels = 2
        sources = list(stem_names)

        def eval(self):
            return self

        def to(self, *_args, **_kwargs):
            return self

    def fake_get_model(name: str):
        return FakeModel()

    def fake_apply_model(model, mix, **_kwargs):
        # mix shape: (B, C, T). Return (B, S, C, T) all zeros.
        b, c, t = mix.shape
        s = len(model.sources)
        return torch.zeros((b, s, c, t))

    def fake_save_audio(source, path, **_kwargs):
        Path(path).write_bytes(b"\x00")

    fake_apply = SimpleNamespace(apply_model=fake_apply_model)
    fake_audio = SimpleNamespace(AudioFile=_FakeAudioFile, save_audio=fake_save_audio)
    fake_pretrained = SimpleNamespace(get_model=fake_get_model)

    monkeypatch.setitem(sys.modules, "demucs.apply", fake_apply)
    monkeypatch.setitem(sys.modules, "demucs.audio", fake_audio)
    monkeypatch.setitem(sys.modules, "demucs.pretrained", fake_pretrained)
    # Blow the process-wide model cache so our fake get_model is
    # actually called — otherwise a real demucs run from a previous
    # test would still be cached.
    _MODEL_CACHE.clear()


def test_separate_stems_happy_path_four_stems(tmp_path: Path, monkeypatch):
    """All four stems are written and routed onto the SeparatedStems slots."""
    pytest.importorskip("torch")

    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 100)

    _install_fake_demucs(
        monkeypatch, ["drums", "bass", "other", "vocals"],
    )

    stems, stats = separate_stems(audio, device="cpu")

    try:
        assert stems is not None
        assert not stats.skipped
        assert stats.model_name == DEFAULT_MODEL_NAME
        assert stats.device == "cpu"
        assert set(stats.stems_written) == {"drums", "bass", "other", "vocals"}

        assert stems.vocals is not None and stems.vocals.exists()
        assert stems.bass is not None and stems.bass.exists()
        assert stems.drums is not None and stems.drums.exists()
        assert stems.other is not None and stems.other.exists()

        # All stems live in the same tempdir so cleanup can rmtree them.
        assert stems._tempdir is not None
        assert stems.vocals.parent == stems._tempdir
        assert stems.bass.parent == stems._tempdir
    finally:
        if stems is not None:
            stems.cleanup()


def test_separate_stems_two_stem_bag_leaves_missing_slots_none(
    tmp_path: Path, monkeypatch,
):
    """A custom 2-source bag still works — missing stems stay None."""
    pytest.importorskip("torch")

    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 100)

    # Only vocals + accompaniment (non-standard but valid — tests the
    # per-slot None fallback without needing a real 2-stem model).
    _install_fake_demucs(monkeypatch, ["vocals", "other"])

    stems, stats = separate_stems(audio, device="cpu")

    try:
        assert stems is not None
        assert not stats.skipped
        assert stems.vocals is not None
        assert stems.other is not None
        # Bass / drums aren't emitted by this fake bag.
        assert stems.bass is None
        assert stems.drums is None
    finally:
        if stems is not None:
            stems.cleanup()


def test_separate_stems_apply_failure_skips_cleanly(tmp_path: Path, monkeypatch):
    """apply_model raising must leave no tempdir behind and set skipped."""
    pytest.importorskip("torch")

    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 100)

    class FakeModel:
        samplerate = 44_100
        audio_channels = 2
        sources = ["drums", "bass", "other", "vocals"]

        def eval(self):
            return self

    def fake_get_model(name):
        return FakeModel()

    def fake_apply_model(model, mix, **_kwargs):
        raise RuntimeError("CUDA OOM")

    def fake_save_audio(*_args, **_kwargs):
        raise AssertionError("save_audio should not be reached")

    fake_apply = SimpleNamespace(apply_model=fake_apply_model)
    fake_audio = SimpleNamespace(AudioFile=_FakeAudioFile, save_audio=fake_save_audio)
    fake_pretrained = SimpleNamespace(get_model=fake_get_model)

    monkeypatch.setitem(sys.modules, "demucs.apply", fake_apply)
    monkeypatch.setitem(sys.modules, "demucs.audio", fake_audio)
    monkeypatch.setitem(sys.modules, "demucs.pretrained", fake_pretrained)
    _MODEL_CACHE.clear()

    stems, stats = separate_stems(audio, device="cpu")
    assert stems is None
    assert stats.skipped
    assert any("apply failed" in w for w in stats.warnings)


# ---------------------------------------------------------------------------
# Config defaults sanity check
# ---------------------------------------------------------------------------

def test_config_defaults_match_module_defaults():
    from backend.config import Settings

    s = Settings()
    # On by default — the stems path is the preferred pipeline, with
    # a transparent fallback to the single-mix path when demucs/torch
    # aren't installed. Commercial deployments that can't use the
    # CC BY-NC htdemucs weights must opt out via
    # OHSHEET_DEMUCS_ENABLED=0.
    assert s.demucs_enabled is True
    assert s.demucs_model == DEFAULT_MODEL_NAME
    assert s.demucs_shifts == DEFAULT_SHIFTS
    assert s.demucs_overlap == DEFAULT_OVERLAP
    # Per-consumer flags default on so flipping demucs_enabled alone
    # wires up all four stems without further config work.
    assert s.demucs_use_vocals_for_melody is True
    assert s.demucs_use_bass_stem is True
    assert s.demucs_use_other_for_chords is True
    assert s.demucs_use_drums_for_beats is True
