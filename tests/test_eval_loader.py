"""Tests for ``eval/loader.py``.

Phase 3 acceptance per
``docs/research/transcription-improvement-implementation-plan.md`` §Phase 3:

* ``eval/loader.py`` is pytest-tested on a fixture song.
* Slots that haven't been delivered yet are visible via ``iter_songs``
  but excluded from ``iter_delivered``.
* ``structural.yaml`` loads cleanly into a ``HarmonicAnalysis``-compatible
  dict (the contract the consumer expects).

Tests assemble synthetic eval sets under ``tmp_path`` so they don't
depend on the real ``eval/pop_eval_v1/`` slots being populated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval import loader  # noqa: E402
from eval.loader import (  # noqa: E402
    AudioMetadata,
    EvalManifestError,
    EvalSong,
    StructuralReference,
    iter_delivered,
    iter_songs,
    load_manifest,
    load_song_by_slug,
)

# ---------------------------------------------------------------------------
# Fixture builder — assembles a minimal eval set on disk
# ---------------------------------------------------------------------------

def _write_manifest(eval_set_dir: Path, songs: list[dict]) -> None:
    manifest = {
        "schema_version": 1,
        "eval_set": "test_eval_set",
        "eval_set_version": "0.0.1",
        "target_duration_sec": 0,
        "chord_recognition_key": "auto",
        "songs": songs,
    }
    eval_set_dir.mkdir(parents=True, exist_ok=True)
    (eval_set_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))


def _populate_song_artifacts(
    eval_set_dir: Path,
    slug: str,
    *,
    license_bucket: str = "fma_redistributable",
    write_audio: bool = True,
    write_audio_meta: bool = True,
    write_midi: bool = True,
    write_xml: bool = True,
    write_structural: bool = True,
    structural_overrides: dict | None = None,
) -> Path:
    songs_dir = eval_set_dir / "songs" / slug
    songs_dir.mkdir(parents=True, exist_ok=True)

    if write_audio:
        # 12 bytes — content doesn't matter for the loader, only presence.
        (songs_dir / "source.audio.mp3").write_bytes(b"\x00" * 12)
    if write_audio_meta:
        (songs_dir / loader.AUDIO_METADATA_FILENAME).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "content_hash": f"sha256:{'a' * 64}",
                    "sample_rate": 44100,
                    "duration_sec": 180.0,
                    "format": "mp3",
                    "license": "cc-by-3.0",
                    "license_bucket": license_bucket,
                    "source_url": "https://example.org/song.mp3",
                    "internal_storage_uri": (
                        "s3://oh-sheet-eval-private/foo.mp3"
                        if license_bucket == "commercial_sync_internal"
                        else None
                    ),
                }
            )
        )
    if write_midi:
        (songs_dir / loader.REFERENCE_MIDI_FILENAME).write_bytes(
            b"MThd\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00"
        )
    if write_xml:
        (songs_dir / loader.REFERENCE_MUSICXML_FILENAME).write_text(
            '<?xml version="1.0"?><score-partwise version="3.1"><part id="P1"/></score-partwise>'
        )
    if write_structural:
        structural = {
            "key": "C:major",
            "time_signature": "4/4",
            "tempo_bpm": 120.0,
            "tempo_map": [{"time_sec": 0.0, "beat": 0.0, "bpm": 120.0}],
            "sections": [
                {"name": "intro", "start_sec": 0.0, "end_sec": 8.0, "bars": "1-4"},
            ],
            "chord_progression": [
                {"start_sec": 0.0, "label": "C:maj"},
                {"start_sec": 4.0, "label": "G:maj"},
            ],
            "downbeat_sec": [0.0, 2.0, 4.0, 6.0],
            "license": "cc-by-3.0",
        }
        if structural_overrides:
            structural.update(structural_overrides)
        (songs_dir / loader.STRUCTURAL_FILENAME).write_text(
            yaml.safe_dump(structural, sort_keys=False)
        )
    return songs_dir


@pytest.fixture
def small_eval_set(tmp_path):
    """3 slots: one fully delivered, one undelivered, one sync-only (no audio file)."""
    eval_set = tmp_path / "test_eval_set"
    _write_manifest(
        eval_set,
        [
            {
                "slug": "song_delivered",
                "title": "Delivered Pop Song",
                "artist": "Tester",
                "genre": "pop_mainstream",
                "license_bucket": "fma_redistributable",
                "intended_source": {
                    "kind": "fma_track",
                    "url": "https://freemusicarchive.org/x",
                    "license": "cc-by",
                    "notes": "test",
                },
            },
            {
                "slug": "song_undelivered",
                "title": "TBD",
                "artist": "TBD",
                "genre": "ballad",
                "license_bucket": "fma_redistributable",
                "intended_source": {
                    "kind": "fma_track",
                    "url": "https://freemusicarchive.org/y",
                    "license": "cc-by",
                    "notes": "test",
                },
            },
            {
                "slug": "song_sync_delivered",
                "title": "Sync Delivered",
                "artist": "Major Label",
                "genre": "kpop",
                "license_bucket": "commercial_sync_internal",
                "intended_source": {
                    "kind": "sync_licensed",
                    "url": "",
                    "license": "sync-internal-research",
                    "notes": "test",
                },
            },
        ],
    )
    _populate_song_artifacts(eval_set, "song_delivered")
    _populate_song_artifacts(
        eval_set,
        "song_sync_delivered",
        license_bucket="commercial_sync_internal",
        write_audio=False,                  # sync slots don't ship audio
    )
    return eval_set


# ---------------------------------------------------------------------------
# load_manifest
# ---------------------------------------------------------------------------

def test_load_manifest_returns_full_song_list(small_eval_set):
    manifest = load_manifest(small_eval_set)
    assert manifest["eval_set"] == "test_eval_set"
    assert len(manifest["songs"]) == 3


def test_load_manifest_raises_on_missing_file(tmp_path):
    with pytest.raises(EvalManifestError, match="No manifest.yaml"):
        load_manifest(tmp_path / "nope")


def test_load_manifest_raises_on_malformed_yaml(tmp_path):
    eval_set = tmp_path / "broken"
    eval_set.mkdir()
    (eval_set / "manifest.yaml").write_text("not_a_mapping: [1, 2, 3\n  bad")
    # PyYAML raises a YAMLError, not our custom EvalManifestError.
    with pytest.raises(yaml.YAMLError):
        load_manifest(eval_set)


def test_load_manifest_raises_when_songs_missing(tmp_path):
    eval_set = tmp_path / "no_songs"
    eval_set.mkdir()
    (eval_set / "manifest.yaml").write_text(yaml.safe_dump({"schema_version": 1}))
    with pytest.raises(EvalManifestError, match="songs"):
        load_manifest(eval_set)


# ---------------------------------------------------------------------------
# iter_songs / iter_delivered
# ---------------------------------------------------------------------------

def test_iter_songs_yields_every_slot_delivered_or_not(small_eval_set):
    songs = list(iter_songs(small_eval_set))
    assert [s.slug for s in songs] == [
        "song_delivered",
        "song_undelivered",
        "song_sync_delivered",
    ]


def test_iter_delivered_excludes_undelivered_slots(small_eval_set):
    delivered = list(iter_delivered(small_eval_set))
    slugs = {s.slug for s in delivered}
    assert "song_delivered" in slugs
    assert "song_sync_delivered" in slugs
    assert "song_undelivered" not in slugs
    assert len(delivered) == 2


def test_undelivered_slot_has_none_artifacts(small_eval_set):
    songs = {s.slug: s for s in iter_songs(small_eval_set)}
    s = songs["song_undelivered"]
    assert s.audio_metadata is None
    assert s.audio_path is None
    assert s.ref_midi_path is None
    assert s.ref_musicxml_path is None
    assert s.structural is None
    assert s.is_fully_delivered is False


def test_delivered_fma_slot_resolves_local_audio(small_eval_set):
    songs = {s.slug: s for s in iter_songs(small_eval_set)}
    s = songs["song_delivered"]
    assert s.is_fully_delivered is True
    assert s.audio_path is not None
    assert s.audio_path.is_file()
    assert s.audio_path.suffix == ".mp3"
    # FMA bucket → audio_uri returns the local path.
    assert s.audio_uri == str(s.audio_path)


def test_delivered_sync_slot_uses_internal_storage_uri(small_eval_set):
    songs = {s.slug: s for s in iter_songs(small_eval_set)}
    s = songs["song_sync_delivered"]
    assert s.is_fully_delivered is True
    # Sync bucket → no local audio file, but audio_uri falls through to
    # the bucket pointer.
    assert s.audio_path is None
    assert s.audio_uri == "s3://oh-sheet-eval-private/foo.mp3"


# ---------------------------------------------------------------------------
# Partial delivery cases
# ---------------------------------------------------------------------------

def test_fma_slot_missing_audio_is_not_fully_delivered(tmp_path):
    eval_set = tmp_path / "partial"
    _write_manifest(
        eval_set,
        [{"slug": "x", "title": "x", "artist": "x", "genre": "pop_mainstream",
          "license_bucket": "fma_redistributable",
          "intended_source": {"kind": "fma_track", "url": "", "license": "cc-by", "notes": ""}}],
    )
    _populate_song_artifacts(eval_set, "x", write_audio=False)
    s = load_song_by_slug(eval_set, "x")
    # Audio metadata + MIDI + XML + structural are present, but the
    # actual audio file is missing — FMA bucket gates require it.
    assert s.audio_metadata is not None
    assert s.ref_midi_path is not None
    assert s.is_fully_delivered is False


def test_slot_missing_structural_is_not_fully_delivered(tmp_path):
    eval_set = tmp_path / "no_struct"
    _write_manifest(
        eval_set,
        [{"slug": "x", "title": "x", "artist": "x", "genre": "pop_mainstream",
          "license_bucket": "fma_redistributable",
          "intended_source": {"kind": "fma_track", "url": "", "license": "cc-by", "notes": ""}}],
    )
    _populate_song_artifacts(eval_set, "x", write_structural=False)
    s = load_song_by_slug(eval_set, "x")
    assert s.structural is None
    assert s.is_fully_delivered is False


# ---------------------------------------------------------------------------
# StructuralReference → HarmonicAnalysis projection
# ---------------------------------------------------------------------------

def test_structural_reference_loads_keys_correctly(small_eval_set):
    s = load_song_by_slug(small_eval_set, "song_delivered")
    ref = s.structural
    assert isinstance(ref, StructuralReference)
    assert ref.key == "C:major"
    assert ref.time_signature == (4, 4)
    assert ref.tempo_bpm == 120.0
    assert ref.downbeat_sec == [0.0, 2.0, 4.0, 6.0]
    assert ref.chord_progression[0]["label"] == "C:maj"


def test_structural_projects_to_harmonic_analysis_dict(small_eval_set):
    s = load_song_by_slug(small_eval_set, "song_delivered")
    proj = s.structural.as_harmonic_analysis_dict()
    # Keys that HarmonicAnalysis expects.
    assert proj["key"] == "C:major"
    assert proj["time_signature"] == (4, 4)
    assert proj["downbeats"] == [0.0, 2.0, 4.0, 6.0]
    # ``chords`` is the field name on HarmonicAnalysis; structural
    # exposes it as ``chord_progression`` and the projection renames.
    assert proj["chords"][0]["label"] == "C:maj"


def test_structural_projection_round_trips_into_harmonic_analysis():
    """The projection actually constructs a valid ``HarmonicAnalysis``."""
    from backend.contracts import HarmonicAnalysis

    ref = StructuralReference.from_mapping(
        {
            "key": "G:major",
            "time_signature": "3/4",
            "tempo_bpm": 90.0,
            "tempo_map": [{"time_sec": 0.0, "beat": 0.0, "bpm": 90.0}],
            "sections": [],
            "chord_progression": [],
            "downbeat_sec": [0.0, 2.0],
            "license": "cc-by",
        }
    )
    proj = ref.as_harmonic_analysis_dict()
    ha = HarmonicAnalysis(**proj)
    assert ha.key == "G:major"
    assert ha.time_signature == (3, 4)
    assert ha.downbeats == [0.0, 2.0]


def test_structural_accepts_list_time_signature():
    """``[4, 4]`` as YAML list parses just like ``"4/4"``."""
    ref = StructuralReference.from_mapping(
        {
            "key": "C:major",
            "time_signature": [6, 8],
            "tempo_bpm": 120.0,
            "tempo_map": [],
            "sections": [],
            "chord_progression": [],
            "downbeat_sec": [],
            "license": "cc-by",
        }
    )
    assert ref.time_signature == (6, 8)


# ---------------------------------------------------------------------------
# load_song_by_slug
# ---------------------------------------------------------------------------

def test_load_song_by_slug_raises_on_unknown(small_eval_set):
    with pytest.raises(EvalManifestError, match="no song with slug"):
        load_song_by_slug(small_eval_set, "does_not_exist")


# ---------------------------------------------------------------------------
# Real pop_eval_v1 manifest sanity check
# ---------------------------------------------------------------------------

def test_real_pop_eval_v1_manifest_loads_30_slots():
    """The committed pop_eval_v1 manifest parses + has the 30 expected slots.

    Locks the slot count so a future reorganization (say, splitting
    pop_eval_v1 into pop_eval_v1_fma + pop_eval_v1_commercial) doesn't
    silently halve the eval surface.
    """
    eval_set = REPO_ROOT / "eval" / "pop_eval_v1"
    if not (eval_set / "manifest.yaml").exists():
        pytest.skip("pop_eval_v1 not present in this checkout")
    manifest = load_manifest(eval_set)
    assert manifest["eval_set"] == "pop_eval_v1"
    assert len(manifest["songs"]) == 30
    # 10 FMA + 20 sync slots per the plan.
    fma = [s for s in manifest["songs"] if s.get("license_bucket") == "fma_redistributable"]
    sync = [
        s for s in manifest["songs"]
        if s.get("license_bucket") == "commercial_sync_internal"
    ]
    assert len(fma) == 10
    assert len(sync) == 20


def test_real_pop_eval_v1_undelivered_slots_iter_songs_succeeds():
    """``iter_songs`` returns 30 EvalSong even though no slot is delivered."""
    eval_set = REPO_ROOT / "eval" / "pop_eval_v1"
    if not (eval_set / "manifest.yaml").exists():
        pytest.skip("pop_eval_v1 not present in this checkout")
    songs = list(iter_songs(eval_set))
    assert len(songs) == 30
    assert all(isinstance(s, EvalSong) for s in songs)
    # As of Phase 3 init, no slot is delivered yet.
    assert all(not s.is_fully_delivered for s in songs)


# ---------------------------------------------------------------------------
# AudioMetadata edge cases
# ---------------------------------------------------------------------------

def test_audio_metadata_handles_missing_internal_storage_uri():
    meta = AudioMetadata.from_mapping(
        {
            "content_hash": "sha256:x",
            "sample_rate": 44100,
            "duration_sec": 1.0,
            "format": "wav",
            "license": "cc-by",
            "license_bucket": "fma_redistributable",
            "source_url": "https://example.org/x",
        }
    )
    assert meta.internal_storage_uri is None
