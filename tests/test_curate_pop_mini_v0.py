"""Tests for the pop_mini_v0 curation tooling.

Covers:
* :mod:`scripts.curate_pop_mini_v0` — manifest rewrite preserves
  ``intended_source``, comments, and other songs verbatim; local-file
  source path works end-to-end; refuses to clobber without ``--force``;
  errors on unknown slug.
* :mod:`scripts.fma_catalog_filter` — license / genre / duration filter
  on a synthetic in-memory CSV (no network); markdown rendering shape;
  commercial-use detection on real CC URL strings.

These tests run in <60 s and don't touch the network. yt-dlp / URL
download paths aren't exercised here — they're trivial wrappers around
yt-dlp's API and the manifest-rewrite logic is identical for URL and
local-file inputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import curate_pop_mini_v0 as curate_mod  # noqa: E402
from scripts import fma_catalog_filter as fma_mod  # noqa: E402

# ---------------------------------------------------------------------------
# Manifest fixture
# ---------------------------------------------------------------------------

# A miniature pop_mini_v0-shaped manifest. The bootstrap source for
# ``alpha`` will be replaced by the curate script; ``beta`` and the
# blank lines / comments must survive verbatim.
_FIXTURE_MANIFEST = """\
schema_version: 1
eval_set: synthetic_test
description: |
  Synthetic manifest fixture for curate tests.
target_duration_sec: 30.0
chord_recognition_key: auto

songs:
  - slug: alpha
    title: "Alpha Track"
    artist: "Test Alpha"
    genre: pop_mainstream
    intended_source:
      kind: fma_track
      url: https://freemusicarchive.org/  # populate with real URL
      license: cc-by-4.0
      notes: |
        Target slot: pop mainstream.
    source:
      kind: synthetic_from_midi
      midi_path: ../fixtures/clean_midi/Alpha/alpha.mid
      content_hash: sha256:0000000000000000000000000000000000000000000000000000000000000000
      license: cc-by-4.0
      provenance: lakh_clean_midi_subset
      bootstrap: true

  - slug: beta
    title: "Beta Track"
    artist: "Test Beta"
    genre: ballad
    intended_source:
      kind: fma_track
      url: https://freemusicarchive.org/
      license: cc-by-4.0
      notes: |
        Target slot: ballad.
    source:
      kind: synthetic_from_midi
      midi_path: ../fixtures/clean_midi/Beta/beta.mid
      content_hash: sha256:1111111111111111111111111111111111111111111111111111111111111111
      license: cc-by-4.0
      provenance: lakh_clean_midi_subset
      bootstrap: true
"""


def _write_fixture_manifest(tmp_path: Path) -> Path:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(_FIXTURE_MANIFEST)
    return manifest_path


def _write_fake_audio(tmp_path: Path, name: str = "fake.mp3") -> Path:
    """Write a tiny non-zero file with an .mp3 extension.

    The curate script copies + hashes the file but doesn't decode the
    audio — any non-empty bytes work. (``shutil.copy2`` preserves the
    extension, which is the only field the curate path inspects.)
    """
    p = tmp_path / name
    p.write_bytes(b"ID3\x04\x00\x00\x00\x00\x00\x00fake-mp3-bytes-for-curate-test")
    return p


# ---------------------------------------------------------------------------
# curate_pop_mini_v0 — happy path on a local file
# ---------------------------------------------------------------------------

def test_curate_local_file_replaces_only_target_song(tmp_path):
    manifest_path = _write_fixture_manifest(tmp_path)
    audio_src = _write_fake_audio(tmp_path)

    result = curate_mod.curate(
        slug="alpha",
        source=str(audio_src),
        manifest_path=manifest_path,
        license_str="cc-by-4.0",
        title="Alpha Track",
        artist="Test Alpha",
        notes="curated via test fixture",
    )

    # Audio landed at the canonical path under the manifest dir.
    landed = result["audio_path"]
    assert landed.is_file()
    assert landed.relative_to(tmp_path) == Path("songs/alpha/source.audio.mp3")

    # Hash matches the source file's bytes.
    expected_hex = curate_mod._sha256_of(audio_src)
    assert result["source"]["content_hash"] == f"sha256:{expected_hex}"

    # Manifest reads back as a valid mapping with kind=audio_file on alpha
    # and kind=synthetic_from_midi unchanged on beta.
    import yaml as _yaml
    parsed = _yaml.safe_load(manifest_path.read_text())
    by_slug = {s["slug"]: s for s in parsed["songs"]}
    assert by_slug["alpha"]["source"]["kind"] == "audio_file"
    assert by_slug["alpha"]["source"]["bootstrap"] is False
    assert by_slug["alpha"]["source"]["path"] == "songs/alpha/source.audio.mp3"
    assert by_slug["alpha"]["source"]["license"] == "cc-by-4.0"
    # intended_source must be preserved verbatim.
    assert by_slug["alpha"]["intended_source"]["kind"] == "fma_track"
    assert by_slug["alpha"]["intended_source"]["license"] == "cc-by-4.0"

    # Beta is untouched.
    assert by_slug["beta"]["source"]["kind"] == "synthetic_from_midi"
    assert by_slug["beta"]["source"]["bootstrap"] is True


def test_curate_preserves_inline_comments_and_blank_lines(tmp_path):
    """Manifest rewrite is line-level: comments + blank lines stay put."""
    manifest_path = _write_fixture_manifest(tmp_path)
    audio_src = _write_fake_audio(tmp_path)

    curate_mod.curate(
        slug="alpha",
        source=str(audio_src),
        manifest_path=manifest_path,
        license_str="cc-by-4.0",
    )

    text = manifest_path.read_text()
    # The intended_source.url comment from the fixture must survive.
    assert "# populate with real URL" in text
    # The description block (multi-line | scalar) must survive.
    assert "Synthetic manifest fixture for curate tests." in text
    # The blank line separator between songs must survive.
    assert "\n\n  - slug: beta\n" in text


def test_curate_unknown_slug_raises(tmp_path):
    manifest_path = _write_fixture_manifest(tmp_path)
    audio_src = _write_fake_audio(tmp_path)

    with pytest.raises(ValueError, match="not found in manifest"):
        curate_mod.curate(
            slug="zeta",
            source=str(audio_src),
            manifest_path=manifest_path,
            license_str="cc-by-4.0",
        )


def test_curate_refuses_to_clobber_without_force(tmp_path):
    manifest_path = _write_fixture_manifest(tmp_path)
    audio_src = _write_fake_audio(tmp_path)

    curate_mod.curate(
        slug="alpha",
        source=str(audio_src),
        manifest_path=manifest_path,
        license_str="cc-by-4.0",
    )
    # Second invocation without --force: bail.
    with pytest.raises(FileExistsError, match="--force"):
        curate_mod.curate(
            slug="alpha",
            source=str(audio_src),
            manifest_path=manifest_path,
            license_str="cc-by-4.0",
        )

    # With force=True it succeeds.
    curate_mod.curate(
        slug="alpha",
        source=str(audio_src),
        manifest_path=manifest_path,
        license_str="cc-by-4.0",
        force=True,
    )


def test_curate_records_source_url_when_input_is_url(tmp_path, monkeypatch):
    """URL inputs land ``source_url`` in the rewritten source block.

    The yt-dlp download is mocked so this stays offline; what we verify
    is the rewrite-side handling of URL vs. local-path inputs.
    """
    manifest_path = _write_fixture_manifest(tmp_path)

    def fake_download(url, target_dir, target_stem):
        # Simulate yt-dlp landing an mp3 at the conventional path.
        target_dir.mkdir(parents=True, exist_ok=True)
        out = target_dir / f"{target_stem}.mp3"
        out.write_bytes(b"ID3\x04mocked-ytdlp-output")
        return out

    monkeypatch.setattr(curate_mod, "_download_via_ytdlp", fake_download)

    fake_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    result = curate_mod.curate(
        slug="alpha",
        source=fake_url,
        manifest_path=manifest_path,
        license_str="cc-by",
    )
    assert result["source"]["source_url"] == fake_url

    import yaml as _yaml
    parsed = _yaml.safe_load(manifest_path.read_text())
    by_slug = {s["slug"]: s for s in parsed["songs"]}
    assert by_slug["alpha"]["source"]["source_url"] == fake_url


def test_curate_rejects_unsupported_extension(tmp_path):
    manifest_path = _write_fixture_manifest(tmp_path)
    bad = tmp_path / "audio.txt"
    bad.write_bytes(b"not audio")
    with pytest.raises(ValueError, match="Unsupported audio extension"):
        curate_mod.curate(
            slug="alpha",
            source=str(bad),
            manifest_path=manifest_path,
            license_str="cc-by-4.0",
        )


# ---------------------------------------------------------------------------
# fma_catalog_filter — pure-data tests (no network)
# ---------------------------------------------------------------------------

# Simulates the FMA tracks.csv 2-row multi-index header + one synthetic
# row each for a CC-BY pop track, a CC-BY-NC pop track (filtered out by
# default), a too-short pop track, a too-long classical track, and a
# CC-BY classical track. First column is the integer track_id (empty
# header cells in both rows); subsequent columns are <section>.<field>.
_FAKE_FMA_CSV = """\
,album,album,artist,artist,track,track,track,track,track
,id,title,id,name,duration,genre_top,license,listens,title
1,10,Album One,100,Alice,180,Pop,http://creativecommons.org/licenses/by/4.0/,5000,Sunny Day
2,11,Album Two,101,Bob,210,Pop,http://creativecommons.org/licenses/by-nc/3.0/,12000,Late Night
3,12,Album Three,102,Carol,30,Pop,http://creativecommons.org/licenses/by/4.0/,400,Brief Burst
4,13,Album Four,103,Dave,3600,Classical,http://creativecommons.org/licenses/by/4.0/,800,Long Sonata
5,14,Album Five,104,Eve,200,Classical,http://creativecommons.org/licenses/by/4.0/,2500,Soft Theme
"""


def _write_fake_fma_csv(tmp_path: Path) -> Path:
    p = tmp_path / "tracks.csv"
    p.write_text(_FAKE_FMA_CSV)
    return p


def test_fma_filter_drops_non_commercial_by_default(tmp_path):
    csv_path = _write_fake_fma_csv(tmp_path)
    matches = fma_mod.filter_tracks(
        csv_path,
        genre="Pop",
        min_duration_sec=60,
        max_duration_sec=360,
        query=None,
        commercial_only=True,
    )
    # Sunny Day (CC BY) survives; Late Night (CC BY-NC) is dropped;
    # Brief Burst is too short.
    titles = sorted(t.title for t in matches)
    assert titles == ["Sunny Day"]


def test_fma_filter_includes_non_commercial_when_allowed(tmp_path):
    csv_path = _write_fake_fma_csv(tmp_path)
    matches = fma_mod.filter_tracks(
        csv_path,
        genre="Pop",
        min_duration_sec=60,
        max_duration_sec=360,
        query=None,
        commercial_only=False,
    )
    titles = sorted(t.title for t in matches)
    assert titles == ["Late Night", "Sunny Day"]


def test_fma_filter_genre_match(tmp_path):
    csv_path = _write_fake_fma_csv(tmp_path)
    matches = fma_mod.filter_tracks(
        csv_path,
        genre="Classical",
        min_duration_sec=60,
        max_duration_sec=360,
        query=None,
        commercial_only=True,
    )
    titles = sorted(t.title for t in matches)
    # Long Sonata is too long; Soft Theme survives.
    assert titles == ["Soft Theme"]


def test_fma_filter_query_substring(tmp_path):
    csv_path = _write_fake_fma_csv(tmp_path)
    matches = fma_mod.filter_tracks(
        csv_path,
        genre=None,
        min_duration_sec=60,
        max_duration_sec=360,
        query="bob",
        commercial_only=False,
    )
    # Bob's track is the CC-BY-NC one, included with commercial_only=False.
    assert {t.artist for t in matches} == {"Bob"}


def test_fma_filter_render_markdown_shape(tmp_path):
    csv_path = _write_fake_fma_csv(tmp_path)
    matches = fma_mod.filter_tracks(
        csv_path,
        genre="Pop",
        min_duration_sec=60,
        max_duration_sec=360,
        query=None,
        commercial_only=True,
    )
    md = fma_mod.render_markdown(matches, top_n=10)
    assert "| listens |" in md
    assert "Sunny Day" in md
    assert "https://freemusicarchive.org/track/1/" in md
    # Dropped tracks must not appear.
    assert "Late Night" not in md
    assert "Brief Burst" not in md


def test_fma_track_license_short_handles_versioned_urls():
    cases = [
        ("http://creativecommons.org/licenses/by/4.0/", "CC BY 4.0"),
        ("http://creativecommons.org/licenses/by-sa/3.0/", "CC BY-SA 3.0"),
        ("http://creativecommons.org/licenses/by-nc/2.5/", "CC BY-NC 2.5"),
        ("http://creativecommons.org/publicdomain/zero/1.0/", "CC0"),
        ("https://example.com/random-license", "https://example.com/random-license"),
    ]
    for url, expected in cases:
        track = fma_mod.TrackRow(
            track_id=1, title="t", artist="a", genre_top="Pop",
            duration_sec=120, license_url=url, listens=1,
            fma_track_url="https://example.com/",
        )
        assert track.license_short() == expected


def test_fma_track_commercial_ok_blocks_nc_variants():
    nc_urls = [
        "http://creativecommons.org/licenses/by-nc/3.0/",
        "http://creativecommons.org/licenses/by-nc-sa/4.0/",
        "http://creativecommons.org/licenses/by-nc-nd/4.0/",
    ]
    for url in nc_urls:
        track = fma_mod.TrackRow(
            track_id=1, title="t", artist="a", genre_top="Pop",
            duration_sec=120, license_url=url, listens=1,
            fma_track_url="https://example.com/",
        )
        assert not track.commercial_ok(), f"{url} should be flagged non-commercial"

    commercial_urls = [
        "http://creativecommons.org/licenses/by/4.0/",
        "http://creativecommons.org/licenses/by-sa/3.0/",
        "http://creativecommons.org/publicdomain/zero/1.0/",
    ]
    for url in commercial_urls:
        track = fma_mod.TrackRow(
            track_id=1, title="t", artist="a", genre_top="Pop",
            duration_sec=120, license_url=url, listens=1,
            fma_track_url="https://example.com/",
        )
        assert track.commercial_ok(), f"{url} should be commercial-ok"


def test_fma_filter_iter_skips_blank_track_id_rows(tmp_path):
    """A row with an empty track_id (extracted from a malformed FMA dump) is skipped, not raised."""
    p = tmp_path / "tracks.csv"
    p.write_text(
        ",album,album,artist,artist,track,track,track,track,track\n"
        ",id,title,id,name,duration,genre_top,license,listens,title\n"
        ",,,,,,,,,Empty Row\n"  # missing track_id
        "9,99,A,99,X,120,Pop,http://creativecommons.org/licenses/by/4.0/,10,Real\n"
    )
    rows = list(fma_mod.iter_tracks(p))
    assert [r.track_id for r in rows] == [9]
