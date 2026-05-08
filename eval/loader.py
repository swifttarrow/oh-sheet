"""Eval-set loader for Phase 3+ paid eval sets (``pop_eval_v1`` and beyond).

Reads ``manifest.yaml`` from an eval-set directory and yields per-song
``EvalSong`` records carrying paths to the four artifacts a delivered
song bundles:

* ``source.audio.json`` + the audio file (or a private-bucket URI for
  sync-licensed tracks)
* ``reference.piano_cover.mid`` — contract-transcribed ground truth
* ``reference.piano_cover.musicxml`` — engraved reference
* ``structural.yaml`` — human-verified key, time-sig, tempo, sections,
  chord progression, and downbeats

Slots that haven't been delivered yet are still listed so callers can
report progress (``n_delivered / n_total``) but excluded from
``iter_delivered``. The loader is the single point of truth for "is this
song eval-ready?" — the harness, holdout split, and CI all gate on
:func:`EvalSong.is_fully_delivered`.

The eval-set manifest schema is documented in
``eval/pop_eval_v1/README.md``. The structural.yaml schema mirrors the
strategy doc §3.2 example and loads directly into a
``HarmonicAnalysis``-compatible dict (with an extra ``downbeat_sec``
list — already accepted as ``HarmonicAnalysis.downbeats``).

This module does **not** depend on the Phase-0 ``tier_rf`` stack or on
Basic Pitch — it's a thin schema layer so the test harness can stay
deterministic and fast. The metric machinery (``scripts/eval_mini.py``,
the future ``scripts/eval.py``) consumes ``EvalSong`` records and pairs
them with the live pipeline output.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Paths inside ``songs/<slug>/`` for each deliverable. Constants so tests
# and the holdout / curate scripts can reference them by name instead of
# hard-coding the filenames in three places.
AUDIO_METADATA_FILENAME = "source.audio.json"
REFERENCE_MIDI_FILENAME = "reference.piano_cover.mid"
REFERENCE_MUSICXML_FILENAME = "reference.piano_cover.musicxml"
REFERENCE_PDF_FILENAME = "reference.piano_cover.pdf"
STRUCTURAL_FILENAME = "structural.yaml"
NOTES_FILENAME = "notes.md"

# Audio file extensions we accept for the redistributable bucket. A
# delivered FMA slot has exactly one of these alongside the JSON
# metadata; commercial_sync_internal slots have only the JSON (audio is
# fetched from the bucket URI at eval time).
_AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".m4a")


# ---------------------------------------------------------------------------
# Manifest schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IntendedSource:
    """Curator hint for an undelivered slot — the kind / URL / license to chase."""

    kind: str
    url: str
    license: str
    notes: str

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> IntendedSource:
        if not raw:
            return cls(kind="", url="", license="", notes="")
        return cls(
            kind=str(raw.get("kind", "")),
            url=str(raw.get("url", "")),
            license=str(raw.get("license", "")),
            notes=str(raw.get("notes", "")),
        )


@dataclass(frozen=True)
class AudioMetadata:
    """Contents of ``source.audio.json``."""

    content_hash: str
    sample_rate: int
    duration_sec: float
    format: str
    license: str
    license_bucket: str
    source_url: str
    internal_storage_uri: str | None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> AudioMetadata:
        return cls(
            content_hash=str(raw.get("content_hash", "")),
            sample_rate=int(raw.get("sample_rate", 0)),
            duration_sec=float(raw.get("duration_sec", 0.0)),
            format=str(raw.get("format", "")),
            license=str(raw.get("license", "")),
            license_bucket=str(raw.get("license_bucket", "")),
            source_url=str(raw.get("source_url", "")),
            internal_storage_uri=raw.get("internal_storage_uri") or None,
        )


@dataclass(frozen=True)
class StructuralReference:
    """Contents of ``structural.yaml``.

    Mirrors the strategy doc §3.2 schema. Loadable into a
    ``HarmonicAnalysis``-compatible dict via :meth:`as_harmonic_analysis_dict`.
    """

    key: str
    time_signature: tuple[int, int]
    tempo_bpm: float
    tempo_map: list[dict[str, float]]
    sections: list[dict[str, Any]]
    chord_progression: list[dict[str, Any]]
    downbeat_sec: list[float]
    license: str

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> StructuralReference:
        ts_raw = raw.get("time_signature", "4/4")
        time_signature = _parse_time_signature(ts_raw)
        return cls(
            key=str(raw.get("key", "")),
            time_signature=time_signature,
            tempo_bpm=float(raw.get("tempo_bpm", 0.0)),
            tempo_map=list(raw.get("tempo_map") or []),
            sections=list(raw.get("sections") or []),
            chord_progression=list(raw.get("chord_progression") or []),
            downbeat_sec=[float(x) for x in (raw.get("downbeat_sec") or [])],
            license=str(raw.get("license", "")),
        )

    def as_harmonic_analysis_dict(self) -> dict[str, Any]:
        """Project to the keys :class:`HarmonicAnalysis` accepts.

        Producers that import the contract live (``backend.contracts``)
        can do ``HarmonicAnalysis(**ref.as_harmonic_analysis_dict())``.
        Kept as a dict so the loader stays import-light — ``HarmonicAnalysis``
        pulls in the whole shared contracts module.
        """
        return {
            "key": self.key,
            "time_signature": self.time_signature,
            "tempo_map": [
                {
                    "time_sec": float(entry.get("time_sec", 0.0)),
                    "beat": float(entry.get("beat", 0.0)),
                    "bpm": float(entry.get("bpm", self.tempo_bpm)),
                }
                for entry in self.tempo_map
            ],
            "sections": list(self.sections),
            "chords": list(self.chord_progression),
            "downbeats": list(self.downbeat_sec),
        }


@dataclass(frozen=True)
class EvalSong:
    """One song from the eval-set manifest — delivered or not."""

    slug: str
    title: str
    artist: str
    genre: str
    license_bucket: str
    intended_source: IntendedSource
    songs_dir: Path                     # ``<eval_set>/songs/<slug>/``

    # Below are populated only when the artifacts are present on disk;
    # ``None`` for an undelivered slot.
    audio_metadata: AudioMetadata | None
    audio_path: Path | None
    ref_midi_path: Path | None
    ref_musicxml_path: Path | None
    ref_pdf_path: Path | None
    structural: StructuralReference | None
    notes_path: Path | None

    @property
    def is_fully_delivered(self) -> bool:
        """True iff the four release-gate artifacts are all present.

        ``ref_pdf_path`` and ``notes_path`` are nice-to-have for UX but
        not required for tier-1 mir_eval scoring, so they're not
        included in the gate. ``audio_path`` may be ``None`` for the
        ``commercial_sync_internal`` bucket — for those slots, having
        ``audio_metadata`` (with ``internal_storage_uri``) is sufficient
        because the release-bot fetches audio at eval time.
        """
        if self.audio_metadata is None:
            return False
        if (
            self.audio_metadata.license_bucket == "fma_redistributable"
            and self.audio_path is None
        ):
            return False
        return (
            self.ref_midi_path is not None
            and self.ref_musicxml_path is not None
            and self.structural is not None
        )

    @property
    def audio_uri(self) -> str | None:
        """Local path or private-bucket URI; whichever is appropriate.

        FMA-bucket slots return ``str(self.audio_path)``; sync-licensed
        slots return the ``internal_storage_uri``. Returns ``None`` when
        neither is resolvable (slot not delivered).
        """
        if self.audio_path is not None:
            return str(self.audio_path)
        if self.audio_metadata is not None and self.audio_metadata.internal_storage_uri:
            return self.audio_metadata.internal_storage_uri
        return None


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

class EvalManifestError(ValueError):
    """Raised when a manifest is malformed or missing required fields."""


def load_manifest(eval_set_path: Path) -> dict[str, Any]:
    """Load ``manifest.yaml`` from an eval-set directory.

    Validates only the top-level structure (``schema_version`` /
    ``eval_set`` / ``songs``); per-song validation is deferred to
    :func:`load_song` so a single broken slot doesn't sink a whole load.
    """
    eval_set_path = Path(eval_set_path)
    manifest_path = eval_set_path / "manifest.yaml"
    if not manifest_path.is_file():
        raise EvalManifestError(f"No manifest.yaml at {eval_set_path}")
    raw = yaml.safe_load(manifest_path.read_text())
    if not isinstance(raw, dict):
        raise EvalManifestError(
            f"manifest.yaml at {manifest_path} did not parse as a mapping"
        )
    if "songs" not in raw or not isinstance(raw["songs"], list):
        raise EvalManifestError("manifest.yaml must contain a 'songs' list")
    return raw


def load_song(eval_set_path: Path, song_entry: dict[str, Any]) -> EvalSong:
    """Build an :class:`EvalSong` for one manifest entry, populating any artifacts present."""
    eval_set_path = Path(eval_set_path)
    slug = song_entry.get("slug")
    if not isinstance(slug, str) or not slug:
        raise EvalManifestError(f"song entry missing slug: {song_entry!r}")

    songs_dir = eval_set_path / "songs" / slug

    audio_meta = _maybe_load_audio_metadata(songs_dir)
    audio_path = _maybe_locate_audio_file(songs_dir)
    ref_midi = _maybe_path(songs_dir / REFERENCE_MIDI_FILENAME)
    ref_xml = _maybe_path(songs_dir / REFERENCE_MUSICXML_FILENAME)
    ref_pdf = _maybe_path(songs_dir / REFERENCE_PDF_FILENAME)
    structural = _maybe_load_structural(songs_dir)
    notes = _maybe_path(songs_dir / NOTES_FILENAME)

    return EvalSong(
        slug=slug,
        title=str(song_entry.get("title", "")),
        artist=str(song_entry.get("artist", "")),
        genre=str(song_entry.get("genre", "")),
        license_bucket=str(song_entry.get("license_bucket", "")),
        intended_source=IntendedSource.from_mapping(song_entry.get("intended_source")),
        songs_dir=songs_dir,
        audio_metadata=audio_meta,
        audio_path=audio_path,
        ref_midi_path=ref_midi,
        ref_musicxml_path=ref_xml,
        ref_pdf_path=ref_pdf,
        structural=structural,
        notes_path=notes,
    )


def iter_songs(eval_set_path: Path) -> Iterator[EvalSong]:
    """Yield every slot in the manifest — delivered or not.

    Useful for progress reporting (``sum(s.is_fully_delivered for s in
    iter_songs(...))``). Most metric runners want :func:`iter_delivered`
    instead.
    """
    manifest = load_manifest(eval_set_path)
    for entry in manifest["songs"]:
        if not isinstance(entry, dict):
            continue
        yield load_song(eval_set_path, entry)


def iter_delivered(eval_set_path: Path) -> Iterator[EvalSong]:
    """Yield only the eval-ready slots."""
    for song in iter_songs(eval_set_path):
        if song.is_fully_delivered:
            yield song


def load_song_by_slug(eval_set_path: Path, slug: str) -> EvalSong:
    """Look up a single song by slug. Raises if not in the manifest."""
    manifest = load_manifest(eval_set_path)
    for entry in manifest["songs"]:
        if isinstance(entry, dict) and entry.get("slug") == slug:
            return load_song(eval_set_path, entry)
    raise EvalManifestError(f"no song with slug={slug!r} in {eval_set_path}")


# ---------------------------------------------------------------------------
# Helpers — file-presence probes
# ---------------------------------------------------------------------------

def _maybe_path(p: Path) -> Path | None:
    return p if p.is_file() else None


def _maybe_load_audio_metadata(songs_dir: Path) -> AudioMetadata | None:
    meta_path = songs_dir / AUDIO_METADATA_FILENAME
    if not meta_path.is_file():
        return None
    raw = json.loads(meta_path.read_text())
    if not isinstance(raw, dict):
        return None
    return AudioMetadata.from_mapping(raw)


def _maybe_locate_audio_file(songs_dir: Path) -> Path | None:
    for ext in _AUDIO_EXTENSIONS:
        candidate = songs_dir / f"source.audio{ext}"
        if candidate.is_file():
            return candidate
    return None


def _maybe_load_structural(songs_dir: Path) -> StructuralReference | None:
    path = songs_dir / STRUCTURAL_FILENAME
    if not path.is_file():
        return None
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        return None
    return StructuralReference.from_mapping(raw)


def _parse_time_signature(value: Any) -> tuple[int, int]:
    """Accept ``"4/4"`` strings or ``[4, 4]`` lists; return a 2-tuple of ints."""
    if isinstance(value, str) and "/" in value:
        num, den = value.split("/", 1)
        return int(num), int(den)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise EvalManifestError(f"unparseable time_signature: {value!r}")
