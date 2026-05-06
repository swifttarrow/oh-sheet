"""Phase 0 reference-free mini-eval CLI.

Runs the Oh Sheet pipeline (transcribe → arrange → humanize → midi-render)
on every song in an eval-set manifest, computes the three reference-free
metrics from :mod:`eval.tier_rf`, and writes ``aggregate.json`` plus a
stdout summary.

Usage::

    python scripts/eval_mini.py eval/pop_mini_v0/ eval/runs/$(date -u +%Y%m%dT%H%M%SZ)/

    # Run a single song (debugging)
    python scripts/eval_mini.py eval/pop_mini_v0/ /tmp/run/ --slug mini_pop_001

The CLI is structured around a top-level :func:`run` function so the
future ``scripts/eval.py`` Click app from Phase 7 can wrap it as a
subcommand without rewriting the orchestration logic. See
``docs/research/transcription-improvement-implementation-plan.md``
§Phase 0 / §Phase 7 for the migration path.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import io
import json
import logging
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent

# Drop scripts/ from sys.path so ``import eval`` resolves to the
# eval/ package at REPO_ROOT, not to ``scripts/eval.py`` (the Phase 7
# Click CLI). Python auto-adds the script dir to sys.path[0] when
# this file is run as ``__main__``; without the scrub, ``eval``
# shadows the package.
sys.path[:] = [p for p in sys.path if Path(p).resolve() != SCRIPTS_DIR]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from eval.tier_rf import (  # noqa: E402
    TierRfResult,
    compute_tier_rf,
    fluidsynth_resynth,
)

log = logging.getLogger(__name__)

# Cache directory for FluidSynth-rendered bootstrap WAVs and pipeline
# artifacts. Mirrors ``scripts/eval_transcription.py``'s
# ``.cache/eval_transcription`` convention so eval state stays
# repo-local and gitignored.
CACHE_DIR = REPO_ROOT / ".cache" / "eval_mini"


# ---------------------------------------------------------------------------
# Manifest + audio resolution
# ---------------------------------------------------------------------------

def _load_manifest(eval_set_path: Path) -> dict[str, Any]:
    """Load and minimally validate ``manifest.yaml`` from an eval-set dir."""
    manifest_path = eval_set_path / "manifest.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No manifest.yaml at {eval_set_path}")
    manifest = yaml.safe_load(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest.yaml at {manifest_path} did not parse as a mapping")
    if "songs" not in manifest or not isinstance(manifest["songs"], list):
        raise ValueError("manifest.yaml must contain a 'songs' list")
    return manifest


def _resolve_input_audio(
    song: dict[str, Any],
    manifest_dir: Path,
    target_duration_sec: float,
) -> Path:
    """Return the audio Path for a song manifest entry.

    Supports both the legacy ``pop_mini_v0`` manifest schema (``source``
    block with ``kind: audio_file`` / ``synthetic_from_midi``) and the
    Phase 3 ``pop_eval_v1`` delivery schema (no ``source`` field;
    audio file lands under ``songs/<slug>/source.audio.{mp3,wav,…}``).
    Slots with neither raise ``FileNotFoundError`` so
    :func:`eval.harness.run_eval_set` records them as a per-song
    resolve error rather than crashing the whole run.
    """
    src = song.get("source")
    if src is None:
        # ``pop_eval_v1`` schema: no ``source`` in the manifest. The
        # delivered audio lands under ``songs/<slug>/source.audio.*``;
        # the loader treats any of mp3/wav/flac/m4a as acceptable.
        slug = song.get("slug")
        if not isinstance(slug, str) or not slug:
            raise FileNotFoundError(
                "song entry has no 'source' block and no slug — "
                "cannot resolve delivered audio"
            )
        songs_dir = manifest_dir / "songs" / slug
        for ext in ("mp3", "wav", "flac", "m4a"):
            candidate = songs_dir / f"source.audio.{ext}"
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(
            f"slug={slug!r} not delivered: no source.audio.* under {songs_dir}"
        )

    kind = src["kind"]
    if kind == "audio_file":
        path = (manifest_dir / src["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"audio_file source missing: {path}")
        return path

    if kind == "synthetic_from_midi":
        midi_path = (manifest_dir / src["midi_path"]).resolve()
        if not midi_path.is_file():
            raise FileNotFoundError(
                f"synthetic_from_midi source MIDI missing: {midi_path}"
            )
        return _bootstrap_wav_from_midi(midi_path, target_duration_sec)

    raise ValueError(f"unknown source.kind: {kind!r} for slug={song.get('slug')!r}")


def _read_pretty_midi_robust(midi_path: Path) -> Any:
    """Load a MIDI through ``pretty_midi``, clipping malformed bytes if needed.

    Some clean_midi entries (notably ``Aqua/Dr Jones.mid``) contain raw
    bytes outside ``0..127`` that ``mido``'s strict reader rejects with
    ``OSError: data byte must be in range 0..127``. ``mido(clip=True)``
    clamps those bytes to the valid range; piping that through
    ``pretty_midi.PrettyMIDI`` recovers a usable ``PrettyMIDI`` for
    downstream synthesis.
    """
    import pretty_midi  # noqa: PLC0415
    try:
        return pretty_midi.PrettyMIDI(str(midi_path))
    except OSError:
        import mido  # noqa: PLC0415

        clipped = mido.MidiFile(filename=str(midi_path), clip=True)
        buf = io.BytesIO()
        clipped.save(file=buf)
        buf.seek(0)
        return pretty_midi.PrettyMIDI(buf)


def _bootstrap_wav_from_midi(
    midi_path: Path,
    target_duration_sec: float,
) -> Path:
    """FluidSynth-render a MIDI file to a cached WAV.

    Truncates to ``target_duration_sec`` before synthesis (matches
    ``scripts/eval_transcription.py``'s 30-second clipping convention so
    a Phase 0 run finishes inside the <2 min window). Cache key is the
    same triple the existing eval harness uses, so a future ``eval.py``
    that shares a cache directory won't re-render the same fixtures.
    """
    import pretty_midi  # noqa: PLC0415  (only needed for the soundfont path)

    bootstrap_dir = CACHE_DIR / "bootstrap"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    soundfont = Path(pretty_midi.__file__).parent / "TimGM6mb.sf2"

    payload = f"{midi_path.resolve()}|{target_duration_sec}|{soundfont.resolve()}"
    key = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    out_wav = bootstrap_dir / f"{key}.wav"
    if out_wav.is_file():
        return out_wav

    pm = _read_pretty_midi_robust(midi_path)
    if target_duration_sec > 0:
        for inst in pm.instruments:
            kept = []
            for n in inst.notes:
                if n.start >= target_duration_sec:
                    continue
                if n.end > target_duration_sec:
                    n.end = target_duration_sec
                kept.append(n)
            inst.notes = kept
            inst.control_changes = [
                cc for cc in inst.control_changes if cc.time < target_duration_sec
            ]
            inst.pitch_bends = [
                pb for pb in inst.pitch_bends if pb.time < target_duration_sec
            ]
        # Track-level meta events also extend get_end_time() and keep
        # FluidSynth rendering silence past the last note. Same fix
        # ``scripts/eval_transcription.py:_truncate_midi`` applies.
        pm.lyrics = [lyr for lyr in pm.lyrics if lyr.time < target_duration_sec]
        pm.text_events = [te for te in pm.text_events if te.time < target_duration_sec]
        pm.time_signature_changes = [
            ts for ts in pm.time_signature_changes if ts.time < target_duration_sec
        ]
        pm.key_signature_changes = [
            ks for ks in pm.key_signature_changes if ks.time < target_duration_sec
        ]

    buf = io.BytesIO()
    pm.write(buf)
    midi_bytes = buf.getvalue()

    audio, sr = fluidsynth_resynth(midi_bytes)
    import soundfile as sf  # noqa: PLC0415
    sf.write(str(out_wav), audio, sr)
    return out_wav


# ---------------------------------------------------------------------------
# Pipeline (in-process, no Celery) — mirrors PipelineRunner's stage chain
# ---------------------------------------------------------------------------

@dataclass
class PipelineArtifacts:
    score: Any                # PianoScore
    midi_bytes: bytes         # output of render_midi_bytes(perf)
    key_label: str            # HarmonicAnalysis.key — passed to chord recog
    transcription: Any        # TranscriptionResult, kept for diagnostics


def _run_pipeline(audio_path: Path) -> PipelineArtifacts:
    """Drive transcribe → arrange → humanize → render in-process.

    The Celery-based PipelineRunner serializes stage IO through a blob
    store; for an offline mini-eval we call the service entry points
    directly. This is the same pattern ``scripts/eval_transcription.py``
    uses (``_run_basic_pitch_sync``).

    Returns the score (for playability_rf) and the engraved MIDI bytes
    (for FluidSynth re-synth → chord_rf + chroma_rf), plus the key
    label so chord recognition uses a symmetric HMM prior on both sides.
    """
    from backend.services.arrange import ArrangeService  # noqa: PLC0415
    from backend.services.humanize import HumanizeService  # noqa: PLC0415
    from backend.services.midi_render import render_midi_bytes  # noqa: PLC0415
    from backend.services.transcribe import _run_basic_pitch_sync  # noqa: PLC0415

    # Phase 8: ``_run_basic_pitch_sync`` returns a 3-tuple
    # ``(TranscriptionResult, midi_bytes, realtime_pedal_events)`` —
    # the third element is the AMT-APC / Kong pedal stream and is
    # unused by the offline mini-eval (engrave handles pedals).
    txr, _midi_bytes, _pedals = _run_basic_pitch_sync(audio_path)
    score = asyncio.run(ArrangeService().run(txr))
    perf = asyncio.run(HumanizeService().run(score))
    midi_bytes = render_midi_bytes(perf)
    return PipelineArtifacts(
        score=score,
        midi_bytes=midi_bytes,
        key_label=txr.analysis.key,
        transcription=txr,
    )


# ---------------------------------------------------------------------------
# Per-song eval + aggregation
# ---------------------------------------------------------------------------

@dataclass
class SongRow:
    slug: str
    title: str | None = None
    artist: str | None = None
    genre: str | None = None
    audio_path: str | None = None
    key_label: str | None = None
    chord_rf: float = 0.0
    playability_rf: float = 0.0
    chroma_rf: float = 0.0
    n_chord_segments_input: int = 0
    n_chord_segments_resynth: int = 0
    n_playable_chords: int = 0
    n_total_chords: int = 0
    n_beats: int = 0
    wall_sec: float = 0.0
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out = {
            "slug": self.slug,
            "title": self.title,
            "artist": self.artist,
            "genre": self.genre,
            "audio_path": self.audio_path,
            "key_label": self.key_label,
            "chord_rf": round(self.chord_rf, 4),
            "playability_rf": round(self.playability_rf, 4),
            "chroma_rf": round(self.chroma_rf, 4),
            "n_chord_segments_input": self.n_chord_segments_input,
            "n_chord_segments_resynth": self.n_chord_segments_resynth,
            "n_playable_chords": self.n_playable_chords,
            "n_total_chords": self.n_total_chords,
            "n_beats": self.n_beats,
            "wall_sec": round(self.wall_sec, 2),
            "notes": list(self.notes),
            "error": self.error,
        }
        return out


def _per_song(
    song: dict[str, Any],
    manifest_dir: Path,
    target_duration_sec: float,
    chord_recognition_key: str,
) -> SongRow:
    slug = song["slug"]
    row = SongRow(
        slug=slug,
        title=song.get("title"),
        artist=song.get("artist"),
        genre=song.get("genre"),
    )
    t0 = time.perf_counter()
    try:
        audio_path = _resolve_input_audio(song, manifest_dir, target_duration_sec)
        row.audio_path = (
            str(audio_path.relative_to(REPO_ROOT))
            if audio_path.is_relative_to(REPO_ROOT)
            else str(audio_path)
        )

        artifacts = _run_pipeline(audio_path)
        row.key_label = artifacts.key_label

        # Manifest can pin the chord-recognition key prior; ``auto``
        # defers to whatever the transcribe stage just emitted.
        key_for_recog = (
            artifacts.key_label
            if chord_recognition_key == "auto"
            else chord_recognition_key
        )

        result: TierRfResult = compute_tier_rf(
            audio_path,
            artifacts.score,
            artifacts.midi_bytes,
            key_label=key_for_recog,
        )
        row.chord_rf = result.chord_rf
        row.playability_rf = result.playability_rf
        row.chroma_rf = result.chroma_rf
        row.n_chord_segments_input = result.n_chord_segments_input
        row.n_chord_segments_resynth = result.n_chord_segments_resynth
        row.n_playable_chords = result.n_playable_chords
        row.n_total_chords = result.n_total_chords
        row.n_beats = result.n_beats
        row.notes = list(result.notes)
    except Exception as exc:  # noqa: BLE001 — one bad song must not sink the run
        log.exception("per-song eval failed slug=%s", slug)
        row.error = f"{type(exc).__name__}: {exc}"
    row.wall_sec = time.perf_counter() - t0
    return row


def _aggregate(rows: list[SongRow]) -> dict[str, Any]:
    ok = [r for r in rows if r.error is None]
    agg: dict[str, Any] = {
        "n_songs_total": len(rows),
        "n_songs_scored": len(ok),
        "n_songs_errored": len(rows) - len(ok),
    }
    if ok:
        agg.update({
            "mean_chord_rf": round(statistics.fmean(r.chord_rf for r in ok), 4),
            "mean_playability_rf": round(
                statistics.fmean(r.playability_rf for r in ok), 4,
            ),
            "mean_chroma_rf": round(statistics.fmean(r.chroma_rf for r in ok), 4),
            "median_chord_rf": round(
                statistics.median(r.chord_rf for r in ok), 4,
            ),
            "median_playability_rf": round(
                statistics.median(r.playability_rf for r in ok), 4,
            ),
            "median_chroma_rf": round(
                statistics.median(r.chroma_rf for r in ok), 4,
            ),
            "mean_wall_sec": round(statistics.fmean(r.wall_sec for r in ok), 2),
        })
    return agg


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _print_table(rows: list[SongRow], agg: dict[str, Any]) -> None:
    """6-line stdout summary matching the Phase 0 demo checkpoint shape."""
    print()
    header = (
        f"  {'song':<22}  {'chord-rf':>8}  "
        f"{'playability-rf':>14}  {'chroma-rf':>9}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        if r.error:
            print(f"  {r.slug:<22}  ERROR: {r.error[:60]}")
            continue
        print(
            f"  {r.slug:<22}  "
            f"{r.chord_rf:>8.3f}  "
            f"{r.playability_rf:>14.3f}  "
            f"{r.chroma_rf:>9.3f}"
        )
    if "mean_chord_rf" in agg:
        print(
            f"  {'mean':<22}  "
            f"{agg['mean_chord_rf']:>8.3f}  "
            f"{agg['mean_playability_rf']:>14.3f}  "
            f"{agg['mean_chroma_rf']:>9.3f}"
        )


# ---------------------------------------------------------------------------
# Top-level run() — the function a future eval.py subcommand will wrap
# ---------------------------------------------------------------------------

def run(
    eval_set_path: Path,
    output_dir: Path,
    *,
    only_slug: str | None = None,
) -> dict[str, Any]:
    """Run the mini-eval and return the payload that gets written to disk.

    Parameters
    ----------
    eval_set_path
        Directory containing ``manifest.yaml`` (e.g. ``eval/pop_mini_v0/``).
    output_dir
        Where ``aggregate.json`` will be written. Created if missing.
    only_slug
        If set, runs only the song with that slug — useful for debugging
        a single fixture without paying the full 5-song latency.
    """
    eval_set_path = Path(eval_set_path).resolve()
    output_dir = Path(output_dir).resolve()

    manifest = _load_manifest(eval_set_path)
    songs = manifest["songs"]
    if only_slug is not None:
        songs = [s for s in songs if s.get("slug") == only_slug]
        if not songs:
            raise ValueError(f"no song with slug={only_slug!r} in manifest")
    target_duration = float(manifest.get("target_duration_sec", 30.0))
    chord_key = str(manifest.get("chord_recognition_key", "auto"))

    print("=== mini-eval ===")
    print(f"  manifest:  {eval_set_path / 'manifest.yaml'}")
    print(f"  songs:     {len(songs)} of {len(manifest['songs'])}")
    print(f"  duration:  {target_duration:.1f}s per song")
    print(f"  cache:     {CACHE_DIR.relative_to(REPO_ROOT)}")
    print()

    rows: list[SongRow] = []
    for song in songs:
        slug = song.get("slug", "<no-slug>")
        print(f"  [{slug}] running…", flush=True)
        row = _per_song(song, eval_set_path, target_duration, chord_key)
        rows.append(row)
        if row.error:
            print(f"    ! error: {row.error}")
        else:
            print(
                f"    chord_rf={row.chord_rf:.3f}  "
                f"playability_rf={row.playability_rf:.3f}  "
                f"chroma_rf={row.chroma_rf:.3f}  ({row.wall_sec:.1f}s)"
            )

    agg = _aggregate(rows)
    payload = {
        "schema_version": 1,
        "eval_set": manifest.get("eval_set", "unknown"),
        "manifest_relpath": (
            str((eval_set_path / "manifest.yaml").relative_to(REPO_ROOT))
            if eval_set_path.is_relative_to(REPO_ROOT)
            else str(eval_set_path / "manifest.yaml")
        ),
        "run_id": dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ"),
        "git_sha": _git_sha(),
        "config": {
            "target_duration_sec": target_duration,
            "chord_recognition_key": chord_key,
        },
        "songs": [r.as_dict() for r in rows],
        "aggregate": agg,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "aggregate.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    _print_table(rows, agg)
    print(f"\nWrote {out_path}")
    return payload


def main() -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "eval_set_path",
        type=Path,
        help="Directory containing manifest.yaml (e.g. eval/pop_mini_v0/).",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Where to write aggregate.json (created if missing).",
    )
    parser.add_argument(
        "--slug",
        type=str,
        default=None,
        help="Only run the song with this slug (useful for debugging).",
    )
    args = parser.parse_args()

    run(
        args.eval_set_path,
        args.output_dir,
        only_slug=args.slug,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
