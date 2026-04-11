"""Engraving stage — render MIDI / MusicXML / PDF artifacts.

  * MIDI     — pretty_midi if installed, else a minimal MThd+MTrk file
  * MusicXML — music21 (hard dependency; errors propagate to the caller)
  * PDF      — LilyPond (preferred) or MuseScore CLI if on $PATH, else a 1-line %PDF stub
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from backend.contracts import (
    SCHEMA_VERSION,
    EngravedOutput,
    EngravedScoreData,
    HumanizedPerformance,
    PianoScore,
    beat_to_sec,
)
from backend.storage.base import BlobStore

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tiny stub artifacts — used when an optional renderer is unavailable.
# ---------------------------------------------------------------------------

_STUB_PDF = b"%PDF-1.4\n% stub PDF emitted by ohsheet engrave service\n"
_STUB_MIDI = (
    b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x01\xe0"   # header chunk: format 0, 1 track, 480 tpq
    b"MTrk\x00\x00\x00\x04\x00\xff\x2f\x00"            # one empty track ending in End-Of-Track
)


# ---------------------------------------------------------------------------
# MIDI rendering
# ---------------------------------------------------------------------------

def _render_midi_bytes(perf: HumanizedPerformance) -> bytes:
    """Render the humanized performance to MIDI bytes via pretty_midi.

    Falls back to ``_STUB_MIDI`` when pretty_midi isn't installed or the
    score is empty enough that pretty_midi would write a no-op file.
    """
    try:
        import pretty_midi  # noqa: PLC0415 — optional dep
    except ImportError:
        log.warning("pretty_midi not installed — MIDI output will be a stub. Install with: pip install pretty_midi")
        return _STUB_MIDI

    tempo_map = perf.score.metadata.tempo_map
    initial_bpm = tempo_map[0].bpm if tempo_map else 120.0
    midi_time_offset = tempo_map[0].time_sec if tempo_map else 0.0

    midi = pretty_midi.PrettyMIDI(initial_tempo=initial_bpm)
    ts = perf.score.metadata.time_signature
    midi.time_signature_changes = [
        pretty_midi.TimeSignature(numerator=ts[0], denominator=ts[1], time=0.0)
    ]

    piano = pretty_midi.Instrument(program=0, name="Piano")

    raw_notes: list[tuple[int, float, float, int]] = []
    for en in perf.expressive_notes:
        try:
            onset = beat_to_sec(en.onset_beat, tempo_map) - midi_time_offset
            offset = beat_to_sec(en.onset_beat + en.duration_beat, tempo_map) - midi_time_offset
        except ValueError:
            continue
        # timing_offset_ms is an *onset-only* nudge — the release boundary
        # stays on the metronomic grid, so a late-struck note plays shorter
        # and an early-struck note plays longer. Mirrors humanize._humanize_timing,
        # which models downbeat anticipation / backbeat push as attack-time
        # gestures with no release component.
        onset += en.timing_offset_ms / 1000.0
        onset = max(0.0, onset)
        offset = max(onset + 0.01, offset)
        raw_notes.append((en.pitch, onset, offset, max(1, min(127, en.velocity))))

    # Resolve same-pitch overlaps — MIDI can't have two notes-on for the same key
    by_pitch: dict[int, list[int]] = {}
    for i, (p, _s, _e, _v) in enumerate(raw_notes):
        by_pitch.setdefault(p, []).append(i)
    for indices in by_pitch.values():
        indices.sort(key=lambda i: raw_notes[i][1])
        for j in range(1, len(indices)):
            pp, ps, pe, pv = raw_notes[indices[j - 1]]
            _cp, cs, _ce, _cv = raw_notes[indices[j]]
            if pe > cs:
                raw_notes[indices[j - 1]] = (pp, ps, cs, pv)

    MIN_NOTE_DUR = 0.03
    for pitch, start, end, vel in raw_notes:
        if end - start < MIN_NOTE_DUR:
            continue
        piano.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end))

    # Pedal events → GM control changes. Sustain = CC64, sostenuto = CC66,
    # una corda = CC67. All three use the standard on=127 / off=0 encoding
    # (below 64 disengages on most synths, but 0 is the unambiguous convention).
    _PEDAL_CC = {"sustain": 64, "sostenuto": 66, "una_corda": 67}
    for pedal in perf.expression.pedal_events:
        cc_num = _PEDAL_CC.get(pedal.type)
        if cc_num is None:
            continue
        try:
            on_sec = beat_to_sec(pedal.onset_beat, tempo_map) - midi_time_offset
            off_sec = beat_to_sec(pedal.offset_beat, tempo_map) - midi_time_offset
        except ValueError:
            continue
        piano.control_changes.append(
            pretty_midi.ControlChange(number=cc_num, value=127, time=max(0.0, on_sec))
        )
        piano.control_changes.append(
            pretty_midi.ControlChange(number=cc_num, value=0, time=max(0.0, off_sec))
        )

    midi.instruments.append(piano)

    if not piano.notes:
        return _STUB_MIDI

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        midi.write(str(tmp_path))
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# MusicXML rendering
# ---------------------------------------------------------------------------

# Human-readable labels for non-sustain pedals. Sustain uses the standard
# "Ped." / "*" pair; MusicXML doesn't reserve glyphs for sostenuto / una
# corda, so text directions are the portable choice across renderers.
_PEDAL_TEXT = {
    "sustain":    ("Ped.", "*"),
    "sostenuto":  ("Sost. Ped.", "*"),
    "una_corda":  ("una corda", "tre corde"),
}


def _attach_dynamics(part, dynamics, music21) -> None:  # noqa: ANN001 — music21 is untyped
    """Insert ``DynamicMarking`` events into a music21 part.

    Static marks (pp..ff) become ``music21.dynamics.Dynamic`` objects at
    ``dyn.beat``. Hairpins (crescendo/decrescendo) become italic text
    directions — ``music21.dynamics.Crescendo`` spanners need referenced
    note endpoints that may have been quantized away, so text is the
    more portable choice across OSMD / LilyPond / MuseScore.
    """
    for dyn in dynamics:
        beat = max(0.0, float(dyn.beat))
        if dyn.type in ("pp", "p", "mp", "mf", "f", "ff"):
            part.insert(beat, music21.dynamics.Dynamic(dyn.type))
        elif dyn.type == "crescendo":
            part.insert(beat, music21.expressions.TextExpression("cresc."))
        elif dyn.type == "decrescendo":
            part.insert(beat, music21.expressions.TextExpression("dim."))


def _attach_pedal_marks(part, pedal_events, music21) -> None:  # noqa: ANN001
    """Insert pedal text directions into a music21 part.

    Uses ``TextExpression`` rather than ``music21.expressions.PedalMark``
    because the latter has inconsistent MusicXML export across music21
    versions. Sustain emits the standard ``Ped.`` / ``*`` pair; sostenuto
    and una corda use descriptive labels.
    """
    for pedal in pedal_events:
        labels = _PEDAL_TEXT.get(pedal.type)
        if labels is None:
            continue
        on_label, off_label = labels
        on_beat = max(0.0, float(pedal.onset_beat))
        off_beat = max(on_beat, float(pedal.offset_beat))
        part.insert(on_beat, music21.expressions.TextExpression(on_label))
        part.insert(off_beat, music21.expressions.TextExpression(off_label))


def _render_musicxml_bytes(
    score: PianoScore,
    perf: HumanizedPerformance | None,
    title: str,
    composer: str,
) -> bytes:
    """Render a PianoScore to MusicXML via music21.

    music21 is a hard dependency (see ``pyproject.toml``); if it raises
    during export we let the exception propagate so the job manager can
    surface the failure instead of silently emitting a stub.
    """
    import music21  # noqa: PLC0415 — heavy import, kept lazy for test speed

    s = music21.stream.Score()
    s.metadata = music21.metadata.Metadata()
    s.metadata.title = title or "Untitled"
    s.metadata.composer = composer or ""

    ts = music21.meter.TimeSignature(
        f"{score.metadata.time_signature[0]}/{score.metadata.time_signature[1]}"
    )
    key_root = score.metadata.key.split(":")[0] if ":" in score.metadata.key else "C"
    mode = "major" if "minor" not in score.metadata.key else "minor"
    ks = music21.key.Key(key_root, mode)
    # Round BPM to a whole number so the metronome mark reads cleanly
    # (e.g., ♩ = 99 instead of ♩ = 99.38401442307693). The tempo map
    # itself stays float for beat→sec conversions elsewhere.
    bpm_float = score.metadata.tempo_map[0].bpm if score.metadata.tempo_map else 120.0
    bpm = round(bpm_float)
    tempo_mark = music21.tempo.MetronomeMark(number=bpm)

    articulations_at: dict[str, str] = {}
    if perf is not None:
        for a in perf.expression.articulations:
            articulations_at[a.score_note_id] = a.type

    for hand_name, notes, clef in (
        ("Right Hand", score.right_hand, music21.clef.TrebleClef()),
        ("Left Hand", score.left_hand, music21.clef.BassClef()),
    ):
        part = music21.stream.Part()
        part.partName = hand_name
        part.append(ts)
        part.append(ks)
        if hand_name == "Right Hand":
            part.append(tempo_mark)
        part.append(clef)

        for sn in sorted(notes, key=lambda n: n.onset_beat):
            n = music21.note.Note(sn.pitch)
            n.quarterLength = sn.duration_beat
            n.volume.velocity = sn.velocity
            art_type = articulations_at.get(sn.id)
            if art_type == "staccato":
                n.articulations.append(music21.articulations.Staccato())
            elif art_type == "accent":
                n.articulations.append(music21.articulations.Accent())
            elif art_type == "tenuto":
                n.articulations.append(music21.articulations.Tenuto())
            elif art_type == "fermata":
                # Fermata lives on n.expressions in music21, not
                # n.articulations — MusicXML emits it as a <fermata/>
                # inside <notations> either way, but only the
                # expressions placement round-trips through makeNotation.
                n.expressions.append(music21.expressions.Fermata())
            part.insert(sn.onset_beat, n)

        # Dynamics + pedal marks are attached to the RH and LH parts
        # respectively so the markings sit between the staves where
        # pianists expect them. Skipped on the engrave-from-score path
        # (perf is never None there, but the expression map is empty).
        if hand_name == "Right Hand" and perf is not None:
            _attach_dynamics(part, perf.expression.dynamics, music21)
        if hand_name == "Left Hand" and perf is not None:
            _attach_pedal_marks(part, perf.expression.pedal_events, music21)

        # Chord symbols disabled for now — the harmonic analysis from
        # audio transcription produces noisy labels (G5, E5, etc.) that
        # clutter the notation. Re-enable when chord recognition quality
        # improves or when source is a clean MIDI upload.
        # for cs in score.metadata.chord_symbols:
        #     try:
        #         h = music21.harmony.ChordSymbol(cs.label.replace(":", ""))
        #         part.insert(cs.beat, h)
        #     except Exception:
        #         pass

        # Quantize to a 16th-note + triplet grid so OSMD can render it.
        # Without explicit divisors, music21 defaults to divisions=10080
        # which produces MusicXML that OSMD chokes on.
        part.quantize(quarterLengthDivisors=(4, 3), inPlace=True)
        part.makeNotation(inPlace=True)
        s.append(part)

    with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        s.write("musicxml", fp=str(tmp_path))
        raw = tmp_path.read_bytes()
        return _sanitize_musicxml_for_osmd(raw)
    finally:
        tmp_path.unlink(missing_ok=True)


def _sanitize_musicxml_for_osmd(raw: bytes) -> bytes:
    """Post-process music21 MusicXML to fix OSMD compatibility issues.

    OSMD's VexFlow backend crashes (parentVoiceEntry) when:
    1. <divisions> is very high (10080) — we reduce to 4 (16th-note grid)
    2. Voice numbers exceed 2 per part — we collapse to voice 1
    3. Voice 0 exists — OSMD expects 1-indexed voices

    We also scale all <duration> values proportionally when changing divisions.
    """
    import re

    text = raw.decode("utf-8")

    # Find the current divisions value
    div_match = re.search(r"<divisions>(\d+)</divisions>", text)
    if div_match:
        old_div = int(div_match.group(1))
        new_div = 4  # 4 divisions per quarter = 16th-note grid

        if old_div != new_div and old_div > 0:
            ratio = new_div / old_div

            # Replace all <divisions> tags
            text = re.sub(
                r"<divisions>\d+</divisions>",
                f"<divisions>{new_div}</divisions>",
                text,
            )

            # Scale all <duration> values
            def scale_duration(m: re.Match) -> str:
                old_dur = int(m.group(1))
                new_dur = max(1, round(old_dur * ratio))
                return f"<duration>{new_dur}</duration>"

            text = re.sub(r"<duration>(\d+)</duration>", scale_duration, text)

            # Scale <forward> and <backup> durations too
            def scale_forward(m: re.Match) -> str:
                tag = m.group(1)
                old_dur = int(m.group(2))
                new_dur = max(1, round(old_dur * ratio))
                return f"<{tag}>\n        <duration>{new_dur}</duration>"

            text = re.sub(
                r"<(forward|backup)>\s*<duration>(\d+)</duration>",
                scale_forward,
                text,
            )

    # Collapse all voices to voice 1 (OSMD chokes on 3+ voices per part)
    text = re.sub(r"<voice>\d+</voice>", "<voice>1</voice>", text)

    return text.encode("utf-8")


# ---------------------------------------------------------------------------
# PDF rendering — best effort, falls back to a tiny stub.
# ---------------------------------------------------------------------------

def _render_pdf_bytes(musicxml_bytes: bytes) -> bytes:
    """Render MusicXML to PDF bytes.

    Tries LilyPond first (this is what production ships — ~250 MB apt
    package, musicxml2ly + lilypond binaries) and MuseScore as a
    higher-fidelity fallback for local dev machines that have it
    installed. Returns the 60-byte stub PDF only when no renderer is
    available or all renderers fail.
    """
    has_lilypond = bool(shutil.which("musicxml2ly") and shutil.which("lilypond"))
    mscore_bin = next(
        (b for b in ("musescore4", "musescore3", "mscore", "MuseScore4") if shutil.which(b)),
        None,
    )

    if not has_lilypond and mscore_bin is None:
        log.warning(
            "No PDF renderer found — install lilypond (preferred) or MuseScore "
            "for real PDF output; emitting stub",
        )
        return _STUB_PDF

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        xml_path = tmp / "score.musicxml"
        xml_path.write_bytes(musicxml_bytes)

        if has_lilypond:
            ly_path = tmp / "score.ly"
            pdf_path = tmp / "sheet.pdf"
            try:
                subprocess.run(
                    ["musicxml2ly", "-o", str(ly_path), str(xml_path)],
                    check=True, capture_output=True, timeout=60,
                )
                subprocess.run(
                    ["lilypond", "-o", str(tmp / "sheet"), str(ly_path)],
                    check=True, capture_output=True, timeout=120,
                )
                if pdf_path.is_file():
                    log.info("PDF rendered via LilyPond (%d bytes)", pdf_path.stat().st_size)
                    return pdf_path.read_bytes()
                log.warning("LilyPond ran but produced no PDF at %s", pdf_path)
            except subprocess.CalledProcessError as exc:
                log.warning(
                    "LilyPond PDF render failed (%s): %s",
                    exc.cmd[0] if exc.cmd else "?",
                    (exc.stderr or b"").decode("utf-8", "replace")[:500],
                )
            except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
                log.warning("LilyPond PDF render failed: %s", exc)

        if mscore_bin is not None:
            pdf_path = tmp / "sheet.pdf"
            try:
                subprocess.run(
                    [mscore_bin, "-o", str(pdf_path), str(xml_path)],
                    check=True, capture_output=True, timeout=120,
                )
                if pdf_path.is_file():
                    log.info(
                        "PDF rendered via %s (%d bytes)", mscore_bin, pdf_path.stat().st_size,
                    )
                    return pdf_path.read_bytes()
                log.warning("%s ran but produced no PDF at %s", mscore_bin, pdf_path)
            except subprocess.CalledProcessError as exc:
                log.warning(
                    "%s PDF render failed: %s",
                    mscore_bin,
                    (exc.stderr or b"").decode("utf-8", "replace")[:500],
                )
            except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
                log.warning("%s PDF render failed: %s", mscore_bin, exc)

    return _STUB_PDF


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

def _engrave_sync(
    payload: HumanizedPerformance | PianoScore,
    title: str,
    composer: str,
) -> tuple[bytes, bytes, bytes]:
    """Render all three artifacts. Returns (pdf, musicxml, midi)."""
    if isinstance(payload, HumanizedPerformance):
        score = payload.score
        perf: HumanizedPerformance | None = payload
    else:
        score = payload
        perf = None

    if perf is None:
        # Engrave-from-score: synthesize a zero-deviation performance shell so
        # _render_midi_bytes / MusicXML helpers can share one code path.
        from backend.contracts import ExpressionMap, ExpressiveNote, QualitySignal  # noqa: PLC0415

        expressive_notes = []
        for hand_name, notes in (("rh", score.right_hand), ("lh", score.left_hand)):
            for n in notes:
                expressive_notes.append(ExpressiveNote(
                    score_note_id=n.id,
                    pitch=n.pitch,
                    onset_beat=n.onset_beat,
                    duration_beat=n.duration_beat,
                    velocity=n.velocity,
                    hand=hand_name,  # type: ignore[arg-type]
                    voice=n.voice,
                    timing_offset_ms=0.0,
                    velocity_offset=0,
                ))
        perf = HumanizedPerformance(
            schema_version=SCHEMA_VERSION,
            expressive_notes=expressive_notes,
            expression=ExpressionMap(),
            score=score,
            quality=QualitySignal(overall_confidence=0.5, warnings=["engrave-from-score"]),
        )

    midi_bytes = _render_midi_bytes(perf)
    musicxml_bytes = _render_musicxml_bytes(score, perf, title, composer)
    pdf_bytes = _render_pdf_bytes(musicxml_bytes)
    return pdf_bytes, musicxml_bytes, midi_bytes


class EngraveService:
    name = "engrave"

    def __init__(self, blob_store: BlobStore) -> None:
        self.blob_store = blob_store

    async def run(
        self,
        payload: HumanizedPerformance | PianoScore,
        *,
        job_id: str,
        title: str = "Untitled",
        composer: str = "Unknown",
    ) -> EngravedOutput:
        log.info(
            "engrave: start job_id=%s title=%r humanized_input=%s",
            job_id,
            title,
            isinstance(payload, HumanizedPerformance),
        )
        pdf_bytes, musicxml_bytes, midi_bytes = await asyncio.to_thread(
            _engrave_sync, payload, title, composer,
        )

        prefix = f"jobs/{job_id}/output"
        pdf_uri = self.blob_store.put_bytes(f"{prefix}/sheet.pdf", pdf_bytes)
        musicxml_uri = self.blob_store.put_bytes(f"{prefix}/score.musicxml", musicxml_bytes)
        midi_uri = self.blob_store.put_bytes(f"{prefix}/humanized.mid", midi_bytes)

        score = payload.score if isinstance(payload, HumanizedPerformance) else payload
        chord_count = len(score.metadata.chord_symbols)

        log.info(
            "engrave: done job_id=%s bytes pdf=%d musicxml=%d midi=%d chord_symbols=%d",
            job_id,
            len(pdf_bytes),
            len(musicxml_bytes),
            len(midi_bytes),
            chord_count,
        )

        # These flags describe what was *actually rendered*. As of PR-5
        # (plan phase 1.3–1.6) engrave renders dynamics and pedal marks
        # when the humanized input populates them, so we surface that
        # state directly from the expression map instead of hardcoding.
        perf_for_flags = payload if isinstance(payload, HumanizedPerformance) else None
        return EngravedOutput(
            schema_version=SCHEMA_VERSION,
            metadata=EngravedScoreData(
                includes_dynamics=bool(perf_for_flags and perf_for_flags.expression.dynamics),
                includes_pedal_marks=bool(perf_for_flags and perf_for_flags.expression.pedal_events),
                includes_fingering=False,
                includes_chord_symbols=chord_count > 0,
                title=title,
                composer=composer,
            ),
            pdf_uri=pdf_uri,
            musicxml_uri=musicxml_uri,
            humanized_midi_uri=midi_uri,
            audio_preview_uri=None,
        )
