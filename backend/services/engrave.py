"""Engraving stage — render MIDI / MusicXML / PDF artifacts.

Tries the real renderers when their optional deps are present and falls
back to small but valid stub bytes otherwise so the rest of the pipeline
keeps working in environments without pretty_midi / music21 / LilyPond.

  * MIDI     — pretty_midi if installed, else a minimal MThd+MTrk file
  * MusicXML — music21 if installed, else a minimal score-partwise body
  * PDF      — MuseScore / LilyPond CLI if on $PATH, else a 1-line %PDF stub
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

from backend.contracts import (
    SCHEMA_VERSION,
    EngravedOutput,
    EngravedScoreData,
    HumanizedPerformance,
    PianoScore,
    ScoreNote,
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
_STUB_MUSICXML = (
    b'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
    b'<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" '
    b'"http://www.musicxml.org/dtds/partwise.dtd">\n'
    b'<score-partwise version="3.1"/>\n'
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

    for pedal in perf.expression.pedal_events:
        if pedal.type != "sustain":
            continue
        try:
            on_sec = beat_to_sec(pedal.onset_beat, tempo_map) - midi_time_offset
            off_sec = beat_to_sec(pedal.offset_beat, tempo_map) - midi_time_offset
        except ValueError:
            continue
        piano.control_changes.append(
            pretty_midi.ControlChange(number=64, value=127, time=max(0.0, on_sec))
        )
        piano.control_changes.append(
            pretty_midi.ControlChange(number=64, value=0, time=max(0.0, off_sec))
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

def _render_musicxml_bytes(
    score: PianoScore,
    perf: HumanizedPerformance | None,
    title: str,
    composer: str,
) -> bytes:
    """Render a PianoScore to MusicXML.

    Tries music21 first; on failure (or if music21 isn't installed) falls
    back to a hand-rolled minimal score-partwise body that at least lists
    the notes for each hand.
    """
    try:
        import music21  # noqa: PLC0415 — optional heavy dep
    except ImportError:
        log.warning("music21 not installed — MusicXML output will be a minimal stub. Install with: pip install music21")
        return _minimal_musicxml(score, title, composer)

    try:
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
                part.insert(sn.onset_beat, n)

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
    except Exception as exc:  # noqa: BLE001 — music21 has many failure modes
        log.warning("music21 MusicXML render failed (%s); falling back to minimal", exc)
        return _minimal_musicxml(score, title, composer)


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


def _minimal_musicxml(score: PianoScore, title: str, composer: str) -> bytes:
    """Hand-rolled minimal score-partwise body.

    Not bar-aware — emits each note as a single ``<note>`` element with
    its pitch step, octave, and duration in tenths of a beat. Good enough
    for round-trip tests; the music21 path is what real consumers want.
    """
    bpm = score.metadata.tempo_map[0].bpm if score.metadata.tempo_map else 120.0
    divisions = 4  # 4 divisions per quarter note → 16th-note grid

    def note_xml(sn: ScoreNote, hand: int) -> str:
        step, alter, octave = _midi_to_step_alter_octave(sn.pitch)
        duration = max(1, int(round(sn.duration_beat * divisions)))
        alter_xml = f"<alter>{alter}</alter>" if alter else ""
        return (
            "<note>"
            f"<pitch><step>{step}</step>{alter_xml}<octave>{octave}</octave></pitch>"
            f"<duration>{duration}</duration>"
            f"<voice>{sn.voice}</voice>"
            f"<staff>{hand}</staff>"
            "</note>"
        )

    rh_notes = "".join(note_xml(n, 1) for n in sorted(score.right_hand, key=lambda n: n.onset_beat))
    lh_notes = "".join(note_xml(n, 2) for n in sorted(score.left_hand, key=lambda n: n.onset_beat))

    ts = score.metadata.time_signature
    body = (
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" '
        '"http://www.musicxml.org/dtds/partwise.dtd">\n'
        '<score-partwise version="3.1">'
        '<work>'
        f'<work-title>{escape(title or "Untitled")}</work-title>'
        '</work>'
        '<identification>'
        f'<creator type="composer">{escape(composer or "Unknown")}</creator>'
        '</identification>'
        '<part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>'
        '<part id="P1">'
        '<measure number="1">'
        '<attributes>'
        f'<divisions>{divisions}</divisions>'
        f'<key><fifths>0</fifths></key>'
        f'<time><beats>{ts[0]}</beats><beat-type>{ts[1]}</beat-type></time>'
        '<staves>2</staves>'
        '<clef number="1"><sign>G</sign><line>2</line></clef>'
        '<clef number="2"><sign>F</sign><line>4</line></clef>'
        '</attributes>'
        f'<sound tempo="{bpm:.2f}"/>'
        f'{rh_notes}'
        '<backup>'
        f'<duration>{max(1, int(round(sum(n.duration_beat for n in score.right_hand) * divisions)))}</duration>'
        '</backup>'
        f'{lh_notes}'
        '</measure>'
        '</part>'
        '</score-partwise>\n'
    )
    return body.encode("utf-8")


_PITCH_NAMES: list[tuple[str, int]] = [
    ("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0),
    ("F", 1), ("G", 0), ("G", 1), ("A", 0), ("A", 1), ("B", 0),
]


def _midi_to_step_alter_octave(midi: int) -> tuple[str, int, int]:
    midi = max(0, min(127, midi))
    octave = (midi // 12) - 1
    step, alter = _PITCH_NAMES[midi % 12]
    return step, alter, octave


# ---------------------------------------------------------------------------
# PDF rendering — best effort, falls back to a tiny stub.
# ---------------------------------------------------------------------------

def _render_pdf_bytes(musicxml_bytes: bytes) -> bytes:
    """Try MuseScore CLI then LilyPond. Returns the stub PDF on failure."""
    if not (shutil.which("musescore4") or shutil.which("musescore3")
            or shutil.which("mscore") or shutil.which("MuseScore4")
            or (shutil.which("musicxml2ly") and shutil.which("lilypond"))):
        log.warning("No PDF renderer found — install LilyPond or MuseScore for real PDF output")
        return _STUB_PDF

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        xml_path = tmp / "score.musicxml"
        pdf_path = tmp / "sheet.pdf"
        xml_path.write_bytes(musicxml_bytes)

        for mscore in ("musescore4", "musescore3", "mscore", "MuseScore4"):
            if not shutil.which(mscore):
                continue
            try:
                subprocess.run(
                    [mscore, "-o", str(pdf_path), str(xml_path)],
                    check=True, capture_output=True, timeout=120,
                )
                if pdf_path.is_file():
                    return pdf_path.read_bytes()
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                continue

        if shutil.which("musicxml2ly") and shutil.which("lilypond"):
            try:
                ly_path = tmp / "score.ly"
                subprocess.run(
                    ["musicxml2ly", "-o", str(ly_path), str(xml_path)],
                    check=True, capture_output=True, timeout=60,
                )
                subprocess.run(
                    ["lilypond", "-o", str(tmp / "sheet"), str(ly_path)],
                    check=True, capture_output=True, timeout=120,
                )
                if pdf_path.is_file():
                    return pdf_path.read_bytes()
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                pass

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

        # NOTE: these flags describe what was *actually rendered*, not what
        # the input contained. Dynamics and pedal marks are still
        # unimplemented in the MusicXML render path (plan phase 1.3/1.4),
        # so both stay False until those PRs land — even when the humanized
        # input has dynamics or pedal events populated.
        return EngravedOutput(
            schema_version=SCHEMA_VERSION,
            metadata=EngravedScoreData(
                includes_dynamics=False,
                includes_pedal_marks=False,
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
