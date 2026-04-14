"""Engraving stage — render MIDI / MusicXML / PDF artifacts.

  * MIDI     — pretty_midi if installed, else a minimal MThd+MTrk file
  * MusicXML — music21 (hard dependency; errors propagate to the caller)
  * PDF      — LilyPond (preferred) or MuseScore CLI ($PATH, ``MUSESCORE_PATH``, or macOS
    ``.app`` bundle), else a 1-line %PDF stub
"""
from __future__ import annotations

import asyncio
import logging
import re
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

from shared.musescore_cli import musescore_executable_paths

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


# Chord symbols below this transcriber confidence are dropped — the
# audio-harmony pipeline routinely emits low-confidence labels that look
# like pitch-octave pairs ("G5", "E5") rather than chord qualities, and
# they clutter the notation more than they help.
_CHORD_MIN_CONFIDENCE = 0.5

# Anything that lexically looks like a pitch-and-octave (e.g. "G5",
# "C#4", "Db3") is an artifact of the transcription pipeline treating
# a single pitch as a chord. Filter these out before handing to music21
# so ``ChordSymbol`` doesn't build a one-note "chord".
_PITCH_OCTAVE_RE = re.compile(r"^[A-G][#b]?\d+$")


# Minimum Krumhansl-Schmuckler correlation coefficient for an override
# to fire. Empirically the KS analyzer reports ~0.90+ on clean tonal
# material (e.g. two_hand_chordal at 0.965), ~0.78 on a bare diatonic
# scale, and drops below 0.4 on highly chromatic content (jazz_voicings
# at 0.35). 0.80 is the floor for a clearly-tonal excerpt — above this,
# disagreements with the declared key almost always indicate upstream
# mis-labeling rather than a confused analyzer. Below this we trust
# the declared key and skip the override.
_KEY_OVERRIDE_MIN_CORRELATION = 0.80


# Export divisions per quarter — matches the ``divisionsPerQuarter=12``
# we force on the music21 exporter below. LCM(3, 4) covers 16ths
# (3/12 qL) and 8th-note triplets (4/12 qL), the finest grids arrange
# emits. Every note onset and duration is snapped to this grid before
# handoff so float drift from decimal-approximated grids (e.g. arrange's
# ``0.167`` for 1/6) can't leave a sub-1/12 residual after ``makeNotation``
# splits notes at barlines. Such residuals map to music21 types like
# ``2048th`` / ``inexpressible`` that the MusicXML exporter rejects.
# ``Fraction`` is essential here — rounding a float to 1/12 still yields
# an inexact binary (e.g. ``23/6`` → ``3.8333333333333335``), and the drift
# is exactly what reopens the barline-split bug. music21 consumes Fractions
# as exact rationals.
_EXPORT_DIVISIONS = 12
_MIN_SNAPPED_QL = Fraction(1, _EXPORT_DIVISIONS)


def _snap_quarter(qL: float) -> Fraction:
    """Snap a quarterLength to the export grid as an exact ``Fraction``."""
    return Fraction(round(qL * _EXPORT_DIVISIONS), _EXPORT_DIVISIONS)


def _resolve_key_signature(score: PianoScore, music21) -> tuple[str, str, bool]:  # noqa: ANN001
    """Resolve the effective key signature using Krumhansl-Schmuckler.

    Audio transcription / arrange can label a piece with the wrong key
    — e.g. tagging an F# minor piece as "C:major" — and every accidental
    downstream explodes because music21 emits sharps/flats for every
    non-diatonic note. KS is the canonical pitch-histogram tonic finder
    and ships with music21; five lines of analyzer gate + one override
    fix the bug before makeNotation ever sees it.

    Returns ``(root_name, mode, overridden)``. The override fires only
    when **both** conditions hold:

    1. Analyzer correlation ≥ ``_KEY_OVERRIDE_MIN_CORRELATION``. Low
       correlations mean the pitch histogram is ambiguous (short
       excerpts, highly chromatic jazz) — trust the upstream label.
    2. The analyzer's (tonic, mode) disagrees with the declared key.
       If KS agrees, the declared key is already correct and we skip
       the override to keep logs quiet.

    Any exception from the analyzer (empty stream, pitch-less notes,
    library internals) falls back to the declared key so this never
    blocks engrave.
    """
    declared = score.metadata.key or "C:major"
    declared_root = declared.split(":")[0] if ":" in declared else "C"
    declared_mode = "minor" if "minor" in declared else "major"

    # Build a throwaway Stream from every note so the pitch histogram
    # is the whole piece, not one hand or one measure. Velocity /
    # duration / voice are irrelevant for KS — it only reads pitch
    # classes.
    all_notes = list(score.right_hand) + list(score.left_hand)
    if not all_notes:
        return declared_root, declared_mode, False

    try:
        # Pin every note to quarterLength=1 so the KS pitch-class histogram
        # is duration-independent. Raw score durations can over-weight
        # sustained LH pedal tones (a 4-beat F#2 gets 8× the histogram
        # mass of a 0.5-beat RH note), which looks to KS like a single-
        # pitch stream rather than a real key. Note-count weighting is
        # what we actually want — the question is "what scale are these
        # notes drawn from", not "which note is longest".
        analysis_stream = music21.stream.Stream()
        for sn in all_notes:
            n = music21.note.Note(sn.pitch)
            n.quarterLength = 1.0
            analysis_stream.append(n)
        analyzer = music21.analysis.discrete.KrumhanslSchmuckler()
        result = analyzer.process(analysis_stream)
    except Exception as exc:  # noqa: BLE001 — analyzer errors must not crash engrave
        log.debug("key-signature analyzer failed; trusting declared %r: %s", declared, exc)
        return declared_root, declared_mode, False

    try:
        tonic_pitch, analyzer_mode, correlation = result[0]
    except (TypeError, IndexError, ValueError) as exc:
        log.debug("unexpected KS result shape %r: %s", result, exc)
        return declared_root, declared_mode, False

    analyzer_root = tonic_pitch.name
    if correlation < _KEY_OVERRIDE_MIN_CORRELATION:
        return declared_root, declared_mode, False
    if analyzer_root == declared_root and analyzer_mode == declared_mode:
        return declared_root, declared_mode, False

    log.info(
        "engrave: KS key override — declared=%s:%s analyzer=%s:%s correlation=%.3f",
        declared_root, declared_mode, analyzer_root, analyzer_mode, correlation,
    )
    return analyzer_root, analyzer_mode, True


def _attach_chord_symbols(part, chord_symbols, music21) -> int:  # noqa: ANN001
    """Insert chord symbols onto a music21 part, returning the count rendered.

    Chord symbols from audio transcription are noisy — single-pitch
    false positives, low-confidence guesses, and Harte labels music21
    can't parse all end up here. Applies three filters before committing:

    1. **Confidence gate.** Drop anything below
       ``_CHORD_MIN_CONFIDENCE``. The transcriber already reports its
       own uncertainty; trust it.
    2. **Shape gate.** Reject labels that match ``[A-G][#b]?\\d+`` —
       those are pitch-octave pairs (e.g. "G5"), not chord qualities.
    3. **Parse gate.** ``music21.harmony.ChordSymbol`` must parse the
       label cleanly AND the resulting chord must carry ≥ 3 distinct
       pitches. Everything else is a one-note or two-note artifact
       that clutters the staff.

    music21's ChordSymbol parser uses a custom pitch-name grammar that
    doesn't accept Harte's colon separator ("C:maj7"), so we strip the
    colon before parsing. Any parser error lands in a broad except —
    ChordSymbol raises a grab-bag of exception types and the cost of a
    missed symbol is much lower than the cost of an exploding engrave.
    """
    rendered = 0
    for cs in chord_symbols:
        if cs.confidence < _CHORD_MIN_CONFIDENCE:
            continue
        label = cs.label.strip()
        if not label or _PITCH_OCTAVE_RE.match(label):
            continue
        # music21 doesn't accept Harte's "root:quality" colon form.
        parse_label = label.replace(":", "")
        try:
            h = music21.harmony.ChordSymbol(parse_label)
        except Exception:  # noqa: BLE001 — ChordSymbol raises many types
            continue
        try:
            pitches = h.pitches
        except Exception:  # noqa: BLE001 — defensive, some labels parse but can't resolve
            continue
        if len({p.midi % 12 for p in pitches}) < 3:
            continue
        try:
            part.insert(max(0.0, float(cs.beat)), h)
        except Exception:  # noqa: BLE001 — insertion shouldn't fail, but don't crash engrave
            continue
        rendered += 1
    return rendered


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


def _approximate_ql_for_musicxml_export(ql: float, music21) -> float:  # noqa: ANN001
    """Pick a ``quarterLength`` near ``ql`` that music21 can encode to MusicXML.

    Transcription grids often produce floats that are not exact rationals music21
    can express as a single ``Duration`` (``type == 'inexpressible'``). Try
    ``Fraction(...).limit_denominator`` at increasing denominators and keep the
    closest candidate that passes the exporter's type rules.
    """
    from fractions import Fraction

    dur_mod = music21.duration
    min_safe = float(dur_mod.convertTypeToQuarterLength("1024th"))
    if ql <= 0:
        return min_safe

    best: float | None = None
    best_err = float("inf")
    for lim in (
        32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048,
        4096, 8192, 16384, 32768, 65536, 131072, 262144,
    ):
        cand = float(Fraction(ql).limit_denominator(lim))
        if cand <= 0:
            continue
        d = dur_mod.Duration(cand)
        if _duration_needs_musicxml_coercion(d):
            continue
        err = abs(cand - ql)
        if err < best_err:
            best_err = err
            best = cand

    if best is not None:
        return best

    # Last resort: snap to 1/384 quarter (finer than our divisions=12 grid).
    step = 1.0 / 384.0
    k = max(1, round(ql / step))
    return k * step


def _duration_needs_musicxml_coercion(d) -> bool:  # noqa: ANN001 — music21 Duration
    """True if music21 would raise when exporting this duration to MusicXML.

    ``makeNotation`` can attach tuplets whose ``durationNormal`` is a
    ``2048th`` — valid internally but rejected by ``typeToMusicXMLType``
    (see music21 ``m21ToXml``). We also treat any other duration type the
    exporter refuses the same way so future music21 versions stay covered.
    """
    from music21.musicxml.m21ToXml import typeToMusicXMLType  # noqa: PLC0415
    from music21.musicxml.xmlObjects import MusicXMLExportException  # noqa: PLC0415

    try:
        typeToMusicXMLType(d.type)
    except MusicXMLExportException:
        return True
    for tup in d.tuplets:
        dn = tup.durationNormal
        if dn is not None:
            try:
                typeToMusicXMLType(dn.type)
            except MusicXMLExportException:
                return True
        da = tup.durationActual
        if da is not None:
            try:
                typeToMusicXMLType(da.type)
            except MusicXMLExportException:
                return True
    return False


def _coerce_durations_for_musicxml_export(part, music21) -> None:  # noqa: ANN001
    """Normalize note/rest durations so MusicXML export cannot fail on types
    like ``2048th`` tuplet brackets.

    First pass rebuilds each offending duration from its ``quarterLength``
    alone (music21 usually picks sane tuplets). If types are still rejected,
    very short lengths are clamped to a ``1024th``. If the duration is still
    bad (common for float noise → ``inexpressible``), snap ``quarterLength``
    to a nearby rational via ``_approximate_ql_for_musicxml_export``.
    """
    dur_mod = music21.duration
    min_safe_ql = float(dur_mod.convertTypeToQuarterLength("1024th"))

    for elem in part.recurse().notesAndRests:
        d = elem.duration
        if not _duration_needs_musicxml_coercion(d):
            continue
        ql_orig = float(d.quarterLength)
        if ql_orig <= 0:
            continue
        ql = ql_orig
        elem.duration = dur_mod.Duration(ql)
        if not _duration_needs_musicxml_coercion(elem.duration):
            continue
        if ql < min_safe_ql:
            ql = min_safe_ql
            elem.duration = dur_mod.Duration(ql)
            if not _duration_needs_musicxml_coercion(elem.duration):
                log.warning(
                    "engrave: raised sub–1024th duration ql=%s to %s for MusicXML",
                    ql_orig,
                    ql,
                )
                continue
        approx = _approximate_ql_for_musicxml_export(ql, music21)
        elem.duration = dur_mod.Duration(approx)
        if not _duration_needs_musicxml_coercion(elem.duration):
            if approx != ql_orig:
                log.warning(
                    "engrave: snapped unexportable duration ql=%s -> %s for MusicXML",
                    ql_orig,
                    approx,
                )
            continue
        # Should be unreachable; keep export from crashing.
        elem.duration = dur_mod.Duration(min_safe_ql)
        log.warning(
            "engrave: fell back to minimum duration ql=%s -> %s for MusicXML",
            ql_orig,
            min_safe_ql,
        )
def _resolve_same_pitch_overlaps_per_voice(notes: list[ScoreNote]) -> list[ScoreNote]:
    """Truncate an earlier same-(pitch, voice) note when a later one
    attacks before it ends.

    Mirrors the overlap resolver in ``_render_midi_bytes`` for the
    MusicXML path. When two notes of the same pitch in the same voice
    overlap AND the earlier one crosses a bar line, music21's
    ``makeTies`` can't reliably split the bar-crossing portion inside
    a ``Voice`` sub-stream — the note lands verbatim in its starting
    measure and pushes the total voice duration past the time
    signature's budget. MuseScore flags that overflow as "corrupted
    score".

    Fixing it upstream (shorten the earlier note to end at the next
    attack) keeps music21 seeing a clean, sequential voice line.
    Preserves a 0.01-beat floor so no note collapses to zero.
    """
    by_key: dict[tuple[int, int], list[int]] = {}
    for i, n in enumerate(notes):
        by_key.setdefault((n.pitch, n.voice), []).append(i)
    new_durations: dict[int, float] = {}
    for idxs in by_key.values():
        if len(idxs) < 2:
            continue
        idxs.sort(key=lambda i: notes[i].onset_beat)
        for j in range(len(idxs) - 1):
            prev = notes[idxs[j]]
            nxt = notes[idxs[j + 1]]
            if prev.onset_beat + prev.duration_beat > nxt.onset_beat:
                new_durations[idxs[j]] = max(0.01, nxt.onset_beat - prev.onset_beat)
    if not new_durations:
        return notes
    return [
        n.model_copy(update={"duration_beat": new_durations[i]}) if i in new_durations else n
        for i, n in enumerate(notes)
    ]


def _split_bar_crossing_notes(
    notes: list[ScoreNote],
    beats_per_measure: float,
) -> list[tuple[ScoreNote, bool, bool]]:
    """Split notes that cross bar lines into consecutive in-measure pieces.

    Returns a list of ``(piece, tie_in, tie_out)`` tuples — ``tie_in``
    means the piece is a continuation of an earlier piece, ``tie_out``
    means the piece will be continued by a later piece. The piece is a
    ScoreNote copy whose ``onset_beat``/``duration_beat`` lie entirely
    within one measure.

    Why pre-split: music21's ``makeTies`` is the alternative, but when
    the score has 2 voices per hand with sparse voice-2 content,
    music21 assigns a bar-crossing note's continuation a fresh voice
    number in the next measure. The OSMD sanitizer then clamps that
    fresh number back to voice 2, breaking the tie chain (start on
    voice 1, stop on voice 2). MuseScore 4 flags the dangling ties as
    a corrupt score.

    Pre-splitting keeps each piece entirely within one measure so
    music21 never renumbers during a split — voices stay stable and
    the tie marks we set here round-trip cleanly through export.
    """
    if beats_per_measure <= 0:
        return [(n, False, False) for n in notes]
    out: list[tuple[ScoreNote, bool, bool]] = []
    for n in notes:
        remaining = n.duration_beat
        cur_onset = n.onset_beat
        prev_piece_tied_out = False
        while remaining > 1e-6:
            bar_index = int(cur_onset / beats_per_measure + 1e-9)
            bar_end = (bar_index + 1) * beats_per_measure
            piece_dur = min(remaining, bar_end - cur_onset)
            if piece_dur < 1e-6:
                break
            tie_out = remaining - piece_dur > 1e-6
            piece = n.model_copy(update={
                "onset_beat": cur_onset,
                "duration_beat": piece_dur,
            })
            out.append((piece, prev_piece_tied_out, tie_out))
            prev_piece_tied_out = tie_out
            cur_onset += piece_dur
            remaining -= piece_dur
    return out


def _render_musicxml_bytes(
    score: PianoScore,
    perf: HumanizedPerformance | None,
    title: str,
    composer: str,
) -> tuple[bytes, int]:
    """Render a PianoScore to MusicXML via music21.

    Returns a ``(bytes, chord_symbols_rendered)`` tuple so the caller
    can report how many chord symbols actually survived the filter gate
    in ``_attach_chord_symbols`` (see plan phase 3.2). The bare length
    of ``score.metadata.chord_symbols`` lies about rendered content —
    low-confidence, unparseable, or pitch-octave-shaped labels are
    dropped before they ever reach the staff.

    music21 is a hard dependency (see ``pyproject.toml``); if it raises
    during export we let the exception propagate so the job manager can
    surface the failure instead of silently emitting a stub.
    """
    import music21  # noqa: PLC0415 — heavy import, kept lazy for test speed

    chord_symbols_rendered = 0

    s = music21.stream.Score()
    s.metadata = music21.metadata.Metadata()
    s.metadata.title = title or "Untitled"
    s.metadata.composer = composer or ""

    ts_num, ts_den = score.metadata.time_signature
    ts = music21.meter.TimeSignature(f"{ts_num}/{ts_den}")
    # beats_per_measure in quarter-note units (music21's internal clock).
    # For 4/4 this is 4; for 3/8 it's 1.5 (three eighth-notes = 1.5 quarters).
    beats_per_measure = ts_num * 4.0 / ts_den
    # Verify the declared key against a KS pitch-histogram analysis
    # before trusting it. Audio transcription can mis-label a piece and
    # every accidental downstream turns into a sharp/flat explosion.
    # ``_resolve_key_signature`` only overrides when the analyzer is
    # confident (correlation ≥ 0.85) AND disagrees with the upstream
    # label — otherwise the declared key wins.
    key_root, mode, _key_overridden = _resolve_key_signature(score, music21)
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

    # Render as a real piano grand staff: two ``Part`` objects bound by a
    # braced ``StaffGroup``. music21 emits this as two separate
    # ``<part>`` elements joined in ``<part-list>`` by a ``<part-group>``
    # with ``<group-symbol>brace</group-symbol>`` — the canonical
    # MusicXML grand-staff idiom that ``musicxml2ly`` / LilyPond, OSMD,
    # MuseScore, and Verovio all render identically. The older
    # ``PartStaff`` path emitted one merged ``<part>`` with
    # ``<staves>2</staves>`` and per-note ``<staff>`` tags, which
    # ``musicxml2ly`` mishandled: the LH clef was dropped and both staves
    # rendered in treble. The ``StaffGroup`` name "Piano" sits on the
    # brace; the individual ``<part>`` elements have empty
    # ``<part-name/>`` tags, so renderers show a single "Piano" label.
    piano_parts: list = []
    for hand_name, notes, clef, part_id in (
        ("Right Hand", score.right_hand, music21.clef.TrebleClef(), "P-RH"),
        ("Left Hand", score.left_hand, music21.clef.BassClef(), "P-LH"),
    ):
        part = music21.stream.Part(id=part_id)
        # Intentionally no partName / partAbbreviation. The StaffGroup
        # owns the "Piano" / "Pno." label; per-part names would render
        # twice (once per staff) in some tools.
        part.append(ts)
        part.append(ks)
        if hand_name == "Right Hand":
            part.append(tempo_mark)
        part.append(clef)

        # Group notes by voice so music21 emits <voice>1</voice> /
        # <voice>2</voice> instead of collapsing everything to a single
        # stream. Arrange now caps at 2 voices per hand (PR-10 / plan
        # phase 3.1), but the older condense pipeline can still produce
        # voice numbers up to 16 — for that high-voice path we fall back
        # to a single stream on the staff, matching the pre-PR-10
        # collapse-to-voice-1 behavior. Piano notation is *defined* by
        # ≤2 voices per staff, so clamping here matches the physical
        # reality of stems-up melody + stems-down accompaniment.
        max_voice_in_hand = max((sn.voice for sn in notes), default=1)
        use_explicit_voices = 1 < max_voice_in_hand <= 2

        # Resolve same-(pitch, voice) overlaps before music21 sees the
        # notes. An overlapping same-pitch attack near a bar line defeats
        # ``makeTies`` inside a ``Voice`` sub-stream, producing a measure
        # whose voice duration exceeds the time signature — MuseScore 4
        # renders that as a "Score corrupted" banner.
        notes = _resolve_same_pitch_overlaps_per_voice(notes)

        # Pre-split bar-crossing notes into tied in-measure pieces so
        # music21 never needs to split during ``makeTies``. Splitting
        # inside ``makeTies`` can reassign the continuation to a fresh
        # voice number in the next measure, which the OSMD sanitizer
        # then clamps back and the tie chain ends up spanning two
        # different voice tags — MuseScore 4 reports that as a corrupt
        # score (dangling ties).
        split = _split_bar_crossing_notes(notes, float(beats_per_measure))

        notes_by_voice: dict[int, list] = {}
        for sn, tie_in, tie_out in sorted(split, key=lambda row: row[0].onset_beat):
            # Snap onset / duration to the export grid so a note's end
            # lands exactly on a division boundary — otherwise a barline
            # split inside ``makeNotation`` can leave a sub-1/12-qL residual
            # that the MusicXML exporter rejects as '2048th' / 'inexpressible'.
            onset = max(0.0, _snap_quarter(sn.onset_beat))
            dur = max(_snap_quarter(sn.duration_beat), _MIN_SNAPPED_QL)
            n = music21.note.Note(sn.pitch)
            n.quarterLength = dur
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
            if tie_in and tie_out:
                n.tie = music21.tie.Tie("continue")
            elif tie_in:
                n.tie = music21.tie.Tie("stop")
            elif tie_out:
                n.tie = music21.tie.Tie("start")
            voice_num = sn.voice if use_explicit_voices else 1
            notes_by_voice.setdefault(voice_num, []).append((onset, n))

        if use_explicit_voices:
            # Insert voice 1 before voice 2 so music21's internal
            # enumeration (0-indexed, in insertion order) maps
            # deterministically back to 1/2 after makeNotation. The
            # string ``id`` we set here is discarded by ``makeMeasures``
            # — it re-creates per-measure Voice streams with fresh
            # integer ids — so the 1..N rename after ``makeNotation``
            # is the authoritative step.
            for voice_num in sorted(notes_by_voice.keys()):
                v = music21.stream.Voice(id=str(voice_num))
                for onset, n in notes_by_voice[voice_num]:
                    v.insert(onset, n)
                part.insert(0, v)
        else:
            for _, nlist in notes_by_voice.items():
                for onset, n in nlist:
                    part.insert(onset, n)

        # Dynamics + pedal marks are attached to the RH and LH parts
        # respectively so the markings sit between the staves where
        # pianists expect them. Skipped on the engrave-from-score path
        # (perf is never None there, but the expression map is empty).
        if hand_name == "Right Hand" and perf is not None:
            _attach_dynamics(part, perf.expression.dynamics, music21)
        if hand_name == "Left Hand" and perf is not None:
            _attach_pedal_marks(part, perf.expression.pedal_events, music21)

        # Chord symbols ride above the RH staff where pianists read them.
        # ``_attach_chord_symbols`` handles confidence / shape / parse
        # filtering so noisy transcriber output doesn't reach the score.
        if hand_name == "Right Hand":
            chord_symbols_rendered += _attach_chord_symbols(
                part, score.metadata.chord_symbols, music21,
            )

        # Trust arrange's grid. arrange already quantizes every onset
        # and duration via ``_estimate_best_grid`` (see
        # ``backend/services/arrange.py``), picking the best of
        # {triplet-16th, 16th, triplet-8th, 8th} for the incoming
        # material. Re-quantizing here to a coarser ``(4, 3)`` divisor
        # tuple would throw away whatever triplet-16th content arrange
        # preserved. ``makeNotation`` auto-detects tuplet brackets from
        # the raw quarterLength without an explicit hint, so the safety
        # net no longer buys anything. PR-9 forces
        # ``defaults.divisionsPerQuarter=12`` at the export boundary,
        # which is LCM(2, 3, 4) — every grid value arrange can emit
        # lands on an integer ``<duration>``.
        part.makeNotation(inPlace=True)
        # makeNotation can still emit tuplets MusicXML refuses (e.g. a
        # ``2048th`` tuplet "normal" type). Coerce before ``write()``.
        _coerce_durations_for_musicxml_export(part, music21)

        # makeMeasures rebuilds per-measure Voice sub-streams with fresh
        # integer ids (music21 treats large ints as memory locations and
        # re-numbers them starting from 0), so the original "1"/"2"
        # string ids we set before makeNotation are gone. Rename them
        # back to the MusicXML-valid 1..N range. Enumeration order
        # matches our insertion order on the part, so voice 0 → "1" and
        # voice 1 → "2" — consistent across every measure.
        for meas in part.getElementsByClass(music21.stream.Measure):
            for idx, v in enumerate(meas.voices):
                v.id = str(idx + 1)

        s.insert(0, part)
        piano_parts.append(part)

    # Bind the two Parts into a braced grand staff. music21 emits this
    # as a ``<part-group type="start"><group-symbol>brace</group-symbol>
    # </part-group>`` wrapper in ``<part-list>``, followed by two
    # ``<part>`` elements — one for RH, one for LH. No ``<staves>`` tag,
    # no per-note ``<staff>`` tag; clef lives in each part's own
    # measure-1 ``<attributes>`` block.
    s.insert(
        0,
        music21.layout.StaffGroup(
            piano_parts,
            name="Piano",
            abbreviation="Pno.",
            symbol="brace",
            barTogether=True,
        ),
    )

    # ``makeNotation=False`` skips the exporter's second ``makeNotation`` pass.
    # Per-part ``makeNotation`` can still leave one ``Note``/``Rest`` carrying a
    # ``complex`` multi-component duration or other MusicXML-unwritable shapes;
    # ``m21ToXml.noteToXml`` then raises (``inexpressible`` / ``complex``). The
    # supported fix is ``splitAtDurations(recurse=True)`` before export — see
    # ``music21.converter.subConverters`` tests around ``makeNotation=False``.
    s.splitAtDurations(recurse=True)
    _coerce_durations_for_musicxml_export(s, music21)
    # Rare: rational snap can reintroduce a ``complex`` multi-component view;
    # a second split is cheap insurance before export.
    s.splitAtDurations(recurse=True)
    _coerce_durations_for_musicxml_export(s, music21)

    # Force divisions=12 at the exporter boundary. music21's MeasureExporter
    # reads ``defaults.divisionsPerQuarter`` verbatim when stamping the
    # ``<divisions>`` tag (m21ToXml.setMxAttributesObjectForStartOfMeasure)
    # and the shipped default is 10080 — producing MusicXML that OSMD
    # chokes on. 12 = LCM(4, 3) which is the finest grid we need to
    # represent the (4, 3) quarterLengthDivisors quantization: 16th = 3
    # divisions, 8th = 6, triplet-8th = 4, quarter = 12. Done here
    # (process-global, restored after write) rather than at module import
    # so non-engrave music21 callers keep the upstream default.
    with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    prior_divisions = music21.defaults.divisionsPerQuarter
    try:
        music21.defaults.divisionsPerQuarter = 12
        # music21's MusicXML writer defaults to ``makeNotation=True``, which
        # deep-copies the score and runs ``makeNotation`` again on every part.
        # That second pass can emit tuplets MusicXML cannot encode (e.g. a
        # ``2048th`` normal-type) even after our per-part cleanup. We already
        # called ``makeNotation`` on each ``PartStaff`` above — export the
        # score as-is. (Requires a well-formed ``Score``; see GeneralObjectExporter.)
        s.write("musicxml", fp=str(tmp_path), makeNotation=False)
        raw = tmp_path.read_bytes()
        return _sanitize_musicxml_for_osmd(raw), chord_symbols_rendered
    finally:
        music21.defaults.divisionsPerQuarter = prior_divisions
        tmp_path.unlink(missing_ok=True)


def _sanitize_musicxml_for_osmd(raw: bytes) -> bytes:
    """Post-process music21 MusicXML to fix OSMD + MuseScore issues.

    Three mechanical fixups after music21 export:

    1. ``<voice>0</voice>`` → ``<voice>1</voice>`` — music21 emits 0 when
       a note isn't wrapped in a ``Voice`` sub-stream and the part has
       multiple voices elsewhere. MusicXML requires voice numbers ≥ 1,
       and OSMD rejects zero outright.
    2. ``<voice>N</voice>`` with N ≥ 3 → ``<voice>2</voice>`` — defense-in-
       depth for any upstream stage (e.g. the condense path) that
       produces more than two voices per hand. OSMD's VexFlow backend
       crashes (``parentVoiceEntry undefined``) on 3+ voices per part;
       clamping to 2 is the minimum-damage fallback.
    3. Tie-chain voice alignment — music21 can reassign a bar-crossing
       note's tie continuation to a different voice in the next
       measure (see ``_split_bar_crossing_notes`` for the upstream
       attempt). MuseScore 4 flags mismatched tie-start / tie-stop
       voices as a corrupt score. Walk the XML part by part, track
       open ties by ``(step, octave, alter)``, and rewrite the
       tie-stop's ``<voice>`` to match the tie-start's voice. The
       voice reassignment only affects the single tied note;
       surrounding voice content keeps its tags.
    """
    text = raw.decode("utf-8")

    # MusicXML voice=0 is invalid and OSMD rejects it outright.
    text = re.sub(r"<voice>0</voice>", "<voice>1</voice>", text)
    remapped = _remap_voices_per_staff(text.encode("utf-8"))
    return _align_tie_chain_voices(remapped)


def _remap_voices_per_staff(raw: bytes) -> bytes:
    """Renumber voice tags per staff so each staff uses ``{1, 2}``.

    music21's exporter numbers voices globally within a ``<part>``
    — a grand staff with two voices per hand gets voices ``{1..4}``.
    OSMD's VexFlow backend crashes on ``voice ≥ 3``, so the old
    sanitizer clamped every ``voice>=3`` to ``voice=2``. That merged
    distinct musical lines into one voice, which MuseScore 4 reports
    as "Score corrupted" when the merged voice's duration adds up
    past the time signature.

    Correct fix: remap per staff. Collect the set of voice numbers
    actually used on each staff and compress them to ``1..N``. Staves
    independently get ``voice=1`` and ``voice=2`` — MusicXML voices
    are part-scoped per the spec, but both MuseScore and OSMD treat
    them per-staff, so this is the renderer-pleasing compromise.
    """
    import xml.etree.ElementTree as ET  # noqa: PLC0415

    root = ET.fromstring(raw)

    for part in root.findall("part"):
        for measure in part.findall("measure"):
            # Collect the voices used on each staff in this measure.
            voices_by_staff: dict[str, list[str]] = {}
            cur_staff = "1"
            for note in measure.findall("note"):
                staff_el = note.find("staff")
                staff = staff_el.text if staff_el is not None else cur_staff
                cur_staff = staff
                voice_el = note.find("voice")
                if voice_el is None:
                    continue
                v = voice_el.text or "1"
                slot = voices_by_staff.setdefault(staff, [])
                if v not in slot:
                    slot.append(v)

            # Build the per-staff remap: first seen voice → "1", second
            # → "2", and clamp everything beyond that to "2" (OSMD
            # can't handle 3 voices on one staff anyway).
            remap: dict[tuple[str, str], str] = {}
            for staff, vs in voices_by_staff.items():
                for idx, v in enumerate(vs):
                    remap[(staff, v)] = str(min(idx + 1, 2))

            # Apply the remap.
            cur_staff = "1"
            for note in measure.findall("note"):
                staff_el = note.find("staff")
                staff = staff_el.text if staff_el is not None else cur_staff
                cur_staff = staff
                voice_el = note.find("voice")
                if voice_el is None:
                    continue
                v = voice_el.text or "1"
                new_v = remap.get((staff, v))
                if new_v is not None and new_v != v:
                    voice_el.text = new_v

    prefix_end = raw.find(b"<score-partwise")
    prefix = raw[:prefix_end] if prefix_end > 0 else b""
    body = ET.tostring(root, encoding="utf-8", xml_declaration=False)
    return prefix + body


def _align_tie_chain_voices(raw: bytes) -> bytes:
    """Rewrite tie-stop / tie-continue notes to match the voice of the
    preceding tie-start for the same pitch within the same ``<part>``.

    music21 sometimes assigns a bar-crossing note's continuation to a
    different voice in the next measure. After the voice-clamp step
    this shows up as tie-start on voice 1, tie-stop on voice 2 — which
    MuseScore 4 reports as a dangling tie corruption. We walk the XML
    part by part (ties never cross hands), tracking open ties by
    ``(step, octave, alter)``, and rewrite the tie-stop's ``<voice>``
    tag to match the tie-start's voice.
    """
    import xml.etree.ElementTree as ET  # noqa: PLC0415 — only needed on this path

    parser = ET.XMLParser()
    root = ET.fromstring(raw, parser=parser)

    rewrites = 0
    for part in root.findall("part"):
        # Ties never cross hands. Reset the open-tie map for each <part>
        # so an RH tie-start can't "match" an LH same-pitch attack in
        # the two-part encoding (where <staff> tags are absent and the
        # old (pitch, staff) key degenerated to (pitch,)).
        open_ties: dict[tuple[str, str, str], str] = {}
        for measure in part.findall("measure"):
            for note in measure.findall("note"):
                pitch = note.find("pitch")
                if pitch is None:
                    continue
                key = (
                    pitch.findtext("step") or "",
                    pitch.findtext("octave") or "",
                    pitch.findtext("alter") or "0",
                )
                voice_el = note.find("voice")
                voice = voice_el.text if voice_el is not None else "1"
                for tie in note.findall("tie"):
                    typ = tie.get("type")
                    if typ == "start":
                        open_ties[key] = voice
                    elif typ == "stop":
                        expected = open_ties.pop(key, None)
                        if expected is not None and expected != voice and voice_el is not None:
                            voice_el.text = expected
                            rewrites += 1

    if rewrites == 0:
        return raw

    # Preserve the original XML declaration + DOCTYPE that ElementTree drops.
    prefix_end = raw.find(b"<score-partwise")
    prefix = raw[:prefix_end] if prefix_end > 0 else b""
    body = ET.tostring(root, encoding="utf-8", xml_declaration=False)
    return prefix + body


# ---------------------------------------------------------------------------
# PDF rendering — best effort, falls back to a tiny stub.
# ---------------------------------------------------------------------------

def _render_pdf_bytes(musicxml_bytes: bytes) -> bytes:
    """Render MusicXML to PDF bytes.

    Tries LilyPond first (this is what production ships — ~250 MB apt
    package, musicxml2ly + lilypond binaries) and MuseScore as a
    higher-fidelity fallback for local dev machines that have it
    installed (including macOS ``.app`` bundles via ``musescore_executable_paths``).
    Returns the 60-byte stub PDF only when no renderer is
    available or all renderers fail.
    """
    ms_paths = musescore_executable_paths()
    has_lilypond = bool(shutil.which("musicxml2ly") and shutil.which("lilypond"))
    if not has_lilypond and not ms_paths:
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

        for mscore in ms_paths:
            pdf_path = tmp / "sheet.pdf"
            try:
                subprocess.run(
                    [mscore, "-o", str(pdf_path), str(xml_path)],
                    check=True, capture_output=True, timeout=120,
                )
                if pdf_path.is_file():
                    log.info(
                        "PDF rendered via %s (%d bytes)", mscore, pdf_path.stat().st_size,
                    )
                    return pdf_path.read_bytes()
                log.warning("%s ran but produced no PDF at %s", mscore, pdf_path)
            except subprocess.CalledProcessError as exc:
                log.warning(
                    "%s PDF render failed: %s",
                    mscore,
                    (exc.stderr or b"").decode("utf-8", "replace")[:500],
                )
            except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
                log.warning("%s PDF render failed: %s", mscore, exc)

    return _STUB_PDF


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

def _engrave_sync(
    payload: HumanizedPerformance | PianoScore,
    title: str,
    composer: str,
) -> tuple[bytes, bytes, bytes, int]:
    """Render all three artifacts.

    Returns ``(pdf, musicxml, midi, chord_symbols_rendered)``. The trailing
    count reflects chord symbols that *actually made it to the staff* after
    ``_attach_chord_symbols`` filtering — not the raw input count — so
    ``EngravedScoreData.includes_chord_symbols`` can tell the truth.
    """
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
    musicxml_bytes, chord_symbols_rendered = _render_musicxml_bytes(
        score, perf, title, composer,
    )
    pdf_bytes = _render_pdf_bytes(musicxml_bytes)
    return pdf_bytes, musicxml_bytes, midi_bytes, chord_symbols_rendered


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
        pdf_bytes, musicxml_bytes, midi_bytes, chord_symbols_rendered = (
            await asyncio.to_thread(_engrave_sync, payload, title, composer)
        )

        prefix = f"jobs/{job_id}/output"
        pdf_uri = self.blob_store.put_bytes(f"{prefix}/sheet.pdf", pdf_bytes)
        musicxml_uri = self.blob_store.put_bytes(f"{prefix}/score.musicxml", musicxml_bytes)
        midi_uri = self.blob_store.put_bytes(f"{prefix}/humanized.mid", midi_bytes)

        score = payload.score if isinstance(payload, HumanizedPerformance) else payload
        chord_input_count = len(score.metadata.chord_symbols)

        log.info(
            "engrave: done job_id=%s bytes pdf=%d musicxml=%d midi=%d "
            "chord_symbols=%d/%d rendered",
            job_id,
            len(pdf_bytes),
            len(musicxml_bytes),
            len(midi_bytes),
            chord_symbols_rendered,
            chord_input_count,
        )

        # These flags describe what was *actually rendered*. As of PR-5
        # (plan phase 1.3–1.6) engrave renders dynamics and pedal marks
        # when the humanized input populates them, so we surface that
        # state directly from the expression map instead of hardcoding.
        # PR-11 (plan phase 3.2) extends the same truth-telling to chord
        # symbols: the input count can lie (low-confidence / unparseable
        # labels are dropped), so gate on the filtered-rendered count.
        perf_for_flags = payload if isinstance(payload, HumanizedPerformance) else None
        return EngravedOutput(
            schema_version=SCHEMA_VERSION,
            metadata=EngravedScoreData(
                includes_dynamics=bool(perf_for_flags and perf_for_flags.expression.dynamics),
                includes_pedal_marks=bool(perf_for_flags and perf_for_flags.expression.pedal_events),
                includes_fingering=False,
                includes_chord_symbols=chord_symbols_rendered > 0,
                title=title,
                composer=composer,
            ),
            pdf_uri=pdf_uri,
            musicxml_uri=musicxml_uri,
            humanized_midi_uri=midi_uri,
            audio_preview_uri=None,
        )
