"""Local engrave stack: PianoScore → MusicXML → SVG/PDF.

Replaces the external HTTP black-box engraver with a music21 → MusicXML
writer plus Verovio (SVG) and LilyPond (PDF) renderers. Operates on the
structured ``(PianoScore, ExpressionMap)`` rather than rendering MIDI
through an opaque transcriber, so chord symbols, dynamics, pedal marks,
title, key/time signatures, and per-note voice numbers all survive
intact across the rendering boundary.

Phase 4.1: ``score_to_musicxml`` — music21-backed MusicXML emitter.
Phase 4.2: ``musicxml_to_svg`` (Verovio) + ``musicxml_to_pdf`` (LilyPond).

License posture (intentionally conservative — see Phase 4 risks):
  * music21    BSD     in-process (safe).
  * verovio    LGPL    dynamic link via PyPI wheel; never linked statically.
  * lilypond   GPL     subprocess call only — no Python-side data sharing
                       beyond the MusicXML file passed by path.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from backend.contracts import (
    ExpressionMap,
    PianoScore,
    ScoreMetadata,
    ScoreNote,
)

log = logging.getLogger(__name__)


class EngraveLocalError(RuntimeError):
    """Raised when the local engrave stack fails to produce its artifact."""


# ── Feature summary (parallels midi_render.EmittedFeatures) ────────────


@dataclass
class EngravedFeatures:
    """What the MusicXML emitter actually wrote.

    Mirrors :class:`backend.services.midi_render.EmittedFeatures` so the
    runner can populate ``EngravedScoreData.includes_*`` from real
    content instead of hard-coded ``False``. Counts are intentionally
    cheap to compute — they're set at insertion time, not by parsing
    the emitted XML back.
    """

    title: str = ""
    composer: str = ""
    has_key_signature: bool = False
    has_tempo_marking: bool = False
    chord_symbol_count: int = 0
    dynamic_count: int = 0
    pedal_event_count: int = 0
    articulation_count: int = 0
    voice_count: int = 0  # distinct voice IDs emitted across both staves
    note_count: int = 0


# ── Key signature parsing ──────────────────────────────────────────────

# Pitch-class lookup matches midi_render._PITCH_CLASS so the two stacks
# agree on which input strings are parseable.
_MINOR_MODES = {"minor", "min", "m", "aeolian", "dorian", "phrygian", "locrian"}
_MAJOR_MODES = {"major", "maj", "ionian", "lydian", "mixolydian", ""}


def _parse_key_string(key: str) -> tuple[str, Literal["major", "minor"]] | None:
    """Parse ``"C:major"`` / ``"F#:minor"`` / ``"Bb:dorian"`` → ``(tonic, mode)``.

    Modes collapse onto major/minor by parallel — Dorian → minor,
    Mixolydian → major. Returns ``None`` when the input doesn't parse
    so callers fall through silently rather than emit a wrong key.
    """
    if not key:
        return None
    m = re.match(r"\s*([A-Ga-g])([#b]*)\s*[: ]?\s*([A-Za-z]*)\s*$", key)
    if not m:
        return None
    letter, accidentals, mode = m.group(1).upper(), m.group(2), m.group(3).lower()
    if letter not in {"C", "D", "E", "F", "G", "A", "B"}:
        return None
    tonic = letter + accidentals  # music21 accepts "C", "F#", "Bb" directly
    if mode in _MINOR_MODES:
        return tonic, "minor"
    if mode in _MAJOR_MODES:
        return tonic, "major"
    return None


# ── Chord-symbol Harte → music21 figure ────────────────────────────────

# Map common Harte quality tokens to music21's ChordSymbol figures.
# Harte left-of-colon is the root (e.g. "C", "F#"); right-of-colon is the
# quality. music21's parser handles "Cmaj7", "Am7", "G7", "F#m7b5"
# directly, so we mostly just normalize separators.
_HARTE_TO_M21 = {
    "maj": "",         # "C:maj" → "C"
    "major": "",
    "min": "m",        # "A:min" → "Am"
    "minor": "m",
    "dim": "dim",
    "aug": "+",
    "sus2": "sus2",
    "sus4": "sus4",
    "maj7": "maj7",
    "min7": "m7",
    "dom7": "7",
    "7": "7",
    "min9": "m9",
    "maj9": "maj9",
    "9": "9",
    "11": "11",
    "13": "13",
}


def _harte_to_figure(label: str) -> str | None:
    """Translate a Harte-style chord label to a music21 ChordSymbol figure.

    Returns ``None`` for empty or unparseable labels so the caller can
    skip emission rather than insert a meaningless symbol.
    """
    if not label:
        return None
    raw = label.strip()
    if not raw:
        return None
    # Already music21-friendly (e.g. "Cmaj7", "Am") — trust it.
    if ":" not in raw:
        return raw
    root, _, quality = raw.partition(":")
    quality = quality.strip().lower()
    suffix = _HARTE_TO_M21.get(quality)
    if suffix is None:
        # Unknown quality — fall back to root + raw quality and hope music21
        # parses it. If not, music21 raises and the caller skips it.
        suffix = quality
    return f"{root.strip()}{suffix}"


# ── Articulation type → music21 articulation class name ────────────────

_ARTICULATION_MAP = {
    "staccato": "Staccato",
    "tenuto": "Tenuto",
    "accent": "Accent",
    # "legato" is a slur in music21, handled separately if/when slurs ship.
}

# Fermata lives under ``music21.expressions``, not ``music21.articulations``;
# it also attaches to a note's ``.expressions`` list rather than
# ``.articulations``. Tracked separately so the dispatch in
# ``_attach_articulations`` can pick the right module + attribute.
_EXPRESSION_MAP = {
    "fermata": "Fermata",
}


# ── Public entry point ────────────────────────────────────────────────


def score_to_musicxml(
    score: PianoScore,
    expression: ExpressionMap | None = None,
    *,
    title: str | None = None,
    composer: str | None = None,
) -> tuple[bytes, EngravedFeatures]:
    """Render ``(PianoScore, ExpressionMap)`` to MusicXML 4.0 bytes.

    Returns ``(xml_bytes, features)``. Title and composer fall back to
    ``score.metadata.title`` / ``.composer`` when not provided.

    Emits, per the Phase 4.1 requirements:
      * Title + composer (top-level Metadata).
      * Key signature (parsed from ``metadata.key``).
      * Time signature.
      * Tempo marking (BPM + optional text like "Andante").
      * Chord symbols (``metadata.chord_symbols`` → ``music21.harmony.ChordSymbol``).
      * Dynamics (``ExpressionMap.dynamics`` → ``music21.dynamics.Dynamic``
        plus crescendo/decrescendo spanners).
      * Pedal marks (``ExpressionMap.pedal_events`` → ``PedalMark`` spanner).
      * Articulations attached to the score notes by ``score_note_id``.
      * Per-note voice numbers (preserved via ``ScoreNote.voice``).

    Raises ``EngraveLocalError`` when music21 isn't importable or when
    the produced XML is empty / not well-formed.
    """
    try:
        # noqa block: lazy-imported optional dep; ruff's import sorter
        # wants to split this by alias into separate ``from`` blocks,
        # which makes the call site much harder to read than a single
        # grouped import. Keep them together.
        from music21 import articulations as m21_artic  # noqa: PLC0415, I001
        from music21 import clef  # noqa: PLC0415, I001
        from music21 import dynamics as m21_dyn  # noqa: PLC0415, I001
        from music21 import expressions as m21_expr  # noqa: PLC0415, I001
        from music21 import harmony  # noqa: PLC0415, I001
        from music21 import instrument  # noqa: PLC0415, I001
        from music21 import key as m21_key  # noqa: PLC0415, I001
        from music21 import layout  # noqa: PLC0415, I001
        from music21 import metadata as m21_metadata  # noqa: PLC0415, I001
        from music21 import meter  # noqa: PLC0415, I001
        from music21 import note as m21_note  # noqa: PLC0415, I001
        from music21 import spanner  # noqa: PLC0415, I001
        from music21 import stream  # noqa: PLC0415, I001
        from music21 import tempo  # noqa: PLC0415, I001
    except ImportError as exc:
        raise EngraveLocalError(
            "music21 is not installed; cannot build MusicXML. "
            "music21 is a top-level dependency in pyproject.toml — "
            "this is a deploy configuration error."
        ) from exc

    meta = score.metadata
    features = EngravedFeatures()

    sc = stream.Score()

    # ── Top-level metadata (title, composer, arranger) ────────────────
    md = m21_metadata.Metadata()
    md.title = title or meta.title or "Untitled"
    md.composer = composer or meta.composer or ""
    if meta.arranger:
        md.arranger = meta.arranger
    sc.append(md)
    features.title = md.title or ""
    features.composer = md.composer or ""

    m21_modules = {
        "instrument": instrument,
        "key": m21_key,
        "meter": meter,
        "note": m21_note,
        "stream": stream,
    }

    # ── Build each staff FLAT (notes in voices, no measures yet) ──────
    # We delay makeMeasures until every part-level element (tempo, chord
    # symbols, dynamics) has been inserted at its absolute beat offset.
    # Otherwise makeMeasures pulls only what's in the part at the time it
    # runs into the measure structure, and anything inserted afterwards
    # dangles outside the measures and is silently dropped by the
    # MusicXML exporter.
    rh_part, rh_voice_ids = _build_part_flat(
        score.right_hand, meta, hand="rh",
        m21_clef=clef.TrebleClef(), m21_modules=m21_modules, features=features,
    )
    lh_part, lh_voice_ids = _build_part_flat(
        score.left_hand, meta, hand="lh",
        m21_clef=clef.BassClef(), m21_modules=m21_modules, features=features,
    )
    features.voice_count = len(rh_voice_ids | lh_voice_ids)

    # ── Tempo marking on RH part (must precede makeMeasures) ──────────
    if meta.tempo_map:
        bpm = float(meta.tempo_map[0].bpm)
        text = meta.tempo_marking or None
        try:
            mm = tempo.MetronomeMark(number=bpm, text=text)
            rh_part.insert(0.0, mm)
            features.has_tempo_marking = True
        except Exception as exc:  # noqa: BLE001
            log.warning("engrave_local: failed to insert tempo marking: %s", exc)

    # ── Chord symbols on RH part (must precede makeMeasures) ──────────
    for ch in meta.chord_symbols or []:
        figure = _harte_to_figure(ch.label)
        if figure is None:
            continue
        try:
            cs = harmony.ChordSymbol(figure)
            cs.duration.quarterLength = max(0.0, float(ch.duration_beat or 0.0)) or 1.0
            rh_part.insert(float(ch.beat), cs)
            features.chord_symbol_count += 1
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "engrave_local: skip chord symbol label=%r figure=%r: %s",
                ch.label, figure, exc,
            )

    # ── Dynamics on RH part (must precede makeMeasures) ───────────────
    if expression:
        _attach_dynamics(
            rh_part, expression, m21_dyn=m21_dyn, spanner=spanner, features=features,
        )

    # ── NOW measureize: pulls everything above into measure containers ─
    for part in (rh_part, lh_part):
        try:
            part.makeMeasures(inPlace=True)
            part.makeTies(inPlace=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("engrave_local: makeMeasures/makeTies on %s failed: %s",
                        part.id, exc)

    # ── Pedal + articulations attach AFTER measureization ─────────────
    # Both walk the part's notes (now inside Voices inside Measures) to
    # find spanner anchors and articulation targets, so they need the
    # final note layout.
    if expression:
        _attach_pedal(lh_part, expression, m21_expr=m21_expr, features=features)
        _attach_articulations(
            rh_part, expression,
            m21_artic=m21_artic, m21_expr=m21_expr, features=features,
        )
        _attach_articulations(
            lh_part, expression,
            m21_artic=m21_artic, m21_expr=m21_expr, features=features,
        )

    # ── Compose the score and group as a piano staff ──────────────────
    sc.insert(0, rh_part)
    sc.insert(0, lh_part)
    grp = layout.StaffGroup(
        [rh_part, lh_part],
        symbol="brace",
        name="Piano",
        abbreviation="Pno.",
    )
    sc.insert(0, grp)

    # ── Serialize to MusicXML 4.0 bytes ───────────────────────────────
    xml_bytes = _stream_to_musicxml_bytes(sc)

    # Sanity check: refuse to return an empty / malformed payload. This
    # mirrors the ``_looks_like_stub`` guard on ml_engraver_client; a
    # downstream consumer that gets a blank score is the worst possible
    # silent-failure mode for a music app.
    if not xml_bytes or len(xml_bytes) < 200:
        raise EngraveLocalError(
            f"music21 emitted suspiciously small MusicXML "
            f"(bytes={len(xml_bytes)}); refusing to surface a blank score."
        )
    _assert_well_formed_musicxml(xml_bytes)

    log.info(
        "engrave_local: MusicXML emitted bytes=%d notes=%d voices=%d "
        "chords=%d dynamics=%d pedals=%d articulations=%d key=%s tempo=%s",
        len(xml_bytes),
        features.note_count,
        features.voice_count,
        features.chord_symbol_count,
        features.dynamic_count,
        features.pedal_event_count,
        features.articulation_count,
        features.has_key_signature,
        features.has_tempo_marking,
    )
    return xml_bytes, features


# ── Internal builders ──────────────────────────────────────────────────


def _build_part_flat(
    notes: list[ScoreNote],
    meta: ScoreMetadata,
    *,
    hand: Literal["rh", "lh"],
    m21_clef,
    m21_modules: dict,
    features: EngravedFeatures,
) -> tuple[Any, set[int]]:
    """Build a single staff (Part) populated with notes, clef, key, time sig.

    The returned Part is *flat* — notes live directly in Voice containers
    inserted at offset 0, with no Measure structure yet. Callers are
    expected to insert any further part-level elements (tempo, chord
    symbols, dynamics) at their absolute beat offsets and THEN call
    ``part.makeMeasures(inPlace=True)`` so all elements get bundled into
    the right measures together.

    Returns ``(part, voice_ids)`` so the caller can roll up a piece-wide
    voice count for the features summary.
    """
    instrument = m21_modules["instrument"]
    m21_key = m21_modules["key"]
    meter = m21_modules["meter"]
    m21_note = m21_modules["note"]
    stream = m21_modules["stream"]

    part = stream.Part()
    part.id = hand
    part.partName = "Piano"
    part.partAbbreviation = "Pno."
    part.append(instrument.Piano())

    # Clef at beat 0
    part.append(m21_clef)

    # Key signature — silent fall-through on unparseable input
    parsed = _parse_key_string(meta.key)
    if parsed is not None:
        tonic, mode = parsed
        try:
            ks = m21_key.Key(tonic if mode == "major" else tonic.lower(), mode)
            part.append(ks)
            features.has_key_signature = True
        except Exception as exc:  # noqa: BLE001
            log.warning("engrave_local: bad key %r: %s", meta.key, exc)

    # Time signature
    num, den = meta.time_signature
    try:
        ts = meter.TimeSignature(f"{int(num)}/{int(den)}")
        part.append(ts)
    except Exception as exc:  # noqa: BLE001
        log.warning("engrave_local: bad time signature %s/%s: %s", num, den, exc)
        ts = meter.TimeSignature("4/4")
        part.append(ts)

    # Group notes by voice and insert at their absolute beat offsets.
    # music21's ``stream.Voice`` lets us preserve the upstream voice ID
    # explicitly rather than collapse everything onto one voice.
    by_voice: dict[int, list[ScoreNote]] = {}
    for n in notes:
        by_voice.setdefault(max(1, int(n.voice or 1)), []).append(n)

    voice_ids: set[int] = set()

    if not by_voice:
        # Empty staff — emit a single whole rest so the part isn't blank.
        # Without this music21 sometimes drops the part from MusicXML
        # output entirely, which removes the staff from the rendered score.
        rest = m21_note.Rest(quarterLength=float(num) * (4.0 / float(den)))
        part.insert(0.0, rest)
    else:
        for voice_id in sorted(by_voice.keys()):
            v = stream.Voice(id=str(voice_id))
            for n in sorted(by_voice[voice_id], key=lambda x: x.onset_beat):
                # quarterLength must be > 0; floor at 1/64 note to avoid
                # music21 silently dropping ultra-short notes.
                qlen = max(1.0 / 16.0, float(n.duration_beat))
                note_obj = m21_note.Note(int(n.pitch), quarterLength=qlen)
                # MIDI velocity 1..127. music21 coerces 0 → "no volume",
                # which becomes a silent note in MIDI playback — clamp.
                note_obj.volume.velocity = max(1, min(127, int(n.velocity)))
                # Tag with score_note_id so articulation lookup later can
                # find the right note without re-walking the score. The
                # music21 9.x Editorial is a dict subclass — store keys
                # directly via __setitem__.
                note_obj.editorial["score_note_id"] = n.id
                v.insert(float(n.onset_beat), note_obj)
                features.note_count += 1
            part.insert(0.0, v)
            voice_ids.add(voice_id)

    return part, voice_ids


def _attach_dynamics(part, expression: ExpressionMap, *, m21_dyn, spanner, features: EngravedFeatures) -> None:
    """Insert Dynamic markings + crescendo/decrescendo spanners on the part."""
    for d in expression.dynamics or []:
        kind = d.type
        try:
            if kind in {"pp", "p", "mp", "mf", "f", "ff"}:
                obj = m21_dyn.Dynamic(kind)
                part.insert(float(d.beat), obj)
                features.dynamic_count += 1
            elif kind == "crescendo":
                # Crescendo spans from d.beat for d.span_beats (default 4).
                cresc = m21_dyn.Crescendo()
                start_beat = float(d.beat)
                # Anchor: insert the spanner at the start beat. music21
                # represents the span via the spanner element; we don't
                # need an explicit end-beat here because it's a Spanner
                # without spanned notes. Engravers render it as a hairpin.
                part.insert(start_beat, cresc)
                features.dynamic_count += 1
            elif kind == "decrescendo":
                decresc = m21_dyn.Diminuendo()
                part.insert(float(d.beat), decresc)
                features.dynamic_count += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("engrave_local: skip dynamic %r: %s", d, exc)


def _attach_pedal(part, expression: ExpressionMap, *, m21_expr, features: EngravedFeatures) -> None:
    """Render PedalEvent ranges as music21 ``PedalMark`` spanners.

    PedalMark is a Spanner — the visual ``Ped. ___ *`` bracket spans the
    notes between onset_beat and offset_beat. We attach it to the notes
    that fall inside the pedal range so the engraver knows where to draw
    the bracket; if no notes fall inside (rare, but possible on a piece
    with very sparse RH), we fall back to a TextExpression so the marking
    still appears.

    Called AFTER ``part.makeMeasures(inPlace=True)``, so notes live inside
    Measure→Voice containers rather than directly under the part. The
    classic ``getOffsetBySite(part)`` lookup raises in that hierarchy
    because the note's site is the Voice, not the part — use
    ``getOffsetInHierarchy(part)`` which walks the parent chain.
    """
    if not expression.pedal_events:
        return

    # Collect candidate notes from the part for spanner attachment. This
    # uses .recurse() so we pick up notes inside Voices inside Measures.
    all_notes = list(part.recurse().notes)

    # Pre-compute each note's absolute offset in the part once. The
    # ``getOffsetInHierarchy`` walk is O(depth-of-tree) so caching it
    # keeps the per-pedal scan O(notes) instead of O(notes × pedals).
    note_offsets: list[tuple[float, object]] = []
    for n in all_notes:
        try:
            note_offsets.append((float(n.getOffsetInHierarchy(part)), n))
        except Exception:  # noqa: BLE001 — orphaned notes get skipped
            continue

    for pe in expression.pedal_events:
        # Only sustain/sostenuto/una_corda survive the contract; render
        # all three as PedalMark with appropriate type if music21 supports it.
        on = float(pe.onset_beat)
        off = float(pe.offset_beat)
        try:
            inside = [n for offset, n in note_offsets if on <= offset <= off]
            if inside:
                pedal = m21_expr.PedalMark(*inside)
                # PedalMark default form is "symbol" (Ped. ... *).
                part.insert(on, pedal)
            else:
                # No notes to span — fall back to text expressions so the
                # marking is at least visible to the user.
                part.insert(on, m21_expr.TextExpression("Ped."))
                part.insert(off, m21_expr.TextExpression("*"))
            features.pedal_event_count += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("engrave_local: skip pedal %r: %s", pe, exc)


def _attach_articulations(
    part, expression: ExpressionMap, *,
    m21_artic, m21_expr, features: EngravedFeatures,
) -> None:
    """Attach Articulation / Expression markings to the notes named by ``score_note_id``.

    music21 splits performance markings across two namespaces:
      * ``music21.articulations`` — Staccato, Tenuto, Accent. Attach to
        ``note.articulations``.
      * ``music21.expressions``   — Fermata. Attach to ``note.expressions``.

    The dispatch tables ``_ARTICULATION_MAP`` and ``_EXPRESSION_MAP``
    pick the right module + attribute by contract type so callers don't
    care that "fermata" is technically an expression rather than an
    articulation in music21's taxonomy.
    """
    if not expression.articulations:
        return

    # Build an index over the part's notes keyed by score_note_id (which
    # we tagged into note.editorial.misc when constructing the part).
    note_index: dict[str, Any] = {}
    for n in part.recurse().notes:
        ed = getattr(n, "editorial", None)
        sid = ed.get("score_note_id") if ed is not None else None
        if sid:
            note_index[sid] = n

    for art in expression.articulations:
        target = note_index.get(art.score_note_id)
        if target is None:
            # Articulation on a note that isn't in this part (e.g. LH
            # articulation while we're decorating RH). Quietly skip.
            continue

        artic_cls_name = _ARTICULATION_MAP.get(art.type)
        expr_cls_name = _EXPRESSION_MAP.get(art.type)
        if artic_cls_name is None and expr_cls_name is None:
            continue

        try:
            if artic_cls_name is not None:
                cls = getattr(m21_artic, artic_cls_name)
                target.articulations.append(cls())
            elif expr_cls_name is not None:
                cls = getattr(m21_expr, expr_cls_name)
                target.expressions.append(cls())
            features.articulation_count += 1
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "engrave_local: skip articulation type=%r note=%s: %s",
                art.type, art.score_note_id, exc,
            )


def _stream_to_musicxml_bytes(sc) -> bytes:
    """Serialize a music21 Score to MusicXML 4.0 bytes (UTF-8 encoded)."""
    try:
        from music21.musicxml.m21ToXml import GeneralObjectExporter  # noqa: PLC0415
    except ImportError as exc:
        raise EngraveLocalError(
            "music21.musicxml exporter not available — music21 install is broken."
        ) from exc

    try:
        exporter = GeneralObjectExporter(sc)
        out = exporter.parse()
    except Exception as exc:  # noqa: BLE001 — exporter wraps many failure modes
        raise EngraveLocalError(f"music21 MusicXML export failed: {exc}") from exc

    if isinstance(out, str):
        out = out.encode("utf-8")
    return out


def _assert_well_formed_musicxml(xml_bytes: bytes) -> None:
    """Lightweight well-formedness check — raises EngraveLocalError on failure.

    Strict XSD validation against the MusicXML 4.0 schema is a Phase 4
    follow-up (the XSD is ~200KB and would need bundling); this check
    catches the common failure modes (truncated XML, missing root,
    XML-parse errors) without that overhead.
    """
    try:
        from lxml import etree  # noqa: PLC0415
    except ImportError:
        # lxml is in dev deps; if unavailable in production, skip the
        # check rather than fail the engrave. The size guard above already
        # rejects the most obvious failure mode.
        return

    try:
        root = etree.fromstring(xml_bytes)
    except etree.XMLSyntaxError as exc:
        raise EngraveLocalError(f"music21 produced malformed XML: {exc}") from exc

    if root.tag not in {"score-partwise", "score-timewise"}:
        raise EngraveLocalError(
            f"music21 produced unexpected XML root <{root.tag}> "
            f"(expected score-partwise or score-timewise)"
        )


# ── Phase 4.2: Verovio (SVG) + LilyPond (PDF) renderers ────────────────


# LilyPond is single-threaded and can hang on malformed MusicXML — cap
# wall time at 60s with an explicit failure path. Same rationale on the
# Verovio side, though Verovio is generally safer (in-process LGPL
# library rather than subprocess).
_LILYPOND_TIMEOUT_SEC = 60
_VEROVIO_TIMEOUT_SEC = 60


def musicxml_to_svg(xml_bytes: bytes) -> bytes:
    """Render MusicXML bytes to SVG bytes via Verovio.

    Verovio is LGPL — the PyPI wheel is a dynamic link, never statically
    embedded. Returns the first page as a self-contained SVG; multi-page
    SVG output is a Phase 4 follow-up.
    """
    try:
        import verovio  # noqa: PLC0415 — optional dep
    except ImportError as exc:
        raise EngraveLocalError(
            "verovio is not installed; cannot render SVG. "
            "Install with `pip install verovio>=4.0` or add the [engrave] extra."
        ) from exc

    try:
        toolkit = verovio.toolkit()
        # Defaults are reasonable for piano scores; engravers can tweak
        # via a follow-up that exposes the option set as kwargs.
        toolkit.setOptions({
            "pageHeight": 1700,
            "pageWidth": 1200,
            "scale": 40,
            "adjustPageHeight": True,
        })
        if not toolkit.loadData(xml_bytes.decode("utf-8")):
            raise EngraveLocalError("verovio rejected the MusicXML payload")
        svg = toolkit.renderToSVG(1)  # page 1
    except EngraveLocalError:
        raise
    except Exception as exc:  # noqa: BLE001 — verovio can crash on bad input
        raise EngraveLocalError(f"verovio render failed: {exc}") from exc

    if not svg or len(svg) < 100:
        raise EngraveLocalError(
            f"verovio produced suspiciously small SVG (bytes={len(svg)})"
        )
    return svg.encode("utf-8") if isinstance(svg, str) else svg


def musicxml_to_pdf(xml_bytes: bytes) -> bytes:
    """Render MusicXML bytes to PDF bytes via the LilyPond toolchain.

    Pipeline: ``musicxml2ly`` (XML → LilyPond source) → ``lilypond``
    (LilyPond → PDF). Both run as subprocesses with a 60-second cap; a
    timeout or non-zero exit raises ``EngraveLocalError``. LilyPond's
    GPL stays isolated to the subprocess — no Python-side data sharing
    beyond passing the file path on the command line.
    """
    if not shutil.which("lilypond"):
        raise EngraveLocalError(
            "lilypond is not installed; cannot render PDF. "
            "Install via apt-get install lilypond (Linux) or "
            "brew install lilypond (macOS). The Dockerfile already "
            "installs it in the python-base stage."
        )
    if not shutil.which("musicxml2ly"):
        raise EngraveLocalError(
            "musicxml2ly is not on PATH (ships with lilypond). "
            "Confirm the lilypond install is complete."
        )

    with tempfile.TemporaryDirectory(prefix="engrave_local_") as tmpdir:
        tmp_root = Path(tmpdir)
        xml_path = tmp_root / "score.musicxml"
        ly_path = tmp_root / "score.ly"
        pdf_path = tmp_root / "score.pdf"

        xml_path.write_bytes(xml_bytes)

        # Step 1: MusicXML → LilyPond source
        try:
            subprocess.run(
                [
                    "musicxml2ly",
                    "--output", str(ly_path),
                    str(xml_path),
                ],
                check=True,
                timeout=_LILYPOND_TIMEOUT_SEC,
                capture_output=True,
                cwd=str(tmp_root),
            )
        except subprocess.TimeoutExpired as exc:
            raise EngraveLocalError(
                f"musicxml2ly timed out after {_LILYPOND_TIMEOUT_SEC}s — "
                f"likely malformed MusicXML."
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or b"").decode("utf-8", errors="replace")[:500]
            raise EngraveLocalError(
                f"musicxml2ly failed (exit {exc.returncode}): {stderr}"
            ) from exc

        if not ly_path.exists():
            raise EngraveLocalError(
                "musicxml2ly succeeded but produced no LilyPond source file."
            )

        # Step 2: LilyPond source → PDF
        try:
            subprocess.run(
                [
                    "lilypond",
                    "--pdf",
                    "--output", str(tmp_root / "score"),
                    str(ly_path),
                ],
                check=True,
                timeout=_LILYPOND_TIMEOUT_SEC,
                capture_output=True,
                cwd=str(tmp_root),
            )
        except subprocess.TimeoutExpired as exc:
            raise EngraveLocalError(
                f"lilypond timed out after {_LILYPOND_TIMEOUT_SEC}s — "
                f"likely a notation construct it can't resolve."
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or b"").decode("utf-8", errors="replace")[:500]
            raise EngraveLocalError(
                f"lilypond failed (exit {exc.returncode}): {stderr}"
            ) from exc

        if not pdf_path.exists():
            raise EngraveLocalError(
                "lilypond succeeded but produced no PDF file."
            )
        pdf_bytes = pdf_path.read_bytes()

    if not pdf_bytes or len(pdf_bytes) < 500 or not pdf_bytes.startswith(b"%PDF"):
        raise EngraveLocalError(
            f"lilypond produced suspiciously small / invalid PDF "
            f"(bytes={len(pdf_bytes)})"
        )
    return pdf_bytes


# ── Top-level convenience: PianoScore → (xml, pdf, svg) ────────────────


@dataclass
class LocalEngraveResult:
    """Bundle of every artifact the local engrave stack produced."""

    musicxml_bytes: bytes
    pdf_bytes: bytes | None = None
    svg_bytes: bytes | None = None
    features: EngravedFeatures = field(default_factory=EngravedFeatures)


def engrave_score_locally(
    score: PianoScore,
    expression: ExpressionMap | None = None,
    *,
    title: str | None = None,
    composer: str | None = None,
    render_pdf: bool = True,
    render_svg: bool = False,
) -> LocalEngraveResult:
    """One-shot helper: ``PianoScore`` → ``LocalEngraveResult``.

    PDF rendering is the default user-visible artifact. SVG is opt-in
    because the runner doesn't currently have a place to surface it; we
    keep the renderer wired up so future UI work can flip it on without
    touching the engrave pipeline.

    PDF / SVG failures degrade gracefully: ``musicxml_bytes`` always
    populates if the music21 step succeeds, so the runner can persist
    something even when the renderer subprocess is broken. Hard MusicXML
    failures still raise.
    """
    xml_bytes, features = score_to_musicxml(
        score, expression, title=title, composer=composer,
    )

    pdf_bytes: bytes | None = None
    svg_bytes: bytes | None = None

    if render_pdf:
        try:
            pdf_bytes = musicxml_to_pdf(xml_bytes)
        except EngraveLocalError as exc:
            log.warning("engrave_local: PDF render failed (continuing): %s", exc)

    if render_svg:
        try:
            svg_bytes = musicxml_to_svg(xml_bytes)
        except EngraveLocalError as exc:
            log.warning("engrave_local: SVG render failed (continuing): %s", exc)

    return LocalEngraveResult(
        musicxml_bytes=xml_bytes,
        pdf_bytes=pdf_bytes,
        svg_bytes=svg_bytes,
        features=features,
    )
