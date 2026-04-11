"""Hand-authored score fixtures for the engrave evaluation harness.

The ten fixtures below cover the cases that matter for engrave quality:
trivial baselines, voice/staff separation, accidentals, irregular meter,
multi-segment tempo, timing offsets, edge cases (empty LH, same-pitch
overlap). They are the foundation for the L1–L5 test layers described in
``docs/engrave-improvement-plan.md``.

Each fixture is built as a Pydantic model (``PianoScore`` or
``HumanizedPerformance``) and serialized to JSON under
``tests/fixtures/scores/``. The loader re-parses via Pydantic so schema
drift is caught as a validation error at test-collection time.

To regenerate all fixture JSON files after a contract change::

    python -m tests.fixtures._builders
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from backend.contracts import (
    SCHEMA_VERSION,
    Articulation,
    DynamicMarking,
    ExpressionMap,
    ExpressiveNote,
    HumanizedPerformance,
    PedalEvent,
    PianoScore,
    QualitySignal,
    ScoreChordEvent,
    ScoreMetadata,
    ScoreNote,
    TempoMapEntry,
)

FIXTURES_DIR = Path(__file__).parent / "scores"


# ---------------------------------------------------------------------------
# Small helpers to keep the builders below readable
# ---------------------------------------------------------------------------

def _tempo(bpm: float = 120.0) -> list[TempoMapEntry]:
    return [TempoMapEntry(time_sec=0.0, beat=0.0, bpm=bpm)]


def _meta(
    *,
    key: str = "C:major",
    time_signature: tuple[int, int] = (4, 4),
    tempo_map: list[TempoMapEntry] | None = None,
    chord_symbols: list[ScoreChordEvent] | None = None,
) -> ScoreMetadata:
    return ScoreMetadata(
        key=key,
        time_signature=time_signature,
        tempo_map=tempo_map or _tempo(),
        difficulty="intermediate",
        sections=[],
        chord_symbols=chord_symbols or [],
    )


def _rh(idx: int, pitch: int, onset: float, duration: float,
        *, velocity: int = 80, voice: int = 1) -> ScoreNote:
    return ScoreNote(
        id=f"rh-{idx:04d}",
        pitch=pitch,
        onset_beat=onset,
        duration_beat=duration,
        velocity=velocity,
        voice=voice,
    )


def _lh(idx: int, pitch: int, onset: float, duration: float,
        *, velocity: int = 70, voice: int = 1) -> ScoreNote:
    return ScoreNote(
        id=f"lh-{idx:04d}",
        pitch=pitch,
        onset_beat=onset,
        duration_beat=duration,
        velocity=velocity,
        voice=voice,
    )


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def build_single_note() -> PianoScore:
    """Trivial baseline — one RH C4 whole note."""
    return PianoScore(
        schema_version=SCHEMA_VERSION,
        right_hand=[_rh(0, pitch=60, onset=0.0, duration=4.0)],
        left_hand=[],
        metadata=_meta(),
    )


def build_c_major_scale() -> PianoScore:
    """RH one-octave C major scale in quarters; LH two whole-note C3 pedal tones."""
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]  # C4..C5
    rh = [_rh(i, pitch=p, onset=float(i), duration=1.0) for i, p in enumerate(pitches)]
    lh = [
        _lh(0, pitch=48, onset=0.0, duration=4.0),
        _lh(1, pitch=48, onset=4.0, duration=4.0),
    ]
    return PianoScore(
        schema_version=SCHEMA_VERSION,
        right_hand=rh,
        left_hand=lh,
        metadata=_meta(),
    )


def build_two_hand_chordal() -> PianoScore:
    """Four quarter-note triads over octave bass — exercises <staff> separation."""
    # RH: C-E-G, F-A-C, G-B-D, C-E-G (I – IV – V – I)
    triads = [
        (60, 64, 67),  # C major
        (65, 69, 72),  # F major
        (67, 71, 74),  # G major
        (60, 64, 67),  # C major
    ]
    rh: list[ScoreNote] = []
    idx = 0
    for beat, triad in enumerate(triads):
        for p in triad:
            rh.append(_rh(idx, pitch=p, onset=float(beat), duration=1.0))
            idx += 1

    # LH: octaves on the root of each chord (e.g. C2 + C3)
    roots = [36, 41, 43, 36]
    lh: list[ScoreNote] = []
    idx = 0
    for beat, root in enumerate(roots):
        lh.append(_lh(idx, pitch=root, onset=float(beat), duration=1.0))
        idx += 1
        lh.append(_lh(idx, pitch=root + 12, onset=float(beat), duration=1.0))
        idx += 1

    return PianoScore(
        schema_version=SCHEMA_VERSION,
        right_hand=rh,
        left_hand=lh,
        metadata=_meta(),
    )


def build_bach_invention_excerpt() -> PianoScore:
    """Eight quarters of two-voice counterpoint in the RH — the voice-handling fixture.

    Not a real Bach quote; just two independent lines in the same staff
    with different rhythms and pitches so the voice=1/voice=2 assignment
    is unambiguous and any collapse to a single voice is visible.
    """
    # Voice 1 — upper line, quarter notes ascending
    v1_pitches = [72, 74, 76, 77, 79, 77, 76, 74]  # C5..
    voice1 = [
        _rh(i, pitch=p, onset=float(i), duration=1.0, voice=1)
        for i, p in enumerate(v1_pitches)
    ]
    # Voice 2 — lower line, half-note pulses
    v2_pitches = [60, 64, 65, 64]
    voice2 = [
        _rh(100 + i, pitch=p, onset=float(i * 2), duration=2.0, voice=2)
        for i, p in enumerate(v2_pitches)
    ]
    return PianoScore(
        schema_version=SCHEMA_VERSION,
        right_hand=voice1 + voice2,
        left_hand=[],
        metadata=_meta(time_signature=(4, 4)),
    )


def build_jazz_voicings() -> PianoScore:
    """Chromatic walking bass + 7th-chord shells above — exercises accidentals."""
    # LH walking bass: C2, C#2, D2, D#2, E2, F2, F#2, G2
    lh = [
        _lh(i, pitch=36 + i, onset=float(i), duration=1.0)
        for i in range(8)
    ]
    # RH shells — 3rd and 7th of each chord. Cmaj7, C#7, Dm7, D#dim7,
    # Emaj7, F7, F#m7b5, G7. Just two voices per hit.
    shells = [
        (64, 71),  # Cmaj7: E + B
        (61, 70),  # C#7: C#(Db) + B(Bb as 7)
        (65, 72),  # Dm7: F + C
        (66, 72),  # D#dim7: F#(Gb) + C
        (68, 75),  # Emaj7: G#(Ab) + D#(Eb)
        (69, 75),  # F7: A + Eb
        (69, 76),  # F#m7b5: A + E
        (71, 77),  # G7: B + F
    ]
    rh: list[ScoreNote] = []
    idx = 0
    for beat, (low, high) in enumerate(shells):
        rh.append(_rh(idx, pitch=low, onset=float(beat), duration=1.0))
        idx += 1
        rh.append(_rh(idx, pitch=high, onset=float(beat), duration=1.0))
        idx += 1
    return PianoScore(
        schema_version=SCHEMA_VERSION,
        right_hand=rh,
        left_hand=lh,
        metadata=_meta(),
    )


def build_seven_eight() -> PianoScore:
    """7/8 irregular-meter fixture — tests time-sig propagation & grouping."""
    # Seven eighth-note pulses in the RH (2+2+3 grouping).
    durations = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    pitches = [60, 62, 64, 65, 67, 69, 71]
    rh: list[ScoreNote] = []
    onset = 0.0
    for i, (p, d) in enumerate(zip(pitches, durations)):
        rh.append(_rh(i, pitch=p, onset=onset, duration=d))
        onset += d

    # LH: one dotted-half per 7/8 bar, pinned to the downbeat
    lh = [_lh(0, pitch=48, onset=0.0, duration=3.5)]

    return PianoScore(
        schema_version=SCHEMA_VERSION,
        right_hand=rh,
        left_hand=lh,
        metadata=_meta(time_signature=(7, 8)),
    )


def build_tempo_change() -> PianoScore:
    """Multi-segment tempo map (120 → 90 halfway through)."""
    tempo_map = [
        TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0),
        TempoMapEntry(time_sec=2.0, beat=4.0, bpm=90.0),
    ]
    rh = [
        _rh(i, pitch=60 + (i * 2), onset=float(i), duration=1.0)
        for i in range(8)
    ]
    return PianoScore(
        schema_version=SCHEMA_VERSION,
        right_hand=rh,
        left_hand=[],
        metadata=_meta(tempo_map=tempo_map),
    )


def build_empty_left_hand() -> PianoScore:
    """RH-only — catches the no-note backup / empty-staff logic in engrave."""
    rh = [
        _rh(0, pitch=67, onset=0.0, duration=1.0),
        _rh(1, pitch=69, onset=1.0, duration=1.0),
        _rh(2, pitch=71, onset=2.0, duration=1.0),
        _rh(3, pitch=72, onset=3.0, duration=1.0),
    ]
    return PianoScore(
        schema_version=SCHEMA_VERSION,
        right_hand=rh,
        left_hand=[],
        metadata=_meta(),
    )


def build_overlapping_same_pitch() -> PianoScore:
    """Two overlapping C4s — exercises the same-pitch overlap resolver in engrave."""
    rh = [
        _rh(0, pitch=60, onset=0.0, duration=2.0),
        _rh(1, pitch=60, onset=1.0, duration=2.0),
    ]
    return PianoScore(
        schema_version=SCHEMA_VERSION,
        right_hand=rh,
        left_hand=[],
        metadata=_meta(),
    )


def build_mislabeled_key() -> PianoScore:
    """F# minor piece mislabeled as C major — the PR-12 KS override fixture.

    Every RH note is drawn from the F# natural minor scale
    (F#, G#, A, B, C#, D, E) with F# as the tonal center. The LH
    reinforces F# as bass. The metadata key lies: it claims
    ``C:major``. The Krumhansl-Schmuckler analyzer in
    ``_resolve_key_signature`` should detect the real F# minor tonic,
    report correlation ≥0.85, and override to F# minor before the key
    signature reaches ``<key><fifths>`` in MusicXML.

    Expected MusicXML ``<fifths>`` value after override: ``3``
    (F# minor has three sharps). Without the override it would be ``0``
    (C major, no sharps or flats), and every F#/G#/C# in the piece
    would come out as an explicit accidental.
    """
    # F# minor scale + tonic arpeggios, thick enough that KS returns
    # a solid correlation well above the 0.80 override floor. Tonic
    # F#4 = pitch 66; scale is F#, G#, A, B, C#, D, E (natural minor).
    scale = [66, 68, 69, 71, 73, 74, 76]  # F#4..E5
    tonic_arp = [66, 69, 73, 78]          # F# A C# F# (i triad, emphasize tonic)
    # Repeat scale up+down twice, separated by i-chord arpeggios, so
    # the pitch histogram is emphatically weighted toward the F# minor
    # scale degrees. Also rules out the "relative major" (A:major)
    # trap — plain scale notes alone can correlate just as well with
    # the parallel major.
    pattern = (
        scale + list(reversed(scale))
        + tonic_arp + tonic_arp
        + scale + list(reversed(scale))
        + tonic_arp + [66, 66, 66, 66]     # hammer the tonic home
    )
    rh = [
        _rh(i, pitch=p, onset=float(i) * 0.5, duration=0.5)
        for i, p in enumerate(pattern)
    ]
    # LH reinforces F# as bass — F#2 pedal tones under every phrase so
    # the analyzer sees the real tonic loud and clear.
    lh_beats = 4
    lh = [
        _lh(i, pitch=42, onset=float(i * lh_beats), duration=float(lh_beats))
        for i in range(len(rh) // (lh_beats * 2))
    ] or [_lh(0, pitch=42, onset=0.0, duration=float(lh_beats))]
    return PianoScore(
        schema_version=SCHEMA_VERSION,
        right_hand=rh,
        left_hand=lh,
        metadata=_meta(key="C:major"),  # the lie
    )


def build_chord_symbols() -> PianoScore:
    """Mixed-quality chord symbols — the PR-11 filter-gate fixture.

    Seven input chord symbols that exercise every branch of
    ``_attach_chord_symbols``:

    1. ``C:maj7`` @ beat 0, conf 0.95 — clean, parses, ≥3 pitches → **rendered**.
    2. ``Dm7``    @ beat 1, conf 0.90 — no colon, parses, ≥3 pitches → **rendered**.
    3. ``F:maj7`` @ beat 2, conf 0.85 — colon form, ≥3 pitches → **rendered**.
    4. ``G5``     @ beat 3, conf 0.80 — pitch-octave shape
       (``^[A-G][#b]?\\d+$``) → **dropped (shape gate)**.
    5. ``C:maj``  @ beat 4, conf 0.30 — would parse, but below the
       0.5 confidence threshold → **dropped (confidence gate)**.
    6. ``???``    @ beat 5, conf 0.90 — unparseable garbage → **dropped (parse gate)**.
    7. ``C5``     @ beat 6, conf 0.95 — pitch-octave shape → **dropped (shape gate)**.

    Expected rendered count: **3**. Note the deliberate choice not to
    test ``G7`` / ``C7`` / ``C9`` — plan phase 3.2 accepts that the
    ``^[A-G][#b]?\\d+$`` shape gate is aggressive enough to kill
    dominant-sevenths written as bare ``G7``, because the audio
    transcriber's false positives dominate the legitimate uses. Callers
    that want dominant sevenths through this gate should emit Harte
    colon form (``G:7``) — which the filter strips to ``G7`` for
    parsing but which still *lexically* fails the shape check.

    Paired with a simple C major melody so the test can focus on
    harmony output, not note plumbing.
    """
    # Two-bar single-line RH so there's something to hang the symbols on.
    rh = [
        _rh(i, pitch=p, onset=float(i), duration=1.0)
        for i, p in enumerate([60, 62, 64, 65, 67, 69, 71, 72])
    ]
    chord_symbols = [
        ScoreChordEvent(beat=0.0, duration_beat=1.0, label="C:maj7", root=60, confidence=0.95),
        ScoreChordEvent(beat=1.0, duration_beat=1.0, label="Dm7",    root=62, confidence=0.90),
        ScoreChordEvent(beat=2.0, duration_beat=1.0, label="F:maj7", root=65, confidence=0.85),
        ScoreChordEvent(beat=3.0, duration_beat=1.0, label="G5",     root=67, confidence=0.80),
        ScoreChordEvent(beat=4.0, duration_beat=1.0, label="C:maj",  root=60, confidence=0.30),
        ScoreChordEvent(beat=5.0, duration_beat=1.0, label="???",    root=60, confidence=0.90),
        ScoreChordEvent(beat=6.0, duration_beat=1.0, label="C5",     root=60, confidence=0.95),
    ]
    return PianoScore(
        schema_version=SCHEMA_VERSION,
        right_hand=rh,
        left_hand=[],
        metadata=_meta(chord_symbols=chord_symbols),
    )


def build_humanized_with_offsets() -> HumanizedPerformance:
    """C major scale wrapped as a HumanizedPerformance with timing offsets + a pedal event.

    This is **the timing-bug fixture** — ``timing_offset_ms`` values are
    non-zero and alternate in sign so the round-trip test can verify
    whether the shift is onset-only or a whole-note translation (see
    ``engrave.py:87-88``). Also includes one sustain-pedal event so L1
    can verify pedal events reach the MIDI output.
    """
    base = build_c_major_scale()

    # Alternating ±20ms offsets across the RH scale notes; LH whole notes
    # stay at 0 so any timing drift in the LH clearly comes from engrave,
    # not the fixture.
    timing = [20.0, -20.0, 15.0, -15.0, 10.0, -10.0, 5.0, -5.0]
    expressive_notes = [
        ExpressiveNote(
            score_note_id=n.id,
            pitch=n.pitch,
            onset_beat=n.onset_beat,
            duration_beat=n.duration_beat,
            velocity=n.velocity,
            hand="rh",
            voice=n.voice,
            timing_offset_ms=t,
            velocity_offset=0,
        )
        for n, t in zip(base.right_hand, timing)
    ]
    expressive_notes.extend(
        ExpressiveNote(
            score_note_id=n.id,
            pitch=n.pitch,
            onset_beat=n.onset_beat,
            duration_beat=n.duration_beat,
            velocity=n.velocity,
            hand="lh",
            voice=n.voice,
            timing_offset_ms=0.0,
            velocity_offset=0,
        )
        for n in base.left_hand
    )

    expression = ExpressionMap(
        dynamics=[],
        articulations=[],
        pedal_events=[
            PedalEvent(onset_beat=0.0, offset_beat=8.0, type="sustain"),
        ],
        tempo_changes=[],
    )

    return HumanizedPerformance(
        schema_version=SCHEMA_VERSION,
        expressive_notes=expressive_notes,
        expression=expression,
        score=base,
        quality=QualitySignal(overall_confidence=0.9, warnings=[]),
    )


def build_humanized_with_expression() -> HumanizedPerformance:
    """The "stop dropping data" fixture — dynamics, pedal variants, fermata.

    Exercises every branch of plan PR-5 (phase 1.3–1.6): a static
    dynamic (``p``) at bar 1 and a hairpin (``crescendo``) spanning bar
    2, sustain + sostenuto + una_corda pedal events on the LH, and a
    fermata articulation on the last RH note.
    """
    base = build_c_major_scale()

    expressive_notes = [
        ExpressiveNote(
            score_note_id=n.id,
            pitch=n.pitch,
            onset_beat=n.onset_beat,
            duration_beat=n.duration_beat,
            velocity=n.velocity,
            hand="rh",
            voice=n.voice,
            timing_offset_ms=0.0,
            velocity_offset=0,
        )
        for n in base.right_hand
    ]
    expressive_notes.extend(
        ExpressiveNote(
            score_note_id=n.id,
            pitch=n.pitch,
            onset_beat=n.onset_beat,
            duration_beat=n.duration_beat,
            velocity=n.velocity,
            hand="lh",
            voice=n.voice,
            timing_offset_ms=0.0,
            velocity_offset=0,
        )
        for n in base.left_hand
    )

    # Pedal offsets are pulled just inside the final beat. music21 drops
    # direction elements inserted at the exact barline past the last note,
    # so "Ped. / *" lifts at the very end of the piece never reach MusicXML.
    # A ~16th-note release is still audible in the MIDI output.
    last_rh_id = base.right_hand[-1].id
    expression = ExpressionMap(
        dynamics=[
            DynamicMarking(beat=0.0, type="p"),
            DynamicMarking(beat=4.0, type="crescendo", span_beats=3.5),
        ],
        articulations=[
            Articulation(beat=7.0, hand="rh", score_note_id=last_rh_id, type="fermata"),
        ],
        pedal_events=[
            PedalEvent(onset_beat=0.0, offset_beat=3.5, type="sustain"),
            PedalEvent(onset_beat=4.0, offset_beat=7.5, type="sostenuto"),
            PedalEvent(onset_beat=0.0, offset_beat=7.5, type="una_corda"),
        ],
        tempo_changes=[],
    )

    return HumanizedPerformance(
        schema_version=SCHEMA_VERSION,
        expressive_notes=expressive_notes,
        expression=expression,
        score=base,
        quality=QualitySignal(overall_confidence=0.9, warnings=[]),
    )


# ---------------------------------------------------------------------------
# Registry + load + regenerate
# ---------------------------------------------------------------------------

_BUILDERS: dict[str, Callable[[], PianoScore | HumanizedPerformance]] = {
    "single_note": build_single_note,
    "c_major_scale": build_c_major_scale,
    "two_hand_chordal": build_two_hand_chordal,
    "bach_invention_excerpt": build_bach_invention_excerpt,
    "jazz_voicings": build_jazz_voicings,
    "seven_eight": build_seven_eight,
    "tempo_change": build_tempo_change,
    "humanized_with_offsets": build_humanized_with_offsets,
    "humanized_with_expression": build_humanized_with_expression,
    "empty_left_hand": build_empty_left_hand,
    "overlapping_same_pitch": build_overlapping_same_pitch,
    "chord_symbols": build_chord_symbols,
    "mislabeled_key": build_mislabeled_key,
}

FIXTURE_NAMES: tuple[str, ...] = tuple(_BUILDERS.keys())

# Fixtures that build a HumanizedPerformance rather than a raw PianoScore.
_HUMANIZED_FIXTURES: frozenset[str] = frozenset({
    "humanized_with_offsets",
    "humanized_with_expression",
})


def load_score_fixture(name: str) -> PianoScore | HumanizedPerformance:
    """Load a committed fixture JSON and re-validate via Pydantic.

    Parsing through the model catches contract drift at test-collection
    time — if a field is renamed or removed, the fixture is flagged
    immediately instead of silently producing garbage MusicXML.
    """
    path = FIXTURES_DIR / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Score fixture {name!r} not found at {path}. "
            "Run `python -m tests.fixtures._builders` to regenerate.",
        )
    raw = json.loads(path.read_text())
    model = HumanizedPerformance if name in _HUMANIZED_FIXTURES else PianoScore
    return model.model_validate(raw)


def regenerate_all() -> None:
    """Write every builder's output to ``tests/fixtures/scores/<name>.json``."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    for name, builder in _BUILDERS.items():
        model = builder()
        path = FIXTURES_DIR / f"{name}.json"
        path.write_text(model.model_dump_json(indent=2) + "\n")
        print(f"wrote {path.relative_to(FIXTURES_DIR.parent.parent)}")


if __name__ == "__main__":
    regenerate_all()
