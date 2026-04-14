"""Engrave quality harness — L1 (MIDI round-trip) + L2 (notation lints).

This is the first layer of the evaluation harness described in
``docs/engrave-improvement-plan.md`` Phase 0.2. It runs every score
fixture in ``tests/fixtures/scores/`` through ``_engrave_sync`` and:

- **L1 — MIDI round-trip.** Parses the emitted MIDI with ``pretty_midi``
  and asserts that every fixture note is present in the output (by
  pitch + attack count). Catches dropped notes and wrong-pitch
  regressions.
- **L2 — Notation quality lints.** Parses the emitted MusicXML with
  ``lxml`` and runs structural xpath checks: ``<voice>`` in range,
  ``<divisions>`` ≤ 480, pitches within the 88-key piano range, note
  count consistent with the fixture.

A dedicated test pins the onset-only semantics of ``timing_offset_ms``
against the ``humanized_with_offsets`` fixture (see plan Phase 1.2).
"""
from __future__ import annotations

import io
from collections import Counter

import pytest

from backend.contracts import HumanizedPerformance, PianoScore, beat_to_sec
from backend.services.engrave import _engrave_sync
from tests.fixtures import FIXTURE_NAMES, load_score_fixture

# ---------------------------------------------------------------------------
# MusicXML step → semitone table for pitch decoding
# ---------------------------------------------------------------------------

_STEP_TO_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _musicxml_pitch_to_midi(pitch_elem) -> int:
    step = pitch_elem.findtext("step")
    octave = int(pitch_elem.findtext("octave"))
    alter_text = pitch_elem.findtext("alter")
    alter = int(float(alter_text)) if alter_text else 0
    return (octave + 1) * 12 + _STEP_TO_SEMITONE[step] + alter


def _expected_note_pitches(fixture) -> list[int]:
    """Return the MIDI pitches of every note in a fixture.

    For a ``PianoScore`` this is ``rh + lh``. For a
    ``HumanizedPerformance`` it's ``expressive_notes`` — same count, but
    with the timing offsets baked in.
    """
    if isinstance(fixture, HumanizedPerformance):
        return [n.pitch for n in fixture.expressive_notes]
    if isinstance(fixture, PianoScore):
        return [n.pitch for n in fixture.right_hand] + [n.pitch for n in fixture.left_hand]
    raise TypeError(f"Unexpected fixture type: {type(fixture).__name__}")


# ---------------------------------------------------------------------------
# Engrave-once-per-fixture cache
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engraved_artifacts() -> dict[str, tuple[bytes, bytes]]:
    """Run each fixture through ``_engrave_sync`` once per test module.

    Returns ``{name: (musicxml_bytes, midi_bytes)}``. Running engrave is
    cheap but music21's importer is slow (~1s) and we don't want to pay
    it for every parametrized test.
    """
    cache: dict[str, tuple[bytes, bytes]] = {}
    for name in FIXTURE_NAMES:
        fixture = load_score_fixture(name)
        _pdf, musicxml, midi, _chord_count = _engrave_sync(fixture, title=name, composer="test")
        cache[name] = (musicxml, midi)
    return cache


# ---------------------------------------------------------------------------
# L1 — MIDI round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_l1_midi_round_trip(name: str, engraved_artifacts):
    """Every fixture note round-trips to MIDI with the right pitch + count."""
    import pretty_midi

    fixture = load_score_fixture(name)
    _, midi_bytes = engraved_artifacts[name]
    assert midi_bytes, f"engrave emitted empty MIDI for fixture {name!r}"

    midi = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    actual_pitches = [note.pitch for inst in midi.instruments for note in inst.notes]
    expected_pitches = _expected_note_pitches(fixture)

    actual_counter = Counter(actual_pitches)
    expected_counter = Counter(expected_pitches)

    # The same-pitch overlap resolver in engrave.py:94-103 may collapse
    # two overlapping attacks into a single sustained note, so for that
    # specific fixture we only assert "at least one attack per pitch."
    # Every other fixture is strict.
    if name == "overlapping_same_pitch":
        assert set(actual_counter.keys()) == set(expected_counter.keys()), (
            f"{name}: pitch set mismatch — expected {sorted(expected_counter)} "
            f"got {sorted(actual_counter)}"
        )
        assert sum(actual_counter.values()) >= 1
    else:
        assert actual_counter == expected_counter, (
            f"{name}: MIDI pitch counts diverge from the fixture.\n"
            f"  expected: {sorted(expected_counter.items())}\n"
            f"  got:      {sorted(actual_counter.items())}"
        )


def test_l1_humanized_timing_offset_is_onset_only(engraved_artifacts):
    """``timing_offset_ms`` is an onset-only nudge — release stays fixed.

    The fixture's first RH note is C4 at beat 0, duration 1 beat, with
    ``timing_offset_ms = +20``. At 120 BPM one beat is 0.5 s, so an
    onset-only shift produces ``onset = 0.020 s`` and
    ``offset = 0.500 s`` (duration compressed to 0.480 s).

    Mirrors ``humanize._humanize_timing``, which models downbeat
    anticipation / backbeat push as attack-time gestures only.
    """
    import pretty_midi

    fixture = load_score_fixture("humanized_with_offsets")
    assert isinstance(fixture, HumanizedPerformance)

    _, midi_bytes = engraved_artifacts["humanized_with_offsets"]
    midi = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    notes_by_pitch = {}
    for inst in midi.instruments:
        for n in inst.notes:
            notes_by_pitch.setdefault(n.pitch, []).append(n)

    # First RH note: pitch 60 (C4), beat 0, timing_offset_ms=+20, duration=1 beat
    rh0 = fixture.expressive_notes[0]
    assert rh0.pitch == 60
    assert rh0.timing_offset_ms == pytest.approx(20.0)

    c4_notes = sorted(notes_by_pitch[60], key=lambda n: n.start)
    first = c4_notes[0]

    tempo_map = fixture.score.metadata.tempo_map
    unshifted_offset_sec = beat_to_sec(rh0.onset_beat + rh0.duration_beat, tempo_map)
    assert first.end == pytest.approx(unshifted_offset_sec, abs=0.001), (
        f"expected offset unchanged at {unshifted_offset_sec:.3f}s, got {first.end:.3f}s "
        "(timing_offset_ms leaked into the release boundary)"
    )


def test_l1_humanized_pedal_reaches_midi(engraved_artifacts):
    """Sustain pedal events must emit CC64 on/off in the output MIDI."""
    import pretty_midi

    _, midi_bytes = engraved_artifacts["humanized_with_offsets"]
    midi = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    cc64 = [
        cc
        for inst in midi.instruments
        for cc in inst.control_changes
        if cc.number == 64
    ]
    assert any(cc.value >= 64 for cc in cc64), "no sustain pedal ON (CC64 ≥ 64)"
    assert any(cc.value < 64 for cc in cc64), "no sustain pedal OFF (CC64 < 64)"


def test_l1_humanized_expression_emits_all_pedal_ccs(engraved_artifacts):
    """Plan phase 1.5 — sostenuto → CC66, una corda → CC67 in addition to sustain → CC64.

    The ``humanized_with_expression`` fixture carries one of each pedal
    type; the engraver must write on/off edges for all three controllers.
    """
    import pretty_midi

    _, midi_bytes = engraved_artifacts["humanized_with_expression"]
    midi = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    by_cc: dict[int, list[int]] = {}
    for inst in midi.instruments:
        for cc in inst.control_changes:
            by_cc.setdefault(cc.number, []).append(cc.value)

    for cc_num, label in ((64, "sustain"), (66, "sostenuto"), (67, "una_corda")):
        assert cc_num in by_cc, f"{label}: no CC{cc_num} events in MIDI"
        values = by_cc[cc_num]
        assert any(v >= 64 for v in values), f"{label}: no CC{cc_num} ON edge"
        assert any(v < 64 for v in values), f"{label}: no CC{cc_num} OFF edge"


# ---------------------------------------------------------------------------
# L2 — Notation quality lints
# ---------------------------------------------------------------------------

_PIANO_MIN_MIDI = 21   # A0
_PIANO_MAX_MIDI = 108  # C8


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_l2_voice_numbers_in_range(name: str, engraved_artifacts):
    """Every ``<voice>`` element must be an integer in [1, 4].

    music21 omits ``<voice>`` entirely for single-voice parts, so a
    missing element is also fine — we only complain about values that
    are present and out of range.
    """
    from lxml import etree

    musicxml, _ = engraved_artifacts[name]
    root = etree.fromstring(musicxml)
    voices = [int(e.text) for e in root.iter("voice")]
    bad = [v for v in voices if not 1 <= v <= 4]
    assert not bad, f"{name}: out-of-range <voice> values: {sorted(set(bad))}"


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_l2_divisions_is_twelve(name: str, engraved_artifacts):
    """``<divisions>`` must be exactly 12 per quarter.

    PR-9 (plan phase 2.4) sets ``music21.defaults.divisionsPerQuarter=12``
    before export so the grid is fixed by construction — 16th = 3
    divisions, 8th = 6, triplet-8th = 4, quarter = 12. Prior to PR-9 the
    value was music21's shipped default of 10080, which OSMD cannot
    consume; this test would catch either regression.
    """
    from lxml import etree

    musicxml, _ = engraved_artifacts[name]
    root = etree.fromstring(musicxml)
    divisions = [int(e.text) for e in root.iter("divisions")]
    assert divisions, f"{name}: no <divisions> element in MusicXML"
    assert set(divisions) == {12}, (
        f"{name}: expected <divisions>12</divisions>, got {sorted(set(divisions))}"
    )


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_l2_pitches_in_piano_range(name: str, engraved_artifacts):
    """Every pitched ``<note>`` must land on an 88-key piano (A0..C8)."""
    from lxml import etree

    musicxml, _ = engraved_artifacts[name]
    root = etree.fromstring(musicxml)
    out_of_range: list[int] = []
    for pitch_elem in root.iter("pitch"):
        midi = _musicxml_pitch_to_midi(pitch_elem)
        if not _PIANO_MIN_MIDI <= midi <= _PIANO_MAX_MIDI:
            out_of_range.append(midi)
    assert not out_of_range, f"{name}: pitches outside piano range: {sorted(set(out_of_range))}"


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_l2_note_count_matches_fixture(name: str, engraved_artifacts):
    """MusicXML note-attack count agrees with the fixture's note count.

    Tied continuations (``<tie type="stop"/>``) are excluded — they
    represent the same attack that spans a barline. Rests are skipped.
    """
    from lxml import etree

    fixture = load_score_fixture(name)
    expected = len(_expected_note_pitches(fixture))

    musicxml, _ = engraved_artifacts[name]
    root = etree.fromstring(musicxml)

    attack_count = 0
    for note in root.iter("note"):
        if note.find("rest") is not None:
            continue
        tie_stops = [t for t in note.findall("tie") if t.get("type") == "stop"]
        if tie_stops and len(tie_stops) == len(note.findall("tie")):
            # Pure tie-continuation (no concurrent start) — not a new attack.
            continue
        attack_count += 1

    assert attack_count == expected, (
        f"{name}: MusicXML has {attack_count} note attacks, fixture has {expected}"
    )


# ---------------------------------------------------------------------------
# L2 — "Stop dropping data" lints (plan phase 1.3–1.6 / PR-5)
# ---------------------------------------------------------------------------


def test_l2_humanized_dynamics_rendered(engraved_artifacts):
    """Static dynamic + hairpin from the expression map reach MusicXML.

    The ``humanized_with_expression`` fixture carries a ``p`` at beat 0
    and a ``crescendo`` at beat 4. The static mark becomes a
    ``<dynamics>`` element; the hairpin becomes a ``<words>cresc.</words>``
    text direction.
    """
    from lxml import etree

    musicxml, _ = engraved_artifacts["humanized_with_expression"]
    root = etree.fromstring(musicxml)

    dyn_children = [c.tag for dyn in root.iter("dynamics") for c in dyn]
    assert "p" in dyn_children, (
        f"expected <dynamics><p/></dynamics>; got dynamics children {dyn_children}"
    )

    words = [w.text for w in root.iter("words")]
    assert "cresc." in words, f"expected 'cresc.' text direction; got {words}"


def test_l2_humanized_pedal_text_rendered(engraved_artifacts):
    """Sustain / sostenuto / una corda all surface as MusicXML text directions.

    Sustain uses the standard "Ped." / "*" pair; sostenuto and una corda
    use descriptive labels because MusicXML has no dedicated glyphs.
    """
    from lxml import etree

    musicxml, _ = engraved_artifacts["humanized_with_expression"]
    root = etree.fromstring(musicxml)
    words = [w.text for w in root.iter("words")]

    for expected in ("Ped.", "*", "Sost. Ped.", "una corda", "tre corde"):
        assert expected in words, (
            f"expected pedal word {expected!r} in MusicXML; got {words}"
        )


def test_l2_humanized_fermata_rendered(engraved_artifacts):
    """A ``fermata`` articulation emits a ``<fermata>`` element in <notations>."""
    from lxml import etree

    musicxml, _ = engraved_artifacts["humanized_with_expression"]
    root = etree.fromstring(musicxml)
    fermatas = list(root.iter("fermata"))
    assert fermatas, "no <fermata> element in rendered MusicXML"


# ---------------------------------------------------------------------------
# L2 — Grand-staff structure (plan phase 2.3 / PR-8)
# ---------------------------------------------------------------------------


def _count_part_notes(musicxml: bytes, part_index: int) -> int:
    """Count pitched ``<note>`` elements in the N-th ``<part>`` (0-based)."""
    from lxml import etree

    parts = etree.fromstring(musicxml).findall("part")
    if part_index >= len(parts):
        return 0
    count = 0
    for note in parts[part_index].iter("note"):
        if note.find("rest") is not None:
            continue
        count += 1
    return count


def test_l2_grand_staff_two_parts_braced(engraved_artifacts):
    """Piano scores render as two ``<part>`` elements joined by a brace group.

    We emit MusicXML in the "two parts + part-group(brace)" idiom instead
    of the "one part + <staves>2" idiom. Rationale: ``musicxml2ly`` (the
    MusicXML → LilyPond bridge) does not reliably honor the one-part /
    multi-staff encoding, so LilyPond renders both staves with a treble
    clef and left-hand notes end up as ledger-line stacks below the
    staff. Two-part + brace is handled correctly by LilyPond, MuseScore,
    Verovio, and OSMD.
    """
    from lxml import etree

    # two_hand_chordal is the canonical grand-staff fixture: 12 RH notes
    # (triads) and 8 LH notes (octaves).
    musicxml, _ = engraved_artifacts["two_hand_chordal"]
    root = etree.fromstring(musicxml)

    parts = root.findall("part")
    assert len(parts) == 2, f"expected two parts (RH + LH), got {len(parts)}"

    # No <staves> element under any part — that's the old one-part idiom.
    assert not list(root.iter("staves")), (
        "unexpected <staves> element — two-part encoding should not declare it"
    )

    # part-list carries a brace-group wrapping both score-parts.
    part_list = root.find("part-list")
    assert part_list is not None
    groups = part_list.findall("part-group")
    assert groups, "expected a <part-group> in <part-list>"
    brace_starts = [
        g for g in groups
        if g.get("type") == "start" and g.findtext("group-symbol") == "brace"
    ]
    assert brace_starts, "expected a brace-type part-group in part-list"
    assert brace_starts[0].findtext("group-name") == "Piano"

    # Clef sanity: part 1 = treble (G on line 2), part 2 = bass (F on line 4).
    def clef_of(part_elem) -> tuple[str, str]:
        clef = part_elem.find("measure/attributes/clef")
        assert clef is not None
        return (clef.findtext("sign") or "", clef.findtext("line") or "")

    assert clef_of(parts[0]) == ("G", "2"), f"RH clef: {clef_of(parts[0])}"
    assert clef_of(parts[1]) == ("F", "4"), f"LH clef: {clef_of(parts[1])}"

    # 12 RH notes on part 1, 8 LH notes on part 2 (same counts as before).
    assert _count_part_notes(musicxml, 0) == 12
    assert _count_part_notes(musicxml, 1) == 8


def test_l2_two_voices_preserved_on_same_staff(engraved_artifacts):
    """Plan phase 3.1 / PR-10: two-voice RH content keeps both voices.

    The ``bach_invention_excerpt`` fixture carries 8 RH notes on
    ``voice=1`` (upper line, quarters) and 4 RH notes on ``voice=2``
    (lower line, half notes). Before PR-10 the engrave sanitizer
    collapsed every ``<voice>`` tag to 1, destroying music21's
    stems-up-melody / stems-down-accompaniment separation. The fix is
    explicit music21 ``Voice`` sub-streams + a post-``makeNotation``
    id-rename from the 0-indexed integers music21 stamps during
    measure construction back to the MusicXML-valid ``1``/``2``.

    What we check: (a) both voice numbers are present, (b) at least
    one ``<backup>`` element exists (music21 emits ``<backup>`` before
    switching from voice 1 to voice 2 within a measure), and (c) the
    note counts per voice match the fixture — allowing one extra
    voice-2 attack for the tie-continuation music21 inserts where a
    half note bridges the barline.
    """
    from lxml import etree

    musicxml, _ = engraved_artifacts["bach_invention_excerpt"]
    root = etree.fromstring(musicxml)

    voice_counts: dict[str, int] = {}
    for note in root.iter("note"):
        if note.find("rest") is not None:
            continue
        v = note.findtext("voice")
        if v is not None:
            voice_counts[v] = voice_counts.get(v, 0) + 1

    assert "1" in voice_counts and "2" in voice_counts, (
        f"expected both <voice>1</voice> and <voice>2</voice>; got {voice_counts}"
    )
    assert voice_counts["1"] == 8, f"voice 1 count {voice_counts['1']} != 8"
    # Voice 2 = 4 half notes in the fixture; expect 4 or 5 (tie split at barline).
    assert voice_counts["2"] in (4, 5), f"voice 2 count {voice_counts['2']} not in (4, 5)"
    assert "0" not in voice_counts, (
        "<voice>0</voice> is invalid MusicXML; sanitizer should have remapped it"
    )

    backups = list(root.iter("backup"))
    assert backups, "expected <backup> elements separating voice 1 from voice 2"


def test_l2_rh_only_fixture_still_braced(engraved_artifacts):
    """An empty-LH fixture still emits a two-part braced grand staff.

    The LH ``<part>`` will contain measures with only rests, but the
    ``<part-group>`` (brace) must still be present so renderers draw a
    grand staff with the expected shape.
    """
    from lxml import etree

    musicxml, _ = engraved_artifacts["empty_left_hand"]
    root = etree.fromstring(musicxml)

    parts = root.findall("part")
    assert len(parts) == 2, f"expected two parts even with empty LH, got {len(parts)}"

    brace_starts = [
        g for g in root.findall("part-list/part-group")
        if g.get("type") == "start" and g.findtext("group-symbol") == "brace"
    ]
    assert brace_starts, "expected brace part-group even when LH is empty"

    # LH part exists but has no pitched notes.
    assert _count_part_notes(musicxml, 1) == 0


def test_l2_bar_crossing_note_in_voice_is_split_with_tie():
    """A voice-2 note that crosses a bar line must be split into two
    tied notes — never emitted as a single over-long duration that
    overflows its starting measure.

    MuseScore 4 renders a "Score corrupted" banner for any MusicXML
    measure whose voice durations sum to more than the time signature
    allows. Before the fix, engrave's explicit-``Voice`` path relied on
    ``part.makeNotation(inPlace=True)`` to split bar-crossing notes via
    ``makeTies`` — but music21's ``makeTies`` does not reliably descend
    into ``Voice`` sub-streams, so voice-2 notes straddling a barline
    got written verbatim into their starting measure with an oversized
    ``<duration>``.
    """
    import xml.etree.ElementTree as etree

    from backend.contracts import (
        PianoScore,
        ScoreMetadata,
        ScoreNote,
        TempoMapEntry,
    )
    from backend.services.engrave import _engrave_sync

    # Reproduces the production failure (job 9334f0368dfe, measure 10):
    # LH voice 1 contains two same-pitch B3 notes that overlap at a
    # measure boundary:
    #   * a 4-beat note ending at m2 beat 0.5 (tied tail from m1),
    #   * an eighth at m2 beat 0 (attacks during that tied tail),
    #   * another 4-beat note at m2 beat 0.5 (crosses into m3).
    # When the same pitch attacks at the same beat as a tied
    # continuation in a ``Voice`` sub-stream, music21's ``makeTies``
    # can't split the bar-crossing note cleanly — m2 voice 1 ends up
    # with 6+6+42 = 54 divisions instead of 48. MuseScore flags the
    # overflow as a "corrupted score".
    rh = [
        ScoreNote(id=f"rh-{i}", pitch=72, onset_beat=float(i), duration_beat=1.0, velocity=80, voice=1)
        for i in range(12)  # 3 measures of quarters
    ]
    lh = [
        # M1 LH v1: eighth at beat 0, then 4-beat note at beat 0.5
        # (crosses into M2 by 0.5 beats).
        ScoreNote(id="lh-m1-v1a", pitch=59, onset_beat=0.0, duration_beat=0.5, velocity=64, voice=1),
        ScoreNote(id="lh-m1-v1b", pitch=59, onset_beat=0.5, duration_beat=4.0, velocity=77, voice=1),
        # M1 LH v2 whole note — forces use_explicit_voices=True.
        ScoreNote(id="lh-m1-v2", pitch=55, onset_beat=0.0, duration_beat=4.0, velocity=70, voice=2),
        # M2 LH v1: ANOTHER eighth at beat 4 (onset=4, dur=0.5) that
        # overlaps with the tied tail of lh-m1-v1b, then another
        # 4-beat note crossing into M3.
        ScoreNote(id="lh-m2-v1a", pitch=59, onset_beat=4.0, duration_beat=0.5, velocity=64, voice=1),
        ScoreNote(id="lh-m2-v1b", pitch=59, onset_beat=4.5, duration_beat=4.0, velocity=77, voice=1),
        ScoreNote(id="lh-m2-v2", pitch=48, onset_beat=4.0, duration_beat=4.0, velocity=70, voice=2),
        # M3 padding so the tied tail has somewhere to land.
        ScoreNote(id="lh-m3-v1", pitch=55, onset_beat=8.5, duration_beat=3.5, velocity=70, voice=1),
        ScoreNote(id="lh-m3-v2", pitch=48, onset_beat=8.0, duration_beat=4.0, velocity=70, voice=2),
    ]

    score = PianoScore(
        metadata=ScoreMetadata(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
            difficulty="intermediate",
            sections=[],
            chord_symbols=[],
        ),
        right_hand=rh,
        left_hand=lh,
    )

    _pdf, musicxml, _midi, _chord_count = _engrave_sync(
        score, title="bar_crossing", composer="test",
    )

    root = etree.fromstring(musicxml)
    parts = root.findall("part")
    assert len(parts) == 1, "engrave should emit a single grand-staff <part>"

    # Pull the divisions + time signature from the first measure's
    # <attributes> block so this test stays honest if we change defaults.
    first_attrs = parts[0].find("measure/attributes")
    divisions = int(first_attrs.findtext("divisions"))
    beats = int(first_attrs.find("time").findtext("beats"))
    beat_type = int(first_attrs.find("time").findtext("beat-type"))
    expected_per_measure = divisions * 4 * beats // beat_type

    overflows: list[tuple[str, int, int]] = []
    for measure in parts[0].findall("measure"):
        # Walk the measure top-to-bottom tracking a per-voice cursor.
        # Each voice's duration in the measure is the max cursor it
        # reaches before the terminating ``<backup>`` (or end-of-measure).
        voice_max: dict[str, int] = {}
        cursor_by_voice: dict[str, int] = {}
        current_voice = "1"
        for el in measure:
            if el.tag == "note":
                v_el = el.find("voice")
                voice = v_el.text if v_el is not None else current_voice
                current_voice = voice
                is_chord = el.find("chord") is not None
                dur = int(el.findtext("duration") or 0)
                if not is_chord:
                    cursor_by_voice[voice] = cursor_by_voice.get(voice, 0) + dur
                voice_max[voice] = max(voice_max.get(voice, 0), cursor_by_voice.get(voice, 0))
            elif el.tag == "backup":
                dur = int(el.findtext("duration") or 0)
                cursor_by_voice[current_voice] = max(0, cursor_by_voice.get(current_voice, 0) - dur)
            elif el.tag == "forward":
                dur = int(el.findtext("duration") or 0)
                cursor_by_voice[current_voice] = cursor_by_voice.get(current_voice, 0) + dur
                voice_max[current_voice] = max(
                    voice_max.get(current_voice, 0), cursor_by_voice[current_voice]
                )

        for voice, filled in voice_max.items():
            if filled > expected_per_measure:
                overflows.append((measure.get("number"), voice, filled))

    assert not overflows, (
        f"voice duration overflow(s) — MuseScore will flag this as a corrupt "
        f"score. expected ≤ {expected_per_measure} divisions per voice per "
        f"measure, got: {overflows}"
    )


def test_l2_ties_do_not_cross_voices():
    """A tie pair (``type="start"`` / ``type="stop"``) must stay in the
    same (pitch, voice, staff). When music21's ``makeTies`` writes the
    continuation of a bar-crossing note on a DIFFERENT voice in the
    next measure, the sanitizer's voice-clamp can merge it into a
    voice that has no matching start — leaving an orphan tie that
    MuseScore 4 flags as a corrupt score.

    This reproduces the second failure mode from job 9334f0368dfe:
    a LH voice-1 note that crosses into a measure whose only other
    LH content is in voice 2 (forcing ``use_explicit_voices=True``)
    ends up with its tie continuation renumbered and the sanitizer
    turns the resulting voice mismatch into a corrupt tie chain.
    """
    import xml.etree.ElementTree as etree

    from backend.contracts import (
        PianoScore,
        ScoreMetadata,
        ScoreNote,
        TempoMapEntry,
    )
    from backend.services.engrave import _engrave_sync

    # Both hands need 2 voices with sparse voice-2 content — that's
    # the pattern that makes music21 assign a fresh voice number to
    # the LH bar-crossing continuation in the next measure. The
    # sanitizer then clamps that fresh number back into voice 2, which
    # on the NEXT measure doesn't match voice 1 where the tie started.
    rh = [
        ScoreNote(id=f"rh-v1-{i}", pitch=72, onset_beat=float(i), duration_beat=1.0, velocity=80, voice=1)
        for i in range(12)
    ]
    # RH voice 2 appears only in M1 and M3 — sparse.
    rh.append(ScoreNote(id="rh-v2-m1", pitch=67, onset_beat=3.5, duration_beat=0.25, velocity=70, voice=2))
    rh.append(ScoreNote(id="rh-v2-m3", pitch=67, onset_beat=9.5, duration_beat=0.25, velocity=70, voice=2))
    lh = [
        # M1 LH v1: eighth + 4-beat note crossing into M2 by 0.5 beats.
        ScoreNote(id="lh-m1-v1a", pitch=59, onset_beat=0.0, duration_beat=0.5, velocity=64, voice=1),
        ScoreNote(id="lh-m1-v1b", pitch=59, onset_beat=0.5, duration_beat=4.0, velocity=77, voice=1),
        # M1 LH v2: F3 whole note forces explicit voices on LH.
        ScoreNote(id="lh-m1-v2", pitch=53, onset_beat=0.0, duration_beat=4.0, velocity=70, voice=2),
        # M2 LH v1: only a single G#3 at beat 0.5 — no voice 2 in M2 LH.
        ScoreNote(id="lh-m2-v1a", pitch=56, onset_beat=4.5, duration_beat=1.0, velocity=70, voice=1),
        # M3 LH v1+v2 so the LH has voice 2 content elsewhere.
        ScoreNote(id="lh-m3-v1", pitch=55, onset_beat=8.0, duration_beat=4.0, velocity=70, voice=1),
        ScoreNote(id="lh-m3-v2", pitch=48, onset_beat=8.0, duration_beat=4.0, velocity=70, voice=2),
    ]
    score = PianoScore(
        metadata=ScoreMetadata(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
            difficulty="intermediate",
            sections=[],
            chord_symbols=[],
        ),
        right_hand=rh,
        left_hand=lh,
    )
    _pdf, musicxml, _midi, _chord_count = _engrave_sync(
        score, title="t", composer="c",
    )

    root = etree.fromstring(musicxml)
    # Walk every tie and confirm each (pitch, voice, staff) start has
    # a matching stop — no orphans, no doubles, no unclosed chains.
    open_ties: dict[tuple, str] = {}
    issues: list[tuple[str, str, tuple]] = []
    for measure in root.find("part").findall("measure"):
        mnum = measure.get("number")
        for n in measure.findall("note"):
            pitch = n.find("pitch")
            if pitch is None:
                continue
            key = (
                pitch.findtext("step"),
                pitch.findtext("octave"),
                pitch.findtext("alter") or "0",
                n.findtext("voice") or "1",
                n.findtext("staff") or "1",
            )
            for tie in n.findall("tie"):
                typ = tie.get("type")
                if typ == "start":
                    if key in open_ties:
                        issues.append(("double-start", mnum, key))
                    open_ties[key] = mnum
                elif typ == "stop":
                    if key not in open_ties:
                        issues.append(("orphan-stop", mnum, key))
                    else:
                        del open_ties[key]
    for key, mnum in open_ties.items():
        issues.append(("unclosed-start", mnum, key))

    assert not issues, (
        f"tie chain issues — MuseScore will flag this as corrupt: {issues}"
    )


def test_engrave_does_not_leak_music21_defaults():
    """``music21.defaults.divisionsPerQuarter`` must be restored after engrave.

    PR-9 overrides this global to 12 around ``s.write("musicxml", ...)``
    so OSMD-friendly divisions come out by construction. The override
    must be wrapped in try/finally so that other music21 callers (eval
    scripts, arrange's makeNotation in a shared process, etc.) keep
    seeing the upstream default of 10080.
    """
    import music21

    from backend.services.engrave import _engrave_sync

    prior = music21.defaults.divisionsPerQuarter
    _pdf, _xml, _midi, _chord_count = _engrave_sync(
        load_score_fixture("c_major_scale"), title="t", composer="c"
    )
    assert music21.defaults.divisionsPerQuarter == prior


def test_coerce_musicxml_durations_fixes_2048th_tuplet_bracket() -> None:
    """``makeNotation``-style tuplets can use a 2048th normal type; MusicXML rejects that."""
    import music21
    from music21.duration import Tuplet, durationTupleFromTypeDots
    from music21.musicxml.m21ToXml import typeToMusicXMLType

    from backend.services.engrave import _coerce_durations_for_musicxml_export

    part = music21.stream.PartStaff()
    r = music21.note.Rest()
    r.duration = music21.duration.Duration(1 / 6)
    nt = Tuplet(3, 2)
    nt.durationNormal = durationTupleFromTypeDots("2048th", 0)
    nt.durationActual = durationTupleFromTypeDots("2048th", 0)
    r.duration.tuplets = (nt,)
    part.insert(0, r)

    _coerce_durations_for_musicxml_export(part, music21)

    r_out = next(part.recurse().notesAndRests)
    typeToMusicXMLType(r_out.duration.type)
    for tup in r_out.duration.tuplets:
        if tup.durationNormal is not None:
            typeToMusicXMLType(tup.durationNormal.type)
        if tup.durationActual is not None:
            typeToMusicXMLType(tup.durationActual.type)


def test_coerce_musicxml_durations_clamps_raw_2048th() -> None:
    """A bare 2048th note type is also unexportable; coerce to at least 1024th."""
    import music21
    from music21.musicxml.m21ToXml import typeToMusicXMLType

    from backend.services.engrave import _coerce_durations_for_musicxml_export

    part = music21.stream.PartStaff()
    n = music21.note.Note("C4")
    n.duration = music21.duration.Duration(
        music21.duration.convertTypeToQuarterLength("2048th"),
    )
    part.insert(0, n)

    _coerce_durations_for_musicxml_export(part, music21)

    n_out = next(part.recurse().notesAndRests)
    typeToMusicXMLType(n_out.duration.type)
    assert n_out.duration.type != "2048th"


def test_coerce_durations_snaps_inexpressible_float_ql() -> None:
    """Float ``quarterLength`` noise can yield ``duration.type == 'inexpressible'``."""
    import music21

    from backend.services.engrave import _coerce_durations_for_musicxml_export

    n = music21.note.Note("C4", quarterLength=2.718281828)
    p = music21.stream.PartStaff()
    p.append(music21.meter.TimeSignature("4/4"))
    p.insert(0, n)
    p.makeNotation(inPlace=True)
    _coerce_durations_for_musicxml_export(p, music21)
    n_out = next(p.recurse().notesAndRests)
    assert n_out.duration.type != "inexpressible"


def test_musicxml_write_make_notation_false_needs_split_at_durations() -> None:
    """Regression: ``complex`` duration on one note breaks MusicXML unless split.

    music21's per-part ``makeNotation`` can leave a single ``Note`` with
    ``duration.type == 'complex'``. ``write(..., makeNotation=False)`` then
    raises; ``splitAtDurations(recurse=True)`` is the documented fix (see
    ``music21.converter.subConverters`` tests).
    """
    import tempfile
    from pathlib import Path

    import music21
    from music21 import duration, meter, note, stream

    from backend.services.engrave import _coerce_durations_for_musicxml_export

    p = stream.PartStaff()
    p.append(meter.TimeSignature("4/4"))
    n = note.Note("C4")
    d = duration.Duration()
    d.addDurationTuple(duration.durationTupleFromTypeDots("quarter", 0))
    d.addDurationTuple(duration.durationTupleFromTypeDots("eighth", 0))
    n.duration = d
    p.insert(0, n)
    p.makeNotation(inPlace=True)

    s = stream.Score()
    s.insert(0, p)

    from music21.musicxml.xmlObjects import MusicXMLExportException

    with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as tmp:
        bad_path = Path(tmp.name)
    try:
        with pytest.raises(MusicXMLExportException):
            s.write("musicxml", fp=str(bad_path), makeNotation=False)
    finally:
        bad_path.unlink(missing_ok=True)

    s.splitAtDurations(recurse=True)
    _coerce_durations_for_musicxml_export(s, music21)

    with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as tmp:
        good_path = Path(tmp.name)
    try:
        s.write("musicxml", fp=str(good_path), makeNotation=False)
        assert good_path.stat().st_size > 500
    finally:
        good_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# L2 — Tuplet survival (plan phase 3.4 / PR-13)
# ---------------------------------------------------------------------------


def test_l2_triplet_eighths_render_as_tuplet(engraved_artifacts):
    """Plan phase 3.4 — triplet-8ths survive without engrave re-quantizing.

    The ``triplet_eighths`` fixture carries 12 RH notes at 1/3-beat
    spacing over 4 beats — a clean triplet grid from arrange's
    ``_estimate_best_grid`` path. Before PR-13, engrave re-quantized
    every Part to ``quarterLengthDivisors=(4, 3)`` as a safety net
    inherited from the divisions=10080 era; the coarser divisor tuple
    could not represent triplet-16ths and the re-quantize silently
    dropped fine resolution.

    After PR-13 engrave trusts arrange's grid directly and music21's
    ``makeNotation`` auto-detects the tuplet bracket. What we check:

    - Each triplet-8th duration = divisions/3 (at divisions=12, that's 4).
    - The MusicXML carries ``<time-modification>`` elements with
      ``<actual-notes>3</actual-notes>`` / ``<normal-notes>2</normal-notes>``
      on each triplet member — the standard "3 in the time of 2"
      encoding for an 8th-note triplet.
    - At least 12 such time-modification blocks land on RH (one per
      triplet 8th).
    """
    from lxml import etree

    musicxml, _ = engraved_artifacts["triplet_eighths"]
    root = etree.fromstring(musicxml)

    # Every triplet 8th should have <time-modification> with 3/2 ratio.
    triplet_members = 0
    for note in root.iter("note"):
        if note.find("rest") is not None:
            continue
        tm = note.find("time-modification")
        if tm is None:
            continue
        actual = tm.findtext("actual-notes")
        normal = tm.findtext("normal-notes")
        if actual == "3" and normal == "2":
            triplet_members += 1

    assert triplet_members >= 12, (
        f"expected ≥12 triplet-8th members with <time-modification>3:2</time-modification>, "
        f"got {triplet_members}"
    )


# ---------------------------------------------------------------------------
# L2 — Key signature verification (plan phase 3.3 / PR-12)
# ---------------------------------------------------------------------------


def test_l2_key_signature_override_on_mislabel(engraved_artifacts):
    """Plan phase 3.3 — Krumhansl-Schmuckler overrides a mislabeled key.

    The ``mislabeled_key`` fixture is an unambiguous F# minor piece
    tagged as ``C:major``. After the KS analyzer runs in
    ``_resolve_key_signature``, the MusicXML ``<key><fifths>`` element
    should show ``3`` (F# minor / A major — three sharps) instead of
    ``0`` (C major — no accidentals).

    Without the override, every F#/G#/C# in the piece would render as
    an explicit accidental on each note head — unreadable.
    """
    from lxml import etree

    musicxml, _ = engraved_artifacts["mislabeled_key"]
    root = etree.fromstring(musicxml)
    fifths = [int(e.text) for e in root.iter("fifths")]
    assert fifths, "no <fifths> element in mislabeled_key MusicXML"
    assert set(fifths) == {3}, (
        f"expected <fifths>3</fifths> (F# minor override), got {sorted(set(fifths))}"
    )
    modes = [e.text for e in root.iter("mode")]
    assert modes and all(m == "minor" for m in modes), (
        f"expected <mode>minor</mode>, got {modes}"
    )


def test_l2_key_signature_low_correlation_trusts_label():
    """Plan phase 3.3 — low KS correlation leaves the declared key alone.

    Builds a throwaway score out of a highly chromatic tone row so the
    Krumhansl-Schmuckler correlation lands well below the 0.80 override
    floor. Even if the analyzer's tonic guess disagrees with the
    declared key, the override must **not** fire — the histogram is too
    ambiguous to trust.
    """
    import music21

    from backend.contracts import (
        SCHEMA_VERSION,
        PianoScore,
        ScoreMetadata,
        ScoreNote,
        TempoMapEntry,
    )
    from backend.services.engrave import _resolve_key_signature

    # All 12 pitch classes once each — maximum chromatic, no tonal
    # center, correlation should collapse.
    row = [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
    score = PianoScore(
        schema_version=SCHEMA_VERSION,
        right_hand=[
            ScoreNote(id=f"rh-{i:04d}", pitch=p, onset_beat=float(i),
                      duration_beat=1.0, velocity=80, voice=1)
            for i, p in enumerate(row)
        ],
        left_hand=[],
        metadata=ScoreMetadata(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
            difficulty="intermediate",
            sections=[],
            chord_symbols=[],
        ),
    )

    root, mode, overridden = _resolve_key_signature(score, music21)
    assert not overridden, (
        f"chromatic tone row should not trigger override; got {root}:{mode}"
    )
    assert (root, mode) == ("C", "major"), (
        f"expected declared C:major to survive, got {root}:{mode}"
    )


def test_l2_key_signature_trusts_correct_label(engraved_artifacts):
    """Plan phase 3.3 — KS does **not** override when the label agrees.

    Every existing fixture except ``mislabeled_key`` is honestly
    labeled as C:major. The override must keep quiet on all of them so
    we don't drift the declared key based on analyzer noise.
    """
    from lxml import etree

    for name in FIXTURE_NAMES:
        if name == "mislabeled_key":
            continue
        musicxml, _ = engraved_artifacts[name]
        root = etree.fromstring(musicxml)
        fifths = [int(e.text) for e in root.iter("fifths")]
        assert set(fifths) == {0}, (
            f"{name}: unexpected key-signature override — expected <fifths>0</fifths>, "
            f"got {sorted(set(fifths))}"
        )


# ---------------------------------------------------------------------------
# L2 — Chord symbol filter gate (plan phase 3.2 / PR-11)
# ---------------------------------------------------------------------------


def test_l2_chord_symbols_filter_gate(engraved_artifacts):
    """Plan phase 3.2 — only the three clean labels reach MusicXML.

    The ``chord_symbols`` fixture carries seven input labels covering
    every branch of ``_attach_chord_symbols``:

    - ``C:maj7``, ``Dm7``, ``F:maj7`` — high-confidence, parseable, ≥3 pitches **→ rendered**
    - ``G5`` — pitch-octave shape **→ dropped (shape gate)**
    - ``C:maj`` — below 0.5 confidence **→ dropped (confidence gate)**
    - ``???`` — unparseable **→ dropped (parse gate)**
    - ``C5`` — pitch-octave shape **→ dropped (shape gate)**

    Assertion: exactly three ``<harmony>`` elements land in the output,
    with roots ``{C, D, F}``.
    """
    from lxml import etree

    musicxml, _ = engraved_artifacts["chord_symbols"]
    root = etree.fromstring(musicxml)
    harmonies = list(root.iter("harmony"))
    assert len(harmonies) == 3, (
        f"expected 3 rendered chord symbols after filter gate, got {len(harmonies)}"
    )

    roots = {h.findtext("root/root-step") for h in harmonies}
    assert roots == {"C", "D", "F"}, (
        f"expected chord roots {{C, D, F}}, got {roots}"
    )


def test_l2_chord_symbols_flag_reflects_filter(tmp_path):
    """``EngravedScoreData.includes_chord_symbols`` must gate on the *rendered*
    count, not the raw input count. PR-11 (plan phase 3.2) wires the
    filter-gate return value all the way out through ``EngraveService``
    so the flag stops lying about what's on the page.
    """
    import asyncio

    from backend.services.engrave import EngraveService
    from backend.storage.local import LocalBlobStore
    from tests.fixtures import load_score_fixture

    svc = EngraveService(LocalBlobStore(tmp_path))
    fixture = load_score_fixture("chord_symbols")
    out = asyncio.run(
        svc.run(fixture, job_id="test-pr11-chord-flag", title="t", composer="c")
    )
    assert out.metadata.includes_chord_symbols is True


def test_l2_chord_symbols_flag_false_when_all_filtered(tmp_path):
    """Input with *only* bad chord labels → ``includes_chord_symbols`` is False.

    Previously the flag was ``len(chord_symbols) > 0``, which would
    return True even when every label was about to be dropped. Build a
    throwaway score with three shape-gate-failing labels and verify the
    flag flips to False after rendering.
    """
    import asyncio

    from backend.contracts import (
        SCHEMA_VERSION,
        PianoScore,
        ScoreChordEvent,
        ScoreMetadata,
        ScoreNote,
        TempoMapEntry,
    )
    from backend.services.engrave import EngraveService
    from backend.storage.local import LocalBlobStore

    score = PianoScore(
        schema_version=SCHEMA_VERSION,
        right_hand=[
            ScoreNote(id="rh-0000", pitch=60, onset_beat=0.0, duration_beat=1.0,
                      velocity=80, voice=1),
            ScoreNote(id="rh-0001", pitch=62, onset_beat=1.0, duration_beat=1.0,
                      velocity=80, voice=1),
        ],
        left_hand=[],
        metadata=ScoreMetadata(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
            difficulty="intermediate",
            sections=[],
            chord_symbols=[
                ScoreChordEvent(beat=0.0, duration_beat=1.0, label="G5",
                                root=67, confidence=0.99),
                ScoreChordEvent(beat=1.0, duration_beat=1.0, label="C3",
                                root=48, confidence=0.99),
            ],
        ),
    )

    svc = EngraveService(LocalBlobStore(tmp_path))
    out = asyncio.run(
        svc.run(score, job_id="test-pr11-chord-flag-false", title="t", composer="c")
    )
    assert out.metadata.includes_chord_symbols is False, (
        "flag should be False when every input chord is filtered out"
    )


def test_l2_engraved_flags_reflect_rendered_content(engraved_artifacts):
    """``EngravedScoreData.includes_dynamics / includes_pedal_marks`` must
    now report True when the humanized input populates them.

    The flag flip is the PR-5 counterpart to the phase 1.1 "truthful
    flags" change — phase 1.1 could only flip the flags to False because
    engrave wasn't rendering those markings yet.
    """
    import asyncio
    import tempfile
    from pathlib import Path

    from backend.services.engrave import EngraveService
    from backend.storage.local import LocalBlobStore
    from tests.fixtures import load_score_fixture

    # EngraveService writes artifacts to a blob store before returning,
    # so stand up a throwaway local store rooted in a tmp dir.
    with tempfile.TemporaryDirectory() as tmpdir:
        svc = EngraveService(LocalBlobStore(Path(tmpdir)))
        fixture = load_score_fixture("humanized_with_expression")
        out = asyncio.run(
            svc.run(fixture, job_id="test-pr5-flags", title="t", composer="c")
        )
        assert out.metadata.includes_dynamics is True
        assert out.metadata.includes_pedal_marks is True
