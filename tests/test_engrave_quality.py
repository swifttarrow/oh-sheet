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
        _pdf, musicxml, midi = _engrave_sync(fixture, title=name, composer="test")
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
def test_l2_divisions_reasonable(name: str, engraved_artifacts):
    """``<divisions>`` must stay small (≤480) — OSMD chokes on the
    10080 value music21 picks by default."""
    from lxml import etree

    musicxml, _ = engraved_artifacts[name]
    root = etree.fromstring(musicxml)
    divisions = [int(e.text) for e in root.iter("divisions")]
    assert divisions, f"{name}: no <divisions> element in MusicXML"
    for d in divisions:
        assert 1 <= d <= 480, f"{name}: <divisions>{d}</divisions> outside [1,480]"


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
