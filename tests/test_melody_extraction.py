"""Unit tests for Phase 2 melody extraction.

The tests drive the Viterbi tracer with hand-built synthetic contour
matrices — no audio, no basic_pitch inference — so they exercise the
algorithm in isolation. We skip gracefully if numpy isn't installed
(the module itself is tolerant, but the tests need to construct
``ndarray`` inputs directly).
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from backend.services.melody_extraction import (  # noqa: E402
    DEFAULT_BACKFILL_ENABLED,
    DEFAULT_BACKFILL_MAX_AMP,
    DEFAULT_BACKFILL_MIN_AMP,
    DEFAULT_BACKFILL_MIN_DURATION_SEC,
    DEFAULT_BACKFILL_OVERLAP_FRACTION,
    DEFAULT_MATCH_FRACTION,
    DEFAULT_MAX_TRANSITION_BINS,
    DEFAULT_MELODY_HIGH_MIDI,
    DEFAULT_MELODY_LOW_MIDI,
    DEFAULT_TRANSITION_WEIGHT,
    DEFAULT_VOICING_FLOOR,
    FRAME_RATE_HZ,
    N_CONTOUR_BINS,
    _path_to_midi_runs,
    _trace_f0_contour,
    backfill_melody_notes,
    bin_to_midi,
    extract_melody,
    midi_to_bin,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_contour(n_frames: int, baseline: float = 0.02):
    """Uniformly-low salience matrix — the canvas for each test."""
    return np.full((n_frames, N_CONTOUR_BINS), baseline, dtype=np.float32)


def _paint(contour, start_frame: int, end_frame: int, midi: int, salience: float = 0.9):
    """Stamp salience at (midi, frame range) in-place."""
    contour[start_frame:end_frame, midi_to_bin(midi)] = salience


def _run(contour):
    return _trace_f0_contour(
        contour,
        low_bin=midi_to_bin(DEFAULT_MELODY_LOW_MIDI),
        high_bin=midi_to_bin(DEFAULT_MELODY_HIGH_MIDI),
        voicing_floor=DEFAULT_VOICING_FLOOR,
        transition_weight=DEFAULT_TRANSITION_WEIGHT,
        max_transition_bins=DEFAULT_MAX_TRANSITION_BINS,
        voiced_enter_cost=1.0,
        unvoiced_enter_cost=1.0,
    )


# ---------------------------------------------------------------------------
# Bin ↔ MIDI mapping
# ---------------------------------------------------------------------------

def test_bin_midi_roundtrip_on_integer_pitches():
    for midi in (21, 55, 60, 69, 90, 108):
        assert bin_to_midi(midi_to_bin(midi)) == midi


def test_bin_mapping_matches_basic_pitch_formula():
    # Basic Pitch: bin = 3 * (midi - 21) for integer MIDI.
    assert midi_to_bin(21) == 0
    assert midi_to_bin(60) == 117
    assert midi_to_bin(108) == 261


# ---------------------------------------------------------------------------
# Viterbi tracer — shape and voicing
# ---------------------------------------------------------------------------

def test_trace_follows_single_stable_peak():
    c = _blank_contour(100)
    _paint(c, 0, 100, 60)
    path = _run(c)
    # All frames should be voiced at MIDI 60.
    assert (path >= 0).all()
    assert all(bin_to_midi(int(p)) == 60 for p in path)


def test_trace_jumps_to_new_peak():
    c = _blank_contour(100)
    _paint(c, 0, 50, 60)
    _paint(c, 50, 100, 64)
    path = _run(c)
    assert all(bin_to_midi(int(p)) == 60 for p in path[:50])
    assert all(bin_to_midi(int(p)) == 64 for p in path[50:])


def test_trace_enters_unvoiced_on_silence():
    c = _blank_contour(100)
    _paint(c, 0, 30, 60)
    # frames 30..70 are below voicing floor
    _paint(c, 70, 100, 60)
    path = _run(c)
    # Voiced at the ends, unvoiced in the middle.
    assert (path[:30] >= 0).all()
    assert (path[30:70] < 0).all()
    assert (path[70:] >= 0).all()


def test_trace_masks_out_of_band_peaks():
    # A strong peak below the melody band must be ignored.
    c = _blank_contour(100)
    _paint(c, 0, 100, 40)  # MIDI 40 < C3 (48)
    path = _run(c)
    assert (path < 0).all()


def test_trace_prefers_small_jumps_over_large():
    # Two candidate peaks at 60 and 72, with 72 slightly stronger early
    # but 60 stronger later. The transition penalty should keep the
    # path near 60 throughout because a big jump costs more than a
    # small delta in emission.
    c = _blank_contour(200)
    _paint(c, 0, 200, 60, salience=0.7)   # steady
    _paint(c, 50, 150, 72, salience=0.75) # slightly stronger briefly
    path = _run(c)
    # Path should stay at 60; the ~12-bin jump isn't worth the 0.05
    # emission improvement once transition cost is counted.
    midis = [bin_to_midi(int(p)) for p in path if p >= 0]
    assert all(m == 60 for m in midis), f"path wandered: {set(midis)}"


def test_trace_reports_expected_shape():
    c = _blank_contour(50)
    _paint(c, 0, 50, 60)
    path = _run(c)
    assert path.shape == (50,)
    assert path.dtype == np.int32


# ---------------------------------------------------------------------------
# Path → MIDI runs
# ---------------------------------------------------------------------------

def test_path_to_runs_groups_consecutive_frames():
    c = _blank_contour(100)
    _paint(c, 0, 40, 60)
    _paint(c, 40, 80, 64)
    _paint(c, 80, 100, 67)
    path = _run(c)
    runs = _path_to_midi_runs(path)
    pitches = [midi for _, _, midi in runs]
    assert pitches == [60, 64, 67]
    lengths = [end - start for start, end, _ in runs]
    assert lengths == [40, 40, 20]


def test_path_to_runs_splits_on_unvoiced():
    c = _blank_contour(100)
    _paint(c, 0, 30, 60)
    _paint(c, 70, 100, 60)  # same pitch, but with an unvoiced gap in the middle
    path = _run(c)
    runs = _path_to_midi_runs(path)
    # Two separate runs at MIDI 60 because the gap broke continuity.
    assert len(runs) == 2
    assert all(midi == 60 for _, _, midi in runs)


# ---------------------------------------------------------------------------
# End-to-end extract_melody
# ---------------------------------------------------------------------------

def _ne(start: float, end: float, pitch: int, amp: float = 0.8):
    """Build a Basic Pitch note_event tuple."""
    return (start, end, pitch, amp, None)


def test_extract_melody_tags_melody_and_chord_notes():
    # Melody: C4 → E4 → G4, one second each
    # Chords: below the melody band at all times
    frames_per_sec = FRAME_RATE_HZ
    c = _blank_contour(int(3.1 * frames_per_sec))
    _paint(c, 0, int(frames_per_sec), 60)
    _paint(c, int(frames_per_sec), int(2 * frames_per_sec), 64)
    _paint(c, int(2 * frames_per_sec), int(3 * frames_per_sec), 67)

    events = [
        _ne(0.0, 1.0, 60),
        _ne(1.0, 2.0, 64),
        _ne(2.0, 3.0, 67),
        _ne(0.0, 1.0, 47),  # B2 — below band → chord
        _ne(0.0, 1.0, 46),  # A#2 — below band → chord
        _ne(1.0, 2.0, 45),
        _ne(2.0, 3.0, 43),
    ]
    melody, chords, stats = extract_melody(c, events)
    assert not stats.skipped
    assert sorted(e[2] for e in melody) == [60, 64, 67]
    assert sorted(e[2] for e in chords) == [43, 45, 46, 47]
    assert stats.melody_note_count == 3
    assert stats.chord_note_count == 4
    assert stats.voiced_frame_fraction > 0.9


def test_extract_melody_skips_when_contour_is_none():
    events = [_ne(0.0, 1.0, 60), _ne(1.0, 2.0, 64)]
    melody, chords, stats = extract_melody(None, events)
    assert stats.skipped
    assert melody == []
    assert chords == events
    assert any("skipped" in w for w in stats.as_warnings())


def test_extract_melody_skips_on_malformed_contour_shape():
    bad = np.zeros((100, 128), dtype=np.float32)  # wrong width
    events = [_ne(0.0, 1.0, 60)]
    melody, chords, stats = extract_melody(bad, events)
    assert stats.skipped
    assert chords == events


def test_extract_melody_sends_out_of_band_notes_to_chords():
    c = _blank_contour(int(1.1 * FRAME_RATE_HZ))
    _paint(c, 0, int(FRAME_RATE_HZ), 60)
    events = [
        _ne(0.0, 1.0, 60),   # in band, matches → melody
        _ne(0.0, 1.0, 30),   # well below band → chord
        _ne(0.0, 1.0, 100),  # above band → chord
    ]
    melody, chords, stats = extract_melody(c, events)
    assert [e[2] for e in melody] == [60]
    assert sorted(e[2] for e in chords) == [30, 100]


def test_extract_melody_band_filter_low_boundary():
    # The melody band is inclusive at the low edge (``DEFAULT_MELODY_LOW_MIDI``
    # = 48), so MIDI 48 must *not* be rejected by the band filter while MIDI
    # 47 must be. This test isolates the band check — the painted contour
    # happens to match the event pitches so path-agreement (a separate
    # guard) isn't what's being exercised here.
    c = _blank_contour(int(1.1 * FRAME_RATE_HZ))
    # Paint the contour at MIDI 48 so the in-band event has a matching run.
    _paint(c, 0, int(FRAME_RATE_HZ), DEFAULT_MELODY_LOW_MIDI)
    events = [
        _ne(0.0, 1.0, DEFAULT_MELODY_LOW_MIDI),       # 48 — boundary, in band
        _ne(0.0, 1.0, DEFAULT_MELODY_LOW_MIDI - 1),   # 47 — below band
    ]
    melody, chords, stats = extract_melody(c, events, backfill_enabled=False)
    assert not stats.skipped
    # Boundary pitch 48 survives the band filter — it reaches the
    # path-agreement stage rather than being punted into chords by the
    # strict ``< low_midi`` comparison.
    assert DEFAULT_MELODY_LOW_MIDI not in {e[2] for e in chords}
    # MIDI 47 is unambiguously below the band and must land in chords.
    assert DEFAULT_MELODY_LOW_MIDI - 1 in {e[2] for e in chords}


def test_extract_melody_note_disagreeing_with_path_goes_to_chords():
    # Path traces MIDI 60 throughout, but the event is at MIDI 72.
    # With match_fraction = 0.6 the note should land in the chord bucket.
    # Back-fill is off here so the test isolates the disagreement path —
    # otherwise the stable MIDI 60 run would synthesize a new melody note.
    c = _blank_contour(int(1.1 * FRAME_RATE_HZ))
    _paint(c, 0, int(FRAME_RATE_HZ), 60)
    events = [_ne(0.0, 1.0, 72)]
    melody, chords, stats = extract_melody(c, events, backfill_enabled=False)
    assert melody == []
    assert chords == events


def test_extract_melody_empty_events():
    # Back-fill off so an empty input → empty output (no synthesis from
    # the painted peak). Back-fill's own behavior is covered in the
    # dedicated tests below.
    c = _blank_contour(50)
    _paint(c, 0, 50, 60)
    melody, chords, stats = extract_melody(c, [], backfill_enabled=False)
    assert melody == [] and chords == []
    assert stats.melody_note_count == 0 and stats.chord_note_count == 0


def test_extract_melody_tiny_contour_is_skipped():
    c = np.zeros((1, N_CONTOUR_BINS), dtype=np.float32)
    events = [_ne(0.0, 1.0, 60)]
    melody, chords, stats = extract_melody(c, events)
    assert stats.skipped
    assert chords == events


# ---------------------------------------------------------------------------
# Back-fill of stable Viterbi runs the upstream note tracker missed
# ---------------------------------------------------------------------------

def test_backfill_adds_stable_run_without_matching_event():
    # 300 ms stable peak at MIDI 67 (G4), no upstream events at all.
    frames = int(round(0.35 * FRAME_RATE_HZ))
    c = _blank_contour(frames)
    run_end = int(round(0.30 * FRAME_RATE_HZ))
    _paint(c, 0, run_end, 67, salience=0.9)

    melody, chords, stats = extract_melody(c, [])
    assert not stats.skipped
    # Nothing was passed in → chord bucket is empty.
    assert chords == []
    # Exactly one back-filled note at MIDI 67 spanning ~300 ms.
    assert len(melody) == 1
    assert stats.backfilled_note_count == 1
    start, end, pitch, amp, bends = melody[0]
    assert pitch == 67
    assert 0.25 <= (end - start) <= 0.35
    assert bends is None
    assert DEFAULT_BACKFILL_MIN_AMP <= amp <= DEFAULT_BACKFILL_MAX_AMP


def test_backfill_skips_runs_shorter_than_min_duration():
    # 80 ms peak — below the 120 ms back-fill floor.
    frames = int(round(0.12 * FRAME_RATE_HZ))
    c = _blank_contour(frames)
    run_end = int(round(0.08 * FRAME_RATE_HZ))
    _paint(c, 0, run_end, 67, salience=0.9)

    melody, chords, stats = extract_melody(c, [])
    assert stats.backfilled_note_count == 0
    assert melody == []


def test_backfill_skips_when_existing_event_covers_the_run():
    # 300 ms stable peak at MIDI 67, plus an upstream Basic Pitch event
    # covering the same window — back-fill should skip (no duplication).
    frames = int(round(0.35 * FRAME_RATE_HZ))
    c = _blank_contour(frames)
    run_end = int(round(0.30 * FRAME_RATE_HZ))
    _paint(c, 0, run_end, 67, salience=0.9)
    events = [_ne(0.0, 0.30, 67, amp=0.8)]

    melody, chords, stats = extract_melody(c, events)
    assert stats.backfilled_note_count == 0
    # Only the original upstream event survives, un-duplicated.
    assert len(melody) == 1
    assert melody[0][3] == 0.8  # original amplitude preserved


def test_backfill_amplitude_is_clipped_to_max():
    # Peak salience 0.95 > max_amp (0.60) — synthesized amp must be clipped.
    frames = int(round(0.35 * FRAME_RATE_HZ))
    c = _blank_contour(frames)
    run_end = int(round(0.30 * FRAME_RATE_HZ))
    _paint(c, 0, run_end, 67, salience=0.95)

    melody, _, stats = extract_melody(c, [])
    assert stats.backfilled_note_count == 1
    _, _, _, amp, _ = melody[0]
    assert amp == pytest.approx(DEFAULT_BACKFILL_MAX_AMP, abs=1e-6)


def test_backfill_respects_enabled_flag():
    frames = int(round(0.35 * FRAME_RATE_HZ))
    c = _blank_contour(frames)
    _paint(c, 0, int(round(0.30 * FRAME_RATE_HZ)), 67, salience=0.9)

    melody, _, stats = extract_melody(c, [], backfill_enabled=False)
    assert stats.backfilled_note_count == 0
    assert melody == []


def test_backfill_does_not_invent_notes_below_melody_band():
    # Peak inside the low-register slice that would belong to bass.
    # The Viterbi band mask already prevents a path at MIDI 40, so we
    # should not see a back-filled note here either — the tracer is
    # unvoiced for this frame range.
    frames = int(round(0.35 * FRAME_RATE_HZ))
    c = _blank_contour(frames)
    _paint(c, 0, int(round(0.30 * FRAME_RATE_HZ)), 40, salience=0.9)

    melody, _, stats = extract_melody(c, [])
    assert stats.backfilled_note_count == 0
    assert melody == []


# ---------------------------------------------------------------------------
# Back-fill clamping against ``max_time_sec``
# ---------------------------------------------------------------------------

def test_backfill_skips_runs_that_start_past_max_time_sec():
    # 500 ms contour with a stable painted peak from 250 ms to 450 ms.
    # The real audio is only 200 ms long (the rest is BP's contour
    # padding), so the run starts *after* the audio ends and must be
    # dropped entirely — otherwise the back-fill invents a ghost note
    # in silence past the song. This is the regression the stems-path
    # bug surfaced: a 1.5s phantom note at the end of ``basic-pitch.mid``.
    frames = int(round(0.50 * FRAME_RATE_HZ))
    c = _blank_contour(frames)
    run_start = int(round(0.25 * FRAME_RATE_HZ))
    run_end = int(round(0.45 * FRAME_RATE_HZ))
    _paint(c, run_start, run_end, 67, salience=0.9)

    melody, _, stats = extract_melody(c, [], max_time_sec=0.20)
    assert stats.backfilled_note_count == 0
    assert melody == []


def test_backfill_truncates_runs_straddling_max_time_sec():
    # Run from 100 ms to 400 ms, but the audio only lasts 250 ms.
    # The back-fill must keep the note (start is inside the audio)
    # but clamp its ``end`` to ``max_time_sec``.
    frames = int(round(0.50 * FRAME_RATE_HZ))
    c = _blank_contour(frames)
    run_start = int(round(0.10 * FRAME_RATE_HZ))
    run_end = int(round(0.40 * FRAME_RATE_HZ))
    _paint(c, run_start, run_end, 67, salience=0.9)

    melody, _, stats = extract_melody(c, [], max_time_sec=0.25)
    assert stats.backfilled_note_count == 1
    start, end, pitch, _amp, _bends = melody[0]
    assert pitch == 67
    assert start == pytest.approx(0.10, abs=0.02)
    assert end == pytest.approx(0.25, abs=1e-9)


def test_backfill_drops_straddling_run_when_clamp_leaves_too_little():
    # Run from 100 ms to 400 ms at ``max_time_sec=0.18``. The clamped
    # duration (~80 ms) is below the 120 ms ``min_duration_sec`` floor,
    # so the note must be dropped entirely — a straddling run with
    # only a sliver in the valid window isn't worth synthesizing.
    frames = int(round(0.50 * FRAME_RATE_HZ))
    c = _blank_contour(frames)
    run_start = int(round(0.10 * FRAME_RATE_HZ))
    run_end = int(round(0.40 * FRAME_RATE_HZ))
    _paint(c, run_start, run_end, 67, salience=0.9)

    melody, _, stats = extract_melody(c, [], max_time_sec=0.18)
    assert stats.backfilled_note_count == 0
    assert melody == []


def test_backfill_without_max_time_sec_preserves_legacy_behavior():
    # Belt-and-suspenders: when ``max_time_sec`` is None (the default)
    # the clamp is inert and back-fill behaves exactly as it did
    # before the fix — synthesizing a note over the full painted run.
    frames = int(round(0.35 * FRAME_RATE_HZ))
    c = _blank_contour(frames)
    _paint(c, 0, int(round(0.30 * FRAME_RATE_HZ)), 67, salience=0.9)

    melody, _, stats = extract_melody(c, [])
    assert stats.backfilled_note_count == 1
    start, end, _, _, _ = melody[0]
    assert 0.25 <= (end - start) <= 0.35


# ---------------------------------------------------------------------------
# ``split_enabled=False`` — additive-only mode for the stems path
# ---------------------------------------------------------------------------

def test_split_enabled_false_keeps_all_input_events_as_melody():
    # Painted peak at MIDI 60 throughout the contour, but the caller
    # passes events at MIDI 72 (which would go to ``chords`` in the
    # default split mode — the path disagrees with the event pitch).
    # With ``split_enabled=False`` the event must be kept in melody
    # and ``chords`` must be empty. Back-fill is off so the assertion
    # isolates the split bypass.
    c = _blank_contour(int(1.1 * FRAME_RATE_HZ))
    _paint(c, 0, int(FRAME_RATE_HZ), 60)
    events = [_ne(0.0, 1.0, 72)]

    melody, chords, stats = extract_melody(
        c, events, backfill_enabled=False, split_enabled=False,
    )
    assert melody == events
    assert chords == []
    assert stats.melody_note_count == 1
    assert stats.chord_note_count == 0


def test_split_enabled_false_still_runs_backfill_against_all_events():
    # Contour has a stable MIDI 67 peak for 300 ms. The caller passes
    # a single MIDI 72 event — under the default split semantics that
    # would go to chords, and back-fill would synthesize a MIDI 67
    # note because no *melody* event overlaps it. In ``split_enabled=
    # False`` mode the MIDI 72 event is kept in melody, and back-fill
    # still runs against the full event list — the 67 peak doesn't
    # match the 72 event (different pitch), so the synthesized note
    # is still added. End state: two melody notes, no chords.
    frames = int(round(0.35 * FRAME_RATE_HZ))
    c = _blank_contour(frames)
    _paint(c, 0, int(round(0.30 * FRAME_RATE_HZ)), 67, salience=0.9)
    events = [_ne(0.0, 0.30, 72)]

    melody, chords, stats = extract_melody(c, events, split_enabled=False)
    pitches = sorted(e[2] for e in melody)
    assert pitches == [67, 72]
    assert chords == []
    assert stats.backfilled_note_count == 1


def test_backfill_melody_notes_wrapper_matches_extract_melody_additive_mode():
    """The ``backfill_melody_notes`` wrapper is a load-bearing call site.

    The stems path in ``backend.services.transcribe._run_with_stems``
    routes through this wrapper instead of ``extract_melody`` so the
    call site doesn't have to unpack the always-empty chord list. This
    test pins the contract: the wrapper must produce the same melody
    event list as ``extract_melody(..., split_enabled=False)`` with
    back-fill enabled, and it must return a 2-tuple (events, stats)
    rather than the 3-tuple the full pipeline returns.
    """
    frames = int(round(0.35 * FRAME_RATE_HZ))
    c = _blank_contour(frames)
    _paint(c, 0, int(round(0.30 * FRAME_RATE_HZ)), 67, salience=0.9)
    events = [_ne(0.0, 0.30, 72)]

    wrapper_events, wrapper_stats = backfill_melody_notes(c, events)
    direct_melody, direct_chords, direct_stats = extract_melody(
        c, events, backfill_enabled=True, split_enabled=False,
    )

    # Wrapper drops the always-empty chord list from the tuple.
    assert direct_chords == []
    # Same melody events: the wrapper is a pure delegation.
    assert wrapper_events == direct_melody
    # Stats line up — both paths ran the same Viterbi + back-fill.
    assert wrapper_stats.backfilled_note_count == direct_stats.backfilled_note_count
    assert wrapper_stats.melody_note_count == direct_stats.melody_note_count
    # And the wrapper actually back-filled the MIDI 67 stable run that
    # no input event covered — the whole point of the additive path.
    assert wrapper_stats.backfilled_note_count == 1
    assert sorted(e[2] for e in wrapper_events) == [67, 72]


def test_split_enabled_false_backfill_still_skips_duplicates():
    # When ``split_enabled=False`` passes all events through,
    # back-fill's overlap check sees the full list — so a synthesized
    # run that would duplicate an existing event at the same pitch
    # must still be suppressed. This is the reason the split bypass
    # exists in ``extract_melody`` itself rather than via a separate
    # ``backfill_only`` helper that only sees the melody subset.
    frames = int(round(0.35 * FRAME_RATE_HZ))
    c = _blank_contour(frames)
    _paint(c, 0, int(round(0.30 * FRAME_RATE_HZ)), 67, salience=0.9)
    events = [_ne(0.0, 0.30, 67, amp=0.8)]

    melody, _, stats = extract_melody(c, events, split_enabled=False)
    assert stats.backfilled_note_count == 0
    assert len(melody) == 1
    assert melody[0][3] == 0.8  # original amplitude preserved


# ---------------------------------------------------------------------------
# Config defaults sanity check
# ---------------------------------------------------------------------------

def test_config_defaults_match_module_defaults():
    from backend.config import Settings

    s = Settings()
    assert s.melody_low_midi == DEFAULT_MELODY_LOW_MIDI
    assert s.melody_high_midi == DEFAULT_MELODY_HIGH_MIDI
    assert s.melody_voicing_floor == DEFAULT_VOICING_FLOOR
    assert s.melody_transition_weight == DEFAULT_TRANSITION_WEIGHT
    assert s.melody_max_transition_bins == DEFAULT_MAX_TRANSITION_BINS
    assert s.melody_match_fraction == DEFAULT_MATCH_FRACTION
    assert s.melody_extraction_enabled is True
    assert s.melody_backfill_enabled is DEFAULT_BACKFILL_ENABLED
    assert s.melody_backfill_min_duration_sec == DEFAULT_BACKFILL_MIN_DURATION_SEC
    assert s.melody_backfill_overlap_fraction == DEFAULT_BACKFILL_OVERLAP_FRACTION
    assert s.melody_backfill_min_amp == DEFAULT_BACKFILL_MIN_AMP
    assert s.melody_backfill_max_amp == DEFAULT_BACKFILL_MAX_AMP
