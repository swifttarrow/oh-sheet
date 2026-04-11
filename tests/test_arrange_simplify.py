"""TDD tests for the post-arrangement simplification pass.

These tests are written FIRST, before implementation, per the repo TDD rule.
They describe the behavior of ``simplify_score()`` which takes a PianoScore
and returns a simplified copy suitable for sheet-music engraving.
"""
from __future__ import annotations

from shared.contracts import (
    PianoScore,
    ScoreMetadata,
    ScoreNote,
    TempoMapEntry,
)


def _score(right: list[ScoreNote], left: list[ScoreNote]) -> PianoScore:
    """Build a minimal PianoScore for testing."""
    return PianoScore(
        right_hand=right,
        left_hand=left,
        metadata=ScoreMetadata(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
            difficulty="intermediate",
        ),
    )


def _note(
    id: str,
    pitch: int = 60,
    onset_beat: float = 0.0,
    duration_beat: float = 1.0,
    velocity: int = 80,
    voice: int = 0,
) -> ScoreNote:
    return ScoreNote(
        id=id,
        pitch=pitch,
        onset_beat=onset_beat,
        duration_beat=duration_beat,
        velocity=velocity,
        voice=voice,
    )


# ---------------------------------------------------------------------------
# Step 1: velocity threshold
# ---------------------------------------------------------------------------


class TestVelocityFilter:
    def test_drops_quiet_notes(self):
        from backend.services.arrange_simplify import simplify_score

        # Distinct onset beats so chord-merging doesn't collapse them.
        rh = [
            _note("rh-0", velocity=80, onset_beat=0.0),   # loud → keep
            _note("rh-1", velocity=30, onset_beat=1.0),   # quiet → drop
            _note("rh-2", velocity=50, onset_beat=2.0),   # at threshold → keep
            _note("rh-3", velocity=10, onset_beat=3.0),   # very quiet → drop
        ]
        result = simplify_score(_score(rh, []), min_velocity=40)
        kept_ids = {n.id for n in result.right_hand}
        assert kept_ids == {"rh-0", "rh-2"}

    def test_keeps_everything_when_threshold_zero(self):
        from backend.services.arrange_simplify import simplify_score

        # Distinct pitches and onsets so chord-merging leaves them alone.
        rh = [
            _note("rh-0", velocity=10, pitch=60, onset_beat=0.0),
            _note("rh-1", velocity=1,  pitch=62, onset_beat=1.0),
        ]
        result = simplify_score(_score(rh, []), min_velocity=0)
        assert len(result.right_hand) == 2


# ---------------------------------------------------------------------------
# Step 2: duration snapping
# ---------------------------------------------------------------------------


class TestDurationSnapping:
    def test_snaps_to_nearest_standard_duration(self):
        from backend.services.arrange_simplify import simplify_score

        rh = [
            _note("rh-0", duration_beat=0.27),   # → 0.25 (16th)
            _note("rh-1", duration_beat=0.48, onset_beat=1.0),  # → 0.5 (8th)
            _note("rh-2", duration_beat=0.95, onset_beat=2.0),  # → 1.0 (quarter)
            _note("rh-3", duration_beat=1.8, onset_beat=3.0),   # → 2.0 (half)
        ]
        result = simplify_score(_score(rh, []))
        durations = sorted(n.duration_beat for n in result.right_hand)
        assert durations == [0.25, 0.5, 1.0, 2.0]


# ---------------------------------------------------------------------------
# Step 3: micro-note pruning (runs AFTER snapping)
# ---------------------------------------------------------------------------


class TestMicroNotePruning:
    def test_drops_notes_shorter_than_min_duration(self):
        from backend.services.arrange_simplify import simplify_score

        # 0.1 snaps to 0.25 which is >= 0.25, so it stays
        # 0.05 snaps to 0.25 which is >= 0.25, so it stays
        # (The grid is [0.25, 0.5, 1.0, 2.0, 4.0] so nothing snaps below 0.25)
        # Test the explicit min_duration_beats > 0.25 case.
        rh = [
            _note("rh-0", duration_beat=0.25),  # → 0.25, kept
            _note("rh-1", duration_beat=0.5, onset_beat=1.0),  # → 0.5, kept
        ]
        result = simplify_score(_score(rh, []), min_duration_beats=0.5)
        kept = {n.id for n in result.right_hand}
        assert kept == {"rh-1"}


# ---------------------------------------------------------------------------
# Step 4: chord cluster merging
# ---------------------------------------------------------------------------


class TestChordMerging:
    def test_merges_near_onsets_to_single_chord_time(self):
        from backend.services.arrange_simplify import simplify_score

        # Three notes within 0.05 beats of each other — should become a C-major chord
        # at the earliest onset (0.0).
        rh = [
            _note("rh-0", pitch=60, onset_beat=0.0),
            _note("rh-1", pitch=64, onset_beat=0.05),
            _note("rh-2", pitch=67, onset_beat=0.09),
        ]
        result = simplify_score(_score(rh, []), chord_merge_beats=0.125)
        onsets = {n.onset_beat for n in result.right_hand}
        pitches = sorted(n.pitch for n in result.right_hand)
        assert onsets == {0.0}
        assert pitches == [60, 64, 67]

    def test_dedupes_same_pitch_in_cluster_keeping_loudest(self):
        from backend.services.arrange_simplify import simplify_score

        # Two notes at the same pitch within the merge window — keep the loudest.
        rh = [
            _note("rh-0", pitch=60, onset_beat=0.0, velocity=50),
            _note("rh-1", pitch=60, onset_beat=0.02, velocity=100),
            _note("rh-2", pitch=64, onset_beat=0.05, velocity=80),
        ]
        result = simplify_score(_score(rh, []), chord_merge_beats=0.125)
        pitch_60 = [n for n in result.right_hand if n.pitch == 60]
        assert len(pitch_60) == 1
        assert pitch_60[0].velocity == 100

    def test_distant_notes_not_merged(self):
        from backend.services.arrange_simplify import simplify_score

        # 0.5 beats apart — outside the merge window, should remain separate
        rh = [
            _note("rh-0", pitch=60, onset_beat=0.0),
            _note("rh-1", pitch=64, onset_beat=0.5),
        ]
        result = simplify_score(_score(rh, []), chord_merge_beats=0.125)
        onsets = {n.onset_beat for n in result.right_hand}
        assert onsets == {0.0, 0.5}


# ---------------------------------------------------------------------------
# Step 5: density cap
# ---------------------------------------------------------------------------


class TestDensityCap:
    def test_caps_distinct_onsets_per_beat(self):
        from backend.services.arrange_simplify import simplify_score

        # 8 distinct onsets in beat 0 (every 1/8 beat), cap is 4 — keep loudest 4
        rh = [
            _note(f"rh-{i}", pitch=60 + i, onset_beat=i * 0.125, velocity=50 + i * 5)
            for i in range(8)
        ]
        result = simplify_score(
            _score(rh, []),
            chord_merge_beats=0.01,  # tight so nothing merges
            max_onsets_per_beat=4,
            min_velocity=0,
        )
        distinct_onsets = {round(n.onset_beat, 4) for n in result.right_hand}
        assert len(distinct_onsets) == 4
        # loudest 4: i=4,5,6,7 → onsets 0.5, 0.625, 0.75, 0.875
        assert distinct_onsets == {0.5, 0.625, 0.75, 0.875}

    def test_under_cap_keeps_everything(self):
        from backend.services.arrange_simplify import simplify_score

        # 3 distinct onsets, cap 6 → all kept
        rh = [
            _note("rh-0", onset_beat=0.0),
            _note("rh-1", onset_beat=0.25, pitch=62),
            _note("rh-2", onset_beat=0.5, pitch=64),
        ]
        result = simplify_score(_score(rh, []), max_onsets_per_beat=6)
        assert len(result.right_hand) == 3


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------


class TestMetadataUntouched:
    def test_metadata_is_unchanged(self):
        from backend.services.arrange_simplify import simplify_score

        score = _score([_note("rh-0")], [])
        original_key = score.metadata.key
        original_tempo = score.metadata.tempo_map[0].bpm
        result = simplify_score(score)
        assert result.metadata.key == original_key
        assert result.metadata.tempo_map[0].bpm == original_tempo

    def test_both_hands_processed(self):
        from backend.services.arrange_simplify import simplify_score

        rh = [_note("rh-0", velocity=80), _note("rh-1", velocity=20)]
        lh = [_note("lh-0", pitch=40, velocity=80), _note("lh-1", pitch=41, velocity=20)]
        result = simplify_score(_score(rh, lh), min_velocity=40)
        assert len(result.right_hand) == 1
        assert len(result.left_hand) == 1
        assert result.right_hand[0].id == "rh-0"
        assert result.left_hand[0].id == "lh-0"
