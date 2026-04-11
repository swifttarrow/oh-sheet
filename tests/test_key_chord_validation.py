"""Unit tests for chord-based key cross-validation.

Tests the diatonic chord set builder and the ``refine_key_with_chords``
function that resolves relative major/minor confusions (Am vs C) by
checking whether detected chords better support the runner-up key.
"""
from __future__ import annotations

from backend.contracts import RealtimeChordEvent
from backend.services.key_estimation import (
    _diatonic_chords_for_key,
    refine_key_with_chords,
)

# ---------------------------------------------------------------------------
# Helpers — build chord event lists
# ---------------------------------------------------------------------------

def _chord(
    label: str,
    root: int,
    time_sec: float = 0.0,
    duration_sec: float = 1.0,
    confidence: float = 0.8,
) -> RealtimeChordEvent:
    return RealtimeChordEvent(
        label=label,
        root=root,
        time_sec=time_sec,
        duration_sec=duration_sec,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Diatonic chord set builder
# ---------------------------------------------------------------------------

class TestDiatonicChords:
    def test_c_major_diatonic_set(self):
        chords = _diatonic_chords_for_key("C:major")
        assert len(chords) == 7
        # I = C:maj
        assert (0, "maj") in chords
        # ii = D:min
        assert (2, "min") in chords
        # iii = E:min
        assert (4, "min") in chords
        # IV = F:maj
        assert (5, "maj") in chords
        # V = G:maj
        assert (7, "maj") in chords
        # vi = A:min
        assert (9, "min") in chords
        # vii = B:min (approximated)
        assert (11, "min") in chords

    def test_a_minor_diatonic_set(self):
        chords = _diatonic_chords_for_key("A:minor")
        assert len(chords) == 7
        # i = A:min
        assert (9, "min") in chords
        # ii = B:min (approximated)
        assert (11, "min") in chords
        # III = C:maj
        assert (0, "maj") in chords
        # iv = D:min
        assert (2, "min") in chords
        # v = E:min (natural minor)
        assert (4, "min") in chords
        # VI = F:maj
        assert (5, "maj") in chords
        # VII = G:maj
        assert (7, "maj") in chords

    def test_g_major_diatonic_set(self):
        chords = _diatonic_chords_for_key("G:major")
        assert len(chords) == 7
        # I = G:maj
        assert (7, "maj") in chords
        # ii = A:min
        assert (9, "min") in chords
        # V = D:maj
        assert (2, "maj") in chords

    def test_invalid_key_label_returns_empty(self):
        assert _diatonic_chords_for_key("") == set()
        assert _diatonic_chords_for_key("invalid") == set()
        assert _diatonic_chords_for_key("X:major") == set()


# ---------------------------------------------------------------------------
# refine_key_with_chords — flipping logic
# ---------------------------------------------------------------------------

class TestRefineKeyWithChords:
    def test_flips_am_to_c_when_chords_start_end_on_c(self):
        """Am and C share diatonic sets, so we rely on boundary chords.

        When the first and last strong chords land on C (root=0) and
        KS reported Am with C as runner-up, the function should flip.
        """
        chords = [
            _chord("C:maj", root=0, time_sec=0.0, confidence=0.9),
            _chord("F:maj", root=5, time_sec=1.0, confidence=0.8),
            _chord("G:maj", root=7, time_sec=2.0, confidence=0.8),
            _chord("C:maj", root=0, time_sec=3.0, confidence=0.9),
        ]
        key, stats = refine_key_with_chords(
            "A:minor", 0.75,
            "C:major", 0.72,
            chords,
            flip_margin=0.15,
        )
        assert key == "C:major"
        assert stats.chord_flipped
        assert stats.chord_validated

    def test_does_not_flip_when_confidence_gap_too_large(self):
        """A large KS confidence gap should prevent flipping."""
        chords = [
            _chord("C:maj", root=0, time_sec=0.0, confidence=0.9),
            _chord("F:maj", root=5, time_sec=1.0, confidence=0.8),
            _chord("G:maj", root=7, time_sec=2.0, confidence=0.8),
            _chord("C:maj", root=0, time_sec=3.0, confidence=0.9),
        ]
        key, stats = refine_key_with_chords(
            "A:minor", 0.90,
            "C:major", 0.60,
            chords,
            flip_margin=0.15,
        )
        assert key == "A:minor"
        assert not stats.chord_flipped
        assert stats.chord_validated

    def test_handles_empty_chord_labels(self):
        """Empty chord list should return the original KS key."""
        key, stats = refine_key_with_chords(
            "C:major", 0.80,
            "A:minor", 0.75,
            [],
        )
        assert key == "C:major"
        assert not stats.chord_flipped
        assert stats.chord_validated

    def test_keeps_ks_key_when_diatonic_fraction_high(self):
        """When most chords are diatonic to the KS key, keep it."""
        # Non-relative keys: D:major vs A:major
        chords = [
            _chord("D:maj", root=2, time_sec=0.0, confidence=0.9),
            _chord("G:maj", root=7, time_sec=1.0, confidence=0.8),
            _chord("A:maj", root=9, time_sec=2.0, confidence=0.8),
            _chord("D:maj", root=2, time_sec=3.0, confidence=0.9),
        ]
        key, stats = refine_key_with_chords(
            "D:major", 0.70,
            "A:major", 0.65,
            chords,
            diatonic_threshold=0.6,
        )
        assert key == "D:major"
        assert not stats.chord_flipped

    def test_flips_non_relative_when_runner_up_better(self):
        """When the runner-up has higher diatonic fraction and gap is small,
        flip to the runner-up (non-relative key pair)."""
        # Chords are all diatonic to G major but not to C major.
        chords = [
            _chord("G:maj", root=7, time_sec=0.0, confidence=0.8),
            _chord("D:maj", root=2, time_sec=1.0, confidence=0.8),
            _chord("F#:min", root=6, time_sec=2.0, confidence=0.8),
            _chord("G:maj", root=7, time_sec=3.0, confidence=0.8),
        ]
        key, stats = refine_key_with_chords(
            "C:major", 0.60,
            "G:major", 0.55,
            chords,
            diatonic_threshold=0.6,
            flip_margin=0.15,
        )
        assert key == "G:major"
        assert stats.chord_flipped

    def test_chords_with_7ths_still_match(self):
        """Chord labels like 'C:maj7' should still be recognized as diatonic
        by stripping the trailing digit."""
        chords = [
            _chord("C:maj7", root=0, time_sec=0.0, confidence=0.9),
            _chord("F:maj7", root=5, time_sec=1.0, confidence=0.8),
            _chord("G:maj7", root=7, time_sec=2.0, confidence=0.8),
            _chord("C:maj7", root=0, time_sec=3.0, confidence=0.9),
        ]
        key, stats = refine_key_with_chords(
            "C:major", 0.80,
            "A:minor", 0.75,
            chords,
        )
        # KS key's diatonic fraction should be 1.0 — all are C-major diatonic.
        assert key == "C:major"
        assert stats.chord_diatonic_fraction > 0.9

    def test_am_not_flipped_when_boundary_chords_favor_am(self):
        """When Am is the KS key and boundary chords start/end on Am's
        tonic, it should NOT flip to C."""
        chords = [
            _chord("A:min", root=9, time_sec=0.0, confidence=0.9),
            _chord("D:min", root=2, time_sec=1.0, confidence=0.8),
            _chord("E:min", root=4, time_sec=2.0, confidence=0.8),
            _chord("A:min", root=9, time_sec=3.0, confidence=0.9),
        ]
        key, stats = refine_key_with_chords(
            "A:minor", 0.75,
            "C:major", 0.72,
            chords,
            flip_margin=0.15,
        )
        assert key == "A:minor"
        assert not stats.chord_flipped

    def test_does_not_flip_relative_when_boundary_tied(self):
        """When boundary chords tie between KS tonic and runner-up tonic,
        keep the KS key (no flip on equal evidence)."""
        chords = [
            _chord("C:maj", root=0, time_sec=0.0, confidence=0.9),
            _chord("G:maj", root=7, time_sec=1.0, confidence=0.8),
            _chord("A:min", root=9, time_sec=2.0, confidence=0.9),
        ]
        key, stats = refine_key_with_chords(
            "A:minor", 0.75,
            "C:major", 0.72,
            chords,
            flip_margin=0.15,
        )
        # First chord = C (runner-up tonic), last chord = A:min (KS tonic)
        # That's 1 vs 1 — tied, so no flip.
        assert key == "A:minor"
        assert not stats.chord_flipped

    def test_stats_warnings_include_chord_validation(self):
        """The as_warnings output should mention chord cross-validation."""
        chords = [
            _chord("C:maj", root=0, time_sec=0.0, confidence=0.9),
        ]
        _key, stats = refine_key_with_chords(
            "C:major", 0.80,
            "A:minor", 0.75,
            chords,
        )
        warnings = stats.as_warnings()
        assert any("chord cross-validation" in w for w in warnings)

    def test_graceful_on_none_like_inputs(self):
        """Should not crash on edge cases."""
        # Chord labels with unparseable label
        chords = [
            _chord("N", root=-1, time_sec=0.0, confidence=0.1),
        ]
        key, stats = refine_key_with_chords(
            "C:major", 0.80,
            "A:minor", 0.75,
            chords,
        )
        assert key == "C:major"
        assert stats.chord_validated


# ---------------------------------------------------------------------------
# Config defaults sync
# ---------------------------------------------------------------------------

def test_config_chord_validation_defaults_match_module():
    """Settings defaults must match the module-level constants."""
    from backend.config import Settings
    from backend.services.key_estimation import (
        DEFAULT_KEY_CHORD_DIATONIC_THRESHOLD,
        DEFAULT_KEY_CHORD_FLIP_MARGIN,
    )

    s = Settings()
    assert s.key_chord_diatonic_threshold == DEFAULT_KEY_CHORD_DIATONIC_THRESHOLD
    assert s.key_chord_flip_margin == DEFAULT_KEY_CHORD_FLIP_MARGIN
