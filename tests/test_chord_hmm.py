"""Unit tests for HMM smoothing and 7th chord templates in chord recognition.

Tests cover:
  * Template construction with and without 7th chords
  * HMM Viterbi smoothing with synthetic score matrices
  * Diatonic transition matrix for major and minor keys
  * Dominant-7th to major resolution preference in C major
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from backend.services.chord_recognition import (  # noqa: E402
    DEFAULT_CHORD_HMM_SELF_TRANSITION,
    DEFAULT_CHORD_HMM_TEMPERATURE,
    _build_chord_templates,
    _diatonic_labels_for_key,
    _smooth_chords_hmm,
)


# ---------------------------------------------------------------------------
# Template construction
# ---------------------------------------------------------------------------

class TestBuildChordTemplates:
    """Verify template shape, labels, and normalization."""

    def test_60_templates_when_seventh_enabled(self):
        templates, labels, roots = _build_chord_templates(seventh_enabled=True)
        assert templates.shape == (60, 12)
        assert len(labels) == 60
        assert len(roots) == 60

    def test_24_templates_when_seventh_disabled(self):
        templates, labels, roots = _build_chord_templates(seventh_enabled=False)
        assert templates.shape == (24, 12)
        assert len(labels) == 24
        assert len(roots) == 24

    def test_triad_labels_structure(self):
        _, labels, _ = _build_chord_templates(seventh_enabled=True)
        # First 12: major triads; next 12: minor triads.
        assert all(lbl.endswith(":maj") for lbl in labels[:12])
        assert all(lbl.endswith(":min") for lbl in labels[12:24])
        # 7th chord labels.
        assert all(lbl.endswith(":maj7") for lbl in labels[24:36])
        assert all(lbl.endswith(":min7") for lbl in labels[36:48])
        assert all(lbl.endswith(":7") for lbl in labels[48:60])

    def test_templates_are_l2_normalized(self):
        templates, _, _ = _build_chord_templates(seventh_enabled=True)
        norms = np.linalg.norm(templates, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_templates_are_l2_normalized_triads_only(self):
        templates, _, _ = _build_chord_templates(seventh_enabled=False)
        norms = np.linalg.norm(templates, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_seventh_template_has_four_nonzero_bins(self):
        """Each 7th chord template should activate exactly 4 chroma bins."""
        templates, _, _ = _build_chord_templates(seventh_enabled=True)
        # Check a major 7th (index 24 = C:maj7).
        nonzero = np.count_nonzero(templates[24])
        assert nonzero == 4
        # Check a dominant 7th (index 48 = C:7).
        nonzero = np.count_nonzero(templates[48])
        assert nonzero == 4

    def test_default_returns_60(self):
        """Default call (no args) returns 60 templates."""
        templates, labels, roots = _build_chord_templates()
        assert templates.shape == (60, 12)


# ---------------------------------------------------------------------------
# Diatonic label construction
# ---------------------------------------------------------------------------

class TestDiatonicLabels:
    """Verify diatonic set construction for key-aware HMM transitions."""

    def test_c_major_diatonic_triads(self):
        labels = _diatonic_labels_for_key("C:major")
        # Expected triads: C:maj, D:min, E:min, F:maj, G:maj, A:min, B:min
        for expected in ["C:maj", "D:min", "E:min", "F:maj", "G:maj", "A:min", "B:min"]:
            assert expected in labels, f"{expected} not in C:major diatonic set"

    def test_c_major_includes_seventh_extensions(self):
        labels = _diatonic_labels_for_key("C:major")
        # Major roots get maj7 and 7 extensions.
        assert "C:maj7" in labels
        assert "C:7" in labels
        assert "G:maj7" in labels
        # Minor roots get min7.
        assert "D:min7" in labels
        assert "A:min7" in labels

    def test_a_minor_diatonic_triads(self):
        labels = _diatonic_labels_for_key("A:minor")
        # Natural minor: A:min, B:min, C:maj, D:min, E:min, F:maj, G:maj
        for expected in ["A:min", "B:min", "C:maj", "D:min", "E:min", "F:maj", "G:maj"]:
            assert expected in labels, f"{expected} not in A:minor diatonic set"

    def test_unknown_key_returns_empty(self):
        labels = _diatonic_labels_for_key("X:dorian")
        assert labels == set()

    def test_invalid_format_returns_empty(self):
        labels = _diatonic_labels_for_key("no-colon")
        assert labels == set()

    def test_g_major_diatonic(self):
        labels = _diatonic_labels_for_key("G:major")
        # G major: G:maj, A:min, B:min, C:maj, D:maj, E:min, F#:min
        assert "G:maj" in labels
        assert "D:maj" in labels
        assert "F#:min" in labels


# ---------------------------------------------------------------------------
# HMM Viterbi smoothing
# ---------------------------------------------------------------------------

class TestSmoothChordsHMM:
    """Verify Viterbi decoding on synthetic score matrices."""

    def _make_simple_setup(self, seventh_enabled: bool = False):
        """Helper: build templates and return (labels, roots, n_templates)."""
        _, labels, roots = _build_chord_templates(seventh_enabled=seventh_enabled)
        return labels, roots, len(labels)

    def test_hmm_smooths_flickering_labels(self):
        """Raw argmax flickers A-B-A-B-A; HMM should smooth to all A."""
        labels, roots, n = self._make_simple_setup(seventh_enabled=False)

        n_spans = 10
        scores = np.full((n, n_spans), 0.1, dtype=np.float32)

        # C:maj is index 0, G:maj is index 7.
        c_idx = labels.index("C:maj")
        g_idx = labels.index("G:maj")

        # Alternate: C gets slightly higher score on even beats,
        # G gets slightly higher on odd beats. The raw argmax would
        # flicker between C and G.
        for t in range(n_spans):
            if t % 2 == 0:
                scores[c_idx, t] = 0.9
                scores[g_idx, t] = 0.85
            else:
                scores[c_idx, t] = 0.85
                scores[g_idx, t] = 0.87

        path = _smooth_chords_hmm(
            scores, labels, roots,
            key_label="C:major",
            self_transition=0.8,
            temperature=1.0,
        )

        # The HMM should smooth — the path should NOT flicker every beat.
        # Count how many transitions occur.
        transitions = sum(1 for i in range(1, len(path)) if path[i] != path[i - 1])
        # With self_transition=0.8, the path should have at most 1-2
        # transitions, not 9 (alternating every beat).
        assert transitions <= 2, (
            f"HMM did not smooth flickering: {transitions} transitions in {n_spans} spans"
        )

    def test_hmm_returns_correct_length(self):
        labels, roots, n = self._make_simple_setup()
        n_spans = 5
        scores = np.random.default_rng(42).random((n, n_spans)).astype(np.float32)
        path = _smooth_chords_hmm(scores, labels, roots)
        assert len(path) == n_spans

    def test_hmm_empty_spans(self):
        labels, roots, n = self._make_simple_setup()
        scores = np.zeros((n, 0), dtype=np.float32)
        path = _smooth_chords_hmm(scores, labels, roots)
        assert path == []

    def test_hmm_single_span(self):
        """Single-span input should return the argmax."""
        labels, roots, n = self._make_simple_setup(seventh_enabled=False)
        scores = np.zeros((n, 1), dtype=np.float32)
        c_idx = labels.index("C:maj")
        scores[c_idx, 0] = 1.0
        path = _smooth_chords_hmm(scores, labels, roots)
        assert len(path) == 1
        assert path[0] == c_idx

    def test_hmm_strong_signal_overrides_transition(self):
        """When the emission is overwhelmingly strong, HMM follows it."""
        labels, roots, n = self._make_simple_setup(seventh_enabled=False)
        n_spans = 6
        scores = np.full((n, n_spans), 0.01, dtype=np.float32)

        c_idx = labels.index("C:maj")
        g_idx = labels.index("G:maj")

        # First 3 spans: strong C:maj. Last 3: strong G:maj.
        for t in range(3):
            scores[c_idx, t] = 0.99
        for t in range(3, 6):
            scores[g_idx, t] = 0.99

        path = _smooth_chords_hmm(
            scores, labels, roots,
            key_label="C:major",
            self_transition=0.8,
        )
        # Should be C,C,C,G,G,G (one transition).
        assert all(path[t] == c_idx for t in range(3))
        assert all(path[t] == g_idx for t in range(3, 6))

    def test_dominant_to_tonic_resolution_preferred(self):
        """G:7 -> C:maj should have higher transition probability than
        G:7 -> F#:min in C:major key.

        We use self_transition=0.3 so the HMM is willing to change chords
        between spans, and give both C:maj and F#:min equal emission on
        span 1 so the transition prior is the tiebreaker.
        """
        labels, roots, n = self._make_simple_setup(seventh_enabled=True)

        g7_idx = labels.index("G:7")
        c_maj_idx = labels.index("C:maj")

        # Build a 3-span score matrix. First span is clearly G:7,
        # second span has equal scores for C:maj and a non-diatonic chord,
        # third span also favours C:maj to anchor the path.
        scores = np.full((n, 3), 0.01, dtype=np.float32)
        scores[g7_idx, 0] = 0.95

        # Second span: C:maj and F#:min have equal emission.
        fsharp_min_idx = labels.index("F#:min")
        scores[c_maj_idx, 1] = 0.7
        scores[fsharp_min_idx, 1] = 0.7

        # Third span: strongly C:maj so the path has somewhere to go.
        scores[c_maj_idx, 2] = 0.95

        path = _smooth_chords_hmm(
            scores, labels, roots,
            key_label="C:major",
            self_transition=0.3,  # low self-transition so HMM is willing to change
        )
        # The HMM should prefer C:maj at span 1 because it's diatonic in
        # C:major, giving it a higher transition probability from G:7.
        assert path[1] == c_maj_idx, (
            f"Expected C:maj ({c_maj_idx}) but got {labels[path[1]]} ({path[1]})"
        )

    def test_temperature_sharpening(self):
        """Lower temperature should make the HMM more confident in emission."""
        labels, roots, n = self._make_simple_setup(seventh_enabled=False)
        n_spans = 5
        scores = np.full((n, n_spans), 0.3, dtype=np.float32)
        c_idx = labels.index("C:maj")
        # C:maj only slightly above the crowd.
        scores[c_idx, :] = 0.4

        path_cold = _smooth_chords_hmm(
            scores, labels, roots,
            temperature=0.5,  # sharper
        )
        # With temperature < 1, the slight score advantage for C:maj is
        # amplified, so we expect mostly C:maj.
        c_count = sum(1 for p in path_cold if p == c_idx)
        assert c_count >= 3, f"Expected mostly C:maj with cold temperature, got {c_count}/5"

    def test_diatonic_transition_major_vs_minor(self):
        """Diatonic sets for major and minor keys should differ."""
        c_major = _diatonic_labels_for_key("C:major")
        c_minor = _diatonic_labels_for_key("C:minor")
        # In C major, E:min is diatonic; in C minor, D#:maj (Eb:maj) is.
        assert "E:min" in c_major
        assert "E:min" not in c_minor
        # D#:maj is diatonic in C minor (III = Eb:maj = D#:maj in our naming).
        assert "D#:maj" in c_minor
        assert "D#:maj" not in c_major
