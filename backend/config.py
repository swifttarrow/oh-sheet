"""Application settings.

All values can be overridden via environment variables prefixed with
``OHSHEET_`` (e.g. ``OHSHEET_BLOB_ROOT=/var/lib/ohsheet/blob``).
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OHSHEET_",
        env_file=".env",
        extra="ignore",
    )

    # Where the LocalBlobStore writes its files. Returned URIs are file:// based.
    blob_root: Path = Path("./blob")

    # CORS — wide open for dev; tighten in deployment.
    cors_origins: list[str] = ["*"]

    # Worker timeout used by OrchestratorCommand envelopes.
    job_timeout_sec: int = 600

    # ---- Basic Pitch transcription -----------------------------------------
    # Tunable knobs passed through to basic_pitch.inference.predict(). Defaults
    # mirror upstream (basic_pitch.constants.DEFAULT_*). The ONNX model ships
    # inside the basic-pitch wheel, so there's no checkpoint path to configure.
    basic_pitch_onset_threshold: float = 0.5
    basic_pitch_frame_threshold: float = 0.3
    basic_pitch_minimum_note_length_ms: float = 127.7

    # ---- Transcription cleanup (Phase 1 post-processing) -------------------
    # Heuristic thresholds applied to Basic Pitch's note_events before we
    # rebuild pretty_midi. See backend/services/transcription_cleanup.py for
    # the semantics; these pass through as keyword args to cleanup_note_events.
    cleanup_merge_gap_sec: float = 0.03
    cleanup_octave_amp_ratio: float = 0.6
    cleanup_octave_onset_tol_sec: float = 0.05
    cleanup_ghost_max_duration_sec: float = 0.06
    cleanup_ghost_amp_median_scale: float = 0.5

    # ---- Melody extraction (Phase 2 post-processing) -----------------------
    # Viterbi-based melody / chord split driven by Basic Pitch's
    # ``model_output["contour"]`` salience matrix. See
    # backend/services/melody_extraction.py for semantics. Disable via
    # ``OHSHEET_MELODY_EXTRACTION_ENABLED=false`` to keep the legacy
    # single-PIANO output. Defaults mirror the DEFAULT_* constants in the
    # extraction module so config and tests agree.
    melody_extraction_enabled: bool = True
    melody_low_midi: int = 55                    # G3
    melody_high_midi: int = 90                   # F#6
    melody_voicing_floor: float = 0.15
    melody_transition_weight: float = 0.25
    melody_max_transition_bins: int = 12         # ≈ 4 semitones / frame
    melody_match_fraction: float = 0.6

    # Back-fill of stable Viterbi runs with no matching Basic Pitch note.
    # See _backfill_missed_melody_notes in melody_extraction.py.
    melody_backfill_enabled: bool = True
    melody_backfill_min_duration_sec: float = 0.12
    melody_backfill_overlap_fraction: float = 0.5
    melody_backfill_min_amp: float = 0.15
    melody_backfill_max_amp: float = 0.60

    # ---- Bass extraction (Phase 3 post-processing) ------------------------
    # Same Viterbi trick as melody extraction, run over the low-register
    # slice of the contour matrix. Accepts the non-melody events from
    # Phase 2 and splits them into BASS / remaining buckets. Defaults
    # mirror bass_extraction.DEFAULT_* so config and tests agree.
    bass_extraction_enabled: bool = True
    bass_low_midi: int = 28                      # E1
    bass_high_midi: int = 55                     # G3
    bass_voicing_floor: float = 0.12
    bass_transition_weight: float = 0.40
    bass_max_transition_bins: int = 9            # ≈ 3 semitones / frame
    bass_match_fraction: float = 0.55

    # ---- Chord recognition (Phase 3 post-processing) ----------------------
    # librosa chroma_cqt + 24 triad templates, beat-synced via the same
    # beat tracker that drives the tempo map. Labels attach to
    # ``HarmonicAnalysis.chords``; notes are unaffected. Disable via
    # ``OHSHEET_CHORD_RECOGNITION_ENABLED=false``.
    chord_recognition_enabled: bool = True
    chord_min_template_score: float = 0.55
    chord_hpss_margin: float = 3.0               # librosa.effects.harmonic margin


settings = Settings()
