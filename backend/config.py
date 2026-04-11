"""Application settings.

All values can be overridden via environment variables prefixed with
``OHSHEET_`` (e.g. ``OHSHEET_BLOB_ROOT=/var/lib/ohsheet/blob``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.contracts import ScorePipelineMode


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OHSHEET_",
        env_file=".env",
        extra="ignore",
    )

    # Where the LocalBlobStore writes its files. Returned URIs are file:// based.
    blob_root: Path = Path("./blob")

    # Redis URL for Celery broker + result backend.
    redis_url: str = "redis://localhost:6379/0"

    # CORS — wide open for dev; tighten in deployment.
    cors_origins: list[str] = ["*"]

    # Worker timeout used by OrchestratorCommand envelopes.
    job_timeout_sec: int = 600

    # Logging level for ``backend.*`` (pipeline, jobs, services). The root
    # logger defaults to WARNING, so without configuration ``log.info`` would
    # not appear when you run the API server.
    log_level: str = "INFO"

    # ---- Basic Pitch transcription -----------------------------------------
    # Tunable knobs passed through to basic_pitch.inference.predict(). Defaults
    # mirror upstream (basic_pitch.constants.DEFAULT_*). The ONNX model ships
    # inside the basic-pitch wheel, so there's no checkpoint path to configure.
    basic_pitch_onset_threshold: float = 0.5
    basic_pitch_frame_threshold: float = 0.3
    basic_pitch_minimum_note_length_ms: float = 127.7

    # Per-stem overrides — applied when running Basic Pitch on Demucs-
    # separated stems instead of the full mix. ``None`` means "use the
    # base value above". Tested 0.25 — regressed (-0.018 F1) because
    # stem separation artifacts produce false positives at lower
    # thresholds.
    #
    # Generic per-stem fallbacks (used when no stem-specific override
    # exists below). Kept for backward compatibility — operators who set
    # ``OHSHEET_BASIC_PITCH_STEM_ONSET_THRESHOLD`` still get their
    # value applied to any stem without a dedicated override.
    basic_pitch_stem_onset_threshold: float | None = None
    basic_pitch_stem_frame_threshold: float | None = None

    # Stem-specific Basic Pitch thresholds — take precedence over the
    # generic ``basic_pitch_stem_*`` above, which in turn take precedence
    # over the global ``basic_pitch_*`` defaults. Each stem has different
    # signal characteristics after Demucs separation.

    # Vocals stem: monophonic singing — raise onset threshold to reject
    # consonant-driven false positives, lower frame threshold slightly to
    # recover soft sustained notes that BP's polyphonic default gates out.
    basic_pitch_stem_onset_threshold_vocals: float = 0.6
    basic_pitch_stem_frame_threshold_vocals: float = 0.25

    # Bass stem: low-frequency notes produce weaker activations in BP's
    # mel spectrogram. Use global defaults — tested onset=0.4/frame=0.35
    # but it regressed bass F1 by -0.019 due to over-detection of
    # sub-harmonic artifacts at lower thresholds.
    basic_pitch_stem_onset_threshold_bass: float = 0.5
    basic_pitch_stem_frame_threshold_bass: float = 0.3

    # Other stem (chords/accompaniment): polyphonic, use defaults close
    # to global but slightly tighter onset to reduce bleed artifacts.
    basic_pitch_stem_onset_threshold_other: float = 0.5
    basic_pitch_stem_frame_threshold_other: float = 0.3

    # ---- Audio pre-processing (runs before Basic Pitch) --------------------
    # HPSS + RMS normalization applied to the waveform before inference.
    # See backend/services/audio_preprocess.py for semantics; the defaults
    # here mirror the DEFAULT_* constants in that module.
    #
    # Enabled=False by default. The measurement run against
    # assets/rising-sun-{1,2,3}.mp3 (scripts/bench_preprocess.py) showed:
    #   * Note counts shift by at most ±10% with preprocessing on — the
    #     cleanup_* and basic_pitch_*_threshold defaults above are robust
    #     to preprocessing and do NOT need a preprocessing-aware profile.
    #   * Cleanup's merge-fragmented-sustains pass drops ~25% of its
    #     workload because HPSS heals most frame-level activation dips
    #     at the source. A welcome side-effect, not a retune driver.
    #   * Octave-ghost counts stay flat (2→1, 3→3, 1→0) — the
    #     octave_amp_ratio threshold is ratio-based and self-corrects.
    #   * Overall confidence improves by +0.00 to +0.04 on the three
    #     fixtures. Real but marginal.
    #   * Cost: ~1.3–1.5s of extra wall time per inference for HPSS +
    #     normalize + tempfile write.
    # The three fixtures are correlated (same piece, three takes) so the
    # evidence is not strong enough to flip the default globally. Users
    # with drum-heavy or dynamic-range-varied material can opt in via
    # OHSHEET_AUDIO_PREPROCESS_ENABLED=1 without touching any other knob.
    audio_preprocess_enabled: bool = False
    audio_preprocess_hpss_enabled: bool = True
    audio_preprocess_hpss_margin: float = 1.0        # librosa default — gentle
    audio_preprocess_normalize_enabled: bool = True
    audio_preprocess_target_rms_dbfs: float = -20.0
    audio_preprocess_peak_ceiling_dbfs: float = -1.0

    # ---- Transcription cleanup (Phase 1 post-processing) -------------------
    # Heuristic thresholds applied to Basic Pitch's note_events before we
    # rebuild pretty_midi. See backend/services/transcription_cleanup.py for
    # the semantics; these pass through as keyword args to cleanup_note_events.
    cleanup_merge_gap_sec: float = 0.03
    cleanup_octave_amp_ratio: float = 0.6
    cleanup_octave_onset_tol_sec: float = 0.05
    cleanup_ghost_max_duration_sec: float = 0.05
    cleanup_ghost_amp_median_scale: float = 0.5

    # Per-stem cleanup overrides — applied when running Basic Pitch
    # on Demucs-separated stems. Stems have cleaner separation than
    # full mixes, so octave ghosts at 0.5x amplitude ratio are more
    # likely artifacts, and ghost notes can be filtered more aggressively.
    cleanup_stem_octave_amp_ratio: float = 0.5
    cleanup_stem_ghost_max_duration_sec: float = 0.04

    # Per-role cleanup — bass sustains longer than melody, chords have
    # more legitimate octave doublings than vocals. These override the
    # global cleanup_* defaults when cleanup_for_role() is used.
    cleanup_melody_merge_gap_sec: float = 0.02   # tighter than global 0.03
    cleanup_melody_ghost_max_duration_sec: float = 0.04  # tighter than global 0.05
    cleanup_bass_merge_gap_sec: float = 0.04     # slightly looser than global 0.03
    cleanup_bass_ghost_max_duration_sec: float = 0.06    # slightly looser than global 0.05
    cleanup_chords_merge_gap_sec: float = 0.04   # moderate
    cleanup_chords_octave_amp_ratio: float = 0.5  # stricter — real chord octave doublings are common

    # Energy gating (Pass 5) — trims suspiciously long note offsets
    # based on amplitude envelope decay or a duration/amplitude heuristic.
    cleanup_energy_gate_enabled: bool = True
    cleanup_energy_gate_max_sustain_sec: float = 2.0
    cleanup_energy_gate_floor_ratio: float = 0.1
    cleanup_energy_gate_tail_sec: float = 0.05

    # ---- Onset refinement (post-cleanup spectral onset snapping) -----------
    # After cleanup passes complete, snap note onsets to nearby peaks in a
    # higher-resolution spectral onset-strength function. This corrects the
    # ~23 ms quantization inherent in Basic Pitch's hop_length=512 grid.
    # See backend/services/onset_refine.py for semantics.
    onset_refine_enabled: bool = True
    onset_refine_max_shift_sec: float = 0.05     # maximum onset adjustment
    onset_refine_hop_length: int = 256           # hop length for onset detection (finer than BP's 512)

    # ---- Melody extraction (Phase 2 post-processing) -----------------------
    # Viterbi-based melody / chord split driven by Basic Pitch's
    # ``model_output["contour"]`` salience matrix. See
    # backend/services/melody_extraction.py for semantics. Disable via
    # ``OHSHEET_MELODY_EXTRACTION_ENABLED=false`` to keep the legacy
    # single-PIANO output. Defaults mirror the DEFAULT_* constants in the
    # extraction module so config and tests agree.
    melody_extraction_enabled: bool = True
    melody_low_midi: int = 48                    # C3 — head voice / contralto floor
    melody_high_midi: int = 96                   # C7 — riffing-soprano ceiling
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
    chord_seventh_templates_enabled: bool = True
    chord_hmm_enabled: bool = True
    chord_hmm_self_transition: float = 0.8
    chord_hmm_temperature: float = 1.0

    # ---- Post-arrangement simplification (sheet music readability) --------
    # Aggressive filter that runs AFTER ArrangeService to drop density from
    # raw transcription levels (~2000 notes per song) down to something a
    # human could actually read and play. See
    # backend/services/arrange_simplify.py for the five-step pipeline.
    #
    # Tuning history:
    # v1 (min_vel=40, max_onsets=6): RH -19%, LH -0% on real songs. LH
    #    untouched because bass notes from Basic Pitch rarely dip below 40.
    # v2 (min_vel=55, max_onsets=4): raises the floor above typical bass
    #    velocity range and caps density at sight-reading levels.
    arrange_simplify_enabled: bool = True
    arrange_simplify_min_velocity: int = 55       # drop anything quieter
    arrange_simplify_chord_merge_beats: float = 0.125   # merge within 1/32 note
    arrange_simplify_max_onsets_per_beat: int = 4       # density cap
    arrange_simplify_min_duration_beats: float = 0.25   # 16th note floor

    # ---- Demucs source separation (pre-Basic Pitch) -----------------------
    # When enabled, the transcribe stage runs Demucs over the source
    # waveform to split it into {drums, bass, other, vocals} and routes
    # each stem to a dedicated downstream consumer:
    #   * vocals  → Basic Pitch → MELODY events
    #   * bass    → Basic Pitch → BASS events
    #   * other   → Basic Pitch → CHORDS events (+ chord recognition)
    #   * drums   → tempo_map beat tracking
    # See backend/services/stem_separation.py for the semantics; the
    # defaults here mirror the DEFAULT_* constants in that module.
    #
    # Enabled by default — the stems path is the preferred pipeline
    # whenever the demucs extra is installed. Any failure (missing
    # dep, load crash, apply OOM, all-stems-empty, ...) falls back
    # transparently to the original single-mix Basic Pitch path, so
    # leaving this on is safe even on boxes where demucs/torch
    # aren't installed.
    #
    # Latency budget operators should know about:
    #   * Demucs separation itself is heavy: ~80 MB weights,
    #     0.2–0.5x real-time on CPU, 2–3x real-time on Apple MPS,
    #     5–10x on CUDA. Pay this once per job.
    #   * After separation the stems path runs Basic Pitch three
    #     times — once per stem (vocals/bass/other). The three
    #     passes run **in parallel** by default (see
    #     ``demucs_parallel_stems`` below), and since the bulk of
    #     Basic Pitch's wall time happens in GIL-releasing C
    #     extensions (ONNX Runtime, librosa, numpy) on a multi-core
    #     host the three passes overlap down to roughly one Basic
    #     Pitch pass of wall time. On a single-core host or with
    #     parallelism disabled, expect ~3x Basic Pitch cost instead.
    #
    # Rough wall-clock with the defaults (multi-core + parallel):
    #     stems path ≈ 1x Demucs + 1x Basic Pitch
    #     single-mix path ≈ 1x Basic Pitch
    # So flipping Demucs on costs you approximately one full Demucs
    # pass in additional wall time — roughly 2–10x the Basic Pitch
    # cost depending on CPU/GPU.
    #
    # The default htdemucs pretrained weights are CC BY-NC 4.0.
    # Commercial deployments must either swap in a commercially
    # licensed model, train their own, or set
    # ``OHSHEET_DEMUCS_ENABLED=0`` to force the single-mix path.
    demucs_enabled: bool = True
    demucs_model: str = "htdemucs"
    demucs_device: str | None = None             # None → auto: cuda → mps → cpu
    demucs_segment_sec: float | None = None      # None → model's own default
    demucs_shifts: int = 1                       # upstream default; >1 improves SDR
    demucs_overlap: float = 0.25
    demucs_split: bool = True

    # Parallel Basic Pitch inference across stems. When enabled, the
    # three per-stem passes run concurrently in a ThreadPoolExecutor
    # sharing the cached ``basic_pitch.inference.Model`` — safe because
    # the underlying ONNX / CoreML sessions are documented thread-safe
    # and ``basic_pitch.inference`` has no module-level mutable state.
    # Disable to reproduce the old serial behavior (useful for
    # debugging single-thread traces or for hosts where the process
    # is already CPU-saturated and parallelism would just thrash).
    demucs_parallel_stems: bool = True
    # Upper bound on concurrent stem workers. The effective worker
    # count is ``min(this, active_stem_count)`` — usually 3
    # (vocals/bass/other). Raising this above 3 has no effect today
    # but leaves room for future per-stem fan-out (e.g. a secondary
    # accompaniment pass on the ``other`` stem).
    demucs_parallel_max_workers: int = 3

    # Per-consumer routing. Each flag gates whether the corresponding
    # stem is used; flipping individual switches off is the escape
    # hatch when one stem turns out to be unreliable on a given corpus
    # (e.g. Demucs drums on a cappella material is noisy — disable
    # beats-from-drums and let audio_timing fall back to the mix).
    demucs_use_vocals_for_melody: bool = True
    demucs_use_bass_stem: bool = True
    demucs_use_other_for_chords: bool = True     # both notes + chord labels
    demucs_use_drums_for_beats: bool = True

    # ---- CREPE vocal melody (replaces Basic Pitch on vocals stem) ---------
    # Basic Pitch is a polyphonic tracker and has known weaknesses on
    # monophonic singing — ghost notes on legato phrases, poor vibrato
    # tracking, consonant-driven onset jitter. CREPE (Kim et al. 2018)
    # is a trained-on-singing F0 estimator and is SOTA for this exact
    # task. When enabled, ``extract_vocal_melody_crepe`` runs on the
    # vocals stem and its output is routed directly to MELODY, skipping
    # the Basic Pitch vocals pass entirely. Missing torchcrepe or any
    # runtime failure falls back to the Basic Pitch vocals pass so
    # flipping this on is always safe.
    #
    # Default: **on**. The first A/B against the 25-file clean_midi
    # baseline (seed=42, max_duration=30) came in net-neutral:
    #
    #                          no-crepe   with-crepe     delta
    #   mean F1 (no-offset)      0.375       0.377      +0.002
    #   median F1 (no-offset)    0.382       0.368      -0.014
    #   mean precision           0.424       0.445      +0.021
    #   mean wall sec/file        4.2         9.4        2.2x
    #   melody role F1           0.089       0.057      -0.032
    #
    # Per-file: 14 improvements, 10 regressions, similar magnitudes
    # on both sides. CREPE is more precise but emits fewer notes, and
    # the role-breakdown metric (lower bound against full ground
    # truth) regresses because CREPE's selectivity hurts the mis-match
    # accounting.
    #
    # After a parameter sweep (voicing_threshold ∈ [0.35..0.55],
    # merge_gap_sec ∈ [0.03..0.30]) on the 25-file clean_midi baseline,
    # the vt=0.45 / mg=0.15 combination is the best trade-off:
    #
    #                    baseline   CREPE(0.41/0.3)   CREPE(0.45/0.15)
    #   mean F1            0.361       0.373             0.375
    #   melody F1          0.073       0.052             0.063
    #   chords F1          0.345       0.341             0.346
    #   mean precision     0.389       0.442             0.434
    #   mean recall        0.376       0.359             0.370
    #   per-file W/L       —           10/5              14/6
    #   wall sec/file      9.1         28.4              17.2
    #
    # vt=0.41/mg=0.3 over-merged notes (melody F1 dropped); vt=0.45
    # rejects more consonant noise while mg=0.15 bridges legato without
    # merging distinct notes. Enabled by default — any failure (missing
    # torchcrepe, runtime crash) falls back to Basic Pitch on the vocals
    # stem, so leaving this on is always safe.
    #
    # All knobs wire through to ``extract_vocal_melody_crepe`` and can
    # be swept via ``OHSHEET_CREPE_*`` env vars. Defaults mirror
    # ``DEFAULT_*`` in ``backend.services.crepe_melody``.
    crepe_vocal_melody_enabled: bool = True
    crepe_model: str = "full"                    # "tiny" (2 MB) or "full" (22 MB)
    crepe_device: str | None = None              # None → auto: cuda → mps → cpu
    crepe_voicing_threshold: float = 0.45
    crepe_median_filter_frames: int = 7          # 70 ms at 100 Hz — better vibrato smoothing
    crepe_min_note_duration_sec: float = 0.06
    crepe_merge_gap_sec: float = 0.15
    crepe_amp_min: float = 0.25
    crepe_amp_max: float = 0.85

    # Hybrid CREPE+BP fusion — use CREPE's pitch accuracy with BP's
    # onset/offset timing. When enabled, both CREPE and the BP vocals
    # pass run and their outputs are fused. When disabled, CREPE
    # replaces BP entirely (the pre-hybrid behavior).
    crepe_hybrid_enabled: bool = True
    crepe_hybrid_bp_min_amp: float = 0.35  # BP notes below this amp are dropped in fusion
    crepe_hybrid_overlap_threshold: float = 0.5  # min temporal overlap fraction to fuse
    crepe_max_pitch_leap: int = 12  # max semitone leap before octave-snap kicks in

    # ---- Key + time-signature detection -----------------------------------
    # Basic Pitch does not estimate key or meter, so the transcribe stage
    # otherwise hardcodes ``"C:major"`` and ``(4, 4)`` into every
    # HarmonicAnalysis — the engrave stage then renders those literally
    # and every piece ships in C major 4/4 regardless of actual tonality.
    # See backend/services/key_estimation.py for the estimators; both are
    # best-effort and fall back to the hardcoded defaults on any failure.
    #
    # Enabled by default — the eval harness (scripts/eval_transcription.py)
    # shows no quality regression relative to the hardcoded defaults
    # because downstream engrave only reads ``analysis.key`` for the key
    # signature rendering (wrong key → wrong accidentals, which is the
    # single most visible symptom in the PDF output). A confidence floor
    # on the key estimator keeps it from claiming labels on atonal /
    # percussion-heavy audio.
    key_detection_enabled: bool = True
    key_min_confidence: float = 0.55
    meter_detection_enabled: bool = True
    meter_confidence_margin: float = 0.05
    meter_min_beats: int = 8

    # Chord-based key cross-validation. Refines the KS key estimate
    # by checking what fraction of detected chords are diatonic to the
    # candidate keys. Helps resolve Am/C relative-key confusions.
    key_chord_validation_enabled: bool = True
    key_chord_diatonic_threshold: float = 0.6
    key_chord_flip_margin: float = 0.15

    # ---- Beat-synchronous note snapping (post-quantization) ---------------
    # After quantization, notes can drift from true beat positions due to
    # cumulative tempo-map error. This correction layer checks whether
    # shifting each note onset by ±1 grid step better aligns it with
    # detected beats (integer beats + subdivision). Only shifts that
    # improve alignment by more than ``snap_weight`` are applied, and
    # voice collisions are respected. Pure beat-space pass — no I/O.
    arrange_beat_snap_enabled: bool = True
    arrange_beat_snap_weight: float = 0.3
    arrange_beat_snap_subdivision: float = 0.5

    # ---- Beat tracker backend ------------------------------------------------
    # "madmom" (default) uses madmom DBNBeatProcessor — more robust for
    # variable-tempo music.  "librosa" uses the legacy librosa.beat.beat_track.
    # "auto" tries madmom first, falls back to librosa.
    beat_tracker: Literal["madmom", "librosa", "auto"] = "madmom"

    # ---- Adaptive quantization grid (arrange stage) --------------------------
    # Instead of a rigid 1/16th-note grid, estimate the best-fit grid per
    # piece from candidates. Improves triplet-feel and swing music.
    arrange_adaptive_grid_enabled: bool = True
    arrange_grid_candidates: str = "0.167,0.25,0.333,0.5"
    arrange_min_notes_for_grid_estimation: int = 4

    # ---- Duration refinement (per-pitch CQT energy gating) ------------------
    # Refine note offsets by tracking energy decay at each note's specific
    # pitch via CQT spectrogram.  More precise than global RMS gating.
    #
    # Tuned via eval sweep (25-file clean_midi, seed=42):
    #   floor_ratio  tail_sec  F1(no-off)  F1(w/off)  dur_MAE  dur_ratio
    #       0.15       0.03       0.373      0.132     639ms     4.28
    #       0.40       0.01       0.378      0.136     590ms     3.97  ← chosen
    #       0.55       0.01       0.371      0.130     569ms     4.27
    # floor=0.40 recovers baseline F1 while cutting duration MAE by 49ms.
    duration_refine_enabled: bool = True
    duration_refine_floor_ratio: float = 0.40
    duration_refine_tail_sec: float = 0.01
    duration_refine_min_duration_sec: float = 0.03
    duration_refine_hop_length: int = 256

    @field_validator(
        "cleanup_energy_gate_floor_ratio",
        "cleanup_octave_amp_ratio",
        "cleanup_stem_octave_amp_ratio",
        "cleanup_chords_octave_amp_ratio",
        "duration_refine_floor_ratio",
        "crepe_voicing_threshold",
        "melody_backfill_overlap_fraction",
        "crepe_hybrid_overlap_threshold",
        "key_chord_diatonic_threshold",
        "key_chord_flip_margin",
        "demucs_overlap",
    )
    @classmethod
    def _validate_unit_ratio(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator(
        "chord_hmm_self_transition",
    )
    @classmethod
    def _validate_probability(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError(f"must be between 0.0 and 1.0 (exclusive), got {v}")
        return v

    @field_validator(
        "chord_hmm_temperature",
        "chord_hpss_margin",
        "audio_preprocess_hpss_margin",
    )
    @classmethod
    def _validate_positive_float(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError(f"must be positive, got {v}")
        return v

    # Score path after transcription (or MIDI-derived TranscriptionResult).
    # Env: ``OHSHEET_SCORE_PIPELINE`` — ``arrange`` (default) or ``condense_transform``.
    score_pipeline: ScorePipelineMode = "arrange"


settings = Settings()
