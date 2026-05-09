"""Application settings.

All values can be overridden via environment variables prefixed with
``OHSHEET_`` (e.g. ``OHSHEET_BLOB_ROOT=/var/lib/ohsheet/blob``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import computed_field, field_validator
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

    # Maximum upload sizes for /v1/uploads/{audio,midi}. Enforced by streaming
    # the request body and aborting with HTTP 413 once the cumulative byte
    # count exceeds the cap (Content-Length is not trusted). MIDI files are
    # tiny in practice, so the MIDI cap is much smaller than the audio cap.
    max_audio_upload_mb: int = 100
    max_midi_upload_mb: int = 5

    # Redis URL for Celery broker + result backend.
    redis_url: str = "redis://localhost:6379/0"

    # CORS — wide open for dev; tighten in deployment.
    cors_origins: list[str] = ["*"]

    # Worker timeout used by OrchestratorCommand envelopes.
    job_timeout_sec: int = 600

    # Cap on the number of completed (succeeded/failed) JobRecords retained
    # in the in-memory JobManager. When this limit is exceeded, the oldest
    # completed records are evicted in insertion order. Running/pending jobs
    # are never evicted. Set high enough to cover your typical
    # status-polling window; long-running deployments without this cap
    # accumulate JobRecords for the process lifetime.
    max_completed_jobs: int = 1000

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
    # Minimum allowed output frequency in Hz. Default 30 Hz lets through
    # the lowest piano A0 (~27.5 Hz) plus typical sub-bass content; raise
    # it (e.g. 80 Hz) to suppress hip-hop kick artifacts that BP loves to
    # mistake for sustained low pitches. ``None`` would mean "no floor"
    # — we keep an explicit floor to avoid regressions on hyped-bass mixes.
    minimum_frequency_hz: float = 30.0
    # Allow Basic Pitch to emit overlapping notes carrying pitch-bend
    # contours. Required to preserve vibrato/portamento on vocal/guitar
    # input; the lift-through is wired in transcribe_inference.py.
    multiple_pitch_bends: bool = True

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
    # v3 (min_vel=25, max_onsets=4): lowered from 55 after the Phase 0
    #    pop_mini_v0 eval showed v2's floor was clipping legitimate quiet
    #    accompaniment notes (especially on cover-style transcriptions
    #    where AMT-APC emits soft inner voices). 25 still suppresses the
    #    sub-noise-floor garbage that v1 was tuned against. Note this is
    #    only validated against the pop_mini_v0 5-song set; revisit if
    #    bass-artifact regressions surface on a larger corpus.
    # Two-hand voice caps. Standard piano notation supports up to four
    # voices per staff (two stems-up, two stems-down) but engraver
    # backends differ in how cleanly they render voices 3 and 4. Raised
    # from 2 to 4 to preserve inner-voice density that a stricter cap
    # would silently drop; revisit if the engraver chokes on real input.
    arrange_max_voices_rh: int = 4
    arrange_max_voices_lh: int = 4

    arrange_simplify_enabled: bool = True
    arrange_simplify_min_velocity: int = 25       # drop anything quieter
    arrange_simplify_chord_merge_beats: float = 0.125   # merge within 1/32 note
    arrange_simplify_max_onsets_per_beat: int = 4       # density cap
    arrange_simplify_min_duration_beats: float = 0.25   # 16th note floor

    # ---- Pop2Piano transcription (replaces Demucs + Basic Pitch) -----------
    # When enabled, the transcribe stage runs Pop2Piano (sweetcocoa/pop2piano)
    # on the source audio to produce a single piano MIDI directly, replacing
    # both Demucs source separation and Basic Pitch polyphonic tracking in
    # one shot. The output is a pretty_midi object whose notes are converted
    # to NoteEvent tuples and fed through the same post-processing pipeline
    # (melody/bass extraction, onset/duration refinement, key/chord/tempo
    # estimation) as the single-mix Basic Pitch path.
    #
    # Pop2Piano is a seq2seq transformer trained on pop music → piano covers;
    # it produces cleaner piano reductions than the Demucs+BP pipeline on
    # most pop/rock material because it was explicitly trained for the task
    # rather than doing generic pitch tracking on separated stems.
    #
    # Any failure (missing deps, model load crash, inference error) falls
    # back transparently to the old Demucs+BP path, so leaving this on is
    # safe even on boxes where torch/transformers aren't installed.
    #
    # Dependencies: torch, transformers, librosa, resampy, scipy,
    # pretty_midi, essentia (see [pop2piano] extra in pyproject.toml).
    pop2piano_enabled: bool = False
    pop2piano_model: str = "sweetcocoa/pop2piano"
    pop2piano_sample_rate: int = 44100
    pop2piano_device: str | None = None  # None → auto: cuda → mps → cpu
    pop2piano_composer: str = "composer1"  # Pop2Piano "composer" token

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

    # ---- Kong piano-stem AMT (Phase 6) -----------------------------------
    # When enabled, the transcribe stage routes a piano-ish summed stem
    # (Demucs ``bass + other``) into ByteDance Kong's high-resolution
    # piano transcription model — a CRNN-Regress AMT that emits per-note
    # onset/offset/velocity AND sustain pedal CC64 on/off events. Pedal
    # tracking is the unique deliverable here: no other commercially-
    # licensed pip-installable transcriber emits sustained-pedal data,
    # and pedal is the difference between "blizzard of staccato eighths"
    # and a readable score on piano-cover material.
    #
    # Routing rules (see ``backend/services/transcribe.py``):
    #   * Pre-separated stems must exist (Phase 5 Demucs ran successfully).
    #     Kong is MAESTRO-overfit (Edwards et al. 2024: −19.2 F1 on
    #     pitch-shift, −10.1 on reverb), so we never feed it raw audio.
    #   * Vocal energy on the vocals stem must be below a threshold OR
    #     the user has explicitly hinted ``piano`` — i.e. the source is
    #     piano-dominant. On vocal-heavy pop, Basic Pitch on the
    #     ``other`` stem still wins.
    #
    # Any failure (missing piano_transcription_inference, weight fetch,
    # GPU OOM) falls back transparently to the Basic Pitch stems path,
    # so flipping this on is always safe.
    #
    # Dependencies: piano_transcription_inference (MIT), torch, librosa
    # — see [kong] extra in pyproject.toml. Weights are fetched from
    # Zenodo on first use; pre-cache in Docker build to avoid Cloud Run
    # cold-start latency (~170 MB).
    kong_enabled: bool = True
    # When ``True``, route through Kong only when the bundle's metadata
    # asks for it (user_hint == "piano"). Default False so the routing
    # gates on vocal-energy heuristics for unhinted jobs.
    kong_user_hint_only: bool = False
    # Vocal-energy threshold (RMS on the vocals stem, normalized to
    # [0, 1]). Below this we treat the input as piano-dominant and
    # prefer Kong; at or above we keep the Basic Pitch stems pipeline.
    # 0.05 was chosen against the Phase 0 mini-eval — sparse-vocals
    # singer-songwriter material lands ~0.03; pop with full vocal
    # presence lands ~0.10–0.20.
    kong_vocal_energy_threshold: float = 0.05
    # Device override for the Kong model. None → auto: cuda → mps → cpu.
    kong_device: str | None = None
    # Optional checkpoint override. None → use the model's bundled default.
    kong_checkpoint_path: str | None = None
    # Pedal-event confidence floor — Kong emits a probability per pedal
    # segment; below this we drop the event rather than render a noisy
    # ``Ped. ___ *`` bracket.
    kong_pedal_min_confidence: float = 0.5

    # ---- AMT-APC piano cover (Phase 8) ------------------------------------
    # Audio→Piano-Cover transcriber from misya11p/amt-apc — an MIT-licensed
    # hFT-Transformer descendant trained on YouTube piano-cover videos
    # paired with their source pop tracks. Unlike Kong (faithful
    # transcription) and Basic Pitch (generic polyphonic AMT), AMT-APC
    # produces a *pianistic cover*: idiomatic LH accompaniment patterns,
    # melody re-voicings, and arrangement decisions that a human cover
    # pianist would make. Surfaced as the "Piano cover" UI toggle.
    #
    # Routing: bound to the ``pop_cover`` PipelineVariant. The dispatcher
    # in ``transcribe.py`` invokes AMT-APC when ``user_hint == "cover"``
    # OR ``settings.amt_apc_enabled`` is True and the bundle came in
    # with the cover variant. The ingest-stage user_hint is set from the
    # frontend's "Faithful / Piano cover" toggle.
    #
    # Pop2Piano sibling: AMT-APC supersedes Pop2Piano for cover-style
    # transcription (license-clean: AMT-APC is MIT, Pop2Piano has no
    # LICENSE file in the upstream repo, see strategy doc §G3).
    #
    # Any failure (missing dep, weight fetch, inference crash) raises
    # ImportError/RuntimeError and the dispatcher falls back to the
    # Kong/BP path so a deploy without the [amt_apc] extra still
    # produces a transcription (just not the cover-mode rearrangement).
    #
    # Dependencies: amt_apc (via the [amt_apc] extra in pyproject.toml),
    # torch, librosa. Weights are fetched on first use; pre-cache in
    # Docker build to avoid Cloud Run cold-start latency (~100 MB).
    #
    # Defaults to True because the user-facing "Piano cover" toggle in
    # the frontend sets ``user_hint=cover`` / ``variant=pop_cover`` and
    # would otherwise be silently ignored at the kill-switch. When the
    # ``[amt_apc]`` extra isn't installed the dispatcher still falls
    # back to the faithful Kong/Basic-Pitch path, so leaving this on
    # is safe even on deployments without the optional dep.
    amt_apc_enabled: bool = True
    # Sample rate AMT-APC expects — 22.05 kHz mono per the upstream README.
    amt_apc_sample_rate: int = 22050
    # Device override. None → auto: cuda → mps → cpu.
    amt_apc_device: str | None = None
    # Optional checkpoint override. None → use the model's bundled default.
    amt_apc_checkpoint_path: str | None = None
    # AMT-APC's inference style preset. The upstream model exposes
    # ``style_token`` (e.g. "amateur" / "professional") that nudges the
    # decoder toward thicker or thinner arrangements. Default to
    # ``professional`` for richer voicings; expose as a setting so an
    # operator can A/B without touching code.
    amt_apc_style: str = "professional"

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
    # "beat_this" (default) — CPJKU's MIT-licensed transformer; emits
    # beats AND downbeats. "librosa" uses the legacy librosa.beat.beat_track
    # (beats only). "auto" tries Beat This! first, falls back to librosa.
    beat_tracker: Literal["beat_this", "librosa", "auto"] = "beat_this"

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

    # ---- Piano cover search (ingest fast path) ----------------------------
    # When a job arrives with ``prefer_clean_source=True``, the ingest
    # stage probes the user's YouTube URL for (title, artist), searches
    # for a clean piano cover via yt-dlp + scoring, and swaps the URL
    # for the cover's URL before transcription. Basic Pitch is a
    # polyphonic tracker that transcribes every audible pitch — drums,
    # vocals, bass — as piano notes, so feeding it a monophonic piano
    # cover produces a dramatically cleaner result than a full-band mix.
    # See backend/services/cover_search.py for the matching logic.
    #
    # ``cover_search_enabled`` is the operator kill switch. Leave it
    # on by default so the per-job ``prefer_clean_source`` flag decides
    # whether to run; flip to False in production to disable the whole
    # feature without needing a code change (e.g. if yt-dlp search
    # breaks or the allowlist is producing bad matches on a new corpus).
    #
    # ``cover_search_min_score`` is the scoring threshold that a
    # candidate must clear to trigger a URL swap. See the module
    # docstring in cover_search.py for the scoring rules; the default
    # of 60 was chosen from dry-run testing against real YouTube.
    # Raise to 70 for "allowlist-only" strict matching, lower to 50 if
    # you're happy to accept weaker title-only matches.
    cover_search_enabled: bool = True
    cover_search_min_score: int = 60

    # ---- Refine stage (LLM-driven score annotation) -------------------------
    # The refine stage uses Anthropic Claude + the built-in web_search tool to
    # produce human-readable score metadata (title, composer, key, tempo
    # marking, section structure, repeats). See backend/services/refine.py.
    # refine_enabled is derived: True when anthropic_api_key is set.
    # Set OHSHEET_REFINE_ENABLED=false to force-disable even with a key.
    refine_enabled: bool | None = None        # None = auto (key present → on)
    refine_model: str = "claude-sonnet-4-6"
    refine_max_searches: int = 5              # web_search cap per refinement
    refine_budget_sec: int = 300              # overall wall-time budget
    refine_call_timeout_sec: int = 120        # per-API-call timeout
    anthropic_api_key: str | None = None

    # ---- Interpret stage (prompt → ArrangementHints via Claude) -------------
    # Global kill switch. Still gated per-job by whether the user supplied
    # a prompt. Set OHSHEET_INTERPRET_ENABLED=false to disable globally
    # (e.g. during Anthropic outages) without needing a code change.
    interpret_enabled: bool = True
    # Model for the interpret stage. Haiku is fast and cheap — sufficient
    # for structured extraction from a short user prompt.
    interpret_model: str = "claude-haiku-4-5"
    # Maximum characters accepted from the user's arrangement prompt.
    # Prompts longer than this are truncated before being sent to Claude.
    interpret_prompt_max_chars: int = 1000
    # Per-process sliding-window cap on Anthropic calls from the interpret
    # stage. Bounds blast radius when a user batch-submits many prompted
    # jobs against the same worker (each Celery worker process maintains
    # its own counter — global production cap is roughly
    # ``cap × worker_concurrency``). Set to 0 to disable the limit. When
    # exceeded, the stage skips the LLM call and the input txr passes
    # through unchanged with a ``rate_limited`` warning appended.
    interpret_max_calls_per_minute: int = 30

    @computed_field  # type: ignore[prop-decorator]
    @property
    def refine_active(self) -> bool:
        """Whether the refine stage actually runs.

        Auto-derives from key presence unless explicitly overridden via
        OHSHEET_REFINE_ENABLED.
        """
        if self.refine_enabled is not None:
            return self.refine_enabled
        return self.anthropic_api_key is not None

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
    # Env: ``OHSHEET_SCORE_PIPELINE`` — ``arrange`` or ``condense_only`` (default).
    # condense_only splits arrangement into two stages: condense
    # (flatten all tracks into one piano stream) then transform (apply
    # difficulty shaping). Produces a denser, more complete MIDI than
    # the single-pass arrange mode.
    #
    # ``condense_transform`` is accepted as a deprecated alias of
    # ``condense_only`` for one release — deployed environments may
    # still carry the old value in their .env.
    score_pipeline: ScorePipelineMode = "arrange"

    @field_validator("score_pipeline", mode="before")
    @classmethod
    def _alias_condense_transform(cls, v: object) -> object:
        if v == "condense_transform":
            import warnings
            warnings.warn(
                "OHSHEET_SCORE_PIPELINE='condense_transform' is deprecated; "
                "use 'condense_only'. Support will be removed in the next release.",
                DeprecationWarning,
                stacklevel=2,
            )
            return "condense_only"
        return v

    # ── TuneChat integration ──────────────────────────────────────────
    # When enabled, Oh Sheet sends audio to TuneChat's transcription
    # API in parallel with its own pipeline. TuneChat returns a
    # higher-quality 30-second preview (sheet music + MIDI). Oh Sheet
    # shows its own result immediately, then upgrades to TuneChat's
    # result when it arrives.
    #
    # Env: OHSHEET_TUNECHAT_ENABLED (default: false — opt-in)
    # Env: OHSHEET_TUNECHAT_URL (the base URL of a running TuneChat)
    # Env: OHSHEET_TUNECHAT_API_KEY (Bearer token for /api/v1/transcribe)
    # Env: OHSHEET_TUNECHAT_TIMEOUT_SEC (max wait before giving up)
    #
    # Note: ``tunechat_timeout_sec`` is the synchronous upper bound for
    # ``httpx.AsyncClient.post`` inside the ingest stage. Because the
    # runner ``await``s the response before returning, the Celery worker
    # handling the job is pinned for up to this duration (5 min default).
    # Size worker concurrency accordingly when TuneChat is enabled.
    tunechat_enabled: bool = False
    tunechat_url: str = "http://localhost:3000"
    tunechat_api_key: str = ""
    tunechat_timeout_sec: int = 300

    # ---- Engrave backend selection ----------------------------------------
    # ``local``       — music21 → MusicXML + LilyPond → PDF, in-process.
    #                   Phase 4 default. The structured ``(PianoScore,
    #                   ExpressionMap)`` flows directly through music21 so
    #                   chord symbols, dynamics, pedal marks, and per-note
    #                   voices survive the rendering boundary intact.
    # ``remote_http`` — POST MIDI bytes to oh-sheet-ml-pipeline /engrave,
    #                   receive MusicXML bytes (no PDF). Pre-Phase 4 path.
    #
    # Both backends are wired in :mod:`backend.jobs.runner` — the local
    # backend additionally falls through to the remote HTTP service when
    # ``EngraveLocalError`` fires, so a missing LilyPond install on a dev
    # machine degrades gracefully to MusicXML-only via the remote path.
    #
    # Env: ``OHSHEET_ENGRAVE_BACKEND`` — ``local`` or ``remote_http``.
    engrave_backend: Literal["local", "remote_http"] = "local"

    # ---- ML engraver service ----------------------------------------------
    # When ``engrave_backend = "remote_http"`` (or when the local backend
    # falls through), audio_upload and midi_upload jobs route engraving
    # through the oh-sheet-ml-pipeline HTTP service (POST {url}/engrave,
    # MIDI bytes → MusicXML bytes). title_lookup jobs are expected to
    # resolve upstream via TuneChat; if they reach the engrave stage, the
    # local backend renders a fallback artifact rather than crashing.
    #
    # Env: OHSHEET_ENGRAVER_SERVICE_URL, OHSHEET_ENGRAVER_SERVICE_TIMEOUT_SEC
    engraver_service_url: str = "http://localhost:8080"
    engraver_service_timeout_sec: int = 60

    # ── YouTube bot-detection bypass (yt-dlp cookies) ─────────────────
    # YouTube periodically flags known data-center IPs (GCP, AWS, etc.)
    # as bot traffic and demands a signed-in session. When that happens,
    # ingest + cover_search users see "Sign in to confirm you're not a
    # bot" from yt-dlp and the job fails before it reaches transcription.
    #
    # Setting this to a path of a Netscape-format cookies.txt file (from
    # a logged-in browser session) routes yt-dlp's requests through that
    # session and bypasses the bot check. Leave empty to run anonymously
    # — works until YouTube rate-limits the VM.
    #
    # Deploy story: the deploy.yml writes the OHSHEET_YTDLP_COOKIES
    # GitHub secret to ~/oh-sheet/youtube-cookies.txt on the VM, and
    # docker-compose bind-mounts that file at /app/youtube-cookies.txt
    # inside the orchestrator + worker-ingest containers. Services
    # check file size > 0 before using the path, so an unset/empty
    # secret is a safe no-op.
    #
    # Env: OHSHEET_YTDLP_COOKIES_PATH
    # Refresh: see docs/ytdlp-cookies.md for browser-export instructions
    ytdlp_cookies_path: str | None = None

    # ---- Score-HPT velocity refinement (Phase 9A) -------------------------
    # Heuristic stand-in for Foscarin et al.'s Score-HPT (Hierarchical
    # Performance Transformer; paper-only at writing — the real ~1M-param
    # BiLSTM/Transformer head is intended to plug into the same input/
    # output contract once it lands). Re-estimates per-note velocities
    # from metric position, register, and onset density before arrange's
    # percentile-band remap runs. See backend/services/score_hpt.py.
    #
    # Default off until validated against the Phase 0 mini-eval and
    # Phase 3 pop eval set — the heuristic stand-in is documented to
    # improve dynamics on the bench fixtures but unverified on real pop.
    score_hpt_enabled: bool = False
    # 0.0 = keep transcriber velocities; 1.0 = full predicted blend.
    score_hpt_blend_alpha: float = 0.5
    # Velocity points added on a downbeat (when downbeats are tracked).
    score_hpt_downbeat_boost: float = 8.0
    # Velocity points added on a strong (integer) beat.
    score_hpt_beat_boost: float = 4.0
    # Velocity points subtracted on an off-beat onset.
    score_hpt_offbeat_attenuation: float = 4.0
    # Bell curve depth for register attenuation (0=disabled, 1=max).
    score_hpt_register_curve_strength: float = 0.3
    # Velocity adjustment magnitude for dense / sparse onset windows.
    score_hpt_density_compensation: float = 6.0
    # Hard floor + ceiling for the refined velocity range.
    score_hpt_min_velocity: int = 20
    score_hpt_max_velocity: int = 120

    # ---- Voice / staff GNN hand assignment (Phase 9B) ---------------------
    # Heuristic stand-in for Karystinaios & Widmer 2024 (arXiv:2407.21030)
    # — clusters notes into streams, then separates streams to RH/LH by
    # pitch centroid. Replaces the naive ``pitch >= 60`` middle-C split
    # in arrange.py for non-melody/non-bass tracks. See
    # backend/services/voice_gnn.py.
    #
    # Default off until validated. The real GNN replacement is intended
    # to load via the same ``assign_hands_gnn`` API once a labeled hand-
    # split eval is curated (strategy doc M8).
    voice_gnn_enabled: bool = False
    voice_gnn_pitch_weight: float = 1.0
    voice_gnn_time_weight: float = 4.0
    voice_gnn_velocity_weight: float = 0.05
    voice_gnn_join_threshold: float = 8.0
    voice_gnn_min_split_hint: int = 55          # A3 — keep split above
    voice_gnn_max_split_hint: int = 65          # F4 — keep split below

    # ---- Arrange backend (rules vs HF MIDI path) ---------------------------
    # ``rules`` — existing hand assignment + quantization in arrange.
    # ``hf_midi_identity`` — materialize MIDI → HF inference (see
    # ``arrange_hf_inference_mode``) → parse output → same ``_arrange_sync``.
    # Env: ``OHSHEET_ARRANGE_BACKEND``.
    arrange_backend: Literal["rules", "hf_midi_identity"] = "rules"
    # Sub-mode for ``hf_midi_identity`` (extensible when real weights ship).
    arrange_hf_inference_mode: Literal["identity"] = "identity"
    # On materialize/HF/parse failure, run classic arrange on the original txr.
    arrange_hf_fallback_to_rules: bool = True


settings = Settings()
