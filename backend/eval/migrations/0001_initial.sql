-- Phase 7 telemetry tables — strategy doc §6.1.
--
-- Apply via:
--   psql "$OHSHEET_EVAL_TELEMETRY_DSN" -f backend/eval/migrations/0001_initial.sql
--
-- Idempotent: re-running on an existing schema is a noop. Bump the
-- file number (0002_…sql) for additive changes; never edit a
-- migration in place once it's run on a deployed Postgres.

-- ---------------------------------------------------------------------------
-- eval_runs — one row per ``scripts/eval.py {ci|nightly|end-to-end}`` run.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id            UUID PRIMARY KEY,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    eval_set_version  TEXT NOT NULL,
    oh_sheet_sha      TEXT NOT NULL,
    config_hash       TEXT NOT NULL,
    label             TEXT,
    is_release_run    BOOLEAN NOT NULL DEFAULT FALSE,
    is_nightly        BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_eval_runs_created_at ON eval_runs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eval_runs_nightly    ON eval_runs (is_nightly, created_at DESC);

-- ---------------------------------------------------------------------------
-- eval_song_scores — per-song, per-tier, per-metric tall table.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eval_song_scores (
    run_id        UUID NOT NULL REFERENCES eval_runs(run_id) ON DELETE CASCADE,
    song_slug     TEXT NOT NULL,
    tier          TEXT NOT NULL CHECK (tier IN ('1', '2', '3', '4', '5', 'rf', 'composite')),
    metric_name   TEXT NOT NULL,
    metric_value  DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (run_id, song_slug, tier, metric_name)
);

CREATE INDEX IF NOT EXISTS idx_eval_song_scores_run         ON eval_song_scores (run_id);
CREATE INDEX IF NOT EXISTS idx_eval_song_scores_metric      ON eval_song_scores (metric_name, song_slug);

-- ---------------------------------------------------------------------------
-- eval_production_quality_scores — one row per live job.
--
-- Production telemetry surface: the runner emits a row here on every
-- engrave-stage success, so Grafana can sparkline the Q distribution
-- over recent uploads. ``user_audio_hash`` is SHA-256 of the audio
-- bytes — not the audio itself — so the table is GDPR-clean.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eval_production_quality_scores (
    id                                 BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id                             TEXT NOT NULL CHECK (length(job_id) > 0),
    created_at                         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_audio_hash                    TEXT NOT NULL,
    composite_quality_score            DOUBLE PRECISION NOT NULL,
    q_version                          TEXT NOT NULL,

    -- Tier 3 surface — the bulk of the production composite.
    tier3_playability_fraction         DOUBLE PRECISION,
    tier3_voice_leading_smoothness     DOUBLE PRECISION,
    tier3_polyphony_in_target_range    DOUBLE PRECISION,
    tier3_sight_readability            DOUBLE PRECISION,
    tier3_engraving_warning_count      INTEGER,
    tier3_composite                    DOUBLE PRECISION,

    -- Tier 2 lite (chord-symbol presence + key sanity).
    tier2_has_key                      BOOLEAN,
    tier2_chord_symbol_count           INTEGER,

    -- Diagnostics.
    engrave_route                      TEXT,
    title                              TEXT
);

CREATE INDEX IF NOT EXISTS idx_eval_prod_q_created_at  ON eval_production_quality_scores (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eval_prod_q_job_id      ON eval_production_quality_scores (job_id);
CREATE INDEX IF NOT EXISTS idx_eval_prod_q_q_version   ON eval_production_quality_scores (q_version, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eval_prod_q_audio_hash  ON eval_production_quality_scores (user_audio_hash);

-- Optional: 30-day retention via pg_cron — operations choose whether to enable.
-- DELETE FROM eval_production_quality_scores WHERE created_at < NOW() - INTERVAL '30 days';
