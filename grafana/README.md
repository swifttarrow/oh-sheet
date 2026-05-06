# Grafana — Oh Sheet eval dashboards

Phase 7 production telemetry surface. The runner emits a row to
`eval_production_quality_scores` (Postgres) on every successful
engrave; this directory holds the matching Grafana dashboards.

## Apply

1. Stand up a Postgres reachable from Grafana and the Oh Sheet runner.
2. Apply the schema:

   ```bash
   psql "$OHSHEET_EVAL_TELEMETRY_DSN" \
     -f backend/eval/migrations/0001_initial.sql
   ```
3. In Grafana, add a Postgres datasource with UID `postgres-eval`
   pointing at that DSN. Read-only Grafana credentials are fine —
   the dashboards only `SELECT`.
4. Import `dashboards/oh-sheet-eval.json` via
   **Dashboards → Import → Upload JSON file**.
5. Set the dashboard's `datasource` template variable to your
   Postgres datasource if the default UID doesn't match.

## What you get

* **Production composite-Q (per-job)** — sparkline of `composite_q`
  from `backend.eval.telemetry.compute_production_quality_report`.
  Strategy doc §8.2 weighting (Phase 7 production-flavored: Tier 3 +
  Tier 2-lite; Tier 4 stays in nightly because resynth is too heavy
  inline).
* **Distribution histogram** — left-skew or bimodality is the signal
  to chase. A bump <0.55 is the "lower confidence" bucket; a bump
  >0.75 is "ready to play".
* **Tier 3 sub-metrics** — playability, voice-leading, sight-
  readability, all hourly means. Each has its own panel so a
  regression on a single sub-metric isn't laundered by the
  composite.
* **Tier 2 chord-symbol presence rate** — the metric that proves
  Phase 2's `_emit_chord_markers` is still firing in production.
* **Engraving warnings per job** — count of ledger-excess /
  voice-crossing / hand-crossing warnings; spikes here mean a
  scarier-on-the-page score is shipping.
* **Engrave route mix** — local vs. remote_http vs. fallback.
  Phase 4's local engraver should dominate; fallback growth means
  a regression in `engrave_local`.
* **Recent low-quality jobs (Q < 0.55)** — top-20 worst jobs in the
  window; click `job_id` to dig in.

## Privacy

`user_audio_hash` is SHA-256 of the user's audio bytes — never the
audio itself. The audio bytes are never written to Postgres. The
hash is stable across re-runs so the dashboard can de-dup repeat
uploads of the same source.

## Operations

* `q_version` is bumped when the §8.2 weights or sub-metric set
  changes (e.g. when Tier 5 calibration ships and we get true v1.0
  weights). Dashboards filter on `$q_version` template variable so
  v0.1 readings aren't conflated with future v1.0 readings.
* The schema includes a commented-out 30-day retention DELETE you
  can wire to `pg_cron` if disk pressure becomes an issue.
