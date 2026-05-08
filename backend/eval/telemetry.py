"""Production-side composite-Q telemetry — Phase 7 §6.1 + §8.2.

Every production job that survives the engrave stage emits a
composite quality score ``Q ∈ [0, 1]`` into the
``eval_production_quality_scores`` Postgres table. Grafana reads
from there and surfaces a per-job sparkline + percentile chart
(see ``grafana/dashboards/oh-sheet-eval.json``).

Two halves of this module:

* :func:`compute_production_quality_report` — a *fast* metric pass
  that runs purely on the engraved ``HumanizedPerformance`` /
  ``PianoScore`` data already produced by the pipeline. No
  re-synthesis, no audio reload — only ``Tier 3`` (arrangement
  quality) is computed inline, plus an optional Tier 2 chord-symbol
  presence check that's cheap because chord events live on
  :class:`ScoreMetadata`. Tier 4 is too heavy for inline production
  telemetry; the nightly eval job handles that surface.

* :class:`TelemetryClient` — a thin wrapper over ``psycopg`` that
  inserts one row per production job. **No-ops when the DSN env var
  is unset**, so dev environments and Docker tests run without a
  database. Real Postgres is opt-in via
  ``OHSHEET_EVAL_TELEMETRY_DSN``.

The Postgres schema lives at ``backend/eval/migrations/0001_initial.sql``
(strategy doc §6.1 verbatim, lightly modernized for ``TIMESTAMPTZ`` +
``GENERATED ALWAYS AS IDENTITY``).
"""
from __future__ import annotations

import logging
import os
import uuid
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shared.contracts import HumanizedPerformance, PianoScore

log = logging.getLogger(__name__)

# Env var holding the Postgres DSN. Empty/unset = telemetry no-ops.
TELEMETRY_DSN_ENV = "OHSHEET_EVAL_TELEMETRY_DSN"

# Composite Q version. Bump when the §8.2 weights or sub-metric set
# changes. Stored alongside each row so dashboards can group by
# Q-version when a recalibration drops.
PRODUCTION_Q_VERSION = "Q_v0.1_phase7"


# ---------------------------------------------------------------------------
# Production quality report — what gets emitted per job
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProductionQualityReport:
    """Per-job composite + sub-metric breakdown.

    Mirrors the strategy doc §8.2 composite, but uses the inline-cheap
    sub-metric subset described in this module's header. The full
    nightly composite has Tier 4 (perceptual); this production-side
    composite drops to Tier 2 (chord-presence) + Tier 3 only.

    The ``q_version`` string is persisted alongside so a future
    Q recalibration (after Tier 5 calibration in Phase 8+) can
    distinguish v0.1 readings from v1.0 readings on the same chart.
    """

    composite_q: float
    q_version: str = PRODUCTION_Q_VERSION

    # Tier 3 surface — the bulk of the production composite.
    tier3_playability_fraction: float | None = None
    tier3_voice_leading_smoothness: float | None = None
    tier3_polyphony_in_target_range: float | None = None
    tier3_sight_readability: float | None = None
    tier3_engraving_warning_count: int | None = None
    tier3_composite: float | None = None

    # Tier 2 surface (chord-symbol presence + key-signature sanity).
    tier2_has_key: bool | None = None
    tier2_chord_symbol_count: int | None = None

    # Diagnostic: what subset actually contributed to ``composite_q``.
    contributing_terms: tuple[str, ...] = field(default_factory=tuple)

    def as_db_row(
        self,
        *,
        job_id: str,
        user_audio_hash: str,
        engrave_route: str | None = None,
        title: str | None = None,
    ) -> dict[str, Any]:
        """Return a dict with the column names the Postgres INSERT expects.

        ``job_id`` is the canonical Oh Sheet job id (UUID). Stored as
        TEXT in case future job systems use non-UUID ids; the schema
        has a ``CHECK (length(job_id) > 0)`` rather than a UUID type.
        ``user_audio_hash`` is the SHA-256 of the user's audio bytes —
        never the audio itself per §6.1. The audio hash is stable
        across re-runs so the dashboard can de-dup repeat uploads.
        """
        return {
            "job_id": job_id,
            "created_at": datetime.now(UTC),
            "user_audio_hash": user_audio_hash,
            "composite_quality_score": self.composite_q,
            "q_version": self.q_version,
            "tier3_playability_fraction": self.tier3_playability_fraction,
            "tier3_voice_leading_smoothness": self.tier3_voice_leading_smoothness,
            "tier3_polyphony_in_target_range": self.tier3_polyphony_in_target_range,
            "tier3_sight_readability": self.tier3_sight_readability,
            "tier3_engraving_warning_count": self.tier3_engraving_warning_count,
            "tier3_composite": self.tier3_composite,
            "tier2_has_key": self.tier2_has_key,
            "tier2_chord_symbol_count": self.tier2_chord_symbol_count,
            "engrave_route": engrave_route,
            "title": title,
        }

    def as_evaluation_report(self) -> dict[str, Any]:
        """Return a dict suitable for ``EngravedOutput.evaluation_report``.

        Trims the DB-only diagnostic fields and rounds floats so the
        JSON-serialized contract isn't full of 18-digit doubles.
        """
        out: dict[str, Any] = {
            "composite_q": round(self.composite_q, 4),
            "q_version": self.q_version,
            "contributing_terms": list(self.contributing_terms),
        }
        for k, v in asdict(self).items():
            if k in ("composite_q", "q_version", "contributing_terms"):
                continue
            if isinstance(v, float):
                out[k] = round(v, 4)
            else:
                out[k] = v
        return out


# ---------------------------------------------------------------------------
# Compute — fast inline pass over the engraved score
# ---------------------------------------------------------------------------

def compute_production_quality_report(
    *,
    score: PianoScore,
    perf: HumanizedPerformance | None = None,
) -> ProductionQualityReport:
    """Run the inline-cheap metric subset and assemble the §8.2 composite.

    Tier 3 is computed via :func:`eval.tier3_arrangement.compute_tier3`
    on the post-arrange ``PianoScore``. Tier 2 is the
    chord-symbol-presence + key-signature-presence quick check on
    ``ScoreMetadata`` — no audio reload, no resynth.

    The composite ``Q`` is a weighted average of the present terms:

    * Tier 3 composite (``0.5·play + 0.3·vleading + 0.2·density``) at weight 0.7
    * Tier 2 lite (``has_key + chord_symbol_presence`` averaged) at weight 0.3

    Different from the offline §8.2 composite (which weights Tier 4
    at 0.40) because production telemetry skips the resynth-heavy
    Tier 4 pass — see this module's header for rationale.
    """
    contributing: list[str] = []
    weights: list[tuple[float, float]] = []  # (weight, value)

    # ── Tier 3 ────────────────────────────────────────────────────────
    tier3_play: float | None = None
    tier3_vlead: float | None = None
    tier3_density: float | None = None
    tier3_readability: float | None = None
    tier3_warning_count: int | None = None
    tier3_composite: float | None = None
    try:
        from eval.tier3_arrangement import compute_tier3  # noqa: PLC0415

        t3 = compute_tier3(score)
        tier3_play = t3.playability_fraction
        tier3_vlead = t3.voice_leading_smoothness
        tier3_density = t3.polyphony_in_target_range
        tier3_readability = t3.sight_readability
        tier3_warning_count = len(t3.engraving_warnings)
        tier3_composite = t3.composite
        contributing.append("tier3_composite")
        weights.append((0.7, t3.composite))
    except Exception:  # noqa: BLE001
        log.exception("compute_production_quality_report: tier3 failed")

    # ── Tier 2 lite (chord-symbol + key presence) ─────────────────────
    has_key: bool | None = None
    chord_count: int | None = None
    try:
        meta = score.metadata
        has_key = bool(meta.key)
        chord_count = len(meta.chord_symbols)
        # Lightweight Tier 2 score: 1.0 when both key and chord symbols
        # are present, 0.5 when only one is present, 0.0 when neither.
        tier2_lite = 0.5 * (1.0 if has_key else 0.0) + 0.5 * (1.0 if chord_count else 0.0)
        contributing.append("tier2_lite")
        weights.append((0.3, tier2_lite))
    except Exception:  # noqa: BLE001
        log.exception("compute_production_quality_report: tier2 lite failed")

    # ── Composite ─────────────────────────────────────────────────────
    if weights:
        total_w = sum(w for w, _ in weights)
        composite_q = sum(w * v for w, v in weights) / total_w
    else:
        composite_q = 0.0

    return ProductionQualityReport(
        composite_q=max(0.0, min(1.0, composite_q)),
        tier3_playability_fraction=tier3_play,
        tier3_voice_leading_smoothness=tier3_vlead,
        tier3_polyphony_in_target_range=tier3_density,
        tier3_sight_readability=tier3_readability,
        tier3_engraving_warning_count=tier3_warning_count,
        tier3_composite=tier3_composite,
        tier2_has_key=has_key,
        tier2_chord_symbol_count=chord_count,
        contributing_terms=tuple(contributing),
    )


# ---------------------------------------------------------------------------
# Postgres telemetry client — opt-in via env var
# ---------------------------------------------------------------------------

class TelemetryClient:
    """Thin wrapper over psycopg that inserts one row per job.

    No-ops when ``OHSHEET_EVAL_TELEMETRY_DSN`` is unset OR when
    ``psycopg`` isn't importable. The pipeline runner instantiates a
    singleton at startup and calls :meth:`record` from the engrave
    stage; failures here MUST NOT take down a job, so every public
    method swallows + logs and returns ``None``.
    """

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = dsn if dsn is not None else os.environ.get(TELEMETRY_DSN_ENV, "")
        self._enabled = bool(self._dsn)
        self._psycopg = None
        if self._enabled:
            try:
                import psycopg  # type: ignore[import-not-found]  # noqa: PLC0415
                self._psycopg = psycopg
            except ImportError:
                log.warning(
                    "TelemetryClient: psycopg not installed; %s set but no driver. "
                    "Install with `pip install psycopg[binary]` to emit production "
                    "quality scores.",
                    TELEMETRY_DSN_ENV,
                )
                self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def record(
        self,
        report: ProductionQualityReport,
        *,
        job_id: str,
        user_audio_hash: str,
        engrave_route: str | None = None,
        title: str | None = None,
    ) -> uuid.UUID | None:
        """Insert one row into ``eval_production_quality_scores``.

        Returns the row's ``id`` UUID on success, ``None`` when
        telemetry is disabled or the insert failed. Job-id collisions
        on the same audio re-emit a fresh row — the dashboard
        de-dups by ``user_audio_hash`` over a moving window.
        """
        if not self._enabled or self._psycopg is None:
            return None

        row = report.as_db_row(
            job_id=job_id,
            user_audio_hash=user_audio_hash,
            engrave_route=engrave_route,
            title=title,
        )
        sql = _build_insert_sql(row)
        try:
            with self._psycopg.connect(self._dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, list(row.values()))
                    fetched = cur.fetchone()
                conn.commit()
        except Exception:  # noqa: BLE001
            log.exception(
                "TelemetryClient.record failed; quality score not persisted "
                "(job_id=%s)",
                job_id,
            )
            return None
        if fetched is None:
            return None
        try:
            return uuid.UUID(str(fetched[0]))
        except (ValueError, TypeError):
            return None


def _build_insert_sql(row: dict[str, Any]) -> str:
    """Build a parametric INSERT for the ``eval_production_quality_scores`` table.

    Keys in ``row`` map 1:1 to column names. Returns a positional-
    parameter SQL string (psycopg style) so the caller passes
    ``list(row.values())``. Centralized here so the SQL stays in
    sync with :meth:`ProductionQualityReport.as_db_row`.
    """
    cols = list(row.keys())
    placeholders = ", ".join(["%s"] * len(cols))
    cols_sql = ", ".join(cols)
    return (
        f"INSERT INTO eval_production_quality_scores ({cols_sql}) "
        f"VALUES ({placeholders}) RETURNING id"
    )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_client: TelemetryClient | None = None


def get_telemetry_client() -> TelemetryClient:
    """Module-level singleton — instantiated lazily on first call."""
    global _default_client
    if _default_client is None:
        _default_client = TelemetryClient()
    return _default_client


def reset_telemetry_client() -> None:
    """Reset the singleton — used by tests that monkeypatch ``OHSHEET_EVAL_TELEMETRY_DSN``."""
    global _default_client
    _default_client = None


# ---------------------------------------------------------------------------
# Convenience: fire-and-forget for the runner
# ---------------------------------------------------------------------------

def emit_production_quality(
    *,
    score: PianoScore,
    job_id: str,
    user_audio_hash: str,
    engrave_route: str | None = None,
    title: str | None = None,
    perf: HumanizedPerformance | None = None,
) -> ProductionQualityReport | None:
    """Compute + persist a Q score for one production job.

    Returns the in-memory :class:`ProductionQualityReport` on success
    (so the runner can attach it to ``EngravedOutput.evaluation_report``),
    or ``None`` if computation itself failed. The Postgres insert is
    handled inside :meth:`TelemetryClient.record` and tolerates
    failures (logs + returns) — never raises out of this call site.
    """
    try:
        report = compute_production_quality_report(score=score, perf=perf)
    except Exception:  # noqa: BLE001
        log.exception(
            "emit_production_quality: compute_production_quality_report failed "
            "(job_id=%s) — telemetry skipped",
            job_id,
        )
        return None

    client = get_telemetry_client()
    with suppress(Exception):
        # ``record`` already swallows + logs internally; the suppress
        # here is belt-and-braces in case psycopg raises during
        # connection setup before our try/except wraps it.
        client.record(
            report,
            job_id=job_id,
            user_audio_hash=user_audio_hash,
            engrave_route=engrave_route,
            title=title,
        )
    return report
