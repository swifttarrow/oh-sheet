"""Backend-side eval — production telemetry for Phase 7.

Mirrors the offline ``eval/`` package's metric library, but emits
per-job composite quality scores into Postgres so the production
dashboard can sparkline live quality trends. See strategy doc §6.1
for the schema and §8.2 for the composite Q computation.
"""
