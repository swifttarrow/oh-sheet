"""Eval harness package.

Phase 0 / Phase 3 surface — reference-free metrics, eval-set loaders,
encrypted holdout split:

* :mod:`eval.tier_rf` — reference-free Tier RF (chord / playability / chroma)
* :mod:`eval.loader` — Phase 3 paid eval-set loader
* :mod:`eval.holdout` — encrypted tune/holdout split

Phase 7 surface — Tier 2 / Tier 3 / Tier 4 metric ladder + harness:

* :mod:`eval.tier2_structural` — key / tempo / beat / chord / section
* :mod:`eval.tier3_arrangement` — playability / vleading / density / readability
* :mod:`eval.tier4_perceptual` — chroma / round-trip / CLAP / MERT
* :mod:`eval.harness` — unified per-song runner + CI gate evaluator

The ``scripts/eval.py`` Click CLI wraps :mod:`eval.harness` with one
subcommand per strategy doc §4.2 use case (``ci``, ``nightly``,
``end-to-end``, ``arrange``, ``engrave``, ``round-trip``,
``transcribe`` + ``compare`` pass-throughs to existing tools).

See ``docs/research/transcription-improvement-strategy.md`` Part III
for the metric definitions and
``transcription-improvement-implementation-plan.md``
§Phase 0 / §Phase 3 / §Phase 7 for the phasing.
"""
