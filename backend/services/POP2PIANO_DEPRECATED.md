# Pop2Piano is deprecated

**Status:** retained for backwards compatibility only. New deployments should
use AMT-APC cover mode (Phase 8) instead.

## Why deprecated

`sweetcocoa/pop2piano` (the upstream model + checkpoint repo) ships without
a `LICENSE` file. The strategy doc §G3 ("Licensing risks") flags this as a
legal blocker for commercial deployments — without an explicit grant from
Choi & Lee, redistributing the weights or shipping derivative output is
risky.

## What replaces it

[AMT-APC](https://github.com/misya11p/amt-apc) (`backend/services/transcribe_amt_apc.py`)
is an MIT-licensed hFT-Transformer descendant trained on similar
pop-track / piano-cover pairs. It is the supported cover-mode
transcriber for the `pop_cover` `PipelineVariant` and is surfaced as
the "Piano cover" toggle in the upload UI.

## Migration path

* Set `OHSHEET_POP2PIANO_ENABLED=false` (already the default since Phase 6).
* Set `OHSHEET_AMT_APC_ENABLED=true` to enable cover-mode transcription.
* Frontend clients should use the `cover_mode` request flag (see
  `backend/api/routes/jobs.py`) instead of relying on Pop2Piano's
  global-enable behavior.

## Removal target

Pop2Piano modules will remain importable until at least the next major
contract bump. Tests under `tests/test_transcribe_pop2piano.py` continue
to exercise the path so fallback behavior stays correct, but the
dispatcher in `backend/services/transcribe.py` now emits a runtime
deprecation warning when `pop2piano_enabled=True`.
