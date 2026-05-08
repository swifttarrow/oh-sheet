# Engraver Protocol

A frozen HTTP contract for a third-party engraver service. Implement
this and Oh Sheet can route its `remote_http` engrave stage to your
service instead of the proprietary `oh-sheet-ml-pipeline`.

The current version is **v0.1** ([`openapi.yaml`](./openapi.yaml)).

## Why this exists

Oh Sheet has two engrave backends ([`OHSHEET_ENGRAVE_BACKEND`](../../backend/config.py)):

- `local` — `music21` → MusicXML, `LilyPond` → PDF, in-process. Default.
- `remote_http` — POST MIDI bytes to an external HTTP service that
  returns MusicXML.

The `remote_http` mode currently calls Oh Sheet's hosted, proprietary
`oh-sheet-ml-pipeline`. Self-hosters can't run it, which leaves an
output-quality gap relative to the hosted product (see [#107] and the
[RFC]).

This protocol freezes the HTTP contract so anyone can build their own
engraver — a different ML model, a `music21`-only renderer, an Audiveris
wrapper, whatever — and Oh Sheet will route to it without code changes.

## Who this is for

- **Self-hosters** who want output-quality parity with the hosted
  pipeline today, before any decision lands on whether/how to publish
  `oh-sheet-ml-pipeline` itself.
- **Researchers** experimenting with alternative engraver models who
  want a production pipeline to plug into.
- **Maintainers**, as a stable target the in-tree client and tests can
  be written against.

## How Oh Sheet calls the protocol

The canonical client lives at
[`backend/services/ml_engraver_client.py`](../../backend/services/ml_engraver_client.py).
It is the source of truth — if the spec disagrees, the spec is the bug.

Configure Oh Sheet to point at your service:

```bash
export OHSHEET_ENGRAVE_BACKEND=remote_http
export OHSHEET_ENGRAVER_SERVICE_URL=http://your-engraver:8080
export OHSHEET_ENGRAVER_SERVICE_TIMEOUT_SEC=60   # optional
```

## Conformance checklist

A v0.1-conforming implementation:

1. Accepts `POST /engrave` with `Content-Type: application/octet-stream`
   and a raw Standard MIDI File body.
2. Returns 200 with a MusicXML 3.x document **longer than 500 bytes** on
   success. (Oh Sheet refuses smaller 200 responses as stub
   placeholders.)
3. Returns a 4xx on invalid input or a deterministic engraving failure.
   These will not be retried.
4. Returns a 5xx on transient failure. These will be retried up to three
   times by Oh Sheet's client with exponential backoff (0.5s base).
5. Treats the same MIDI input as deterministic across the retry window —
   non-deterministic output during retry surfaces as flaky downstream
   rendering.

A minimal conformance smoke test is to POST a MusicXML round-trip
fixture's MIDI rendering and confirm the response parses as valid
MusicXML 3.x. Oh Sheet does not currently ship a stand-alone
conformance harness; if you build one, please open a PR.

## Versioning policy

- **v0.x** is unstable. Breaking changes are allowed if they unblock
  one of the [RFC] open questions, but each one bumps the minor version
  and is called out in the changelog below.
- **v1.0** will be cut once a v0.x has been deployed against at least
  one third-party engraver and one significant gap has been identified
  and fixed (or explicitly accepted as a v1 limitation). No date
  committed.
- Evolution from "MIDI in / MusicXML out" to "structured `PianoScore` in"
  is tracked as Q5 in the [RFC] and would land as v0.2+. v0.1 is the
  byte-for-byte freeze of what the client speaks today.

## Changelog

- **v0.1.0** — Initial freeze. `POST /engrave`, MIDI bytes in, MusicXML
  bytes out. Matches the contract of `oh-sheet-ml-pipeline` as of
  [`backend/services/ml_engraver_client.py`](../../backend/services/ml_engraver_client.py).

[#107]: https://github.com/Oh-Sheet-Team/oh-sheet/issues/107
[RFC]: ../rfc-ml-pipeline-publishing.md
