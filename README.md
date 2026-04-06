# Oh Sheet

FastAPI backend for the **Song to Humanized Piano Sheet Music** pipeline.

This is the REST + WebSocket layer that exposes the 5-stage pipeline
(`ingest → transcribe → arrange → humanize → engrave`) defined in
[`api-contracts-v2.md`](../temp1/api-contracts-v2.md), schema version
`3.0.0`.

## Status

This repo currently contains the **API skeleton**. Each stage service is a
stub that returns shape-correct contract objects but does **not** run real ML
inference, `music21`, LilyPond, etc. The wrappers will delegate to the
existing pipeline implementations in a follow-up PR.

## Layout

```
ohsheet/
├── main.py              # FastAPI app factory + uvicorn entry
├── config.py            # Pydantic settings
├── contracts.py         # Pydantic models — mirrors api-contracts-v2.md (3.0.0)
├── storage/             # BlobStore abstraction (Claim-Check pattern)
│   ├── base.py
│   └── local.py         # file:// backed local store; S3 stub goes here next
├── services/            # Stage workers — STUBS
│   ├── ingest.py
│   ├── transcribe.py    # → wraps MT4 v5
│   ├── arrange.py       # → wraps temp1/arrange.py
│   ├── humanize.py      # → wraps temp1/humanize.py
│   └── engrave.py       # → wraps temp1/engrave.py (LilyPond)
├── jobs/                # Async job runner + WebSocket pub/sub
│   ├── manager.py       # In-memory JobManager (swap for Redis later)
│   ├── runner.py        # Walks PipelineConfig.get_execution_plan()
│   └── events.py        # JobEvent schema
└── api/
    └── routes/
        ├── health.py
        ├── uploads.py   # POST /v1/uploads/{audio,midi} → Claim-Check URIs
        ├── jobs.py      # POST /v1/jobs, GET /v1/jobs/{id}, list
        ├── stages.py    # POST /v1/stages/{ingest,…} — worker envelope per §1
        └── ws.py        # WS  /v1/jobs/{id}/ws — live event stream
```

## Run

```bash
pip install -e ".[dev]"
ohsheet                              # or: uvicorn ohsheet.main:app --reload
```

OpenAPI docs at <http://localhost:8000/docs>.

## Submit a job

```bash
# 1. Upload an audio file → returns a RemoteAudioFile (Claim-Check URI)
curl -F "file=@song.mp3" http://localhost:8000/v1/uploads/audio

# 2. Submit a job referencing the upload
curl -X POST http://localhost:8000/v1/jobs \
  -H "content-type: application/json" \
  -d '{"audio": <RemoteAudioFile from step 1>, "title": "My Song"}'

# 3. Stream live updates over WebSocket
wscat -c ws://localhost:8000/v1/jobs/<job_id>/ws
```

## Per-stage worker endpoints

For Temporal / Step Functions style orchestration, each stage is also exposed
as a stateless worker that takes an `OrchestratorCommand` and returns a
`WorkerResponse` (see contracts §1):

- `POST /v1/stages/ingest`
- `POST /v1/stages/transcribe`
- `POST /v1/stages/arrange`
- `POST /v1/stages/humanize`
- `POST /v1/stages/engrave`

## Tests

```bash
pytest
```

## Wiring real services

Each file under `ohsheet/services/` has a top-of-file docstring marking what
to replace. The plan:

1. Move (or `pip install -e`) the existing pipeline modules so they're
   importable.
2. Replace the stub bodies with calls into the real implementations.
3. Use `asyncio.to_thread()` for CPU-bound stages (MT4 inference, LilyPond).
4. Add an `S3BlobStore` alongside `LocalBlobStore`.
