# System Design: Oh Sheet!

> **Last updated:** 2026-04-10

## 1. Architecture Overview

"Oh Sheet!" is an automated pipeline that converts songs (MP3, MIDI, or YouTube links) into playable, two-staff piano sheet music with expressive, humanized playback.

The current architecture is a **monolith with module separation**, where each pipeline stage runs as an independent **Celery worker** sharing a single codebase. Heavy media files are exchanged via the **Claim-Check Pattern** through a local blob store, keeping the Redis message broker lightweight. A FastAPI orchestrator manages job state in-memory and streams progress to clients over WebSocket.

### Core Tech Stack
* **API / Orchestrator:** Python 3.10+ (3.12 in Docker), FastAPI, in-memory job state
* **Task Queue:** Celery with Redis (broker + result backend)
* **Blob Storage:** Local filesystem (`LocalBlobStore`, `file://` URIs)
* **Pipeline Workers:** Python modules in `backend/workers/` + `backend/services/`
* **Key Libraries:** Basic Pitch (transcription), music21 (MusicXML), pretty_midi (MIDI), yt-dlp (ingest)
* **Frontend:** Flutter (Dart), cross-platform (web primary)
* **Deployment:** Docker Compose on GCP VM, Caddy reverse proxy, GitHub Actions CI/CD

---

## 2. Architecture Diagram (Current State)

```text
                              +-----------------+
                              |                 |
                              |  Flutter Client |
                              |  (Web / Mobile) |
                              +---+---------+---+
                                  |         ^
                   1. Upload file |         | 2. Poll / WebSocket
                      & config    v         |    (Get status + URLs)
                          +-----------------+----------------+
                          |                                  |
                          |  API Gateway & Orchestrator      |
                          |  (FastAPI, in-memory job state)  |
                          |                                  |
                          +-------+------------------+-------+
                                  |                  |
                      3. Task IDs |                  | 4. Media files
                     & URIs only  |                  |    (Claim-Check)
                                  v                  v
                          +---------------+  +---------------+
                          |               |  |               |
                          |  Redis Queue  |  |  Local Blob   |
                          |  (Celery)     |  |  Store (fs)   |
                          |               |  |               |
                          +-------+-------+  +-------+-------+
                                  |                  |
                                  |                  | 5. Read / Write
                                  |                  |    files via URIs
                                  v                  v
            +---------------------+------------------+---------------------+
            |                    Celery Workers (same codebase)            |
            |                                                              |
            | +-----------------+   +-----------------+                    |
            | | worker-ingest   |   | worker-transcribe|                   |
            | | (yt-dlp, ffprobe|   | (Basic Pitch)   |                    |
            | +-----------------+   +-----------------+                    |
            |                                                              |
            | +-----------------+   +-----------------+                    |
            | | worker-arrange  |   | worker-humanize |                    |
            | | (music21)       |   | (rule-based     |                    |
            | +-----------------+   |  expression)    |                    |
            |                       +-----------------+                    |
            | +-----------------+                                          |
            | | worker-engrave  |                                          |
            | | (pretty_midi,   |                                          |
            | |  music21,       |                                          |
            | |  LilyPond)      |                                          |
            | +-----------------+                                          |
            +--------------------------------------------------------------+
```

---

## 3. Core Component Breakdown

### 3.1 API Gateway & Orchestrator (`backend/`)
* **Role:** Entry point for clients and pipeline coordinator.
* **Responsibilities:**
  * Receives file uploads via `POST /v1/uploads/{audio,midi}` and stores them in blob storage (Claim-Check).
  * Creates jobs via `POST /v1/jobs` with a `PipelineConfig` specifying the variant.
  * `PipelineRunner` walks the execution plan, dispatching each stage sequentially via Celery (`apply_async` for local tasks, `send_task` for remote).
  * `JobManager` tracks in-memory job state and fans out `JobEvent`s to WebSocket subscribers via asyncio Queues.
  * Serves artifacts (PDF, MIDI, MusicXML) via `GET /v1/artifacts/{job_id}/{kind}`.

### 3.2 Worker: Ingest (`backend/workers/ingest.py`)
* **Role:** Stage 1 — validate and probe input files.
* **Responsibilities:**
  * Probes uploaded audio/MIDI with ffprobe for metadata (duration, format, sample rate).
  * Produces an `InputBundle` manifest with remote file references and metadata.

### 3.3 Worker: Transcribe (`backend/workers/transcribe.py`)
* **Role:** Stage 2 — audio-to-MIDI transcription.
* **Responsibilities:**
  * Uses **Spotify Basic Pitch** for polyphonic note detection from audio.
  * Post-processes raw predictions into structured `MidiTrack` objects with tempo map.
  * Produces a `TranscriptionResult` with note-level data and quality signals.
  * Falls back to a shape-correct stub when Basic Pitch is not installed.

### 3.4 Worker: Arrange (`backend/workers/arrange.py`)
* **Role:** Stage 3 — piano reduction.
* **Responsibilities:**
  * Assigns transcribed notes to right hand (melody) and left hand (accompaniment).
  * Applies difficulty constraints (e.g., max hand span <= 12 semitones).
  * Quantizes note timings to a rhythmic grid.
  * Produces a `PianoScore` with per-hand note lists and score metadata.

### 3.5 Worker: Humanize (`backend/workers/humanize.py`)
* **Role:** Stage 4 — expressive performance.
* **Responsibilities:**
  * Takes the rigid, quantized `PianoScore` and applies musical expression.
  * Adds micro-timing deviations (rubato), dynamic velocity shaping, and articulations.
  * Currently rule-based; ML model (transformer) planned for future.
  * Produces a `HumanizedPerformance` with expressive notes and dynamic markings.

### 3.6 Worker: Engrave (`backend/workers/engrave.py`)
* **Role:** Stage 5 — render final outputs.
* **Responsibilities:**
  * Fuses note positions from `PianoScore` with dynamics from `HumanizedPerformance`.
  * Renders MIDI via pretty_midi, MusicXML via music21, PDF via LilyPond (when available).
  * Produces an `EngravedOutput` with URIs to `sheet.pdf`, `score.musicxml`, and `humanized.mid`.
  * Falls back to stub outputs when rendering dependencies are missing.

---

## 4. Pipeline Execution & Data Flow

### 4.1 Pipeline Variants

| Variant | Stages | Use Case |
|---------|--------|----------|
| `full` | ingest → transcribe → arrange → humanize → engrave | Audio input, full pipeline |
| `audio_upload` | ingest → transcribe → arrange → humanize → engrave | Same as full (explicit audio) |
| `midi_upload` | ingest → arrange → humanize → engrave | MIDI input (skip transcription) |
| `sheet_only` | ingest → transcribe → arrange → engrave | Sheet music only (skip humanizer) |

All variants respect the `skip_humanizer` config flag to dynamically remove the humanize stage.

### 4.2 Sequence Diagram

```text
Client         Orchestrator        Redis Queue         Blob Store          Celery Workers
  |                 |                   |                  |                    |
  |-- 1. POST /v1/uploads/audio ------>|                  |                    |
  |                 |-- 2. Store file ---------------------->|                  |
  |<-- 3. RemoteAudioFile (URI) -------|                  |                    |
  |                 |                   |                  |                    |
  |-- 4. POST /v1/jobs (config) ------>|                  |                    |
  |<-- 5. Return job_id (202) ---------|                  |                    |
  |                 |                   |                  |                    |
  |-- 6. WS /v1/jobs/{id}/ws -------->|                   |                    |
  |                 |                   |                  |                    |
  |                 |-- 7. Dispatch `ingest.run` --------->|                    |
  |                 |                   |-- 8. Execute --->| (worker-ingest)    |
  |                 |                   |                  |<-- 9. Write bundle-|
  |<- 10. Event ----|<-- Result --------|                  |                    |
  |                 |                   |                  |                    |
  |                 |-- 11. Dispatch `transcribe.run` ---->|                    |
  |                 |                   |-- 12. Execute -->| (worker-transcribe)|
  |                 |                   |                  |<-- 13. Write MIDI -|
  |<- 14. Event ----|<-- Result --------|                  |                    |
  |                 |                   |                  |                    |
  |                 |-- 15. Dispatch `arrange.run` ------->|                    |
  |                 |                   |-- 16. Execute -->| (worker-arrange)   |
  |                 |                   |                  |<-- 17. Write score-|
  |<- 18. Event ----|<-- Result --------|                  |                    |
  |                 |                   |                  |                    |
  |                 |-- 19. Dispatch `humanize.run` ------>|                    |
  |                 |                   |-- 20. Execute -->| (worker-humanize)  |
  |                 |                   |                  |<-- 21. Write perf.-|
  |<- 22. Event ----|<-- Result --------|                  |                    |
  |                 |                   |                  |                    |
  |                 |-- 23. Dispatch `engrave.run` ------->|                    |
  |                 |                   |-- 24. Execute -->| (worker-engrave)   |
  |                 |                   |                  |<-- 25. Write PDF --|
  |<- 26. Event ----|<-- Result --------|                  |                    |
  |                 |                   |                  |                    |
  |-- 27. GET /v1/artifacts/{id}/pdf ->|                   |                    |
  |<-- 28. File stream ----------------|                  |                    |
```

---

## 5. Data Contracts (Schema v3.0.0)

Defined in `shared/shared/contracts.py`, re-exported by `backend/contracts.py`.

| Contract | Produced By | Consumed By | Key Fields |
|----------|------------|-------------|------------|
| `InputBundle` | Ingest | Transcribe, Arrange | metadata, audio_file, midi_file |
| `TranscriptionResult` | Transcribe | Arrange | tracks (MidiTrack[]), harmonic_analysis, quality |
| `PianoScore` | Arrange | Humanize, Engrave | right_hand, left_hand (ScoreNote[]), metadata |
| `HumanizedPerformance` | Humanize | Engrave | expressive_notes, expression_map, original_score |
| `EngravedOutput` | Engrave | Client (download) | pdf_uri, musicxml_uri, humanized_midi_uri |

---

## 6. API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/health` | Health check |
| `POST` | `/v1/uploads/audio` | Upload audio file → `RemoteAudioFile` |
| `POST` | `/v1/uploads/midi` | Upload MIDI file → `RemoteMidiFile` |
| `POST` | `/v1/jobs` | Create pipeline job → `JobSummary` (202) |
| `GET` | `/v1/jobs` | List all jobs |
| `GET` | `/v1/jobs/{id}` | Get job status |
| `GET` | `/v1/jobs/{id}/events` | Get job event history |
| `WS` | `/v1/jobs/{id}/ws` | Stream `JobEvent`s in real-time |
| `GET` | `/v1/artifacts/{job_id}/{kind}` | Download artifact (pdf/midi/musicxml) |
| `POST` | `/v1/stages/{name}` | Stateless stage endpoint (external orchestrator) |

OpenAPI docs available at `/docs`.

---

## 7. Infrastructure & Deployment

* **Container Images:** Single multi-stage Dockerfile builds Flutter web assets then packages the Python backend. Each Celery worker runs from the same image with a different `-Q` flag.
* **Docker Compose (dev):** Redis + orchestrator + 5 worker containers (one per stage queue).
* **Docker Compose (prod):** Same services + Caddy reverse proxy (TLS via `oh-sheet.duckdns.org`), persistent volumes for Redis data and blob storage.
* **CI (GitHub Actions):** Lint (ruff), typecheck (mypy), test (pytest), Flutter analyze — runs on PRs to main.
* **CD (GitHub Actions):** Manual trigger builds images, pushes to GCP Artifact Registry, deploys to VM via SSH, health-checks `/v1/health`, Slack notifications on success/failure.
* **System Dependencies:** The engrave worker requires `lilypond` and `fluidsynth` (installed in the Docker image) for PDF rendering and audio preview.

---

## 8. Known Limitations (Current State)

* **In-memory job state:** Jobs are lost on orchestrator restart. No persistence layer yet.
* **Local blob store:** Files stored on local filesystem with `file://` URIs. Not suitable for multi-node deployments.
* **No external integrations:** Tunechat export and YouTube link ingestion are not yet implemented.
* **Rule-based humanizer:** Expression is applied via heuristics, not the planned ML transformer model.
* **Single-node deployment:** All containers run on one GCP VM. No horizontal scaling.

---

## 9. Future Design (Target Architecture)

> **Note:** This section describes the aspirational target. The specific approach
> for decomposition, grade-tuning, and ML humanization may evolve as the project
> matures.

The target architecture retains the monolith-with-Celery-workers pattern but expands the pipeline with **grade-tuning workers** that adapt sheet music difficulty to player skill level, and replaces stubs with production implementations.

### Planned Changes

* **Grade-Tuning Workers:** Additional stages (or sub-stages within arrange) that adapt the arrangement to a target difficulty level. Two known approaches under consideration, though others may emerge:
  * **Decompose:** Isolate melody from accompaniment via source separation (e.g., Demucs) before arrangement, giving the arranger cleaner input to work with at each difficulty level.
  * **Condense:** Work top-down from the full transcription, selectively simplifying rhythms, reducing hand span, dropping voices, or collapsing chord voicings to fit a target grade.
* **ML Humanizer:** Replace the rule-based humanizer with a transformer encoder/decoder model exported via ONNX for CPU inference.
* **S3 Blob Store:** Swap `LocalBlobStore` for S3 to enable multi-node deployment and presigned URL downloads.
* **Persistent Job State:** Move from in-memory to PostgreSQL (or Redis Streams) for job durability across restarts.
* **Tunechat Integration:** Post-pipeline webhook to export final artifacts (presigned S3 URLs) to the Tunechat platform.

### Target Architecture Diagram

```text
                              +-----------------+
                              |                 |
                              |  Flutter Client  |
                              |  (Web / Mobile) |
                              +---+---------+---+
                                  |         ^
                   1. Upload file |         | 2. Poll / WebSocket
                      & config    v         |    (Get status + URLs)
+---------------+         +-----------------+----------------+        +----------------+
|               |         |                                  |        |                |
|  PostgreSQL   | <-----> |  API Gateway & Orchestrator      | -----> | Tunechat API   |
|  (Job State)  |         |  (FastAPI)                       |        | (External)     |
|               |         |                                  |        |                |
+---------------+         +-------+------------------+-------+        +----------------+
                                  |                  |
                      3. Task IDs |                  | 4. Media files
                     & URIs only  |                  |    (Claim-Check)
                                  v                  v
                          +---------------+  +---------------+
                          |               |  |               |
                          |  Redis Queue  |  |   Amazon S3   |
                          |  (Celery)     |  |   (Storage)   |
                          |               |  |               |
                          +-------+-------+  +-------+-------+
                                  |                  |
                                  |                  | 5. Read / Write
                                  |                  |    files via URIs
                                  v                  v
            +---------------------+------------------+---------------------+
            |                    Celery Workers                            |
            |                                                              |
            | +-----------------+   +-----------------+                    |
            | | worker-ingest   |   | worker-transcribe|                   |
            | | (yt-dlp, probe) |   | (Basic Pitch)   |                    |
            | +-----------------+   +-----------------+                    |
            |                                                              |
            | +-----------------+   +-----------------+                    |
            | | worker-decompose|   | worker-arrange  |                    |
            | | (Demucs / ML)   |   | + grade-tuning  |                    |
            | +-----------------+   +-----------------+                    |
            |                                                              |
            | +-----------------+   +-----------------+                    |
            | | worker-humanize |   | worker-engrave  |                    |
            | | (ONNX/ML model) |   | (LilyPond)      |                    |
            | +-----------------+   +-----------------+                    |
            +--------------------------------------------------------------+
```

### Target Pipeline

```text
ingest → transcribe → decompose → arrange (+ grade-tune) → humanize (ML) → engrave
```

Reference: https://miro.com/app/board/uXjVGm_LRe0=/
