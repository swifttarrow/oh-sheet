Here is the updated High-Level System Design (HLD) document. It now maps the complete end-to-end flow from your Miro architecture, fully integrating the **Humanizer Service** (Stage 4), the **Engraver/Renderer** (Stage 5), and the post-pipeline integration for **Tunechat**. 

> **Note:** This document describes the aspirational target architecture.
> The current implementation uses **local filesystem blob storage** (not S3),
> **in-memory job state** (not PostgreSQL), and has not yet integrated with
> the Tunechat API. See `CLAUDE.md` for the as-built architecture.

***

# System Design: Oh Sheet! (Complete MVP Architecture)

## 1. Architecture Overview
"Oh Sheet!" is an automated pipeline that converts raw, multi-track MIDI files into human-readable, playable, two-staff piano sheet music with expressive, humanized playback. 

The MVP utilizes an asynchronous, event-driven microservices architecture optimized for speed of delivery. It leverages **Celery + Redis** for task orchestration and strictly enforces the **Claim-Check Pattern** via **Amazon S3**, ensuring heavy media files and massive JSON arrays never bottleneck the message broker. The pipeline concludes with an optional webhook integration to export artifacts directly to the **Tunechat** platform.

### Core Tech Stack
* **API / Orchestrator:** Python 3.11, FastAPI, PostgreSQL
* **Message Broker:** Redis (Celery backend)
* **Blob Storage:** Amazon S3
* **Decomposer Service (Stage 2):** Python, `mido`, `musicpy`
* **Assembler Service (Stage 3):** Python, `music21`
* **Humanizer Service (Stage 4):** Python, ONNX Runtime (Transformer Encoder/Decoder)
* **Engraver Service (Stage 5):** Python, `music21`, LilyPond / Verovio
* **Deployment:** Railway.app (Production), Docker Compose (Local)

---

## 2. High-Level Architecture Diagram

```text
                               +-----------------+
                               |                 |
                               |  Client / UI    |
                               |                 |
                               +---+---------+---+
                                   |         ^
                    1. Upload MIDI |         | 2. Poll Status 
                       & Config    v         |    (Get URLs)
+---------------+          +-----------------+----------------+        +----------------+
|               |          |                                  |        |                |
|  PostgreSQL   | <------> |  API Gateway & Orchestrator      | ---->  | Tunechat API   |
|  (Job State)  |          |  (FastAPI)                       | Option | (External)     |
|               |          |                                  |        |                |
+---------------+          +-------+------------------+-------+        +----------------+
                                   |                  |
                       3. Job IDs  |                  | 4. Media & JSON
                      & URIs Only  |                  | (Claim-Check)
                                   v                  v
                           +---------------+  +---------------+
                           |               |  |               |
                           |  Redis Queue  |  |   Amazon S3   |
                           |  (Celery)     |  |   (Storage)   |
                           |               |  |               |
                           +-------+-------+  +-------+-------+
                                   |                  |
                                   |                  | 5. Read / Write 
                                   |                  | Payloads & Files
                                   v                  v
             +---------------------+------------------+---------------------+
             |                                                              |
             | +-----------------+   +-----------------+                    |
             | | svc-decomposer  |   | svc-assembler   |                    |
             | | (musicpy)       |   | (music21)       |                    |
             | +-----------------+   +-----------------+                    |
             |                                                              |
             | +-----------------+   +-----------------+                    |
             | | svc-humanizer   |   | svc-engraver    |                    |
             | | (ONNX/ML Model) |   | (LilyPond)      |                    |
             | +-----------------+   +-----------------+                    |
             +--------------------------------------------------------------+
```

---

## 3. Core Component Breakdown

### 3.1 API Gateway & Orchestrator (`svc-api`)
* **Role:** The entry point for users and the "brain" of the state machine.
* **Responsibilities:** * Receives file uploads and generates `InputBundle` manifests in S3.
  * Dispatches ordered tasks sequentially to Redis (`cmd.decompose` $\rightarrow$ `cmd.arrange` $\rightarrow$ `cmd.humanize` $\rightarrow$ `cmd.engrave`).
  * If the user selected the "Export to Tunechat" option, the Orchestrator fires a final webhook with S3 presigned URLs to the Tunechat API upon pipeline completion.

### 3.2 Decomposer Service (`svc-decomposer`)
* **Role:** Stage 2 (Transcribe & Isolate).
* **Responsibilities:**
  * Pulls raw multi-track MIDI files from S3.
  * Uses `musicpy`'s `split_all()` logic to isolate the primary melodic line from the harmonic accompaniment.
  * Outputs `melody.mid` and `accompaniment.mid` to S3, returning a `TranscriptionResult` JSON.

### 3.3 Assembler Service (`svc-assembler`)
* **Role:** Stage 3 (Piano Arrangement).
* **Responsibilities:**
  * Maps the melody to the Right Hand (RH) and lowest accompaniment notes to the Left Hand (LH).
  * Applies difficulty constraints (e.g., Max span $\leq$ 12 semitones).
  * Quantizes note timings to a rigid 16th-note grid.
  * Outputs the strictly formatted `PianoScore` JSON.

### 3.4 Humanizer Service (`svc-humanizer`)
* **Role:** Stage 4 (Humanize Performance).
* **Responsibilities:**
  * Ingests the rigid, quantized `PianoScore`.
  * Runs a 4-layer transformer encoder / 1-layer decoder (exported via ONNX for CPU-friendly MVP inference) to analyze phrase structure and harmonic tension.
  * Generates micro-timing deviations (rubato), dynamic velocity changes, and articulations.
  * Outputs an expressive `.mid` file and a `HumanizedPerformance` JSON containing the expression metadata (dynamics markings) required for the engraver.

### 3.5 Engraver Service (`svc-engraver`)
* **Role:** Stage 5 & 6 (Engrave & Output).
* **Responsibilities:**
  * Fuses the discrete note positions from the `PianoScore` with the continuous dynamic markings from the `HumanizedPerformance`.
  * Renders professional sheet music via LilyPond / Verovio binaries.
  * Generates an audio preview from the expressive MIDI using a SoundFont (.sf2).
  * Outputs final assets to S3: `sheet.pdf`, `score.musicxml`, `preview.mp3`.

---

## 4. Pipeline Execution & Data Flow

```text
Client         Orchestrator        Redis Queue         S3 Storage          Worker Nodes        Tunechat API
  |                 |                   |                  |                    |                   |
  |-- 1. POST Job ->|                   |                  |                    |                   |
  |                 |-- 2. Init State & Upload MIDI ------>|                    |                   |
  |<-- 3. Return ID-|                                      |                    |                   |
  |                 |-- 4. Push `cmd.decompose` ---------->|                    |                   |
  |                 |                                      |-- 5. Pop Task ---->| (Decomposer)      |
  |                 |                                      |<--6. Save Stems ---|                   |
  |                 |<-- 7. Decomposer Success Event ---------------------------|                   |
  |                 |                                      |                    |                   |
  |                 |-- 8. Push `cmd.arrange` ------------>|                    |                   |
  |                 |                                      |-- 9. Pop Task ---->| (Assembler)       |
  |                 |                                      |<--10. Save Score --|                   |
  |                 |<-- 11. Assembler Success Event ---------------------------|                   |
  |                 |                                      |                    |                   |
  |                 |-- 12. Push `cmd.humanize` ---------->|                    |                   |
  |                 |                                      |-- 13. Pop Task --->| (Humanizer)       |
  |                 |                                      |<--14. Save Expr. --|                   |
  |                 |<-- 15. Humanizer Success Event ---------------------------|                   |
  |                 |                                      |                    |                   |
  |                 |-- 16. Push `cmd.engrave` ----------->|                    |                   |
  |                 |                                      |-- 17. Pop Task --->| (Engraver)        |
  |                 |                                      |<--18. Save PDF/MP3-|                   |
  |                 |<-- 19. Engraver Success Event ----------------------------|                   |
  |                 |                                      |                    |                   |
  |                 |-- 20. [If Opt-in] POST /api/v1/import --------------------------------------->|
  |-- 21. Poll GET->|                                      |                    |                   |
  |<-- 22. S3 URLs -|                                      |                    |                   |
```

---

## 5. Tunechat Integration Strategy

To ensure Oh Sheet! remains decoupled from third-party outages, the Tunechat export is handled entirely asynchronously at the very end of the DAG (Directed Acyclic Graph).

1. **User Opt-in:** The client passes `{"export_to_tunechat": true, "tunechat_token": "xyz"}` during the initial `POST /jobs` request.
2. **Post-Processing:** Once the Engraver successfully uploads the final PDF and Audio Preview to S3, the Orchestrator evaluates the user's config.
3. **Delivery:** The Orchestrator generates short-lived Presigned S3 URLs for the PDF, MP3, and MIDI, and fires a POST request to Tunechat's ingestion API.
4. **Resilience:** If the Tunechat API is down, the Orchestrator marks the export step as `failed` but marks the overall Oh Sheet! job as `success`, ensuring the user still receives their sheet music locally.

---

## 6. Infrastructure & Deployment Notes

* **Machine Learning Runtime:** The Humanizer utilizes a Transformer model. To avoid the cost and complexity of GPU scheduling for the MVP, the model weights will be exported to **ONNX**. This allows the `svc-humanizer` to run efficiently on standard CPU containers alongside the rest of the services.
* **System Dependencies:** The `svc-engraver` Docker image MUST include system-level dependencies for sheet music rendering (e.g., `apt-get install lilypond fluidsynth`).
* **Storage Cleanup:** Because the pipeline generates 5-8 intermediate files per song (raw stems, quantized MIDIs, expression JSONs), S3 Lifecycle Policies will be configured to automatically expire intermediate `stems/` and `json/` objects after 7 days to manage MVP storage costs.

Reference: https://miro.com/app/board/uXjVGm_LRe0=/