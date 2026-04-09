Here are the revised, finalized Product Requirements Documents (PRDs) for both the Decomposer and Assembler microservices.

These updated versions incorporate the new research, specifically pivoting the infrastructure to **Celery \+ Redis** for faster MVP delivery on Railway, and integrating the highly efficient **Musicpy \+ Mido** stack for algorithmic separation.

# ---

**📄 PRD 1: Decomposer Microservice (svc-decomposer)**

## **1\. Meta Information**

* **Service Name:** svc-decomposer  
* **Status:** READY FOR DEVELOPMENT  
* **Tech Stack:** Python 3.11+, FastAPI, Celery \+ Redis (Task Queue), mido (Parsing), musicpy (Splitting Logic), Boto3 (S3).  
* **Pipeline Stage:** Stage 2 (Transcribe & Isolate \- MIDI Only)

## **2\. Overview & Objective**

The Decomposer service ingests multi-track or single-track MIDI files and intelligently separates the primary melody from the accompaniment. Based on current research, the service will completely bypass heavy ML models and manual heuristic scoring in favor of musicpy's built-in logical separation algorithms, paired with mido for rock-solid file handling.

## **3\. User Personas**

* **Producer Penny (Content Creator):** Uploads dense DAW MIDI exports. She needs the system to accurately identify her lead synth line (Melody) and separate it from her chord pads (Accompaniment) so the resulting sheet music is readable and logically split.

## **4\. Expected System Flow (Celery \+ S3)**

1. **Trigger:** The API Gateway/Orchestrator pushes a job ID and payload\_uri to the Redis task queue.  
2. **Hydration:** The Celery worker picks up the task, downloads the InputBundle.json from S3, and fetches the raw .mid file to the local /tmp directory.  
3. **Processing:** The worker uses mido to parse the file, then passes the structure to musicpy to extract the melody and harmony.  
4. **Export:** It uses mido to write melody\_isolated.mid and accompaniment.mid back to /tmp and uploads them to S3.  
5. **Completion:** It generates the TranscriptionResult JSON (v3.0.0), uploads it, and fires a webhook/status update back to the Orchestrator.

## **5\. The Split Heuristics Plan (Implementation Steps)**

The engineer should implement the following breakdown for the separation logic:

* **Step 1: Low-Level Parsing (mido)**  
  * Load the raw MIDI file using mido.MidiFile. This ensures that all MIDI ticks, tempo changes, and structural metadata are accurately preserved before any logical manipulation begins.  
* **Step 2: The Logic Split (musicpy)**  
  * Convert the mido object into a musicpy data structure.  
  * Execute the split\_all() function. This built-in method uses pitch density and timing to automatically isolate the monophonic lead line from polyphonic chordal structures.  
* **Step 3: Track Routing & Fallbacks**  
  * Route the extracted lead line to the melody track.  
  * Group the remaining extracted harmony and bass parts into the accompaniment track.  
  * *Fallback:* If split\_all() fails to confidently separate a highly complex track, default to a custom **Skyline Algorithm** (extracting the highest-pitched continuous note at any given 16th-note interval).  
* **Step 4: Safe Serialization (mido)**  
  * Convert the separated musicpy objects back to mido tracks and save them as strictly formatted .mid files to prevent downstream corruption.

## **6\. Data Contracts (v3.0.0)**

* **Consumes:** InputBundle  
* **Produces:** TranscriptionResult containing Claim-Check URIs.  
  JSON  
  "midi\_tracks": \[  
    {  
      "instrument": "melody",  
      "source\_stem": "s3://oh-sheet-pipeline/jobs/{job\_id}/melody.mid"  
    },  
    {  
      "instrument": "other",  
      "source\_stem": "s3://oh-sheet-pipeline/jobs/{job\_id}/accompaniment.mid"  
    }  
  \]  
