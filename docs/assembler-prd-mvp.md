# **PRD: Assembler Microservice (svc-assembler)**

## **1\. Meta Information**

* **Service Name:** svc-assembler  
* **Status:** READY FOR DEVELOPMENT  
* **Tech Stack:** Python 3.11+, FastAPI (for health/metrics), Celery \+ Redis (Task Queue), music21, Boto3 (S3).  
* **Pipeline Stage:** Stage 3 (Piano Arrangement)

## **2\. Overview & Objective**

The Assembler service acts as a strict mathematical filter and arranger. It ingests the separated melody and accompaniment MIDI tracks produced by the Decomposer, stacks them onto a two-staff piano arrangement, and rigorously prunes away notes that violate physical playability constraints for a human pianist. To ensure a highly predictable and testable MVP, this service is hardcoded to output an **"Intermediate"** difficulty level using rigid rule-based heuristics.

## **3\. User Personas**

* **Beginner/Intermediate Ben (The Target User):** Wants to play pop songs but cannot physically stretch his hands to play dense, 10-note DAW chords. He relies on the Assembler to brutally (but musically) prune out unnecessary inner voices so the sheet music is readable and structurally sound.

## **4\. Exact Data Ingestion Contract (Developer Handoff)**

Because this service is being built in parallel with the Decomposer, there must be zero ambiguity regarding file locations and naming conventions. The Assembler worker must strictly follow this ingestion sequence:

**Step 1: The Celery Payload**

The Orchestrator places a task in the Redis queue. The Assembler's Celery worker receives a payload containing the job\_id and the payload\_uri.

* *Example Payload:*  
  JSON  
  {  
    "job\_id": "job\_99887766",  
    "payload\_uri": "s3://oh-sheet-pipeline/jobs/job\_99887766/transcription\_result.json"  
  }

**Step 2: Fetching the Contract (The JSON Manifest)**

Use boto3 to fetch the TranscriptionResult JSON from the payload\_uri. This JSON acts as the manifest for the raw MIDI files.

* *Expected S3 Path:* s3://{BUCKET\_NAME}/jobs/{job\_id}/transcription\_result.json

**Step 3: Resolving the Claim-Checks (The File Paths)**

Parse the JSON's midi\_tracks array. The Decomposer is contracted to output exactly two nodes here with strict instrument string values: "melody" and "other" (accompaniment). Extract the S3 URIs:

JSON

"midi\_tracks": \[  
  {  
    "instrument": "melody",  
    "source\_stem": "s3://oh-sheet-pipeline/jobs/job\_99887766/stems/melody.mid"  
  },  
  {  
    "instrument": "other",  
    "source\_stem": "s3://oh-sheet-pipeline/jobs/job\_99887766/stems/accompaniment.mid"  
  }  
\]

**Step 4: Container-Safe Local Hydration**

To avoid file collisions during concurrent Celery processing, **do not** download the files to a generic /tmp/melody.mid. Use boto3 to download the objects to local ephemeral storage using the job\_id as a prefix:

* **Local Melody Path:** /tmp/{job\_id}\_melody.mid  
* **Local Accompaniment Path:** /tmp/{job\_id}\_accompaniment.mid

**Step 5: Initialization & Cleanup**

Pass those two local file paths directly into music21.converter.parse() to begin filtering. **CRITICAL:** Once the job completes (success or failure), explicitly delete these /tmp files to prevent the Railway container from running out of disk space.

## **5\. Functional Requirements (The Rule Engine)**

The Assembler is a pure, predictable function hardcoded to the "Intermediate" difficulty profile. The developer must build this exact pipeline:

**1\. Key Signature Handling**

* **Baseline MVP:** Read the key signature from the ingested MIDI and keep it exactly as is (Pass-Through). Do not attempt to shift pitches.  
* **Stretch Goal (Transposition):** Use music21 to analyze the key. If the original key has more than 3 sharps or flats (e.g., E Major, Ab Major), automatically transpose the entire score to the nearest simple key (C Major, F Major, or G Major). *Note to Dev: If implemented, you must ensure music21.chord.simplifyEnharmonics() is run afterward to prevent unreadable enharmonic spelling (like E\#\#).*

**2\. Rigid 16th-Note Quantization**

* Snap every single onset\_beat and duration\_beat to the nearest 0.25 beat grid.

**3\. Right Hand (RH) Logic: The Melody \+ 1**

* **Primary (Immunity):** Map 100% of the notes from melody.mid to RH Voice 1\. These cannot be deleted.  
* **Secondary (Filler):** Look at notes in accompaniment.mid above Middle C (MIDI 60). Allow a **maximum of 1 concurrent note** to be added below the melody line, provided the distance between that note and the melody note is $\\leq$ 12 semitones. Discard all other RH filler notes.

**4\. Left Hand (LH) Logic: The Bass \+ 1**

* **Primary (Foundation):** Map the lowest concurrent notes from accompaniment.mid to LH Voice 1\. These cannot be deleted.  
* **Secondary (Filler):** Look at notes in accompaniment.mid below Middle C (MIDI 60). Allow a **maximum of 1 concurrent note** to be added above the bass note, provided the distance between the bass note and the inner note is $\\leq$ 12 semitones. Discard all other LH filler notes.

## **6\. Data Contracts (v3.0.0)**

* **Consumes:** TranscriptionResult (JSON) \+ Raw .mid files via S3.  
* **Produces:** PianoScore (JSON). Must map the quantized notes to the PianoScore JSON schema, assigning unique IDs (rh-0001, lh-0042), and upload the result to S3.  
  JSON  
  "right\_hand": \[  
    {  
      "id": "rh-0001",  
      "pitch": 72,  
      "onset\_beat": 0.0,  
      "duration\_beat": 1.0,  
      "velocity": 80,  
      "voice": 1  
    }  
  \]

## **7\. Scale, Load, & Deployment**

* **Local Dev:** docker-compose utilizing Redis (for Celery) and MinIO (for local S3 emulation).  
* **Production Deployment:** Railway.app. Deploy as an independent Railway service worker attached to a shared managed Redis instance.  
* **Autoscaling:** Triggered on CPU utilization \> 70%.  
* **Memory Profile:** Allocate 1GB RAM minimum per container. music21 generates massive object trees in memory when parsing dense MIDI files.