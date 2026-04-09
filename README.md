<p align="center">
  <img src="frontend/assets/mascots/mascot-home-happy.svg" alt="Oh Sheet! mascot" width="280">
</p>

<h1 align="center">Oh Sheet!</h1>

<p align="center">
  <strong>Turn any song into playable piano sheet music with AI.</strong><br>
  Paste a YouTube link, upload audio, or drop a MIDI file — get a PDF score, MusicXML, and playable MIDI in seconds.
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#features">Features</a> &bull;
  <a href="#how-it-works">How It Works</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://github.com/swifttarrow/oh-sheet/actions/workflows/ci.yml/badge.svg" alt="CI">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/flutter-3.19%2B-02569B" alt="Flutter 3.19+">
  <img src="https://img.shields.io/badge/license-Proprietary-lightgrey" alt="License">
</p>

---

## Features

- **YouTube URL support** — Paste a YouTube link, Oh Sheet downloads the audio and transcribes it automatically
- **AI transcription** — Spotify's Basic Pitch detects notes from audio; optional Demucs stem separation isolates instruments first
- **Two-hand piano arrangement** — Melody goes to right hand, bass + harmony to left hand, with intelligent voice assignment
- **Humanized playback** — Micro-timing, velocity dynamics, pedal marks, and articulations make it sound natural
- **Publication-quality engraving** — LilyPond or MuseScore renders clean PDF sheet music; music21 generates MusicXML
- **Interactive viewer** — OSMD renders notation in the browser with Tone.js playback and cursor sync
- **Custom piano roll** — Canvas-based visualization with color-coded hands, Y-axis note labels, and tempo-synced beat grid
- **Real-time progress** — WebSocket events stream pipeline status with kawaii mascot animations per stage
- **TuneChat integration** — Push results to [TuneChat](https://github.com/robin-raq/TuneChat) rooms for collaborative practice

<p align="center">
  <img src="docs/wireframes/wireframe-option-c-landing.png" alt="Landing page wireframe" width="45%">
  &nbsp;&nbsp;
  <img src="docs/wireframes/wireframe-option-d-job-party.png" alt="Progress screen wireframe" width="45%">
</p>

## Quick Start

**Requirements:** Python 3.10+, Flutter SDK, ffmpeg

```bash
# Clone and install
git clone https://github.com/swifttarrow/oh-sheet.git
cd oh-sheet
make install                  # backend + frontend deps

# Optional: install ML deps for real transcription
make install-basic-pitch      # Spotify Basic Pitch (CPU, ~10s per song)

# Run
make backend                  # API on http://localhost:8000
make frontend                 # Flutter Web on Chrome
```

Open the app, paste a YouTube URL, and hit **Let's go!**

OpenAPI docs: [localhost:8000/docs](http://localhost:8000/docs)

## How It Works

```
YouTube URL / MP3 / MIDI
        |
        v
  ┌── INGEST ──┐    Download audio (yt-dlp), probe metadata
  └─────┬──────┘
        v
  ┌─ SEPARATE ─┐    Demucs splits vocals/drums/bass/other (optional)
  └─────┬──────┘
        v
  ┌ TRANSCRIBE ┐    Basic Pitch: audio → MIDI notes
  │            │    Beat tracking, tempo map, key detection
  └─────┬──────┘
        v
  ┌── ARRANGE ─┐    MIDI → two-hand piano score
  │            │    Melody → RH, bass + chords → LH
  └─────┬──────┘
        v
  ┌─ HUMANIZE ─┐    Add micro-timing, dynamics, pedal marks
  └─────┬──────┘
        v
  ┌── ENGRAVE ─┐    Score → PDF + MusicXML + MIDI
  └─────┬──────┘
        v
  PDF + MusicXML + Humanized MIDI
```

### Pipeline Variants

| Variant        | Input           | Stages                        | Use Case                           |
| -------------- | --------------- | ----------------------------- | ---------------------------------- |
| `full`         | YouTube URL     | All 6 stages                  | Paste a link, get sheet music      |
| `audio_upload` | MP3/WAV file    | Ingest → Transcribe → ... → Engrave | Upload your own recording     |
| `midi_upload`  | MIDI file       | Ingest → Arrange → ... → Engrave    | Skip transcription             |
| `sheet_only`   | Audio/MIDI      | Skip humanize                 | Clean quantized output             |

## Architecture

### Backend (Python 3.10+, FastAPI)

```
backend/
├── main.py                  # FastAPI app + uvicorn entry
├── config.py                # Pydantic settings (OHSHEET_* env vars)
├── contracts.py             # Pydantic v2 models (Schema v3.0.0)
├── services/
│   ├── ingest.py            # yt-dlp download + metadata probe
│   ├── stem_separation.py   # Demucs source separation
│   ├── audio_preprocess.py  # Normalization, silence trimming
│   ├── transcribe.py        # Basic Pitch (ONNX) + beat tracking
│   ├── arrange.py           # Two-hand piano reduction
│   ├── humanize.py          # Rule-based expression
│   └── engrave.py           # music21 → MusicXML, LilyPond → PDF
├── jobs/
│   ├── manager.py           # In-memory job state + WebSocket pub/sub
│   ├── runner.py            # Pipeline orchestration
│   └── events.py            # JobEvent schema
├── storage/
│   ├── base.py              # BlobStore protocol (Claim-Check pattern)
│   └── local.py             # file:// backed store (S3 next)
└── api/routes/
    ├── uploads.py           # POST /v1/uploads/{audio,midi}
    ├── jobs.py              # POST /v1/jobs, GET /v1/jobs/{id}
    ├── artifacts.py         # GET /v1/artifacts/{job_id}/{kind}
    ├── ws.py                # WS /v1/jobs/{id}/ws (live events)
    └── stages.py            # POST /v1/stages/{name} (worker endpoints)
```

### Frontend (Flutter 3.19+, Dart)

```
frontend/lib/
├── main.dart                # App shell + bottom nav (Home/Library/Profile)
├── theme.dart               # Kawaii sticker design system
├── screens/
│   ├── upload_screen.dart   # Audio / MIDI / Title / YouTube input
│   ├── progress_screen.dart # Mascot animations + stage badges
│   └── result_screen.dart   # Sheet music viewer + piano roll + downloads
└── widgets/
    ├── sheet_music_viewer.dart   # OSMD + Tone.js interactive notation
    ├── piano_roll.dart           # Custom canvas piano roll
    └── sticker_widgets.dart      # Kawaii UI components
```

### API Endpoints

| Method | Endpoint                          | Description                    |
| ------ | --------------------------------- | ------------------------------ |
| POST   | `/v1/uploads/audio`               | Upload MP3/WAV/FLAC/M4A       |
| POST   | `/v1/uploads/midi`                | Upload MIDI file               |
| POST   | `/v1/jobs`                        | Submit pipeline job             |
| GET    | `/v1/jobs/{id}`                   | Poll job status                |
| WS     | `/v1/jobs/{id}/ws`                | Live event stream              |
| GET    | `/v1/artifacts/{id}/{kind}`       | Download PDF/MIDI/MusicXML     |
| GET    | `/v1/health`                      | Health check                   |

### Mascot Gallery

The Oh Sheet! mascot has expressions for every pipeline stage:

<p align="center">
  <img src="frontend/assets/mascots/mascot-progress-ingest.svg" alt="Listening" height="100">
  <img src="frontend/assets/mascots/mascot-progress-transcribe.svg" alt="Transcribing" height="100">
  <img src="frontend/assets/mascots/mascot-progress-arrange.svg" alt="Arranging" height="100">
  <img src="frontend/assets/mascots/mascot-progress-engrave.svg" alt="Engraving" height="100">
  <img src="frontend/assets/mascots/mascot-success.svg" alt="Success!" height="100">
</p>

## Deployment

Oh Sheet runs on a single GCP VM with Docker Compose:

```bash
# Build and deploy
docker compose up -d

# Or use the GitHub Actions workflow (auto-deploys on push to main)
```

See `.github/workflows/deploy.yml` and `docker-compose.yml` for details.

## TuneChat Integration

Oh Sheet powers the sheet music in [TuneChat](https://github.com/robin-raq/TuneChat) — a real-time collaborative music learning platform. TuneChat uploads files to Oh Sheet's API, polls for results, and renders the MusicXML with OSMD in shared rooms.

```
TuneChat Client → TuneChat Server → Oh Sheet API → Pipeline → Artifacts → TuneChat Client
```

## Testing

```bash
make test          # pytest (backend) + flutter test (frontend)
make lint          # ruff check + flutter analyze
make typecheck     # mypy
```

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Write tests first (TDD)
4. Commit in small, focused chunks
5. Open a PR against `main`

See `CONTRIBUTING.md` for detailed guidelines.

## Tech Stack

| Component       | Technology                                          |
| --------------- | --------------------------------------------------- |
| Backend         | Python 3.10+, FastAPI, Pydantic v2                  |
| Transcription   | Basic Pitch (ONNX), Demucs (stem separation)        |
| Arrangement     | Custom Python (quantization, voice assignment)       |
| Engraving       | music21, LilyPond, pretty_midi                      |
| Frontend        | Flutter 3.19+ (Web + Mobile)                        |
| Sheet Viewer    | OpenSheetMusicDisplay (OSMD), Tone.js               |
| Deployment      | Docker Compose, GCP VM, GitHub Actions               |
| CI              | ruff, mypy, pytest, flutter analyze                  |

---

<p align="center">
  <img src="frontend/assets/mascots/mascot-success.svg" alt="Your sheet music is ready!" width="200">
  <br>
  <strong>Star this repo if you find it useful!</strong>
</p>
