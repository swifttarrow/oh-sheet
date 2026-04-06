# Oh Sheet! 🎵

An automated pipeline that transforms any song (MP3/MIDI) into playable piano sheet music — with a web/mobile frontend for uploading, tracking progress, and downloading results.

## What it does

```
MP3 or Song Link → Transcription → Piano Arrangement → Humanization → Sheet Music (PDF + MusicXML)
```

1. **Upload** — User submits an MP3 file or song link via the Oh Sheet web app
2. **Transcribe** — Full-mix audio transcription using MT3 (or custom conformer model)
3. **Arrange** — Reduce multi-instrument transcription to a two-handed piano score
4. **Humanize** — Add micro-timing, dynamics, and pedaling for natural-sounding playback
5. **Engrave** — Generate MusicXML and PDF sheet music
6. **Deliver** — User downloads PDF or opens results in TuneChat for collaborative practice

## Architecture

Oh Sheet has two components:

### Frontend (SPA)
Upload/search interface with live progress bars via WebSocket updates, results view with client-side playback, and PDF download.

### Pipeline (Python)
ML-powered music processing service that handles transcription, arrangement, humanization, and engraving.

```
┌──────────────┐                ┌──────────────┐                ┌──────────────┐
│  Oh Sheet    │   job request  │  Oh Sheet    │   artifacts    │  TuneChat    │
│  Frontend    │ ──────────────►│  Pipeline    │ ──────────────►│  (optional)  │
│  (SPA)       │                │  (Python)    │                │              │
│              │◄─── progress ──│              │                │  Rooms       │
│  Upload      │   via WS      │  MT3         │                │  Shared Piano│
│  Progress    │                │  music21     │                │  AI Coach    │
│  Download    │                │  LilyPond    │                │  Live Playback│
└──────────────┘                └──────────────┘                └──────────────┘
```

## Integration with TuneChat

Oh Sheet can optionally push results to [TuneChat](https://github.com/robin-raq/TuneChat) — a real-time collaborative music learning platform built by [Raq Dominique](https://github.com/robin-raq). When results are delivered to a TuneChat room, users can:

- View the sheet music as interactive notation
- Play along on a shared piano (on-screen, computer keyboard, or USB MIDI controller)
- Practice together in real-time with other users
- Ask an AI coach for help with specific passages

### Delivering results to TuneChat

```bash
# Send progress updates to a TuneChat room
curl -X POST https://<tunechat-host>/api/v1/service/rooms/{room_id}/messages \
  -H "Authorization: Bearer $SERVICE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "Arranging for piano...", "display_as": "Pipeline"}'

# Deliver completed artifacts
curl -X POST https://<tunechat-host>/api/v1/service/rooms/{room_id}/artifacts \
  -H "Authorization: Bearer $SERVICE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "job": {"job_id": "abc-123", "status": "completed"},
    "artifacts": [
      {"kind": "musicxml", "url": "https://storage.example.com/score.xml"},
      {"kind": "pdf", "url": "https://storage.example.com/sheet.pdf"},
      {"kind": "humanized_midi", "url": "https://storage.example.com/piano.mid"}
    ],
    "title": "Apple - Charli XCX"
  }'
```

Full API spec: [TuneChat API Contracts](https://github.com/robin-raq/TuneChat/blob/master/docs/api-contracts.md)

## Tech Stack

| Component | Tech |
|-----------|------|
| Frontend | TBD (React Native / Flutter) |
| Transcription | MT3 / Custom Conformer |
| Arrangement | music21 |
| Humanization | Rule-based + ML |
| Engraving | music21 → MusicXML, LilyPond → PDF |

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `SERVICE_TOKEN` | Shared secret for authenticating with TuneChat service endpoints |
| `TUNECHAT_URL` | TuneChat server URL (e.g. `http://localhost:3000` for local dev) |

## Contributors

Built by the Oh Sheet team. Collaborative playback platform powered by [TuneChat](https://github.com/robin-raq/TuneChat) by [Raq Dominique](https://github.com/robin-raq).
