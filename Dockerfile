# ---- Stage 1: Build Flutter web app ----
FROM ghcr.io/cirruslabs/flutter:stable AS flutter-build

WORKDIR /app/frontend
COPY frontend/ .
RUN flutter pub get
# Empty API_BASE_URL → client uses same-origin relative URLs (/v1/...)
ARG APP_VERSION=dev
RUN flutter build web --release --dart-define=API_BASE_URL= --dart-define=APP_VERSION=${APP_VERSION}

# ---- Stage 2: Python runtime ----
# Python 3.12 for forward-compatibility with torch (MT3 model) which
# does not yet support 3.13.
FROM python:3.12-slim

WORKDIR /app

# System deps:
#   ffmpeg   — yt-dlp audio extraction
#   lilypond — MusicXML → PDF via musicxml2ly + lilypond (the engrave
#              service falls back to a 60-byte stub PDF without this)
#   curl     — health checks / debug
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        lilypond \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install shared package first (changes less often)
COPY shared/ shared/
RUN pip install --no-cache-dir ./shared

# Install Python package with Pop2Piano transcription deps.
# NOTE: essentia only ships x86_64 Linux wheels — build with
# --platform linux/amd64 if targeting Apple Silicon hosts.
COPY pyproject.toml .
COPY backend/ backend/
RUN pip install --no-cache-dir ".[pop2piano]"

# Copy Flutter web build into a directory the backend will serve
COPY --from=flutter-build /app/frontend/build/web /app/static

# Cloud Run sets PORT; default to 8080
ENV PORT=8080

EXPOSE ${PORT}

CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}
