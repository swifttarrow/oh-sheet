# ---- Stage 1: Build Flutter web app ----
FROM ghcr.io/cirruslabs/flutter:stable AS flutter-build

WORKDIR /app/frontend
COPY frontend/ .
RUN flutter pub get
# Empty API_BASE_URL → client uses same-origin relative URLs (/v1/...)
ARG APP_VERSION=dev
RUN flutter build web --release --dart-define=API_BASE_URL= --dart-define=APP_VERSION=${APP_VERSION}

# ---- Stage 2: Shared Python base ----
# Layers common to both the `api` and `ml` final targets: interpreter,
# system packages, the ohsheet-shared wheel, and the backend source tree.
# Splitting this out means BuildKit caches it once and both downstream
# targets reuse the same base layers, so building both images costs
# barely more than building one.
#
# Python 3.12 for forward-compatibility with torch (MT3 model) which
# does not yet support 3.13.
FROM python:3.12-slim AS python-base

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

# Install shared package first (changes less often). --retries/--timeout
# harden the install against transient PyPI TLS flakes (`SSLError: record
# layer failure`) — see incident in run 24407031842 / job 71293097136.
COPY shared/ shared/
RUN pip install --no-cache-dir --retries 5 --timeout 120 ./shared

COPY pyproject.toml .
COPY backend/ backend/

# ---- Stage 3a: ML image target ----
# Staging ground for step 3 of the image-split migration. Not yet
# consumed by any compose service — the ml target exists today only so
# CI exercises the multi-target build and catches regressions before
# worker-transcribe / worker-arrange start using it. Build explicitly
# with `docker build --target ml .` to produce this image.
#
# NOTE: essentia only ships x86_64 Linux wheels — build with
# --platform linux/amd64 if targeting Apple Silicon hosts.
FROM python-base AS ml

RUN pip install --no-cache-dir --retries 5 --timeout 120 ".[pop2piano]"

CMD celery -A backend.workers.celery_app worker --loglevel=warning

# ---- Stage 3b: API image target (default) ----
# This MUST remain the last stage in the file so that `docker build .`
# (no --target flag) continues to produce the same image the deploy
# workflow has always published. Every compose service that references
# ${IMAGE} in docker-compose.prod.yml resolves to this target. Reordering
# or renaming this stage without also updating .github/workflows/deploy.yml
# will silently ship the wrong image.
FROM python-base AS api

# Install Python package with Pop2Piano transcription deps.
# NOTE: essentia only ships x86_64 Linux wheels — build with
# --platform linux/amd64 if targeting Apple Silicon hosts.
RUN pip install --no-cache-dir --retries 5 --timeout 120 ".[pop2piano]"

# Copy Flutter web build into a directory the backend will serve
COPY --from=flutter-build /app/frontend/build/web /app/static

# Cloud Run sets PORT; default to 8080
ENV PORT=8080

EXPOSE ${PORT}

CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}
