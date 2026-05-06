# ---- Stage 1: Build frontend-v2 (vanilla-JS SPA via Vite) ----
#
# Replaces the previous Flutter build stage. frontend-v2 uses vanilla
# JavaScript + Vite; build artifacts go to frontend-v2/dist. Bundle size
# drops from ~6 MB (CanvasKit WASM + Dart runtime) to ~100 KB.
#
# node:20-slim is ~300 MB vs the Flutter SDK's ~3 GB — saves ~3-5 min
# off every qa deploy.
#
# TUNECHAT_URL dart-define is no longer needed: views.js derives the
# TuneChat origin at runtime from the backend's tunechat_preview_image_url
# field (which is stamped with the configured public base URL). No build
# arg required.
#
# API_BASE_URL is unused too — frontend-v2 uses same-origin relative
# URLs exclusively (served by the same backend on /).
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend-v2

# package-lock.json first so Docker caches npm ci separately from src
COPY frontend-v2/package.json frontend-v2/package-lock.json ./
RUN npm ci

# Sources + assets, then build
COPY frontend-v2/ .
RUN npm run build

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

RUN pip install --no-cache-dir --retries 5 --timeout 120 ".[pop2piano,demucs]"

# Pre-cache HTDemucs pretrained weights (~80 MB). Without this, every
# fresh container hits the first separate.run job with a cold-start
# weight download from dl.fbaipublicfiles.com — in Cloud Run that's
# ~10–20 s of avoidable latency per worker spawn. Pinning into the
# image is the standard "bake the model into the layer" trick used
# for Pop2Piano and Basic Pitch already.
RUN python -c "from demucs.pretrained import get_model; get_model('htdemucs')" \
    || echo "warning: htdemucs weights pre-cache failed (will fetch on first use)"

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
RUN pip install --no-cache-dir --retries 5 --timeout 120 ".[pop2piano,demucs]"

# Pre-cache HTDemucs weights (~80 MB) so the api image can run the
# separate worker via task_always_eager in tests / dev without paying
# a cold download. Mirrors the cache step in the ``ml`` target.
RUN python -c "from demucs.pretrained import get_model; get_model('htdemucs')" \
    || echo "warning: htdemucs weights pre-cache failed (will fetch on first use)"

# Copy frontend-v2's Vite build output into the static dir the backend
# serves. backend/main.py mounts /app/static as a catch-all StaticFiles
# route (see main.py _STATIC_DIR), so whatever lands here is what users
# see at the root URL.
COPY --from=frontend-build /app/frontend-v2/dist /app/static

# Cloud Run sets PORT; default to 8080
ENV PORT=8080

EXPOSE ${PORT}

CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}
