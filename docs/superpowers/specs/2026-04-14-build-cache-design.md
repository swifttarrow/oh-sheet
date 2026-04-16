# Build Cache for `make backend`

**Linear:** [GAU-108](https://linear.app/gauntletai-kevin/issue/GAU-108/speed-up-make-backend-build-currently-10min)
**Date:** 2026-04-14
**Status:** Approved — ready for implementation plan

## Problem

`make backend` (which runs `docker compose up --build`) takes >10 minutes on every invocation, even when nothing has changed. The pain is felt most on no-change reruns — layer caching should make these near-instant, but doesn't.

## Root causes

1. **6 independent builds per run.** Compose builds `Dockerfile.dev` once per service (orchestrator + 5 workers). Even on cache hits, BuildKit evaluates context and tags 6 images serially.
2. **`COPY backend/` invalidates the ML install layer** on any `.py` edit, despite `./backend` being bind-mounted at runtime (so the COPY is useless for dev).
3. **No BuildKit cache mounts** — pip and apt caches are discarded each build. A cache miss re-downloads torch, essentia, transformers, and rebuilds madmom from source via Cython.
4. **`--build` flag forces re-evaluation every run**, even when nothing changed.

## Goals

- `make backend` with no changes: **< 10s** (down from ~10 min).
- `make build` warm, no dep changes: **< 15s**.
- `make build` after `pyproject.toml` edit: **< 3 min** (pip cache avoids redownload).
- Onboarding cold path unchanged for now; GHCR publishing is a follow-up.

## Architecture

Split into two images:

1. **`ohsheet-dev-base`** — heavy, slow-changing. Python 3.12 + system deps (ffmpeg, lilypond, gcc) + shared package + ML deps (pop2piano, basic-pitch, madmom). Tagged `ghcr.io/oh-sheet-team/ohsheet-dev-base:latest` from day one so GHCR adoption later is a one-line CI change, not a refactor.
2. **Compose services** — all 6 services reference `image: ohsheet-dev-base` directly with **no `build:` block**. Source continues to arrive via bind mounts on `./backend` and `./shared`.

## Changes

### `Dockerfile.dev` (rewrite)

```dockerfile
# syntax=docker/dockerfile:1.6
FROM python:3.12-slim

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg gcc g++ lilypond \
    && rm -rf /var/lib/apt/lists/*

COPY shared/ shared/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e ./shared

COPY pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install setuptools Cython numpy \
 && pip install --no-build-isolation "madmom>=0.16" \
 && pip install ".[pop2piano,basic-pitch]" \
 && pip install --no-deps "basic-pitch>=0.4"

# backend/ intentionally NOT copied — bind-mounted by compose at runtime.
```

Key deltas:
- Cache mounts on apt + pip survive across rebuilds.
- `--no-cache-dir` removed (cache mount handles persistence properly).
- `COPY backend/` removed — was the primary cache-buster.
- Image cannot run standalone; requires compose bind mount. Prod `Dockerfile` is untouched.

### `Makefile`

```make
BASE_IMAGE ?= ghcr.io/oh-sheet-team/ohsheet-dev-base:latest

backend: require-port-free
	docker compose up

build:
	DOCKER_BUILDKIT=1 docker build \
		--platform linux/amd64 \
		-f Dockerfile.dev \
		-t $(BASE_IMAGE) .

rebuild: build backend
```

- `make backend` — `docker compose up` only. No rebuild.
- `make build` — rebuild base image. Run when `pyproject.toml`, `Dockerfile.dev`, or `shared/` changes.
- `make rebuild` — convenience shortcut.
- `make backend` surfaces a clear error if the base image is missing, instructing the user to run `make build`.

### `docker-compose.yml`

For every service (orchestrator + 5 workers):
- Remove the `build:` block.
- Add `image: ${BASE_IMAGE:-ghcr.io/oh-sheet-team/ohsheet-dev-base:latest}`.

Bind mounts, environment, depends_on, command all unchanged.

### Documentation

Update Makefile `help` target and README onboarding to reflect the new flow:

```
make install
make build       # one-time; re-run when pyproject.toml or Dockerfile.dev changes
make backend
```

## Verification

1. `docker system prune -a`, then time `make build` — baseline cold build.
2. `make build` again with no changes — should complete in seconds (cache mounts + layer cache).
3. Touch `backend/main.py`, run `make backend` — should start in seconds (no rebuild path).
4. Touch `pyproject.toml`, run `make build` — should reuse pip cache; only affected layer rebuilds.
5. `make backend` brings up all 6 services from the single base image. Run a sample pipeline job end-to-end and confirm all celery workers and uvicorn respond.

## Out of scope

- Publishing `ohsheet-dev-base` to GHCR from CI (follow-up ticket).
- Prod `Dockerfile` (multi-stage, production) — untouched.
- `svc-decomposer` and `svc-assembler` Dockerfiles — separate services, separate tickets.
- Dropping `platform: linux/amd64` / Rosetta — blocked by essentia wheel availability.
