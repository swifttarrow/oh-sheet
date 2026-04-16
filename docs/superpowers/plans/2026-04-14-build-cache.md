# Build Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `make backend` start in seconds instead of 10+ minutes by introducing a shared, cached dev base image.

**Architecture:** Split `Dockerfile.dev` into a single base image (`ohsheet-dev-base`) with BuildKit cache mounts, referenced by `image:` from every compose service. Remove `COPY backend/` (bind mount covers it). Add Makefile `build`/`backend`/`rebuild` split so `make backend` never rebuilds.

**Tech Stack:** Docker + BuildKit (cache mounts, `# syntax=docker/dockerfile:1.6`), Docker Compose v2, GNU Make, Python 3.12-slim.

**Spec:** [`docs/superpowers/specs/2026-04-14-build-cache-design.md`](../specs/2026-04-14-build-cache-design.md)
**Linear:** [GAU-108](https://linear.app/gauntletai-kevin/issue/GAU-108/speed-up-make-backend-build-currently-10min)

---

## File Structure

**Modified files:**
- `Dockerfile.dev` — rewrite: BuildKit cache mounts, drop `COPY backend/`, add syntax directive.
- `Makefile` — split `backend` target; add `build` and `rebuild` targets; add `BASE_IMAGE` variable and help text.
- `docker-compose.yml` — remove all six `build:` blocks; replace with `image: ${BASE_IMAGE:-...}`.
- `README.md` — update onboarding/run instructions (if it documents `make backend`; verify in Task 0).

**No new files.** No code changes to `backend/` or `shared/`.

---

## Task 0: Baseline & context check

**Files:** read-only.

- [ ] **Step 1: Confirm current state**

Run: `git status && git rev-parse --abbrev-ref HEAD`
Expected: clean tree (or only the spec commit), branch `feat/build-cache`.

- [ ] **Step 2: Confirm BuildKit is available**

Run: `docker buildx version`
Expected: prints a version string (e.g., `github.com/docker/buildx v0.x`). If this fails, stop and tell the user — the plan requires BuildKit.

- [ ] **Step 3: Identify README sections to update**

Run: `grep -n -E "make backend|docker compose" README.md || echo "no matches"`
Record the line numbers; Task 5 uses them. If no matches, Task 5 becomes a no-op and should be skipped.

- [ ] **Step 4: Time the baseline (optional but recommended)**

Run: `time docker compose build --progress=plain 2>&1 | tail -5`
Record wall-clock time for before/after comparison in the final verification task. Skip if the user doesn't want to wait ~10 min.

No commit — this task only gathers info.

---

## Task 1: Rewrite `Dockerfile.dev`

**Files:**
- Modify: `Dockerfile.dev` (full rewrite)

- [ ] **Step 1: Replace the file contents**

Write this exact content to `Dockerfile.dev`:

```dockerfile
# syntax=docker/dockerfile:1.6
# Dev base image for Oh Sheet — shared by every compose service.
#
# NOT standalone: backend/ is intentionally NOT copied in. docker-compose
# bind-mounts ./backend and ./shared at runtime so source edits are live
# without a rebuild. Rebuild this image only when pyproject.toml,
# shared/, or this Dockerfile changes (see `make build`).
#
# Platform note: essentia (required by Pop2Piano) only ships x86_64 Linux
# wheels, so compose pins platform: linux/amd64. On Apple Silicon this
# runs under Rosetta (~10-20% slower).
FROM python:3.12-slim

WORKDIR /app

# System deps: ffmpeg (yt-dlp), lilypond (engrave), gcc/g++ (madmom Cython build).
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg gcc g++ lilypond \
    && rm -rf /var/lib/apt/lists/*

# Shared package first (changes less often than pyproject.toml).
COPY shared/ shared/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e ./shared

# Heavy ML install. madmom needs Cython + numpy at build time; basic-pitch
# 0.4 pins tensorflow-macos on Darwin so we install it with --no-deps and
# rely on the [basic-pitch] extra for actual runtime deps.
COPY pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install setuptools Cython numpy \
 && pip install --no-build-isolation "madmom>=0.16" \
 && pip install ".[pop2piano,basic-pitch]" \
 && pip install --no-deps "basic-pitch>=0.4"
```

- [ ] **Step 2: Lint the Dockerfile syntax**

Run: `docker buildx build --platform linux/amd64 -f Dockerfile.dev -t ohsheet-dev-base:test --progress=plain . 2>&1 | tail -20`
Expected: build succeeds. This is the real cold build — expect several minutes. On success, you'll see `=> naming to docker.io/library/ohsheet-dev-base:test`.

- [ ] **Step 3: Verify the image does NOT contain backend/**

Run: `docker run --rm ohsheet-dev-base:test ls /app`
Expected: output shows `pyproject.toml` and `shared` but **no `backend`** directory. This confirms the COPY removal took effect.

- [ ] **Step 4: Verify cache-mount behavior with a no-op rebuild**

Run: `time docker buildx build --platform linux/amd64 -f Dockerfile.dev -t ohsheet-dev-base:test . 2>&1 | tail -3`
Expected: completes in under ~10 seconds, all layers `CACHED`.

- [ ] **Step 5: Commit**

```bash
git add Dockerfile.dev
git commit -m "build(dev): add BuildKit cache mounts, drop backend COPY (GAU-108)"
```

---

## Task 2: Update `Makefile` — targets and variables

**Files:**
- Modify: `Makefile` (edits around lines 17, 19, 21-46, 109-110)

- [ ] **Step 1: Add `BASE_IMAGE` variable**

In `Makefile`, after the existing `FLUTTER ?= flutter` line (around line 15), add:

```make
BASE_IMAGE ?= ghcr.io/oh-sheet-team/ohsheet-dev-base:latest
```

- [ ] **Step 2: Add new targets to `.PHONY`**

Find the existing `.PHONY:` line (around line 19) and add `build rebuild` to the list. The line should read:

```make
.PHONY: help install install-backend install-basic-pitch install-pop2piano install-demucs install-eval install-frontend backend build rebuild frontend test test-backend test-e2e eval lint typecheck clean require-flutter require-port-free require-base-image
```

(Also includes `require-base-image` — added in Step 4.)

- [ ] **Step 3: Replace the `backend:` target body**

Find the existing `backend:` target (around line 109-110). Replace:

```make
backend:
	docker compose up --build
```

with:

```make
backend: require-base-image require-port-free
	docker compose up
```

- [ ] **Step 4: Add `build`, `rebuild`, and `require-base-image` targets**

Insert the following immediately after the new `backend:` target:

```make
build:
	DOCKER_BUILDKIT=1 docker build \
		--platform linux/amd64 \
		-f Dockerfile.dev \
		-t $(BASE_IMAGE) .

rebuild: build backend

require-base-image:
	@if ! docker image inspect $(BASE_IMAGE) >/dev/null 2>&1; then \
		echo "Base image $(BASE_IMAGE) not found locally."; \
		echo "Run 'make build' first (one-time; re-run when pyproject.toml,"; \
		echo "shared/, or Dockerfile.dev changes)."; \
		exit 1; \
	fi
```

- [ ] **Step 5: Update the `help` target**

Find the `help:` target (around line 21-42). Replace the "run" section block (the lines describing `make backend` and `make frontend`) with:

```make
	@echo "  make build              build the shared dev base image ($(BASE_IMAGE))"
	@echo "                          re-run when pyproject.toml, shared/, or Dockerfile.dev changes"
	@echo "  make backend            docker compose up (Redis + Celery workers + API on :8000)"
	@echo "                          requires 'make build' first"
	@echo "  make rebuild            shortcut for: make build && make backend"
	@echo "  make frontend           $(FLUTTER) run -d $(DEVICE) (override DEVICE=ios|android|macos|...)"
	@echo "                          set API_BASE_URL=http://host:port to point at a non-default backend"
	@echo "                          set FLUTTER=/path/to/flutter if the SDK is not on your PATH"
```

- [ ] **Step 6: Verify Makefile parses**

Run: `make help | head -30`
Expected: shows the new `make build`, `make backend`, `make rebuild` lines. No "missing separator" or similar errors.

- [ ] **Step 7: Verify `require-base-image` gate works**

Run: `docker image rm ghcr.io/oh-sheet-team/ohsheet-dev-base:latest 2>/dev/null; make backend 2>&1 | head -5`
Expected: exits non-zero with the "Base image ... not found locally" message. Do not let it start compose.

- [ ] **Step 8: Build the base image via `make build`**

Run: `make build`
Expected: succeeds. Because Task 1 already populated BuildKit cache layers for the same Dockerfile, this should be fast (mostly CACHED). Confirms the target works end-to-end.

- [ ] **Step 9: Commit**

```bash
git add Makefile
git commit -m "build(dev): split make backend into build/backend/rebuild (GAU-108)"
```

---

## Task 3: Update `docker-compose.yml` to use the prebuilt image

**Files:**
- Modify: `docker-compose.yml` (lines 12-32, 34-50, 52-68, 70-86, 88-104, 106-122 — all six services)

- [ ] **Step 1: Replace each service's `build:` block with `image:`**

For **each** of the six services (`orchestrator`, `worker-ingest`, `worker-transcribe`, `worker-arrange`, `worker-humanize`, `worker-engrave`), find this block:

```yaml
    build:
      context: .
      dockerfile: Dockerfile.dev
```

and replace it with:

```yaml
    image: ${BASE_IMAGE:-ghcr.io/oh-sheet-team/ohsheet-dev-base:latest}
```

Keep `platform: linux/amd64`, `command:`, `ports:`, `environment:`, `volumes:`, `depends_on:` exactly as they are. Do NOT change the `redis` service (it uses `image: redis:7-alpine` already).

- [ ] **Step 2: Validate the compose file parses**

Run: `docker compose config --quiet`
Expected: no output, exit 0. Any YAML or reference error surfaces here.

- [ ] **Step 3: Confirm all six services resolved to the base image**

Run: `docker compose config | grep -c 'image: ghcr.io/oh-sheet-team/ohsheet-dev-base'`
Expected: `6`.

- [ ] **Step 4: Smoke-test `make backend`**

Run: `make backend` in one terminal.
Expected: containers start in seconds (no build phase). Wait until you see uvicorn log `Application startup complete` and at least one `celery@... ready` line.

In another terminal, verify: `curl -sf http://localhost:8000/docs -o /dev/null && echo OK`
Expected: `OK`.

Shut down: Ctrl-C in the `make backend` terminal, then `docker compose down`.

- [ ] **Step 5: Smoke-test bind mount still works**

Edit any comment in `backend/main.py` (add a trailing `# smoke test`), then start `make backend` again. The orchestrator container should pick up the edit without a rebuild. Revert the edit before committing. (If `uvicorn --reload` is active, you'll see it reload; otherwise presence of the edited line inside the container via `docker compose exec orchestrator grep 'smoke test' backend/main.py` confirms the mount.)

Shut down when done.

- [ ] **Step 6: Commit**

```bash
git add docker-compose.yml
git commit -m "build(dev): use shared ohsheet-dev-base image across services (GAU-108)"
```

---

## Task 4: Update `README.md` onboarding (conditional)

**Files:**
- Modify: `README.md` (use line numbers from Task 0 Step 3)

Skip this entire task if Task 0 Step 3 reported "no matches."

- [ ] **Step 1: Locate the run instructions**

Open `README.md` at the line numbers recorded in Task 0. Identify any block that tells users to run `make backend` as a first step after install.

- [ ] **Step 2: Insert the `make build` step**

Wherever the README shows an onboarding sequence like:

```
make install
make backend
```

replace with:

```
make install
make build       # one-time; re-run when pyproject.toml, shared/, or Dockerfile.dev changes
make backend
```

If the README describes `make backend` prose-style ("run `make backend` to start the stack"), add one sentence: "The first time, run `make build` to build the shared dev image."

- [ ] **Step 3: Verify no other references are stale**

Run: `grep -n "docker compose up --build" README.md docs/ 2>/dev/null || echo clean`
Expected: `clean`, or only the spec/plan files under `docs/superpowers/` (those are historical and shouldn't be edited).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document make build step in onboarding (GAU-108)"
```

---

## Task 5: End-to-end verification

**Files:** none.

- [ ] **Step 1: Clean-slate warm-path verification**

With the base image already built (from Task 2), run:

```bash
docker compose down -v 2>/dev/null
time make backend
```

In another terminal once it's up: `curl -sf http://localhost:8000/docs -o /dev/null && echo OK`
Expected: `make backend` reaches "Application startup complete" in well under 30 seconds. `OK` prints. Ctrl-C to stop, then `docker compose down`.

- [ ] **Step 2: No-change rebuild verification**

Run: `time make build`
Expected: completes in under ~15 seconds; every layer `CACHED` in the output.

- [ ] **Step 3: Backend-source-edit verification**

Edit `backend/main.py` (add a trailing comment), then:

```bash
time make build
```

Expected: still fast (< 15s, all CACHED). Because `COPY backend/` is gone, backend edits no longer invalidate the ML layer. Revert the edit.

- [ ] **Step 4: Dep-edit verification (optional)**

Add a harmless comment line to `pyproject.toml`, then `time make build`. Expected: reruns the pip install step but hits the pip cache mount — faster than cold (~2-3 min vs. ~10 min). Revert.

- [ ] **Step 5: Run a real pipeline job (golden path)**

Start the stack: `make backend` (in one terminal).
In another terminal, trigger a job using the existing test harness or manual endpoint. Minimal check:

```bash
curl -sf http://localhost:8000/v1/uploads/audio -X POST -F "file=@eval/fixtures/clean_midi/<any>.mid" | jq .
```

(Adjust to whatever fixture exists; the point is to confirm all six services are responsive, not to test pipeline correctness.) Shut down with Ctrl-C + `docker compose down`.

- [ ] **Step 6: Record timings in Linear**

Update GAU-108 with before/after numbers captured in Task 0 Step 4 vs. this task's Steps 1-2. No code commit.

---

## Self-Review Notes

- **Spec coverage:** Dockerfile rewrite (Task 1), Makefile split (Task 2), compose service changes (Task 3), README (Task 4), verification plan (Task 5). All spec sections mapped.
- **Out-of-scope respected:** no GHCR publish workflow, no prod Dockerfile edits, no svc-decomposer / svc-assembler changes, no Rosetta / platform changes.
- **No placeholders:** all code blocks and shell commands are literal.
- **Identifier consistency:** `BASE_IMAGE`, `ohsheet-dev-base`, `require-base-image` used identically across all tasks.
- **Commit cadence:** one commit per task (1-4), no commit for verification (Task 5), matches project convention of small focused commits.
