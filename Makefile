# Oh Sheet — top-level orchestrator.
# Backend (Python): pyproject.toml + tests/ at the repo root, package at backend/.
# Frontend (Flutter): everything under frontend/.

FRONTEND := frontend

# Override on the command line, e.g.:
#   make frontend DEVICE=ios
#   make frontend API_BASE_URL=http://192.168.1.42:8000
#   make frontend FLUTTER=/opt/flutter/bin/flutter
DEVICE       ?= chrome
API_BASE_URL ?=
HOST         ?= 0.0.0.0
PORT         ?= 8000
FLUTTER      ?= flutter
BASE_IMAGE   ?= ghcr.io/oh-sheet-team/ohsheet-dev-base:latest

DART_DEFINE := $(if $(API_BASE_URL),--dart-define=API_BASE_URL=$(API_BASE_URL),)

.PHONY: help install install-backend install-basic-pitch install-pop2piano install-demucs install-eval install-frontend backend build rebuild frontend test test-backend test-e2e eval lint typecheck clean require-flutter require-port-free require-base-image

help:
	@echo "Oh Sheet — make targets"
	@echo ""
	@echo "  make install              full install: backend + Basic Pitch deps + flutter pub get"
	@echo "  make install-backend      pip install -e .[dev]  (API only — TranscribeService"
	@echo "                            will fall back to a 4-note stub without Basic Pitch)"
	@echo "  make install-basic-pitch  pip install -e .[basic-pitch]  (basic-pitch[onnx] + pretty_midi)"
	@echo "  make install-pop2piano    pip install -e .[pop2piano]  (Pop2Piano transformer; preferred path)"
	@echo "  make install-demucs       pip install -e .[demucs]  (demucs + torch; legacy stem split)"
	@echo "  make install-eval         pip install -e .[eval]  (mir_eval for the offline eval harness)"
	@echo "  make install-frontend     $(FLUTTER) pub get inside frontend/"
	@echo ""
	@echo "  make build              build the shared dev base image ($(BASE_IMAGE))"
	@echo "                          re-run when pyproject.toml, shared/, or Dockerfile.dev changes"
	@echo "  make backend            docker compose up (Redis + Celery workers + API on :8000)"
	@echo "                          requires 'make build' first"
	@echo "  make rebuild            shortcut for: make build && make backend"
	@echo "  make frontend           $(FLUTTER) run -d $(DEVICE) (override DEVICE=ios|android|macos|...)"
	@echo "                          set API_BASE_URL=http://host:port to point at a non-default backend"
	@echo "                          set FLUTTER=/path/to/flutter if the SDK is not on your PATH"
	@echo ""
	@echo "  make test               run backend pytest suite"
	@echo "  make eval               score TranscribeService on the eval/fixtures/clean_midi subset"
	@echo "                          (requires .[basic-pitch] + .[eval] + fluidsynth on PATH)"
	@echo "  make lint               ruff + $(FLUTTER) analyze"
	@echo "  make clean              remove build artifacts and the local blob store"

# ---- install ----------------------------------------------------------------

install: install-backend install-pop2piano install-basic-pitch install-frontend

require-flutter:
	@if [ -x "$(FLUTTER)" ] || command -v "$(FLUTTER)" >/dev/null 2>&1; then \
		:; \
	else \
		echo "Flutter SDK not found."; \
		echo "Install Flutter and make sure its bin directory is on your PATH."; \
		echo "Or rerun make with FLUTTER=/absolute/path/to/flutter."; \
		echo "Example: make frontend FLUTTER=\$$HOME/flutter/bin/flutter"; \
		exit 127; \
	fi

install-backend:
	pip install -e ".[dev]"

install-basic-pitch:
	# madmom has no pre-built wheels and its setup.py requires Cython +
	# setuptools at build time.  pip's default build-isolation doesn't
	# expose packages already in the venv, so we pre-install the build
	# deps and then build madmom without isolation.
	pip install setuptools Cython numpy
	pip install --no-build-isolation "madmom>=0.16"
	pip install -e ".[basic-pitch]"
	# basic-pitch 0.4.0 hard-codes tensorflow-macos as a base dep on
	# Darwin+Python>3.11 (no wheels for 3.13), so install it with
	# --no-deps and rely on [basic-pitch] above for the actual runtime
	# deps. See pyproject.toml comment for details.
	pip install --no-deps "basic-pitch>=0.4"

install-pop2piano:
	# Pop2Piano audio-to-piano transformer. On by default when deps are
	# installed (OHSHEET_POP2PIANO_ENABLED=1). Replaces Demucs + Basic
	# Pitch with a single transformer pass. essentia only ships x86_64
	# Linux + macOS wheels; Docker builds use platform: linux/amd64.
	pip install -e ".[pop2piano]"

install-demucs:
	# Optional stem-separation stack (demucs + torch). Off by default;
	# flip on via OHSHEET_DEMUCS_ENABLED=1. The htdemucs pretrained
	# weights are CC BY-NC 4.0 — see pyproject.toml for the commercial
	# caveat before enabling in production.
	pip install -e ".[demucs]"

install-eval:
	# Offline eval harness — mir_eval only. Assumes ``.[basic-pitch]``
	# is already installed (the harness drives the real TranscribeService
	# to score). Does not install fluidsynth; that's a system binary.
	pip install -e ".[eval]"

install-frontend: require-flutter
	cd $(FRONTEND) && $(FLUTTER) pub get

# ---- run --------------------------------------------------------------------

require-port-free:
	@if command -v lsof >/dev/null 2>&1 && lsof -tiTCP:$(PORT) -sTCP:LISTEN >/dev/null 2>&1; then \
		echo "Port $(PORT) is already in use."; \
		echo "Stop the existing process or rerun with a different port, e.g. make backend PORT=8001"; \
		lsof -nP -iTCP:$(PORT) -sTCP:LISTEN; \
		exit 1; \
	fi

backend: require-base-image require-port-free
	docker compose up

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

frontend: require-flutter
	cd $(FRONTEND) && $(FLUTTER) run -d $(DEVICE) $(DART_DEFINE)

# ---- quality ----------------------------------------------------------------

test: test-backend

test-backend:
	pytest

test-e2e:
	cd e2e && npx playwright test

# Score TranscribeService end-to-end against the tracked clean_midi
# subset at ``eval/fixtures/clean_midi/`` and write the full report
# to ``eval-baseline.json``. Requires the ``.[basic-pitch]`` and
# ``.[eval]`` extras plus ``fluidsynth`` on PATH (used to render the
# ground-truth MIDIs to WAV for the audio-in transcriber).
# See the script's module docstring for tuning / sampling options.
eval:
	python scripts/eval_transcription.py --out eval-baseline.json

lint:
	ruff check backend tests
	@$(MAKE) require-flutter
	cd $(FRONTEND) && $(FLUTTER) analyze

typecheck:
	mypy

# ---- housekeeping -----------------------------------------------------------

clean:
	rm -rf blob .pytest_cache backend/__pycache__ backend/**/__pycache__
	@if [ -x "$(FLUTTER)" ] || command -v "$(FLUTTER)" >/dev/null 2>&1; then \
		cd $(FRONTEND) && $(FLUTTER) clean; \
	fi
