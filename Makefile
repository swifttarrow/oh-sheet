# Oh Sheet — top-level orchestrator.
# Backend (Python): pyproject.toml + tests/ at the repo root, package at backend/.
# Frontend (Flutter): everything under frontend/.

FRONTEND := frontend

# Override on the command line, e.g.:
#   make frontend DEVICE=ios
#   make frontend API_BASE_URL=http://192.168.1.42:8000
DEVICE       ?= chrome
API_BASE_URL ?=
HOST         ?= 0.0.0.0
PORT         ?= 8000

DART_DEFINE := $(if $(API_BASE_URL),--dart-define=API_BASE_URL=$(API_BASE_URL),)

.PHONY: help install install-backend install-basic-pitch install-demucs install-eval install-frontend backend frontend test test-backend test-e2e eval lint typecheck clean

help:
	@echo "Oh Sheet — make targets"
	@echo ""
	@echo "  make install              full install: backend + Basic Pitch deps + flutter pub get"
	@echo "  make install-backend      pip install -e .[dev]  (API only — TranscribeService"
	@echo "                            will fall back to a 4-note stub without Basic Pitch)"
	@echo "  make install-basic-pitch  pip install -e .[basic-pitch]  (basic-pitch[onnx] + pretty_midi)"
	@echo "  make install-demucs       pip install -e .[demucs]  (demucs + torch; opt-in stem split)"
	@echo "  make install-eval         pip install -e .[eval]  (mir_eval for the offline eval harness)"
	@echo "  make install-frontend     flutter pub get inside frontend/"
	@echo ""
	@echo "  make backend            docker-compose up (Redis + Celery workers + API on :8000)"
	@echo "  make frontend           flutter run -d $(DEVICE) (override DEVICE=ios|android|macos|...)"
	@echo "                          set API_BASE_URL=http://host:port to point at a non-default backend"
	@echo ""
	@echo "  make test               run backend pytest suite"
	@echo "  make eval               score TranscribeService on the eval/fixtures/clean_midi subset"
	@echo "                          (requires .[basic-pitch] + .[eval] + fluidsynth on PATH)"
	@echo "  make lint               flutter analyze"
	@echo "  make clean              remove build artifacts and the local blob store"

# ---- install ----------------------------------------------------------------

install: install-backend install-basic-pitch install-frontend

install-backend:
	pip install -e ".[dev]"

install-basic-pitch:
	pip install -e ".[basic-pitch]"
	# basic-pitch 0.4.0 hard-codes tensorflow-macos as a base dep on
	# Darwin+Python>3.11 (no wheels for 3.13), so install it with
	# --no-deps and rely on [basic-pitch] above for the actual runtime
	# deps. See pyproject.toml comment for details.
	pip install --no-deps "basic-pitch>=0.4"

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

install-frontend:
	cd $(FRONTEND) && flutter pub get

# ---- run --------------------------------------------------------------------

backend:
	docker compose up --build

frontend:
	cd $(FRONTEND) && flutter run -d $(DEVICE) $(DART_DEFINE)

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
	cd $(FRONTEND) && flutter analyze

typecheck:
	mypy

# ---- housekeeping -----------------------------------------------------------

clean:
	rm -rf blob .pytest_cache backend/__pycache__ backend/**/__pycache__
	cd $(FRONTEND) && flutter clean || true
