# Version Tracking Design

**Date**: 2026-04-10
**Status**: Approved

## Overview

Automated semantic versioning for the Oh Sheet monorepo. Every PR merged to `main` automatically increments the project version based on conventional commit prefixes, and the version is displayed in the frontend UI and backend health endpoint.

## Decisions

- **Single unified version**: All components (backend, shared, decomposer, assembler, frontend) share one version number.
- **Conventional Commits**: `fix:` = patch, `feat:` = minor, `BREAKING CHANGE` = major.
- **Tool**: `python-semantic-release` — Python-native, mature, configurable via `pyproject.toml`.
- **UI display**: Persistent footer on every screen.
- **Backend display**: `GET /v1/health` response includes version and commit SHA.

## Version Source of Truth & Sync

`pyproject.toml` (root) holds the canonical version. On every bump, the following files are updated to match:

| File | Field | Example |
|------|-------|---------|
| `pyproject.toml` | `version` | `"0.2.0"` |
| `backend/__init__.py` | `__version__` | `"0.2.0"` |
| `shared/pyproject.toml` | `version` | `"0.2.0"` |
| `svc-decomposer/pyproject.toml` | `version` | `"0.2.0"` |
| `svc-assembler/pyproject.toml` | `version` | `"0.2.0"` |
| `frontend/pubspec.yaml` | `version` | `0.2.0+<build_number>` |

`python-semantic-release` handles `pyproject.toml`, `backend/__init__.py`, and the sub-package TOML files via its `version_variables` and `version_toml` config. A small shell step syncs `frontend/pubspec.yaml` using `sed`: the semver portion is set to match the new version, and the build number (the `+N` suffix) is incremented by 1 from its current value.

## CI/CD Workflow: release.yml

Triggers on push to `main`. Steps:

1. **Checkout** with `fetch-depth: 0` (full history for commit parsing).
2. **Run `python-semantic-release version`** — parses commits since last tag, determines bump, updates version files, creates commit and tag (e.g., `v0.2.0`).
3. **Sync pubspec.yaml** — reads new version from `pyproject.toml`, writes to `frontend/pubspec.yaml`, amends the release commit.
4. **Push** release commit + tag to `main` using a GitHub App or PAT (so the push triggers subsequent workflows).
5. **Exit cleanly** if no bumpable commits (only `docs:`, `chore:`, etc.).

Existing `deploy.yml` triggers on the subsequent push to `main`, building and deploying with the new version baked in.

### Token Requirement

The default `GITHUB_TOKEN` does not trigger downstream workflows when pushing. A GitHub App installation token or PAT with `contents: write` is required for the push step so that `deploy.yml` fires.

## Frontend Version Display

Version injected at Docker build time via `--dart-define`:

```dockerfile
# In Dockerfile, Flutter build stage
ARG APP_VERSION=dev
RUN flutter build web --release --dart-define=APP_VERSION=${APP_VERSION}
```

Dart reads it as a compile-time constant:

```dart
const appVersion = String.fromEnvironment('APP_VERSION', defaultValue: 'dev');
```

Displayed as a low-opacity text widget anchored to the bottom of every screen's `Scaffold`. Consistent across upload, progress, and result screens.

The `deploy.yml` workflow extracts the version from `pyproject.toml` and passes it as a Docker build arg.

## Backend Health Endpoint

Extend `GET /v1/health` to return:

```json
{
  "status": "ok",
  "version": "0.2.0",
  "commit": "b86fc58"
}
```

- `version`: read from `backend.__version__` (synced by semantic-release).
- `commit`: read from the existing `COMMIT_SHA` environment variable (already passed in `docker-compose.prod.yml`).

## Slack Notification Enhancement

The existing deploy success Slack message can include the version, read from the `pyproject.toml` in the workflow or from the git tag.

## python-semantic-release Configuration

Added to root `pyproject.toml`:

```toml
[tool.semantic_release]
version_toml = [
    "pyproject.toml:project.version",
    "shared/pyproject.toml:project.version",
    "svc-decomposer/pyproject.toml:project.version",
    "svc-assembler/pyproject.toml:project.version",
]
version_variables = [
    "backend/__init__.py:__version__",
]
branch = "main"
commit_message = "chore(release): v{version}"
tag_format = "v{version}"
build_command = false
upload_to_repository = false
```

## What Does Not Change

- Commit convention: already in use, no behavior change for contributors.
- Deploy workflow: still triggers on push to main, still uses commit SHA for Docker image tags.
- Docker compose: `COMMIT_SHA` env var remains, version is additive.

## Out of Scope

- Changelog generation (can be added later via semantic-release's built-in support).
- GitHub Releases (can be enabled in semantic-release config when desired).
- Per-service independent versioning.
