"""V20-V24 pre-commit hook verification (CFG-08, T-1-06).

Layered tests:
  V20 — gitignore excludes .env
  V21 — .pre-commit-config.yaml exists and references detect-secrets
  V22 — .secrets.baseline is valid JSON
  V23 — LOAD-BEARING: hook blocks a file containing a mock Anthropic key
  V24 — no false positives: hook returns 0 on the clean repo

The load-bearing test (V23) proves the hook actually works — not just that
the config files exist. A passing V21+V22 with a failing V23 means the
config is present but detectable secrets still sneak through; a config-only
test suite would miss that.
"""
from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_PRECOMMIT_CONFIG = _REPO_ROOT / ".pre-commit-config.yaml"
_SECRETS_BASELINE = _REPO_ROOT / ".secrets.baseline"
_GITIGNORE = _REPO_ROOT / ".gitignore"


# ---------------------------------------------------------------------------
# V20 — .env is gitignored
# ---------------------------------------------------------------------------


def test_gitignore_excludes_dotenv() -> None:
    """CFG-08 success criterion 5a: .env is gitignored."""
    assert _GITIGNORE.is_file(), f"{_GITIGNORE} not found"
    lines = [line.strip() for line in _GITIGNORE.read_text().splitlines()]
    assert ".env" in lines, (
        f".env not found in .gitignore lines. Current lines: {lines!r}. "
        "CFG-08 requires .env to be gitignored to prevent accidental commits "
        "of OHSHEET_ANTHROPIC_API_KEY."
    )


# ---------------------------------------------------------------------------
# V21 — .pre-commit-config.yaml exists and references detect-secrets
# ---------------------------------------------------------------------------


def test_precommit_config_exists() -> None:
    """CFG-08: .pre-commit-config.yaml exists at repo root."""
    assert _PRECOMMIT_CONFIG.is_file(), (
        f"{_PRECOMMIT_CONFIG} does not exist. CFG-08 requires a pre-commit config."
    )


def test_precommit_config_has_detect_secrets_hook() -> None:
    """CFG-08: config references Yelp/detect-secrets at v1.5.0 with --baseline."""
    content = _PRECOMMIT_CONFIG.read_text()
    assert "Yelp/detect-secrets" in content, (
        ".pre-commit-config.yaml must reference the Yelp/detect-secrets repo. "
        f"Content:\n{content}"
    )
    assert "v1.5.0" in content, (
        ".pre-commit-config.yaml should pin detect-secrets to v1.5.0 for reproducibility."
    )
    assert ".secrets.baseline" in content, (
        ".pre-commit-config.yaml must reference .secrets.baseline in hook args."
    )
    assert "detect-secrets" in content  # hook id


# ---------------------------------------------------------------------------
# V22 — baseline is valid JSON
# ---------------------------------------------------------------------------


def test_secrets_baseline_exists() -> None:
    """CFG-08: .secrets.baseline exists — operators can whitelist false positives."""
    assert _SECRETS_BASELINE.is_file(), (
        f"{_SECRETS_BASELINE} does not exist. Create via: "
        "uvx --from detect-secrets==1.5.0 detect-secrets scan "
        "--exclude-files '^(tests/fixtures/|.*\\.ipynb$|\\.secrets\\.baseline$)' "
        "> .secrets.baseline"
    )


def test_secrets_baseline_is_valid_json() -> None:
    """CFG-08: .secrets.baseline must parse as JSON (detect-secrets baseline format)."""
    data = json.loads(_SECRETS_BASELINE.read_text())
    # detect-secrets baseline has top-level keys: version, plugins_used, etc.
    assert "version" in data, list(data.keys())
    assert "plugins_used" in data, list(data.keys())


# ---------------------------------------------------------------------------
# V23 — LOAD-BEARING end-to-end test
# ---------------------------------------------------------------------------


def test_detect_secrets_blocks_mock_anthropic_key(tmp_path: Path) -> None:
    """CFG-08 + T-1-06: hook blocks a file containing a mock Anthropic key.

    The load-bearing CFG-08 assertion. Proves the full pre-commit → detect-secrets
    chain actually intercepts a commit-shaped secret, not just that config files
    exist. Uses a 70-char high-entropy mock key designed to trip detect-secrets's
    HighEntropyString detector regardless of whether a named AnthropicDetector
    plugin exists in v1.5.0.
    """
    # Mock Anthropic-style key: 70 chars, high entropy. The pragma keeps
    # detect-secrets from flagging THIS test file (line below) while still
    # letting the subprocess assertion flag the tmp file we build from it.
    mock_key = "sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789AbCdEfGhIjKlMnOpQrSt"  # pragma: allowlist secret
    mock_key_file = tmp_path / "leak.py"
    mock_key_file.write_text(
        textwrap.dedent(
            f"""
            # Accidentally committed key for testing detect-secrets
            ANTHROPIC_API_KEY = "{mock_key}"
            """
        ).lstrip()
    )

    result = subprocess.run(
        ["pre-commit", "run", "detect-secrets", "--files", str(mock_key_file)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,  # allow first-run hook venv bootstrap
    )

    # Non-zero exit = hook rejected the file (load-bearing assertion).
    assert result.returncode != 0, (
        f"pre-commit hook did NOT block a mock Anthropic key.\n"
        f"stdout: {result.stdout!r}\n"
        f"stderr: {result.stderr!r}\n"
        f"Check .pre-commit-config.yaml and .secrets.baseline configuration."
    )
    # Sanity: output should name detect-secrets or contain a secret-detection marker.
    combined = result.stdout + result.stderr
    assert any(
        marker in combined.lower()
        for marker in ("detect-secrets", "potential", "secret")
    ), (
        f"hook failed but output does not contain a recognizable detect-secrets "
        f"signal: {combined!r}"
    )


# ---------------------------------------------------------------------------
# V24 — no false positives on the clean repo
# ---------------------------------------------------------------------------


def test_detect_secrets_does_not_false_positive_on_repo_code() -> None:
    """CFG-08: `pre-commit run detect-secrets --all-files` on the clean repo exits 0.

    Guards against a too-narrow baseline (hook misses real secrets) or a too-broad
    baseline (hook flags legitimate code). The committed .secrets.baseline should
    record every pre-existing high-entropy-but-safe string so future commits only
    fail on NEW secret-shaped content.
    """
    result = subprocess.run(
        ["pre-commit", "run", "detect-secrets", "--all-files"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,  # full-repo scan is slower
    )
    assert result.returncode == 0, (
        f"detect-secrets --all-files failed on the clean repo — baseline may be too narrow, "
        f"or an unbaseline-d high-entropy string exists in the working tree.\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
