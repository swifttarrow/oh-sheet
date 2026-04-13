"""V14-V19 Settings-level tests for CFG-03, CFG-05, CFG-07.

Covers:
  - V14: SecretStr masks across 6 serialization paths
  - V15: .get_secret_value() is the only reveal path
  - V16-V18: model allowlist rejection (unsupported, opus gating, frozen)
  - V19: refine knob defaults and env-var override

All tests use direct Settings(...) constructor calls — do NOT re-import
backend.config.settings (module singleton) because that's cached at import
time and doesn't re-read env vars between tests. See 01-RESEARCH.md Pitfall 7.
"""
from __future__ import annotations

import json

import pytest
from pydantic import SecretStr, ValidationError

from backend.config import (
    _ALLOWED_REFINE_MODELS_OPUS,
    _ALLOWED_REFINE_MODELS_SONNET,
    Settings,
)

_RAW_MOCK_KEY = "sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"  # pragma: allowlist secret


# ---------------------------------------------------------------------------
# V14 — SecretStr masking
# ---------------------------------------------------------------------------


def test_secret_str_does_not_leak_key_in_model_dump() -> None:
    """CFG-03: SecretStr's model_dump output is a masked representation, not raw."""
    s = Settings(anthropic_api_key=SecretStr(_RAW_MOCK_KEY))
    dumped = s.model_dump()
    # Pydantic's SecretStr serializes as SecretStr('**********') in model_dump
    # (the default serialize-as is "display" which masks). Either it's not the
    # raw value, or we assert specifically.
    assert _RAW_MOCK_KEY not in str(dumped)


def test_secret_str_does_not_leak_key_in_model_dump_json() -> None:
    """CFG-03: model_dump_json output must not contain the raw key."""
    s = Settings(anthropic_api_key=SecretStr(_RAW_MOCK_KEY))
    dumped_json = s.model_dump_json()
    assert _RAW_MOCK_KEY not in dumped_json
    # Sanity: the JSON is still valid
    parsed = json.loads(dumped_json)
    assert "anthropic_api_key" in parsed


def test_secret_str_does_not_leak_key_in_repr_or_fstring() -> None:
    """CFG-03: repr, str, f-string, and %s-format all mask the key."""
    s = Settings(anthropic_api_key=SecretStr(_RAW_MOCK_KEY))
    key = s.anthropic_api_key
    assert key is not None
    assert _RAW_MOCK_KEY not in repr(key)
    assert _RAW_MOCK_KEY not in str(key)
    assert _RAW_MOCK_KEY not in f"{key}"
    assert _RAW_MOCK_KEY not in "%s" % key


# ---------------------------------------------------------------------------
# V15 — reveal only on get_secret_value
# ---------------------------------------------------------------------------


def test_secret_str_reveals_raw_only_on_get_secret_value() -> None:
    """CFG-03: .get_secret_value() is the only documented reveal path."""
    s = Settings(anthropic_api_key=SecretStr(_RAW_MOCK_KEY))
    assert s.anthropic_api_key is not None
    assert s.anthropic_api_key.get_secret_value() == _RAW_MOCK_KEY


# ---------------------------------------------------------------------------
# V16-V18 — allowlist enforcement
# ---------------------------------------------------------------------------


def test_allowlist_rejects_unsupported_model() -> None:
    """CFG-05: arbitrary model names rejected at Settings() instantiation."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(refine_model="gpt-4")
    assert "allowlist" in str(exc_info.value).lower() or "not in" in str(exc_info.value).lower()


def test_allowlist_rejects_opus_without_flag() -> None:
    """CFG-05: Opus models rejected when refine_allow_opus=False (default)."""
    with pytest.raises(ValidationError):
        Settings(refine_model="claude-opus-4-6", refine_allow_opus=False)


def test_allowlist_accepts_opus_with_flag() -> None:
    """CFG-05: Opus models accepted when refine_allow_opus=True."""
    s = Settings(refine_model="claude-opus-4-6", refine_allow_opus=True)
    assert s.refine_model == "claude-opus-4-6"
    assert s.refine_allow_opus is True


def test_allowlist_accepts_default_sonnet() -> None:
    """CFG-05: Default settings (refine_model='claude-sonnet-4-6') succeed."""
    s = Settings()
    assert s.refine_model == "claude-sonnet-4-6"
    assert s.refine_allow_opus is False


def test_allowlist_is_frozen_and_not_dynamically_mutable() -> None:
    """CFG-05 (integrity): frozenset disallows runtime mutation of the allowlist."""
    assert isinstance(_ALLOWED_REFINE_MODELS_SONNET, frozenset)
    assert isinstance(_ALLOWED_REFINE_MODELS_OPUS, frozenset)
    # frozenset has no .add(); attempting it raises AttributeError
    with pytest.raises(AttributeError):
        _ALLOWED_REFINE_MODELS_SONNET.add("arbitrary-model")  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        _ALLOWED_REFINE_MODELS_OPUS.add("arbitrary-model")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# V19 — refine knob defaults + env-var override
# ---------------------------------------------------------------------------


def test_refine_knobs_load_from_env_with_defaults() -> None:
    """CFG-07: refine knobs have documented defaults AND can be overridden via env."""
    # Defaults
    defaults = Settings()
    assert defaults.refine_max_tokens == 4096
    assert defaults.refine_web_search_max_uses == 5
    assert defaults.refine_max_retries == 3
    assert defaults.refine_kill_switch is False


def test_refine_knobs_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """CFG-07: OHSHEET_REFINE_MAX_TOKENS etc. override the field defaults."""
    monkeypatch.setenv("OHSHEET_REFINE_MAX_TOKENS", "8192")
    monkeypatch.setenv("OHSHEET_REFINE_WEB_SEARCH_MAX_USES", "10")
    monkeypatch.setenv("OHSHEET_REFINE_MAX_RETRIES", "5")
    monkeypatch.setenv("OHSHEET_REFINE_KILL_SWITCH", "true")

    s = Settings()
    assert s.refine_max_tokens == 8192
    assert s.refine_web_search_max_uses == 10
    assert s.refine_max_retries == 5
    assert s.refine_kill_switch is True


def test_anthropic_api_key_default_is_none() -> None:
    """CFG-03: anthropic_api_key defaults to None — the CFG-04 400-on-missing-key case."""
    s = Settings()
    assert s.anthropic_api_key is None


# ---------------------------------------------------------------------------
# D-13 — refine_ghost_velocity_max field (Phase 2, Plan 01)
# ---------------------------------------------------------------------------


def test_refine_ghost_velocity_max_default_is_40(monkeypatch: pytest.MonkeyPatch) -> None:
    """D-13: default 40 — configurable via OHSHEET_REFINE_GHOST_VELOCITY_MAX."""
    monkeypatch.delenv("OHSHEET_REFINE_GHOST_VELOCITY_MAX", raising=False)
    s = Settings(_env_file=None)  # _env_file=None prevents .env pollution
    assert s.refine_ghost_velocity_max == 40


def test_refine_ghost_velocity_max_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """D-13: env var overrides default — Phase-4 A/B tuning hook."""
    monkeypatch.setenv("OHSHEET_REFINE_GHOST_VELOCITY_MAX", "25")
    s = Settings(_env_file=None)
    assert s.refine_ghost_velocity_max == 25


@pytest.mark.parametrize("bad", ["-1", "128", "256"])
def test_refine_ghost_velocity_max_rejects_out_of_range(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    """D-13: bounds [0, 127] enforced at Settings() instantiation."""
    monkeypatch.setenv("OHSHEET_REFINE_GHOST_VELOCITY_MAX", bad)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
