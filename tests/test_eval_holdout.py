"""Tests for ``eval/holdout.py`` — the Phase 3 encrypted holdout split.

Phase 3 acceptance per
``docs/research/transcription-improvement-implementation-plan.md`` §Phase 3:

* 50/50 tune/holdout split is deterministic given a fixed seed.
* Holdout is encrypted; reading it requires the passphrase.
* The split is locked: re-running ``init`` with the same seed +
  passphrase produces the same tune-side slug list.

Tests assemble a small synthetic manifest via the loader fixture
helpers so they're hermetic — no dependency on the real
``eval/pop_eval_v1/`` slots being populated.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.holdout import (  # noqa: E402
    HOLDOUT_FILENAME,
    HOLDOUT_FORMAT_VERSION,
    HOLDOUT_KEY_ENV,
    TUNE_FILENAME,
    HoldoutError,
    HoldoutKeyMissingError,
    compute_split,
    init_holdout,
    load_holdout_slugs,
    load_tune_slugs,
)


def _write_manifest(eval_set_dir: Path, slugs: list[str]) -> None:
    eval_set_dir.mkdir(parents=True, exist_ok=True)
    songs = [
        {
            "slug": slug,
            "title": "x",
            "artist": "x",
            "genre": "pop_mainstream",
            "license_bucket": "fma_redistributable",
            "intended_source": {
                "kind": "fma_track", "url": "", "license": "cc-by", "notes": "",
            },
        }
        for slug in slugs
    ]
    (eval_set_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "eval_set": "test_split",
                "songs": songs,
            },
            sort_keys=False,
        )
    )


@pytest.fixture
def manifest_30(tmp_path):
    """30-slug manifest, mirroring pop_eval_v1's slot count."""
    eval_set = tmp_path / "set30"
    slugs = [f"song_{i:03d}" for i in range(30)]
    _write_manifest(eval_set, slugs)
    return eval_set, slugs


@pytest.fixture
def passphrase_env(monkeypatch):
    monkeypatch.setenv(HOLDOUT_KEY_ENV, "test-passphrase-do-not-use-in-prod")
    yield "test-passphrase-do-not-use-in-prod"


# ---------------------------------------------------------------------------
# compute_split — pure function, deterministic
# ---------------------------------------------------------------------------

def test_compute_split_50_50_for_even_count():
    slugs = [f"s{i}" for i in range(20)]
    split = compute_split(slugs, seed="abcd")
    assert len(split.tune) == 10
    assert len(split.holdout) == 10
    assert set(split.tune) | set(split.holdout) == set(slugs)
    assert set(split.tune).isdisjoint(split.holdout)


def test_compute_split_odd_count_extra_goes_to_holdout():
    """Odd totals: extra slug goes to holdout (conservative — never tuned on)."""
    slugs = [f"s{i}" for i in range(31)]
    split = compute_split(slugs, seed="abcd")
    assert len(split.tune) == 15
    assert len(split.holdout) == 16
    # Union still equals input.
    assert set(split.tune) | set(split.holdout) == set(slugs)


def test_compute_split_is_deterministic_for_same_seed():
    slugs = [f"s{i}" for i in range(30)]
    s1 = compute_split(slugs, seed="seed-A")
    s2 = compute_split(slugs, seed="seed-A")
    assert s1.tune == s2.tune
    assert s1.holdout == s2.holdout


def test_compute_split_changes_with_seed():
    slugs = [f"s{i}" for i in range(30)]
    s1 = compute_split(slugs, seed="seed-A")
    s2 = compute_split(slugs, seed="seed-B")
    assert (s1.tune, s1.holdout) != (s2.tune, s2.holdout)


def test_compute_split_seed_round_trip():
    split = compute_split(["a", "b", "c", "d"], seed="hello")
    assert split.seed == "hello"


# ---------------------------------------------------------------------------
# init_holdout — writes both files, idempotent given seed
# ---------------------------------------------------------------------------

def test_init_holdout_writes_both_manifests(manifest_30, passphrase_env):
    eval_set, _slugs = manifest_30
    split = init_holdout(eval_set, seed="locked-seed-1234")
    assert (eval_set / TUNE_FILENAME).is_file()
    assert (eval_set / HOLDOUT_FILENAME).is_file()
    assert len(split.tune) == 15
    assert len(split.holdout) == 15
    assert split.seed == "locked-seed-1234"


def test_init_holdout_refuses_to_overwrite_existing_split(manifest_30, passphrase_env):
    eval_set, _ = manifest_30
    init_holdout(eval_set, seed="x")
    with pytest.raises(HoldoutError, match="already exists"):
        init_holdout(eval_set, seed="x")


def test_init_holdout_overwrite_flag_replaces_existing(manifest_30, passphrase_env):
    eval_set, _ = manifest_30
    init_holdout(eval_set, seed="seed1")
    tune1 = load_tune_slugs(eval_set)
    init_holdout(eval_set, seed="seed2", overwrite=True)
    tune2 = load_tune_slugs(eval_set)
    # Different seeds → different tune sets.
    assert tune1 != tune2


def test_init_holdout_requires_passphrase_when_env_missing(manifest_30, monkeypatch):
    eval_set, _ = manifest_30
    monkeypatch.delenv(HOLDOUT_KEY_ENV, raising=False)
    with pytest.raises(HoldoutKeyMissingError):
        init_holdout(eval_set, seed="x")


def test_init_holdout_uses_explicit_passphrase_arg(manifest_30, monkeypatch):
    """Explicit ``passphrase=`` kwarg bypasses the env var."""
    eval_set, _ = manifest_30
    monkeypatch.delenv(HOLDOUT_KEY_ENV, raising=False)
    init_holdout(eval_set, seed="x", passphrase="explicit-passphrase")
    # Should have written the encrypted file.
    assert (eval_set / HOLDOUT_FILENAME).is_file()


# ---------------------------------------------------------------------------
# load_tune_slugs / load_holdout_slugs — the core read paths
# ---------------------------------------------------------------------------

def test_tune_slugs_readable_without_passphrase(manifest_30, passphrase_env, monkeypatch):
    eval_set, _ = manifest_30
    init_holdout(eval_set, seed="loaded")
    # Drop the env var; tune side should still be readable.
    monkeypatch.delenv(HOLDOUT_KEY_ENV)
    slugs = load_tune_slugs(eval_set)
    assert len(slugs) == 15


def test_holdout_slugs_decrypt_round_trip(manifest_30, passphrase_env):
    eval_set, _ = manifest_30
    split = init_holdout(eval_set, seed="loaded")
    decrypted = load_holdout_slugs(eval_set)
    assert sorted(decrypted) == sorted(split.holdout)


def test_holdout_slugs_require_correct_passphrase(manifest_30, passphrase_env):
    eval_set, _ = manifest_30
    init_holdout(eval_set, seed="loaded")
    # Passing the wrong passphrase should raise (Fernet InvalidToken).
    with pytest.raises(HoldoutError, match="decryption failed"):
        load_holdout_slugs(eval_set, passphrase="wrong-passphrase")


def test_holdout_slugs_raise_when_env_missing(manifest_30, passphrase_env, monkeypatch):
    eval_set, _ = manifest_30
    init_holdout(eval_set, seed="loaded")
    monkeypatch.delenv(HOLDOUT_KEY_ENV)
    with pytest.raises(HoldoutKeyMissingError):
        load_holdout_slugs(eval_set)


def test_holdout_slugs_disjoint_from_tune(manifest_30, passphrase_env):
    eval_set, slugs = manifest_30
    init_holdout(eval_set, seed="disjoint-test")
    tune = set(load_tune_slugs(eval_set))
    holdout_set = set(load_holdout_slugs(eval_set))
    assert tune.isdisjoint(holdout_set)
    assert tune | holdout_set == set(slugs)


# ---------------------------------------------------------------------------
# Encrypted-file shape — locked so accidental format drift is caught
# ---------------------------------------------------------------------------

def test_encrypted_envelope_has_expected_fields(manifest_30, passphrase_env):
    eval_set, _ = manifest_30
    init_holdout(eval_set, seed="x")
    envelope = yaml.safe_load((eval_set / HOLDOUT_FILENAME).read_text())
    assert envelope["format_version"] == HOLDOUT_FORMAT_VERSION
    assert isinstance(envelope["salt_hex"], str)
    assert len(envelope["salt_hex"]) == 32   # 16 bytes hex-encoded
    assert isinstance(envelope["ciphertext"], str)
    # Ciphertext must NOT contain any slug names in plaintext.
    for slug in (load_tune_slugs(eval_set)):
        assert slug not in envelope["ciphertext"]


def test_tune_manifest_has_expected_fields(manifest_30, passphrase_env):
    eval_set, _ = manifest_30
    init_holdout(eval_set, seed="x")
    tune_doc = yaml.safe_load((eval_set / TUNE_FILENAME).read_text())
    assert tune_doc["side"] == "tune"
    assert tune_doc["split_seed"] == "x"
    assert tune_doc["n_total"] == 30
    assert tune_doc["n_tune"] == 15
    assert tune_doc["n_holdout"] == 15
    assert isinstance(tune_doc["slugs"], list)


def test_load_holdout_rejects_unknown_format_version(manifest_30, passphrase_env):
    eval_set, _ = manifest_30
    init_holdout(eval_set, seed="x")
    envelope = yaml.safe_load((eval_set / HOLDOUT_FILENAME).read_text())
    envelope["format_version"] = 99
    (eval_set / HOLDOUT_FILENAME).write_text(yaml.safe_dump(envelope))
    with pytest.raises(HoldoutError, match="format_version"):
        load_holdout_slugs(eval_set)


# ---------------------------------------------------------------------------
# Empty / malformed manifest
# ---------------------------------------------------------------------------

def test_init_holdout_raises_on_empty_song_list(tmp_path, passphrase_env):
    eval_set = tmp_path / "empty"
    _write_manifest(eval_set, [])
    with pytest.raises(HoldoutError, match="no songs"):
        init_holdout(eval_set, seed="x")


def test_load_tune_raises_when_init_not_run(manifest_30):
    eval_set, _ = manifest_30
    with pytest.raises(HoldoutError, match="run init_holdout"):
        load_tune_slugs(eval_set)


def test_load_holdout_raises_when_init_not_run(manifest_30, passphrase_env):
    eval_set, _ = manifest_30
    with pytest.raises(HoldoutError, match="run init_holdout"):
        load_holdout_slugs(eval_set)
