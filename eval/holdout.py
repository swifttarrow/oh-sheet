"""Encrypted tune/holdout split for the Phase 3 paid eval set.

Splits an eval-set manifest 50/50 into a tune set (engineer-readable,
used for hyperparameter tuning + CI gates) and a holdout set
(encrypted, used only for release-gate runs). The holdout is encrypted
with Fernet (AES-128-CBC + HMAC-SHA-256) using a passphrase-derived
key stored in the ``OHSHEET_HOLDOUT_KEY`` environment variable.

The threat model is **"engineer accidentally tunes against holdout
while debugging a tune-set regression."** A determined engineer with
filesystem access can find the holdout slugs by reading the loader
output once they get the passphrase from the release manager. We don't
defend against that; we just make accidental peeking impossible.

The split itself is deterministic:

    bucket = sha256(slug || split_seed)[0:8]
    sorted_slugs = slugs sorted by bucket
    tune     = first  half of sorted_slugs
    holdout  = second half of sorted_slugs

``split_seed`` is committed in plaintext alongside the encrypted
manifest so the split is reproducible after a key rotation. Once
published as ``pop_eval_v1.0.0``, the seed is frozen.

Public surface
--------------

* :func:`compute_split` — pure function, deterministic 50/50 split
* :func:`init_holdout` — produce ``holdout_manifest.yaml.enc`` from a
  manifest + passphrase. Idempotent given the same inputs.
* :func:`load_tune_slugs` / :func:`load_holdout_slugs` — read back the
  tune / holdout side. ``load_holdout_slugs`` requires the passphrase.

CLI
---

::

    OHSHEET_HOLDOUT_KEY=… python -m eval.holdout init eval/pop_eval_v1/
    python -m eval.holdout list-tune eval/pop_eval_v1/
    OHSHEET_HOLDOUT_KEY=… python -m eval.holdout list-holdout eval/pop_eval_v1/
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import secrets
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Module-level constants stay close to the encryption code that uses
# them; tests assert against these to lock the on-disk format.
HOLDOUT_FILENAME = "holdout_manifest.yaml.enc"
TUNE_FILENAME = "tune_manifest.yaml"          # plaintext companion for the tune side

# Environment variable holding the passphrase. Mirrors the convention
# in scripts/curate_pop_mini_v0.py for ``OHSHEET_*`` env vars.
HOLDOUT_KEY_ENV = "OHSHEET_HOLDOUT_KEY"

# PBKDF2-HMAC-SHA-256 derivation parameters. Iterations chosen for
# ~50ms on a 2024-class M-series Mac — fast enough that ``init`` is
# still snappy, slow enough that brute-forcing a leaked encrypted
# manifest is non-trivial. The salt is per-eval-set, stored in the
# ``salt`` field of the encrypted manifest.
PBKDF2_ITERATIONS = 200_000
PBKDF2_KEY_LEN = 32                            # AES-256 key material (Fernet uses urlsafe-base64-encoded 32-byte key)
PBKDF2_SALT_BYTES = 16

# Format version for the encrypted manifest. Bump when changing the
# wire format so the loader can refuse to decrypt an unfamiliar blob
# instead of silently producing garbage.
HOLDOUT_FORMAT_VERSION = 1

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class HoldoutError(RuntimeError):
    """Generic holdout-pipeline error (split mismatch, decrypt failure, etc.)."""


class HoldoutKeyMissingError(HoldoutError):
    """Raised when the holdout passphrase is not available in the environment.

    Caught explicitly by the CLI so engineers see "set OHSHEET_HOLDOUT_KEY"
    instead of a cryptic Fernet decryption error.
    """


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HoldoutSplit:
    """Result of :func:`compute_split` — both sides + the seed used."""

    tune: list[str]
    holdout: list[str]
    seed: str                                  # hex-encoded; committed in the manifest

    def all_slugs(self) -> list[str]:
        return [*self.tune, *self.holdout]


# ---------------------------------------------------------------------------
# Pure split — deterministic, no encryption
# ---------------------------------------------------------------------------

def compute_split(slugs: Iterable[str], *, seed: str) -> HoldoutSplit:
    """Return a deterministic 50/50 split of ``slugs`` keyed on ``seed``.

    Slugs are bucketed by ``sha256(slug || seed)[0:8]`` and sorted by
    bucket; the lower half goes to tune, the upper half to holdout.
    Even-numbered totals split exactly 50/50; odd-numbered totals put
    the extra slug in the holdout (so 31 songs → 15 tune / 16 holdout).
    Putting the extra in holdout is the conservative choice: holdout
    is never seen by tuning loops, so an extra song there only widens
    the release gate.
    """
    seed_bytes = seed.encode("utf-8")
    keyed = []
    for slug in slugs:
        h = hashlib.sha256(seed_bytes + b"|" + slug.encode("utf-8")).digest()
        bucket = int.from_bytes(h[:8], "big")
        keyed.append((bucket, slug))
    keyed.sort()                                # ascending by bucket
    sorted_slugs = [slug for _, slug in keyed]
    n = len(sorted_slugs)
    tune_n = n // 2
    return HoldoutSplit(
        tune=sorted_slugs[:tune_n],
        holdout=sorted_slugs[tune_n:],
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Encryption — Fernet with PBKDF2-derived key
# ---------------------------------------------------------------------------

def _derive_fernet_key(passphrase: str, salt: bytes) -> bytes:
    """Derive a Fernet key (32 bytes, urlsafe-base64-encoded) from a passphrase + salt."""
    if not passphrase:
        raise HoldoutKeyMissingError(
            f"Empty passphrase. Set ${HOLDOUT_KEY_ENV} to a non-empty value."
        )
    raw = hashlib.pbkdf2_hmac(
        "sha256",
        passphrase.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
        dklen=PBKDF2_KEY_LEN,
    )
    return base64.urlsafe_b64encode(raw)


def _encrypt_payload(payload: dict[str, Any], passphrase: str) -> dict[str, Any]:
    """Encrypt ``payload`` (a JSON-serializable dict) with Fernet.

    Returns the on-disk envelope: ``{format_version, salt_hex,
    ciphertext}``. The salt is stored in plaintext alongside the
    ciphertext so anyone with the passphrase can decrypt; without the
    passphrase the salt alone reveals nothing.
    """
    from cryptography.fernet import Fernet  # noqa: PLC0415 — keep cryptography import lazy

    salt = secrets.token_bytes(PBKDF2_SALT_BYTES)
    key = _derive_fernet_key(passphrase, salt)
    fernet = Fernet(key)
    plaintext = json.dumps(payload, sort_keys=True).encode("utf-8")
    ciphertext = fernet.encrypt(plaintext)
    return {
        "format_version": HOLDOUT_FORMAT_VERSION,
        "salt_hex": salt.hex(),
        "ciphertext": ciphertext.decode("ascii"),
    }


def _decrypt_payload(envelope: dict[str, Any], passphrase: str) -> dict[str, Any]:
    from cryptography.fernet import Fernet, InvalidToken  # noqa: PLC0415

    fmt = envelope.get("format_version")
    if fmt != HOLDOUT_FORMAT_VERSION:
        raise HoldoutError(
            f"unsupported holdout envelope format_version={fmt}; "
            f"this build expects {HOLDOUT_FORMAT_VERSION}"
        )
    salt_hex = envelope.get("salt_hex")
    ciphertext = envelope.get("ciphertext")
    if not isinstance(salt_hex, str) or not isinstance(ciphertext, str):
        raise HoldoutError("malformed holdout envelope (salt_hex / ciphertext missing)")
    salt = bytes.fromhex(salt_hex)
    key = _derive_fernet_key(passphrase, salt)
    fernet = Fernet(key)
    try:
        plaintext = fernet.decrypt(ciphertext.encode("ascii"))
    except InvalidToken as exc:
        raise HoldoutError(
            "holdout decryption failed — wrong passphrase or tampered ciphertext"
        ) from exc
    return json.loads(plaintext)


# ---------------------------------------------------------------------------
# init / load — top-level operations
# ---------------------------------------------------------------------------

def get_passphrase(*, required: bool = True) -> str | None:
    """Read ``$OHSHEET_HOLDOUT_KEY``; raise (or return None) when missing."""
    value = os.environ.get(HOLDOUT_KEY_ENV)
    if value:
        return value
    if required:
        raise HoldoutKeyMissingError(
            f"${HOLDOUT_KEY_ENV} is unset. Ask the release manager for the "
            f"holdout passphrase, or run `python -m eval.holdout init` to "
            f"create one for a new eval set."
        )
    return None


def init_holdout(
    eval_set_path: Path,
    *,
    passphrase: str | None = None,
    seed: str | None = None,
    overwrite: bool = False,
) -> HoldoutSplit:
    """Generate ``holdout_manifest.yaml.enc`` and ``tune_manifest.yaml``.

    Reads the manifest's ``songs`` list, computes the 50/50 split,
    writes the tune slugs as plaintext YAML and the holdout slugs as
    Fernet-encrypted JSON. Idempotent: re-running with the same
    ``passphrase`` + ``seed`` produces the same split (Fernet's
    ciphertext changes per-call due to its random IV, but the
    underlying slug list does not).

    ``seed`` defaults to a freshly-generated 16-byte hex string; pin a
    specific seed when re-creating an existing split.
    """
    eval_set_path = Path(eval_set_path)
    if passphrase is None:
        passphrase = get_passphrase(required=True) or ""

    holdout_path = eval_set_path / HOLDOUT_FILENAME
    tune_path = eval_set_path / TUNE_FILENAME
    if (holdout_path.exists() or tune_path.exists()) and not overwrite:
        raise HoldoutError(
            f"split already exists at {holdout_path} (or {tune_path}); "
            f"pass overwrite=True to regenerate"
        )

    # Lazy-load the loader to avoid a circular import (loader doesn't
    # touch holdout, but holdout reads the manifest via loader).
    from eval.loader import load_manifest  # noqa: PLC0415

    manifest = load_manifest(eval_set_path)
    slugs = [
        entry["slug"]
        for entry in manifest["songs"]
        if isinstance(entry, dict) and entry.get("slug")
    ]
    if not slugs:
        raise HoldoutError(f"no songs in {eval_set_path / 'manifest.yaml'}")

    if seed is None:
        seed = secrets.token_hex(16)

    split = compute_split(slugs, seed=seed)

    # Write the tune side as plaintext YAML — readable by anyone with
    # the repo. The schema mirrors the encrypted holdout side so a
    # reader can union them and recover the full slug list.
    tune_doc = {
        "format_version": HOLDOUT_FORMAT_VERSION,
        "split_seed": seed,
        "side": "tune",
        "n_total": len(slugs),
        "n_tune": len(split.tune),
        "n_holdout": len(split.holdout),
        "slugs": list(split.tune),
    }
    tune_path.write_text(yaml.safe_dump(tune_doc, sort_keys=False))

    holdout_doc = {
        "split_seed": seed,
        "side": "holdout",
        "n_total": len(slugs),
        "slugs": list(split.holdout),
    }
    envelope = _encrypt_payload(holdout_doc, passphrase)
    holdout_path.write_text(yaml.safe_dump(envelope, sort_keys=False))

    return split


def load_tune_slugs(eval_set_path: Path) -> list[str]:
    """Read the plaintext tune-side slug list. No passphrase needed."""
    eval_set_path = Path(eval_set_path)
    tune_path = eval_set_path / TUNE_FILENAME
    if not tune_path.is_file():
        raise HoldoutError(f"no tune manifest at {tune_path}; run init_holdout first")
    raw = yaml.safe_load(tune_path.read_text())
    if not isinstance(raw, dict) or "slugs" not in raw:
        raise HoldoutError(f"malformed tune manifest at {tune_path}")
    return [str(s) for s in raw["slugs"]]


def load_holdout_slugs(
    eval_set_path: Path,
    *,
    passphrase: str | None = None,
) -> list[str]:
    """Decrypt the holdout side. **Requires the passphrase.**"""
    eval_set_path = Path(eval_set_path)
    holdout_path = eval_set_path / HOLDOUT_FILENAME
    if not holdout_path.is_file():
        raise HoldoutError(
            f"no holdout manifest at {holdout_path}; run init_holdout first"
        )

    if passphrase is None:
        passphrase = get_passphrase(required=True) or ""

    envelope = yaml.safe_load(holdout_path.read_text())
    if not isinstance(envelope, dict):
        raise HoldoutError(f"malformed holdout envelope at {holdout_path}")

    payload = _decrypt_payload(envelope, passphrase)
    return [str(s) for s in payload.get("slugs", [])]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_init(args: argparse.Namespace) -> int:
    split = init_holdout(
        Path(args.eval_set_path),
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print("split written:")
    print(f"  tune   ({len(split.tune)}): {', '.join(split.tune)}")
    print(f"  holdout({len(split.holdout)}): <encrypted>")
    print(f"  seed:                    {split.seed}")
    return 0


def _cmd_list_tune(args: argparse.Namespace) -> int:
    slugs = load_tune_slugs(Path(args.eval_set_path))
    for s in slugs:
        print(s)
    return 0


def _cmd_list_holdout(args: argparse.Namespace) -> int:
    try:
        slugs = load_holdout_slugs(Path(args.eval_set_path))
    except HoldoutKeyMissingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except HoldoutError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for s in slugs:
        print(s)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Tune/holdout split for Oh Sheet eval sets.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Generate the encrypted holdout split.")
    p_init.add_argument("eval_set_path", type=str)
    p_init.add_argument(
        "--seed", type=str, default=None,
        help="Pin a specific seed (hex); default auto-generates one.",
    )
    p_init.add_argument(
        "--overwrite", action="store_true",
        help="Replace an existing split (CAUTION — invalidates prior baselines).",
    )
    p_init.set_defaults(func=_cmd_init)

    p_tune = sub.add_parser("list-tune", help="Print the tune-side slug list.")
    p_tune.add_argument("eval_set_path", type=str)
    p_tune.set_defaults(func=_cmd_list_tune)

    p_hold = sub.add_parser(
        "list-holdout",
        help=f"Decrypt and print the holdout-side slug list. Requires ${HOLDOUT_KEY_ENV}.",
    )
    p_hold.add_argument("eval_set_path", type=str)
    p_hold.set_defaults(func=_cmd_list_holdout)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
