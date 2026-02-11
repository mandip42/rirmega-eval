from __future__ import annotations

import os
import secrets
from pathlib import Path

from rirmega_eval.utils.hashing import sha256_str


def _install_id_path() -> Path:
    return Path(".rirmega_eval_install_id")


def get_or_create_install_id() -> str:
    p = _install_id_path()
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    v = secrets.token_hex(16)
    p.write_text(v, encoding="utf-8")
    return v


def privacy_hash(value: str, salt: str) -> str:
    return sha256_str(f"{salt}::{value}")


def get_local_salt() -> str:
    # Optional: user can provide their own salt via env var to rotate anonymity scope.
    s = os.environ.get("RIRMEGA_EVAL_LEDGER_SALT")
    if s:
        return s
    # Otherwise derive from install id (still local-only).
    return sha256_str(get_or_create_install_id())[:32]

