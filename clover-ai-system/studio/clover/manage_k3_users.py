"""Manage the list of users allowed to access the k3 hosted models (`LFM3-Omni (k3)`, `Clover-35B-A3B`,
`Nemotron 3.5 Super VL`, `Kimi K3.1 Pro`) on `k3` (`allowed-accounts.json`).

Usage on k3 (`/opt/clover-k3/web-studio/clover/manage_k3_users.py`):
  python3.12 manage_k3_users.py list
  python3.12 manage_k3_users.py add --email user@gmail.com --github username --name "Full Name"
  python3.12 manage_k3_users.py remove user@gmail.com
  python3.12 manage_k3_users.py token user@gmail.com

Each approved user in `allowed-accounts.json` can unlock the 4 k3 models on their Phone, Windows Laptop, or Web Studio
using ANY of:
  1. Their GitHub Personal Access Token (`ghp_...` / `github_pat_...`) — either in `Settings -> GitHub account` or
     `Settings -> Model and voice (+ Add model…)`.
  2. Their Google / Gmail OIDC / OAuth2 token.
  3. Their signed `acct:<account>:<hmac>` token printed by this tool.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import secrets
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_BASE = Path(os.environ.get("CLOVER_WEB_BASE") or ("/opt/clover-k3/web-studio" if Path("/opt/clover-k3/web-studio").is_dir() else str(HERE.parent)))
DEFAULT_OWNER_GITHUB = (os.environ.get("CLOVER_OWNER_GITHUB") or "santhoshsathish94").strip().lower().lstrip("@")
DEFAULT_OWNER_EMAIL = (os.environ.get("CLOVER_OWNER_EMAIL") or "santhoshsathish94@gmail.com").strip().lower()
EMAIL_RE = re.compile(r"^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$")
GITHUB_RE = re.compile(r"^[a-z0-9](?:[a-z0-9\-]{0,37}[a-z0-9])?$")


def _chmod_600(fp: Path) -> None:
    try:
        os.chmod(fp, 0o600)
    except OSError:
        pass


def _secret(base: Path) -> bytes:
    env = (os.environ.get("CLOVER_K3_KEY") or os.environ.get("CLOVER_HOST_KEY") or "").strip()
    if env:
        return env.encode("utf-8")
    kf = base / "vault" / "k3_host.key"
    if kf.is_file():
        raw = kf.read_text(encoding="utf-8").strip()
        if len(raw) >= 16:
            return raw.encode("utf-8")
    # Generate and persist a random 256-bit key in the vault if none exists yet (never fall back to a static string)
    target_kf = base / "vault" / "k3_host.key"
    target_kf.parent.mkdir(parents=True, exist_ok=True)
    new_key = secrets.token_hex(32)
    tmp = target_kf.with_suffix(".tmp")
    tmp.write_text(new_key + "\n", encoding="utf-8")
    _chmod_600(tmp)
    os.replace(tmp, target_kf)
    _chmod_600(target_kf)
    return new_key.encode("utf-8")


def make_account_token(account: str, base: Path = DEFAULT_BASE) -> str:
    acct = re.sub(r"[^a-z0-9._@-]", "", str(account or "").strip().lower().lstrip("@"))[:60]
    sig = hmac.new(_secret(base), f"clover-acct:{acct}".encode("utf-8"), hashlib.sha256).hexdigest()[:32]
    return f"acct:{acct}:{sig}"


def allowlist_path(base: Path = DEFAULT_BASE) -> Path:
    return base / "allowed-accounts.json"


def load_allowlist(base: Path = DEFAULT_BASE) -> dict:
    fp = allowlist_path(base)
    if fp.is_file():
        try:
            doc = json.loads(fp.read_text(encoding="utf-8"))
            if isinstance(doc, dict):
                doc.setdefault("accounts", [])
                doc.setdefault("users", [])
                doc.setdefault("keys", {})
                return doc
        except Exception:
            pass
    return {
        "accounts": [DEFAULT_OWNER_GITHUB, DEFAULT_OWNER_EMAIL],
        "users": [
            {
                "github": DEFAULT_OWNER_GITHUB,
                "email": DEFAULT_OWNER_EMAIL,
                "name": "Santhosh Narayanan",
                "role": "owner",
                "models": ["omni", "clover-35b-a3b", "nemotron-3.5-super-vl", "kimi-k3"],
            }
        ],
        "keys": {},
    }


def save_allowlist(doc: dict, base: Path = DEFAULT_BASE) -> Path:
    # Keep `accounts` flat list synchronized with `users`
    accts = set()
    for a in doc.get("accounts") or ():
        if isinstance(a, str) and a.strip():
            accts.add(a.strip().lower().lstrip("@"))
    for u in doc.get("users") or ():
        if isinstance(u, dict):
            for k in ("github", "email"):
                v = str(u.get(k) or "").strip().lower().lstrip("@")
                if v:
                    accts.add(v)
    doc["accounts"] = sorted(accts)
    fp = allowlist_path(base)
    fp.parent.mkdir(parents=True, exist_ok=True)
    tmp = fp.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    _chmod_600(tmp)
    os.replace(tmp, fp)
    _chmod_600(fp)
    # Also mirror into clover-ai-studio/config/allowed-accounts.json
    cfg = base / "clover-ai-studio" / "config" / "allowed-accounts.json"
    try:
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        _chmod_600(cfg)
    except OSError:
        pass
    return fp


def cmd_list(base: Path):
    doc = load_allowlist(base)
    save_allowlist(doc, base)
    print(f"Allowlist file: {allowlist_path(base)}")
    print(f"Total approved identifiers: {len(doc['accounts'])}\n")
    for idx, u in enumerate(doc.get("users") or (), 1):
        gh = u.get("github") or ""
        em = u.get("email") or ""
        nm = u.get("name") or ""
        role = u.get("role") or "user"
        primary = gh or em
        tok = make_account_token(primary, base) if primary else ""
        print(f"  [{idx}] {nm or primary} ({role})")
        if gh:
            print(f"      GitHub : {gh}  (token: {make_account_token(gh, base)})")
        if em:
            print(f"      Gmail  : {em}  (token: {make_account_token(em, base)})")
        if not gh and not em and tok:
            print(f"      Token  : {tok}")


def cmd_add(base: Path, email: str, github: str, name: str, role: str, key: str):
    doc = load_allowlist(base)
    em = (email or "").strip().lower()
    gh = (github or "").strip().lower().lstrip("@")
    if not em and not gh:
        raise ValueError("Provide an email address (e.g. user@gmail.com) and/or a GitHub username.")
    if em and not EMAIL_RE.match(em):
        raise ValueError(f"Invalid email address format: {em!r}")
    if gh and not GITHUB_RE.match(gh):
        raise ValueError(f"Invalid GitHub username format: {gh!r}")
    users = list(doc.get("users") or [])
    existing = None
    for u in users:
        if (em and u.get("email") == em) or (gh and u.get("github") == gh):
            existing = u
            break
    if existing is None:
        existing = {
            "github": gh,
            "email": em,
            "name": name or gh or em,
            "role": role or "user",
            "models": ["omni", "clover-35b-a3b", "nemotron-3.5-super-vl", "kimi-k3"],
        }
        users.append(existing)
    else:
        if gh:
            existing["github"] = gh
        if em:
            existing["email"] = em
        if name:
            existing["name"] = name
        if role:
            existing["role"] = role
    if key:
        kh = hashlib.sha256(key.strip().encode("utf-8")).hexdigest().lower()
        existing["key_sha256"] = kh
        doc.setdefault("keys", {})
        if gh:
            doc["keys"][gh] = kh
        if em:
            doc["keys"][em] = kh
    doc["users"] = users
    fp = save_allowlist(doc, base)
    primary = gh or em
    print(f"Added/updated user in {fp}:")
    if gh:
        print(f"  GitHub : {gh}")
        print(f"  Token  : {make_account_token(gh, base)}")
    if em:
        print(f"  Gmail  : {em}")
        print(f"  Token  : {make_account_token(em, base)}")


def cmd_remove(base: Path, identifier: str):
    ident = (identifier or "").strip().lower().lstrip("@")
    doc = load_allowlist(base)
    before = len(doc.get("users") or [])
    removed_ids = {ident}
    kept = []
    for u in doc.get("users") or ():
        if u.get("github") == ident or u.get("email") == ident:
            if u.get("github"):
                removed_ids.add(u["github"])
            if u.get("email"):
                removed_ids.add(u["email"])
        else:
            kept.append(u)
    doc["users"] = kept
    doc["accounts"] = [a for a in (doc.get("accounts") or ()) if a.lower().lstrip("@") not in removed_ids]
    for rid in removed_ids:
        (doc.get("keys") or {}).pop(rid, None)
    fp = save_allowlist(doc, base)
    print(f"Removed {ident} ({before - len(kept)} user record(s)) from {fp}")


def main():
    ap = argparse.ArgumentParser(description="Manage allowed k3 model users (GitHub & Gmail).")
    ap.add_argument("--base", default=str(DEFAULT_BASE))
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List all approved k3 users and their access tokens")

    p_add = sub.add_parser("add", help="Add or update an approved user by Gmail and/or GitHub username")
    p_add.add_argument("identifier", nargs="?", default="", help="Gmail address (user@gmail.com) or GitHub username")
    p_add.add_argument("--email", default="", help="Gmail / email address")
    p_add.add_argument("--github", default="", help="GitHub username")
    p_add.add_argument("--name", default="", help="Display name")
    p_add.add_argument("--role", default="user", help="Role (user or owner)")
    p_add.add_argument("--key", default="", help="Optional custom access key/passcode for this user")

    p_rm = sub.add_parser("remove", help="Remove a user by Gmail or GitHub username")
    p_rm.add_argument("identifier", help="Gmail address or GitHub username to remove")

    p_tok = sub.add_parser("token", help="Print the signed acct:<user>:<hmac> token for an approved user")
    p_tok.add_argument("identifier", help="Gmail address or GitHub username")

    args = ap.parse_args()
    base = Path(args.base)
    if args.cmd == "list":
        cmd_list(base)
    elif args.cmd == "add":
        em, gh = args.email, args.github
        if args.identifier:
            if "@" in args.identifier:
                em = em or args.identifier
            else:
                gh = gh or args.identifier
        cmd_add(base, em, gh, args.name, args.role, args.key)
    elif args.cmd == "remove":
        cmd_remove(base, args.identifier)
    elif args.cmd == "token":
        print(make_account_token(args.identifier, base))


if __name__ == "__main__":
    main()
