"""Collective Problem-Solving Learning — Git-Backed Core Knowledge Registry (`core_registry.py`).

Manages the versioned `clover-collective/` Git repository:
  clover-collective/
  ├── schema/
  │   ├── learning-event.schema.json
  │   └── core-knowledge.schema.json
  ├── candidates/
  │   └── reviewed-learning-packages/
  ├── core/
  │   ├── semantics/
  │   ├── reasoning-patterns/
  │   ├── problem-solving/
  │   └── tool-use-patterns/
  ├── evaluations/
  │   ├── regression/
  │   ├── generalisation/
  │   └── cross-domain/
  ├── releases/
  │   └── manifest.json
  └── CHANGELOG.md
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path

from collective_gateway import LEARNING_EVENT_SCHEMA

# --- RFC 8032 Ed25519 Asymmetric Signing (private key is NEVER embedded in client builds) ---
_Q = 2**255 - 19
_L = 2**252 + 27742317777372353535851937790883648493
_D = (-121665 * pow(121666, _Q - 2, _Q)) % _Q
_I = pow(2, (_Q - 1) // 4, _Q)


def _xrecover(y: int) -> int:
    xx = (y * y - 1) * pow(_D * y * y + 1, _Q - 2, _Q) % _Q
    x = pow(xx, (_Q + 3) // 8, _Q)
    if (x * x - xx) % _Q != 0:
        x = (x * _I) % _Q
    return _Q - x if x % 2 != 0 else x


_BY = 4 * pow(5, _Q - 2, _Q) % _Q
_BX = _xrecover(_BY)
_B_PT = (_BX % _Q, _BY % _Q, 1, (_BX * _BY) % _Q)
_IDENT = (0, 1, 1, 0)


def _edwards_add(P: tuple[int, int, int, int], Q: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, z1, t1 = P
    x2, y2, z2, t2 = Q
    a = (y1 - x1) * (y2 - x2) % _Q
    b = (y1 + x1) * (y2 + x2) % _Q
    c = t1 * 2 * _D * t2 % _Q
    d = z1 * 2 * z2 % _Q
    e = (b - a) % _Q
    f = (d - c) % _Q
    g = (d + c) % _Q
    h = (b + a) % _Q
    return (e * f % _Q, g * h % _Q, f * g % _Q, e * h % _Q)


def _scalarmult(P: tuple[int, int, int, int], e: int) -> tuple[int, int, int, int]:
    Q = _IDENT
    addend = P
    while e > 0:
        if e & 1:
            Q = _edwards_add(Q, addend)
        addend = _edwards_add(addend, addend)
        e >>= 1
    return Q


def _encodepoint(P: tuple[int, int, int, int]) -> bytes:
    x, y, z, _ = P
    zi = pow(z, _Q - 2, _Q)
    x = (x * zi) % _Q
    y = (y * zi) % _Q
    res = bytearray(y.to_bytes(32, "little"))
    res[31] |= (x & 1) << 7
    return bytes(res)


def _normalize_seed(signing_key: str | bytes) -> bytes:
    if isinstance(signing_key, bytes):
        if len(signing_key) == 32:
            return signing_key
        signing_key = signing_key.decode("utf-8").strip()
    s = str(signing_key).strip()
    if len(s) == 64:
        try:
            return bytes.fromhex(s)
        except ValueError:
            pass
    raise ValueError("Ed25519 signing_key must be a 32-byte raw seed or 64-character hex seed")


def ed25519_public_key(seed: bytes) -> bytes:
    """Derive 32-byte RFC 8032 Ed25519 public key from 32-byte private seed."""
    if len(seed) != 32:
        raise ValueError("Ed25519 private seed must be 32 bytes")
    h = bytearray(hashlib.sha512(seed).digest())
    h[0] &= 248
    h[31] &= 127
    h[31] |= 64
    a = int.from_bytes(h[:32], "little")
    return _encodepoint(_scalarmult(_B_PT, a))


def ed25519_sign(seed: bytes, msg: bytes) -> bytes:
    """Sign `msg` with 32-byte RFC 8032 Ed25519 private `seed` -> 64-byte signature."""
    if len(seed) != 32:
        raise ValueError("Ed25519 private seed must be 32 bytes")
    h = bytearray(hashlib.sha512(seed).digest())
    h[0] &= 248
    h[31] &= 127
    h[31] |= 64
    a = int.from_bytes(h[:32], "little")
    prefix = bytes(h[32:64])
    A = _encodepoint(_scalarmult(_B_PT, a))
    r = int.from_bytes(hashlib.sha512(prefix + msg).digest(), "little") % _L
    R = _encodepoint(_scalarmult(_B_PT, r))
    k = int.from_bytes(hashlib.sha512(R + A + msg).digest(), "little") % _L
    S = ((r + k * a) % _L).to_bytes(32, "little")
    return R + S


def generate_ed25519_keypair() -> tuple[str, str]:
    """Generate a fresh 32-byte Ed25519 `(private_seed_hex, public_key_hex)` pair."""
    seed = os.urandom(32)
    return seed.hex(), ed25519_public_key(seed).hex()


def _resolve_signing_seed(signing_key: str | bytes | None) -> bytes | None:
    if signing_key is not None:
        return _normalize_seed(signing_key)
    env_key = os.environ.get("CLOVER_CPSL_PRIVATE_KEY", "").strip()
    if env_key:
        return _normalize_seed(env_key)
    key_file = os.environ.get("CLOVER_CPSL_PRIVATE_KEY_FILE", "").strip()
    if not key_file and os.environ.get("CLOVER_CPSL_USE_SERVER_VAULT") == "1":
        key_file = "/opt/clover-k3/web-studio/vault/cpsl_release_ed25519.key"
    if key_file:
        vault_key = Path(key_file)
        if vault_key.is_file():
            try:
                return _normalize_seed(vault_key.read_text(encoding="utf-8").strip())
            except Exception:
                return None
    return None


CORE_KNOWLEDGE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "CloverCoreKnowledgeRule",
    "type": "object",
    "required": [
        "rule_id",
        "version",
        "category",
        "trigger",
        "target",
        "independent_installations",
        "heldout_accuracy",
        "heldout_lift",
        "regressions",
    ],
    "optional": ["strategy"],
    "additionalProperties": False,
}


def _writable(func, path, _exc):
    os.chmod(path, stat.S_IWRITE)
    func(path)


class CoreRegistry:
    """Manages approved collective knowledge in a Git repository and publishes Ed25519-signed releases.

    No private signing key is ever embedded in source code or client builds. Publishing a release requires an explicit
    Ed25519 private seed passed via `signing_key`, `CLOVER_CPSL_PRIVATE_KEY`, or the protected release server vault.
    """

    def __init__(self, repo_dir: str | Path, signing_key: str | bytes | None = None):
        self.root = Path(repo_dir)
        self._seed = _resolve_signing_seed(signing_key)
        self.public_key_hex = ed25519_public_key(self._seed).hex() if self._seed is not None else None
        self.git = shutil.which("git")
        self.version_num = 0
        self._init_repo()

    def _git(self, *args):
        if not self.git:
            return b""
        r = subprocess.run([self.git, "-C", str(self.root), "-c", "core.hooksPath=", *args], capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"git {args[0]}: {r.stderr.decode(errors='replace').strip()}")
        return r.stdout

    def _write_json(self, rel: str, obj):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8"))

    def _init_repo(self):
        for sub in [
            "schema",
            "candidates/reviewed-learning-packages",
            "core/semantics",
            "core/reasoning-patterns",
            "core/problem-solving",
            "core/tool-use-patterns",
            "core/jepa-growth",
            "evaluations/regression",
            "evaluations/generalisation",
            "evaluations/cross-domain",
            "releases",
            "releases/history",
        ]:
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self.revoked_versions: set[str] = set()
        manifest_file = self.root / "releases" / "manifest.json"
        if manifest_file.is_file():
            try:
                cur_m = json.loads(manifest_file.read_text(encoding="utf-8"))
                v_str = str(cur_m.get("version") or "")
                if v_str.startswith("1.0.") and v_str[4:].isdigit():
                    self.version_num = int(v_str[4:])
                self.revoked_versions.update(str(x) for x in (cur_m.get("revoked_versions") or ()))
            except Exception:
                pass

        if self.git and not (self.root / ".git").exists():
            self._git("init", "-q")
            for k, v in (("user.name", "CloverCoreRegistry"), ("user.email", "registry@cloverframework.local"),
                         ("commit.gpgsign", "false")):
                self._git("config", k, v)

        if not (self.root / "schema" / "learning-event.schema.json").exists():
            self._write_json("schema/learning-event.schema.json", LEARNING_EVENT_SCHEMA)
        if not (self.root / "schema" / "core-knowledge.schema.json").exists():
            self._write_json("schema/core-knowledge.schema.json", CORE_KNOWLEDGE_SCHEMA)
        if not (self.root / "CHANGELOG.md").exists():
            (self.root / "CHANGELOG.md").write_text("# Clover Collective Core — Changelog\n\n", encoding="utf-8")
        self._commit("Initialize clover-collective repository structure and schemas")

    def _commit(self, msg: str):
        if not self.git:
            return
        self._git("add", "-A")
        if self._git("status", "--porcelain").strip():
            self._git("commit", "-q", "-m", msg)

    def publish_release(
        self,
        eval_summary: dict,
        note: str = "Automated evaluation & promotion",
        *,
        expires_utc: str | None = None,
        revoked_versions: list[str] | set[str] | None = None,
        jepa_growth: dict | None = None,
    ) -> dict:
        self.version_num += 1
        version = f"1.0.{self.version_num}"
        if revoked_versions is not None:
            self.revoked_versions.update(str(x) for x in revoked_versions if x)

        for cat in ("semantics", "reasoning-patterns", "problem-solving", "tool-use-patterns"):
            cat_dir = self.root / "core" / cat
            for old_f in cat_dir.glob("*.json"):
                old_f.unlink()

        self._write_json(
            f"candidates/reviewed-learning-packages/review-v{version}.json",
            {
                "version": version,
                "total_evaluated": eval_summary["total_evaluated"],
                "promoted_count": eval_summary["promoted_count"],
                "rejected_count": eval_summary["rejected_count"],
                "candidates": eval_summary["all_reports"],
            },
        )
        self._write_json(
            f"evaluations/regression/regression-v{version}.json",
            [
                {"candidate_id": r["candidate_id"], "regressions": r["metrics"]["regressions"],
                 "personal_hijacks": r["metrics"]["personal_hijacks"], "pass": r["gates"]["G5_zero_regression"]}
                for r in eval_summary["all_reports"]
            ],
        )
        self._write_json(
            f"evaluations/generalisation/generalisation-v{version}.json",
            [
                {"candidate_id": r["candidate_id"], "heldout_accuracy": r["metrics"]["heldout_accuracy"],
                 "heldout_lift": r["metrics"]["heldout_lift"], "pass": r["gates"]["G3_generalisation"]}
                for r in eval_summary["all_reports"]
            ],
        )
        self._write_json(
            f"evaluations/cross-domain/cross-domain-v{version}.json",
            [
                {"candidate_id": r["candidate_id"], "matched_domains": r["metrics"]["matched_domains"],
                 "pass": r["gates"]["G4_cross_domain"]}
                for r in eval_summary["all_reports"]
            ],
        )

        promoted_rules = []
        file_hashes = {}
        files_content = {}
        for r in eval_summary["promoted"]:
            rule_id = f"core_{r['candidate_id'][4:]}"
            rule_obj = {
                "rule_id": rule_id,
                "version": version,
                "category": r["category"],
                "trigger": r["trigger"],
                "target": r["target"],
                "strategy": r.get("strategy", ""),
                "independent_installations": r["independent_installations"],
                "heldout_accuracy": r["metrics"]["heldout_accuracy"],
                "heldout_lift": r["metrics"]["heldout_lift"],
                "regressions": r["metrics"]["regressions"],
            }
            rel_path = f"core/{r['category']}/{rule_id}.json"
            self._write_json(rel_path, rule_obj)
            raw_bytes = (self.root / rel_path).read_bytes()
            file_hashes[rel_path] = hashlib.sha256(raw_bytes).hexdigest()
            files_content[rel_path] = raw_bytes.decode("utf-8")
            promoted_rules.append(rule_obj)

        jepa_rel = "core/jepa-growth/jepa_collective_growth.json"
        if isinstance(jepa_growth, dict) and jepa_growth.get("tensors"):
            self._write_json(jepa_rel, jepa_growth)
        if (self.root / jepa_rel).is_file():
            j_bytes = (self.root / jepa_rel).read_bytes()
            file_hashes[jepa_rel] = hashlib.sha256(j_bytes).hexdigest()
            files_content[jepa_rel] = j_bytes.decode("utf-8")

        if self._seed is None:
            raise RuntimeError(
                "CoreRegistry.publish_release requires an Ed25519 private signing key "
                "(pass signing_key=... or set CLOVER_CPSL_PRIVATE_KEY); client builds only ship the public key."
            )
        signed_obj = {"version": version, "rules": promoted_rules, "file_hashes": file_hashes}
        if expires_utc:
            signed_obj["expires_utc"] = str(expires_utc)
        if self.revoked_versions:
            signed_obj["revoked_versions"] = sorted(self.revoked_versions)
        payload_to_sign = json.dumps(signed_obj, sort_keys=True).encode("utf-8")
        signature = ed25519_sign(self._seed, payload_to_sign).hex()

        manifest = {
            "version": version,
            "published_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": note,
            "rules_count": len(promoted_rules),
            "rules": promoted_rules,
            "file_hashes": file_hashes,
            "signature_alg": "Ed25519",
            "public_key": self.public_key_hex,
            "signature": signature,
        }
        if expires_utc:
            manifest["expires_utc"] = str(expires_utc)
        if self.revoked_versions:
            manifest["revoked_versions"] = sorted(self.revoked_versions)

        self._write_json("releases/manifest.json", manifest)
        self._write_json(
            f"releases/history/v{version}.json",
            {"manifest": manifest, "files": files_content},
        )

        changelog = self.root / "CHANGELOG.md"
        prev_log = changelog.read_text(encoding="utf-8")
        entry = (
            f"## v{version} ({manifest['published_utc']})\n"
            f"- Note: {note}\n"
            f"- Evaluated candidates: {eval_summary['total_evaluated']} "
            f"(promoted: {eval_summary['promoted_count']}, rejected: {eval_summary['rejected_count']})\n\n"
        )
        changelog.write_text(prev_log + entry, encoding="utf-8")
        self._commit(f"Release v{version}: {len(promoted_rules)} validated collective rules ({note})")
        return manifest

    def revoke_version(self, version_to_revoke: str) -> dict:
        """Mark a release version as revoked and re-sign the current manifest."""
        if self._seed is None:
            raise RuntimeError("CoreRegistry.revoke_version requires an Ed25519 private signing key")
        self.revoked_versions.add(str(version_to_revoke))
        manifest_path = self.root / "releases" / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("No manifest.json exists to update revocation list")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["revoked_versions"] = sorted(self.revoked_versions)
        if manifest.get("version") == str(version_to_revoke):
            manifest["revoked"] = True
        signed_obj = {
            "version": manifest["version"],
            "rules": manifest.get("rules", []),
            "file_hashes": manifest.get("file_hashes", {}),
            "revoked_versions": manifest["revoked_versions"],
        }
        if manifest.get("expires_utc"):
            signed_obj["expires_utc"] = manifest["expires_utc"]
        if manifest.get("revoked"):
            signed_obj["revoked"] = True
        manifest["signature"] = ed25519_sign(self._seed, json.dumps(signed_obj, sort_keys=True).encode("utf-8")).hex()
        self._write_json("releases/manifest.json", manifest)
        self._commit(f"Revoke collective release v{version_to_revoke}")
        return manifest

    def get_release_bundle(self, since_version: str | None = None) -> dict:
        """Return a signed release distribution bundle supporting incremental synchronization from `since_version`."""
        manifest_path = self.root / "releases" / "manifest.json"
        if not manifest_path.is_file():
            return {"available": False, "version": None, "manifest": None, "files": {}, "removed_files": []}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cur_ver = manifest.get("version")
        revoked = set(manifest.get("revoked_versions") or ())
        if since_version and since_version == cur_ver and cur_ver not in revoked and not manifest.get("revoked"):
            return {
                "available": True,
                "up_to_date": True,
                "incremental": True,
                "version": cur_ver,
                "since_version": since_version,
                "manifest": manifest,
                "files": {},
                "removed_files": [],
            }

        prev_hashes = {}
        if since_version:
            hist_file = self.root / "releases" / "history" / f"v{since_version}.json"
            if hist_file.is_file():
                try:
                    prev_doc = json.loads(hist_file.read_text(encoding="utf-8"))
                    prev_hashes = dict((prev_doc.get("manifest") or {}).get("file_hashes") or {})
                except Exception:
                    prev_hashes = {}

        cur_hashes = dict(manifest.get("file_hashes") or {})
        changed_files = {}
        for rel_path, sha in cur_hashes.items():
            if prev_hashes.get(rel_path) == sha:
                continue
            fpath = self.root / rel_path
            if fpath.is_file():
                changed_files[rel_path] = fpath.read_text(encoding="utf-8")

        removed_files = sorted(set(prev_hashes.keys()) - set(cur_hashes.keys()))
        return {
            "available": True,
            "up_to_date": False,
            "incremental": bool(prev_hashes),
            "version": cur_ver,
            "since_version": since_version,
            "manifest": manifest,
            "files": changed_files,
            "removed_files": removed_files,
        }

    def find_strings_anywhere_in_repo(self, strings: list[str]) -> list[str]:
        blob = (
            subprocess.run(
                [self.git, "-C", str(self.root), "cat-file", "--batch-all-objects", "--batch"],
                capture_output=True,
            ).stdout
            if self.git
            else b""
        )
        for p in self.root.rglob("*"):
            if p.is_file() and p.relative_to(self.root).parts[:2] != (".git", "objects"):
                blob += p.read_bytes()
        return [s for s in strings if s.encode("utf-8") in blob]
