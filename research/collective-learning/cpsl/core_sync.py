"""`core_sync.py` — Client-Side Core Synchronizer & Layered Collective Assistant.

Responsibilities (`whats-needed-for-clover.txt` §§1, 5, 6):
1. Consent Check (`consent_receive`): Downloads and applies collective core releases ONLY when `consent_receive` is True.
2. Asymmetric Public-Key Verification: Verifies `releases/manifest.json` RFC 8032 Ed25519 signature against the pinned
   32-byte public verification key and per-file SHA-256 hashes before loading any collective rule. Clients hold ONLY
   the public verification key and can never manufacture or sign shared-core releases.
3. Strict Priority Hierarchy in `CollectiveAssistant`:
     1. Validated Personal Understanding (`self.learner`, PPL) — always takes precedence for that individual user!
     2. Signed Collective Core (`self.collective_rules`, CPSL) — applies general problem-solving strategies and semantic
        improvements when no personal override matches.
     3. Frozen General Interpreter (`self.general`, B0 + CSI v0.1r3) — immutable baseline fallback.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ppl.assistant import Assistant
from ppl.learner import trigger_features

# Official 32-byte RFC 8032 Ed25519 public verification key for Clover Collective Core releases.
# The corresponding private signing key is held exclusively in the protected release environment vault.
DEFAULT_VERIFY_PUBLIC_KEY = "dba3f7e2e969a26c57a24d50e2ecd654249b42e25eecc1f9140a514d7dcedb8a"

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


def _decodepoint(s: bytes) -> tuple[int, int, int, int]:
    if len(s) != 32:
        raise ValueError("Invalid Ed25519 point length")
    y = int.from_bytes(s, "little") & ((1 << 255) - 1)
    if y >= _Q:
        raise ValueError("Ed25519 point y out of range")
    x_sign = (s[31] >> 7) & 1
    u = (y * y - 1) % _Q
    v = (_D * y * y + 1) % _Q
    x = (u * pow(v, 3, _Q) * pow(u * pow(v, 7, _Q), (_Q - 5) // 8, _Q)) % _Q
    vxx = (v * x * x) % _Q
    if vxx == u:
        pass
    elif vxx == (-u) % _Q:
        x = (x * _I) % _Q
    else:
        raise ValueError("Point not on Ed25519 curve")
    if x == 0 and x_sign == 1:
        raise ValueError("Invalid Ed25519 x=0 with sign=1")
    if (x & 1) != x_sign:
        x = _Q - x
    return (x, y, 1, (x * y) % _Q)


def ed25519_verify(pub_bytes: bytes, msg: bytes, sig_bytes: bytes) -> bool:
    """Verify a 64-byte RFC 8032 Ed25519 signature against a 32-byte public key."""
    try:
        if len(pub_bytes) != 32 or len(sig_bytes) != 64:
            return False
        R_bytes, S_bytes = sig_bytes[:32], sig_bytes[32:]
        S = int.from_bytes(S_bytes, "little")
        if S >= _L:
            return False
        A = _decodepoint(pub_bytes)
        R = _decodepoint(R_bytes)
        k = int.from_bytes(hashlib.sha512(R_bytes + pub_bytes + msg).digest(), "little") % _L
        lhs = _scalarmult(_B_PT, S)
        rhs = _edwards_add(R, _scalarmult(A, k))
        return _encodepoint(lhs) == _encodepoint(rhs)
    except Exception:
        return False


def _normalize_public_key(verify_key: str | bytes | None) -> bytes:
    if verify_key is None:
        verify_key = os.environ.get("CLOVER_CPSL_PUBLIC_KEY", "").strip() or DEFAULT_VERIFY_PUBLIC_KEY
    if isinstance(verify_key, bytes):
        if len(verify_key) == 32:
            return verify_key
        verify_key = verify_key.decode("utf-8").strip()
    s = str(verify_key).strip()
    if len(s) == 64:
        try:
            return bytes.fromhex(s)
        except ValueError:
            pass
    raise ValueError("Ed25519 verify_key must be a 32-byte raw public key or 64-character hex public key")


class CoreSyncError(RuntimeError):
    pass


class CoreSync:
    """Downloads, verifies, and applies Ed25519-signed releases from `clover-collective` using a public key."""

    def __init__(self, verify_key: str | bytes | None = None):
        self.verify_key = _normalize_public_key(verify_key)
        self.verify_key_hex = self.verify_key.hex()

    def verify_and_load(self, registry_root: str | Path, consent_receive: bool = True) -> list[dict]:
        """Verify `releases/manifest.json` Ed25519 signature and file SHA-256 digests. Return `[]` if `consent_receive` is False."""
        if not consent_receive:
            return []
        root = Path(registry_root)
        manifest_path = root / "releases" / "manifest.json"
        if not manifest_path.exists():
            raise CoreSyncError(f"Missing manifest at {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        version = manifest.get("version")
        rules = manifest.get("rules", [])
        file_hashes = manifest.get("file_hashes", {})
        sig_alg = manifest.get("signature_alg", "")
        sig_hex = manifest.get("signature", "")
        manifest_pub = manifest.get("public_key", "")

        if sig_alg != "Ed25519":
            raise CoreSyncError(f"Unsupported or insecure signature_alg {sig_alg!r}; Ed25519 public-key signature required")
        if manifest_pub and manifest_pub.lower() != self.verify_key_hex.lower():
            raise CoreSyncError("Release public_key does not match trusted verification public key")
        try:
            sig_bytes = bytes.fromhex(sig_hex)
        except ValueError as exc:
            raise CoreSyncError("Malformed hex signature in manifest") from exc

        payload = json.dumps(
            {"version": version, "rules": rules, "file_hashes": file_hashes},
            sort_keys=True,
        ).encode("utf-8")
        if not ed25519_verify(self.verify_key, payload, sig_bytes):
            raise CoreSyncError("Release signature verification failed (invalid Ed25519 signature or tampered manifest)")

        for rel_path, expected_sha in file_hashes.items():
            fpath = root / rel_path
            if not fpath.exists():
                raise CoreSyncError(f"Missing core file {rel_path}")
            actual_sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
            if actual_sha != expected_sha:
                raise CoreSyncError(f"File SHA-256 mismatch for {rel_path}")

        return list(rules)


class CollectiveAssistant(Assistant):
    """Clover Assistant combining Personal Process Learning (PPL), Signed Collective Core (CPSL), and Frozen General Core (B0)."""

    def __init__(self, arm: str, general, reader, params: dict, collective_rules: list[dict] | None = None,
                 store=None, personal_csi: bool = True):
        base_arm = "PPL" if arm in ("C1", "C2", "C3", "CPSL", "CPSL-NoCSI") else "B0"
        super().__init__(base_arm, general, reader, params, store=store)
        if arm == "C0":
            self.learner = None
        if not personal_csi:
            self.personal_csi = False
        self.collective_rules = []
        for r in collective_rules or []:
            targ = r["target"]
            target_val = tuple(targ) if isinstance(targ, (list, tuple)) and len(targ) > 1 else (
                targ[0] if isinstance(targ, (list, tuple)) else targ
            )
            self.collective_rules.append({
                "rule_id": r["rule_id"],
                "trigger": frozenset(r["trigger"]),
                "target": target_val,
                "independent_installations": r.get("independent_installations", 1),
                "heldout_accuracy": r.get("heldout_accuracy", 1.0),
            })

    def match_collective(self, feats: frozenset[str]):
        tf = trigger_features(feats, use_ctx=True)
        matches = [r for r in self.collective_rules if r["trigger"] <= tf]
        if not matches:
            return None
        return max(
            matches,
            key=lambda r: (len(r["trigger"]), r["heldout_accuracy"], r["independent_installations"], r["rule_id"]),
        )

    def respond(self, text: str, context: str):
        a = self.analyse(text, context)
        top, conf = a["scores"][0]
        # 1. Personal understanding (PPL) has highest priority for this person
        if self.learner:
            h = self.learner.interpret(a["p"])
            if h is not None:
                return self.act(h.target, "personal", h.id), a
        # 2. Signed Collective Core (CPSL) applies validated collective strategies
        c_rule = self.match_collective(a["p"])
        if c_rule is not None:
            return self.act(c_rule["target"], "collective", c_rule["rule_id"]), a
        # 3. Frozen General Interpreter (B0)
        if conf < self.C:
            personal = []
            if self.learner:
                personal = [
                    h.target
                    for h in sorted(self.learner.candidates(a["p"]), key=lambda h: (-len(h.support), h.id))
                ]
            return self.clarify(personal + [l for l, _ in a["scores"][:3]]), a
        return self.act(top, "general"), a
