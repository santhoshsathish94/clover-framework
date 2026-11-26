"""Collective Problem-Solving Learning — Client Core Synchronizer & Runtime Hook (`core_sync.py`).

Manages:
1. Two independent consent controls (`consent_contribute`, `consent_receive`), persisted locally in
   `<collective_dir>/consent.json` (both default to `False` — strictly opt-in).
2. Asymmetric public-key verification of signed `clover-collective` releases (`releases/manifest.json` RFC 8032 Ed25519
   signature against the pinned 32-byte public verification key + per-file SHA-256 digests). Clients hold ONLY the
   public verification key and can never manufacture or sign shared-core releases.
3. Runtime interpretation (`Collective.interpret(text, frame)`), hooked into `semantics.Runtime.collective` and called
   ONLY when no validated personal habit overrides the message (`Personal > Collective > General`).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from pathlib import Path

import collective as C_EXT
import personal
import semantics

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
    """Verifies and loads Ed25519-signed releases from a `clover-collective` release directory using a public key."""

    def __init__(self, verify_key: str | bytes | None = None):
        self.verify_key = _normalize_public_key(verify_key)
        self.verify_key_hex = self.verify_key.hex()

    def verify_and_load(self, registry_root: str | Path, consent_receive: bool = True) -> dict:
        if not consent_receive:
            return {"version": None, "rules": []}
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

        return {"version": version, "rules": list(rules)}


class Collective:
    """One installation's collective learning state: consent switches, verified shared core rules, and candidate preview."""

    def __init__(
        self,
        root: str | Path,
        device: str = "",
        csi=None,
        verify_key: str | bytes | None = None,
    ):
        self.dir = Path(root)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.csi = csi
        self.syncer = CoreSync(verify_key=verify_key)
        self.lock = threading.RLock()
        self.version: str | None = None
        self.rules: list[dict] = []
        self.normalize_gerunds: bool = True
        self._load_state()

    @property
    def _consent_path(self) -> Path:
        return self.dir / "consent.json"

    @property
    def _applied_path(self) -> Path:
        return self.dir / "applied_release.json"

    def _load_state(self):
        if self._consent_path.exists():
            try:
                data = json.loads(self._consent_path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        else:
            data = {}
        self.consent_contribute = bool(data.get("consent_contribute", False))
        self.consent_receive = bool(data.get("consent_receive", False))
        self.installation_id = str(data.get("installation_id") or os.urandom(8).hex())
        self._save_consent()

        if self.consent_receive and self._applied_path.exists():
            try:
                applied = json.loads(self._applied_path.read_text(encoding="utf-8"))
                self.version = applied.get("version")
                self.rules = list(applied.get("rules") or [])
            except Exception:
                self.version, self.rules = None, []

    def _save_consent(self):
        self._consent_path.write_text(
            json.dumps(
                {
                    "installation_id": self.installation_id,
                    "consent_contribute": self.consent_contribute,
                    "consent_receive": self.consent_receive,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def set_consent(self, contribute: bool | None = None, receive: bool | None = None) -> dict:
        with self.lock:
            if contribute is not None:
                self.consent_contribute = bool(contribute)
            if receive is not None:
                self.consent_receive = bool(receive)
                if not self.consent_receive:
                    self.rules = []
                    self.version = None
                elif self._applied_path.exists():
                    applied = json.loads(self._applied_path.read_text(encoding="utf-8"))
                    self.version = applied.get("version")
                    self.rules = list(applied.get("rules") or [])
            self._save_consent()
            return {"consent_contribute": self.consent_contribute, "consent_receive": self.consent_receive}

    def sync_from_registry(self, registry_root: str | Path) -> dict:
        """Verify and apply a signed release from `registry_root` when `consent_receive` is True."""
        with self.lock:
            res = self.syncer.verify_and_load(registry_root, consent_receive=self.consent_receive)
            self.version = res["version"]
            self.rules = res["rules"]
            if self.consent_receive and self.version is not None:
                self._applied_path.write_text(json.dumps(res, indent=2, sort_keys=True), encoding="utf-8")
            return {"synced": self.consent_receive, "version": self.version, "rules_count": len(self.rules)}

    def preview_candidates(self, personal_obj, extraction_mode: str = "mismatch_only") -> list[dict]:
        with self.lock:
            if personal_obj is None:
                return []
            ext = C_EXT.LocalCollectiveExtractor(
                installation_id=self.installation_id,
                consent_contribute=self.consent_contribute,
                extraction_mode=extraction_mode,
            )
            return ext.preview_candidates(personal_obj)

    def export_candidates(self, personal_obj, extraction_mode: str = "mismatch_only") -> list[dict]:
        with self.lock:
            if not self.consent_contribute or personal_obj is None:
                return []
            ext = C_EXT.LocalCollectiveExtractor(
                installation_id=self.installation_id,
                consent_contribute=True,
                extraction_mode=extraction_mode,
            )
            return ext.export_candidates(personal_obj)

    def _features(self, text: str, analysis=None) -> frozenset[str]:
        if analysis is None and self.csi is not None:
            try:
                analysis = self.csi(text)
            except Exception:
                analysis = None
        raw = personal.features(text, analysis, self.device)
        return C_EXT.normalize_gerund_features(raw) if self.normalize_gerunds else raw

    def interpret(self, text: str, frame) -> dict | None:
        """Called by `semantics._collective(frame, runtime)` when no personal hypothesis overrides the message."""
        with self.lock:
            if not self.consent_receive or not self.rules:
                return None
            feats = self._features(text, analysis=getattr(frame, "_csi", None))
            tf = personal.trigger_features(feats, use_ctx=True)
            matches = []
            for r in self.rules:
                trig = frozenset(r["trigger"])
                trig_no_ctx = frozenset(f for f in trig if not f.startswith("ctx="))
                r_ctxs = {f for f in trig if f.startswith("ctx=")}
                f_ctxs = {f for f in tf if f.startswith("ctx=")}
                if trig <= tf or (not r_ctxs and trig_no_ctx and trig_no_ctx <= tf) or (
                    bool(r_ctxs & f_ctxs) and trig_no_ctx <= tf
                ):
                    matches.append(r)
            if not matches:
                return None
            best = max(
                matches,
                key=lambda r: (
                    len(r["trigger"]),
                    float(r.get("heldout_accuracy", 1.0)),
                    int(r.get("independent_installations", 1)),
                    r["rule_id"],
                ),
            )
            target_caps = list(best.get("target") or [])
            if sorted(target_caps) == sorted(frame.caps) and not best.get("strategy"):
                return None
            out = {
                "rule_id": best["rule_id"],
                "version": best.get("version") or self.version,
                "category": best.get("category", "semantics"),
                "caps": target_caps,
                "independent_installations": int(best.get("independent_installations", 1)),
            }
            if best.get("strategy"):
                out["strategy"] = best["strategy"]
            return out

    def view(self, personal_obj=None) -> dict:
        with self.lock:
            return {
                "enabled": True,
                "store": str(self.dir),
                "consent_contribute": self.consent_contribute,
                "consent_receive": self.consent_receive,
                "version": self.version,
                "active_rules_count": len(self.rules),
                "rules": list(self.rules),
                "candidate_preview": self.preview_candidates(personal_obj) if personal_obj is not None else [],
            }


_PROCESS_COLL = {}
_PROCESS_COLL_LOCK = threading.Lock()
KEEP = object()


def for_process(root, device: str = "", csi=KEEP):
    if not root or os.environ.get("CLOVER_COLLECTIVE", "").strip().lower() in ("0", "off", "false", "no"):
        return None
    key = os.path.abspath(root)
    with _PROCESS_COLL_LOCK:
        if key not in _PROCESS_COLL:
            try:
                _PROCESS_COLL[key] = Collective(root, device=device, csi=None if csi is KEEP else csi)
            except (OSError, ValueError, RuntimeError) as ex:
                print(f"[Clover] collective learning stays off: {ex}", file=sys.stderr, flush=True)
                return None
        elif csi is not KEEP:
            _PROCESS_COLL[key].csi = csi
        return _PROCESS_COLL[key]
