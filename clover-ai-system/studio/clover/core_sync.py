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
import secrets
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import collective as C_EXT
import personal
import semantics

# Official 32-byte RFC 8032 Ed25519 public verification key for Clover Collective Core releases.
# The corresponding private signing key is held exclusively in the protected release environment vault.
DEFAULT_VERIFY_PUBLIC_KEY = "dba3f7e2e969a26c57a24d50e2ecd654249b42e25eecc1f9140a514d7dcedb8a"
DEFAULT_GATEWAY_URL = "https://api.cloverframework.com"

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

    def verify_and_load(
        self,
        registry_root: str | Path,
        consent_receive: bool = True,
        now_utc: str | None = None,
    ) -> dict:
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

        signed_obj = {"version": version, "rules": rules, "file_hashes": file_hashes}
        if "expires_utc" in manifest and manifest["expires_utc"]:
            signed_obj["expires_utc"] = manifest["expires_utc"]
        if "revoked_versions" in manifest and manifest["revoked_versions"]:
            signed_obj["revoked_versions"] = manifest["revoked_versions"]
        if manifest.get("revoked"):
            signed_obj["revoked"] = True
        payload = json.dumps(signed_obj, sort_keys=True).encode("utf-8")
        base_payload = json.dumps(
            {"version": version, "rules": rules, "file_hashes": file_hashes},
            sort_keys=True,
        ).encode("utf-8")
        if not ed25519_verify(self.verify_key, payload, sig_bytes) and not ed25519_verify(
            self.verify_key, base_payload, sig_bytes
        ):
            raise CoreSyncError("Release signature verification failed (invalid Ed25519 signature or tampered manifest)")

        # Reject revoked releases
        revoked_set = {str(x) for x in (manifest.get("revoked_versions") or ())}
        if manifest.get("revoked") is True or (version and str(version) in revoked_set):
            raise CoreSyncError(f"Release version {version!r} has been revoked")

        # Reject expired releases
        expires_utc = manifest.get("expires_utc")
        if expires_utc:
            now_iso = str(now_utc or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            if str(expires_utc) <= now_iso:
                raise CoreSyncError(f"Release version {version!r} expired at {expires_utc}")

        for rel_path, expected_sha in file_hashes.items():
            fpath = root / rel_path
            if not fpath.exists():
                raise CoreSyncError(f"Missing core file {rel_path}")
            actual_sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
            if actual_sha != expected_sha:
                raise CoreSyncError(f"File SHA-256 mismatch for {rel_path}")

        jepa_growth = None
        jepa_path = root / "core" / "jepa-growth" / "jepa_collective_growth.json"
        if "core/jepa-growth/jepa_collective_growth.json" in file_hashes and jepa_path.is_file():
            try:
                jepa_growth = json.loads(jepa_path.read_text(encoding="utf-8"))
            except Exception:
                jepa_growth = None

        out = {"version": version, "rules": list(rules)}
        if isinstance(jepa_growth, dict):
            out["jepa_growth"] = jepa_growth
        return out


class Collective:
    """One installation's collective learning state: consent switches, verified shared core rules, candidate preview,
    and remote gateway synchronization."""

    def __init__(
        self,
        root: str | Path,
        device: str = "",
        csi=None,
        verify_key: str | bytes | None = None,
        gateway_url: str | None = None,
    ):
        self.dir = Path(root)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.csi = csi
        self.syncer = CoreSync(verify_key=verify_key)
        self.gateway_url = (
            gateway_url
            or os.environ.get("CLOVER_COLLECTIVE_GATEWAY_URL", "").strip()
            or DEFAULT_GATEWAY_URL
        ).rstrip("/")
        self.lock = threading.RLock()
        self.version: str | None = None
        self.rules: list[dict] = []
        self.normalize_gerunds: bool = True
        self.submitted_packages: list[dict] = []
        self.last_sync_utc: str | None = None
        self.last_sync_error: str = ""
        self.consent_withdrawn_utc: str | None = None
        self._load_state()

    @property
    def _consent_path(self) -> Path:
        return self.dir / "consent.json"

    @property
    def _applied_path(self) -> Path:
        return self.dir / "applied_release.json"

    @property
    def _sync_state_path(self) -> Path:
        return self.dir / "sync_state.json"

    @property
    def _mirror_dir(self) -> Path:
        return self.dir / "remote_mirror"

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

        if self._sync_state_path.exists():
            try:
                sdata = json.loads(self._sync_state_path.read_text(encoding="utf-8"))
                self.submitted_packages = list(sdata.get("submitted_packages") or [])
                self.last_sync_utc = sdata.get("last_sync_utc")
                self.last_sync_error = str(sdata.get("last_sync_error") or "")
                self.consent_withdrawn_utc = sdata.get("consent_withdrawn_utc")
                if sdata.get("gateway_url"):
                    self.gateway_url = str(sdata["gateway_url"]).rstrip("/")
            except Exception:
                pass

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

    def _save_sync_state(self):
        self._sync_state_path.write_text(
            json.dumps(
                {
                    "gateway_url": self.gateway_url,
                    "submitted_packages": self.submitted_packages,
                    "last_sync_utc": self.last_sync_utc,
                    "last_sync_error": self.last_sync_error,
                    "consent_withdrawn_utc": self.consent_withdrawn_utc,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def set_consent(
        self,
        contribute: bool | None = None,
        receive: bool | None = None,
        *,
        identity_connected: bool | None = None,
        google_connected: bool | None = None,
        require_identity: bool = False,
    ) -> dict:
        connected = identity_connected if identity_connected is not None else google_connected
        with self.lock:
            if require_identity and (contribute is True or receive is True) and not connected:
                raise CoreSyncError("Sign in with GitHub first to enable collective learning.")
            if contribute is not None:
                self.consent_contribute = bool(contribute)
                if self.consent_contribute:
                    self.consent_withdrawn_utc = None
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
            self._save_sync_state()
            return {"consent_contribute": self.consent_contribute, "consent_receive": self.consent_receive}

    def sync_from_registry(self, registry_root: str | Path, now_utc: str | None = None) -> dict:
        """Verify and apply a signed release from `registry_root` when `consent_receive` is True."""
        with self.lock:
            try:
                res = self.syncer.verify_and_load(
                    registry_root, consent_receive=self.consent_receive, now_utc=now_utc
                )
            except CoreSyncError as exc:
                self.last_sync_error = str(exc)
                self._save_sync_state()
                raise
            self.version = res["version"]
            self.rules = res["rules"]
            if self.consent_receive and self.version is not None:
                self._applied_path.write_text(json.dumps(res, indent=2, sort_keys=True), encoding="utf-8")
                if isinstance(res.get("jepa_growth"), dict):
                    jepa_dst = self.dir / "jepa_collective_growth.json"
                    jepa_dst.write_text(json.dumps(res["jepa_growth"], indent=2, sort_keys=True), encoding="utf-8")
                    try:
                        import jepa_engine
                        j_rt = jepa_engine.for_process(self.dir.parent)
                        if j_rt is not None and hasattr(j_rt, "apply_collective_growth"):
                            j_rt.apply_collective_growth(res["jepa_growth"])
                    except Exception:
                        pass
                self.last_sync_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                self.last_sync_error = ""
                self._save_sync_state()
            return {
                "synced": self.consent_receive,
                "version": self.version,
                "rules_count": len(self.rules),
                "jepa_growth_synced": bool(self.consent_receive and isinstance(res.get("jepa_growth"), dict)),
            }

    def _http_call(
        self,
        method: str,
        url: str,
        *,
        id_token: str | None = None,
        body: dict | None = None,
        transport_fn=None,
    ) -> dict:
        if transport_fn is not None:
            return transport_fn(method, url, id_token=id_token, body=body)
        headers = {"Accept": "application/json", "User-Agent": "CloverAIStudio/5.0"}
        if id_token:
            headers["Authorization"] = f"Bearer {id_token}"
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json; charset=utf-8"
            data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw_err = exc.read().decode("utf-8", errors="replace")
            try:
                err_obj = json.loads(raw_err)
                msg = err_obj.get("error") or err_obj.get("message") or raw_err[:240]
            except Exception:
                msg = raw_err[:240] or f"HTTP {exc.code}"
            raise CoreSyncError(msg) from exc
        except Exception as exc:
            raise CoreSyncError(f"Gateway request failed: {exc}") from exc

    def gateway_session(self, *, github_token: str, gateway_url: str | None = None, transport_fn=None) -> dict:
        """The gateway checks a GitHub sign-in once and answers with its own credential for this contributor
        (`cg1.`), used for contributions, releases and withdrawal until it expires."""
        if not github_token or not str(github_token).strip():
            raise CoreSyncError("GitHub sign-in is required for collective learning.")
        gw = (gateway_url or self.gateway_url).rstrip("/")
        resp = self._http_call("POST", f"{gw}/api/collective/gateway/session", id_token=str(github_token).strip(),
                               body={}, transport_fn=transport_fn)
        if not str(resp.get("credential") or "").startswith("cg1."):
            raise CoreSyncError("The collective gateway did not issue a credential.")
        return resp

    def contribute_to_gateway(
        self,
        *,
        id_token: str | None,
        packages: list[dict],
        confirmed: bool = False,
        gateway_url: str | None = None,
        transport_fn=None,
    ) -> dict:
        """Upload explicitly confirmed, privacy-scrubbed candidate packages to the remote Collective Gateway."""
        with self.lock:
            if not self.consent_contribute:
                err = "Contribution consent is disabled. Enable 'Contribute to Collective Learning' first."
                self.last_sync_error = err
                self._save_sync_state()
                raise CoreSyncError(err)
            if not id_token or not str(id_token).strip():
                err = "GitHub sign-in is required to contribute candidate knowledge."
                self.last_sync_error = err
                self._save_sync_state()
                raise CoreSyncError(err)
            if not confirmed:
                err = "Candidate upload requires explicit user confirmation after reviewing the package."
                self.last_sync_error = err
                self._save_sync_state()
                raise CoreSyncError(err)
            if not isinstance(packages, list) or not packages:
                raise CoreSyncError("No candidate packages selected for contribution.")

            # Validate every candidate locally before transmission so unapproved/personal data can never leave
            import collective_gateway
            local_validator = collective_gateway.CollectiveGateway()
            clean_packages = []
            for p in packages:
                if not isinstance(p, dict):
                    raise CoreSyncError("Invalid candidate package structure")
                ok, reason = local_validator.validate_package(p, consent_verified=True)
                if not ok:
                    err = f"Candidate {p.get('package_id')!r} blocked by local privacy/schema gate: {reason}"
                    self.last_sync_error = err
                    self._save_sync_state()
                    raise CoreSyncError(err)
                clean_packages.append(p)

            gw = (gateway_url or self.gateway_url).rstrip("/")
            payload = {
                "confirmed": True,
                "consent_contribute": True,
                "nonce": secrets.token_urlsafe(18),
                "timestamp": time.time(),
                "packages": clean_packages,
            }
            try:
                resp = self._http_call(
                    "POST",
                    f"{gw}/api/collective/gateway/contribute",
                    id_token=id_token,
                    body=payload,
                    transport_fn=transport_fn,
                )
            except CoreSyncError as exc:
                self.last_sync_error = str(exc)
                self._save_sync_state()
                raise

            now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            accepted_set = set(resp.get("accepted_package_ids") or ())
            existing_ids = {x["package_id"] for x in self.submitted_packages}
            for p in clean_packages:
                if p["package_id"] in accepted_set and p["package_id"] not in existing_ids:
                    self.submitted_packages.append({
                        "package_id": p["package_id"],
                        "category": p["category"],
                        "trigger": list(p["trigger"]),
                        "target": list(p["target"]),
                        "submitted_utc": now_iso,
                    })
            self.last_sync_utc = now_iso
            self.last_sync_error = ""
            self._save_sync_state()
            return resp

    def sync_from_gateway(
        self,
        *,
        id_token: str | None = None,
        require_identity: bool = True,
        gateway_url: str | None = None,
        transport_fn=None,
        now_utc: str | None = None,
    ) -> dict:
        """Fetch an incremental or full Ed25519-signed collective release from the remote gateway, verify its
        cryptographic signature, file hashes, expiry, and revocation status, and apply it locally."""
        with self.lock:
            if not self.consent_receive:
                err = "Receive consent is disabled. Enable 'Receive Collective Learning' first."
                self.last_sync_error = err
                self._save_sync_state()
                raise CoreSyncError(err)
            if require_identity and (not id_token or not str(id_token).strip()):
                err = "GitHub sign-in is required to receive collective learning releases."
                self.last_sync_error = err
                self._save_sync_state()
                raise CoreSyncError(err)

            gw = (gateway_url or self.gateway_url).rstrip("/")
            mirror = self._mirror_dir
            since = self.version if (self.version and (mirror / "releases" / "manifest.json").is_file()) else ""
            q = f"?{urllib.parse.urlencode({'since_version': since})}" if since else ""
            try:
                bundle = self._http_call(
                    "GET",
                    f"{gw}/api/collective/gateway/releases{q}",
                    id_token=id_token,
                    transport_fn=transport_fn,
                )
            except CoreSyncError as exc:
                self.last_sync_error = str(exc)
                self._save_sync_state()
                raise

            if not bundle.get("available") or not bundle.get("manifest"):
                self.last_sync_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                self.last_sync_error = ""
                self._save_sync_state()
                return {"synced": True, "up_to_date": True, "version": self.version, "rules_count": len(self.rules)}

            # Stage files in a temporary directory or mirror, then verify before activating
            (mirror / "releases").mkdir(parents=True, exist_ok=True)
            for rel_rm in bundle.get("removed_files") or []:
                rm_path = (mirror / rel_rm).resolve()
                if str(rm_path).startswith(str(mirror.resolve())) and rm_path.is_file():
                    rm_path.unlink()
            for rel_path, content in (bundle.get("files") or {}).items():
                dest = (mirror / rel_path).resolve()
                if not str(dest).startswith(str(mirror.resolve())):
                    raise CoreSyncError(f"Unsafe file path in release bundle: {rel_path!r}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(str(content).encode("utf-8"))
            (mirror / "releases" / "manifest.json").write_bytes(
                json.dumps(bundle["manifest"], indent=2, sort_keys=True).encode("utf-8"),
            )

            res = self.sync_from_registry(mirror, now_utc=now_utc)
            res["up_to_date"] = bool(bundle.get("up_to_date"))
            res["incremental"] = bool(bundle.get("incremental"))
            return res

    def withdraw_from_gateway(
        self,
        *,
        id_token: str | None,
        gateway_url: str | None = None,
        transport_fn=None,
    ) -> dict:
        """Withdraw future contribution consent and remove previously contributed candidates on the remote gateway."""
        with self.lock:
            now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            gw_resp = {}
            if id_token and str(id_token).strip():
                gw = (gateway_url or self.gateway_url).rstrip("/")
                payload = {
                    "nonce": secrets.token_urlsafe(18),
                    "timestamp": time.time(),
                }
                gw_resp = self._http_call(
                    "POST",
                    f"{gw}/api/collective/gateway/withdraw",
                    id_token=id_token,
                    body=payload,
                    transport_fn=transport_fn,
                )
            self.consent_contribute = False
            self.submitted_packages = []
            self.consent_withdrawn_utc = now_iso
            self.last_sync_error = ""
            self._save_consent()
            self._save_sync_state()
            return {
                "ok": True,
                "consent_contribute": False,
                "consent_withdrawn_utc": now_iso,
                "gateway": gw_resp,
            }

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

    def view(self, personal_obj=None, google_status: dict | None = None, identity: dict | None = None) -> dict:
        with self.lock:
            cands = self.preview_candidates(personal_obj) if personal_obj is not None else []
            submitted_ids = {x["package_id"] for x in self.submitted_packages}
            pending_cands = [c for c in cands if c.get("package_id") not in submitted_ids]
            return {
                "enabled": True,
                "store": str(self.dir),
                "gateway_url": self.gateway_url,
                "identity": identity or google_status or {"connected": False, "status_text": "Not signed in"},
                "consent_contribute": self.consent_contribute,
                "consent_receive": self.consent_receive,
                "consent_withdrawn_utc": self.consent_withdrawn_utc,
                "version": self.version,
                "active_rules_count": len(self.rules),
                "rules": list(self.rules),
                "active_rules": [
                    {"rule_id": r.get("rule_id"), "kind": r.get("category", "semantics"),
                     "key": r.get("trigger", []), "target": r.get("target", []), "strategy": r.get("strategy", "")}
                    for r in self.rules
                ],
                "active_defaults": [],
                "candidate_preview": cands,
                "pending_candidates": pending_cands,
                "submitted_packages": list(self.submitted_packages),
                "submitted_count": len(self.submitted_packages),
                "last_sync_utc": self.last_sync_utc,
                "last_sync_error": self.last_sync_error,
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
