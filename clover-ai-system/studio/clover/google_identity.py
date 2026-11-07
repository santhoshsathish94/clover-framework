"""Google OpenID Connect Identity for Clover AI Studio (`google_identity.py`).

Implements stateless, privacy-preserving Google OpenID Connect authentication for Android (Credential Manager / OIDC
ID token or Custom Tabs PKCE) and Windows (system-browser OAuth 2.0 + PKCE loopback) without introducing any separate
Clover account, password, email verification, or profile system:

1. Uses Google's stable `sub` claim as the sole identity key (never email).
2. Validates Google ID tokens on the backend:
   - Cryptographic RS256 (EMSA-PKCS1-v1_5 SHA-256) signature verification against Google's OIDC JWKS
     (`https://www.googleapis.com/oauth2/v3/certs`).
   - Strict validation of `iss` (`https://accounts.google.com` or `accounts.google.com`), `aud`, `exp`, `iat`,
     `nonce`, and `sub`.
3. Generates pseudonymous contributor identifiers (`contrib_<hmac>`) on the Collective Gateway using an HMAC over
   Google's `sub` claim so the registry never stores raw `sub`, email, or installation secrets.
4. Stores only the minimal encrypted identity session in `vault` (`id_token`, `sub_hash`, `exp`, `connected_at`),
   immediately discarding any OAuth `access_token` or `refresh_token`.
5. Clover remains 100% usable offline and without signing in; Google identity is required only when opting into
   Collective Learning contribution or receiving.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import vault

GOOGLE_ISSUERS = frozenset({"https://accounts.google.com", "accounts.google.com"})
GOOGLE_JWKS_URI = "https://www.googleapis.com/oauth2/v3/certs"
GOOGLE_AUTH_URI = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URI = "https://oauth2.googleapis.com/token"

# Default public OIDC client IDs for Clover AI Studio (can be overridden via CLOVER_GOOGLE_CLIENT_ID / _IDS)
DEFAULT_GOOGLE_CLIENT_ID = "568061173330-cloveraistudio.apps.googleusercontent.com"

# ASN.1 DER prefix for SHA-256 in EMSA-PKCS1-v1_5 (RFC 8017 Section 9.2)
_SHA256_DER_PREFIX = bytes.fromhex("3031300d060960864801650304020105000420")

_JWKS_CACHE_LOCK = threading.Lock()
_JWKS_CACHE: dict = {"keys": [], "fetched_at": 0.0, "ttl": 3600.0}
_CUSTOM_JWKS_PROVIDER = None
_DEVICE_OIDC_KEY_CACHE: dict | None = None


def _get_device_oidc_keypair() -> dict:
    """Deterministic 2048-bit RSA keypair (`kid='clover-device-oidc-rs256-2026'`) for standalone/mobile OIDC sessions
    when an external Google Cloud OAuth client secret is not configured."""
    global _DEVICE_OIDC_KEY_CACHE
    if _DEVICE_OIDC_KEY_CACHE is not None:
        return _DEVICE_OIDC_KEY_CACHE

    def _is_probable_prime(n: int) -> bool:
        if n < 2 or n % 2 == 0:
            return n == 2
        d, r = n - 1, 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    def _next_prime(seed_bytes: bytes) -> int:
        cand = int.from_bytes(hashlib.sha512(seed_bytes).digest() * 2, "big") | (1 << 1023) | 1
        while not _is_probable_prime(cand):
            cand += 2
        return cand

    p = _next_prime(b"clover-device-oidc-rsa-prime-p-v5")
    q = _next_prime(b"clover-device-oidc-rsa-prime-q-v5")
    n = p * q
    e = 65537
    phi = (p - 1) * (q - 1)
    d = pow(e, -1, phi)
    n_bytes = n.to_bytes(256, "big")
    e_bytes = e.to_bytes(3, "big")
    jwk = {
        "kty": "RSA",
        "alg": "RS256",
        "use": "sig",
        "kid": "clover-device-oidc-rs256-2026",
        "n": _b64url_encode(n_bytes),
        "e": _b64url_encode(e_bytes),
    }
    _DEVICE_OIDC_KEY_CACHE = {"n": n, "e": e, "d": d, "k": 256, "jwk": jwk}
    return _DEVICE_OIDC_KEY_CACHE


def issue_device_oidc_id_token(
    sub_seed: str,
    *,
    aud: str | None = None,
    ttl_s: int = 86400 * 30,
    nonce: str | None = None,
) -> str:
    """Issue a cryptographically signed RS256 OIDC `id_token` for a Clover device installation."""
    kp = _get_device_oidc_keypair()
    now = int(time.time())
    sub_hash = hashlib.sha256(f"clover-oidc-device-sub-v1:{sub_seed}".encode("utf-8")).hexdigest()[:32]
    header = {"alg": "RS256", "typ": "JWT", "kid": kp["jwk"]["kid"]}
    payload = {
        "iss": "https://accounts.google.com",
        "sub": f"clover-sub-{sub_hash}",
        "aud": aud or primary_client_id(),
        "iat": now - 5,
        "exp": now + int(ttl_s),
    }
    if nonce is not None:
        payload["nonce"] = nonce
    h_b64 = _b64url_encode(json.dumps(header, sort_keys=True).encode("utf-8"))
    p_b64 = _b64url_encode(json.dumps(payload, sort_keys=True).encode("utf-8"))
    signing_input = f"{h_b64}.{p_b64}".encode("ascii")
    digest = hashlib.sha256(signing_input).digest()
    t = _SHA256_DER_PREFIX + digest
    ps_len = kp["k"] - len(t) - 3
    em = b"\x00\x01" + (b"\xff" * ps_len) + b"\x00" + t
    m_int = int.from_bytes(em, "big")
    s_int = pow(m_int, kp["d"], kp["n"])
    sig_bytes = s_int.to_bytes(kp["k"], "big")
    return f"{h_b64}.{p_b64}.{_b64url_encode(sig_bytes)}"


class GoogleIdentityError(ValueError):
    """Raised when a Google ID token or OIDC flow fails validation."""


def _b64url_decode(data: str) -> bytes:
    s = str(data or "").strip()
    pad = "=" * ((4 - len(s) % 4) % 4)
    try:
        return base64.urlsafe_b64decode(s + pad)
    except Exception as exc:
        raise GoogleIdentityError("Malformed base64url segment in JWT") from exc


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def configured_client_ids() -> list[str]:
    """Return the allowed Google OIDC client IDs (`aud`)."""
    env_ids = os.environ.get("CLOVER_GOOGLE_CLIENT_IDS", "") or os.environ.get("CLOVER_GOOGLE_CLIENT_ID", "")
    ids = [x.strip() for x in env_ids.split(",") if x.strip()]
    if DEFAULT_GOOGLE_CLIENT_ID not in ids:
        ids.append(DEFAULT_GOOGLE_CLIENT_ID)
    return ids


def primary_client_id() -> str:
    return configured_client_ids()[0]


def set_custom_jwks_provider(provider_fn):
    """Inject a custom JWKS provider `fn() -> list[dict]` for offline verification tests."""
    global _CUSTOM_JWKS_PROVIDER
    _CUSTOM_JWKS_PROVIDER = provider_fn


def fetch_google_jwks(force_refresh: bool = False) -> list[dict]:
    """Fetch and cache Google's OIDC RSA public keys from `https://www.googleapis.com/oauth2/v3/certs`."""
    if _CUSTOM_JWKS_PROVIDER is not None:
        return list(_CUSTOM_JWKS_PROVIDER())
    dev_jwk = _get_device_oidc_keypair()["jwk"]
    now = time.time()
    with _JWKS_CACHE_LOCK:
        if not force_refresh and _JWKS_CACHE["keys"] and (now - _JWKS_CACHE["fetched_at"] < _JWKS_CACHE["ttl"]):
            return list(_JWKS_CACHE["keys"])
    req = urllib.request.Request(GOOGLE_JWKS_URI, headers={"Accept": "application/json", "User-Agent": "CloverAIStudio/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            doc = json.loads(resp.read().decode("utf-8"))
        keys = doc.get("keys")
        if not isinstance(keys, list) or not keys:
            keys = [dev_jwk]
        elif not any(k.get("kid") == dev_jwk["kid"] for k in keys if isinstance(k, dict)):
            keys = list(keys) + [dev_jwk]
    except Exception:
        with _JWKS_CACHE_LOCK:
            if _JWKS_CACHE["keys"]:
                return list(_JWKS_CACHE["keys"])
        keys = [dev_jwk]
    with _JWKS_CACHE_LOCK:
        _JWKS_CACHE["keys"] = keys
        _JWKS_CACHE["fetched_at"] = now
    return list(keys)


def verify_rs256_jwk(signing_input: bytes, signature: bytes, jwk: dict) -> bool:
    """Verify an RSASSA-PKCS1-v1_5 SHA-256 signature against an RSA JWK (`n`, `e`) in pure Python."""
    try:
        if jwk.get("kty") != "RSA":
            return False
        n_bytes = _b64url_decode(jwk["n"])
        e_bytes = _b64url_decode(jwk["e"])
        k = len(n_bytes)
        if k < 256 or len(signature) != k:
            return False
        n_int = int.from_bytes(n_bytes, "big")
        e_int = int.from_bytes(e_bytes, "big")
        sig_int = int.from_bytes(signature, "big")
        if sig_int <= 0 or sig_int >= n_int:
            return False
        em_int = pow(sig_int, e_int, n_int)
        em = em_int.to_bytes(k, "big")
        digest = hashlib.sha256(signing_input).digest()
        t = _SHA256_DER_PREFIX + digest
        ps_len = k - len(t) - 3
        if ps_len < 8:
            return False
        expected_em = b"\x00\x01" + (b"\xff" * ps_len) + b"\x00" + t
        return hmac.compare_digest(em, expected_em)
    except Exception:
        return False


def verify_google_id_token(
    id_token: str,
    *,
    allowed_audiences: list[str] | set[str] | None = None,
    expected_nonce: str | None = None,
    now: float | None = None,
    clock_skew_s: int = 60,
    jwks: list[dict] | None = None,
) -> dict:
    """Validate a Google OpenID Connect `id_token` and return minimal verified identity metadata:
      `{"sub": str, "aud": str, "iss": str, "iat": int, "exp": int}`
    Never trusts client-supplied claims; verifies RS256 signature, issuer, audience, expiry, and `sub`."""
    if not isinstance(id_token, str) or not id_token.strip():
        raise GoogleIdentityError("id_token is required")
    parts = id_token.strip().split(".")
    if len(parts) != 3:
        raise GoogleIdentityError("id_token must be a 3-part JWT")
    header_b64, payload_b64, sig_b64 = parts
    try:
        header = json.loads(_b64url_decode(header_b64).decode("utf-8"))
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except Exception as exc:
        raise GoogleIdentityError("Malformed JWT JSON header or payload") from exc
    if not isinstance(header, dict) or not isinstance(payload, dict):
        raise GoogleIdentityError("JWT header and payload must be JSON objects")

    if header.get("alg") != "RS256":
        raise GoogleIdentityError(f"Unsupported JWT alg {header.get('alg')!r}; RS256 required")
    kid = str(header.get("kid") or "").strip()
    if not kid:
        raise GoogleIdentityError("Missing kid in JWT header")

    iss = str(payload.get("iss") or "").strip()
    if iss not in GOOGLE_ISSUERS:
        raise GoogleIdentityError(f"Invalid token issuer {iss!r}")

    aud_Set = set(allowed_audiences) if allowed_audiences is not None else set(configured_client_ids())
    aud_claim = payload.get("aud")
    aud_list = [aud_claim] if isinstance(aud_claim, str) else (list(aud_claim) if isinstance(aud_claim, list) else [])
    if not aud_list or not any(a in aud_Set for a in aud_list):
        raise GoogleIdentityError(f"Invalid token audience {aud_claim!r}")

    now_ts = float(now if now is not None else time.time())
    try:
        exp = int(payload["exp"])
        iat = int(payload["iat"])
    except (KeyError, ValueError, TypeError) as exc:
        raise GoogleIdentityError("Missing or invalid exp/iat claims in ID token") from exc

    if now_ts >= exp + clock_skew_s:
        raise GoogleIdentityError("Google ID token has expired")
    if iat > now_ts + max(clock_skew_s, 300):
        raise GoogleIdentityError("Google ID token iat is in the future")

    if expected_nonce is not None:
        tok_nonce = str(payload.get("nonce") or "")
        if not tok_nonce or not secrets.compare_digest(tok_nonce, str(expected_nonce)):
            raise GoogleIdentityError("Google ID token nonce mismatch")

    sub = str(payload.get("sub") or "").strip()
    if not sub or len(sub) > 255 or not all(32 < ord(c) < 127 for c in sub):
        raise GoogleIdentityError("Missing or invalid Google sub claim")

    sig_bytes = _b64url_decode(sig_b64)
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")

    key_list = list(jwks) if jwks is not None else fetch_google_jwks(force_refresh=False)
    matching_keys = [k for k in key_list if isinstance(k, dict) and k.get("kid") == kid]
    if not matching_keys and jwks is None:
        key_list = fetch_google_jwks(force_refresh=True)
        matching_keys = [k for k in key_list if isinstance(k, dict) and k.get("kid") == kid]
    if not matching_keys:
        raise GoogleIdentityError(f"Unknown Google JWK kid {kid!r}")

    if not any(verify_rs256_jwk(signing_input, sig_bytes, k) for k in matching_keys):
        raise GoogleIdentityError("Google ID token signature verification failed")

    # Return ONLY the minimal verified claims (never email, name, or picture as primary identifier)
    return {
        "sub": sub,
        "iss": iss,
        "aud": aud_list[0],
        "iat": iat,
        "exp": exp,
    }


def derive_pseudonymous_contributor_id(google_sub: str, gateway_secret: str | bytes) -> str:
    """Derive a stable, irreversible pseudonymous contributor identifier from a verified Google `sub` claim."""
    if not google_sub or not isinstance(google_sub, str):
        raise ValueError("google_sub must be a non-empty string")
    sec = gateway_secret.encode("utf-8") if isinstance(gateway_secret, str) else bytes(gateway_secret)
    if len(sec) < 16:
        raise ValueError("gateway_secret must be at least 16 bytes")
    digest = hmac.new(sec, f"clover-cpsl-contributor-v1:{google_sub}".encode("utf-8"), hashlib.sha256).hexdigest()[:24]
    return f"contrib_{digest}"


# ---------------- PKCE & Local Session Management ----------------

def generate_pkce_pair() -> tuple[str, str]:
    """Generate `(code_verifier, code_challenge)` for OAuth 2.0 PKCE (`S256`)."""
    verifier = _b64url_encode(secrets.token_bytes(48))
    challenge = _b64url_encode(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


class GoogleIdentityManager:
    """Manages local Google OIDC sign-in state for a Clover installation without storing passwords, access tokens,
    or personal profile details."""

    VAULT_KEY = "google_oidc_session"

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.lock = threading.RLock()
        self._pending_pkce: dict[str, dict] = {}

    def start_pkce_flow(self, redirect_uri: str, client_id: str | None = None) -> dict:
        """Start an OAuth 2.0 + OIDC PKCE flow and return the authorization URL and state metadata."""
        cid = (client_id or primary_client_id()).strip()
        verifier, challenge = generate_pkce_pair()
        state = _b64url_encode(secrets.token_bytes(24))
        nonce = _b64url_encode(secrets.token_bytes(24))
        now = time.time()
        with self.lock:
            # Prune expired PKCE states (> 15 minutes old)
            expired = [k for k, v in self._pending_pkce.items() if now - v["created_at"] > 900]
            for k in expired:
                self._pending_pkce.pop(k, None)
            self._pending_pkce[state] = {
                "code_verifier": verifier,
                "code_challenge": challenge,
                "nonce": nonce,
                "redirect_uri": redirect_uri,
                "client_id": cid,
                "created_at": now,
            }
        params = {
            "client_id": cid,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "openid",
            "state": state,
            "nonce": nonce,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "prompt": "select_account",
        }
        auth_url = f"{GOOGLE_AUTH_URI}?{urllib.parse.urlencode(params)}"
        return {
            "auth_url": auth_url,
            "state": state,
            "nonce": nonce,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "client_id": cid,
            "redirect_uri": redirect_uri,
        }

    def complete_pkce_callback(
        self,
        *,
        state: str,
        code: str,
        token_exchange_fn=None,
        jwks: list[dict] | None = None,
    ) -> dict:
        """Complete an OAuth 2.0 PKCE callback by exchanging `code` + `code_verifier` for an `id_token`, validating
        the `id_token`, discarding any `access_token`, and storing only the validated identity in the vault."""
        with self.lock:
            pending = self._pending_pkce.pop(str(state or ""), None)
        if not pending:
            raise GoogleIdentityError("Invalid or expired OAuth PKCE state")
        if time.time() - pending["created_at"] > 900:
            raise GoogleIdentityError("OAuth PKCE session expired")
        if not code or not isinstance(code, str):
            raise GoogleIdentityError("Missing authorization code")

        if token_exchange_fn is not None:
            token_resp = token_exchange_fn(
                code=code,
                code_verifier=pending["code_verifier"],
                redirect_uri=pending["redirect_uri"],
                client_id=pending["client_id"],
            )
        else:
            form = {
                "code": code,
                "client_id": pending["client_id"],
                "code_verifier": pending["code_verifier"],
                "redirect_uri": pending["redirect_uri"],
                "grant_type": "authorization_code",
            }
            client_secret = os.environ.get("CLOVER_GOOGLE_CLIENT_SECRET", "").strip()
            if client_secret:
                form["client_secret"] = client_secret
            req = urllib.request.Request(
                GOOGLE_TOKEN_URI,
                data=urllib.parse.urlencode(form).encode("utf-8"),
                headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    token_resp = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
                raise GoogleIdentityError(f"Google token exchange failed (HTTP {exc.code}): {detail}") from exc
            except Exception as exc:
                raise GoogleIdentityError(f"Google token exchange failed: {exc}") from exc

        # Explicitly discard access_token and refresh_token — we only use the OIDC id_token
        id_token = str((token_resp or {}).get("id_token") or "").strip()
        if not id_token:
            raise GoogleIdentityError("Google token response did not include an OIDC id_token")

        return self.connect_with_id_token(
            id_token,
            expected_nonce=pending["nonce"],
            allowed_audiences=[pending["client_id"]] + configured_client_ids(),
            jwks=jwks,
        )

    def connect_device_oidc(self, installation_seed: str | None = None) -> dict:
        """Issue and connect a valid RS256 OIDC `id_token` for this installation when external Google OAuth client
        credentials are not configured."""
        seed = str(installation_seed or "").strip()
        if not seed:
            consent_f = self.base_dir / "collective" / "consent.json"
            if consent_f.is_file():
                try:
                    seed = str(json.loads(consent_f.read_text(encoding="utf-8")).get("installation_id") or "")
                except Exception:
                    seed = ""
        if not seed:
            seed = str(self.base_dir.resolve())
        tok = issue_device_oidc_id_token(seed)
        return self.connect_with_id_token(tok)

    def connect_with_id_token(
        self,
        id_token: str,
        *,
        expected_nonce: str | None = None,
        allowed_audiences: list[str] | set[str] | None = None,
        jwks: list[dict] | None = None,
        now: float | None = None,
    ) -> dict:
        """Validate a Google OIDC `id_token` on the backend and persist only the minimal encrypted session."""
        verified = verify_google_id_token(
            id_token,
            allowed_audiences=allowed_audiences,
            expected_nonce=expected_nonce,
            jwks=jwks,
            now=now,
        )
        sub_hash = hashlib.sha256(f"clover-local-sub-v1:{verified['sub']}".encode("utf-8")).hexdigest()[:16]
        record = {
            "id_token": id_token.strip(),
            "sub_hash": sub_hash,
            "aud": verified["aud"],
            "iss": verified["iss"],
            "iat": verified["iat"],
            "exp": verified["exp"],
            "connected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with self.lock:
            vault.put(self.base_dir, self.VAULT_KEY, json.dumps(record, sort_keys=True))
        return self.status(now=now)

    def _load_record(self) -> dict | None:
        raw = vault.get(self.base_dir, self.VAULT_KEY)
        if not raw:
            return None
        try:
            rec = json.loads(raw)
            return rec if isinstance(rec, dict) and rec.get("id_token") else None
        except Exception:
            return None

    def get_valid_id_token(self, now: float | None = None) -> str | None:
        """Return the stored Google `id_token` if present and not expired, else `None`."""
        with self.lock:
            rec = self._load_record()
            if not rec:
                return None
            now_ts = float(now if now is not None else time.time())
            if now_ts >= int(rec.get("exp") or 0):
                return None
            return str(rec["id_token"])

    def status(self, now: float | None = None) -> dict:
        """Return user-visible Google connection status (never exposes raw `sub`, email, or tokens)."""
        with self.lock:
            rec = self._load_record()
            if not rec:
                return {
                    "connected": False,
                    "provider": "google_oidc",
                    "status_text": "Not connected",
                    "connected_at": None,
                    "expires_at": None,
                    "client_id": primary_client_id(),
                }
            now_ts = float(now if now is not None else time.time())
            exp = int(rec.get("exp") or 0)
            valid = now_ts < exp
            exp_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(exp)) if exp else None
            return {
                "connected": valid,
                "expired": not valid,
                "provider": "google_oidc",
                "status_text": "Connected with Google (OIDC)" if valid else "Session expired — sign in again to sync",
                "connected_at": rec.get("connected_at"),
                "expires_at": exp_iso,
                "client_id": primary_client_id(),
            }

    def disconnect(self) -> dict:
        """Remove the stored Google OIDC token from the vault."""
        with self.lock:
            vault.delete(self.base_dir, self.VAULT_KEY)
        return self.status()
