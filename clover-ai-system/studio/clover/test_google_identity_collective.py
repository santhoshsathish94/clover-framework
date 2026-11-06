#!/usr/bin/env python3
"""End-to-End Verification Suite for Google OpenID Connect Identity & Collective Learning Integration.

Verifies all 12 Acceptance Criteria from the specification:
  1. Clover works without Google sign-in.
  2. Personal learning works offline.
  3. Google sign-in works on Android (OIDC ID token) and Windows (OAuth 2.0 PKCE loopback).
  4. No separate Clover account is created (only Google `sub` claim; no passwords, emails, or profiles stored).
  5. Collective contribution is off by default.
  6. Collective receiving is off by default.
  7. Neither can activate without Google identity and explicit consent (never auto-enabled on sign-in).
  8. Only approved candidate knowledge can leave the device (confirmation gate + closed-vocab PII filter).
  9. Collective releases are cryptographically verified (Ed25519 signature, SHA-256 hashes, expiry & revocation).
 10. Two separate installations can contribute, validate, distribute, and receive a signed collective release
     end-to-end over HTTP.
 11. Disabling consent / withdrawing prevents future synchronization and demotes withdrawn quorum rules.
 12. Existing personal knowledge remains intact across upgrades, syncs, and sign-out.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import clover_cycle as C  # noqa: E402
import collective  # noqa: E402
import collective_evaluator  # noqa: E402
import collective_gateway  # noqa: E402
import core_registry  # noqa: E402
import core_sync  # noqa: E402
import google_identity  # noqa: E402
import personal  # noqa: E402
import semantics  # noqa: E402
import vault  # noqa: E402

FAILED: list[str] = []
PASSED = [0]


def check(cond: bool, what: str) -> None:
    if cond:
        PASSED[0] += 1
        print("PASS", what)
    else:
        FAILED.append(what)
        print("FAIL", what)


# ---------------- Deterministic 2048-bit RSA JWK & JWT Signer for Offline OIDC Tests ----------------
# Two prime factors p and q (1024-bit each) generated deterministically for test signing
def _make_test_rsa_keypair():
    # Use a deterministic odd prime pair or generate via standard Miller-Rabin in pure Python
    def is_probable_prime(n: int) -> bool:
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

    def next_prime(seed_bytes: bytes) -> int:
        cand = int.from_bytes(hashlib.sha512(seed_bytes).digest() * 2, "big") | (1 << 1023) | 1
        while not is_probable_prime(cand):
            cand += 2
        return cand

    p = next_prime(b"clover-test-oidc-rsa-prime-p-v1")
    q = next_prime(b"clover-test-oidc-rsa-prime-q-v1")
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
        "kid": "test-google-kid-2026",
        "n": google_identity._b64url_encode(n_bytes),
        "e": google_identity._b64url_encode(e_bytes),
    }
    return {"n": n, "e": e, "d": d, "k": 256, "jwk": jwk}


RSA_KEY = _make_test_rsa_keypair()


def make_signed_google_jwt(
    sub: str,
    *,
    aud: str | None = None,
    iss: str = "https://accounts.google.com",
    exp_offset_s: int = 3600,
    iat_offset_s: int = -5,
    nonce: str | None = None,
    kid: str = "test-google-kid-2026",
    tamper_sig: bool = False,
    extra_claims: dict | None = None,
) -> str:
    now = int(time.time())
    header = {"alg": "RS256", "typ": "JWT", "kid": kid}
    payload = {
        "iss": iss,
        "sub": sub,
        "aud": aud or google_identity.primary_client_id(),
        "iat": now + iat_offset_s,
        "exp": now + exp_offset_s,
    }
    if nonce is not None:
        payload["nonce"] = nonce
    if extra_claims:
        payload.update(extra_claims)
    h_b64 = google_identity._b64url_encode(json.dumps(header, sort_keys=True).encode("utf-8"))
    p_b64 = google_identity._b64url_encode(json.dumps(payload, sort_keys=True).encode("utf-8"))
    signing_input = f"{h_b64}.{p_b64}".encode("ascii")
    digest = hashlib.sha256(signing_input).digest()
    t = google_identity._SHA256_DER_PREFIX + digest
    ps_len = RSA_KEY["k"] - len(t) - 3
    em = b"\x00\x01" + (b"\xff" * ps_len) + b"\x00" + t
    m_int = int.from_bytes(em, "big")
    s_int = pow(m_int, RSA_KEY["d"], RSA_KEY["n"])
    sig_bytes = bytearray(s_int.to_bytes(RSA_KEY["k"], "big"))
    if tamper_sig:
        sig_bytes[-1] ^= 0xFF
    s_b64 = google_identity._b64url_encode(bytes(sig_bytes))
    return f"{h_b64}.{p_b64}.{s_b64}"


google_identity.set_custom_jwks_provider(lambda: [RSA_KEY["jwk"]])


PHONE_TOOLS = {
    "web.search": ("web_search",), "web.page": ("web_search",), "python": ("python_interpreter",),
    "files.read": ("repo_read",), "files.list": ("repo_read",), "files.write": ("repo_write",),
    "shell": ("run_shell",), "status": ("run_shell",), "ssh": ("run_ssh",), "image": ("generate_image",),
    "memory.search": ("repo_search",), "remember": ("remember",),
}
_RISKY = lambda cmd: ("deletes files" if "del " in str(cmd).lower() or "rm " in str(cmd).lower() else None)


class Chat:
    def __init__(self, rt, session):
        self.rt, self.session, self.history = rt, session, []

    def say(self, text):
        last = C.last_exchange(self.history) if self.history else None
        retry = C.retry_requested(text, last[0]["content"]) if last else None
        if self.rt.personal:
            self.rt.personal.before_turn(self.session, text, retry)
        f = semantics.interpret(text, self.rt, history=self.history)
        if retry:
            f.act = "retry"
        if self.rt.personal:
            self.rt.personal.after_turn(self.session, f, {"turn_n": len(self.history) // 2 + 1})
        self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": f"(answer to {text})"}]
        return f


# ==============================================================================
# 1 & 2. Clover works without Google sign-in & Personal learning works offline
# ==============================================================================
with tempfile.TemporaryDirectory() as tmp_dir:
    base_a = Path(tmp_dir) / "inst_a"
    base_b = Path(tmp_dir) / "inst_b"
    gw_dir = Path(tmp_dir) / "gateway_srv"
    for d in (base_a, base_b, gw_dir):
        d.mkdir(parents=True, exist_ok=True)

    id_mgr_a = google_identity.GoogleIdentityManager(base_a)
    id_mgr_b = google_identity.GoogleIdentityManager(base_b)
    check(not id_mgr_a.status()["connected"], "AC1: Installation A starts without Google sign-in")

    pers_a = personal.Personal(base_a / "personal", device="phone", filter_repo=[])
    priv_seed_hex, pub_verify_hex = core_registry.generate_ed25519_keypair()
    coll_a = core_sync.Collective(base_a / "collective", device="phone", verify_key=pub_verify_hex)
    rt_a = semantics.Runtime(tools=PHONE_TOOLS, device="phone", risky=_RISKY, personal=pers_a, collective=coll_a)

    # AC5 & AC6: Collective contribution and receiving are off by default
    check(not coll_a.consent_contribute and not coll_a.consent_receive,
          "AC5 & AC6: Collective contribution and receiving are disabled by default")

    # AC1 & AC2: Normal interpretation and offline personal learning work without Google sign-in
    chat_a = Chat(rt_a, "sess-offline-a")
    chat_a.say("Summarize Macbeth.")
    chat_a.say("Shorter, please.")
    f_offline = chat_a.say("Explain quantum tunneling.")
    check((f_offline.personal.get("presentation") or {}).get("length") == "short" and "keep answers short" in f_offline.guidance,
          "AC1 & AC2: Offline personal learning learns user preferences without Google sign-in")

    # Learn a multi-capability strategy ("corroborate" -> web.search + python) plus a private habit on Installation A
    chat_a.say("Corroborate the claim about fusion (secret_project_alpha_99).")
    chat_a.say("No, search the web and use Python to check.")
    chat_a.say("Corroborate the figures on solar panels for alice@acmecorp.internal.")
    chat_a.say("No, look it up online and use Python to compute.")
    chat_a.say("When I say ping, I mean search the web.")

    # AC7: Cannot enable collective contribution or receiving without Google sign-in
    blocked_without_google = False
    try:
        coll_a.set_consent(contribute=True, google_connected=id_mgr_a.status()["connected"], require_identity=True)
    except core_sync.CoreSyncError:
        blocked_without_google = True
    check(blocked_without_google and not coll_a.consent_contribute,
          "AC7: Enabling collective contribution without Google sign-in is rejected")

    # ==============================================================================
    # 3 & 4. Google OIDC Sign-In (Windows PKCE Loopback & Android ID Token) & Privacy
    # ==============================================================================
    # Windows OAuth 2.0 + OIDC PKCE flow on Installation A
    pkce_start = id_mgr_a.start_pkce_flow(redirect_uri="http://127.0.0.1:8765/api/identity/google/callback")
    check("code_challenge=" in pkce_start["auth_url"] and pkce_start["code_challenge_method"] == "S256"
          and "scope=openid" in pkce_start["auth_url"] and "email" not in pkce_start["auth_url"],
          "AC3 & AC4: Windows OAuth PKCE flow requests only 'openid' scope with S256 challenge")

    def fake_google_token_endpoint(code, code_verifier, redirect_uri, client_id):
        # Verify PKCE S256 challenge matches verifier
        expected_chal = google_identity._b64url_encode(hashlib.sha256(code_verifier.encode("ascii")).digest())
        assert expected_chal == pkce_start["code_challenge"], "PKCE challenge mismatch"
        tok = make_signed_google_jwt(
            "google-sub-user-alpha-10001",
            aud=client_id,
            nonce=pkce_start["nonce"],
            extra_claims={"email": "alpha@example.com", "name": "Alpha User"},
        )
        return {
            "id_token": tok,
            "access_token": "ya29.should_be_discarded_immediately",
            "refresh_token": "1//should_also_be_discarded",
        }

    st_a = id_mgr_a.complete_pkce_callback(
        state=pkce_start["state"],
        code="auth_code_xyz",
        token_exchange_fn=fake_google_token_endpoint,
    )
    check(st_a["connected"] is True, "AC3: Windows OAuth PKCE callback validates RS256 Google ID token and connects")

    # AC4: Verify no email, name, access_token, refresh_token, or raw sub is stored in plaintext or status
    raw_vault_file = (base_a / "vault" / "remote-models.json").read_text(encoding="utf-8")
    stored_rec = json.loads(vault.get(base_a, google_identity.GoogleIdentityManager.VAULT_KEY))
    check("access_token" not in stored_rec and "refresh_token" not in stored_rec
          and "email" not in stored_rec and "name" not in stored_rec
          and "alpha@example.com" not in json.dumps(st_a)
          and "google-sub-user-alpha-10001" not in json.dumps(st_a),
          "AC4: Access/refresh tokens and email/profile are discarded; raw sub is never exposed in status")

    # AC7: Signing in with Google does NOT silently enable collective contribution or receiving
    check(not coll_a.consent_contribute and not coll_a.consent_receive,
          "AC7: Signing in with Google does NOT silently enable collective contribution or receiving")

    # Android Credential Manager / Direct OIDC ID Token flow on Installation B
    tok_b = make_signed_google_jwt("google-sub-user-beta-20002", nonce="android-nonce-42")
    st_b = id_mgr_b.connect_with_id_token(tok_b, expected_nonce="android-nonce-42")
    check(st_b["connected"] is True, "AC3: Android OIDC ID token flow validates RS256 signature, nonce, and sub")

    # Security checks on Google ID token validator: reject tampered signature, wrong issuer, wrong audience, expired token
    for bad_label, bad_kwargs in (
        ("tampered signature", {"tamper_sig": True}),
        ("wrong issuer", {"iss": "https://evil.example.com"}),
        ("wrong audience", {"aud": "wrong-client-id.apps.googleusercontent.com"}),
        ("expired token", {"exp_offset_s": -120}),
    ):
        bad_tok = make_signed_google_jwt("google-sub-attacker", **bad_kwargs)
        rejected = False
        try:
            google_identity.verify_google_id_token(bad_tok)
        except google_identity.GoogleIdentityError:
            rejected = True
        check(rejected, f"AC6/Security: Backend rejects Google ID token with {bad_label}")

    # ==============================================================================
    # 8, 9, 10, 11, 12. Real HTTP Gateway + Two-Installation End-to-End Sync
    # ==============================================================================
    gw = collective_gateway.CollectiveGateway(state_dir=gw_dir / "gw_state", gateway_secret="test-gateway-hmac-secret-32bytes!!")
    ev = collective_evaluator.CollectiveEvaluator(min_installations=2, min_agreement=0.75)
    reg = core_registry.CoreRegistry(gw_dir / "clover-collective", signing_key=priv_seed_hex)
    gw_lock = threading.RLock()

    class GatewayHTTPHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _bearer(self) -> str:
            h = (self.headers.get("Authorization") or "").strip()
            return h[7:].strip() if h.lower().startswith("bearer ") else ""

        def _reply(self, code: int, obj: dict):
            raw = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            path, _, query = self.path.partition("?")
            if path == "/api/collective/gateway/releases":
                q = {k: v[0] for k, v in urllib.parse.parse_qs(query).items()}
                try:
                    gw.authenticate_contributor(self._bearer())
                except collective_gateway.GatewaySecurityError as exc:
                    return self._reply(exc.status, {"error": exc.message, "code": exc.code})
                with gw_lock:
                    return self._reply(200, reg.get_release_bundle(since_version=q.get("since_version")))
            return self._reply(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n).decode("utf-8")) if n > 0 else {}
            try:
                with gw_lock:
                    if self.path == "/api/collective/gateway/contribute":
                        res = gw.submit_authenticated_contribution(id_token=self._bearer(), payload=body)
                    elif self.path == "/api/collective/gateway/withdraw":
                        res = gw.withdraw_authenticated_contributor(id_token=self._bearer(), payload=body)
                    else:
                        return self._reply(404, {"error": "not found"})
                    agg = gw.aggregate_candidates()
                    eval_res = ev.evaluate_all(agg)
                    manifest = reg.publish_release(eval_res, note="Live E2E gateway evaluation")
                    res["release_version"] = manifest["version"]
                    res["promoted_count"] = eval_res["promoted_count"]
                    return self._reply(200, res)
            except collective_gateway.GatewaySecurityError as exc:
                return self._reply(exc.status, {"error": exc.message, "code": exc.code})

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), GatewayHTTPHandler)
    gw_port = httpd.server_address[1]
    gw_url = f"http://127.0.0.1:{gw_port}"
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    try:
        coll_a.gateway_url = gw_url
        pers_b = personal.Personal(base_b / "personal", device="phone", filter_repo=[])
        coll_b = core_sync.Collective(base_b / "collective", device="phone", verify_key=pub_verify_hex, gateway_url=gw_url)
        rt_b = semantics.Runtime(tools=PHONE_TOOLS, device="phone", risky=_RISKY, personal=pers_b, collective=coll_b)

        # Installation B also learns "corroborate" -> web.search + python locally
        chat_b = Chat(rt_b, "sess-b")
        chat_b.say("Corroborate the claim about battery density (confidential_client_77).")
        chat_b.say("No, search the web and use Python to check.")
        chat_b.say("Corroborate the figures on wind turbines.")
        chat_b.say("No, look it up online and use Python to compute.")

        # AC8: Attempt to upload from Installation A BEFORE enabling consent_contribute -> rejected
        cands_a = coll_a.preview_candidates(pers_a)
        check(len(cands_a) == 1 and "tok=corroborate" in cands_a[0]["trigger"],
              "AC8: Installation A previews 1 scrubbed candidate package before opt-in")

        no_consent_blocked = False
        try:
            coll_a.contribute_to_gateway(id_token=id_mgr_a.get_valid_id_token(), packages=cands_a, confirmed=True)
        except core_sync.CoreSyncError:
            no_consent_blocked = True
        check(no_consent_blocked, "AC8: Contribution is blocked when consent_contribute is False")

        # Enable consent_contribute on Installation A (with Google connected)
        coll_a.set_consent(contribute=True, google_connected=id_mgr_a.status()["connected"], require_identity=True)

        # AC8: Attempt to upload WITHOUT explicit confirmation (`confirmed=False`) -> rejected
        unconfirmed_blocked = False
        try:
            coll_a.contribute_to_gateway(id_token=id_mgr_a.get_valid_id_token(), packages=cands_a, confirmed=False)
        except core_sync.CoreSyncError:
            unconfirmed_blocked = True
        check(unconfirmed_blocked, "AC8: Contribution is blocked when user confirmation (confirmed=True) is missing")

        # AC6/Security: Gateway rejects raw conversation histories or personal memories
        raw_mem_rejected = False
        try:
            coll_a._http_call(
                "POST",
                f"{gw_url}/api/collective/gateway/contribute",
                id_token=id_mgr_a.get_valid_id_token(),
                body={
                    "confirmed": True,
                    "consent_contribute": True,
                    "nonce": "nonce_raw_mem_12345",
                    "timestamp": time.time(),
                    "packages": cands_a,
                    "memories": ["secret personal memory"],
                },
            )
        except core_sync.CoreSyncError:
            raw_mem_rejected = True
        check(raw_mem_rejected, "AC4/AC6: Gateway rejects payload containing personal memories or conversation history")

        # Installation A contributes its confirmed candidate package to the Gateway
        res_a = coll_a.contribute_to_gateway(
            id_token=id_mgr_a.get_valid_id_token(), packages=cands_a, confirmed=True
        )
        check(res_a["ok"] and res_a["accepted_count"] == 1 and res_a["promoted_count"] == 0,
              "AC10: Installation A's contribution is accepted by gateway (quorum 1/2 -> not promoted yet)")

        # AC6/Security: Replay protection rejects duplicate nonce
        replay_rejected = False
        try:
            coll_a._http_call(
                "POST",
                f"{gw_url}/api/collective/gateway/contribute",
                id_token=id_mgr_a.get_valid_id_token(),
                body={
                    "confirmed": True,
                    "consent_contribute": True,
                    "nonce": "fixed_replay_nonce_99999",
                    "timestamp": time.time(),
                    "packages": cands_a,
                },
            )
            coll_a._http_call(
                "POST",
                f"{gw_url}/api/collective/gateway/contribute",
                id_token=id_mgr_a.get_valid_id_token(),
                body={
                    "confirmed": True,
                    "consent_contribute": True,
                    "nonce": "fixed_replay_nonce_99999",
                    "timestamp": time.time(),
                    "packages": cands_a,
                },
            )
        except core_sync.CoreSyncError:
            replay_rejected = True
        check(replay_rejected, "AC6/Security: Gateway rejects replayed request nonce (HTTP 409)")

        # Installation B enables contribution consent and contributes its confirmed candidate
        coll_b.set_consent(
            contribute=True, receive=True, google_connected=id_mgr_b.status()["connected"], require_identity=True
        )
        cands_b = coll_b.preview_candidates(pers_b)
        res_b = coll_b.contribute_to_gateway(
            id_token=id_mgr_b.get_valid_id_token(), packages=cands_b, confirmed=True
        )
        check(res_b["ok"] and res_b["promoted_count"] == 1,
              "AC10: Installation B's corroborating contribution reaches quorum (2/2), passes G1..G5, and publishes signed release")

        # Verify pseudonymous contributor IDs: neither raw Google sub nor client installation_id is in registry
        leaked = reg.find_strings_anywhere_in_repo([
            "google-sub-user-alpha-10001",
            "google-sub-user-beta-20002",
            "alpha@example.com",
            "secret_project_alpha_99",
            "confidential_client_77",
            coll_a.installation_id,
            coll_b.installation_id,
        ])
        check(leaked == [], "AC4/AC6: Zero Google subs, emails, installation IDs, or private tokens in collective registry")

        # AC10: Installation A (or a clean recipient Installation C) receives the validated collective release over HTTP
        base_c = Path(tmp_dir) / "inst_c"
        base_c.mkdir(parents=True, exist_ok=True)
        id_mgr_c = google_identity.GoogleIdentityManager(base_c)
        tok_c = make_signed_google_jwt("google-sub-user-gamma-30003")
        id_mgr_c.connect_with_id_token(tok_c)
        pers_c = personal.Personal(base_c / "personal", device="phone", filter_repo=[])
        coll_c = core_sync.Collective(base_c / "collective", device="phone", verify_key=pub_verify_hex, gateway_url=gw_url)
        rt_c = semantics.Runtime(tools=PHONE_TOOLS, device="phone", risky=_RISKY, personal=pers_c, collective=coll_c)

        # Before opt-in on Installation C: collective rule does not fire
        f_c0 = semantics.interpret("Corroborate the claim about geothermal energy.", rt_c)
        check(f_c0.caps == [] and not f_c0.collective, "AC6/AC7: Before receive opt-in on Installation C, collective rule is inactive")

        # Opt in to receive on Installation C and synchronize over HTTP from the Gateway
        coll_c.set_consent(receive=True, google_connected=id_mgr_c.status()["connected"], require_identity=True)
        sync_c1 = coll_c.sync_from_gateway(id_token=id_mgr_c.get_valid_id_token())
        check(sync_c1["synced"] and sync_c1["rules_count"] == 1 and not sync_c1.get("up_to_date"),
              "AC9 & AC10: Installation C downloads, verifies Ed25519 signature & SHA-256 hashes, and activates 1 shared rule")

        # Second sync on Installation C uses incremental sync (`since_version`) and reports `up_to_date=True`
        sync_c2 = coll_c.sync_from_gateway(id_token=id_mgr_c.get_valid_id_token())
        check(sync_c2["synced"] and sync_c2["up_to_date"] is True,
              "AC4/Registry: Incremental synchronization reports up_to_date=True with 0 redundant file transfers")

        # Verify that Installation C now applies the shared collective rule on an unseen prompt
        f_c1 = semantics.interpret("Corroborate the claim about geothermal energy.", rt_c)
        check(set(f_c1.asked) == {"web_search", "python_interpreter"} and f_c1.collective.get("rule_id") is not None,
              "AC10: Installation C routes 'Corroborate...' to web_search + python_interpreter using verified collective release")

        # AC9: Test expired and revoked release rejection
        with tempfile.TemporaryDirectory() as exp_dir:
            reg_exp = core_registry.CoreRegistry(exp_dir, signing_key=priv_seed_hex)
            eval_now = ev.evaluate_all(gw.aggregate_candidates())
            reg_exp.publish_release(eval_now, note="Expired release test", expires_utc="2020-01-01T00:00:00Z")
            expired_rejected = False
            try:
                core_sync.CoreSync(verify_key=pub_verify_hex).verify_and_load(exp_dir)
            except core_sync.CoreSyncError as exc:
                expired_rejected = "expired" in str(exc).lower()
            check(expired_rejected, "AC9: CoreSync rejects expired collective release")

            # Test revoked release rejection
            reg_rev = core_registry.CoreRegistry(Path(exp_dir) / "rev", signing_key=priv_seed_hex)
            m_rev = reg_rev.publish_release(eval_now, note="Revocable release")
            reg_rev.revoke_version(m_rev["version"])
            revoked_rejected = False
            try:
                core_sync.CoreSync(verify_key=pub_verify_hex).verify_and_load(Path(exp_dir) / "rev")
            except core_sync.CoreSyncError as exc:
                revoked_rejected = "revoked" in str(exc).lower()
            check(revoked_rejected, "AC9: CoreSync rejects revoked collective release")

        # AC11: Disabling receive consent on Installation C immediately stops applying collective rules and blocks sync
        coll_c.set_consent(receive=False)
        f_c_disabled = semantics.interpret("Corroborate the claim about geothermal energy.", rt_c)
        check(f_c_disabled.caps == [] and not f_c_disabled.collective,
              "AC11: Disabling receive consent immediately stops collective rules from applying")
        sync_blocked = False
        try:
            coll_c.sync_from_gateway(id_token=id_mgr_c.get_valid_id_token())
        except core_sync.CoreSyncError:
            sync_blocked = True
        check(sync_blocked, "AC11: Disabling receive consent prevents future release synchronization")

        # AC11: Installation A withdraws contribution consent -> Gateway purges Installation A's candidates & demotes rule
        w_res = coll_a.withdraw_from_gateway(id_token=id_mgr_a.get_valid_id_token())
        check(w_res["ok"] and not coll_a.consent_contribute and w_res["gateway"]["promoted_count"] == 0,
              "AC11: Withdrawing Installation A's contribution purges its candidates and demotes the rule in the next release")

        # Re-enable receive on Installation C and sync -> rule is now demoted (0 active rules)
        coll_c.set_consent(receive=True, google_connected=id_mgr_c.status()["connected"], require_identity=True)
        sync_c_after_withdraw = coll_c.sync_from_gateway(id_token=id_mgr_c.get_valid_id_token())
        check(sync_c_after_withdraw["rules_count"] == 0,
              "AC11: Recipient installation receives updated release with withdrawn rule demoted (0 rules)")

        # AC12: Existing personal knowledge on Installation A remains 100% intact after withdrawal & Google disconnect
        id_mgr_a.disconnect()
        coll_a.set_consent(contribute=False, receive=False)
        f_pers_intact = chat_a.say("Explain superconductivity.")
        pers_view_a = pers_a.view()
        check((f_pers_intact.personal.get("presentation") or {}).get("length") == "short"
              and len(pers_view_a.get("experiences") or []) >= 3,
              "AC12: Existing local personal knowledge and preferences remain intact after sync, withdrawal, and sign-out")
    finally:
        httpd.shutdown()

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: {FAILED}")
    sys.exit(1)
print(f"ALL {PASSED[0]} GOOGLE IDENTITY & COLLECTIVE LEARNING CHECKS PASSED")
