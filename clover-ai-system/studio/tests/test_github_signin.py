"""Sign in with GitHub: one account for the memory repository, Workspaces, k3 access and collective learning (human
direction 2026-11-11). Offline: GitHub is a stand-in that answers like github.com, so no real account is touched.
Checks the device flow, token renewal, the encrypted store, the website's redirect sign-in, the collective gateway's
own credential, and that the self-made "Google" key is gone. Usage: python tests/test_github_signin.py"""
import argparse
import hashlib
import base64
import json
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "clover"))
sys.path.insert(0, str(HERE))
import collective_gateway  # noqa: E402
import core_sync  # noqa: E402
import github_auth as G  # noqa: E402
import google_identity  # noqa: E402
import server  # noqa: E402
import ui_harness  # noqa: E402
import workspaces as WS  # noqa: E402

ok = True


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    ok &= bool(cond)


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


class FakeGitHub:
    """Answers GitHub's OAuth endpoints from a script and records every form it was sent."""

    def __init__(self):
        self.calls, self.token_answers = [], []

    def __call__(self, url, form, timeout=20):
        self.calls.append((url, dict(form)))
        if url == G.DEVICE_CODE_URL:
            return {"device_code": "DEVICE-CODE-SECRET", "user_code": "WDJB-MJHT",
                    "verification_uri": "https://github.com/login/device", "expires_in": 900, "interval": 5}
        return self.token_answers.pop(0) if self.token_answers else {"error": "authorization_pending"}


ACCESS = "gho_" + "a" * 36
NEW_ACCESS = "gho_" + "b" * 36
REFRESH = "ghr_" + "c" * 76

# ---------------- device flow ----------------
clock, gh = Clock(), FakeGitHub()
flows = G.DeviceFlows(post=gh, clock=clock)
st = flows.start()
check("start: the page gets a flow id, the user code and GitHub's page, never the device code",
      st["user_code"] == "WDJB-MJHT" and st["verification_uri"] == "https://github.com/login/device"
      and "DEVICE-CODE-SECRET" not in json.dumps(st) and st["interval"] == 5)
check("start: Clover's client id and the repo scope, no client secret",
      gh.calls[0][1] == {"client_id": "Ov23li2tK8VKBD0fTYGG", "scope": "repo"})
n0 = len(gh.calls)
check("poll before the interval: pending without asking GitHub", flows.poll(st["flow_id"])["status"] == "pending"
      and len(gh.calls) == n0)
clock.t += 5
check("poll after the interval: GitHub's authorization_pending -> pending", flows.poll(st["flow_id"])["status"] == "pending")
check("... the poll carries client id, device code and the device grant, no secret",
      gh.calls[-1][1] == {"client_id": G.CLIENT_ID, "device_code": "DEVICE-CODE-SECRET", "grant_type": G.DEVICE_GRANT})
clock.t += 5
gh.token_answers.append({"error": "slow_down", "interval": 10})
r = flows.poll(st["flow_id"])
check("slow_down: the wait grows to GitHub's interval", r == {"status": "pending", "interval": 10})
clock.t += 10
gh.token_answers.append({"access_token": ACCESS, "token_type": "bearer", "scope": "repo", "expires_in": 28800,
                         "refresh_token": REFRESH, "refresh_token_expires_in": 15897600})
r = flows.poll(st["flow_id"])
check("approved: the token record with its 8-hour expiry and 6-month refresh",
      r["status"] == "ok" and r["token"]["access_token"] == ACCESS and r["token"]["refresh_token"] == REFRESH
      and r["token"]["expires_at"] == int(clock.t + 28800) and r["token"]["refresh_expires_at"] == int(clock.t + 15897600))
check("a finished flow cannot be polled again", flows.poll(st["flow_id"])["status"] == "expired")
for err, want in (("access_denied", "denied"), ("expired_token", "expired"), ("incorrect_device_code", "error")):
    f2 = flows.start()
    clock.t += 5
    gh.token_answers.append({"error": err, "error_description": err})
    check(f"GitHub's {err} -> {want}", flows.poll(f2["flow_id"])["status"] == want)
f3 = flows.start()
clock.t += 901
check("a code past its 15 minutes -> expired", flows.poll(f3["flow_id"])["status"] == "expired")
try:
    G.DeviceFlows(post=lambda u, f, timeout=20: {"error": "device_flow_disabled"}).start()
    check("device flow switched off on GitHub is reported", False)
except G.GitHubAuthError as exc:
    check("device flow switched off on GitHub is reported", exc.code == "device_flow_disabled")

# ---------------- renewal ----------------
sent = []
rec = G.refresh(REFRESH, post=lambda u, f, timeout=20: sent.append((u, f)) or {
    "access_token": NEW_ACCESS, "refresh_token": REFRESH[:-1] + "d", "expires_in": 28800, "refresh_token_expires_in": 100},
    now=50.0)
check("renewal: client id, refresh grant and the refresh token, no client secret (device-flow tokens)",
      sent[0][0] == G.TOKEN_URL and sent[0][1] == {"client_id": G.CLIENT_ID, "grant_type": "refresh_token",
                                                    "refresh_token": REFRESH}
      and rec["access_token"] == NEW_ACCESS and rec["expires_at"] == 28850)
try:
    G.refresh(REFRESH, post=lambda u, f, timeout=20: {"error": "bad_refresh_token"})
    check("a refused refresh token raises with GitHub's code", False)
except G.GitHubAuthError as exc:
    check("a refused refresh token raises with GitHub's code", exc.code == "bad_refresh_token")

# ---------------- the encrypted store ----------------
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "github.json"
    renewals = []

    def renew_ok(rt):
        renewals.append(rt)
        return {"access_token": NEW_ACCESS, "refresh_token": "ghr_" + "e" * 76, "scope": "repo",
                "expires_at": int(__import__("time").time()) + 28800, "refresh_expires_at": 0}
    a = WS.GitHubAuth(p, refresher=renew_ok)
    a.sign_in(dict(r["token"], expires_at=int(__import__("time").time()) + 28800), {"login": "octo", "id": 42})
    desc = a.describe()
    check("signed in: who, never the token", desc["signed_in"] and desc["login"] == "octo" and desc["token_set"]
          and ACCESS not in json.dumps(desc) and desc["review_url"].endswith("/Ov23li2tK8VKBD0fTYGG"))
    raw = p.read_text(encoding="utf-8")
    check("the file holds the tokens encrypted, not in plain text", ACCESS not in raw and REFRESH not in raw)
    b = WS.GitHubAuth(p, refresher=renew_ok)
    check("a restart reads the same sign-in", b.token == ACCESS and b.describe()["login"] == "octo" and not renewals)
    b.meta["expires_at"] = int(__import__("time").time()) - 1
    check("an expired token is renewed on use", b.token == NEW_ACCESS and renewals == [REFRESH])
    c = WS.GitHubAuth(p)
    check("... and the new pair is saved before it is used", c.token == NEW_ACCESS and c._refresh_token == "ghr_" + "e" * 76)

    def renew_refused(rt):
        raise G.GitHubAuthError("bad", "bad_refresh_token")

    def renew_offline(rt):
        raise G.GitHubAuthError("offline", "unreachable")
    c._refresher = renew_offline
    c.meta["expires_at"] = 1
    check("GitHub unreachable: the old token is kept (the call that needs it reports the network)",
          c.token == NEW_ACCESS and c.describe()["signed_in"])
    c._refresher = renew_refused
    check("GitHub refuses the refresh token: the sign-in ends", c.token is None and not c.describe()["token_set"]
          and WS.GitHubAuth(p).token is None)
    c.update(token="ghp_" + "f" * 36)
    check("a pasted token still works and is not a sign-in", c.describe()["source"] == "token"
          and not c.describe()["signed_in"])
    c.update(forget=True)
    check("sign out clears it", not WS.GitHubAuth(p).describe()["token_set"])

# ---------------- website redirect sign-in ----------------
os.environ.pop("CLOVER_GITHUB_CLIENT_SECRET", None)
try:
    G.WebSignIns().begin("https://cloverframework.com/clover-ai/")
    check("no client secret on the server: the redirect sign-in says it is not set up", False)
except G.GitHubAuthError as exc:
    check("no client secret on the server: the redirect sign-in says it is not set up", exc.code == "not_configured")
os.environ["CLOVER_GITHUB_CLIENT_SECRET"] = "test-only-secret"
try:
    wgh, wclock = FakeGitHub(), Clock()
    web = G.WebSignIns(post=wgh, clock=wclock)
    url = web.begin("https://cloverframework.com/clover-ai/")
    q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
    check("the browser goes to GitHub with Clover's id, the api.cloverframework.com callback, S256 PKCE and no sign-up",
          url.startswith(G.AUTHORIZE_URL) and q["client_id"] == G.CLIENT_ID
          and q["redirect_uri"] == "https://api.cloverframework.com/api/auth/github/callback"
          and q["code_challenge_method"] == "S256" and q["allow_signup"] == "false" and "client_secret" not in q)
    wgh.token_answers.append({"access_token": ACCESS, "token_type": "bearer", "scope": "read:user"})
    back, tok, err = web.finish(q["state"], "the-code")
    form = wgh.calls[-1][1]
    challenge = base64.urlsafe_b64encode(hashlib.sha256(form["code_verifier"].encode()).digest()).rstrip(b"=").decode()
    check("GitHub came back: the code is exchanged with the server's secret and the PKCE verifier",
          back == "https://cloverframework.com/clover-ai/" and tok == ACCESS and err == ""
          and form["client_secret"] == "test-only-secret" and challenge == q["code_challenge"])
    check("the same state cannot be used twice", web.finish(q["state"], "the-code") == (None, None, "expired"))
    q2 = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(web.begin("https://cloverframework.com/x")).query))
    check("cancel on GitHub -> back to the page with 'cancelled'",
          web.finish(q2["state"], None, "access_denied") == ("https://cloverframework.com/x", None, "cancelled"))
    code = web.issue("octo")
    check("one-time code: redeemed once", web.redeem(code) == "octo" and web.redeem(code) is None)
    code = web.issue("octo")
    wclock.t += 121
    check("one-time code: expires after 2 minutes", web.redeem(code) is None)
finally:
    os.environ.pop("CLOVER_GITHUB_CLIENT_SECRET", None)

# ---------------- collective gateway credential ----------------
with tempfile.TemporaryDirectory() as d:
    gw = collective_gateway.CollectiveGateway(state_dir=Path(d) / "gw")
    sess = gw.issue_session_from_github(ACCESS, get_user_id=lambda t: 4242, now=1000.0)
    check("the gateway answers with its own credential and a pseudonymous id",
          sess["credential"].startswith("cg1.") and sess["contributor_id"].startswith("contrib_")
          and ACCESS not in json.dumps(sess) and "4242" not in sess["credential"])
    check("the credential authenticates the same contributor",
          gw.authenticate_contributor(sess["credential"], now=2000.0) == sess["contributor_id"])
    same = gw.issue_session_from_github("ghp_" + "z" * 36, get_user_id=lambda t: 4242, now=1000.0)
    check("the same GitHub account is the same contributor whatever the token",
          same["contributor_id"] == sess["contributor_id"])
    for label, cred, when in (("a changed credential", sess["credential"][:-1] + ("0" if sess["credential"][-1] != "0" else "1"), 2000.0),
                              ("an expired credential", sess["credential"], 1000.0 + collective_gateway.SESSION_TTL_S + 1),
                              ("non-ASCII rubbish", "cg1.\u00e9\u00e9.\u00e9", 2000.0)):
        try:
            gw.authenticate_contributor(cred, now=when)
            check(f"{label} is refused", False)
        except collective_gateway.GatewaySecurityError as exc:
            check(f"{label} is refused", exc.status == 401)
    gw2 = collective_gateway.CollectiveGateway(state_dir=Path(d) / "gw")
    check("a restarted gateway still accepts it (its secret is kept)",
          gw2.authenticate_contributor(sess["credential"], now=2000.0) == sess["contributor_id"])
    body = sess["credential"][4:].split(".")[0]
    forged = "cg1." + body + "." + __import__("hmac").new(
        __import__("hmac").new(collective_gateway._PUBLIC_DEFAULT_SECRET, b"clover-gateway-session-key-v1",
                               hashlib.sha256).digest(), body.encode(), hashlib.sha256).hexdigest()
    os.environ.pop("CLOVER_GATEWAY_SECRET", None)
    try:
        collective_gateway.CollectiveGateway().authenticate_contributor(forged, now=2000.0)
        check("a gateway on the public fallback secret does not accept credentials made with it", False)
    except collective_gateway.GatewaySecurityError:
        check("a gateway on the public fallback secret does not accept credentials made with it", True)
check("the self-made 'Google' key is gone from google_identity",
      not hasattr(google_identity, "issue_device_oidc_id_token") and not hasattr(google_identity, "_get_device_oidc_keypair")
      and not hasattr(google_identity.GoogleIdentityManager, "connect_device_oidc"))

with tempfile.TemporaryDirectory() as d:
    store = core_sync.for_process(str(Path(d) / "collective"), "laptop", None)
    try:
        store.set_consent(contribute=True, identity_connected=False, require_identity=True)
        check("collective consent needs the GitHub sign-in", False)
    except core_sync.CoreSyncError as exc:
        check("collective consent needs the GitHub sign-in", "GitHub" in str(exc))
    view = store.view(None, identity={"connected": True, "provider": "github", "login": "octo"})
    check("the collective view carries the GitHub identity", view["identity"]["login"] == "octo"
          and "google_identity" not in view)
    got = store.gateway_session(github_token=ACCESS, gateway_url="https://gw.example",
                                transport_fn=lambda m, u, id_token=None, body=None: {
                                    "credential": "cg1.x.y", "expires_at": 5} if (m, u, id_token) == (
                                    "POST", "https://gw.example/api/collective/gateway/session", ACCESS) else {})
    check("the client asks the gateway for its credential with the GitHub token", got["credential"] == "cg1.x.y")

# ---------------- the routes, through the real server ----------------
saved = {k: getattr(server.S, k) for k in ("args", "port", "perms", "settings", "search", "github", "workspaces",
                                            "vscode", "repo", "gh_flows", "gh_web", "gateway_runtime", "token")}
saved_fns = (server.layer_kit, server.verify_k3_credential, G.user,
             collective_gateway.CollectiveGateway.__dict__["_github_user_id"])
import k3_host_engine  # noqa: E402
import paths  # noqa: E402
saved_unlock, saved_web = k3_host_engine.try_unlock_k3_with_credential, paths.WEB
httpd = None


def offline_refresh(rt):
    raise G.GitHubAuthError("tests never renew with github.com", "unreachable")


with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
    base = Path(d)
    try:
        import clover_repo
        import permissions as P
        import vscode_bridge
        import web_search
        S = server.S
        S.args = argparse.Namespace(base=str(base), port=0, mode="lfm3", no_gate=False, idle_exit_min=0.0)
        S.perms = P.PermissionStore(paths.config_file(base, "permissions.json"))
        S.settings = server.Settings(paths.config_file(base, "settings.json"))
        S.search = web_search.SearchConfig.shared(paths.config_file(base, "search.json"),
                                                  paths.vault_file(base, "search-keys.json"))
        S.github = WS.GitHubAuth(paths.vault_file(base, "github.json"), refresher=offline_refresh)
        S.workspaces = WS.Workspaces(paths.config_file(base, "workspaces.json"), base / "workspaces", base, S.github)
        S.vscode = vscode_bridge.Bridge(paths.vault_file(base, "vscode-bridge.json"))
        kit = ui_harness.FakeKit()
        server.layer_kit = lambda: kit
        S.repo = clover_repo.Repo(base, HERE.parent / "clover" / "AGENTS.md", kit=server.layer_kit)
        S.token, S.gateway_runtime = "", None
        rclock, rgh = Clock(), FakeGitHub()
        rclock.t = __import__("time").time()  # token expiry is compared with the real clock
        S.gh_flows, S.gh_web = G.DeviceFlows(post=rgh, clock=rclock), G.WebSignIns(post=rgh, clock=rclock)
        G.user = lambda token, timeout=15: {"login": "octo", "id": 42, "name": "Octo"}
        k3_host_engine.try_unlock_k3_with_credential = lambda *a, **k: {"unlocked": False}
        user_ids = []
        collective_gateway.CollectiveGateway._github_user_id = staticmethod(lambda t: user_ids.append(t) or 4242)
        httpd = server.StudioHTTPServer(("127.0.0.1", 0), server.Handler)
        httpd.daemon_threads = True
        S.port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        root = f"http://127.0.0.1:{S.port}"

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **k):
                return None
        opener = urllib.request.build_opener(NoRedirect)

        def call(method, path, body=None, headers=None):
            data = json.dumps(body).encode() if body is not None else None
            req = urllib.request.Request(root + path, data=data, method=method,
                                         headers=dict({"Content-Type": "application/json"} if data else {}, **(headers or {})))
            try:
                with opener.open(req, timeout=30) as resp:
                    txt = resp.read().decode()
                    return resp.status, (json.loads(txt) if txt.strip().startswith("{") else txt), resp.headers
            except urllib.error.HTTPError as exc:
                txt = exc.read().decode()
                return exc.code, (json.loads(txt) if txt.strip().startswith("{") else txt), exc.headers

        code, st, _ = call("POST", "/api/github/device/start", {})
        check("app route: start answers the user code and a flow id", code == 200 and st.get("user_code") == "WDJB-MJHT"
              and "device_code" not in st)
        code, pr, _ = call("POST", "/api/github/device/poll", {"flow_id": st["flow_id"]})
        check("app route: pending while the user has not approved", code == 200 and pr["status"] == "pending")
        rclock.t += 5
        rgh.token_answers.append({"access_token": ACCESS, "scope": "repo", "expires_in": 28800, "refresh_token": REFRESH,
                                  "refresh_token_expires_in": 15897600})
        code, pr, _ = call("POST", "/api/github/device/poll", {"flow_id": st["flow_id"]})
        check("app route: approved -> signed in as @octo, and the page never sees a token",
              code == 200 and pr["status"] == "ok" and pr["login"] == "octo" and pr["github"]["signed_in"]
              and ACCESS not in json.dumps(pr) and REFRESH not in json.dumps(pr))
        check("... the memory sync and Workspaces now use that sign-in", S.github.token == ACCESS)
        code, cv, _ = call("GET", "/api/collective")
        check("collective learning sees the GitHub account", code == 200 and cv["identity"]["connected"]
              and cv["identity"]["login"] == "octo")
        coll = server.collective_store()
        coll.gateway_url = root
        cred = server.collective_credential(coll)
        cred2 = server.collective_credential(coll)
        check("the gateway checks GitHub once and its credential is reused",
              cred.startswith("cg1.") and cred2 == cred and user_ids == [ACCESS])
        gw_rt = server.collective_gateway_runtime()
        check("... and it authenticates at the gateway", gw_rt["gateway"].authenticate_contributor(cred).startswith("contrib_"))
        code, out, _ = call("POST", "/api/github", {"forget": True})
        import vault
        check("sign out: no token, no gateway credential, collective consent off",
              code == 200 and not out["github"]["token_set"] and not vault.get(str(base), "collective_gateway")
              and not S.settings.get().get("collective_receive"))
        code, out, _ = call("POST", "/api/collective/consent", {"receive": True})
        check("signed out: collective learning cannot be switched on", code == 400 and "GitHub" in out["error"])

        paths.WEB = True  # the website (k3)
        code, out, _ = call("POST", "/api/github/device/start", {})
        check("website: the app sign-in route is closed (a visitor's token never becomes the server's)",
              code in (401, 404))
        code, out, _ = call("GET", "/api/auth/github")
        check("website: says the redirect sign-in is not set up without a secret", code == 200 and out["redirect"] is False)
        code, _, hdr = call("GET", "/api/auth/github/start?return_to=" + urllib.parse.quote("https://cloverframework.com/clover-ai/"))
        check("website: without the secret, back to the page to use a code instead",
              code == 302 and hdr["Location"] == "https://cloverframework.com/clover-ai/#gh_error=not_configured")
        os.environ["CLOVER_GITHUB_CLIENT_SECRET"] = "test-only-secret"
        code, _, hdr = call("GET", "/api/auth/github/start?return_to=" + urllib.parse.quote("https://evil.example/steal"))
        state = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(hdr["Location"]).query)).get("state", "")
        check("website: an outside return address is replaced by Clover's page",
              code == 302 and hdr["Location"].startswith(G.AUTHORIZE_URL)
              and S.gh_web.pending[state]["return_to"] == "https://cloverframework.com/clover-ai/")
        check("website: a return address with a line break cannot reach the Location header",
              server.web_return("https://cloverframework.com/clover-ai/\r\nSet-Cookie: x=1") == "https://cloverframework.com/clover-ai/"
              and server.web_return("https://cloverframework.com@evil.example/") == "https://cloverframework.com/clover-ai/"
              and server.web_return("https://cloverframework.com/clover-ai/?v=2#old") == "https://cloverframework.com/clover-ai/?v=2")
        seen = []
        server.verify_k3_credential = lambda tok, *a, **k: (seen.append(tok) or (True, "octo", 200)) if tok == ACCESS else (False, "no", 403)
        rgh.token_answers.append({"access_token": ACCESS, "scope": "read:user"})
        code, _, hdr = call("GET", f"/api/auth/github/callback?code=abc&state={state}")
        frag = urllib.parse.urlsplit(hdr["Location"]).fragment
        check("website: GitHub's callback checks the account once and returns a one-time code, not a token",
              code == 302 and hdr["Location"].startswith("https://cloverframework.com/clover-ai/#gh_login=")
              and ACCESS not in hdr["Location"] and seen == [ACCESS])
        code, fin, _ = call("POST", "/api/auth/github/finish", {"code": frag.split("=", 1)[1]},
                            {"Origin": "https://cloverframework.com"})
        check("website: the page trades the one-time code for its session", code == 200 and fin["account"] == "octo"
              and fin["token"].startswith("acct:octo:"))
        code, fin, _ = call("POST", "/api/auth/github/finish", {"code": frag.split("=", 1)[1]},
                            {"Origin": "https://cloverframework.com"})
        check("website: the one-time code works once", code == 400)
        code, _, hdr = call("GET", "/api/auth/github/start?return_to=" + urllib.parse.quote("https://cloverframework.com/clover-ai/"))
        state = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(hdr["Location"]).query)).get("state", "")
        rgh.token_answers.append({"access_token": "gho_" + "9" * 36, "scope": "read:user"})
        code, _, hdr = call("GET", f"/api/auth/github/callback?code=abc&state={state}")
        check("website: an account that is not approved goes back with not_allowed",
              hdr["Location"].endswith("#gh_error=not_allowed"))
        code, _, hdr = call("GET", "/api/auth/github/callback?error=access_denied&state=unknown")
        check("website: an unknown state goes back to Clover's page with 'expired'",
              hdr["Location"] == "https://cloverframework.com/clover-ai/#gh_error=expired")
        os.environ.pop("CLOVER_GITHUB_CLIENT_SECRET", None)
        code, st, _ = call("POST", "/api/auth/github/device/start", {}, {"Origin": "https://cloverframework.com"})
        rclock.t += 5
        rgh.token_answers.append({"access_token": ACCESS, "scope": "read:user user:email"})
        code, pr, _ = call("POST", "/api/auth/github/device/poll", {"flow_id": st["flow_id"]},
                           {"Origin": "https://cloverframework.com"})
        check("website without the secret: the code sign-in gives the same session",
              code == 200 and pr["status"] == "ok" and pr["token"].startswith("acct:octo:") and ACCESS not in json.dumps(pr))
        check("website: no GitHub token was kept on the server", S.github.token is None)
    finally:
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        paths.WEB = saved_web
        os.environ.pop("CLOVER_GITHUB_CLIENT_SECRET", None)
        k3_host_engine.try_unlock_k3_with_credential = saved_unlock
        server.layer_kit, server.verify_k3_credential, G.user = saved_fns[0], saved_fns[1], saved_fns[2]
        collective_gateway.CollectiveGateway._github_user_id = saved_fns[3]
        for k, v in saved.items():
            setattr(server.S, k, v)

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
