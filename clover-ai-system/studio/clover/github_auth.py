"""Sign in with GitHub: one account for Clover's memory repository, Workspaces, k3 access and collective learning.

The laptop and phone apps use GitHub's device flow with Clover's OAuth app: the page shows a code, the user approves it
at github.com/login/device, and this server collects the token. GitHub needs no client secret for the device flow or
for renewing its tokens. The website uses GitHub's redirect sign-in back to Clover's own domain, which needs the
client secret; it is kept only on that server (CLOVER_GITHUB_CLIENT_SECRET), and the website keeps no GitHub token.
"""
import base64
import hashlib
import json
import os
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

CLIENT_ID = (os.environ.get("CLOVER_GITHUB_CLIENT_ID") or "Ov23li2tK8VKBD0fTYGG").strip()
APP_SCOPE = "repo"  # the private memory repository and private projects; GitHub has no narrower scope for them
WEB_SCOPE = "read:user user:email"  # the website only checks who signed in against its allowlist
REDIRECT_URI = (os.environ.get("CLOVER_GITHUB_REDIRECT_URI")
                or "https://api.cloverframework.com/api/auth/github/callback").strip()
REVIEW_URL = f"https://github.com/settings/connections/applications/{CLIENT_ID}"
DEVICE_CODE_URL = "https://github.com/login/device/code"
TOKEN_URL = "https://github.com/login/oauth/access_token"
AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
USER_URL = "https://api.github.com/user"
UA = "CloverAIStudio/5.0"
DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"


class GitHubAuthError(ValueError):
    def __init__(self, message, code=""):
        super().__init__(message)
        self.code = code


def _client_secret():
    return (os.environ.get("CLOVER_GITHUB_CLIENT_SECRET") or "").strip()


def web_configured():
    return bool(_client_secret())


def post_form(url, form, timeout=20):
    """GitHub's OAuth endpoints answer 200 with {"error": ...} when they refuse."""
    req = urllib.request.Request(url, urllib.parse.urlencode(form).encode("utf-8"),
                                 {"Accept": "application/json", "User-Agent": UA,
                                  "Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            doc = json.loads(r.read().decode("utf-8", "replace") or "{}")
    except urllib.error.HTTPError as exc:
        try:
            doc = json.loads(exc.read().decode("utf-8", "replace") or "{}")
        except ValueError:
            doc = {"error": f"http_{exc.code}"}
    except (OSError, ValueError) as exc:
        raise GitHubAuthError(f"GitHub could not be reached: {exc}", "unreachable") from exc
    return doc if isinstance(doc, dict) else {"error": "bad_response"}


def user(token, timeout=15):
    """Who a token belongs to: {"login", "id", "name"}."""
    req = urllib.request.Request(USER_URL, headers={"Authorization": f"Bearer {token}", "User-Agent": UA,
                                                    "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            doc = json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        refused = exc.code in (401, 403)
        raise GitHubAuthError("GitHub did not accept the sign-in" if refused else f"GitHub answered HTTP {exc.code}",
                              "bad_token" if refused else "http") from exc
    except (OSError, ValueError) as exc:
        raise GitHubAuthError(f"GitHub could not be reached: {exc}", "unreachable") from exc
    if not isinstance(doc, dict) or not doc.get("login") or not isinstance(doc.get("id"), int):
        raise GitHubAuthError("GitHub did not say who signed in", "bad_user")
    return {"login": str(doc["login"]), "id": int(doc["id"]), "name": str(doc.get("name") or "")}


def token_record(resp, now=None):
    """GitHub's token answer -> what Clover keeps. An expiring token (8 hours) comes with a refresh token (6 months)."""
    now = time.time() if now is None else now
    if not isinstance(resp, dict) or not resp.get("access_token"):
        raise GitHubAuthError("GitHub returned no access token", str((resp or {}).get("error") or "no_token"))
    return {"access_token": str(resp["access_token"]), "refresh_token": str(resp.get("refresh_token") or ""),
            "scope": str(resp.get("scope") or ""),
            "expires_at": int(now + int(resp["expires_in"])) if resp.get("expires_in") else 0,
            "refresh_expires_at": (int(now + int(resp["refresh_token_expires_in"]))
                                   if resp.get("refresh_token_expires_in") else 0)}


def refresh(refresh_token, post=None, now=None):
    """A new token pair. The used refresh token and the old access token stop working at once."""
    resp = (post or post_form)(TOKEN_URL, {"client_id": CLIENT_ID, "grant_type": "refresh_token",
                                           "refresh_token": refresh_token})
    if resp.get("error"):
        raise GitHubAuthError(str(resp.get("error_description") or resp["error"]), str(resp["error"]))
    return token_record(resp, now)


class DeviceFlows:
    """Device sign-ins in progress. The device code stays on this server; the page sees a random id and the user code."""

    MAX_PENDING = 16

    def __init__(self, post=None, clock=time.time):
        self.lock, self.flows, self._post, self._clock = threading.Lock(), {}, post or post_form, clock

    def start(self, scope=APP_SCOPE):
        resp = self._post(DEVICE_CODE_URL, {"client_id": CLIENT_ID, "scope": scope})
        if resp.get("error") or not resp.get("device_code") or not resp.get("user_code"):
            raise GitHubAuthError("GitHub could not start the sign-in: "
                                  + str(resp.get("error_description") or resp.get("error") or "no code"),
                                  str(resp.get("error") or "no_code"))
        now, flow_id = self._clock(), secrets.token_urlsafe(18)
        interval, life = max(5, int(resp.get("interval") or 5)), int(resp.get("expires_in") or 900)
        with self.lock:
            for k in [k for k, f in self.flows.items() if now >= f["expires_at"]]:
                self.flows.pop(k)
            while len(self.flows) >= self.MAX_PENDING:
                self.flows.pop(min(self.flows, key=lambda k: self.flows[k]["expires_at"]))
            self.flows[flow_id] = {"device_code": str(resp["device_code"]), "interval": interval,
                                   "next_poll": now + interval, "expires_at": now + life}
        return {"flow_id": flow_id, "user_code": str(resp["user_code"]),
                "verification_uri": str(resp.get("verification_uri") or "https://github.com/login/device"),
                "expires_in": life, "interval": interval}

    def poll(self, flow_id):
        """-> {"status": "pending" | "ok" | "denied" | "expired" | "error", ...}; "ok" carries the token record."""
        now, flow_id = self._clock(), str(flow_id or "")
        with self.lock:
            f = self.flows.get(flow_id)
            if f is None or now >= f["expires_at"]:
                self.flows.pop(flow_id, None)
                return {"status": "expired"}
            if now < f["next_poll"]:  # GitHub answers slow_down when asked more often than the interval
                return {"status": "pending", "interval": f["interval"]}
            f["next_poll"] = now + f["interval"]
            device_code = f["device_code"]
        resp = self._post(TOKEN_URL, {"client_id": CLIENT_ID, "device_code": device_code, "grant_type": DEVICE_GRANT})
        err = str(resp.get("error") or "")
        with self.lock:
            f = self.flows.get(flow_id)
            if err in ("authorization_pending", "slow_down") and f is not None:
                if err == "slow_down":
                    f["interval"] = max(f["interval"] + 5, int(resp.get("interval") or 0))
                    f["next_poll"] = now + f["interval"]
                return {"status": "pending", "interval": f["interval"]}
            self.flows.pop(flow_id, None)
        if err == "access_denied":
            return {"status": "denied"}
        if err in ("expired_token", "token_expired", "authorization_pending", "slow_down"):
            return {"status": "expired"}
        if err:
            return {"status": "error", "error": str(resp.get("error_description") or err)}
        return {"status": "ok", "token": token_record(resp, now)}


class WebSignIns:
    """Website sign-ins in progress (state -> PKCE verifier and the page to return to) and one-time login codes."""

    TTL, CODE_TTL, MAX_PENDING = 600, 120, 256

    def __init__(self, post=None, clock=time.time):
        self.lock, self._post, self._clock = threading.Lock(), post or post_form, clock
        self.pending, self.codes = {}, {}

    def _prune(self, now):
        for k in [k for k, p in self.pending.items() if now - p["started"] > self.TTL]:
            self.pending.pop(k)
        for k in [k for k, c in self.codes.items() if now >= c["expires_at"]]:
            self.codes.pop(k)

    def begin(self, return_to):
        """-> the GitHub address to send the browser to."""
        if not web_configured():
            raise GitHubAuthError("GitHub sign-in is not set up on this server yet", "not_configured")
        now, state, verifier = self._clock(), secrets.token_urlsafe(24), secrets.token_urlsafe(48)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode()
        with self.lock:
            self._prune(now)
            while len(self.pending) >= self.MAX_PENDING:
                self.pending.pop(min(self.pending, key=lambda k: self.pending[k]["started"]))
            self.pending[state] = {"verifier": verifier, "return_to": return_to, "started": now}
        q = {"client_id": CLIENT_ID, "redirect_uri": REDIRECT_URI, "scope": WEB_SCOPE, "state": state,
             "code_challenge": challenge, "code_challenge_method": "S256", "allow_signup": "false"}
        return f"{AUTHORIZE_URL}?{urllib.parse.urlencode(q)}"

    def finish(self, state, code, error=None):
        """GitHub came back -> (the page to return to or None, a token to check once and drop or None, error code)."""
        now = self._clock()
        with self.lock:
            self._prune(now)
            p = self.pending.pop(str(state or ""), None)
        if p is None:
            return None, None, "expired"
        if error or not code:
            return p["return_to"], None, "cancelled" if error == "access_denied" else "no_code"
        try:
            resp = self._post(TOKEN_URL, {"client_id": CLIENT_ID, "client_secret": _client_secret(), "code": str(code),
                                          "redirect_uri": REDIRECT_URI, "code_verifier": p["verifier"]})
        except GitHubAuthError as exc:
            return p["return_to"], None, exc.code
        if resp.get("error") or not resp.get("access_token"):
            return p["return_to"], None, str(resp.get("error") or "no_token")
        return p["return_to"], str(resp["access_token"]), ""

    def issue(self, account):
        """A one-time code the page exchanges for its session, so no session token travels in an address."""
        now, code = self._clock(), secrets.token_urlsafe(24)
        with self.lock:
            self._prune(now)
            self.codes[code] = {"account": account, "expires_at": now + self.CODE_TTL}
        return code

    def redeem(self, code):
        now = self._clock()
        with self.lock:
            c = self.codes.pop(str(code or ""), None)
        return c["account"] if c and now < c["expires_at"] else None
