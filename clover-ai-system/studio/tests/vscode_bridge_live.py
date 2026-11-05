"""Live check on the laptop: Clover uses VS Code from a chat turn. Each VS Code tool asks first (permission_request),
then allow_once runs it in the VS Code window in front. Needs the Studio on 127.0.0.1:8765 and VS Code with the Clover
extension (0.3.0+) open. Changes no permission settings. Usage: python tests/vscode_bridge_live.py [out.jsonl]"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8765"
OUT = sys.argv[1] if len(sys.argv) > 1 else "vscode_bridge_live.jsonl"
MD = os.path.join(os.environ["USERPROFILE"], "CloverAI", "tmp", "vscode-bridge-test.md")
SIDS = []


def req(method, path, body=None, timeout=1800):
    data = None if body is None else json.dumps(body).encode("utf-8")
    r = urllib.request.Request(BASE + path, data=data, method=method,
                               headers={"Content-Type": "application/json"} if body is not None else {})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read()), time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), time.perf_counter() - t0


def log(name, ok, **kw):
    rec = {"check": name, "ok": bool(ok), **kw}
    print(json.dumps(rec, ensure_ascii=False)[:1500], flush=True)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def calls_of(r, tool):
    return [c for c in r.get("tool_calls") or [] if c.get("tool") == tool]


def brief(r):
    return [{"tool": c.get("tool"), "args": c.get("arguments"), "ran": c.get("ran"), "result": str(c.get("result"))[:400]}
            for c in r.get("tool_calls") or []]


def turn(name, prompt, perm, tool, expect, modes):
    """/api/chat answers without pausing: a tool set to Ask is declined there (the page's stream asks instead)."""
    sid = f"vscode-live-{name}-{int(time.time())}"
    SIDS.extend([sid, sid + "-ok"])
    if modes.get(perm) == "deny":
        return log(f"{name}: skipped ({perm} is off in Permissions)", True)
    if modes.get(perm) == "ask":
        code, r, dt = req("POST", "/api/chat", {"prompt": prompt, "session_id": sid})
        calls = calls_of(r, tool)
        log(f"{name}: model picks {tool}, not run without permission", code == 200 and calls and not any(c.get("ran") for c in calls),
            seconds=round(dt, 1), tools_offered=r.get("tools_offered"), calls=brief(r), reply=(r.get("reply") or "")[:300])
    code, r, dt = req("POST", "/api/chat", {"prompt": prompt, "session_id": sid + "-ok", "allow_once": [perm]})
    calls = [c for c in calls_of(r, tool) if c.get("ran")]
    log(f"{name}: runs in VS Code", code == 200 and bool(calls) and expect(" ".join(str(c.get("result")) for c in calls)),
        seconds=round(dt, 1), tools_offered=r.get("tools_offered"), calls=brief(r), reply=(r.get("reply") or "")[:400])


def main():
    code, st, _ = req("GET", "/api/vscode")
    log("VS Code connected", code == 200 and bool(st.get("connected")), state=st)
    code, p, _ = req("GET", "/api/permissions")
    modes = {k: v["mode"] for k, v in p["permissions"].items()}
    code, s, _ = req("GET", "/api/settings")
    log("vscode permission listed", "vscode" in modes, modes=modes, tools_setting=s["settings"]["tools"])
    turn("open", f"Open the file {MD} in VS Code.", "vscode", "vscode_open", lambda x: "Markdown preview" in x, modes)
    turn("run", "Run the command `git --version` in a VS Code terminal and tell me the version.", "shell", "vscode_run",
         lambda x: "git version" in x and '"exitCode": 0' in x, modes)
    turn("install", "Install the VS Code extension clover-framework.clover-local.", "vscode", "vscode_install",
         lambda x: '"already": true' in x, modes)
    for sid in SIDS:  # test conversations do not stay in the user's records
        req("POST", "/api/session/forget", {"session_id": sid, "delete": True})


if __name__ == "__main__":
    main()
