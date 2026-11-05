"""Live replay of the user's failing turns (2026-11-01) against the running Studio, with the user's own settings.
Runs only the hidden terminal (no VS Code actions on the user's screen); deletes its chats at the end.
Usage (on the laptop): venv\\Scripts\\python.exe tests\\user_chat_replay.py"""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8765"
REPO = r"C:\Users\santh\CloverAI\workspaces\clover-framework"
TURNS = [("replay-commits", f"In {REPO}, check the recent commits using the terminal.", ["shell"]),
         ("replay-install", "then install it use the terminal", [])]  # ambiguous: nothing may run without asking


def call(method, path, body=None, timeout=900):
    data = None if body is None else json.dumps(body).encode("utf-8")
    r = urllib.request.Request(BASE + path, data=data, method=method,
                               headers={"Content-Type": "application/json"} if body is not None else {})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read())


print("settings:", json.dumps(call("GET", "/api/settings")["settings"]))
shell_before = call("GET", "/api/permissions")["permissions"]["shell"]["mode"]
call("POST", "/api/permissions", {"permissions": {"shell": "ask"}})
sids = []
try:
    for name, prompt, allow in TURNS:
        sid = f"{name}-{int(time.time())}"
        sids.append(sid)
        t0 = time.perf_counter()
        res = call("POST", "/api/chat", {"prompt": prompt, "session_id": sid, "allow_once": allow})
        print(json.dumps({
            "prompt": prompt, "seconds": round(time.perf_counter() - t0, 1), "tools_offered": res.get("tools_offered"),
            "permission_request": res.get("permission_request"),
            "calls": [{"tool": c.get("tool"), "args": c.get("arguments"), "ran": c.get("ran"),
                       "result": str(c.get("result"))[:240]} for c in res.get("tool_calls") or []],
            "generation": [{k: g.get(k) for k in ("new_tokens", "finish", "looped", "decode_s")} for g in res.get("generation") or []],
            "layers": [x.get("layer") for x in res.get("layers") or []],
            "reply": (res.get("reply") or "")[:500]}, ensure_ascii=False), flush=True)
finally:
    call("POST", "/api/permissions", {"permissions": {"shell": shell_before}})
    print("terminal permission restored to", call("GET", "/api/permissions")["permissions"]["shell"]["mode"])
    for sid in sids:
        print("deleted", sid, call("POST", "/api/session/forget", {"session_id": sid, "delete": True}).get("deleted"))
