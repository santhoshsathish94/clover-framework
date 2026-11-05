"""Live picture request against the running Studio with the user's settings: the Pictures permission stays on Ask and
this one request is allowed once, as a user would click Allow once. Deletes its chat; the picture stays in the media
folder. Usage (on the laptop): venv\\Scripts\\python.exe tests\\picture_live.py"""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8765"


def call(method, path, body=None, timeout=900):
    data = None if body is None else json.dumps(body).encode("utf-8")
    r = urllib.request.Request(BASE + path, data=data, method=method,
                               headers={"Content-Type": "application/json"} if body is not None else {})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read())


print("settings:", json.dumps(call("GET", "/api/settings")["settings"]))
print("pictures permission:", call("GET", "/api/permissions")["permissions"]["media"]["mode"])
sid = f"picture-live-{int(time.time())}"
try:
    t0 = time.perf_counter()
    res = call("POST", "/api/chat", {"prompt": "can you generate an image of a parrot", "session_id": sid,
                                     "allow_once": ["media"]})
    print(json.dumps({
        "seconds": round(time.perf_counter() - t0, 1), "tools_offered": res.get("tools_offered"),
        "permission_request": res.get("permission_request"),
        "calls": [{"tool": c.get("tool"), "args": c.get("arguments"), "ran": c.get("ran"),
                   "result": str(c.get("result"))[:300]} for c in res.get("tool_calls") or []],
        "images": res.get("images"), "reply": (res.get("reply") or "")[:400]}, ensure_ascii=False), flush=True)
finally:
    print("deleted", sid, call("POST", "/api/session/forget", {"session_id": sid, "delete": True}).get("deleted"))
